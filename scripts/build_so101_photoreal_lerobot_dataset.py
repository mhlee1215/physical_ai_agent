#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
PHOTO_CAMERA_CONTRACT = {
    "observation.images.camera1": "photoreal egocentric_cam",
    "observation.images.camera2": "photoreal wrist_cam",
    "observation.images.camera3": "photoreal wrist_cam duplicate",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an SO101 LeRobot parquet dataset and replace embedded image bytes "
            "with Blender photoreal renders while preserving state/action/timestamps."
        )
    )
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-keys", default=",".join(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--duplicate-camera3-from-camera2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rewrite-color-task-prompts",
        action="store_true",
        help="Rewrite task metadata from the source episode seed so prompts name the actual target color/shape.",
    )
    parser.add_argument("--task-skill-mode", default="pick_cube")
    parser.add_argument("--env-id", default="MuJoCoPickLift-v1")
    parser.add_argument("--env-source", choices=("gym", "high_contrast_picklift"), default="high_contrast_picklift")
    args = parser.parse_args()

    report = build_photoreal_lerobot_dataset(
        source_dataset_root=args.source_dataset_root,
        rendered_dir=args.rendered_dir,
        output_root=args.output_root,
        repo_id=args.repo_id,
        camera_keys=tuple(item.strip() for item in args.camera_keys.split(",") if item.strip()),
        duplicate_camera3_from_camera2=args.duplicate_camera3_from_camera2,
        rewrite_color_task_prompts=args.rewrite_color_task_prompts,
        task_skill_mode=args.task_skill_mode,
        env_id=args.env_id,
        env_source=args.env_source,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_photoreal_lerobot_dataset(
    *,
    source_dataset_root: Path,
    rendered_dir: Path,
    output_root: Path,
    repo_id: str,
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
    duplicate_camera3_from_camera2: bool = True,
    rewrite_color_task_prompts: bool = False,
    task_skill_mode: str = "pick_cube",
    env_id: str = "MuJoCoPickLift-v1",
    env_source: str = "high_contrast_picklift",
    overwrite: bool = False,
) -> dict[str, Any]:
    source_dataset_root = source_dataset_root.resolve()
    rendered_dir = rendered_dir.resolve()
    output_root = output_root.resolve()
    if not (source_dataset_root / "data").exists():
        raise FileNotFoundError(f"missing source LeRobot data directory: {source_dataset_root / 'data'}")
    if not rendered_dir.exists():
        raise FileNotFoundError(f"missing rendered frame directory: {rendered_dir}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)

    rendered_index = _rendered_index(rendered_dir)
    episode_tasks = (
        _episode_color_tasks(source_dataset_root, skill_mode=task_skill_mode, env_id=env_id, env_source=env_source)
        if rewrite_color_task_prompts
        else {}
    )
    task_indices = _task_indices(episode_tasks)
    missing: list[str] = []
    replaced_frames = 0
    replaced_images = 0

    shutil.copytree(
        source_dataset_root,
        output_root,
        ignore=shutil.ignore_patterns("photoreal_preview", "render_replay"),
    )
    data_files = sorted((output_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"missing parquet files under {output_root / 'data'}")

    for parquet_path in data_files:
        table = pq.read_table(parquet_path)
        columns = table.to_pydict()
        row_count = table.num_rows
        replacement_columns: dict[str, list[dict[str, bytes | str | None]]] = {key: [] for key in camera_keys}
        replacement_task_indices: list[int] = []
        for row_index in range(row_count):
            episode = int(columns["episode_index"][row_index])
            frame = int(columns["frame_index"][row_index])
            if episode_tasks:
                replacement_task_indices.append(task_indices[episode_tasks[episode]])
            for camera_key in camera_keys:
                render_path = rendered_index.get((episode, frame, camera_key))
                if render_path is None and duplicate_camera3_from_camera2 and camera_key == "observation.images.camera3":
                    render_path = rendered_index.get((episode, frame, "observation.images.camera2"))
                if render_path is None:
                    missing.append(f"episode={episode} frame={frame} camera={camera_key}")
                    replacement_columns[camera_key].append(columns[camera_key][row_index])
                    continue
                replacement_columns[camera_key].append(
                    {
                        "bytes": _rgb_png_bytes(render_path),
                        "path": f"images/{camera_key.replace('.', '_')}/episode_{episode:04d}_frame_{frame:04d}.png",
                    }
                )
                replaced_images += 1
            replaced_frames += 1
        if missing:
            continue
        updated = table
        for camera_key, values in replacement_columns.items():
            field_index = updated.schema.get_field_index(camera_key)
            if field_index < 0:
                raise ValueError(f"source parquet does not contain camera column: {camera_key}")
            array = pa.array(values, type=updated.schema.field(field_index).type)
            updated = updated.set_column(field_index, camera_key, array)
        if episode_tasks:
            field_index = updated.schema.get_field_index("task_index")
            if field_index < 0:
                raise ValueError("source parquet does not contain task_index")
            array = pa.array(replacement_task_indices, type=updated.schema.field(field_index).type)
            updated = updated.set_column(field_index, "task_index", array)
        pq.write_table(updated, parquet_path)

    if missing:
        shutil.rmtree(output_root)
        sample = ", ".join(missing[:12])
        suffix = f" ... and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise ValueError(f"missing rendered frames: {sample}{suffix}")

    _write_manifest(
        output_root=output_root,
        source_dataset_root=source_dataset_root,
        rendered_dir=rendered_dir,
        repo_id=repo_id,
        camera_keys=camera_keys,
        replaced_frames=replaced_frames,
        replaced_images=replaced_images,
        duplicate_camera3_from_camera2=duplicate_camera3_from_camera2,
        episode_tasks=episode_tasks,
    )
    if episode_tasks:
        _write_task_metadata(output_root, episode_tasks=episode_tasks, task_indices=task_indices)
    return _read_manifest(output_root)


def _rendered_index(rendered_dir: Path) -> dict[tuple[int, int, str], Path]:
    index: dict[tuple[int, int, str], Path] = {}
    patterns = (
        re.compile(r"episode_(\d+)_frame_(\d+)_(camera\d)\.(?:png|jpe?g)$", re.IGNORECASE),
        re.compile(
            r"episode_(\d+)_frame_(\d+)_(observation_images_camera\d)\.(?:png|jpe?g)$",
            re.IGNORECASE,
        ),
    )
    for path in sorted(rendered_dir.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        for pattern in patterns:
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            camera = match.group(3).replace("observation_images_", "")
            index[(int(match.group(1)), int(match.group(2)), f"observation.images.{camera}")] = path
            break
    return index


def _rgb_png_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        buffer = BytesIO()
        rgb.save(buffer, format="PNG")
        return buffer.getvalue()


def _episode_color_tasks(source_dataset_root: Path, *, skill_mode: str, env_id: str, env_source: str) -> dict[int, str]:
    report_path = source_dataset_root / "so101_lerobot_export_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing source export report for color prompts: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes") or []
    if not episodes:
        raise ValueError(f"source export report has no episodes: {report_path}")

    from export_so101_teacher_rollouts_lerobot import _target_object_metadata
    from render_so101_dataset_blender_preview import _make_env

    env = _make_env(env_id=env_id, env_source=env_source)
    tasks: dict[int, str] = {}
    try:
        for episode_index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                continue
            seed = episode.get("seed")
            if seed is None:
                raise ValueError(f"missing seed for episode {episode_index} in {report_path}")
            target = episode.get("target_object")
            if isinstance(target, dict) and target.get("color") and target.get("shape"):
                color = str(target["color"]).strip().lower()
                shape = str(target["shape"]).strip().lower()
            else:
                env.reset(seed=int(seed))
                target = _target_object_metadata(env)
                color = str(target["color"]).strip().lower()
                shape = str(target["shape"]).strip().lower()
            if not color or color == "visible":
                raise ValueError(f"could not resolve concrete target color for episode {episode_index}")
            tasks[episode_index] = _color_task_prompt(skill_mode=skill_mode, color=color, shape=shape)
    finally:
        env.close()
    return tasks


def _color_task_prompt(*, skill_mode: str, color: str, shape: str) -> str:
    object_name = f"{color} {shape}".strip()
    if skill_mode == "pick_from_top_cube":
        return f"From above the {object_name}, grasp it and lift it up."
    if skill_mode == "move_over_cube":
        return f"Move the gripper over the {object_name}."
    return f"Grasp the {object_name} and lift it up."


def _task_indices(episode_tasks: dict[int, str]) -> dict[str, int]:
    task_indices: dict[str, int] = {}
    for episode in sorted(episode_tasks):
        task = episode_tasks[episode]
        if task not in task_indices:
            task_indices[task] = len(task_indices)
    return task_indices


def _write_task_metadata(output_root: Path, *, episode_tasks: dict[int, str], task_indices: dict[str, int]) -> None:
    rows = sorted((index, task) for task, index in task_indices.items())
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([index for index, _task in rows], type=pa.int64()),
                "task": pa.array([task for _index, task in rows], type=pa.large_string()),
            }
        ),
        output_root / "meta" / "tasks.parquet",
    )

    info_path = output_root / "meta" / "info.json"
    info = _read_json(info_path)
    info["total_tasks"] = len(task_indices)
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    for parquet_path in sorted((output_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path)
        columns = table.to_pydict()
        task_values: list[list[str]] = []
        task_index_values: list[int] = []
        for episode in columns["episode_index"]:
            task = episode_tasks[int(episode)]
            task_values.append([task])
            task_index_values.append(task_indices[task])
        updated = table
        tasks_index = updated.schema.get_field_index("tasks")
        if tasks_index >= 0:
            updated = updated.set_column(tasks_index, "tasks", pa.array(task_values, type=updated.schema.field(tasks_index).type))
        for suffix in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
            column = f"stats/task_index/{suffix}"
            field_index = updated.schema.get_field_index(column)
            if field_index >= 0:
                values = [[float(value)] for value in task_index_values]
                if suffix in {"min", "max"}:
                    values = [[int(value)] for value in task_index_values]
                updated = updated.set_column(field_index, column, pa.array(values, type=updated.schema.field(field_index).type))
        field_index = updated.schema.get_field_index("stats/task_index/std")
        if field_index >= 0:
            updated = updated.set_column(field_index, "stats/task_index/std", pa.array([[0.0] for _ in task_index_values], type=updated.schema.field(field_index).type))
        pq.write_table(updated, parquet_path)


def _write_manifest(
    *,
    output_root: Path,
    source_dataset_root: Path,
    rendered_dir: Path,
    repo_id: str,
    camera_keys: tuple[str, ...],
    replaced_frames: int,
    replaced_images: int,
    duplicate_camera3_from_camera2: bool,
    episode_tasks: dict[int, str],
) -> None:
    info = _read_json(output_root / "meta" / "info.json")
    total_episodes = int(info.get("total_episodes") or 0)
    total_frames = int(info.get("total_frames") or replaced_frames)
    manifest = {
        "format": "so101_photoreal_lerobot_v1",
        "repo_id": repo_id,
        "source_dataset_root": str(source_dataset_root),
        "source_dataset_name": source_dataset_root.name,
        "rendered_dir": str(rendered_dir),
        "episodes": total_episodes,
        "frames": total_frames,
        "fps": info.get("fps"),
        "camera_keys": list(camera_keys),
        "camera_contract": PHOTO_CAMERA_CONTRACT,
        "duplicate_camera3_from_camera2": duplicate_camera3_from_camera2,
        "replaced_frames": replaced_frames,
        "replaced_images": replaced_images,
        "training_ready": replaced_images == total_frames * len(camera_keys),
        "task_prompt_rewrite": bool(episode_tasks),
        "task_prompts": sorted(set(episode_tasks.values())),
        "note": "LeRobot parquet root with photoreal image bytes and original state/action/timestamp columns.",
    }
    (output_root / "photoreal_lerobot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_manifest(output_root: Path) -> dict[str, Any]:
    return json.loads((output_root / "photoreal_lerobot_manifest.json").read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
