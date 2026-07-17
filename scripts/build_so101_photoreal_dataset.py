#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_CAMERA_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]
SOURCE_CAMERA_CONTRACT = {
    "observation.images.camera1": "egocentric_cam",
    "observation.images.camera2": "wrist_cam",
    "observation.images.camera3": "wrist_cam duplicate",
}
PHOTO_CAMERA_CONTRACT = {
    "observation.images.camera1": "photoreal egocentric_cam",
    "observation.images.camera2": "photoreal wrist_cam",
    "observation.images.camera3": "photoreal wrist_cam duplicate",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SO101 photoreal image dataset from rendered frames. "
            "By default this is strict: selected episodes must preserve the source "
            "episode lengths and all camera keys."
        )
    )
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", default="0,1,2,3,4", help="Comma-separated source episode indices.")
    parser.add_argument("--camera-keys", default=",".join(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--allow-frame-subset", action="store_true")
    parser.add_argument(
        "--duplicate-camera3-from-camera2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow camera3 to reuse camera2 renders because the source camera3 is the wrist duplicate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = build_photoreal_dataset(
        source_dataset_root=args.source_dataset_root,
        rendered_dir=args.rendered_dir,
        output_root=args.output_root,
        episodes=_parse_int_csv(args.episodes),
        camera_keys=[item.strip() for item in args.camera_keys.split(",") if item.strip()],
        allow_frame_subset=args.allow_frame_subset,
        duplicate_camera3_from_camera2=args.duplicate_camera3_from_camera2,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_photoreal_dataset(
    *,
    source_dataset_root: Path,
    rendered_dir: Path,
    output_root: Path,
    episodes: list[int],
    camera_keys: list[str],
    allow_frame_subset: bool,
    duplicate_camera3_from_camera2: bool,
    overwrite: bool,
) -> dict[str, Any]:
    source_episode_lengths = _episode_lengths(source_dataset_root)
    missing_episodes = [episode for episode in episodes if episode not in source_episode_lengths]
    if missing_episodes:
        raise ValueError(f"source episodes not found: {missing_episodes}")

    rendered_index = _rendered_index(rendered_dir)
    plan = _build_plan(
        source_episode_lengths=source_episode_lengths,
        episodes=episodes,
        camera_keys=camera_keys,
        rendered_index=rendered_index,
        allow_frame_subset=allow_frame_subset,
        duplicate_camera3_from_camera2=duplicate_camera3_from_camera2,
    )
    rows = _source_rows(source_dataset_root, wanted={(row["source_episode"], row["source_frame"]) for row in plan})
    tasks = _tasks(source_dataset_root)
    source_report = _source_export_report(source_dataset_root)
    image_shape = _source_image_shape(source_dataset_root, camera_keys)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    (output_root / "images").mkdir()
    (output_root / "episodes").mkdir()

    episode_summaries: list[dict[str, Any]] = []
    total_frames = 0
    for local_episode, source_episode in enumerate(episodes):
        episode_plan = [row for row in plan if row["source_episode"] == source_episode]
        episode_path = output_root / "episodes" / f"episode_{local_episode:04d}.jsonl"
        source_summary = source_report.get(source_episode, {})
        frames = sorted({int(row["source_frame"]) for row in episode_plan})
        with episode_path.open("w", encoding="utf-8") as file:
            for local_frame, source_frame in enumerate(frames):
                key = (source_episode, source_frame)
                if key not in rows:
                    raise ValueError(f"missing source row for episode={source_episode} frame={source_frame}")
                source_row = rows[key]
                images: dict[str, str] = {}
                for camera_key in camera_keys:
                    render_row = next(
                        row for row in episode_plan if row["source_frame"] == source_frame and row["camera_key"] == camera_key
                    )
                    rel = (
                        Path("images")
                        / f"episode_{local_episode:04d}"
                        / _camera_slug(camera_key)
                        / f"frame_{local_frame:04d}.png"
                    )
                    out_path = output_root / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(render_row["path"], out_path)
                    images[_image_field(camera_key)] = str(rel)
                task_index = int(source_row.get("task_index") or 0)
                record = {
                    "episode_index": local_episode,
                    "frame_index": local_frame,
                    "timestamp": float(source_row["timestamp"]),
                    "task_index": task_index,
                    "task": tasks.get(task_index, ""),
                    "prompt": tasks.get(task_index, ""),
                    "source_episode_index": source_episode,
                    "source_frame_index": source_frame,
                    "source_seed": source_summary.get("seed"),
                    "observation": {
                        "state": [float(value) for value in source_row["observation.state"]],
                        "images": images,
                    },
                    "action": [float(value) for value in source_row["action"]],
                }
                file.write(json.dumps(record, sort_keys=True) + "\n")
        episode_summaries.append(
            {
                "episode_index": local_episode,
                "source_episode_index": source_episode,
                "source_seed": source_summary.get("seed"),
                "frames": len(frames),
                "source_frames": frames,
                "source_length": source_episode_lengths[source_episode],
            }
        )
        total_frames += len(frames)

    manifest = {
        "format": "so101_photoreal_jsonl_v1",
        "source_dataset_root": str(source_dataset_root),
        "source_dataset_name": source_dataset_root.name,
        "rendered_dir": str(rendered_dir),
        "episodes": len(episodes),
        "frames": total_frames,
        "fps": _fps(source_dataset_root),
        "image_mime_type": "image/png",
        "image_shape": image_shape,
        "features": [*camera_keys, "observation.state", "action"],
        "joint_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "camera_contract": PHOTO_CAMERA_CONTRACT,
        "source_camera_contract": SOURCE_CAMERA_CONTRACT,
        "training_ready": not allow_frame_subset,
        "allow_frame_subset": allow_frame_subset,
        "duplicate_camera3_from_camera2": duplicate_camera3_from_camera2,
        "note": (
            "Photoreal image dataset built from rendered source LeRobot rows. "
            "When allow_frame_subset is false, selected episodes preserve source frame counts and camera keys."
        ),
        "episode_summaries": episode_summaries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_root": str(output_root), **manifest}


def _build_plan(
    *,
    source_episode_lengths: dict[int, int],
    episodes: list[int],
    camera_keys: list[str],
    rendered_index: dict[tuple[int, int, str], Path],
    allow_frame_subset: bool,
    duplicate_camera3_from_camera2: bool,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    missing: list[str] = []
    for episode in episodes:
        if allow_frame_subset:
            candidate_frames = sorted({frame for ep, frame, _camera in rendered_index if ep == episode})
        else:
            candidate_frames = list(range(source_episode_lengths[episode]))
        for frame in candidate_frames:
            for camera_key in camera_keys:
                path = rendered_index.get((episode, frame, camera_key))
                if path is None and duplicate_camera3_from_camera2 and camera_key == "observation.images.camera3":
                    path = rendered_index.get((episode, frame, "observation.images.camera2"))
                if path is None:
                    missing.append(f"episode={episode} frame={frame} camera={camera_key}")
                    continue
                plan.append({"source_episode": episode, "source_frame": frame, "camera_key": camera_key, "path": path})
    if missing:
        sample = ", ".join(missing[:12])
        suffix = f" ... and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise ValueError(f"missing rendered frames: {sample}{suffix}")
    return plan


def _rendered_index(rendered_dir: Path) -> dict[tuple[int, int, str], Path]:
    index: dict[tuple[int, int, str], Path] = {}
    patterns = [
        re.compile(r"episode_(\d+)_frame_(\d+)_(camera\d)\.png$"),
        re.compile(r"episode_(\d+)_frame_(\d+)_(observation_images_camera\d)\.png$"),
    ]
    for path in sorted(rendered_dir.rglob("*.png")):
        for pattern in patterns:
            match = pattern.fullmatch(path.name)
            if not match:
                continue
            episode = int(match.group(1))
            frame = int(match.group(2))
            camera = match.group(3).replace("observation_images_", "")
            index[(episode, frame, f"observation.images.{camera}")] = path
            break
    return index


def _episode_lengths(source_dataset_root: Path) -> dict[int, int]:
    files = sorted((source_dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing parquet files under {source_dataset_root}")
    table = pq.read_table([str(path) for path in files], columns=["episode_index", "frame_index"])
    lengths: dict[int, int] = {}
    for row in _rows(table.to_pydict()):
        episode = int(row["episode_index"])
        frame = int(row["frame_index"])
        lengths[episode] = max(lengths.get(episode, 0), frame + 1)
    return lengths


def _source_rows(source_dataset_root: Path, *, wanted: set[tuple[int, int]]) -> dict[tuple[int, int], dict[str, Any]]:
    files = sorted((source_dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing parquet files under {source_dataset_root}")
    table = pq.read_table(
        [str(path) for path in files],
        columns=["episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"],
    )
    output = {}
    for row in _rows(table.to_pydict()):
        key = (int(row["episode_index"]), int(row["frame_index"]))
        if key in wanted:
            output[key] = row
    return output


def _tasks(source_dataset_root: Path) -> dict[int, str]:
    tasks_path = source_dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    table = pq.read_table(str(tasks_path))
    rows = _rows(table.to_pydict())
    return {int(row.get("task_index") or row.get("index") or 0): str(row.get("task") or "") for row in rows}


def _source_export_report(source_dataset_root: Path) -> dict[int, dict[str, Any]]:
    report_path = source_dataset_root / "so101_lerobot_export_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes") or []
    return {index: episode for index, episode in enumerate(episodes) if isinstance(episode, dict)}


def _fps(source_dataset_root: Path) -> int | None:
    info_path = source_dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text(encoding="utf-8")).get("fps")


def _source_image_shape(source_dataset_root: Path, camera_keys: list[str]) -> list[int]:
    info_path = source_dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return [480, 640, 3]
    features = json.loads(info_path.read_text(encoding="utf-8")).get("features") or {}
    for camera_key in camera_keys:
        shape = (features.get(camera_key) or {}).get("shape")
        if isinstance(shape, list) and len(shape) == 3:
            return [int(value) for value in shape]
    return [480, 640, 3]


def _image_field(camera_key: str) -> str:
    return camera_key.removeprefix("observation.images.")


def _camera_slug(camera_key: str) -> str:
    return camera_key.replace(".", "_")


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    count = len(next(iter(columns.values()))) if columns else 0
    return [{key: value[index] for key, value in columns.items()} for index in range(count)]


if __name__ == "__main__":
    main()
