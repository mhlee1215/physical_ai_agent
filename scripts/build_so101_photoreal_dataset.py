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
CAMERA_CONTRACT = {
    "observation.images.camera1": "photoreal_render",
    "source.observation.images.camera1": "egocentric_cam",
    "source.observation.images.camera2": "wrist_cam",
    "source.observation.images.camera3": "wrist_cam duplicate",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small SO101 photoreal image dataset from rendered frames.")
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = build_photoreal_dataset(
        source_dataset_root=args.source_dataset_root,
        rendered_dir=args.rendered_dir,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_photoreal_dataset(
    *,
    source_dataset_root: Path,
    rendered_dir: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    images_root = output_root / "images"
    episodes_root = output_root / "episodes"
    images_root.mkdir()
    episodes_root.mkdir()

    rendered = _rendered_frames(rendered_dir)
    if not rendered:
        raise FileNotFoundError(f"no rendered episode_XXXX_frame_YYYY.png files under {rendered_dir}")
    rows = _source_rows(source_dataset_root, wanted={(episode, frame) for episode, frame, _path in rendered})
    tasks = _tasks(source_dataset_root)
    source_report = _source_export_report(source_dataset_root)

    by_episode: dict[int, list[tuple[int, Path]]] = {}
    for episode, frame, path in rendered:
        by_episode.setdefault(episode, []).append((frame, path))

    episode_summaries: list[dict[str, Any]] = []
    total_frames = 0
    for local_episode, source_episode in enumerate(sorted(by_episode)):
        episode_rows = sorted(by_episode[source_episode])
        episode_path = episodes_root / f"episode_{local_episode:04d}.jsonl"
        source_summary = source_report.get(source_episode, {})
        with episode_path.open("w", encoding="utf-8") as file:
            for local_frame, (source_frame, image_path) in enumerate(episode_rows):
                key = (source_episode, source_frame)
                if key not in rows:
                    raise ValueError(f"missing source row for episode={source_episode} frame={source_frame}")
                row = rows[key]
                image_rel = Path("images") / f"episode_{local_episode:04d}" / f"frame_{local_frame:04d}.png"
                image_out = output_root / image_rel
                image_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, image_out)
                task_index = int(row.get("task_index") or 0)
                record = {
                    "episode_index": local_episode,
                    "frame_index": local_frame,
                    "timestamp": float(row["timestamp"]),
                    "task_index": task_index,
                    "task": tasks.get(task_index, ""),
                    "prompt": tasks.get(task_index, ""),
                    "source_episode_index": source_episode,
                    "source_frame_index": source_frame,
                    "source_seed": source_summary.get("seed"),
                    "observation": {
                        "state": [float(value) for value in row["observation.state"]],
                        "images": {
                            "camera1": str(image_rel),
                        },
                    },
                    "action": [float(value) for value in row["action"]],
                }
                file.write(json.dumps(record, sort_keys=True) + "\n")
        episode_summaries.append(
            {
                "episode_index": local_episode,
                "source_episode_index": source_episode,
                "source_seed": source_summary.get("seed"),
                "frames": len(episode_rows),
                "source_frames": [frame for frame, _path in episode_rows],
            }
        )
        total_frames += len(episode_rows)

    manifest = {
        "format": "so101_photoreal_jsonl_v1",
        "source_dataset_root": str(source_dataset_root),
        "source_dataset_name": source_dataset_root.name,
        "rendered_dir": str(rendered_dir),
        "episodes": len(episode_summaries),
        "frames": total_frames,
        "fps": _fps(source_dataset_root),
        "image_mime_type": "image/png",
        "image_shape": [480, 640, 3],
        "features": ["observation.images.camera1", "observation.state", "action"],
        "joint_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "camera_contract": CAMERA_CONTRACT,
        "training_ready": False,
        "note": "Photoreal image dataset built from rendered source LeRobot rows. It is a compact visual dataset, not a full-frame replacement of the original training set.",
        "episode_summaries": episode_summaries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_root": str(output_root), **manifest}


def _rendered_frames(rendered_dir: Path) -> list[tuple[int, int, Path]]:
    rows = []
    for path in sorted(rendered_dir.glob("episode_*_frame_*.png")):
        match = re.fullmatch(r"episode_(\d+)_frame_(\d+)\.png", path.name)
        if not match:
            continue
        rows.append((int(match.group(1)), int(match.group(2)), path))
    return rows


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


def _rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    count = len(next(iter(columns.values()))) if columns else 0
    return [{key: value[index] for key, value in columns.items()} for index in range(count)]


if __name__ == "__main__":
    main()
