#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from export_so101_teacher_rollouts_lerobot import _lerobot_features, audit_lerobot_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SO101 LeRobotDataset shards without loading all frames.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = merge_shards(
        output_root=args.output_root,
        repo_id=args.repo_id,
        shard_roots=args.shard,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def merge_shards(*, output_root: Path, repo_id: str, shard_roots: list[Path], overwrite: bool) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    shard_reports = [_load_shard_report(root) for root in shard_roots]
    first_report = shard_reports[0]
    include_camera3 = bool(first_report.get("camera3_duplicate", {}).get("enabled", True))
    width, height = _image_size_from_report(first_report)
    features = _lerobot_features(
        height=height,
        width=width,
        use_videos=False,
        include_camera3_duplicate=include_camera3,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(first_report["fps"]),
        features=features,
        root=output_root,
        robot_type="so101",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    merged_episodes = 0
    merged_frames = 0
    shard_summaries = []
    for shard_root, shard_report in zip(shard_roots, shard_reports, strict=True):
        source = LeRobotDataset(str(shard_report["repo_id"]), root=shard_root)
        source_frames = 0
        source_episodes = 0
        for episode in source.meta.episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            for index in range(start, end):
                dataset.add_frame(_frame_from_source(source[index], include_camera3=include_camera3))
                source_frames += 1
            dataset.save_episode()
            source_episodes += 1
        merged_episodes += source_episodes
        merged_frames += source_frames
        shard_summaries.append(
            {
                "root": str(shard_root),
                "repo_id": shard_report["repo_id"],
                "episodes": source_episodes,
                "frames": source_frames,
                "audit_status": shard_report.get("audit", {}).get("status"),
            }
        )

    dataset.finalize()
    action_space_low = np.asarray(first_report["audit"]["action_space_low"], dtype=np.float32)
    action_space_high = np.asarray(first_report["audit"]["action_space_high"], dtype=np.float32)
    audit = audit_lerobot_dataset(
        root=output_root,
        repo_id=repo_id,
        features=features,
        action_space_low=action_space_low,
        action_space_high=action_space_high,
    )
    report = {
        "operation": "merge_so101_lerobot_shards",
        "output_root": str(output_root),
        "repo_id": repo_id,
        "merged_episodes": merged_episodes,
        "merged_frames": merged_frames,
        "shards": shard_summaries,
        "audit": audit,
    }
    report_path = output_root / "so101_lerobot_merge_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_shard_report(root: Path) -> dict[str, Any]:
    report_path = root / "so101_lerobot_export_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing shard report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _image_size_from_report(report: dict[str, Any]) -> tuple[int, int]:
    shape = report["audit"]["declared_features"]["observation.images.camera1"]["shape"]
    return int(shape[1]), int(shape[0])


def _frame_from_source(sample: dict[str, Any], *, include_camera3: bool) -> dict[str, Any]:
    frame = {
        "observation.images.camera1": _image_hwc_uint8(sample["observation.images.camera1"]),
        "observation.images.camera2": _image_hwc_uint8(sample["observation.images.camera2"]),
        "observation.state": _array(sample["observation.state"], dtype=np.float32),
        "action": _array(sample["action"], dtype=np.float32),
        "task": str(sample["task"]),
    }
    if include_camera3:
        frame["observation.images.camera3"] = _image_hwc_uint8(sample["observation.images.camera3"])
    return frame


def _image_hwc_uint8(value: Any) -> np.ndarray:
    array = _array(value, dtype=np.float32)
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return array


def _array(value: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


if __name__ == "__main__":
    main()
