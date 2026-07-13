#!/usr/bin/env python3
"""Fail when two generated SO101 splits overlap or violate their input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--expected-prompt", required=True)
    parser.add_argument("--expected-resolution", default="256x256")
    parser.add_argument("--expected-train-bins", required=True)
    parser.add_argument("--expected-validation-bins", required=True)
    parser.add_argument("--expected-terminal-hold-steps", type=int, required=True)
    parser.add_argument("--max-pre-close-alignment-deg", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    width, height = (int(value) for value in args.expected_resolution.lower().split("x", 1))
    report = audit_splits(
        train_root=args.train_root,
        validation_root=args.validation_root,
        expected_prompt=args.expected_prompt,
        expected_resolution=(width, height),
        expected_train_bins=_parse_bin_counts(args.expected_train_bins),
        expected_validation_bins=_parse_bin_counts(args.expected_validation_bins),
        expected_terminal_hold_steps=args.expected_terminal_hold_steps,
        max_pre_close_alignment_deg=args.max_pre_close_alignment_deg,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def audit_splits(
    *,
    train_root: Path,
    validation_root: Path,
    expected_prompt: str,
    expected_resolution: tuple[int, int],
    expected_train_bins: dict[int, int],
    expected_validation_bins: dict[int, int],
    expected_terminal_hold_steps: int,
    max_pre_close_alignment_deg: float,
) -> dict[str, Any]:
    common = {
        "expected_prompt": expected_prompt,
        "expected_resolution": expected_resolution,
        "expected_terminal_hold_steps": expected_terminal_hold_steps,
        "max_pre_close_alignment_deg": max_pre_close_alignment_deg,
    }
    train = _split_facts(train_root, expected_bins=expected_train_bins, **common)
    validation = _split_facts(
        validation_root, expected_bins=expected_validation_bins, **common
    )
    overlaps = {
        "seeds": sorted(train["seeds"] & validation["seeds"]),
        "spawn_xy": sorted(train["spawn_xy"] & validation["spawn_xy"]),
        "trajectory_hashes": sorted(train["trajectory_hashes"] & validation["trajectory_hashes"]),
    }
    failures = [name for name, values in overlaps.items() if values]
    if failures:
        raise ValueError(f"train/validation overlap detected: {', '.join(failures)}")
    return {
        "operation": "audit_so101_dataset_splits",
        "status": "passed",
        "train": _public_facts(train),
        "validation": _public_facts(validation),
        "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "expected_prompt": expected_prompt,
        "expected_resolution": list(expected_resolution),
        "expected_terminal_hold_steps": expected_terminal_hold_steps,
        "max_pre_close_alignment_deg": max_pre_close_alignment_deg,
    }


def _split_facts(
    root: Path,
    *,
    expected_prompt: str,
    expected_resolution: tuple[int, int],
    expected_bins: dict[int, int],
    expected_terminal_hold_steps: int,
    max_pre_close_alignment_deg: float,
) -> dict[str, Any]:
    report_path = root / "so101_lerobot_export_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes") or []
    actual_bins: dict[int, int] = {}
    for row in episodes:
        bin_id = int(row["grid_balance_bin"])
        actual_bins[bin_id] = actual_bins.get(bin_id, 0) + 1
        if not bool(row.get("success")) or not bool(row.get("task_success")):
            raise ValueError(f"unsuccessful teacher episode at {root}: seed={row.get('seed')}")
        hold_steps = int((row.get("phase_counts") or {}).get("terminal_hold", 0))
        if hold_steps != expected_terminal_hold_steps:
            raise ValueError(f"terminal hold mismatch at {root}: seed={row.get('seed')} hold={hold_steps}")
        alignment = float(row["pre_close_cube_face_normal_parallel_error_deg"])
        if alignment > max_pre_close_alignment_deg:
            raise ValueError(f"pre-close alignment exceeds limit at {root}: seed={row.get('seed')}")
    if actual_bins != expected_bins:
        raise ValueError(f"grid-bin counts mismatch at {root}: {actual_bins} != {expected_bins}")
    seeds = [int(row["seed"]) for row in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds within split: {root}")
    spawn_xy = [tuple(round(float(value), 9) for value in row["forced_spawn_xy"]) for row in episodes]
    if len(spawn_xy) != len(set(spawn_xy)):
        raise ValueError(f"duplicate spawn positions within split: {root}")

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    camera_feature = info["features"]["observation.images.camera1"]
    shape_by_name = dict(zip(camera_feature["names"], camera_feature["shape"], strict=True))
    actual_resolution = (int(shape_by_name["width"]), int(shape_by_name["height"]))
    if actual_resolution != expected_resolution:
        raise ValueError(f"camera1 resolution mismatch at {root}: {actual_resolution}")

    trajectory_hashes, task_indexes = _trajectory_hashes_and_task_indexes(root)
    task_table = pd.read_parquet(root / "meta" / "tasks.parquet").reset_index()
    task_by_index = {int(row.task_index): str(row.task) for row in task_table.itertuples()}
    if not task_indexes.issubset(task_by_index):
        raise ValueError(f"unknown task indexes at {root}: {sorted(task_indexes - task_by_index.keys())}")
    prompts = {task_by_index[index] for index in task_indexes}
    if prompts != {expected_prompt}:
        raise ValueError(f"prompt mismatch at {root}: {sorted(prompts)}")
    if len(trajectory_hashes) != len(set(trajectory_hashes)):
        raise ValueError(f"duplicate action/state trajectories within split: {root}")
    return {
        "root": str(root),
        "episodes": len(episodes),
        "seeds": set(seeds),
        "spawn_xy": set(spawn_xy),
        "trajectory_hashes": set(trajectory_hashes),
        "prompts": prompts,
        "resolution": actual_resolution,
        "bin_counts": actual_bins,
    }


def _trajectory_hashes_and_task_indexes(root: Path) -> tuple[list[str], set[int]]:
    frames = []
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        frames.append(
            pd.read_parquet(path, columns=["episode_index", "action", "observation.state", "task_index"])
        )
    if not frames:
        raise ValueError(f"no parquet files found under {root / 'data'}")
    table = pd.concat(frames, ignore_index=True)
    hashes = []
    for _, episode in table.groupby("episode_index", sort=True):
        digest = hashlib.sha256()
        digest.update(np.stack(episode["action"].to_numpy()).astype(np.float32).tobytes())
        digest.update(np.stack(episode["observation.state"].to_numpy()).astype(np.float32).tobytes())
        hashes.append(digest.hexdigest())
    return hashes, {int(value) for value in table["task_index"].unique()}


def _public_facts(facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": facts["root"],
        "episodes": facts["episodes"],
        "unique_seeds": len(facts["seeds"]),
        "unique_spawn_xy": len(facts["spawn_xy"]),
        "unique_trajectory_hashes": len(facts["trajectory_hashes"]),
        "prompts": sorted(facts["prompts"]),
        "resolution": list(facts["resolution"]),
        "bin_counts": {str(key): value for key, value in sorted(facts["bin_counts"].items())},
    }


def _parse_bin_counts(value: str) -> dict[int, int]:
    return {
        int(item.split(":", 1)[0]): int(item.split(":", 1)[1])
        for item in value.split(",")
        if item
    }


if __name__ == "__main__":
    main()
