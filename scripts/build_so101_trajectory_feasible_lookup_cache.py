#!/usr/bin/env python3
"""Build a deterministic bin -> proven teacher trajectory lookup from an export report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_cache(report: dict[str, Any], *, grid_size: int, resolution: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> dict[str, Any]:
    lookup: dict[str, list[list[float]]] = {}
    seen_seeds: set[int] = set()
    for episode in report.get("episodes", []):
        xy = episode.get("forced_spawn_xy")
        bin_id = episode.get("grid_balance_bin")
        seed = episode.get("seed")
        if bin_id is None or not isinstance(xy, list) or len(xy) < 2 or seed is None:
            raise ValueError("every calibration episode must contain grid_balance_bin, forced_spawn_xy, and seed")
        episode_seed = int(seed)
        if episode_seed in seen_seeds:
            raise ValueError(f"duplicate calibration seed is forbidden: seed={episode_seed}")
        seen_seeds.add(episode_seed)
        lookup.setdefault(str(int(bin_id)), []).append([float(xy[0]), float(xy[1]), episode_seed])
    if not lookup:
        raise ValueError("calibration report contains no forced-spawn episodes")
    return {
        "format": "so101_camera1_spawn_lookup_v1",
        "candidate_kind": "trajectory_feasible",
        "source_report": report.get("report_path"),
        "grid_size": int(grid_size),
        "resolution": int(resolution),
        "x_range": [float(x_range[0]), float(x_range[1])],
        "y_range": [float(y_range[0]), float(y_range[1])],
        "seed_uniqueness": {
            "required": True,
            "unique_seed_count": len(seen_seeds),
            "duplicate_seed_count": 0,
        },
        "lookup": lookup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=21)
    parser.add_argument("--x-min", type=float, default=-0.10)
    parser.add_argument("--x-max", type=float, default=0.55)
    parser.add_argument("--y-min", type=float, default=-0.45)
    parser.add_argument("--y-max", type=float, default=0.45)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    payload = build_cache(
        report,
        grid_size=args.grid_size,
        resolution=args.resolution,
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bins": {key: len(value) for key, value in payload["lookup"].items()}}, indent=2))


if __name__ == "__main__":
    main()
