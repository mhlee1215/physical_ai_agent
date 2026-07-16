#!/usr/bin/env python3
"""Build a deterministic spawn lookup from successful SO101 episode reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-size", required=True, type=int)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--x-min", required=True, type=float)
    parser.add_argument("--x-max", required=True, type=float)
    parser.add_argument("--y-min", required=True, type=float)
    parser.add_argument("--y-max", required=True, type=float)
    parser.add_argument("--bins", required=True)
    parser.add_argument(
        "--candidate-start-index",
        type=int,
        default=0,
        help="Reserve earlier source candidates by slicing every requested bin from this index.",
    )
    args = parser.parse_args()

    payload = build_source_lookup(
        source_reports=args.source_report,
        grid_size=args.grid_size,
        resolution=args.resolution,
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
        bins=[int(value) for value in args.bins.split(",") if value.strip()],
        candidate_start_index=args.candidate_start_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidate_counts": payload["candidate_counts"]}, indent=2))


def build_source_lookup(
    *,
    source_reports: list[Path],
    grid_size: int,
    resolution: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: list[int],
    candidate_start_index: int = 0,
) -> dict:
    if candidate_start_index < 0:
        raise ValueError("candidate_start_index must be non-negative")
    lookup = {int(bin_id): [] for bin_id in range(int(grid_size) ** 2)}
    seen_seeds: set[int] = set()
    sources = []
    for report_path in source_reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        accepted = 0
        for episode in report.get("episodes", []):
            if not bool(episode.get("success", False)):
                continue
            bin_id = episode.get("grid_balance_bin")
            spawn_xy = episode.get("forced_spawn_xy")
            seed = episode.get("seed")
            if bin_id is None or spawn_xy is None or seed is None:
                continue
            bin_id = int(bin_id)
            seed = int(seed)
            if bin_id not in lookup or bin_id not in bins or seed in seen_seeds:
                continue
            lookup[bin_id].append([float(spawn_xy[0]), float(spawn_xy[1]), seed])
            seen_seeds.add(seed)
            accepted += 1
        sources.append({"report": str(report_path), "accepted": accepted})
    raw_candidate_counts = {str(key): len(value) for key, value in sorted(lookup.items())}
    if candidate_start_index:
        lookup = {
            bin_id: values[candidate_start_index:] if bin_id in bins else values
            for bin_id, values in lookup.items()
        }
    missing = [bin_id for bin_id in bins if not lookup[bin_id]]
    if missing:
        raise ValueError(f"source episode reports have no successful candidates for bins: {missing}")
    return {
        "format": "so101_camera1_spawn_lookup_v1",
        "candidate_kind": "source_episode_manifest",
        "grid_size": int(grid_size),
        "resolution": int(resolution),
        "x_range": [float(value) for value in x_range],
        "y_range": [float(value) for value in y_range],
        "lookup": {str(key): value for key, value in sorted(lookup.items())},
        "candidate_counts": {str(key): len(value) for key, value in sorted(lookup.items())},
        "raw_candidate_counts": raw_candidate_counts,
        "candidate_start_index": int(candidate_start_index),
        "source_reports": sources,
    }


if __name__ == "__main__":
    main()
