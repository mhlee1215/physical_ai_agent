#!/usr/bin/env python3
"""Extract a seed-free SO101 spawn catalog from successful episode reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_so101_source_episode_spawn_lookup import build_spawn_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--source-report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-size", required=True, type=int)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--x-min", required=True, type=float)
    parser.add_argument("--x-max", required=True, type=float)
    parser.add_argument("--y-min", required=True, type=float)
    parser.add_argument("--y-max", required=True, type=float)
    parser.add_argument("--bins", required=True)
    args = parser.parse_args()

    payload = build_spawn_catalog(
        catalog_id=args.catalog_id,
        source_reports=args.source_report,
        grid_size=args.grid_size,
        resolution=args.resolution,
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
        bins=[int(value) for value in args.bins.split(",") if value.strip()],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "candidate_counts": payload["candidate_counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
