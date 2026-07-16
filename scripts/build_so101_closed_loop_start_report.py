#!/usr/bin/env python3
"""Build a balanced closed-loop start report from exported dataset episodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def select_balanced_episodes(
    episodes: list[dict[str, Any]], *, count: int, bins: list[int]
) -> list[dict[str, Any]]:
    buckets: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, episode in enumerate(episodes):
        bin_id = episode.get("grid_balance_bin", episode.get("desired_grid_bin"))
        if bin_id in bins and isinstance(episode.get("sim_snapshot"), dict):
            buckets[int(bin_id)].append((source_index, episode))
    missing = [bin_id for bin_id in bins if not buckets[bin_id]]
    if missing:
        raise ValueError(f"source report has no restorable episodes for bins: {missing}")

    selected: list[dict[str, Any]] = []
    offsets = {bin_id: 0 for bin_id in bins}
    while len(selected) < count:
        progressed = False
        for bin_id in bins:
            offset = offsets[bin_id]
            if offset >= len(buckets[bin_id]):
                continue
            source_index, episode = buckets[bin_id][offset]
            selected.append({**episode, "source_validation_episode_index": source_index})
            offsets[bin_id] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"requested {count} starts, but only {len(selected)} are available")
    return selected


def build_report(source: dict[str, Any], *, count: int, bins: list[int], source_path: Path) -> dict[str, Any]:
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("source report must contain an episodes list")
    selected = select_balanced_episodes(episodes, count=count, bins=bins)
    counts = {str(bin_id): 0 for bin_id in bins}
    for episode in selected:
        bin_id = int(episode.get("grid_balance_bin", episode.get("desired_grid_bin")))
        counts[str(bin_id)] += 1
    return {
        "operation": "build_so101_closed_loop_start_report",
        "source_validation_report": str(source_path),
        "selection": "round_robin_grid_bin_from_validation_first_state",
        "requested_episodes": count,
        "grid_bins": bins,
        "grid_bin_counts": counts,
        "episodes": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--grid-bins", default="5,6,9,10")
    args = parser.parse_args()
    bins = [int(value.strip()) for value in args.grid_bins.split(",") if value.strip()]
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    report = build_report(source, count=args.episodes, bins=bins, source_path=args.source_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("requested_episodes", "grid_bin_counts")}, indent=2))


if __name__ == "__main__":
    main()
