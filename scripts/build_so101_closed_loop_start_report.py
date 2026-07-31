#!/usr/bin/env python3
"""Build a balanced closed-loop start report from exported dataset episodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _spawn_key(episode: dict[str, Any]) -> tuple[float, float] | None:
    value = episode.get("forced_spawn_xy")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return (round(float(value[0]), 9), round(float(value[1]), 9))


def collect_excluded_episode_identities(
    reports: list[tuple[Path, dict[str, Any]]],
) -> tuple[set[int], set[tuple[float, float]]]:
    seeds: set[int] = set()
    spawn_xy: set[tuple[float, float]] = set()
    for path, report in reports:
        episodes = report.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"excluded source report must contain an episodes list: {path}")
        for episode in episodes:
            seed = episode.get("seed")
            if seed is not None:
                seeds.add(int(seed))
            key = _spawn_key(episode)
            if key is not None:
                spawn_xy.add(key)
    return seeds, spawn_xy


def select_balanced_episodes(
    episodes: list[dict[str, Any]],
    *,
    count: int,
    bins: list[int],
    excluded_seeds: set[int] | None = None,
    excluded_spawn_xy: set[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    excluded_seeds = excluded_seeds or set()
    excluded_spawn_xy = excluded_spawn_xy or set()
    buckets: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, episode in enumerate(episodes):
        seed = episode.get("seed")
        if seed is not None and int(seed) in excluded_seeds:
            continue
        spawn_key = _spawn_key(episode)
        if spawn_key is not None and spawn_key in excluded_spawn_xy:
            continue
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


def build_report(
    source: dict[str, Any],
    *,
    count: int,
    bins: list[int],
    source_path: Path,
    success_metric: str | None = None,
    lift_success_height: float | None = None,
    excluded_sources: list[tuple[Path, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("source report must contain an episodes list")
    excluded_sources = excluded_sources or []
    excluded_seeds, excluded_spawn_xy = collect_excluded_episode_identities(excluded_sources)
    excluded_validation_episodes = sum(
        1
        for episode in episodes
        if (
            episode.get("seed") is not None
            and int(episode["seed"]) in excluded_seeds
        )
        or (_spawn_key(episode) in excluded_spawn_xy)
    )
    selected = select_balanced_episodes(
        episodes,
        count=count,
        bins=bins,
        excluded_seeds=excluded_seeds,
        excluded_spawn_xy=excluded_spawn_xy,
    )
    counts = {str(bin_id): 0 for bin_id in bins}
    for episode in selected:
        bin_id = int(episode.get("grid_balance_bin", episode.get("desired_grid_bin")))
        counts[str(bin_id)] += 1
    report = {
        "operation": "build_so101_closed_loop_start_report",
        "source_validation_report": str(source_path),
        "selection": "round_robin_grid_bin_from_validation_first_state",
        "requested_episodes": count,
        "grid_bins": bins,
        "grid_bin_counts": counts,
        "episodes": selected,
    }
    if excluded_sources:
        report["exclusion_contract"] = {
            "source_reports": [str(path) for path, _report in excluded_sources],
            "match_keys": ["seed", "forced_spawn_xy_rounded_9dp"],
            "excluded_seed_count": len(excluded_seeds),
            "excluded_spawn_xy_count": len(excluded_spawn_xy),
            "excluded_validation_episodes": excluded_validation_episodes,
        }
    if success_metric is not None:
        report["success_metric"] = str(success_metric)
    if lift_success_height is not None:
        report["lift_success_height"] = float(lift_success_height)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--grid-bins", default="5,6,9,10")
    parser.add_argument("--success-metric")
    parser.add_argument("--lift-success-height", type=float)
    parser.add_argument(
        "--exclude-source-report",
        type=Path,
        action="append",
        default=[],
        help="Exclude validation starts whose seed or forced_spawn_xy appears in this report.",
    )
    args = parser.parse_args()
    bins = [int(value.strip()) for value in args.grid_bins.split(",") if value.strip()]
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    excluded_sources = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in args.exclude_source_report
    ]
    report = build_report(
        source,
        count=args.episodes,
        bins=bins,
        source_path=args.source_report,
        success_metric=args.success_metric,
        lift_success_height=args.lift_success_height,
        excluded_sources=excluded_sources,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("requested_episodes", "grid_bin_counts")}, indent=2))


if __name__ == "__main__":
    main()
