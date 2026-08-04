#!/usr/bin/env python3
"""Build a balanced closed-loop start report from exported dataset episodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from physical_ai_agent.so101_closed_loop_contract import (
    build_executable_loop_test_contract,
    contract_path_for_start_report,
    observation_renderer_from_camera_rig,
    write_executable_loop_test_contract,
)


def _spawn_key(episode: dict[str, Any]) -> tuple[float, float] | None:
    value = episode.get("forced_spawn_xy")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return (round(float(value[0]), 9), round(float(value[1]), 9))


def _episode_grid_bin(episode: dict[str, Any]) -> int | None:
    """Resolve the camera1 bin across legacy and workspace-catalog reports."""
    for key in ("camera1_grid_bin", "grid_balance_bin", "desired_grid_bin"):
        value = episode.get(key)
        if value is not None:
            return int(value)
    workspace_spawn = episode.get("workspace_spawn")
    if isinstance(workspace_spawn, dict) and workspace_spawn.get("camera1_grid_bin") is not None:
        return int(workspace_spawn["camera1_grid_bin"])
    return None


def apply_grid_sidecar(source: dict[str, Any], sidecar: Path) -> dict[str, Any]:
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("source report must contain an episodes list")
    rows = pq.read_table(
        sidecar,
        columns=["episode_index", "visible", "grid_bin"],
    ).to_pylist()
    by_episode = {int(row["episode_index"]): row for row in rows}
    missing = sorted(set(range(len(episodes))) - set(by_episode))
    if missing:
        raise ValueError(
            f"grid sidecar is missing {len(missing)} source episodes: {missing[:8]}"
        )
    enriched = dict(source)
    enriched["episodes"] = []
    for episode_index, episode in enumerate(episodes):
        row = by_episode[episode_index]
        grid_bin = int(row["grid_bin"])
        enriched["episodes"].append(
            {
                **episode,
                "camera1_grid_bin": grid_bin if bool(row["visible"]) else -1,
                "camera1_grid_visible": bool(row["visible"]),
            }
        )
    return enriched


def available_grid_bins(source: dict[str, Any]) -> list[int]:
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("source report must contain an episodes list")
    return sorted(
        {
            int(bin_id)
            for episode in episodes
            if (bin_id := _episode_grid_bin(episode)) is not None and int(bin_id) >= 0
        }
    )


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
        bin_id = _episode_grid_bin(episode)
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
    grid_bin_source: str = "source_report",
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
        bin_id = _episode_grid_bin(episode)
        if bin_id is None:  # Defensive: selection already rejects missing bins.
            raise ValueError("selected episode is missing a camera1 grid bin")
        counts[str(bin_id)] += 1
    report = {
        "operation": "build_so101_closed_loop_start_report",
        "source_validation_report": str(source_path),
        "selection": "round_robin_grid_bin_from_validation_first_state",
        "requested_episodes": count,
        "grid_bins": bins,
        "grid_bin_counts": counts,
        "grid_bin_source": grid_bin_source,
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


def build_contract(
    *,
    start_report_path: Path,
    dataset_root: Path,
    dataset_name: str,
    dataset_repo_id: str,
    test_case_id: str,
    description: str,
    episodes: int,
    steps: int,
    seed: int,
    task_prompt: str,
    success_metric: str,
    camera_rig_config: str,
    target_object_color: str,
    object_half_sizes: list[float],
    spawn_center: tuple[float, float],
    spawn_min_radius: float,
    spawn_max_radius: float,
    spawn_angle_half_range_deg: float,
    source_recipe: str,
    source_split: str,
) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    camera_rig_path = Path(camera_rig_config)
    renderer = observation_renderer_from_camera_rig(
        camera_rig_path,
        camera_rig_config=camera_rig_config,
    )
    return build_executable_loop_test_contract(
        test_case_id=test_case_id,
        description=description,
        episodes=episodes,
        steps=steps,
        seed=seed,
        task_prompt=task_prompt,
        success_metric=success_metric,
        start_report_path=str(start_report_path),
        start_dataset={
            "name": dataset_name,
            "repo_id": dataset_repo_id,
            "root": str(dataset_root),
            "expected_episodes": int(info["total_episodes"]),
            "expected_frames": int(info["total_frames"]),
        },
        env_config={
            "camera_rig_config": camera_rig_config,
            "target_object_color": target_object_color,
            "object_half_sizes": object_half_sizes,
            "spawn_center": list(spawn_center),
            "spawn_min_radius": spawn_min_radius,
            "spawn_max_radius": spawn_max_radius,
            "spawn_angle_half_range_deg": spawn_angle_half_range_deg,
        },
        observation_renderer=renderer,
        source_recipe=source_recipe,
        source_split=source_split,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--grid-bins", default="5,6,9,10")
    parser.add_argument("--grid-sidecar", type=Path)
    parser.add_argument("--success-metric")
    parser.add_argument("--lift-success-height", type=float)
    parser.add_argument("--write-executable-contract", action="store_true")
    parser.add_argument("--contract-output", type=Path)
    parser.add_argument("--contract-id")
    parser.add_argument("--contract-description")
    parser.add_argument("--contract-steps", type=int, default=200)
    parser.add_argument("--contract-seed", type=int, default=98100)
    parser.add_argument("--task-prompt")
    parser.add_argument("--start-dataset-name")
    parser.add_argument("--start-dataset-root", type=Path)
    parser.add_argument("--start-dataset-repo-id")
    parser.add_argument("--source-recipe")
    parser.add_argument("--source-split")
    parser.add_argument("--camera-rig-config")
    parser.add_argument("--target-object-color")
    parser.add_argument("--object-half-sizes")
    parser.add_argument("--spawn-center-x", type=float)
    parser.add_argument("--spawn-center-y", type=float)
    parser.add_argument("--spawn-min-radius", type=float)
    parser.add_argument("--spawn-max-radius", type=float)
    parser.add_argument("--spawn-angle-half-range-deg", type=float)
    parser.add_argument(
        "--exclude-source-report",
        type=Path,
        action="append",
        default=[],
        help="Exclude validation starts whose seed or forced_spawn_xy appears in this report.",
    )
    args = parser.parse_args()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    grid_bin_source = "source_report"
    if args.grid_sidecar is not None:
        source = apply_grid_sidecar(source, args.grid_sidecar)
        grid_bin_source = str(args.grid_sidecar)
    bins = (
        available_grid_bins(source)
        if args.grid_bins == "auto"
        else [int(value.strip()) for value in args.grid_bins.split(",") if value.strip()]
    )
    if not bins:
        raise ValueError("closed-loop source has no visible camera1 grid bins")
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
        grid_bin_source=grid_bin_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    contract_output = None
    if args.write_executable_contract:
        required = {
            "contract_id": args.contract_id,
            "task_prompt": args.task_prompt,
            "start_dataset_name": args.start_dataset_name,
            "start_dataset_root": args.start_dataset_root,
            "start_dataset_repo_id": args.start_dataset_repo_id,
            "source_recipe": args.source_recipe,
            "source_split": args.source_split,
            "camera_rig_config": args.camera_rig_config,
            "target_object_color": args.target_object_color,
            "object_half_sizes": args.object_half_sizes,
            "spawn_center_x": args.spawn_center_x,
            "spawn_center_y": args.spawn_center_y,
            "spawn_min_radius": args.spawn_min_radius,
            "spawn_max_radius": args.spawn_max_radius,
            "spawn_angle_half_range_deg": args.spawn_angle_half_range_deg,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "executable loop-test contract is missing arguments: "
                + ", ".join(missing)
            )
        contract = build_contract(
            start_report_path=args.output,
            dataset_root=args.start_dataset_root,
            dataset_name=args.start_dataset_name,
            dataset_repo_id=args.start_dataset_repo_id,
            test_case_id=args.contract_id,
            description=args.contract_description
            or f"Held-out starts for {args.start_dataset_name}.",
            episodes=args.episodes,
            steps=args.contract_steps,
            seed=args.contract_seed,
            task_prompt=args.task_prompt,
            success_metric=args.success_metric or "env_success",
            camera_rig_config=args.camera_rig_config,
            target_object_color=args.target_object_color,
            object_half_sizes=[
                float(value)
                for value in args.object_half_sizes.split(",")
                if value.strip()
            ],
            spawn_center=(args.spawn_center_x, args.spawn_center_y),
            spawn_min_radius=args.spawn_min_radius,
            spawn_max_radius=args.spawn_max_radius,
            spawn_angle_half_range_deg=args.spawn_angle_half_range_deg,
            source_recipe=args.source_recipe,
            source_split=args.source_split,
        )
        contract_output = write_executable_loop_test_contract(
            args.contract_output or contract_path_for_start_report(args.output),
            contract,
        )
    summary = {
        key: report[key] for key in ("requested_episodes", "grid_bin_counts")
    }
    if contract_output is not None:
        summary["executable_contract"] = str(contract_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
