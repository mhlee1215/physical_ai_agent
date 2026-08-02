#!/usr/bin/env python3
"""Build and gate JSON, Markdown, and HTML SO101 dataset distribution reports."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_dataset_generation_schema import (
    load_dataset_generation_recipe,
)
from physical_ai_agent.so101_workspace_spawn_catalog import (
    load_workspace_spawn_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    report = build_distribution_report(
        dataset_root=args.dataset_root,
        recipe_path=args.recipe,
        split_name=args.split,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["gate"]["status"] != "passed":
        raise SystemExit(2)


def build_distribution_report(
    *,
    dataset_root: Path,
    recipe_path: Path,
    split_name: str,
) -> dict[str, Any]:
    recipe_model = load_dataset_generation_recipe(recipe_path)
    recipe = recipe_model.as_dict()
    if split_name not in recipe["splits"]:
        raise ValueError(f"recipe does not define split: {split_name}")
    split = recipe["splits"][split_name]
    export_path = dataset_root / "so101_lerobot_export_report.json"
    if not export_path.is_file():
        raise FileNotFoundError(f"missing export report: {export_path}")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    episodes = list(export.get("episodes") or [])
    expected_episodes = _expected_episodes(split)

    seeds = [int(row["seed"]) for row in episodes if row.get("seed") is not None]
    successes = [bool(row.get("success", row.get("task_success", False))) for row in episodes]
    frame_counts = [int(row.get("frames", 0)) for row in episodes]
    prompts = Counter(str(row.get("task", "")) for row in episodes)
    workspace_rows = [
        row["workspace_spawn"]
        for row in episodes
        if isinstance(row.get("workspace_spawn"), dict)
    ]
    radii = [float(row["radius_from_base_m"]) for row in workspace_rows]
    angles = [float(row["angle_from_base_deg"]) for row in workspace_rows]
    object_yaws = [
        float(row["object_yaw_deg"])
        for row in workspace_rows
        if row.get("object_yaw_deg") is not None
    ]
    world_xy = [
        [float(value) for value in row["world_xy_m"]]
        for row in workspace_rows
    ]
    source_cells = [str(row["source_cell_id"]) for row in workspace_rows]
    grid_bins = [
        value
        for row in episodes
        if (value := _episode_camera1_grid_bin(row)) is not None
    ]

    sidecar = _load_camera_sidecar(dataset_root)
    if sidecar:
        camera_bin_counts = _flatten_grid_counts(sidecar.get("bin_counts_yx") or [])
        invisible = int(sidecar.get("invisible_episodes", 0))
    else:
        camera_bin_counts = dict(sorted(Counter(grid_bins).items()))
        invisible = 0
    all_policy_cameras_invisible = sum(
        not _episode_has_visible_policy_camera(row) for row in episodes
    )

    workspace_catalog_path = recipe["common"].get("workspace_spawn_catalog")
    workspace_catalog = (
        load_workspace_spawn_catalog(Path(workspace_catalog_path))
        if workspace_catalog_path
        else None
    )
    radial_bin_width_m = float(
        recipe["distribution_report"]["radial_histogram_bin_width_m"]
    )
    if workspace_catalog is not None and workspace_catalog.continuous_distribution:
        actual_radial_counts = _continuous_workspace_radial_counts(
            workspace_rows,
            workspace_catalog,
        )
        expected_radial_counts = _expected_workspace_radial_counts(
            recipe,
            split,
            workspace_catalog,
        )
        radial_area_density = _continuous_radial_area_density(
            actual_radial_counts,
            workspace_catalog,
        )
    else:
        expected_radial_counts = _expected_workspace_radial_counts(
            recipe,
            split,
            workspace_catalog,
        )
        actual_radial_counts = _radial_counts(
            radii,
            bin_width_m=radial_bin_width_m,
        )
        radial_area_density = {
            key: float(value) for key, value in actual_radial_counts.items()
        }
    radial_total_variation = _distribution_total_variation(
        actual_radial_counts,
        expected_radial_counts,
    )
    expected_source_cell_counts = _expected_workspace_cell_counts(
        split,
        workspace_catalog,
    )
    declared_cells = len(expected_source_cell_counts)
    if declared_cells == 0 and workspace_catalog is not None:
        declared_cells = int(workspace_catalog.source_cell_count)
    cell_coverage_ratio = (
        len(set(source_cells)) / declared_cells if declared_cells else None
    )
    source_cell_total_variation = _categorical_total_variation(
        dict(Counter(source_cells)),
        expected_source_cell_counts,
    )
    unique_xy = {
        (round(point[0], 12), round(point[1], 12)) for point in world_xy
    }
    radius_span_m = _numeric_span(radii)
    angle_span_deg = _numeric_span(angles)
    spec = recipe["distribution_report"]
    yaw_distribution = (
        None
        if workspace_catalog is None
        or workspace_catalog.continuous_distribution is None
        else workspace_catalog.continuous_distribution.object_yaw
    )
    yaw_periodicity_deg = float(spec["object_yaw_periodicity_deg"])
    periodic_object_yaws = [
        _periodic_degrees(value, period=yaw_periodicity_deg)
        for value in object_yaws
    ]
    object_yaw_occupancy = _object_yaw_occupancy(
        periodic_object_yaws,
        bins=int(spec["object_yaw_histogram_bins"]),
        bounds=(0.0, yaw_periodicity_deg),
    )
    relative_object_yaws = [
        _periodic_degrees(
            float(row["object_yaw_deg"]) - float(row["angle_from_base_deg"]),
            period=yaw_periodicity_deg,
        )
        for row in workspace_rows
        if row.get("object_yaw_deg") is not None
        and row.get("angle_from_base_deg") is not None
    ]
    relative_object_yaw_occupancy = _object_yaw_occupancy(
        relative_object_yaws,
        bins=int(spec["relative_object_yaw_histogram_bins"]),
        bounds=(0.0, yaw_periodicity_deg),
    )
    polar_occupancy = _polar_occupancy(
        radii,
        angles,
        radial_bins=int(spec["polar_radial_bins"]),
        angular_bins=int(spec["polar_angular_bins"]),
        radius_bounds=(
            None
            if workspace_catalog is None
            or workspace_catalog.continuous_distribution is None
            else (
                workspace_catalog.continuous_distribution.radius_min_m,
                workspace_catalog.continuous_distribution.radius_max_m,
            )
        ),
        angle_bounds=(
            None
            if workspace_catalog is None
            or workspace_catalog.continuous_distribution is None
            else (
                workspace_catalog.continuous_distribution.angle_min_deg,
                workspace_catalog.continuous_distribution.angle_max_deg,
            )
        ),
    )
    nearest_neighbor = _nearest_neighbor_stats(world_xy)
    floor_clearances = [
        float(row["gripper_floor_clearance_gate"]["minimum_clearance_m"])
        for row in episodes
        if isinstance(row.get("gripper_floor_clearance_gate"), dict)
        and row["gripper_floor_clearance_gate"].get("minimum_clearance_m") is not None
    ]
    lift_heights = [
        float(row["final_info"]["lift_height"])
        for row in episodes
        if isinstance(row.get("final_info"), dict)
        and row["final_info"].get("lift_height") is not None
    ]
    gates = {
        "episode_count": {
            "passed": len(episodes) == expected_episodes,
            "actual": len(episodes),
            "expected": expected_episodes,
        },
        "all_teacher_trajectories_successful": {
            "passed": (not spec["require_all_successful"]) or all(successes),
            "successes": sum(successes),
            "episodes": len(successes),
        },
        "unique_episode_seeds": {
            "passed": (not spec["require_unique_seeds"])
            or (len(seeds) == len(episodes) == len(set(seeds))),
            "unique": len(set(seeds)),
            "episodes": len(episodes),
        },
        "camera1_visibility": {
            "passed": (
                invisible / max(1, len(episodes))
                <= float(spec["max_camera1_invisible_fraction"])
            ),
            "invisible": invisible,
            "fraction": invisible / max(1, len(episodes)),
            "maximum_fraction": float(spec["max_camera1_invisible_fraction"]),
        },
        "initial_target_visibility": {
            "passed": (
                all_policy_cameras_invisible / max(1, len(episodes))
                <= float(spec["max_all_policy_cameras_invisible_fraction"])
            ),
            "invisible_in_all_policy_cameras": all_policy_cameras_invisible,
            "fraction": all_policy_cameras_invisible / max(1, len(episodes)),
            "maximum_fraction": float(
                spec["max_all_policy_cameras_invisible_fraction"]
            ),
        },
    }
    if workspace_catalog is not None:
        gates.update(
            {
                "workspace_cell_coverage": {
                    "passed": cell_coverage_ratio
                    >= float(spec["min_workspace_cell_coverage_ratio"]),
                    "covered": len(set(source_cells)),
                    "declared": declared_cells,
                    "ratio": cell_coverage_ratio,
                    "minimum_ratio": float(
                        spec["min_workspace_cell_coverage_ratio"]
                    ),
                },
                "workspace_cell_quota_match": {
                    "passed": source_cell_total_variation is not None
                    and source_cell_total_variation
                    <= float(spec["max_workspace_cell_total_variation"]),
                    "total_variation": source_cell_total_variation,
                    "maximum": float(
                        spec["max_workspace_cell_total_variation"]
                    ),
                },
                "radial_distribution_match": {
                    "passed": radial_total_variation
                    <= float(spec["max_radial_total_variation"]),
                    "total_variation": radial_total_variation,
                    "maximum": float(spec["max_radial_total_variation"]),
                },
                "distance_decay_nonincreasing": {
                    "passed": (
                        not spec["require_distance_decay_nonincreasing"]
                        or _counts_nonincreasing(radial_area_density)
                    ),
                    "required": bool(
                        spec["require_distance_decay_nonincreasing"]
                    ),
                },
                "unique_workspace_positions": {
                    "passed": len(unique_xy) == len(workspace_rows),
                    "unique": len(unique_xy),
                    "episodes": len(workspace_rows),
                },
            }
        )
        if float(spec["min_radius_span_m"]) > 0.0:
            gates["workspace_radius_span"] = {
                "passed": radius_span_m >= float(spec["min_radius_span_m"]),
                "span_m": radius_span_m,
                "minimum_m": float(spec["min_radius_span_m"]),
            }
        if float(spec["min_angle_span_deg"]) > 0.0:
            gates["workspace_angle_span"] = {
                "passed": angle_span_deg >= float(spec["min_angle_span_deg"]),
                "span_deg": angle_span_deg,
                "minimum_deg": float(spec["min_angle_span_deg"]),
            }
        if int(spec["polar_radial_bins"]) > 0:
            gates["polar_2d_cell_coverage"] = {
                "passed": polar_occupancy["coverage_ratio"]
                >= float(spec["min_polar_cell_coverage_ratio"]),
                "occupied": polar_occupancy["occupied_cells"],
                "declared": polar_occupancy["total_cells"],
                "ratio": polar_occupancy["coverage_ratio"],
                "minimum_ratio": float(
                    spec["min_polar_cell_coverage_ratio"]
                ),
            }
            gates["polar_2d_cell_balance"] = {
                "passed": polar_occupancy["count_cv"]
                <= float(spec["max_polar_cell_count_cv"]),
                "count_cv": polar_occupancy["count_cv"],
                "maximum_cv": float(spec["max_polar_cell_count_cv"]),
            }
        if float(spec["min_nearest_neighbor_median_m"]) > 0.0:
            gates["workspace_nearest_neighbor_spacing"] = {
                "passed": (
                    nearest_neighbor["median_m"] is not None
                    and float(nearest_neighbor["median_m"])
                    >= float(spec["min_nearest_neighbor_median_m"])
                ),
                "median_m": nearest_neighbor["median_m"],
                "minimum_m": float(
                    spec["min_nearest_neighbor_median_m"]
                ),
            }
        if float(spec["min_nearest_neighbor_min_m"]) > 0.0:
            gates["workspace_minimum_pair_spacing"] = {
                "passed": (
                    nearest_neighbor["min_m"] is not None
                    and float(nearest_neighbor["min_m"])
                    >= float(spec["min_nearest_neighbor_min_m"])
                ),
                "actual_m": nearest_neighbor["min_m"],
                "minimum_m": float(spec["min_nearest_neighbor_min_m"]),
            }
        if float(spec["min_object_yaw_span_deg"]) > 0.0:
            gates["object_yaw_span"] = {
                "passed": object_yaw_occupancy["span_deg"]
                >= float(spec["min_object_yaw_span_deg"]),
                "span_deg": object_yaw_occupancy["span_deg"],
                "minimum_deg": float(spec["min_object_yaw_span_deg"]),
            }
        if int(spec["object_yaw_histogram_bins"]) > 0:
            yaw_coverage = object_yaw_occupancy["coverage_ratio"]
            yaw_cv = object_yaw_occupancy["count_cv"]
            gates["object_yaw_bin_coverage"] = {
                "passed": yaw_coverage is not None
                and yaw_coverage
                >= float(spec["min_object_yaw_bin_coverage_ratio"]),
                "occupied": object_yaw_occupancy["occupied_bins"],
                "declared": object_yaw_occupancy["bins"],
                "ratio": yaw_coverage,
                "minimum_ratio": float(
                    spec["min_object_yaw_bin_coverage_ratio"]
                ),
            }
            gates["object_yaw_bin_balance"] = {
                "passed": yaw_cv is not None
                and yaw_cv <= float(spec["max_object_yaw_bin_count_cv"]),
                "count_cv": yaw_cv,
                "maximum_cv": float(spec["max_object_yaw_bin_count_cv"]),
            }
        if int(spec["relative_object_yaw_histogram_bins"]) > 0:
            relative_yaw_coverage = relative_object_yaw_occupancy[
                "coverage_ratio"
            ]
            relative_yaw_cv = relative_object_yaw_occupancy["count_cv"]
            gates["robot_relative_object_yaw_bin_coverage"] = {
                "passed": relative_yaw_coverage is not None
                and relative_yaw_coverage
                >= float(spec["min_relative_object_yaw_bin_coverage_ratio"]),
                "occupied": relative_object_yaw_occupancy["occupied_bins"],
                "declared": relative_object_yaw_occupancy["bins"],
                "ratio": relative_yaw_coverage,
                "minimum_ratio": float(
                    spec["min_relative_object_yaw_bin_coverage_ratio"]
                ),
            }
            gates["robot_relative_object_yaw_bin_balance"] = {
                "passed": relative_yaw_cv is not None
                and relative_yaw_cv
                <= float(spec["max_relative_object_yaw_bin_count_cv"]),
                "count_cv": relative_yaw_cv,
                "maximum_cv": float(
                    spec["max_relative_object_yaw_bin_count_cv"]
                ),
            }
    failed = [name for name, value in gates.items() if not value["passed"]]
    report = {
        "format": "so101_dataset_distribution_report_v1",
        "dataset_root": str(dataset_root),
        "recipe": str(recipe_path),
        "recipe_name": recipe["name"],
        "split": split_name,
        "summary": {
            "episodes": len(episodes),
            "frames": sum(frame_counts),
            "success_rate": sum(successes) / max(1, len(successes)),
            "unique_seeds": len(set(seeds)),
            "unique_prompts": len(prompts),
            "camera1_invisible_episodes": invisible,
            "all_policy_cameras_invisible_episodes": all_policy_cameras_invisible,
            "workspace_positions": len(workspace_rows),
            "unique_workspace_positions": len(unique_xy),
            "workspace_cells_covered": len(set(source_cells)),
            "workspace_cells_declared": declared_cells,
            "workspace_cell_coverage_ratio": cell_coverage_ratio,
            "workspace_cell_total_variation": source_cell_total_variation,
            "radial_total_variation": radial_total_variation,
            "radius_span_m": radius_span_m,
            "angle_span_deg": angle_span_deg,
            "polar_cell_coverage_ratio": polar_occupancy["coverage_ratio"],
            "polar_cell_count_cv": polar_occupancy["count_cv"],
            "nearest_neighbor_median_m": nearest_neighbor["median_m"],
            "nearest_neighbor_min_m": nearest_neighbor["min_m"],
            "object_yaw_span_deg": object_yaw_occupancy["span_deg"],
            "object_yaw_bin_coverage_ratio": object_yaw_occupancy[
                "coverage_ratio"
            ],
            "object_yaw_bin_count_cv": object_yaw_occupancy["count_cv"],
            "relative_object_yaw_bin_coverage_ratio": (
                relative_object_yaw_occupancy["coverage_ratio"]
            ),
            "relative_object_yaw_bin_count_cv": (
                relative_object_yaw_occupancy["count_cv"]
            ),
        },
        "episode_length": _numeric_stats(frame_counts),
        "lift_height_m": _numeric_stats(lift_heights),
        "gripper_floor_clearance_m": _numeric_stats(floor_clearances),
        "workspace": {
            "catalog": workspace_catalog_path,
            "world_xy_bounds_m": _xy_bounds(world_xy),
            "radius_m": _numeric_stats(radii),
            "angle_deg": _numeric_stats(angles),
            "radial_counts": actual_radial_counts,
            "expected_radial_counts": expected_radial_counts,
            "radial_histogram_bin_width_m": radial_bin_width_m,
            "angle_histogram_15deg": _angle_histogram(angles, width=15),
            "source_cell_counts": dict(sorted(Counter(source_cells).items())),
            "expected_source_cell_counts": expected_source_cell_counts,
            "polar_occupancy": polar_occupancy,
            "nearest_neighbor_m": nearest_neighbor,
            "radial_area_density": radial_area_density,
            "object_yaw": object_yaw_occupancy,
            "robot_relative_object_yaw": relative_object_yaw_occupancy,
        },
        "camera1_grid": {
            "grid_size": int(sidecar.get("grid_size", 4)) if sidecar else 4,
            "bin_counts": camera_bin_counts,
            "occupied_bins": len([value for value in camera_bin_counts.values() if value]),
            "invisible_episodes": invisible,
        },
        "prompts": dict(sorted(prompts.items())),
        "gate": {
            "status": "passed" if not failed else "failed",
            "failed": failed,
            "checks": gates,
        },
    }
    output_dir = dataset_root / spec["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "distribution.json"
    md_path = output_dir / "distribution.md"
    html_path = output_dir / "distribution.html"
    report["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }
    markdown_text = _markdown_report(report)
    markdown_sha256 = hashlib.sha256(markdown_text.encode("utf-8")).hexdigest()
    report["artifacts"]["markdown_sha256"] = markdown_sha256
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(
        _html_report(report, world_xy, markdown_text),
        encoding="utf-8",
    )
    return report


def _episode_camera1_grid_bin(row: dict[str, Any]) -> int | None:
    value = row.get("camera1_grid_bin")
    if value is None:
        workspace_spawn = row.get("workspace_spawn")
        if isinstance(workspace_spawn, dict):
            value = workspace_spawn.get("camera1_grid_bin")
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def _episode_has_visible_policy_camera(row: dict[str, Any]) -> bool:
    visibility = row.get("start_policy_camera_visibility")
    if not isinstance(visibility, dict):
        return False
    return any(
        isinstance(visibility.get(camera_key), dict)
        and bool(visibility[camera_key].get("visible"))
        for camera_key in ("camera1", "camera2")
    )


def require_distribution_report(
    dataset_root: Path,
    *,
    output_dir: str = "meta/distribution",
) -> dict[str, Any]:
    root = dataset_root / output_dir
    paths = {
        "json": root / "distribution.json",
        "markdown": root / "distribution.md",
        "html": root / "distribution.html",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "dataset distribution report is incomplete: " + ", ".join(missing)
        )
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    expected_markdown_sha256 = report.get("artifacts", {}).get("markdown_sha256")
    actual_markdown_sha256 = hashlib.sha256(
        paths["markdown"].read_bytes()
    ).hexdigest()
    if expected_markdown_sha256 != actual_markdown_sha256:
        raise ValueError(
            "dataset distribution Markdown does not match its JSON/HTML source "
            f"contract: expected={expected_markdown_sha256}, "
            f"actual={actual_markdown_sha256}"
        )
    html_text = paths["html"].read_text(encoding="utf-8")
    if f'data-markdown-sha256="{actual_markdown_sha256}"' not in html_text:
        raise ValueError(
            "dataset distribution HTML was not generated from the current "
            "Markdown report"
        )
    if report.get("gate", {}).get("status") != "passed":
        raise ValueError(
            "dataset distribution report gate failed: "
            + ", ".join(report.get("gate", {}).get("failed", []))
        )
    return report


def _expected_episodes(split: dict[str, Any]) -> int:
    if split.get("expected_episodes") is not None:
        return int(split["expected_episodes"])
    return sum(int(row["episodes"]) for row in split.get("bins", []))


def _load_camera_sidecar(dataset_root: Path) -> dict[str, Any] | None:
    paths = sorted(
        (dataset_root / "meta" / "camera_grid_bins").glob(
            "observation_images_camera1_*_frame0.json"
        )
    )
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def _flatten_grid_counts(rows: list[list[int]]) -> dict[int, int]:
    result: dict[int, int] = {}
    width = len(rows[0]) if rows else 0
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            result[y * width + x] = int(value)
    return result


def _expected_workspace_radial_counts(
    recipe: dict[str, Any],
    split: dict[str, Any],
    catalog: Any,
) -> dict[str, int]:
    if catalog is None:
        return {}
    rows = []
    for shard in split.get("bins", []):
        start = int(shard["lookup_start_index"])
        candidate_count = int(
            shard.get("workspace_candidate_count") or shard["episodes"]
        )
        shard_rows = catalog.candidates[start : start + candidate_count]
        if catalog.continuous_distribution is not None:
            rows.extend(row for row in shard_rows if row.stage == "primary")
        else:
            rows.extend(shard_rows[: int(shard["episodes"])])
    if catalog.continuous_distribution is not None:
        return _continuous_workspace_radial_counts(
            [row.model_dump(mode="json") for row in rows],
            catalog,
        )
    return _radial_counts(
        [float(row.radius_from_base_m) for row in rows],
        bin_width_m=float(
            recipe["distribution_report"]["radial_histogram_bin_width_m"]
        ),
    )


def _expected_workspace_cell_counts(
    split: dict[str, Any],
    catalog: Any,
) -> dict[str, int]:
    if catalog is None:
        return {}
    counts: Counter[str] = Counter()
    for shard in split.get("bins", []):
        start = int(shard["lookup_start_index"])
        candidate_count = int(
            shard.get("workspace_candidate_count") or shard["episodes"]
        )
        rows = catalog.candidates[start : start + candidate_count]
        primary = [row for row in rows if row.stage == "primary"]
        selected = primary if catalog.enforce_cell_local_quota else rows[: int(shard["episodes"])]
        counts.update(row.source_cell_id for row in selected)
    return dict(sorted(counts.items()))


def _continuous_workspace_radial_counts(
    rows: list[Any],
    catalog: Any,
) -> dict[str, int]:
    quota_by_id = {row.cell_id: row for row in catalog.cell_quotas}
    bounds_by_index: dict[int, tuple[float, float]] = {}
    counts: Counter[int] = Counter()
    for row in rows:
        source_cell_id = (
            str(row.source_cell_id)
            if hasattr(row, "source_cell_id")
            else str(row["source_cell_id"])
        )
        quota = quota_by_id[source_cell_id]
        counts[int(quota.radial_index)] += 1
        bounds_by_index[int(quota.radial_index)] = (
            float(quota.radius_bounds_m[0]),
            float(quota.radius_bounds_m[1]),
        )
    return {
        f"{bounds_by_index[index][0]:.4f}..{bounds_by_index[index][1]:.4f}": int(count)
        for index, count in sorted(counts.items())
    }


def _continuous_radial_area_density(
    counts: dict[str, int],
    catalog: Any,
) -> dict[str, float]:
    area_by_label: Counter[str] = Counter()
    for quota in catalog.cell_quotas:
        label = (
            f"{float(quota.radius_bounds_m[0]):.4f}.."
            f"{float(quota.radius_bounds_m[1]):.4f}"
        )
        area_by_label[label] += float(quota.area_m2)
    return {
        label: float(count) / float(area_by_label[label])
        for label, count in counts.items()
    }


def _radial_counts(
    radii: list[float],
    *,
    bin_width_m: float = 0.0001,
) -> dict[str, int]:
    if bin_width_m <= 0.0:
        raise ValueError("radial histogram bin width must be positive")
    return dict(
        sorted(
            Counter(
                f"{round(round(value / bin_width_m) * bin_width_m, 10):.4f}"
                for value in radii
            ).items(),
            key=lambda item: float(item[0]),
        )
    )


def _distribution_total_variation(
    actual: dict[str, int],
    expected: dict[str, int],
) -> float | None:
    if not actual or not expected:
        return None
    keys = sorted(set(actual) | set(expected), key=_histogram_key_value)
    actual_total = sum(actual.values())
    expected_total = sum(expected.values())
    return 0.5 * sum(
        abs(actual.get(key, 0) / actual_total - expected.get(key, 0) / expected_total)
        for key in keys
    )


def _categorical_total_variation(
    actual: dict[str, int],
    expected: dict[str, int],
) -> float | None:
    if not actual or not expected:
        return None
    keys = set(actual) | set(expected)
    actual_total = sum(actual.values())
    expected_total = sum(expected.values())
    return 0.5 * sum(
        abs(
            actual.get(key, 0) / actual_total
            - expected.get(key, 0) / expected_total
        )
        for key in keys
    )


def _histogram_key_value(value: str) -> float:
    return float(str(value).split("..", maxsplit=1)[0])


def _counts_nonincreasing(counts: dict[str, float | int]) -> bool:
    values = [
        value
        for _, value in sorted(
            counts.items(), key=lambda item: _histogram_key_value(item[0])
        )
    ]
    return all(left >= right for left, right in zip(values, values[1:]))


def _numeric_stats(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "max": max(numeric),
    }


def _numeric_span(values: list[float | int]) -> float:
    if not values:
        return 0.0
    numeric = [float(value) for value in values]
    return float(max(numeric) - min(numeric))


def _polar_occupancy(
    radii: list[float],
    angles: list[float],
    *,
    radial_bins: int,
    angular_bins: int,
    radius_bounds: tuple[float, float] | None = None,
    angle_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    if radial_bins <= 0 or angular_bins <= 0 or not radii or not angles:
        return {
            "radial_bins": int(radial_bins),
            "angular_bins": int(angular_bins),
            "total_cells": 0,
            "occupied_cells": 0,
            "coverage_ratio": None,
            "count_cv": None,
            "radius_bounds_m": [None, None],
            "angle_bounds_deg": [None, None],
            "counts": {},
        }
    radius_min, radius_max = radius_bounds or (min(radii), max(radii))
    angle_min, angle_max = angle_bounds or (min(angles), max(angles))
    counts = {
        f"r{r:02d}_a{a:02d}": 0
        for r in range(radial_bins)
        for a in range(angular_bins)
    }
    for radius, angle in zip(radii, angles, strict=True):
        radius_index = _bounded_bin_index(
            radius, lower=radius_min, upper=radius_max, bins=radial_bins
        )
        angle_index = _bounded_bin_index(
            angle, lower=angle_min, upper=angle_max, bins=angular_bins
        )
        counts[f"r{radius_index:02d}_a{angle_index:02d}"] += 1
    values = list(counts.values())
    occupied = sum(value > 0 for value in values)
    mean = statistics.fmean(values)
    count_cv = (
        statistics.pstdev(values) / mean
        if mean > 0.0
        else None
    )
    total = radial_bins * angular_bins
    return {
        "radial_bins": radial_bins,
        "angular_bins": angular_bins,
        "total_cells": total,
        "occupied_cells": occupied,
        "coverage_ratio": occupied / total,
        "count_cv": count_cv,
        "radius_bounds_m": [radius_min, radius_max],
        "angle_bounds_deg": [angle_min, angle_max],
        "counts": counts,
    }


def _bounded_bin_index(
    value: float,
    *,
    lower: float,
    upper: float,
    bins: int,
) -> int:
    if bins <= 1 or math.isclose(lower, upper, abs_tol=1e-12):
        return 0
    unit = (float(value) - float(lower)) / (float(upper) - float(lower))
    return min(bins - 1, max(0, int(math.floor(unit * bins))))


def _periodic_degrees(value: float, *, period: float) -> float:
    """Map an angle to [0, period), preserving cube-face symmetry."""
    if period <= 0.0:
        raise ValueError("period must be positive")
    return float(value) % float(period)


def _object_yaw_occupancy(
    yaws: list[float],
    *,
    bins: int,
    bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    if not yaws:
        return {
            "bins": int(bins),
            "occupied_bins": 0,
            "coverage_ratio": None,
            "count_cv": None,
            "bounds_deg": [None, None],
            "span_deg": 0.0,
            "counts": {},
        }
    lower, upper = bounds or (min(yaws), max(yaws))
    if upper <= lower:
        upper = lower + 1.0
    if bins <= 0:
        return {
            "bins": int(bins),
            "occupied_bins": 0,
            "coverage_ratio": None,
            "count_cv": None,
            "bounds_deg": [float(lower), float(upper)],
            "span_deg": _numeric_span(yaws),
            "counts": {},
        }
    width = (float(upper) - float(lower)) / bins
    labels = [
        f"{lower + index * width:.3f}..{lower + (index + 1) * width:.3f}"
        for index in range(bins)
    ]
    counts = {label: 0 for label in labels}
    for yaw in yaws:
        index = _bounded_bin_index(
            float(yaw),
            lower=float(lower),
            upper=float(upper),
            bins=bins,
        )
        counts[labels[index]] += 1
    values = list(counts.values())
    occupied = sum(value > 0 for value in values)
    mean = statistics.fmean(values)
    count_cv = statistics.pstdev(values) / mean if mean > 0.0 else None
    return {
        "bins": int(bins),
        "occupied_bins": occupied,
        "coverage_ratio": occupied / bins,
        "count_cv": count_cv,
        "bounds_deg": [float(lower), float(upper)],
        "span_deg": _numeric_span(yaws),
        "counts": counts,
    }


def _nearest_neighbor_stats(points: list[list[float]]) -> dict[str, Any]:
    if len(points) < 2:
        return {
            "count": 0,
            "min_m": None,
            "mean_m": None,
            "median_m": None,
            "max_m": None,
        }
    nearest: list[float] = []
    for index, point in enumerate(points):
        px, py = float(point[0]), float(point[1])
        nearest.append(
            min(
                math.hypot(px - float(other[0]), py - float(other[1]))
                for other_index, other in enumerate(points)
                if other_index != index
            )
        )
    return {
        "count": len(nearest),
        "min_m": min(nearest),
        "mean_m": statistics.fmean(nearest),
        "median_m": statistics.median(nearest),
        "max_m": max(nearest),
    }


def _xy_bounds(points: list[list[float]]) -> dict[str, float | None]:
    if not points:
        return {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    return {
        "x_min": min(point[0] for point in points),
        "x_max": max(point[0] for point in points),
        "y_min": min(point[1] for point in points),
        "y_max": max(point[1] for point in points),
    }


def _angle_histogram(angles: list[float], *, width: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for angle in angles:
        lower = math.floor(angle / width) * width
        counts[f"{lower:+d}..{lower + width:+d}"] += 1
    return dict(sorted(counts.items()))


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    gate = report["gate"]
    workspace = report["workspace"]
    rows = [
        f"# SO101 Dataset Distribution: {report['recipe_name']} / {report['split']}",
        "",
        f"**Gate:** `{gate['status']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Episodes | {summary['episodes']} |",
        f"| Frames | {summary['frames']} |",
        f"| Teacher success rate | {summary['success_rate']:.3f} |",
        f"| Unique seeds | {summary['unique_seeds']} |",
        f"| Unique XY positions | {summary['unique_workspace_positions']} |",
        f"| Workspace cell coverage | {_format_ratio(summary['workspace_cell_coverage_ratio'])} |",
        f"| Workspace cell quota TV | {_format_number(summary['workspace_cell_total_variation'])} |",
        f"| Camera1 invisible | {summary['camera1_invisible_episodes']} |",
        f"| Radial distribution TV | {_format_number(summary['radial_total_variation'])} |",
        f"| Radius span | {summary['radius_span_m']:.4f} m |",
        f"| Angle span | {summary['angle_span_deg']:.2f} deg |",
        f"| Polar cell coverage | {_format_ratio(summary['polar_cell_coverage_ratio'])} |",
        f"| Polar cell count CV | {_format_number(summary['polar_cell_count_cv'])} |",
        f"| Object yaw span | {summary['object_yaw_span_deg']:.2f} deg |",
        f"| Object yaw bin coverage | {_format_ratio(summary['object_yaw_bin_coverage_ratio'])} |",
        f"| Object yaw bin count CV | {_format_number(summary['object_yaw_bin_count_cv'])} |",
        f"| Robot-relative yaw bin coverage | {_format_ratio(summary['relative_object_yaw_bin_coverage_ratio'])} |",
        f"| Robot-relative yaw bin count CV | {_format_number(summary['relative_object_yaw_bin_count_cv'])} |",
        f"| Median nearest-neighbor distance | {_format_number(summary['nearest_neighbor_median_m'])} m |",
        f"| Minimum nearest-neighbor distance | {_format_number(summary['nearest_neighbor_min_m'])} m |",
        "",
        "## Radial Distribution",
        "",
        "| Radius (m) | Actual | Expected |",
        "|---:|---:|---:|",
    ]
    keys = sorted(
        set(workspace["radial_counts"]) | set(workspace["expected_radial_counts"]),
        key=_histogram_key_value,
    )
    rows.extend(
        f"| {key} | {workspace['radial_counts'].get(key, 0)} | "
        f"{workspace['expected_radial_counts'].get(key, 0)} |"
        for key in keys
    )
    rows.extend(
        [
            "",
            "## Object Yaw Distribution",
            "",
            "| Yaw range (deg) | Episodes |",
            "|---:|---:|",
            *[
                f"| {key} | {value} |"
                for key, value in workspace["object_yaw"]["counts"].items()
            ],
            "",
            "## Robot-relative Object Yaw",
            "",
            "This is `(cube yaw - spawn angle) mod cube symmetry period`. A collapsed histogram means the same cube face always points toward the robot.",
            "",
            "| Relative yaw range (deg) | Episodes |",
            "|---:|---:|",
            *[
                f"| {key} | {value} |"
                for key, value in workspace["robot_relative_object_yaw"]["counts"].items()
            ],
        ]
    )
    rows.extend(
        [
            "",
            "## Camera1 4x4 Grid",
            "",
            "| Bin | Episodes |",
            "|---:|---:|",
            *[
                f"| {key} | {value} |"
                for key, value in report["camera1_grid"]["bin_counts"].items()
            ],
            "",
            "## Gate Checks",
            "",
            "| Check | Result |",
            "|---|---|",
            *[
                f"| `{name}` | {'PASS' if check['passed'] else 'FAIL'} |"
                for name, check in gate["checks"].items()
            ],
            "",
            "## Prompts",
            "",
            *[
                f"- `{prompt}`: {count}"
                for prompt, count in report["prompts"].items()
            ],
            "",
        ]
    )
    return "\n".join(rows)


def _html_report(
    report: dict[str, Any],
    points: list[list[float]],
    markdown_source: str,
) -> str:
    summary = report["summary"]
    gate = report["gate"]
    markdown_sha256 = hashlib.sha256(markdown_source.encode("utf-8")).hexdigest()
    markdown_html = _markdown_to_html(markdown_source)
    status_class = "pass" if gate["status"] == "passed" else "fail"
    cards = [
        ("Episodes", str(summary["episodes"])),
        ("Success", f"{summary['success_rate'] * 100:.1f}%"),
        ("Unique XY", str(summary["unique_workspace_positions"])),
        ("Cell coverage", _format_ratio(summary["workspace_cell_coverage_ratio"])),
        ("Camera1 hidden", str(summary["camera1_invisible_episodes"])),
        ("Radial TV", _format_number(summary["radial_total_variation"])),
        ("Radius span", f"{summary['radius_span_m'] * 100:.1f} cm"),
        ("Angle span", f"{summary['angle_span_deg']:.1f}°"),
        ("Polar coverage", _format_ratio(summary["polar_cell_coverage_ratio"])),
        ("Yaw coverage", _format_ratio(summary["object_yaw_bin_coverage_ratio"])),
        ("Yaw span", f"{summary['object_yaw_span_deg']:.1f}°"),
        (
            "Relative yaw coverage",
            _format_ratio(summary["relative_object_yaw_bin_coverage_ratio"]),
        ),
        ("NN median", f"{(summary['nearest_neighbor_median_m'] or 0.0) * 1000:.1f} mm"),
    ]
    gate_rows = "".join(
        "<tr><td>{}</td><td class=\"{}\">{}</td></tr>".format(
            html.escape(name),
            "pass" if check["passed"] else "fail",
            "PASS" if check["passed"] else "FAIL",
        )
        for name, check in gate["checks"].items()
    )
    prompt_rows = "".join(
        f"<tr><td>{html.escape(prompt)}</td><td>{count}</td></tr>"
        for prompt, count in report["prompts"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO101 Dataset Distribution</title>
<style>
:root{{--ink:#15213a;--muted:#64748b;--line:#dbe4f0;--panel:#fff;--bg:#f3f6fa;
--blue:#2563eb;--cyan:#0891b2;--green:#15803d;--red:#c2410c;--amber:#d97706}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 20px 64px}} h1{{font-size:30px;margin:0}}
.sub{{color:var(--muted);margin:6px 0 22px}} .status{{display:inline-flex;padding:5px 10px;
border-radius:6px;font-weight:800;margin-left:10px}} .status.pass{{background:#dcfce7;color:#166534}}
.status.fail{{background:#fee2e2;color:#991b1b}} .cards{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}}
.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;
box-shadow:0 5px 18px rgba(30,41,59,.05)}} .card{{padding:15px}}
.card span{{display:block;color:var(--muted);font-size:12px;font-weight:700;text-transform:uppercase}}
.card strong{{font-size:25px}} .grid{{display:grid;grid-template-columns:1.2fr 1fr;gap:18px}}
.panel{{padding:18px;margin-bottom:18px}} h2{{font-size:17px;margin:0 0 14px}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:8px;
border-bottom:1px solid #edf2f7}} td:last-child{{text-align:right}} .pass{{color:var(--green);
font-weight:800}} .fail{{color:var(--red);font-weight:800}} svg{{width:100%;height:auto}}
.legend{{color:var(--muted);font-size:12px;margin-top:8px}}
.source summary{{cursor:pointer;font-weight:800}} .source .markdown{{margin-top:16px}}
.source .markdown h1{{display:none}} .source .markdown h2{{font-size:16px;margin-top:20px}}
.source .markdown code{{background:#eef2ff;border-radius:4px;padding:2px 5px}}
.source .markdown ul{{padding-left:20px}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}} main{{padding:20px 12px}}}}
</style>
</head>
<body data-markdown-sha256="{markdown_sha256}"><main>
<h1>SO101 Dataset Distribution <span class="status {status_class}">{gate['status'].upper()}</span></h1>
<p class="sub">{html.escape(report['recipe_name'])} · {html.escape(report['split'])}</p>
<section class="cards">{''.join(f'<div class="card"><span>{html.escape(k)}</span><strong>{html.escape(v)}</strong></div>' for k,v in cards)}</section>
<section class="grid">
<div class="panel"><h2>Object spawn positions</h2>{_scatter_svg(points)}
<div class="legend">World XY. Each dot is one exported episode.</div></div>
<div class="panel"><h2>Polar 2D occupancy</h2>{_polar_grid_svg(report['workspace']['polar_occupancy'])}
<div class="legend">Rows: radius · columns: base-relative angle.</div></div>
</section>
<section class="grid">
<div class="panel"><h2>Radial distribution</h2>{_radial_bar_svg(report['workspace']['radial_counts'], report['workspace']['expected_radial_counts'])}
<div class="legend">Blue: actual · outline: configured target</div></div>
<div class="panel"><h2>Camera1 4×4 occupancy</h2>{_grid_svg(report['camera1_grid']['bin_counts'])}</div>
</section>
<section class="panel"><h2>Independent cube yaw</h2>{_yaw_bar_svg(report['workspace']['object_yaw'])}
<div class="legend">Cube orientation modulo its declared symmetry interval.</div></section>
<section class="panel"><h2>Robot-relative cube yaw</h2>{_yaw_bar_svg(report['workspace']['robot_relative_object_yaw'])}
<div class="legend">(Cube yaw - spawn angle) modulo cube symmetry. This must stay spread out; a single bin means the same face always points at the robot.</div></section>
<section class="grid">
<div class="panel"><h2>Completion gates</h2><table>{gate_rows}</table></div>
<div class="panel"><h2>Prompts</h2><table>{prompt_rows}</table></div>
</section>
<details class="panel source"><summary>Canonical Markdown report</summary>
<div class="markdown">{markdown_html}</div></details>
</main></body></html>
"""


def _markdown_to_html(markdown_text: str) -> str:
    """Render the small Markdown subset emitted by _markdown_report."""

    lines = markdown_text.splitlines()
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            rendered.append(f"<h1>{_markdown_inline(line[2:])}</h1>")
            index += 1
            continue
        if line.startswith("## "):
            rendered.append(f"<h2>{_markdown_inline(line[3:])}</h2>")
            index += 1
            continue
        if line.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            header = _markdown_table_cells(table_lines[0])
            body_lines = table_lines[2:] if len(table_lines) > 1 else []
            rendered.append("<table><thead><tr>")
            rendered.extend(f"<th>{_markdown_inline(cell)}</th>" for cell in header)
            rendered.append("</tr></thead><tbody>")
            for body_line in body_lines:
                rendered.append("<tr>")
                rendered.extend(
                    f"<td>{_markdown_inline(cell)}</td>"
                    for cell in _markdown_table_cells(body_line)
                )
                rendered.append("</tr>")
            rendered.append("</tbody></table>")
            continue
        if line.startswith("- "):
            rendered.append("<ul>")
            while index < len(lines) and lines[index].startswith("- "):
                rendered.append(
                    f"<li>{_markdown_inline(lines[index][2:])}</li>"
                )
                index += 1
            rendered.append("</ul>")
            continue
        rendered.append(f"<p>{_markdown_inline(line)}</p>")
        index += 1
    return "".join(rendered)


def _markdown_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_inline(value: str) -> str:
    escaped = html.escape(value)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def _scatter_svg(points: list[list[float]]) -> str:
    if not points:
        return "<p>No workspace coordinates recorded.</p>"
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(0.005, (x_max - x_min) * 0.08)
    y_pad = max(0.005, (y_max - y_min) * 0.08)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    circles = []
    for x, y in points:
        sx = 30 + 520 * (x - x_min) / max(1e-9, x_max - x_min)
        sy = 300 - 270 * (y - y_min) / max(1e-9, y_max - y_min)
        circles.append(
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3.0" fill="#2563eb" fill-opacity=".48"/>'
        )
    return (
        '<svg viewBox="0 0 580 330" role="img">'
        '<rect x="30" y="30" width="520" height="270" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>'
        + "".join(circles)
        + "</svg>"
    )


def _radial_bar_svg(actual: dict[str, int], expected: dict[str, int]) -> str:
    keys = sorted(set(actual) | set(expected), key=_histogram_key_value)
    if not keys:
        return "<p>No radial metadata recorded.</p>"
    maximum = max([*actual.values(), *expected.values(), 1])
    bars = []
    for index, key in enumerate(keys):
        x = 65 + index * (430 / max(1, len(keys)))
        width = 54
        actual_h = 220 * actual.get(key, 0) / maximum
        expected_h = 220 * expected.get(key, 0) / maximum
        bars.extend(
            [
                f'<rect x="{x:.1f}" y="{270-expected_h:.1f}" width="{width}" height="{expected_h:.1f}" fill="none" stroke="#d97706" stroke-width="3"/>',
                f'<rect x="{x+8:.1f}" y="{270-actual_h:.1f}" width="{width-16}" height="{actual_h:.1f}" fill="#2563eb" fill-opacity=".82"/>',
                f'<text x="{x+width/2:.1f}" y="292" text-anchor="middle" font-size="12" fill="#475569">{_histogram_label_cm(key)}</text>',
            ]
        )
    return (
        '<svg viewBox="0 0 560 320" role="img">'
        '<line x1="50" y1="270" x2="530" y2="270" stroke="#cbd5e1"/>'
        + "".join(bars)
        + "</svg>"
    )


def _histogram_label_cm(key: str) -> str:
    parts = str(key).split("..", maxsplit=1)
    if len(parts) == 2:
        center = (float(parts[0]) + float(parts[1])) / 2.0
    else:
        center = float(parts[0])
    return f"{center * 100:.1f}cm"


def _yaw_bar_svg(occupancy: dict[str, Any]) -> str:
    counts = occupancy.get("counts") or {}
    if not counts:
        return "<p>Object-yaw occupancy is not configured.</p>"
    maximum = max([*counts.values(), 1])
    width = 500.0 / max(1, len(counts))
    bars = []
    for index, (label, value) in enumerate(counts.items()):
        x = 40.0 + index * width
        height = 190.0 * int(value) / maximum
        bars.extend(
            [
                f'<rect x="{x + 5:.1f}" y="{225 - height:.1f}" '
                f'width="{max(4.0, width - 10):.1f}" height="{height:.1f}" '
                'fill="#d97706" fill-opacity=".82"/>',
                f'<text x="{x + width / 2:.1f}" y="245" text-anchor="middle" '
                f'font-size="11" fill="#475569">{html.escape(label)}</text>',
                f'<text x="{x + width / 2:.1f}" y="{215 - height:.1f}" '
                f'text-anchor="middle" font-size="12" fill="#0f172a">{int(value)}</text>',
            ]
        )
    return (
        '<svg viewBox="0 0 580 270" role="img">'
        '<line x1="35" y1="225" x2="550" y2="225" stroke="#cbd5e1"/>'
        + "".join(bars)
        + "</svg>"
    )


def _grid_svg(counts: dict[int, int]) -> str:
    maximum = max([*counts.values(), 1])
    cells = []
    for index in range(16):
        x = 35 + (index % 4) * 76
        y = 25 + (index // 4) * 76
        value = int(counts.get(index, counts.get(str(index), 0)))
        opacity = 0.08 + 0.86 * value / maximum
        cells.append(
            f'<rect x="{x}" y="{y}" width="64" height="64" rx="5" fill="#0891b2" fill-opacity="{opacity:.3f}"/>'
            f'<text x="{x+32}" y="{y+29}" text-anchor="middle" font-size="12" fill="#0f172a">bin {index}</text>'
            f'<text x="{x+32}" y="{y+48}" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">{value}</text>'
        )
    return '<svg viewBox="0 0 380 350" role="img">' + "".join(cells) + "</svg>"


def _polar_grid_svg(occupancy: dict[str, Any]) -> str:
    radial_bins = int(occupancy.get("radial_bins", 0))
    angular_bins = int(occupancy.get("angular_bins", 0))
    counts = occupancy.get("counts") or {}
    if radial_bins <= 0 or angular_bins <= 0:
        return "<p>Polar occupancy is not configured.</p>"
    maximum = max([*counts.values(), 1])
    cell_width = 460.0 / angular_bins
    cell_height = 230.0 / radial_bins
    cells = []
    for radius_index in range(radial_bins):
        for angle_index in range(angular_bins):
            value = int(counts.get(f"r{radius_index:02d}_a{angle_index:02d}", 0))
            x = 70.0 + angle_index * cell_width
            y = 25.0 + (radial_bins - radius_index - 1) * cell_height
            opacity = 0.05 if value == 0 else 0.20 + 0.78 * value / maximum
            cells.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width - 2:.2f}" '
                f'height="{cell_height - 2:.2f}" rx="3" fill="#2563eb" '
                f'fill-opacity="{opacity:.3f}"/>'
                f'<text x="{x + cell_width / 2:.2f}" '
                f'y="{y + cell_height / 2 + 4:.2f}" text-anchor="middle" '
                f'font-size="11" fill="#0f172a">{value}</text>'
            )
    return (
        '<svg viewBox="0 0 570 300" role="img">'
        + "".join(cells)
        + '<text x="300" y="286" text-anchor="middle" font-size="12" '
        'fill="#64748b">base-relative angle</text>'
        '<text x="18" y="145" text-anchor="middle" font-size="12" '
        'fill="#64748b" transform="rotate(-90 18 145)">radius</text>'
        "</svg>"
    )


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
