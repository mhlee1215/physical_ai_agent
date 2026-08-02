#!/usr/bin/env python3
"""Build a seed-free, weighted SO101 workspace spawn catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.so101_workspace_spawn_catalog import (
    build_continuous_area_workspace_spawn_catalog,
    build_joint_feasible_workspace_spawn_catalog,
    build_workspace_spawn_catalog,
)


def _integer_list(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _float_pair(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--primary-target-count", type=int, required=True)
    parser.add_argument("--backup-count", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--distance-decay-rate-per-m", type=float, default=20.0)
    parser.add_argument("--angular-jitter-max-deg", type=float, default=0.75)
    parser.add_argument("--radial-jitter-max-m", type=float, default=0.0)
    parser.add_argument(
        "--candidate-sequence-offset",
        type=int,
        default=0,
        help=(
            "Deterministic sequence offset for fresh continuous candidates; "
            "use a new value to avoid reusing coordinates from an older catalog."
        ),
    )
    parser.add_argument(
        "--preserve-evidence-object-yaw",
        action="store_true",
        help=(
            "Keep each verified source cell's object yaw and apply only the "
            "configured local yaw jitter."
        ),
    )
    parser.add_argument("--continuous-area", action="store_true")
    parser.add_argument("--joint-feasible", action="store_true")
    parser.add_argument("--radius-min-m", type=float, default=0.25)
    parser.add_argument("--radius-max-m", type=float, default=0.28)
    parser.add_argument("--angle-min-deg", type=float, default=-20.0)
    parser.add_argument("--angle-max-deg", type=float, default=80.0)
    parser.add_argument("--radial-strata", type=int, default=6)
    parser.add_argument("--angular-strata", type=int, default=10)
    parser.add_argument(
        "--far-to-near-area-density-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument("--minimum-spacing-m", type=float, default=0.0015)
    parser.add_argument(
        "--candidate-pool-minimum-spacing-m",
        type=float,
        help=(
            "Optional dense backup-pool spacing. Accepted episodes still use "
            "--minimum-spacing-m."
        ),
    )
    parser.add_argument("--object-yaw-min-deg", type=float)
    parser.add_argument("--object-yaw-max-deg", type=float)
    parser.add_argument(
        "--object-yaw-strata",
        type=int,
        default=0,
        help=(
            "Positive values independently stratify object yaw instead of "
            "coupling one cube face to the base-relative spawn angle."
        ),
    )
    parser.add_argument(
        "--object-yaw-periodicity-deg",
        type=float,
        default=90.0,
        help="Rotational symmetry period; 90 degrees for a textureless cube.",
    )
    parser.add_argument(
        "--object-yaw-reference-frame",
        choices=("robot_relative", "world_absolute"),
        default="robot_relative",
        help=(
            "Stratify the cube face seen from the robot by default; use "
            "world_absolute only when world-frame yaw is the intended contract."
        ),
    )
    parser.add_argument("--radial-primary-counts", type=_integer_list)
    parser.add_argument("--yaw-primary-counts", type=_integer_list)
    parser.add_argument("--radial-backup-counts", type=_integer_list)
    parser.add_argument("--yaw-backup-counts", type=_integer_list)
    parser.add_argument("--object-yaw-center-offset-deg", type=float, default=2.5)
    parser.add_argument(
        "--evidence-radius-half-range-m",
        type=float,
        default=0.004,
        help="Maximum radial distance from a teacher-verified probe center.",
    )
    parser.add_argument(
        "--evidence-angle-half-range-deg",
        type=float,
        default=2.0,
        help="Maximum base-angle distance from a teacher-verified probe center.",
    )
    parser.add_argument(
        "--object-yaw-jitter-half-range-deg",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--max-robot-relative-yaw-count-cv",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--exclude-radius-relative-yaw",
        action="append",
        type=_float_pair,
        default=[],
        help=(
            "Exclude one fragile joint-feasible support pair as "
            "radius_m,robot_relative_yaw_deg; repeat as needed."
        ),
    )
    args = parser.parse_args()

    if args.joint_feasible:
        required = {
            "--radial-primary-counts": args.radial_primary_counts,
            "--yaw-primary-counts": args.yaw_primary_counts,
            "--radial-backup-counts": args.radial_backup_counts,
            "--yaw-backup-counts": args.yaw_backup_counts,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            parser.error("joint-feasible mode requires " + ", ".join(missing))
        catalog = build_joint_feasible_workspace_spawn_catalog(
            source_path=args.source,
            catalog_id=args.catalog_id,
            radial_primary_counts=args.radial_primary_counts,
            yaw_primary_counts=args.yaw_primary_counts,
            radial_backup_counts=args.radial_backup_counts,
            yaw_backup_counts=args.yaw_backup_counts,
            shard_count=args.shards,
            radius_min_m=args.radius_min_m,
            radius_max_m=args.radius_max_m,
            angle_min_deg=args.angle_min_deg,
            angle_max_deg=args.angle_max_deg,
            minimum_spacing_m=args.minimum_spacing_m,
            candidate_pool_minimum_spacing_m=(
                args.candidate_pool_minimum_spacing_m
            ),
            evidence_radius_half_range_m=args.evidence_radius_half_range_m,
            evidence_angle_half_range_deg=args.evidence_angle_half_range_deg,
            object_yaw_center_offset_deg=args.object_yaw_center_offset_deg,
            object_yaw_jitter_half_range_deg=(
                args.object_yaw_jitter_half_range_deg
            ),
            object_yaw_periodicity_deg=args.object_yaw_periodicity_deg,
            max_robot_relative_yaw_count_cv=(
                args.max_robot_relative_yaw_count_cv
            ),
            excluded_radius_relative_yaw_pairs=(
                args.exclude_radius_relative_yaw
            ),
            candidate_sequence_offset=args.candidate_sequence_offset,
        )
    elif args.continuous_area:
        catalog = build_continuous_area_workspace_spawn_catalog(
            source_path=args.source,
            catalog_id=args.catalog_id,
            primary_target_count=args.primary_target_count,
            backup_count=args.backup_count,
            shard_count=args.shards,
            radius_min_m=args.radius_min_m,
            radius_max_m=args.radius_max_m,
            angle_min_deg=args.angle_min_deg,
            angle_max_deg=args.angle_max_deg,
            radial_strata=args.radial_strata,
            angular_strata=args.angular_strata,
            far_to_near_area_density_ratio=(
                args.far_to_near_area_density_ratio
            ),
            minimum_spacing_m=args.minimum_spacing_m,
            object_yaw_min_deg=args.object_yaw_min_deg,
            object_yaw_max_deg=args.object_yaw_max_deg,
            object_yaw_strata=args.object_yaw_strata,
            object_yaw_periodicity_deg=args.object_yaw_periodicity_deg,
            object_yaw_reference_frame=args.object_yaw_reference_frame,
        )
    else:
        catalog = build_workspace_spawn_catalog(
            source_path=args.source,
            catalog_id=args.catalog_id,
            primary_target_count=args.primary_target_count,
            backup_count=args.backup_count,
            shard_count=args.shards,
            distance_decay_rate_per_m=args.distance_decay_rate_per_m,
            angular_jitter_max_deg=args.angular_jitter_max_deg,
            radial_jitter_max_m=args.radial_jitter_max_m,
            candidate_sequence_offset=args.candidate_sequence_offset,
            preserve_evidence_object_yaw=args.preserve_evidence_object_yaw,
            object_yaw_jitter_half_range_deg=(
                args.object_yaw_jitter_half_range_deg
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "candidate_count": catalog.candidate_count,
                "primary_target_count": catalog.primary_target_count,
                "backup_count": catalog.backup_count,
                "source_cell_count": catalog.source_cell_count,
                "sampling_strategy": catalog.sampling_strategy,
                "radial_jitter_max_m": catalog.radial_jitter_max_m,
                "candidate_sequence_offset": catalog.candidate_sequence_offset,
                "preserve_evidence_object_yaw": (
                    catalog.preserve_evidence_object_yaw
                ),
                "shards": [shard.model_dump(mode="json") for shard in catalog.shards],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
