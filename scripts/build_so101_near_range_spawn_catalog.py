#!/usr/bin/env python3
"""Build a seed-free, visibility-gated near-range SO101 spawn catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from export_so101_teacher_rollouts_lerobot import (
    _make_near_range_fixed_jaw_teacher_targets,
    _policy_camera_visibility,
    _set_target_object_xy,
    _set_target_object_yaw,
)
from physical_ai_agent.sim.so101_camera_rig_render_config import (
    load_so101_camera_rig_render_config,
)
from physical_ai_agent.so101_workspace_spawn_catalog import (
    WorkspaceSpawnCandidate,
    WorkspaceSpawnCatalog,
    WorkspaceSpawnShard,
)
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _make_policy_renderers,
    _set_qpos,
    make_high_contrast_picklift_env,
)


DEFAULT_HOME_QPOS = (
    0.0,
    -math.pi / 2.0,
    math.pi / 2.0,
    0.66,
    math.pi / 2.0,
    -0.17453,
)
DEFAULT_BASE_ORIGIN_XY = (0.015478381254367828, 0.00000525602343)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--primary-count", type=int, required=True)
    parser.add_argument("--backup-count", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--radius-min-m", type=float, default=0.10)
    parser.add_argument("--radius-max-m", type=float, default=0.18)
    parser.add_argument("--radial-strata", type=int, default=5)
    parser.add_argument("--angle-min-deg", type=float, default=-75.0)
    parser.add_argument("--angle-max-deg", type=float, default=75.0)
    parser.add_argument("--angle-strata", type=int, default=10)
    parser.add_argument("--yaw-strata", type=int, default=8)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--max-attempt-multiplier", type=int, default=30)
    parser.add_argument("--minimum-spacing-m", type=float, default=0.0015)
    parser.add_argument("--minimum-area-pixels", type=int, default=20)
    parser.add_argument("--min-floor-clearance-m", type=float, default=0.01)
    parser.add_argument(
        "--skip-solver-prefilter",
        action="store_true",
        help=(
            "Build the placement catalog from range/distribution/visibility gates only. "
            "The dataset exporter still applies the authoritative IK, alignment, floor, "
            "grasp, lift, and hold gates to every exported episode."
        ),
    )
    parser.add_argument("--object-half-size-m", type=float, default=0.015)
    parser.add_argument("--object-color", default="green")
    parser.add_argument(
        "--camera-rig-config",
        type=Path,
        default=Path(
            "configs/so101/camera_rigs/"
            "official_32x32_uvc_photoreal_v10_fov_calibrated_direct_square.json"
        ),
    )
    parser.add_argument(
        "--home-qpos",
        default=",".join(str(value) for value in DEFAULT_HOME_QPOS),
    )
    parser.add_argument(
        "--base-origin-xy",
        default=",".join(str(value) for value in DEFAULT_BASE_ORIGIN_XY),
    )
    args = parser.parse_args()

    report = build_near_range_spawn_catalog(
        catalog_output=args.catalog_output,
        evidence_output=args.evidence_output,
        catalog_id=args.catalog_id,
        primary_count=args.primary_count,
        backup_count=args.backup_count,
        shard_count=args.shards,
        radius_min_m=args.radius_min_m,
        radius_max_m=args.radius_max_m,
        radial_strata=args.radial_strata,
        angle_min_deg=args.angle_min_deg,
        angle_max_deg=args.angle_max_deg,
        angle_strata=args.angle_strata,
        yaw_strata=args.yaw_strata,
        sequence_offset=args.sequence_offset,
        max_attempt_multiplier=args.max_attempt_multiplier,
        minimum_spacing_m=args.minimum_spacing_m,
        minimum_area_pixels=args.minimum_area_pixels,
        min_floor_clearance_m=args.min_floor_clearance_m,
        require_solver_prefilter=not args.skip_solver_prefilter,
        object_half_size_m=args.object_half_size_m,
        object_color=args.object_color,
        camera_rig_config=args.camera_rig_config,
        home_qpos=_float_tuple(args.home_qpos, expected=6),
        base_origin_xy=_float_tuple(args.base_origin_xy, expected=2),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_near_range_spawn_catalog(
    *,
    catalog_output: Path,
    evidence_output: Path,
    catalog_id: str,
    primary_count: int,
    backup_count: int,
    shard_count: int,
    radius_min_m: float,
    radius_max_m: float,
    radial_strata: int,
    angle_min_deg: float,
    angle_max_deg: float,
    angle_strata: int,
    yaw_strata: int,
    sequence_offset: int,
    max_attempt_multiplier: int,
    minimum_spacing_m: float,
    minimum_area_pixels: int,
    min_floor_clearance_m: float,
    require_solver_prefilter: bool = True,
    object_half_size_m: float,
    object_color: str,
    camera_rig_config: Path,
    home_qpos: tuple[float, ...],
    base_origin_xy: tuple[float, ...],
) -> dict[str, Any]:
    _validate_args(
        primary_count=primary_count,
        backup_count=backup_count,
        shard_count=shard_count,
        radius_min_m=radius_min_m,
        radius_max_m=radius_max_m,
        radial_strata=radial_strata,
        angle_min_deg=angle_min_deg,
        angle_max_deg=angle_max_deg,
        angle_strata=angle_strata,
        yaw_strata=yaw_strata,
        max_attempt_multiplier=max_attempt_multiplier,
        minimum_spacing_m=minimum_spacing_m,
        minimum_area_pixels=minimum_area_pixels,
    )
    total_count = int(primary_count) + int(backup_count)
    primary_per_radius = _balanced_counts(primary_count, radial_strata)
    backup_per_radius = _balanced_counts(backup_count, radial_strata)
    requested_per_radius = [
        primary + backup
        for primary, backup in zip(primary_per_radius, backup_per_radius, strict=True)
    ]
    camera_rig_declared = str(camera_rig_config)
    camera_rig_config = camera_rig_config.resolve()
    rig = load_so101_camera_rig_render_config(camera_rig_config)
    env = make_high_contrast_picklift_env(
        target_object_color=object_color,
        object_half_sizes=(object_half_size_m,),
        spawn_center=(0.15, 0.0),
        spawn_min_radius=radius_min_m,
        spawn_max_radius=radius_max_m,
        spawn_angle_half_range_deg=max(abs(angle_min_deg), abs(angle_max_deg)),
        camera_rig_preset=rig.preset,
        camera_rig_config=rig,
    )
    renderers = _make_policy_renderers(
        env,
        WristEgoServoConfig(width=256, height=256),
    )
    accepted_by_radius: list[list[dict[str, Any]]] = [
        [] for _ in range(radial_strata)
    ]
    rejections: Counter[str] = Counter()
    used_positions: list[tuple[float, float]] = []
    attempts = 0
    started = time.perf_counter()
    max_attempts = max(1, total_count * int(max_attempt_multiplier))
    try:
        stream_index = int(sequence_offset)
        while sum(len(rows) for rows in accepted_by_radius) < total_count:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "near-range candidate pool exhausted before quotas were filled: "
                    f"accepted={sum(len(rows) for rows in accepted_by_radius)}/"
                    f"{total_count}, per_radius={[len(rows) for rows in accepted_by_radius]}, "
                    f"rejections={dict(rejections)}"
                )
            radial_index = attempts % radial_strata
            attempts += 1
            if len(accepted_by_radius[radial_index]) >= requested_per_radius[radial_index]:
                continue
            sample_index = stream_index + attempts
            radius_lo, radius_hi = _radial_bounds(
                radial_index,
                radial_strata=radial_strata,
                radius_min_m=radius_min_m,
                radius_max_m=radius_max_m,
            )
            radius = _area_uniform_radius(
                radius_lo,
                radius_hi,
                _van_der_corput(sample_index, base=2),
            )
            angle_unit = _van_der_corput(sample_index, base=3)
            angle = angle_min_deg + (angle_max_deg - angle_min_deg) * angle_unit
            yaw_index = sample_index % yaw_strata
            yaw = (yaw_index + _van_der_corput(sample_index, base=5)) * (
                90.0 / yaw_strata
            )
            world_xy = (
                float(base_origin_xy[0] + radius * math.cos(math.radians(angle))),
                float(base_origin_xy[1] + radius * math.sin(math.radians(angle))),
            )
            if any(
                math.dist(world_xy, previous) < minimum_spacing_m
                for previous in used_positions
            ):
                rejections["minimum_spacing"] += 1
                continue

            env.reset(seed=0)
            _set_qpos(env, np.asarray(home_qpos, dtype=np.float32))
            _set_target_object_xy(env, world_xy)
            _set_target_object_yaw(env, yaw)
            visibility = _policy_camera_visibility(
                env,
                renderers,
                minimum_area=minimum_area_pixels,
            )
            if not any(row["visible"] for row in visibility.values()):
                rejections["initial_target_not_visible"] += 1
                continue
            teacher_candidates: list[dict[str, Any]] = []
            best: dict[str, Any] | None = None
            if require_solver_prefilter:
                teacher_candidates = _make_near_range_fixed_jaw_teacher_targets(
                    env,
                    min_floor_clearance_m=min_floor_clearance_m,
                )
                if not teacher_candidates:
                    rejections["near_contact_ik"] += 1
                    continue
                best = max(
                    teacher_candidates,
                    key=lambda row: float(row["meta"].get("score", -1e9)),
                )
            camera1_grid_bin = _camera_grid_bin(
                visibility["camera1"],
                grid_size=4,
            )
            accepted_by_radius[radial_index].append(
                {
                    "world_xy_m": [float(world_xy[0]), float(world_xy[1])],
                    "base_xy_m": [
                        float(world_xy[0] - base_origin_xy[0]),
                        float(world_xy[1] - base_origin_xy[1]),
                    ],
                    "radius_from_base_m": float(radius),
                    "angle_from_base_deg": float(angle),
                    "object_yaw_deg": float(yaw),
                    "camera1_grid_bin": camera1_grid_bin,
                    "initial_visibility": visibility,
                    "solver_candidate_count": (
                        len(teacher_candidates) if require_solver_prefilter else None
                    ),
                    "solver_best_score": (
                        float(best["meta"].get("score", 0.0)) if best else None
                    ),
                    "solver_contact_parallel_error_deg": (
                        float(best["meta"]["cube_face_normal_parallel_error_deg"])
                        if best
                        else None
                    ),
                    "solver_close_sweep_floor_clearance_m": (
                        float(best["meta"]["ik_close_sweep_floor_clearance_m"])
                        if best
                        else None
                    ),
                }
            )
            used_positions.append(world_xy)
            accepted_total = sum(len(rows) for rows in accepted_by_radius)
            if accepted_total % 25 == 0 or accepted_total == total_count:
                print(
                    "[near-catalog] "
                    f"accepted={accepted_total}/{total_count} attempts={attempts} "
                    f"per_radius={[len(rows) for rows in accepted_by_radius]} "
                    f"rejections={dict(rejections)}",
                    flush=True,
                )
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()

    primary_rows: list[dict[str, Any]] = []
    backup_rows: list[dict[str, Any]] = []
    for radial_index, rows in enumerate(accepted_by_radius):
        primary_quota = primary_per_radius[radial_index]
        primary_rows.extend(rows[:primary_quota])
        backup_rows.extend(rows[primary_quota:])
    primary_shards = _stratified_shards(primary_rows, shard_count)
    backup_shards = _stratified_shards(backup_rows, shard_count)

    evidence = {
        "format": "so101_near_range_spawn_evidence_v1",
        "catalog_id": catalog_id,
        "description": (
            "Seed-free placement evidence. Every retained placement is visible in at "
            "least one policy camera at the exact home pose. Solver feasibility is "
            + (
                "also prefiltered before export."
                if require_solver_prefilter
                else "deferred to the authoritative per-episode dataset exporter."
            )
        ),
        "camera_rig_config": camera_rig_declared,
        "camera_rig_sha256": hashlib.sha256(camera_rig_config.read_bytes()).hexdigest(),
        "home_qpos": [float(value) for value in home_qpos],
        "base_origin_world_xy_m": [float(value) for value in base_origin_xy],
        "object_color": object_color,
        "object_half_size_m": float(object_half_size_m),
        "contract": {
            "radius_range_m": [float(radius_min_m), float(radius_max_m)],
            "angle_range_deg": [float(angle_min_deg), float(angle_max_deg)],
            "radial_strata": int(radial_strata),
            "angle_strata": int(angle_strata),
            "yaw_strata": int(yaw_strata),
            "initial_visibility_mode": "any_policy_camera",
            "initial_visibility_min_area_pixels": int(minimum_area_pixels),
            "minimum_spacing_m": float(minimum_spacing_m),
            "minimum_gripper_floor_clearance_m": float(min_floor_clearance_m),
            "solver_prefilter_enabled": bool(require_solver_prefilter),
        },
        "attempts": int(attempts),
        "accepted": int(total_count),
        "rejections": dict(sorted(rejections.items())),
        "placements": [*primary_rows, *backup_rows],
    }
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    candidates: list[WorkspaceSpawnCandidate] = []
    shards: list[WorkspaceSpawnShard] = []
    source_cells: set[str] = set()
    for shard_index in range(shard_count):
        shard_primary = primary_shards[shard_index]
        shard_backup = backup_shards[shard_index]
        start_index = len(candidates)
        for stage, rows in (("primary", shard_primary), ("backup", shard_backup)):
            for local_index, row in enumerate(rows):
                angle_index = min(
                    angle_strata - 1,
                    int(
                        (float(row["angle_from_base_deg"]) - angle_min_deg)
                        / (angle_max_deg - angle_min_deg)
                        * angle_strata
                    ),
                )
                radius_index = min(
                    radial_strata - 1,
                    int(
                        (float(row["radius_from_base_m"]) - radius_min_m)
                        / (radius_max_m - radius_min_m)
                        * radial_strata
                    ),
                )
                yaw_index = min(
                    yaw_strata - 1,
                    int(float(row["object_yaw_deg"]) / 90.0 * yaw_strata),
                )
                source_cell_id = f"r{radius_index:02d}_a{angle_index:02d}_y{yaw_index:02d}"
                source_cells.add(source_cell_id)
                candidates.append(
                    WorkspaceSpawnCandidate(
                        candidate_id=(
                            f"{catalog_id}-s{shard_index:02d}-{stage}-{local_index:04d}"
                        ),
                        source_cell_id=source_cell_id,
                        stage=stage,
                        world_xy_m=row["world_xy_m"],
                        base_xy_m=row["base_xy_m"],
                        radius_from_base_m=row["radius_from_base_m"],
                        angle_from_base_deg=row["angle_from_base_deg"],
                        object_yaw_deg=row["object_yaw_deg"],
                        sampling_weight=1.0 / total_count,
                        camera1_grid_bin=row["camera1_grid_bin"],
                    )
                )
        shards.append(
            WorkspaceSpawnShard(
                shard=f"workspace_{shard_index:02d}",
                start_index=start_index,
                candidate_count=len(shard_primary) + len(shard_backup),
                primary_target_count=len(shard_primary),
            )
        )

    catalog = WorkspaceSpawnCatalog(
        format="so101_workspace_spawn_catalog_v1",
        catalog_id=catalog_id,
        source_workspace_catalog=str(evidence_output),
        source_workspace_catalog_sha256=hashlib.sha256(
            evidence_output.read_bytes()
        ).hexdigest(),
        camera_rig_config=camera_rig_declared,
        home_qpos=[float(value) for value in home_qpos],
        object_color=object_color,
        object_half_size_m=float(object_half_size_m),
        base_origin_world_xy_m=[float(value) for value in base_origin_xy],
        candidate_count=len(candidates),
        primary_target_count=int(primary_count),
        backup_count=int(backup_count),
        source_cell_count=len(source_cells),
        distance_decay_rate_per_m=1.0,
        angular_jitter_max_deg=0.0,
        radial_jitter_max_m=0.0,
        candidate_sequence_offset=int(sequence_offset),
        sampling_strategy="angular_golden_v1",
        enforce_cell_local_quota=False,
        candidates=candidates,
        shards=shards,
    )
    catalog_output.parent.mkdir(parents=True, exist_ok=True)
    catalog_output.write_text(
        json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    return {
        "catalog": str(catalog_output),
        "evidence": str(evidence_output),
        "primary_count": int(primary_count),
        "backup_count": int(backup_count),
        "attempts": int(attempts),
        "accepted": int(total_count),
        "acceptance_rate": float(total_count / max(1, attempts)),
        "rejections": dict(sorted(rejections.items())),
        "elapsed_seconds": float(elapsed),
        "shards": [row.model_dump(mode="json") for row in catalog.shards],
    }


def _validate_args(**values: Any) -> None:
    if int(values["primary_count"]) <= 0 or int(values["backup_count"]) < 0:
        raise ValueError("primary_count must be positive and backup_count non-negative")
    if int(values["shard_count"]) <= 0:
        raise ValueError("shard_count must be positive")
    if int(values["primary_count"]) % int(values["shard_count"]):
        raise ValueError("primary_count must divide evenly across shards")
    if int(values["backup_count"]) % int(values["shard_count"]):
        raise ValueError("backup_count must divide evenly across shards")
    if float(values["radius_max_m"]) <= float(values["radius_min_m"]):
        raise ValueError("radius_max_m must exceed radius_min_m")
    if float(values["angle_max_deg"]) <= float(values["angle_min_deg"]):
        raise ValueError("angle_max_deg must exceed angle_min_deg")
    for key in ("radial_strata", "angle_strata", "yaw_strata", "max_attempt_multiplier"):
        if int(values[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(values["minimum_spacing_m"]) <= 0.0:
        raise ValueError("minimum_spacing_m must be positive")
    if int(values["minimum_area_pixels"]) <= 0:
        raise ValueError("minimum_area_pixels must be positive")


def _balanced_counts(total: int, strata: int) -> list[int]:
    base, remainder = divmod(int(total), int(strata))
    return [base + (index < remainder) for index in range(strata)]


def _radial_bounds(
    index: int,
    *,
    radial_strata: int,
    radius_min_m: float,
    radius_max_m: float,
) -> tuple[float, float]:
    width = (radius_max_m - radius_min_m) / radial_strata
    return radius_min_m + index * width, radius_min_m + (index + 1) * width


def _area_uniform_radius(lo: float, hi: float, unit: float) -> float:
    return math.sqrt(lo * lo + float(unit) * (hi * hi - lo * lo))


def _van_der_corput(index: int, *, base: int) -> float:
    value = 0.0
    denominator = 1.0
    integer = int(index)
    while integer:
        integer, remainder = divmod(integer, int(base))
        denominator *= float(base)
        value += float(remainder) / denominator
    return value


def _stratified_shards(rows: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["radius_from_base_m"]),
            float(row["angle_from_base_deg"]),
            float(row["object_yaw_deg"]),
        ),
    )
    shards = [[] for _ in range(shard_count)]
    for index, row in enumerate(ordered):
        shards[index % shard_count].append(row)
    return shards


def _camera_grid_bin(visibility: dict[str, Any], *, grid_size: int) -> int | None:
    centroid = visibility.get("normalized_centroid")
    if not visibility.get("visible") or centroid is None:
        return None
    x_index = min(grid_size - 1, max(0, int(float(centroid[0]) * grid_size)))
    y_index = min(grid_size - 1, max(0, int(float(centroid[1]) * grid_size)))
    return y_index * grid_size + x_index


def _float_tuple(value: str, *, expected: int) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated values")
    return parsed


if __name__ == "__main__":
    main()
