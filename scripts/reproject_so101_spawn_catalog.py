#!/usr/bin/env python3
"""Re-bin seed-free SO101 spawn coordinates under a new camera contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from export_so101_teacher_rollouts_lerobot import (
    _object_visibility_in_camera,
    _set_target_object_xy,
    _set_target_object_yaw,
)
from physical_ai_agent.sim.so101_camera_rig_render_config import (
    load_so101_camera_rig_render_config,
)
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _make_policy_renderers,
    _set_qpos,
    make_high_contrast_picklift_env,
)


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def collect_unique_spawn_xy(paths: list[Path]) -> tuple[list[list[float]], list[dict[str, Any]]]:
    candidates: dict[tuple[float, float], list[float]] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "so101_spawn_catalog_v1":
            raise ValueError(f"source must be an so101_spawn_catalog_v1 file: {path}")
        source_count = 0
        for values in payload["lookup"].values():
            for candidate in values:
                if not isinstance(candidate, list) or len(candidate) != 2:
                    raise ValueError(f"invalid [x, y] spawn candidate in {path}: {candidate!r}")
                xy = [float(candidate[0]), float(candidate[1])]
                key = (round(xy[0], 12), round(xy[1], 12))
                candidates.setdefault(key, xy)
                source_count += 1
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "candidate_rows": source_count,
            }
        )
    ordered = [candidates[key] for key in sorted(candidates)]
    return ordered, sources


def _camera_bin(centroid: list[float], *, grid_size: int) -> int:
    bx = min(grid_size - 1, max(0, int(float(centroid[0]) * grid_size)))
    by = min(grid_size - 1, max(0, int(float(centroid[1]) * grid_size)))
    return int(by * grid_size + bx)


def translate_spawn_xy(
    candidates: list[list[float]], *, translate_x: float, translate_y: float
) -> list[list[float]]:
    return [
        [float(xy[0]) + float(translate_x), float(xy[1]) + float(translate_y)]
        for xy in candidates
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--camera-rig-config", type=Path, required=True)
    parser.add_argument("--initial-qpos", required=True)
    parser.add_argument("--target-object-yaw-deg", type=float, required=True)
    parser.add_argument("--target-object-color", default="green")
    parser.add_argument("--object-half-size", type=float, default=0.015)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--translate-x", type=float, default=0.0)
    parser.add_argument("--translate-y", type=float, default=0.0)
    args = parser.parse_args()

    initial_qpos = _csv_floats(args.initial_qpos)
    if len(initial_qpos) != 6:
        raise ValueError("--initial-qpos must contain exactly six values")
    candidates, sources = collect_unique_spawn_xy(args.source_catalog)
    if not candidates:
        raise ValueError("source catalogs contain no spawn candidates")
    candidates = translate_spawn_xy(
        candidates,
        translate_x=float(args.translate_x),
        translate_y=float(args.translate_y),
    )

    rig_path = args.camera_rig_config.resolve()
    rig = load_so101_camera_rig_render_config(rig_path)
    env = make_high_contrast_picklift_env(
        target_object_color=args.target_object_color,
        object_half_sizes=[float(args.object_half_size)],
        spawn_center=(0.15, 0.0),
        spawn_min_radius=0.1,
        spawn_max_radius=0.3,
        spawn_angle_half_range_deg=90.0,
        camera_rig_preset=rig.preset,
        camera_rig_config=rig,
    )
    renderers = _make_policy_renderers(
        env, WristEgoServoConfig(width=int(args.width), height=int(args.height))
    )
    lookup = {bin_id: [] for bin_id in range(int(args.grid_size) ** 2)}
    invisible: list[list[float]] = []
    try:
        env.reset(seed=int(args.seed))
        _set_qpos(env, np.asarray(initial_qpos, dtype=np.float32))
        for xy in candidates:
            _set_target_object_xy(env, xy)
            _set_target_object_yaw(env, float(args.target_object_yaw_deg))
            visibility = _object_visibility_in_camera(
                env, renderers["egocentric_cam"], "egocentric_cam"
            )
            centroid = visibility.get("normalized_centroid")
            if not visibility.get("visible") or centroid is None:
                invisible.append(xy)
                continue
            lookup[_camera_bin(centroid, grid_size=int(args.grid_size))].append(xy)
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()

    x_values = [xy[0] for xy in candidates]
    y_values = [xy[1] for xy in candidates]
    payload = {
        "format": "so101_spawn_catalog_v1",
        "catalog_id": str(args.catalog_id),
        "coordinate_frame": "mujoco_world_xy",
        "candidate_kind": "camera_reprojected_seed_free_spawn",
        "grid_size": int(args.grid_size),
        "resolution": len(candidates),
        "x_range": [min(x_values), max(x_values)],
        "y_range": [min(y_values), max(y_values)],
        "target_object_yaw_deg": float(args.target_object_yaw_deg),
        "initial_qpos": initial_qpos,
        "camera_rig_config": str(args.camera_rig_config),
        "camera_rig_sha256": hashlib.sha256(rig_path.read_bytes()).hexdigest(),
        "lookup": {str(key): values for key, values in sorted(lookup.items())},
        "candidate_counts": {
            str(key): len(values) for key, values in sorted(lookup.items())
        },
        "invisible_candidate_count": len(invisible),
        "coordinate_only_sources": sources,
        "coordinate_transform": {
            "kind": "world_xy_translation",
            "translate_xy": [float(args.translate_x), float(args.translate_y)],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "unique_input_candidates": len(candidates),
                "invisible_candidates": len(invisible),
                "candidate_counts": payload["candidate_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
