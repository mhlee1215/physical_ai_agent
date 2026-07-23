#!/usr/bin/env python3
"""Build a seed-free SO101 world-XY catalog using the declared camera contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from export_so101_teacher_rollouts_lerobot import (
    _build_camera1_spawn_lookup,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--camera-rig-config", type=Path, required=True)
    parser.add_argument("--initial-qpos", required=True)
    parser.add_argument("--target-object-yaw-deg", type=float, required=True)
    parser.add_argument("--target-object-color", default="green")
    parser.add_argument("--object-half-size", type=float, default=0.015)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    initial_qpos = _csv_floats(args.initial_qpos)
    if len(initial_qpos) != 6:
        raise ValueError("--initial-qpos must contain exactly six values")
    rig = load_so101_camera_rig_render_config(args.camera_rig_config.resolve())
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
    try:
        env.reset(seed=int(args.seed))
        _set_qpos(env, np.asarray(initial_qpos, dtype=np.float32))
        _set_target_object_yaw(env, float(args.target_object_yaw_deg))
        lookup = _build_camera1_spawn_lookup(
            env,
            renderers,
            grid_size=int(args.grid_size),
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            y_min=float(args.y_min),
            y_max=float(args.y_max),
            resolution=int(args.resolution),
            target_object_yaw_deg=float(args.target_object_yaw_deg),
        )
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()

    rig_path = args.camera_rig_config.resolve()
    all_bins = range(int(args.grid_size) ** 2)
    payload = {
        "format": "so101_spawn_catalog_v1",
        "catalog_id": str(args.catalog_id),
        "coordinate_frame": "mujoco_world_xy",
        "grid_size": int(args.grid_size),
        "resolution": int(args.resolution),
        "x_range": [float(args.x_min), float(args.x_max)],
        "y_range": [float(args.y_min), float(args.y_max)],
        "target_object_yaw_deg": float(args.target_object_yaw_deg),
        "initial_qpos": initial_qpos,
        "camera_rig_config": str(args.camera_rig_config),
        "camera_rig_sha256": hashlib.sha256(rig_path.read_bytes()).hexdigest(),
        "lookup": {
            str(bin_id): lookup.get(bin_id, [])
            for bin_id in all_bins
        },
    }
    payload["candidate_counts"] = {
        key: len(values) for key, values in payload["lookup"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "candidate_counts": payload["candidate_counts"]}, indent=2))


if __name__ == "__main__":
    main()
