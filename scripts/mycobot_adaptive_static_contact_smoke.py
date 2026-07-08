#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (  # noqa: E402
    MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
    MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    run_mycobot_adaptive_static_contact_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gate 7 for the myCobot 320 adaptive gripper: fixed arm, "
            "cube on the table under the finger pads, slow gripper close, "
            "sustained two-pad contact."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_adaptive_static_contact_smoke"),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("MYCOBOT_MUJOCO_ROOT", "_vendor/mycobot_mujoco")),
        help="Local clone of https://github.com/elephantrobotics/mycobot_mujoco.",
    )
    parser.add_argument(
        "--official-gripper-root",
        type=Path,
        default=(
            Path(os.environ["MYCOBOT_ROS2_ROOT"])
            if "MYCOBOT_ROS2_ROOT" in os.environ
            else Path(os.environ["MYCOBOT_ROS_ROOT"])
            if "MYCOBOT_ROS_ROOT" in os.environ
            else Path("_vendor/mycobot_ros2")
        ),
        help="Local clone containing the ROS2 Humble myCobot 320 adaptive gripper URDF.",
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument(
        "--model-profile",
        choices=[MODEL_PROFILE_320_ADAPTIVE_GRIPPER, MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER],
        default=MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
        help="Adaptive myCobot source profile to run through this gate.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--placement-gripper-command", type=float, default=0.25)
    parser.add_argument("--final-gripper-command", type=float, default=-1.0)
    parser.add_argument("--required-sustained-steps", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_mycobot_adaptive_static_contact_smoke(
        output_dir=args.output_dir,
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        model_profile=args.model_profile,
        steps=args.steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        placement_gripper_command=args.placement_gripper_command,
        final_gripper_command=args.final_gripper_command,
        required_sustained_steps=args.required_sustained_steps,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if result.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
