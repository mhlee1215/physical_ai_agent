#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (  # noqa: E402
    MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
    run_mycobot_adaptive_static_contact_smoke,
)
from scripts.mycobot_adaptive_static_contact_smoke import build_parser as _shared_build_parser  # noqa: E402



def build_parser():
    parser = _shared_build_parser()
    parser.description = (
        "Run Gate 7 for the myCobot 280 Pi adaptive gripper: fixed arm, cube on the table under the finger pads, slow gripper close, sustained two-pad contact."
    )
    parser.set_defaults(
        model_profile=MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        output_dir=Path("_workspace/mycobot_280_pi_adaptive_static_contact_smoke"),
        official_gripper_root=(
            Path(os.environ["MYCOBOT_ROS_ROOT"])
            if "MYCOBOT_ROS_ROOT" in os.environ
            else Path("_vendor/mycobot_ros")
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
