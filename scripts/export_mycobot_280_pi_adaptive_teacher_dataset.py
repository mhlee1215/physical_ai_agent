#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_ai_agent.sim.mycobot_nexus_env import MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER
from scripts.export_mycobot_adaptive_teacher_dataset import build_parser, export_dataset


def main() -> None:
    parser = build_parser()
    parser.description = "Export a small myCobot 280 Pi adaptive-gripper teacher dataset from Gate 8."
    parser.set_defaults(
        model_profile=MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        output_dir=Path("_workspace/mycobot_teacher_datasets/mycobot_280_pi_adaptive_gate8_10eps"),
        official_gripper_root=Path("_vendor/mycobot_ros"),
    )
    args = parser.parse_args()
    report = export_dataset(
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        model_profile=args.model_profile,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render_every=args.render_every,
        pregrasp_steps=args.pregrasp_steps,
        close_steps=args.close_steps,
        lift_steps=args.lift_steps,
        placement_gripper_command=args.placement_gripper_command,
        close_gripper_command=args.close_gripper_command,
        cube_half_size=args.cube_half_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["episodes"] != args.episodes or report["failed_episodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
