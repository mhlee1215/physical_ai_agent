#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.sim.so101_visual_rl import SO101VisualRLConfig, run_visual_rl_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test SO101 visual RL observations.")
    parser.add_argument("--env-id", default="MuJoCoPickLift-v1")
    parser.add_argument("--camera-name", default="wrist_cam")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--channel-last", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/so101_visual_rl/smoke"),
    )
    args = parser.parse_args()
    report = run_visual_rl_smoke(
        output_dir=args.output_dir,
        config=SO101VisualRLConfig(
            env_id=args.env_id,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
            include_state=not args.no_state,
            channel_first=not args.channel_last,
        ),
        steps=args.steps,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
