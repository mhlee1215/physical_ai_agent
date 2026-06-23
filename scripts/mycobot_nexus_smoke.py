#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (
    run_mycobot_nexus_smoke,
    write_dry_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a myCobot-in-Nexus-style MuJoCo reset/step/render smoke."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mycobot_nexus_smoke"))
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("MYCOBOT_MUJOCO_ROOT", "_vendor/mycobot_mujoco")),
        help="Local clone of https://github.com/elephantrobotics/mycobot_mujoco.",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--policy", choices=["sample", "cube-approach"], default="sample")
    parser.add_argument("--dry-contract", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_contract:
        contract_path = write_dry_contract(args.output_dir)
        print(json.dumps({"status": "passed", "contract_path": str(contract_path)}, indent=2))
        return
    result = run_mycobot_nexus_smoke(
        output_dir=args.output_dir,
        asset_root=args.asset_root,
        steps=args.steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        policy=args.policy,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
