#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from physical_ai_agent.imagine_then_act.risk_probes import (
    RiskProbeConfig,
    diagnose_direct_libero_sim_tree,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose direct LIBERO OffScreenRenderEnv simulator state handles. "
            "This is a RunPod/Linux probe for MuJoCo get_state/set_state or qpos/qvel restore paths."
        )
    )
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--output-dir", default="_workspace/imagine_then_act/direct_libero_sim_diagnosis")
    parser.add_argument("--json", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> RiskProbeConfig:
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("image dimensions must be > 0")
    return RiskProbeConfig(
        preset="runpod-libero-double-sim-smoke",
        backend="direct-libero",
        suite=args.suite,
        task_ids=(args.task_id,),
        seed=args.seed,
        num_candidates=2,
        chunk_steps=1,
        action_dim=7,
        output_dir=args.output_dir,
        direct_libero_double_sim=True,
        direct_image_width=args.image_width,
        direct_image_height=args.image_height,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = build_config(args)
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    evidence = diagnose_direct_libero_sim_tree(config, Path(args.output_dir))
    payload = {
        "status": evidence["status"],
        "artifact_path": evidence["artifact_path"],
        "selected_handle": evidence.get("selected_handle"),
        "restore": evidence.get("restore"),
        "blockers": evidence.get("blockers", []),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status={payload['status']}")
        print(f"artifact_path={payload['artifact_path']}")
        if payload.get("selected_handle"):
            print(f"selected_handle={payload['selected_handle']}")
        if payload.get("restore"):
            print(f"restore={payload['restore']}")
    return 0 if evidence["status"] in {"PASS", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
