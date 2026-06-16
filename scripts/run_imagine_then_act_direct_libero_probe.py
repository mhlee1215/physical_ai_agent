#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from physical_ai_agent.imagine_then_act.direct_libero_imagination import (
    DirectLiberoProbeConfig,
    direct_probe_payload,
    run_direct_libero_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the direct LIBERO/MuJoCo imagination probe for Imagine-Then-Act Risk 2. "
            "Use --backend mock for dependency-free local dry-run; use --backend direct-libero on RunPod."
        )
    )
    parser.add_argument("--backend", choices=("mock", "direct-libero"), default="direct-libero")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--chunk-steps", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--output-dir", default="_workspace/imagine_then_act/direct_libero_probe")
    parser.add_argument("--json", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> DirectLiberoProbeConfig:
    for name in ("num_candidates", "chunk_steps", "action_dim", "max_steps", "image_width", "image_height"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be > 0")
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be at least 2")
    return DirectLiberoProbeConfig(
        suite=args.suite,
        task_id=args.task_id,
        seed=args.seed,
        num_candidates=args.num_candidates,
        chunk_steps=args.chunk_steps,
        action_dim=args.action_dim,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        camera_name=args.camera_name,
        image_width=args.image_width,
        image_height=args.image_height,
        backend=args.backend,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = build_config(args)
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    report = run_direct_libero_probe(config)
    payload = direct_probe_payload(report)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status={report.status}")
        print(f"summary={payload['summary_path']}")
        print(f"html_report={payload['html_report']}")
        if payload.get("direct_libero_double_sim_evidence"):
            print(f"direct_libero_double_sim_evidence={payload['direct_libero_double_sim_evidence']}")
    return 0 if report.status in {"PASS", "WARN", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
