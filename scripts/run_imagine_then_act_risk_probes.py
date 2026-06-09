#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from physical_ai_agent.imagine_then_act.risk_probes import RiskProbeConfig, run_risk_probes


DEFAULT_TASK_IDS = {
    "local-dry-run": (6,),
    "runpod-libero-smoke": (6,),
    "runpod-libero-breadth": tuple(range(10)),
}


def parse_task_ids(raw_value: str | None, preset: str) -> tuple[int, ...]:
    if raw_value is None:
        return DEFAULT_TASK_IDS[preset]
    task_ids: list[int] = []
    for part in (item.strip() for item in raw_value.split(",")):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid descending task range: {part}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(part))
    if not task_ids:
        raise ValueError("task ids must include at least one id")
    return tuple(dict.fromkeys(task_ids))


def default_output_dir(preset: str) -> str:
    return str(Path("_workspace") / "imagine_then_act" / "risk_probes" / preset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Imagine-Then-Act stage-2 risk probes. BLOCKED exits 0 because it is an "
            "expected contract-only outcome; FAIL exits nonzero."
        )
    )
    parser.add_argument(
        "--preset",
        choices=("local-dry-run", "runpod-libero-smoke", "runpod-libero-breadth"),
        default="local-dry-run",
    )
    parser.add_argument("--backend", choices=("mock", "libero-contract"), default=None)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--chunk-steps", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def backend_for_preset(preset: str, requested_backend: str | None) -> str:
    if requested_backend:
        return requested_backend
    if preset == "local-dry-run":
        return "mock"
    return "libero-contract"


def build_config(args: argparse.Namespace) -> RiskProbeConfig:
    task_ids = parse_task_ids(args.task_ids, args.preset)
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be at least 2")
    if args.chunk_steps <= 0:
        raise ValueError("chunk-steps must be > 0")
    if args.action_dim <= 0:
        raise ValueError("action-dim must be > 0")
    return RiskProbeConfig(
        preset=args.preset,
        backend=backend_for_preset(args.preset, args.backend),
        suite=args.suite,
        task_ids=task_ids,
        seed=args.seed,
        num_candidates=args.num_candidates,
        chunk_steps=args.chunk_steps,
        action_dim=args.action_dim,
        output_dir=args.output_dir or default_output_dir(args.preset),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = build_config(args)
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 2
    report = run_risk_probes(config)
    payload = {
        "status": report.status,
        "risk_verdicts": report.risk_verdicts,
        "output_dir": config.output_dir,
        "summary_path": report.artifacts["summary"],
        "events_path": report.artifacts["events"],
        "html_report": report.artifacts["html_report"],
        "blockers": report.blockers,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status={report.status}")
        print(f"output_dir={config.output_dir}")
        print(f"summary={report.artifacts['summary']}")
        print(f"html_report={report.artifacts['html_report']}")
    return 0 if report.status in {"PASS", "WARN", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
