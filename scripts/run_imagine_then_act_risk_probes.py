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
    "runpod-libero-double-sim-smoke": (6,),
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
        choices=("local-dry-run", "runpod-libero-smoke", "runpod-libero-double-sim-smoke", "runpod-libero-breadth"),
        default="local-dry-run",
    )
    parser.add_argument("--backend", choices=("mock", "libero-contract", "direct-libero"), default=None)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--chunk-steps", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--camera-mapping", default='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}')
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--actual-max-steps", type=int, default=15)
    parser.add_argument("--actual-timeout-sec", type=int, default=1800)
    parser.add_argument("--image-frequency", type=int, default=1)
    parser.add_argument("--direct-libero-double-sim", action="store_true")
    parser.add_argument("--direct-camera-name", default="agentview")
    parser.add_argument("--direct-image-width", type=int, default=128)
    parser.add_argument("--direct-image-height", type=int, default=128)
    parser.add_argument(
        "--renderer-backend",
        choices=("egl", "osmesa", "auto"),
        default="egl",
        help="Renderer backend for actual LIBERO probes. Use osmesa only as an explicit limited smoke fallback.",
    )
    parser.add_argument(
        "--debug-candidate-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Optional plumbing-only noise for non-baseline policy candidates. "
            "Reports remain WARN for debug-noise diversity; do not use for method claims."
        ),
    )
    parser.add_argument(
        "--risk1a-prompt-portfolio",
        action="store_true",
        help="Enable Risk1-A prompt/subgoal portfolio instrumentation. Default baseline remains single-prompt.",
    )
    parser.add_argument(
        "--risk1a-ambiguity",
        action="store_true",
        help="Request ambiguity-triggered Risk1-A strategy prompts. Without this flag Risk1-A preserves one prompt.",
    )
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
    if args.policy_num_steps <= 0 or args.policy_n_action_steps <= 0:
        raise ValueError("policy horizon flags must be > 0")
    if args.actual_max_steps <= 0:
        raise ValueError("actual-max-steps must be > 0")
    if args.actual_timeout_sec < 0:
        raise ValueError("actual-timeout-sec must be >= 0")
    if args.image_frequency <= 0:
        raise ValueError("image-frequency must be > 0")
    if args.direct_image_width <= 0 or args.direct_image_height <= 0:
        raise ValueError("direct image dimensions must be > 0")
    if args.debug_candidate_noise_scale < 0:
        raise ValueError("debug-candidate-noise-scale must be >= 0")
    direct_libero_double_sim = args.direct_libero_double_sim or args.preset == "runpod-libero-double-sim-smoke"
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
        policy_path=args.policy_path,
        camera_mapping=args.camera_mapping,
        policy_num_steps=args.policy_num_steps,
        policy_n_action_steps=args.policy_n_action_steps,
        actual_max_steps=args.actual_max_steps,
        actual_timeout_sec=args.actual_timeout_sec,
        image_frequency=args.image_frequency,
        direct_libero_double_sim=direct_libero_double_sim,
        direct_camera_name=args.direct_camera_name,
        direct_image_width=args.direct_image_width,
        direct_image_height=args.direct_image_height,
        renderer_backend=args.renderer_backend,
        debug_candidate_noise_scale=args.debug_candidate_noise_scale,
        risk1a_prompt_portfolio=args.risk1a_prompt_portfolio,
        risk1a_ambiguity=args.risk1a_ambiguity,
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
    if "risk1a_prompt_portfolio" in report.artifacts:
        payload["risk1a_prompt_portfolio"] = report.artifacts["risk1a_prompt_portfolio"]
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
