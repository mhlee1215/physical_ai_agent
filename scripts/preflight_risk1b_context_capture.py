#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact Risk1-B actual LIBERO context-capture path as a "
            "preflight. This blocks Researcher handoff before Qwen/Gemma if "
            "LeRobot/LIBERO rendering cannot open the selected EGL device."
        )
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--renderer-backend", choices=("egl", "osmesa", "auto"), default="egl")
    parser.add_argument("--actual-timeout-sec", type=int, default=600)
    parser.add_argument("--output-dir", default="_workspace/runpod_results/ita_risk_probes/risk1b_context_preflight")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def build_context_capture_argv(args: argparse.Namespace) -> list[str]:
    return [
        args.python_bin,
        "-B",
        "scripts/capture_risk1b_context.py",
        "--backend",
        "libero",
        "--suite",
        args.suite,
        "--task-id",
        str(args.task_id),
        "--seed",
        str(args.seed),
        "--policy-path",
        args.policy_path,
        "--policy-num-steps",
        str(args.policy_num_steps),
        "--policy-n-action-steps",
        str(args.policy_n_action_steps),
        "--renderer-backend",
        args.renderer_backend,
        "--actual-timeout-sec",
        str(args.actual_timeout_sec),
        "--output-dir",
        str(Path(args.output_dir) / "risk1b_context"),
        "--json",
    ]


def classify_context_capture_failure(stderr: str, stdout: str = "") -> dict[str, str]:
    combined = f"{stderr}\n{stdout}"
    lowered = combined.lower()
    if "permission denied" in lowered and "/dev/dri" in lowered:
        return {
            "category": "CONTEXT_CAPTURE_LIBERO_BLOCKED_EGL_DEVICE_PERMISSION",
            "hint": (
                "The full LeRobot/LIBERO context capture path cannot open /dev/dri "
                "render/card devices. Retry on a Pod/runtime with EGL render device "
                "permissions, or make Manager block handoff before Qwen generation."
            ),
        }
    if "egl device display" in lowered or "platform_device" in lowered:
        return {
            "category": "CONTEXT_CAPTURE_LIBERO_BLOCKED_EGL_PLATFORM_DEVICE",
            "hint": (
                "EGL context creation failed in the exact context-capture path. "
                "Use a known-good renderer host or an explicitly limited OSMesa smoke."
            ),
        }
    if "timed out" in lowered or "timeout" in lowered:
        return {
            "category": "CONTEXT_CAPTURE_LIBERO_BLOCKED_TIMEOUT",
            "hint": "The context capture preflight timed out before producing actual context artifacts.",
        }
    return {
        "category": "CONTEXT_CAPTURE_LIBERO_BLOCKED_UNKNOWN",
        "hint": "Inspect preflight stdout/stderr before Qwen/Gemma generation.",
    }


def parse_json_payload(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.rfind("{")
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                return None
    return None


def run_preflight(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_context_capture_argv(args)
    if args.dry_run:
        return 0, {
            "status": "DRY_RUN",
            "operation": "risk1b_context_capture_preflight",
            "command_argv": argv,
            "claim_boundary": "dry-run only; no actual LIBERO context was captured",
        }
    started = time.time()
    try:
        completed = subprocess.run(argv, cwd=Path.cwd(), capture_output=True, text=True, timeout=args.actual_timeout_sec)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {args.actual_timeout_sec}s"
        completed = subprocess.CompletedProcess(argv, 124, stdout=stdout, stderr=stderr)

    stdout_path = output_dir / "context_capture_preflight_stdout.log"
    stderr_path = output_dir / "context_capture_preflight_stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    payload = parse_json_payload(completed.stdout or "")
    passed = completed.returncode == 0 and isinstance(payload, dict) and payload.get("status") == "PASS"
    if passed:
        report = {
            "status": "PASS",
            "operation": "risk1b_context_capture_preflight",
            "command_argv": argv,
            "elapsed_sec": round(time.time() - started, 3),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "context_capture": payload,
        }
        return 0, report

    failure = classify_context_capture_failure(completed.stderr or "", completed.stdout or "")
    report = {
        "status": "BLOCKED",
        "operation": "risk1b_context_capture_preflight",
        "blocker_category": failure["category"],
        "hint": failure["hint"],
        "command_argv": argv,
        "returncode": completed.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "claim_boundary": "Qwen/Gemma generation and Risk1-C must not run until this actual context preflight passes.",
    }
    return 2, report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, report = run_preflight(args)
    report_path = Path(args.output_dir) / "risk1b_context_capture_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status={report['status']}")
        print(f"report_path={report_path}")
        if report.get("blocker_category"):
            print(f"blocker_category={report['blocker_category']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
