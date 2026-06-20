#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from physical_ai_agent.imagine_then_act.sequential_substeps import (
    task0_push_only_substep_plan,
    task0_primitive_relaxed_substep_plan,
    task0_sequential_substep_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LIBERO-10 task0 with sequential conservative substeps.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--continue-on-required-fail", action="store_true")
    parser.add_argument(
        "--relaxed-progress-verifier",
        action="store_true",
        help=(
            "Use relaxed diagnostic substep pass criteria: target/receptacle progress "
            "or eef-to-target progress can advance a substep. Full LIBERO success is still separate."
        ),
    )
    parser.add_argument(
        "--plan-mode",
        choices=("object-pair", "primitive-relaxed", "push-only"),
        default="object-pair",
        help="Select task0 substep plan. primitive-relaxed uses reach/grasp-lift/place; push-only uses table pushes.",
    )
    parser.add_argument("--primitive-chunk-steps", type=int, default=15)
    parser.add_argument("--primitive-max-attempts", type=int, default=3)
    parser.add_argument("--primitive-lift-progress-threshold", type=float, default=0.015)
    parser.add_argument("--primitive-strict-place-verifier", action="store_true")
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--sequential-lift-assist", action="store_true")
    parser.add_argument("--sequential-lift-assist-z", type=float, default=0.6)
    parser.add_argument("--sequential-lift-assist-gripper", type=float, default=1.0)
    parser.add_argument("--sequential-primitive-assist", action="store_true")
    parser.add_argument("--sequential-primitive-assist-reach-gain", type=float, default=5.0)
    parser.add_argument("--sequential-primitive-assist-push-gain", type=float, default=5.0)
    parser.add_argument("--sequential-primitive-assist-contact-threshold", type=float, default=0.11)
    parser.add_argument("--sequential-primitive-assist-lift-z", type=float, default=0.8)
    parser.add_argument("--sequential-primitive-assist-carry-z", type=float, default=0.15)
    parser.add_argument("--sequential-primitive-assist-release-distance", type=float, default=0.12)
    parser.add_argument("--sequential-primitive-assist-push-behind-offset", type=float, default=0.08)
    parser.add_argument("--sequential-primitive-assist-close-gripper", type=float, default=-1.0)
    parser.add_argument("--sequential-primitive-assist-release-gripper", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "task0_sequential_substeps_plan.json"
    trace_path = output_dir / "benchmark_trace.jsonl"
    eval_logs = output_dir / "eval_logs"
    stdout_path = output_dir / "rollout_stdout.log"
    stderr_path = output_dir / "rollout_stderr.log"
    summary_path = output_dir / "summary.json"

    if args.plan_mode == "primitive-relaxed":
        plan = task0_primitive_relaxed_substep_plan(
            chunk_steps=int(args.primitive_chunk_steps),
            max_attempts=int(args.primitive_max_attempts),
            lift_progress_threshold=float(args.primitive_lift_progress_threshold),
            place_pass_on_progress=not bool(args.primitive_strict_place_verifier),
        )
    elif args.plan_mode == "push-only":
        plan = task0_push_only_substep_plan(
            chunk_steps=int(args.primitive_chunk_steps),
            max_attempts=int(args.primitive_max_attempts),
        )
    else:
        plan = task0_sequential_substep_plan(relaxed_progress=args.relaxed_progress_verifier)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    command = [
        args.python_bin,
        "-B",
        "scripts/run_libero_in_episode_smolvla_instrumented.py",
        "--trace-path",
        str(trace_path),
        "--trigger-mode",
        "semantic_no_progress",
        "--intervention-mode",
        "none",
        "--target-object-key",
        "alphabet_soup_1_pos",
        "--receptacle-object-key",
        "basket_1_pos",
        "--sequential-substeps-json",
        str(plan_path),
        "--output_dir",
        str(eval_logs),
        "--policy.path=lerobot/smolvla_libero",
        "--env.type=libero",
        "--env.task=libero_10",
        "--env.task_ids=[0]",
        "--env.camera_name_mapping={\"agentview_image\": \"camera1\", \"robot0_eye_in_hand_image\": \"camera2\"}",
        f"--eval.n_episodes={int(args.episodes)}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--env.max_parallel_tasks=1",
        "--policy.empty_cameras=0",
        f"--policy.num_steps={int(args.policy_num_steps)}",
        f"--policy.n_action_steps={int(args.policy_n_action_steps)}",
        f"--seed={int(args.seed)}",
    ]
    if args.continue_on_required_fail:
        command.insert(command.index("--output_dir"), "--sequential-substeps-continue-on-required-fail")
    if args.sequential_lift_assist:
        insert_at = command.index("--output_dir")
        command[insert_at:insert_at] = [
            "--sequential-lift-assist",
            "--sequential-lift-assist-z",
            str(float(args.sequential_lift_assist_z)),
            "--sequential-lift-assist-gripper",
            str(float(args.sequential_lift_assist_gripper)),
        ]
    if args.sequential_primitive_assist:
        insert_at = command.index("--output_dir")
        command[insert_at:insert_at] = [
            "--sequential-primitive-assist",
            "--sequential-primitive-assist-reach-gain",
            str(float(args.sequential_primitive_assist_reach_gain)),
            "--sequential-primitive-assist-push-gain",
            str(float(args.sequential_primitive_assist_push_gain)),
            "--sequential-primitive-assist-contact-threshold",
            str(float(args.sequential_primitive_assist_contact_threshold)),
            "--sequential-primitive-assist-lift-z",
            str(float(args.sequential_primitive_assist_lift_z)),
            "--sequential-primitive-assist-carry-z",
            str(float(args.sequential_primitive_assist_carry_z)),
            "--sequential-primitive-assist-release-distance",
            str(float(args.sequential_primitive_assist_release_distance)),
            "--sequential-primitive-assist-push-behind-offset",
            str(float(args.sequential_primitive_assist_push_behind_offset)),
            "--sequential-primitive-assist-close-gripper",
            str(float(args.sequential_primitive_assist_close_gripper)),
            "--sequential-primitive-assist-release-gripper",
            str(float(args.sequential_primitive_assist_release_gripper)),
        ]

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    env.setdefault("LIBERO_CONFIG_PATH", "/workspace/physical-ai/libero_config")
    env.setdefault("HF_HOME", "/workspace/physical-ai/hf_home")
    env["PYTHONPATH"] = env.get("PYTHONPATH") or "src"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)

    summary = build_summary(
        output_dir=output_dir,
        plan_path=plan_path,
        trace_path=trace_path,
        eval_logs=eval_logs,
        command=command,
        exit_code=completed.returncode,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return int(completed.returncode)


def build_summary(
    *,
    output_dir: Path,
    plan_path: Path,
    trace_path: Path,
    eval_logs: Path,
    command: list[str],
    exit_code: int,
) -> dict[str, Any]:
    rollout_summary = read_rollout_summary(trace_path)
    videos = sorted(str(path) for path in eval_logs.rglob("*.mp4"))
    eval_info_path = eval_logs / "eval_info.json"
    eval_info: dict[str, Any] | None = None
    if eval_info_path.exists():
        try:
            eval_info = json.loads(eval_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            eval_info = {"parse_error": "invalid_json", "path": str(eval_info_path)}
    success_rate = extract_success_rate(eval_info, rollout_summary)
    return {
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "lane": "SECURE/shallow OSMesa sequential-substep task0 data-production",
        "claim_boundary": "non-EGL OSMesa; task0 pilot/full-episode rollout evidence, not EGL deployment evidence",
        "output_dir": str(output_dir),
        "plan_path": str(plan_path),
        "trace_path": str(trace_path),
        "eval_logs": str(eval_logs),
        "eval_info_path": str(eval_info_path),
        "video_count": len(videos),
        "videos": videos,
        "success_rate": success_rate,
        "rollout_summary": rollout_summary,
        "command": command,
    }


def read_rollout_summary(trace_path: Path) -> dict[str, Any] | None:
    if not trace_path.exists():
        return None
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "rollout_summary":
            return record
    return None


def extract_success_rate(eval_info: dict[str, Any] | None, rollout_summary: dict[str, Any] | None) -> float | None:
    if isinstance(eval_info, dict):
        for key in ("pc_success", "success_rate", "eval_s/episode_success"):
            value = eval_info.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        nested = eval_info.get("aggregated")
        if isinstance(nested, dict):
            for key in ("pc_success", "success_rate"):
                value = nested.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    if isinstance(rollout_summary, dict) and isinstance(rollout_summary.get("success"), bool):
        return 100.0 if rollout_summary["success"] else 0.0
    return None


if __name__ == "__main__":
    raise SystemExit(main())
