#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMPORTANT_CLOSED_LOOP_SUCCESS_RATE_TAG = "important/closed_loop_success_rate"
from zoneinfo import ZoneInfo

from physical_ai_agent.so101_smolvla_pipeline import (
    SO101TrainingSchedule,
    detect_overfit_stop,
    should_run_closed_loop,
)


LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodically record SO101 training progress for the dashboard.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-s", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--policy-device", default="mps", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Checkpoint directory. Defaults to run-dir/checkpoints or run-dir/model/checkpoints.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--defer-validation-while-training", action="store_true")
    parser.add_argument("--min-validation-step", type=int, default=0)
    parser.add_argument("--train-pid-file", type=Path)
    parser.add_argument("--closed-loop-every-epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=138)
    parser.add_argument("--closed-loop-episodes", type=int, default=10)
    parser.add_argument("--closed-loop-steps", type=int, default=160)
    parser.add_argument("--closed-loop-seed", type=int, default=98100)
    parser.add_argument("--closed-loop-test-id", default="default")
    parser.add_argument(
        "--closed-loop-success-metric",
        default="env_success",
        choices=["env_success", "tcp_to_object_below_threshold"],
    )
    parser.add_argument("--closed-loop-success-threshold", type=float, default=0.06)
    parser.add_argument("--closed-loop-start-contract", default="default_reset")
    parser.add_argument("--closed-loop-start-report-path", type=Path)
    parser.add_argument("--closed-loop-env-id", default="MuJoCoPickLift-v1")
    parser.add_argument("--closed-loop-width", type=int, default=256)
    parser.add_argument("--closed-loop-height", type=int, default=256)
    parser.add_argument(
        "--closed-loop-runner",
        choices=["picklift", "qwen_chain"],
        default="picklift",
        help="Closed-loop evaluator implementation to run for each scheduled checkpoint.",
    )
    parser.add_argument(
        "--mujoco-gl",
        choices=["auto", "glfw", "egl", "osmesa"],
        default="auto",
        help="MuJoCo rendering backend for validation rollouts. auto uses glfw on macOS and egl on Linux.",
    )
    parser.add_argument("--closed-loop-task-prompt")
    parser.add_argument(
        "--closed-loop-action-contract-mode",
        choices=["processor", "legacy", "processor_dataset_clamp", "processor_delta_q", "visual_servo_delta_q"],
        default="processor",
        help="Action conversion mode forwarded to the closed-loop policy evaluator.",
    )
    parser.add_argument(
        "--closed-loop-eval-skill-mode",
        choices=["picklift", "pick_from_top_cube", "pick_and_place_cube"],
        default="picklift",
    )
    parser.add_argument("--closed-loop-record-rollout-gif", action="store_true")
    parser.add_argument("--record-loop-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-loop-media", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop-artifact-width", type=int, default=256)
    parser.add_argument("--loop-artifact-height", type=int, default=256)
    parser.add_argument("--loop-artifact-fps", type=int, default=12)
    parser.add_argument("--loop-artifact-every-n-steps", type=int, default=1)
    parser.add_argument("--qwen-model", default="qwen3-vl-8b-instruct-mlx")
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-api-key")
    parser.add_argument("--qwen-response-json", type=Path)
    parser.add_argument("--qwen-plan-json", type=Path)
    parser.add_argument("--closed-loop-precondition-plan-json", type=Path)
    parser.add_argument("--qwen-object", default="green cube")
    parser.add_argument("--qwen-env-object-color", default="green")
    parser.add_argument("--closed-loop-subgoal-chain-mode", choices=["off", "fixed", "valid-mask"], default="off")
    parser.add_argument("--closed-loop-subgoal-sequence")
    parser.add_argument("--closed-loop-fixed-subgoal-chunks", type=int, default=1)
    parser.add_argument("--closed-loop-valid-mask-checkpoint", type=Path)
    parser.add_argument("--closed-loop-valid-mask-threshold", type=float, default=0.5)
    parser.add_argument("--closed-loop-valid-mask-consecutive", type=int, default=2)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--closed-loop-input-grid-count", type=int, default=16)
    parser.add_argument("--torch-seed", type=int, default=1000)
    parser.add_argument("--skip-closed-loop", action="store_true")
    parser.add_argument(
        "--closed-loop-policy",
        choices=["off", "periodic", "best_only", "best_or_periodic"],
        default=None,
        help="Explicit closed-loop scheduling policy. Overrides --skip-closed-loop/--closed-loop-best-only.",
    )
    parser.add_argument(
        "--closed-loop-best-only",
        action="store_true",
        help="Run closed-loop validation only when this checkpoint is the best supervised validation loss so far.",
    )
    parser.add_argument("--stop-training-on-overfit", action="store_true")
    parser.add_argument("--overfit-patience-checkpoints", type=int, default=3)
    parser.add_argument("--overfit-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "monitor_events.jsonl").touch()
    (metrics_dir / "validation_metrics.jsonl").touch()
    (metrics_dir / "closed_loop_metrics.jsonl").touch()
    iteration = 0
    while True:
        iteration += 1
        check_once(args, run_dir)
        if args.iterations and iteration >= args.iterations:
            break
        time.sleep(max(1, int(args.interval_s)))


def check_once(args: argparse.Namespace, run_dir: Path) -> None:
    checkpoint_root = _checkpoint_root(args, run_dir)
    checkpoints = _checkpoint_names(checkpoint_root)
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    _append_event(
        run_dir,
        {
            "kind": "periodic_check",
            "detail": f"{len(checkpoints)} checkpoints; latest={latest_checkpoint or 'none'}",
            "checkpoint": latest_checkpoint,
            "checkpoint_count": len(checkpoints),
        },
    )
    _update_loss_summary(run_dir, latest_checkpoint)
    if args.skip_validation:
        return
    train_process = _process_status(args.train_pid_file)
    for checkpoint in checkpoints:
        if checkpoint == "last":
            continue
        checkpoint_step = _checkpoint_to_step(checkpoint)
        if checkpoint_step is not None and checkpoint_step < args.min_validation_step:
            continue
        policy_path = checkpoint_root / checkpoint / "pretrained_model"
        if not policy_path.exists():
            continue
        validation_recorded = _validation_already_recorded(run_dir, checkpoint)
        if not validation_recorded:
            if args.defer_validation_while_training and train_process.get("alive"):
                if not _validation_deferred_already_recorded(run_dir, checkpoint):
                    _append_event(
                        run_dir,
                        {
                            "kind": "validation_deferred",
                            "detail": (
                                f"deferred validation for checkpoint {checkpoint}; "
                                "training is still active on the same GPU"
                            ),
                            "checkpoint": checkpoint,
                            "train_process": train_process,
                        },
                    )
                    _update_loss_summary(run_dir, checkpoint)
                continue
            _append_event(
                run_dir,
                {
                    "kind": "validation_start",
                    "detail": f"computing validation loss for checkpoint {checkpoint}",
                    "checkpoint": checkpoint,
                },
            )
            try:
                report = _run_validation_loss(args, run_dir, checkpoint, policy_path)
            except Exception as exc:  # noqa: BLE001
                _append_event(
                    run_dir,
                    {
                        "kind": "validation_error",
                        "detail": str(exc),
                        "checkpoint": checkpoint,
                    },
                )
                continue
            _append_validation_metric(run_dir, checkpoint, report, args)
            _append_event(
                run_dir,
                {
                    "kind": "validation_done",
                    "detail": f"checkpoint {checkpoint} val_loss={report['loss_mean']:.6f}",
                    "checkpoint": checkpoint,
                    "loss": report["loss_mean"],
                },
            )
            _update_loss_summary(run_dir, checkpoint)
            if _maybe_stop_training_on_overfit(args, run_dir, checkpoint, train_process):
                continue
        if not _should_run_closed_loop(args, run_dir, checkpoint):
            continue
        _append_event(
            run_dir,
            {
                "kind": "closed_loop_start",
                "detail": f"running closed-loop validation for checkpoint {checkpoint}",
                "checkpoint": checkpoint,
            },
        )
        try:
            closed_loop_report = _run_closed_loop_eval(args, run_dir, checkpoint, policy_path)
        except Exception as exc:  # noqa: BLE001
            _append_event(
                run_dir,
                {
                    "kind": "closed_loop_error",
                    "detail": str(exc),
                    "checkpoint": checkpoint,
                },
            )
            continue
        _append_closed_loop_metric(run_dir, checkpoint, closed_loop_report)
        _append_event(
            run_dir,
            {
                "kind": "closed_loop_done",
                "detail": (
                    f"checkpoint {checkpoint} "
                    f"success={_fmt_optional_float(closed_loop_report.get('success_rate'))} "
                    f"grasp={_fmt_optional_float(closed_loop_report.get('grasp_rate'))}"
                ),
                "checkpoint": checkpoint,
                "success_rate": closed_loop_report.get("success_rate"),
                "grasp_rate": closed_loop_report.get("grasp_rate"),
            },
        )
        _update_loss_summary(run_dir, checkpoint)


def _run_validation_loss(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: str,
    policy_path: Path,
) -> dict[str, Any]:
    output_path = run_dir / "metrics" / f"loss_eval_{checkpoint}_val24_sample{args.max_batches}.json"
    cmd = [
        args.python,
        str(args.repo_root / "scripts" / "evaluate_smolvla_supervised_loss.py"),
        "--policy-path",
        str(policy_path),
        "--dataset-root",
        str(args.dataset_root),
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--output-path",
        str(output_path),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        "0",
        "--max-batches",
        str(args.max_batches),
        "--device",
        args.policy_device,
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    completed = subprocess.run(cmd, cwd=args.repo_root, env=_runtime_env(args), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or f"validation failed: {completed.returncode}")[-2000:])
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_closed_loop_eval(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: str,
    policy_path: Path,
) -> dict[str, Any]:
    if args.closed_loop_runner == "qwen_chain":
        return _run_qwen_chain_closed_loop_eval(args, run_dir, checkpoint, policy_path)
    return _run_picklift_closed_loop_eval(args, run_dir, checkpoint, policy_path)


def _checkpoint_valid_mask_head(policy_path: Path) -> Path | None:
    candidate = Path(policy_path).parent / "valid_mask_head.pt"
    return candidate if candidate.exists() else None


def _run_picklift_closed_loop_eval(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: str,
    policy_path: Path,
) -> dict[str, Any]:
    closed_loop_test_id = str(getattr(args, "closed_loop_test_id", "default") or "default")
    output_dir = run_dir / "closed_loop_evals" / (
        f"{_safe_id(closed_loop_test_id)}_seed{args.closed_loop_seed}_"
        f"nact{args.policy_n_action_steps}_{checkpoint}"
    )
    cmd = [
        args.python,
        str(args.repo_root / "scripts" / "evaluate_so101_picklift_smolvla_policy.py"),
        "--policy-path",
        str(policy_path),
        "--output-dir",
        str(output_dir),
        "--episodes",
        str(args.closed_loop_episodes),
        "--steps",
        str(args.closed_loop_steps),
        "--seed",
        str(args.closed_loop_seed),
        "--device",
        args.policy_device,
        "--width",
        str(args.closed_loop_width),
        "--height",
        str(args.closed_loop_height),
        "--policy-n-action-steps",
        str(args.policy_n_action_steps),
        "--policy-num-steps",
        str(args.policy_num_steps),
        "--sample-input-grid-count",
        str(args.closed_loop_input_grid_count),
        "--torch-seed",
        str(args.torch_seed),
        "--eval-skill-mode",
        args.closed_loop_eval_skill_mode,
        "--record-rollout-gif" if args.closed_loop_record_rollout_gif else "--no-record-rollout-gif",
        "--subgoal-chain-mode",
        args.closed_loop_subgoal_chain_mode,
        "--fixed-subgoal-chunks",
        str(args.closed_loop_fixed_subgoal_chunks),
        "--valid-mask-threshold",
        str(args.closed_loop_valid_mask_threshold),
        "--valid-mask-consecutive",
        str(args.closed_loop_valid_mask_consecutive),
    ]
    if args.closed_loop_subgoal_sequence:
        cmd.extend(["--subgoal-sequence", args.closed_loop_subgoal_sequence])
    valid_mask_checkpoint = _checkpoint_valid_mask_head(policy_path) or args.closed_loop_valid_mask_checkpoint
    if valid_mask_checkpoint:
        cmd.extend(["--valid-mask-checkpoint", str(valid_mask_checkpoint)])
    if args.closed_loop_task_prompt:
        cmd.extend(["--task-prompt", args.closed_loop_task_prompt])
    if args.closed_loop_eval_skill_mode == "pick_from_top_cube":
        cmd.extend(
            [
                "--no-sweep",
                "--pick-start-min-actual-z",
                "0.05",
                "--pick-start-min-actual-abs-y",
                "0.015",
                "--pick-start-max-actual-abs-y",
                "0.065",
                "--pick-start-z-offset",
                "0.7",
            ]
        )
    if args.local_files_only:
        cmd.append("--local-files-only")
    completed = subprocess.run(cmd, cwd=args.repo_root, env=_runtime_env(args), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or f"closed-loop eval failed: {completed.returncode}")[-2000:])
    report_path = output_dir / "so101_picklift_smolvla_eval_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("closed_loop_test_id", closed_loop_test_id)
    return report


def _run_qwen_chain_closed_loop_eval(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: str,
    policy_path: Path,
) -> dict[str, Any]:
    valid_mask_checkpoint = _checkpoint_valid_mask_head(policy_path) or args.closed_loop_valid_mask_checkpoint
    if valid_mask_checkpoint is None:
        raise RuntimeError("qwen_chain closed-loop requires --closed-loop-valid-mask-checkpoint")
    closed_loop_test_id = str(getattr(args, "closed_loop_test_id", "default") or "default")
    output_dir = run_dir / "closed_loop_evals" / (
        f"qwen_chain_{_safe_id(closed_loop_test_id)}_seed{args.closed_loop_seed}_{checkpoint}"
    )
    cmd = [
        args.python,
        str(args.repo_root / "scripts" / "run_so101_qwen_closed_loop_eval.py"),
        "--task",
        args.closed_loop_task_prompt or "pick and lift the green cube",
        "--object",
        args.qwen_object,
        "--qwen-model",
        args.qwen_model,
        "--env-id",
        args.closed_loop_env_id,
        "--env-object-color",
        args.qwen_env_object_color,
        "--policy-path",
        str(policy_path),
        "--output-dir",
        str(output_dir),
        "--episodes",
        str(args.closed_loop_episodes),
        "--seed",
        str(args.closed_loop_seed),
        "--start-contract",
        args.closed_loop_start_contract,
        "--device",
        args.policy_device,
        "--max-steps-per-primitive",
        str(args.closed_loop_steps),
        "--policy-n-action-steps",
        str(args.policy_n_action_steps),
        "--policy-num-steps",
        str(args.policy_num_steps),
        "--action-contract-mode",
        getattr(args, "closed_loop_action_contract_mode", "processor"),
        "--valid-mask-checkpoint",
        str(valid_mask_checkpoint),
        "--valid-mask-threshold",
        str(args.closed_loop_valid_mask_threshold),
        "--valid-mask-consecutive",
        str(args.closed_loop_valid_mask_consecutive),
    ]
    closed_loop_start_report_path = getattr(args, "closed_loop_start_report_path", None)
    if closed_loop_start_report_path is not None:
        cmd.extend(["--start-report-path", str(closed_loop_start_report_path)])
    if args.record_loop_artifacts:
        cmd.extend(
            [
                "--record-loop-artifacts",
                "--render-loop-media" if args.render_loop_media else "--no-render-loop-media",
                "--artifact-width",
                str(args.loop_artifact_width),
                "--artifact-height",
                str(args.loop_artifact_height),
                "--artifact-fps",
                str(args.loop_artifact_fps),
                "--artifact-every-n-steps",
                str(args.loop_artifact_every_n_steps),
            ]
        )
    if args.qwen_plan_json:
        cmd.extend(["--qwen-plan-json", str(args.qwen_plan_json)])
    elif args.qwen_response_json:
        cmd.extend(["--qwen-response-json", str(args.qwen_response_json)])
    elif args.qwen_base_url:
        cmd.extend(["--qwen-base-url", args.qwen_base_url])
    else:
        raise RuntimeError(
            "qwen_chain closed-loop requires --qwen-response-json, --qwen-plan-json, or --qwen-base-url"
        )
    if args.qwen_api_key:
        cmd.extend(["--qwen-api-key", args.qwen_api_key])
    if getattr(args, "closed_loop_precondition_plan_json", None):
        cmd.extend(["--precondition-plan-json", str(args.closed_loop_precondition_plan_json)])
    if not args.local_files_only:
        cmd.append("--allow-download")
    completed = subprocess.run(cmd, cwd=args.repo_root, env=_runtime_env(args), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or f"Qwen closed-loop eval failed: {completed.returncode}")[-2000:])
    report_path = output_dir / "qwen_closed_loop_eval_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("eval_skill_mode", "qwen_edge_chain")
    report.setdefault("closed_loop_test_id", closed_loop_test_id)
    report.setdefault("start_contract", args.closed_loop_start_contract)
    report.setdefault("task_prompt", args.closed_loop_task_prompt or (report.get("plan") or {}).get("task"))
    _apply_closed_loop_success_metric(report, args)
    return report


def _apply_closed_loop_success_metric(report: dict[str, Any], args: argparse.Namespace) -> None:
    metric = str(getattr(args, "closed_loop_success_metric", "env_success") or "env_success")
    if metric == "env_success":
        return
    if metric != "tcp_to_object_below_threshold":
        raise RuntimeError(f"unsupported closed-loop success metric: {metric}")
    threshold = float(getattr(args, "closed_loop_success_threshold", 0.06))
    episodes = report.get("episodes") or []
    rows = []
    passed = 0
    for index, episode in enumerate(episodes):
        distance = _episode_final_tcp_to_obj_dist(episode)
        episode_passed = distance is not None and distance <= threshold
        rows.append(
            {
                "episode": index,
                "tcp_to_obj_dist": distance,
                "threshold": threshold,
                "passed": bool(episode_passed),
            }
        )
        passed += int(bool(episode_passed))
    report["env_success_rate"] = report.get("success_rate")
    report["success_metric"] = metric
    report["success_threshold"] = threshold
    report["success_rate"] = (passed / len(rows)) if rows else None
    report["gripper_above_object_rate"] = report["success_rate"]
    report["gripper_above_object_episodes"] = rows


def _episode_final_tcp_to_obj_dist(episode: dict[str, Any]) -> float | None:
    final_info = episode.get("final_info") if isinstance(episode, dict) else None
    if isinstance(final_info, dict) and final_info.get("tcp_to_obj_dist") is not None:
        return float(final_info["tcp_to_obj_dist"])
    trace_path = Path(str(episode.get("trace_path") or "")) if isinstance(episode, dict) else Path()
    if not trace_path.exists():
        return None
    rows = _read_jsonl(trace_path)
    for row in reversed(rows):
        info = row.get("info") if isinstance(row, dict) else None
        if isinstance(info, dict) and info.get("tcp_to_obj_dist") is not None:
            return float(info["tcp_to_obj_dist"])
    return None


def _runtime_env(args: argparse.Namespace) -> dict[str, str]:
    mujoco_gl, pyopengl_platform = _mujoco_render_env(args.mujoco_gl)
    env = {
        **os.environ,
        "PATH": str(Path(args.python).parent) + ":" + os.environ.get("PATH", ""),
        "PYTHONPATH": str(args.repo_root / "src"),
        "HF_DATASETS_CACHE": str(args.repo_root / "_workspace" / "hf_datasets_cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MUJOCO_GL": mujoco_gl,
    }
    if pyopengl_platform is None:
        env.pop("PYOPENGL_PLATFORM", None)
    else:
        env["PYOPENGL_PLATFORM"] = pyopengl_platform
    if args.policy_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _mujoco_render_env(requested: str = "auto") -> tuple[str, str | None]:
    if requested != "auto":
        if requested == "egl":
            return requested, os.environ.get("PYOPENGL_PLATFORM", "egl")
        if requested == "glfw":
            return requested, None
        return requested, os.environ.get("PYOPENGL_PLATFORM")
    requested = os.environ.get("MUJOCO_GL")
    system = platform.system().lower()
    if system == "darwin":
        if requested in {"glfw", "osmesa"}:
            return requested, None
        return "glfw", None
    mujoco_gl = requested or "egl"
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM")
    if pyopengl_platform is None and mujoco_gl == "egl":
        pyopengl_platform = "egl"
    return mujoco_gl, pyopengl_platform


def _checkpoint_root(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.checkpoint_root is not None:
        return args.checkpoint_root if args.checkpoint_root.is_absolute() else run_dir / args.checkpoint_root
    for candidate in (run_dir / "checkpoints", run_dir / "model" / "checkpoints"):
        if candidate.exists():
            return candidate
    return run_dir / "checkpoints"


def _checkpoint_names(checkpoints_dir: Path) -> list[str]:
    if not checkpoints_dir.exists():
        return []
    return sorted(path.name for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit())


def _validation_already_recorded(run_dir: Path, checkpoint: str) -> bool:
    path = run_dir / "metrics" / "validation_metrics.jsonl"
    for row in _read_jsonl(path):
        if str(row.get("checkpoint")) == checkpoint:
            return True
    return False


def _validation_deferred_already_recorded(run_dir: Path, checkpoint: str) -> bool:
    path = run_dir / "metrics" / "monitor_events.jsonl"
    for row in _read_jsonl(path):
        if row.get("kind") == "validation_deferred" and str(row.get("checkpoint")) == checkpoint:
            return True
    return False


def _process_status(pid_file: Path | None) -> dict[str, Any]:
    if pid_file is None or not pid_file.exists():
        return {"alive": None}
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return {"alive": False, "pid": None}
    text = _run_text(["ps", "-p", str(pid), "-o", "pid=,stat=,etime=,pcpu=,pmem=,rss="]).strip()
    if not text:
        return {"alive": False, "pid": pid}
    parts = text.split()
    row: dict[str, Any] = {"alive": True, "pid": pid}
    if len(parts) >= 6:
        row.update(
            {
                "stat": parts[1],
                "elapsed": parts[2],
                "cpu_percent": _float_or_none(parts[3]),
                "mem_percent": _float_or_none(parts[4]),
                "rss_gb": _bytes_to_gb(int(parts[5]) * 1024),
            }
        )
    return row


def _run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return ""


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _fmt_optional_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 2)


def _should_run_closed_loop(args: argparse.Namespace, run_dir: Path, checkpoint: str) -> bool:
    policy = _closed_loop_policy(args)
    test_id = str(getattr(args, "closed_loop_test_id", "default") or "default")
    closed_loop_rows = [
        row
        for row in _read_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl")
        if str(row.get("test_id", "default")) == test_id
    ]
    return should_run_closed_loop(
        schedule=SO101TrainingSchedule(
            closed_loop_policy=policy,
            closed_loop_every_epochs=args.closed_loop_every_epochs,
            steps_per_epoch=args.steps_per_epoch,
            stop_on_overfit=args.stop_training_on_overfit,
            overfit_patience_checkpoints=args.overfit_patience_checkpoints,
            overfit_min_delta=args.overfit_min_delta,
        ),
        checkpoint=checkpoint,
        validation_rows=_read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl"),
        closed_loop_rows=closed_loop_rows,
    )


def _closed_loop_policy(args: argparse.Namespace) -> str:
    if args.closed_loop_policy is not None:
        return args.closed_loop_policy
    if args.skip_closed_loop or args.closed_loop_every_epochs <= 0:
        return "off"
    if args.closed_loop_best_only:
        return "best_only"
    return "periodic"


def _maybe_stop_training_on_overfit(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: str,
    train_process: dict[str, Any],
) -> bool:
    if not args.stop_training_on_overfit or not train_process.get("alive"):
        return False
    decision = detect_overfit_stop(
        _read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl"),
        patience_checkpoints=args.overfit_patience_checkpoints,
        min_delta=args.overfit_min_delta,
    )
    if not decision.get("should_stop"):
        return False
    pid = train_process.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 15)
        stopped = True
        detail = f"sent SIGTERM to training pid {pid}: {decision.get('reason')}"
    except OSError as exc:
        stopped = False
        detail = f"failed to stop training pid {pid}: {exc}"
    _append_event(
        run_dir,
        {
            "kind": "training_stop_overfit",
            "detail": detail,
            "checkpoint": checkpoint,
            "stopped": stopped,
            "overfit_decision": decision,
        },
    )
    _update_loss_summary(run_dir, checkpoint)
    return stopped


def _is_best_validation_checkpoint(run_dir: Path, checkpoint: str) -> bool:
    rows = [
        row
        for row in _read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl")
        if row.get("loss") is not None and row.get("checkpoint") is not None
    ]
    if not rows:
        return False
    best = min(rows, key=lambda row: float(row["loss"]))
    return str(best.get("checkpoint")) == checkpoint


def _closed_loop_already_recorded(run_dir: Path, checkpoint: str) -> bool:
    path = run_dir / "metrics" / "closed_loop_metrics.jsonl"
    for row in _read_jsonl(path):
        if str(row.get("checkpoint")) == checkpoint:
            return True
    return False


def _append_validation_metric(
    run_dir: Path,
    checkpoint: str,
    report: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    row = {
        "step": _checkpoint_to_step(checkpoint),
        "loss": report["loss_mean"],
        "checkpoint": checkpoint,
        "split": "validation",
        "batches_evaluated": report.get("batches_evaluated"),
        "batch_size": report.get("batch_size", args.batch_size),
        "samples_seen": report.get("samples_seen"),
        "source": Path(report.get("output_path", "")).name,
    }
    _append_jsonl(run_dir / "metrics" / "validation_metrics.jsonl", row)


def _append_closed_loop_metric(run_dir: Path, checkpoint: str, report: dict[str, Any]) -> None:
    row = {
        "step": _checkpoint_to_step(checkpoint),
        "checkpoint": checkpoint,
        "test_id": report.get("closed_loop_test_id", "default"),
        "loop_validation_id": report.get("closed_loop_test_id", "default"),
        "operation": report.get("operation"),
        "success_rate": report.get("success_rate"),
        "env_success_rate": report.get("env_success_rate"),
        "grasp_rate": report.get("grasp_rate"),
        "eval_skill_mode": report.get("eval_skill_mode"),
        "task_prompt": report.get("task_prompt"),
        "episodes": len(report.get("episodes") or []),
        "duration_s": report.get("duration_s"),
        "report_path": report.get("report_path"),
        "policy_rollout_config": report.get("policy_rollout_config"),
    }
    _append_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl", row)
    _write_closed_loop_tensorboard(run_dir, row, report)


def _write_closed_loop_tensorboard(run_dir: Path, row: dict[str, Any], report: dict[str, Any]) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return

    # Keep loop-test writers out of the main training run. TensorBoard can merge
    # multiple event writers under one run in confusing ways, which hides or
    # de-emphasizes the main train/val loss traces.
    log_dir = run_dir / "tensorboard" / "so101_closed_loop"
    step = int(row.get("step") or 0)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        test_id = _safe_id(str(row.get("test_id") or "default"))
        for key in ("success_rate", "grasp_rate", "episodes", "duration_s"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f"closed_loop/{test_id}/{key}", float(value), global_step=step)
                if key == "success_rate":
                    writer.add_scalar(f"{IMPORTANT_CLOSED_LOOP_SUCCESS_RATE_TAG}/{test_id}", float(value), global_step=step)
        for camera_name, image_path in _first_closed_loop_input_grid_paths(report).items():
            image = _read_hwc_image(Path(image_path))
            if image is not None:
                writer.add_image(
                    f"closed_loop/input_{camera_name}_grid",
                    image,
                    global_step=step,
                    dataformats="HWC",
                )
        for camera_name, video in _closed_loop_policy_camera_side_by_side_videos(report).items():
            writer.add_video(
                f"closed_loop/{test_id}/rollout_{camera_name}",
                video,
                global_step=step,
                fps=12,
            )
        for camera_name, video in _training_reference_camera_side_by_side_videos(run_dir).items():
            writer.add_video(
                f"closed_loop/{test_id}/train_reference_{camera_name}",
                video,
                global_step=step,
                fps=12,
            )


def _first_closed_loop_input_grid_paths(report: dict[str, Any]) -> dict[str, str]:
    for episode in report.get("episodes") or []:
        paths = episode.get("input_grid_paths")
        if isinstance(paths, dict) and paths:
            return {str(key): str(value) for key, value in paths.items()}
    return {}


def _closed_loop_policy_camera_videos(report: dict[str, Any]) -> dict[str, Any]:
    videos = {}
    n_action_steps = _closed_loop_n_action_steps(report)
    for camera_feature, camera_label in (
        ("observation.images.camera1", "camera1"),
        ("observation.images.camera2", "camera2"),
    ):
        for episode_index, episode_frames in _closed_loop_policy_camera_frames_by_episode(
            report,
            camera_feature,
            max_frames_per_episode=96,
            n_action_steps=n_action_steps,
        ).items():
            video = _frames_to_tensorboard_video(episode_frames)
            if video is not None:
                videos[f"{camera_label}_episode_{episode_index:03d}"] = video
    return videos


def _closed_loop_policy_camera_side_by_side_videos(report: dict[str, Any]) -> dict[str, Any]:
    n_action_steps = _closed_loop_n_action_steps(report)
    camera1_by_episode = _closed_loop_policy_camera_frames_by_episode(
        report,
        "observation.images.camera1",
        max_frames_per_episode=96,
        n_action_steps=n_action_steps,
    )
    camera2_by_episode = _closed_loop_policy_camera_frames_by_episode(
        report,
        "observation.images.camera2",
        max_frames_per_episode=96,
        n_action_steps=n_action_steps,
    )
    videos = {}
    for episode_index in sorted(set(camera1_by_episode) & set(camera2_by_episode)):
        pairs = list(zip(camera1_by_episode[episode_index], camera2_by_episode[episode_index]))
        video = _side_by_side_tensorboard_video(pairs, left_title="camera1 egocentric", right_title="camera2 wrist")
        if video is not None:
            videos[f"camera1_camera2_episode_{episode_index:03d}"] = video
    return videos


def _closed_loop_n_action_steps(report: dict[str, Any]) -> int:
    config = report.get("policy_rollout_config")
    if isinstance(config, dict):
        value = _int_or_none(config.get("n_action_steps"))
        if value and value > 0:
            return value
    return 15


def _training_reference_camera_videos(run_dir: Path) -> dict[str, Any]:
    dataset_root = _training_reference_dataset_root(run_dir)
    if dataset_root is None:
        return {}
    frames_by_camera = _training_reference_camera_frames_by_episode(
        dataset_root,
        max_episodes=10,
        max_frames_per_episode=96,
    )
    videos = {}
    for camera_label, episodes in frames_by_camera.items():
        for episode_index, frames in episodes.items():
            video = _image_frames_to_tensorboard_video(frames)
            if video is not None:
                videos[f"{camera_label}_episode_{episode_index:03d}"] = video
    return videos


def _training_reference_camera_side_by_side_videos(run_dir: Path) -> dict[str, Any]:
    dataset_root = _training_reference_dataset_root(run_dir)
    if dataset_root is None:
        return {}
    frames_by_camera = _training_reference_camera_frames_by_episode(
        dataset_root,
        max_episodes=10,
        max_frames_per_episode=96,
    )
    camera1_by_episode = frames_by_camera.get("camera1") or {}
    camera2_by_episode = frames_by_camera.get("camera2") or {}
    videos = {}
    for episode_index in sorted(set(camera1_by_episode) & set(camera2_by_episode)):
        pairs = list(zip(camera1_by_episode[episode_index], camera2_by_episode[episode_index]))
        video = _side_by_side_tensorboard_video(pairs, left_title="camera1 egocentric", right_title="camera2 wrist")
        if video is not None:
            videos[f"camera1_camera2_episode_{episode_index:03d}"] = video
    return videos


def _training_reference_dataset_root(run_dir: Path) -> Path | None:
    summary = _read_json(run_dir / "training_run_summary.json") or {}
    dataset_config = summary.get("dataset_config")
    if not isinstance(dataset_config, dict):
        return None
    dataset: dict[str, Any] | None = None
    train_datasets = dataset_config.get("train_datasets")
    if isinstance(train_datasets, list) and train_datasets:
        first = train_datasets[0]
        if isinstance(first, dict):
            dataset = first
    if dataset is None and isinstance(dataset_config.get("train_dataset"), dict):
        dataset = dataset_config["train_dataset"]
    if dataset is None:
        return None
    root_value = dataset.get("root")
    if not root_value:
        return None
    root = Path(str(root_value))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    return root if root.exists() else None


def _training_reference_camera_frames_by_episode(
    dataset_root: Path,
    *,
    max_episodes: int,
    max_frames_per_episode: int,
) -> dict[str, dict[int, list[tuple[Any, str]]]]:
    data_root = dataset_root / "data"
    parquet_paths = sorted(data_root.rglob("*.parquet"))
    if not parquet_paths:
        return {}
    camera_columns = {
        "camera1": "observation.images.camera1",
        "camera2": "observation.images.camera2",
    }
    required_columns = ["episode_index", "frame_index", *camera_columns.values()]
    try:
        import pandas as pd

        dataframes = []
        for path in parquet_paths:
            try:
                dataframes.append(pd.read_parquet(path, columns=required_columns))
            except Exception:
                continue
        if not dataframes:
            return {}
        df = pd.concat(dataframes, ignore_index=True)
    except Exception:
        return {}
    try:
        df = df.sort_values(["episode_index", "frame_index"])
        episode_indices = [int(value) for value in sorted(df["episode_index"].dropna().unique())[:max_episodes]]
    except Exception:
        return {}
    sidecars = _reference_visual_servo_sidecars(dataset_root)
    result: dict[str, dict[int, list[tuple[Any, str, bool, dict[str, Any] | None]]]] = {
        camera: {} for camera in camera_columns
    }
    for episode_index in episode_indices:
        episode_df = df[df["episode_index"] == episode_index].head(max_frames_per_episode)
        for camera_label, column in camera_columns.items():
            frames: list[tuple[Any, str, bool, dict[str, Any] | None]] = []
            for _, row in episode_df.iterrows():
                image = _decode_lerobot_image_cell(row.get(column))
                if image is None:
                    continue
                label = (
                    f"train ep {_format_frame_index(row.get('episode_index'))} | "
                    f"frame {_format_frame_index(row.get('frame_index'))}"
                )
                frames.append((image, label, False, _reference_target_overlay(sidecars.get(camera_label), row, camera_label)))
            if frames:
                result[camera_label][episode_index] = frames
    return {camera: episodes for camera, episodes in result.items() if episodes}


def _reference_visual_servo_sidecars(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "visual_servo_labels" / "camera1_camera2_green_cube.parquet"
    if not path.exists():
        return {}
    try:
        import pandas as pd

        table = pd.read_parquet(path)
        return {
            camera: table.set_index(["episode_index", "frame_index"])
            for camera in ("camera1", "camera2")
            if f"{camera}_dx_norm" in table.columns
        }
    except Exception:
        return {}


def _reference_target_overlay(sidecar: Any, row: Any, camera: str) -> dict[str, Any] | None:
    if sidecar is None:
        return None
    try:
        target = sidecar.loc[(int(row.get("episode_index")), int(row.get("frame_index")))]
        if not bool(target.get(f"{camera}_visible")):
            return None
        return {
            "dx_norm": float(target.get(f"{camera}_dx_norm")),
            "dy_norm": float(target.get(f"{camera}_dy_norm")),
            "label": "gt",
            "color": (255, 210, 0),
        }
    except Exception:
        return None


def _decode_lerobot_image_cell(value: Any) -> Any | None:
    try:
        import numpy as np
        from PIL import Image

        image_bytes = None
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            image_path = value.get("path")
            if image_bytes is None and image_path:
                path = Path(str(image_path))
                if path.exists():
                    return np.asarray(Image.open(path).convert("RGB"))
        elif isinstance(value, (bytes, bytearray, memoryview)):
            image_bytes = bytes(value)
        if image_bytes is None:
            return None
        return np.asarray(Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB"))
    except Exception:
        return None


def _closed_loop_policy_camera_frames_by_episode(
    report: dict[str, Any],
    camera_feature: str,
    *,
    max_frames_per_episode: int,
    n_action_steps: int,
) -> dict[int, list[tuple[Path, str, bool]]]:
    episodes: dict[int, list[tuple[Path, str, bool]]] = {}
    for episode in report.get("episodes") or []:
        episode_index = _int_or_none(episode.get("episode"))
        if episode_index is None:
            continue
        trace_path_value = episode.get("trace_path")
        if not trace_path_value:
            continue
        trace_path = Path(str(trace_path_value))
        if not trace_path.exists():
            continue
        frames: list[tuple[Path, str, bool, dict[str, Any] | None]] = []
        for row in _read_jsonl(trace_path):
            row_episode = row.get("episode", episode_index)
            row_frame = row.get("global_step", row.get("primitive_step", len(frames)))
            mapping = row.get("image_feature_mapping") or {}
            camera_name = mapping.get(camera_feature)
            if not camera_name:
                continue
            media = row.get("media") or {}
            image_path = (media.get("policy_input_images") or {}).get(camera_name)
            if not image_path:
                continue
            path = Path(str(image_path))
            if path.exists():
                is_inference_frame = _is_closed_loop_inference_frame(row_frame, n_action_steps)
                frames.append(
                    (
                        path,
                        _closed_loop_frame_label(row_episode, row_frame, row.get("prompt")),
                        is_inference_frame,
                        _closed_loop_target_overlay(row, camera_feature),
                    )
                )
            if len(frames) >= max_frames_per_episode:
                break
        if frames:
            episodes[episode_index] = frames
    return episodes


def _closed_loop_target_overlay(row: dict[str, Any], camera_feature: str) -> dict[str, Any] | None:
    camera = "camera1" if camera_feature.endswith("camera1") else "camera2"
    policy_output = row.get("policy_output")
    if isinstance(policy_output, dict):
        target = _closed_loop_target_overlay(policy_output, camera_feature)
        if target is not None:
            return target
    for key in ("visual_servo_prediction", "visual_servo", "target_prediction"):
        value = row.get(key)
        if isinstance(value, dict):
            source = value.get(camera) if isinstance(value.get(camera), dict) else value
            target = _target_overlay_from_mapping(source, camera=camera, label="pred", color=(0, 220, 255))
            if target is not None:
                return target
    return _target_overlay_from_mapping(row, camera=camera, label="pred", color=(0, 220, 255))


def _target_overlay_from_mapping(value: Any, *, camera: str, label: str, color: tuple[int, int, int]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    dx = value.get(f"{camera}_dx_norm", value.get("dx_norm"))
    dy = value.get(f"{camera}_dy_norm", value.get("dy_norm"))
    if dx is None or dy is None:
        return None
    visible = value.get(f"{camera}_visible", value.get("visible", True))
    if not bool(visible):
        return None
    try:
        return {"dx_norm": float(dx), "dy_norm": float(dy), "label": label, "color": color}
    except (TypeError, ValueError):
        return None


def _is_closed_loop_inference_frame(frame: Any, n_action_steps: int) -> bool:
    frame_index = _int_or_none(frame)
    if frame_index is None or n_action_steps <= 0:
        return False
    return frame_index % n_action_steps == 0


def _closed_loop_policy_camera_frames(
    report: dict[str, Any],
    camera_feature: str,
    *,
    max_frames: int,
) -> list[tuple[Path, str]]:
    frames: list[tuple[Path, str]] = []
    for episode in report.get("episodes") or []:
        episode_index = episode.get("episode")
        trace_path_value = episode.get("trace_path")
        if not trace_path_value:
            continue
        trace_path = Path(str(trace_path_value))
        if not trace_path.exists():
            continue
        for row in _read_jsonl(trace_path):
            row_episode = row.get("episode", episode_index)
            row_frame = row.get("global_step", row.get("primitive_step", len(frames)))
            mapping = row.get("image_feature_mapping") or {}
            camera_name = mapping.get(camera_feature)
            if not camera_name:
                continue
            media = row.get("media") or {}
            image_path = (media.get("policy_input_images") or {}).get(camera_name)
            if not image_path:
                continue
            path = Path(str(image_path))
            if path.exists():
                frames.append((path, _closed_loop_frame_label(row_episode, row_frame, row.get("prompt"))))
            if len(frames) >= max_frames:
                return frames
    return frames


def _closed_loop_frame_label(episode: Any, frame: Any, prompt: Any = None) -> str:
    label = f"ep {_format_frame_index(episode)} | frame {_format_frame_index(frame)}"
    prompt_label = _short_closed_loop_prompt(prompt)
    if prompt_label:
        label = f"{label}\nprompt: {prompt_label}"
    return label


def _short_closed_loop_prompt(prompt: Any, *, max_chars: int = 44) -> str:
    if not isinstance(prompt, str):
        return ""
    normalized = " ".join(prompt.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _format_frame_index(value: Any) -> str:
    try:
        return f"{int(value):03d}"
    except (TypeError, ValueError):
        return "???"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frames_to_tensorboard_video(frames_with_labels: list[tuple[Any, ...]]) -> Any | None:
    try:
        import numpy as np
        import torch

        frames = []
        for frame_item in frames_with_labels:
            path, label, is_inference_frame, target = _unpack_video_frame_item(frame_item)
            image = _read_hwc_image(path)
            if image is None:
                continue
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[2] > 3:
                array = array[:, :, :3]
            array = _overlay_closed_loop_frame_label(array, label, inference_frame=is_inference_frame, target=target)
            frames.append(array.astype("uint8", copy=False))
        if not frames:
            return None
        stacked = np.stack(frames, axis=0)  # T,H,W,C
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0)
    except Exception:
        return None


def _image_frames_to_tensorboard_video(frames_with_labels: list[tuple[Any, ...]]) -> Any | None:
    try:
        import numpy as np
        import torch

        frames = []
        for frame_item in frames_with_labels:
            image, label, is_inference_frame, target = _unpack_video_frame_item(frame_item)
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[2] > 3:
                array = array[:, :, :3]
            array = _overlay_closed_loop_frame_label(array, label, inference_frame=is_inference_frame, target=target)
            frames.append(array.astype("uint8", copy=False))
        if not frames:
            return None
        stacked = np.stack(frames, axis=0)  # T,H,W,C
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0)
    except Exception:
        return None


def _side_by_side_tensorboard_video(
    frame_pairs: list[tuple[tuple[Any, ...], tuple[Any, ...]]],
    *,
    left_title: str,
    right_title: str,
) -> Any | None:
    try:
        import numpy as np
        import torch

        frames = []
        for left_item, right_item in frame_pairs:
            left = _video_frame_item_to_array(left_item, title=left_title)
            right = _video_frame_item_to_array(right_item, title=right_title)
            if left is None or right is None:
                continue
            height = min(left.shape[0], right.shape[0])
            width = min(left.shape[1], right.shape[1])
            left = left[:height, :width, :3]
            right = right[:height, :width, :3]
            gutter = np.zeros((height, 6, 3), dtype=np.uint8)
            frames.append(np.concatenate([left, gutter, right], axis=1).astype("uint8", copy=False))
        if not frames:
            return None
        stacked = np.stack(frames, axis=0)  # T,H,W,C
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0)
    except Exception:
        return None


def _video_frame_item_to_array(frame_item: tuple[Any, ...], *, title: str) -> Any | None:
    try:
        import numpy as np

        source, label, is_inference_frame, target = _unpack_video_frame_item(frame_item)
        if isinstance(source, Path):
            image = _read_hwc_image(source)
        else:
            image = source
        if image is None:
            return None
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.shape[2] > 3:
            array = array[:, :, :3]
        return _overlay_closed_loop_frame_label(
            array,
            f"{title}\n{label}",
            inference_frame=is_inference_frame,
            target=target,
        ).astype("uint8", copy=False)
    except Exception:
        return None


def _unpack_video_frame_item(frame_item: tuple[Any, ...]) -> tuple[Any, str, bool, dict[str, Any] | None]:
    if len(frame_item) >= 3:
        target = frame_item[3] if len(frame_item) >= 4 and isinstance(frame_item[3], dict) else None
        return frame_item[0], str(frame_item[1]), bool(frame_item[2]), target
    return frame_item[0], str(frame_item[1]), False, None


def _overlay_closed_loop_frame_label(
    array: Any,
    label: str,
    *,
    inference_frame: bool = False,
    target: dict[str, Any] | None = None,
) -> Any:
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(np.asarray(array).astype("uint8", copy=False)).convert("RGB")
        draw = ImageDraw.Draw(image)
        border_width = max(3, image.width // 64) if inference_frame else 0
        if inference_frame:
            for offset in range(border_width):
                draw.rectangle(
                    (offset, offset, image.width - 1 - offset, image.height - 1 - offset),
                    outline=(0, 255, 0),
                )
        font = ImageFont.load_default()
        _draw_target_overlay(draw, image.width, image.height, target, font=font)
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=1)
        pad_x = 3
        pad_y = 2
        width = min(image.width, bbox[2] - bbox[0] + 2 * pad_x)
        height = min(image.height, bbox[3] - bbox[1] + 2 * pad_y)
        draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
        draw.multiline_text((pad_x, pad_y), label, fill=(255, 255, 255), font=font, spacing=1)
        if inference_frame:
            for offset in range(border_width):
                draw.rectangle(
                    (offset, offset, image.width - 1 - offset, image.height - 1 - offset),
                    outline=(0, 255, 0),
                )
        return np.asarray(image)
    except Exception:
        return array


def _draw_target_overlay(draw: Any, width: int, height: int, target: dict[str, Any] | None, *, font: Any) -> None:
    if not target:
        return
    try:
        dx = float(target["dx_norm"])
        dy = float(target["dy_norm"])
        x = int(round(((width - 1) * 0.5) + dx * ((width - 1) * 0.5)))
        y = int(round(((height - 1) * 0.5) + dy * ((height - 1) * 0.5)))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        color = tuple(target.get("color") or (255, 210, 0))
        label = str(target.get("label") or "target")
        radius = max(5, min(width, height) // 28)
        draw.line((x - radius, y, x + radius, y), fill=color, width=2)
        draw.line((x, y - radius, x, y + radius), fill=color, width=2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.text((min(width - 1, x + radius + 2), max(0, y - radius - 2)), label, fill=color, font=font)
    except Exception:
        return


def _read_hwc_image(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        import imageio.v2 as imageio

        return imageio.imread(path)
    except Exception:
        try:
            import numpy as np
            from PIL import Image

            return np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            return None


def _update_loss_summary(run_dir: Path, checkpoint: str | None) -> None:
    summary_path = run_dir / "metrics" / "loss_summary.json"
    summary = _read_json(summary_path) or {}
    checkpoints = _checkpoint_names(run_dir)
    train_rows = _read_jsonl(run_dir / "metrics" / "training_metrics.jsonl")
    val_rows = _read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl")
    if train_rows:
        summary["latest_train_loss"] = train_rows[-1].get("loss")
        summary["latest_train_step"] = train_rows[-1].get("step")
        if "epoch" in train_rows[-1]:
            summary["latest_train_epoch"] = train_rows[-1].get("epoch")
    if val_rows:
        summary["latest_val_loss"] = val_rows[-1].get("loss")
        summary["latest_val_step"] = val_rows[-1].get("step")
        summary["latest_val_checkpoint"] = val_rows[-1].get("checkpoint")
    closed_loop_rows = _read_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl")
    if closed_loop_rows:
        summary["latest_closed_loop_success_rate"] = closed_loop_rows[-1].get("success_rate")
        summary["latest_closed_loop_grasp_rate"] = closed_loop_rows[-1].get("grasp_rate")
        summary["latest_closed_loop_checkpoint"] = closed_loop_rows[-1].get("checkpoint")
    monitor_rows = _read_jsonl(run_dir / "metrics" / "monitor_events.jsonl")
    validation_status_rows = [
        row
        for row in monitor_rows
        if row.get("kind")
        in {"validation_deferred", "validation_error", "validation_start", "validation_done", "validation_done_cpu"}
    ]
    if validation_status_rows:
        latest_validation_status = validation_status_rows[-1]
        summary["latest_validation_status"] = latest_validation_status.get("kind")
        summary["latest_validation_status_checkpoint"] = latest_validation_status.get("checkpoint")
        summary["latest_validation_status_detail"] = latest_validation_status.get("detail")
    stop_rows = [row for row in monitor_rows if row.get("kind") == "training_stop_overfit"]
    if stop_rows:
        latest_stop = stop_rows[-1]
        summary["status"] = "stopped_overfit"
        summary["stopped_reason"] = latest_stop.get("detail")
        summary["stopped_checkpoint"] = latest_stop.get("checkpoint")
        summary["overfit_decision"] = latest_stop.get("overfit_decision")
    summary["latest_checkpoint"] = checkpoint
    summary["checkpoint_count"] = len(checkpoints)
    summary["last_monitor_check_local"] = _now_local()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_event(run_dir: Path, event: dict[str, Any]) -> None:
    _append_jsonl(
        run_dir / "metrics" / "monitor_events.jsonl",
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checked_at_local": _now_local(),
            **event,
        },
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint_to_step(checkpoint: str) -> int | None:
    try:
        return int(checkpoint)
    except ValueError:
        return None


def _now_local() -> str:
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value)).strip("_") or "default"


if __name__ == "__main__":
    main()
