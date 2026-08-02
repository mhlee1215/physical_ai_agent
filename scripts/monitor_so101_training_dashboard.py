#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from physical_ai_agent.so101_resolution_contract import (
    require_lerobot_dataset_256,
    require_so101_image_resolution,
    resolve_lerobot_root_for_start_report,
)
from physical_ai_agent.so101_smolvla_pipeline import (
    SO101TrainingSchedule,
    detect_overfit_stop,
    should_run_closed_loop,
)
from physical_ai_agent.so101_training_config_schema import resolve_so101_training_config_defaults

IMPORTANT_CLOSED_LOOP_SUCCESS_RATE_TAG = "important/closed_loop_success_rate"
IMPORTANT_VAL_LOSS_TAG = "important/val_loss"
IMPORTANT_VAL_POSTPROCESSED_ACTION_RMSE_TAG = "important/val_postprocessed_action_rmse"


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
    parser.add_argument(
        "--checkpoint-name",
        help="Evaluate exactly one checkpoint directory instead of scanning retained checkpoint aliases.",
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
    parser.add_argument("--closed-loop-every-epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=138)
    parser.add_argument("--closed-loop-episodes", type=int, default=10)
    parser.add_argument("--closed-loop-steps", type=int, default=160)
    parser.add_argument("--closed-loop-seed", type=int, default=98100)
    parser.add_argument("--closed-loop-test-id", default="default")
    parser.add_argument(
        "--closed-loop-success-metric",
        default="env_success",
        choices=["env_success", "grasped", "tcp_to_object_below_threshold"],
    )
    parser.add_argument("--closed-loop-success-threshold", type=float, default=0.06)
    parser.add_argument("--closed-loop-start-contract", default="default_reset")
    parser.add_argument("--closed-loop-start-report-path", type=Path)
    parser.add_argument(
        "--closed-loop-episode-indices",
        help="Comma-separated report episode indices used in deterministic order.",
    )
    parser.add_argument(
        "--closed-loop-phase-contract-json",
        help="JSON primitive/chain verifier and handoff contract forwarded to the evaluator.",
    )
    parser.add_argument(
        "--closed-loop-observation-renderer-json",
        help="JSON policy-camera renderer contract forwarded to the closed-loop evaluator.",
    )
    parser.add_argument("--closed-loop-env-id", default="MuJoCoPickLift-v1")
    parser.add_argument("--closed-loop-env-object-color")
    parser.add_argument(
        "--closed-loop-env-config-json",
        help="JSON environment contract forwarded unchanged to the closed-loop evaluator.",
    )
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
        choices=[
            "processor",
            "legacy",
            "processor_dataset_clamp",
            "processor_gripper_snap",
            "processor_delta_q",
            "visual_servo_delta_q",
            "visual_servo_gt_delta_q",
        ],
        default="processor",
        help="Action conversion mode forwarded to the closed-loop policy evaluator.",
    )
    parser.add_argument("--closed-loop-eval-skill-mode", default="picklift")
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
    parser.add_argument("--closed-loop-valid-mask-requery-confirmations", type=int)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--closed-loop-temporal-ensemble", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--closed-loop-temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument(
        "--closed-loop-action-rmse-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After each Qwen closed-loop test, run a one-episode n_action_steps sweep against "
            "the matching teacher episode and attach the RMSE plot to TensorBoard."
        ),
    )
    parser.add_argument(
        "--closed-loop-action-rmse-sweep-n-action-steps",
        default="1,3,5,10,15,30,40,50",
        help="Comma-separated n_action_steps values for the closed-loop teacher-action RMSE sweep.",
    )
    parser.add_argument(
        "--closed-loop-action-rmse-sweep-y-axis-max",
        type=float,
        help="Fixed upper bound for the action RMSE sweep plot y-axis.",
    )
    parser.add_argument(
        "--closed-loop-action-rmse-sweep-render-policy-inference-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render sweep camera inputs only when the policy is queried.",
    )
    parser.add_argument(
        "--closed-loop-action-rmse-sweep-timeline-mode",
        choices=["start_dataset", "phase_chain"],
        default="start_dataset",
        help="Use one start dataset or concatenate every configured phase into one diagnostic timeline.",
    )
    parser.add_argument("--closed-loop-action-rmse-sweep-test-cases", default="")
    parser.add_argument("--closed-loop-action-rmse-sweep-phase-contract-test-case-id")
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
    require_lerobot_dataset_256(args.dataset_root, context="monitor validation dataset")
    require_so101_image_resolution(
        height=int(args.closed_loop_height),
        width=int(args.closed_loop_width),
        context="monitor closed-loop policy inputs",
    )
    require_so101_image_resolution(
        height=int(args.loop_artifact_height),
        width=int(args.loop_artifact_width),
        context="monitor loop artifact media",
    )
    if args.closed_loop_start_report_path is not None:
        require_lerobot_dataset_256(
            resolve_lerobot_root_for_start_report(
                args.closed_loop_start_report_path,
                repo_root=args.repo_root,
            ),
            context="monitor closed-loop start dataset",
        )
        args.closed_loop_episodes = _clamp_closed_loop_episodes_to_start_report(args)

    run_dir = args.run_dir.resolve()
    _refresh_closed_loop_runtime_from_config(args, run_dir)
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
    if args.checkpoint_name:
        requested = str(args.checkpoint_name)
        if requested not in checkpoints:
            raise FileNotFoundError(f"requested checkpoint not found under {checkpoint_root}: {requested}")
        checkpoints = [requested]
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
    train_process = _process_status(args.train_pid_file)
    for checkpoint in checkpoints:
        if checkpoint == "last":
            continue
        checkpoint_step = _checkpoint_step(run_dir, checkpoint)
        if checkpoint_step is not None and checkpoint_step < args.min_validation_step:
            continue
        policy_path = checkpoint_root / checkpoint / "pretrained_model"
        if not policy_path.exists():
            continue
        validation_recorded = args.skip_validation or _validation_already_recorded(run_dir, checkpoint)
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


def _clamp_closed_loop_episodes_to_start_report(args: argparse.Namespace) -> int:
    report_path = Path(args.closed_loop_start_report_path)
    if not report_path.is_absolute():
        report_path = args.repo_root / report_path
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return int(args.closed_loop_episodes)
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return int(args.closed_loop_episodes)
    return min(int(args.closed_loop_episodes), len(episodes))


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
    record_rollout_gif = bool(
        args.closed_loop_record_rollout_gif
        or getattr(args, "record_loop_artifacts", False)
        or getattr(args, "render_loop_media", False)
    )
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
        "--record-rollout-gif" if record_rollout_gif else "--no-record-rollout-gif",
        "--subgoal-chain-mode",
        args.closed_loop_subgoal_chain_mode,
        "--fixed-subgoal-chunks",
        str(args.closed_loop_fixed_subgoal_chunks),
        "--valid-mask-threshold",
        str(args.closed_loop_valid_mask_threshold),
        "--valid-mask-consecutive",
        str(args.closed_loop_valid_mask_consecutive),
    ]
    if args.closed_loop_valid_mask_requery_confirmations is not None:
        cmd.extend(
            [
                "--valid-mask-requery-confirmations",
                str(args.closed_loop_valid_mask_requery_confirmations),
            ]
        )
    cmd.extend(
        [
            "--temporal-ensemble" if args.closed_loop_temporal_ensemble else "--no-temporal-ensemble",
            "--temporal-ensemble-decay",
            str(args.closed_loop_temporal_ensemble_decay),
        ]
    )
    if args.closed_loop_start_report_path is not None:
        cmd.extend(["--start-report-path", str(args.closed_loop_start_report_path)])
    if args.closed_loop_episode_indices:
        cmd.extend(["--episode-indices", str(args.closed_loop_episode_indices)])
    if args.closed_loop_phase_contract_json:
        cmd.extend(["--phase-contract-json", str(args.closed_loop_phase_contract_json)])
    if args.closed_loop_observation_renderer_json:
        cmd.extend(
            [
                "--observation-renderer-json",
                _action_rmse_sweep_observation_renderer_json(
                    args.closed_loop_observation_renderer_json,
                    render_policy_inference_only=bool(
                        args.closed_loop_action_rmse_sweep_render_policy_inference_only
                    ),
                ),
            ]
        )
    env_object_color = args.closed_loop_env_object_color or args.qwen_env_object_color
    if env_object_color:
        cmd.extend(["--env-object-color", str(env_object_color)])
    if args.closed_loop_env_config_json:
        cmd.extend(["--env-config-json", args.closed_loop_env_config_json])
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
    _attach_action_rmse_sweep(
        args=args,
        checkpoint=checkpoint,
        policy_path=policy_path,
        valid_mask_checkpoint=valid_mask_checkpoint,
        closed_loop_test_id=closed_loop_test_id,
        output_dir=output_dir,
        report=report,
    )
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
    _attach_action_rmse_sweep(
        args=args,
        checkpoint=checkpoint,
        policy_path=policy_path,
        valid_mask_checkpoint=valid_mask_checkpoint,
        closed_loop_test_id=closed_loop_test_id,
        output_dir=output_dir,
        report=report,
    )
    return report


def _apply_closed_loop_success_metric(report: dict[str, Any], args: argparse.Namespace) -> None:
    metric, metric_source = _configured_closed_loop_success_metric(report, args)
    report["success_metric_source"] = metric_source
    if metric == "env_success":
        return
    if metric == "grasped":
        episodes = report.get("episodes") or []
        rows = []
        passed = 0
        for index, episode in enumerate(episodes):
            final_is_grasped = _episode_final_is_grasped(episode)
            episode_passed = final_is_grasped is not None and final_is_grasped > 0.5
            rows.append(
                {
                    "episode": index,
                    "final_is_grasped": final_is_grasped,
                    "threshold": 0.5,
                    "passed": bool(episode_passed),
                }
            )
            passed += int(bool(episode_passed))
        report["task_success_rate"] = report.get("success_rate")
        report["success_metric"] = metric
        report["success_rate"] = (passed / len(rows)) if rows else None
        report["grasp_rate"] = report["success_rate"]
        report["grasped_episodes"] = rows
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


def _configured_closed_loop_success_metric(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str]:
    fallback = str(getattr(args, "closed_loop_success_metric", "env_success") or "env_success")
    run_dir_value = getattr(args, "run_dir", None)
    if run_dir_value is None:
        return fallback, "command"
    try:
        config = _resolved_training_config_for_run(
            Path(run_dir_value),
            repo_root=Path(getattr(args, "repo_root", Path.cwd())),
        )
        if config is None:
            return fallback, "command"
        closed_loop = ((config.get("training_config") or {}).get("closed_loop") or config.get("closed_loop") or {})
        test_id = str(report.get("closed_loop_test_id") or getattr(args, "closed_loop_test_id", "default"))
        for test_case in closed_loop.get("test_cases") or []:
            if str(test_case.get("id")) == test_id and test_case.get("success_metric"):
                return str(test_case["success_metric"]), "training_config"
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return fallback, "command"
    return fallback, "command"


def _resolved_training_config_for_run(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    summary_path = Path(run_dir) / "training_run_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path_value = (summary.get("dataset_config") or {}).get("config_path")
    if not config_path_value:
        return None
    config_path = Path(str(config_path_value))
    if not config_path.is_absolute():
        config_path = Path(repo_root) / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return resolve_so101_training_config_defaults(
        config,
        path=config_path,
        repo_root=Path(repo_root),
    )


def _refresh_closed_loop_runtime_from_config(args: argparse.Namespace, run_dir: Path) -> None:
    config = _resolved_training_config_for_run(
        run_dir,
        repo_root=Path(getattr(args, "repo_root", Path.cwd())),
    )
    if config is None:
        return
    closed_loop = (
        (config.get("training_config") or {}).get("closed_loop")
        or config.get("closed_loop")
        or {}
    )
    renderer = closed_loop.get("observation_renderer")
    if isinstance(renderer, dict):
        args.closed_loop_observation_renderer_json = json.dumps(renderer, sort_keys=True)
    action_rmse_sweep = closed_loop.get("action_rmse_sweep")
    if isinstance(action_rmse_sweep, dict):
        if "render_policy_inference_only" in action_rmse_sweep:
            args.closed_loop_action_rmse_sweep_render_policy_inference_only = bool(
                action_rmse_sweep["render_policy_inference_only"]
            )
        n_action_steps = action_rmse_sweep.get("n_action_steps")
        if isinstance(n_action_steps, list) and n_action_steps:
            args.closed_loop_action_rmse_sweep_n_action_steps = ",".join(
                str(int(value)) for value in n_action_steps
            )
        if action_rmse_sweep.get("y_axis_max") is not None:
            args.closed_loop_action_rmse_sweep_y_axis_max = float(
                action_rmse_sweep["y_axis_max"]
            )
        if action_rmse_sweep.get("timeline_mode") is not None:
            args.closed_loop_action_rmse_sweep_timeline_mode = str(
                action_rmse_sweep["timeline_mode"]
            )
        rmse_test_cases = action_rmse_sweep.get("test_cases")
        if isinstance(rmse_test_cases, list):
            args.closed_loop_action_rmse_sweep_test_cases = ",".join(
                str(value) for value in rmse_test_cases
            )
        source_test_case_id = action_rmse_sweep.get("phase_contract_test_case_id")
        if source_test_case_id is not None:
            args.closed_loop_action_rmse_sweep_phase_contract_test_case_id = str(
                source_test_case_id
            )
            source_test_case = next(
                (
                    case
                    for case in closed_loop.get("test_cases") or []
                    if str(case.get("id")) == str(source_test_case_id)
                ),
                None,
            )
            if not isinstance(source_test_case, dict) or not isinstance(
                source_test_case.get("phase_contract"),
                dict,
            ):
                raise ValueError(
                    "action RMSE phase contract source test case is missing: "
                    f"{source_test_case_id}"
                )
            args.closed_loop_action_rmse_sweep_phase_contract_json = json.dumps(
                source_test_case["phase_contract"],
                sort_keys=True,
            )
    if isinstance(closed_loop.get("record_rollout_gif"), bool):
        args.closed_loop_record_rollout_gif = bool(closed_loop["record_rollout_gif"])
    media = closed_loop.get("tensorboard_media")
    render_test_cases = media.get("render_test_cases") if isinstance(media, dict) else None
    if isinstance(render_test_cases, list):
        test_id = str(getattr(args, "closed_loop_test_id", "default") or "default")
        media_enabled = test_id in {str(value) for value in render_test_cases}
        args.record_loop_artifacts = media_enabled
        args.render_loop_media = media_enabled
        args.closed_loop_record_rollout_gif = (
            media_enabled and bool(closed_loop.get("record_rollout_gif"))
        )


def _attach_action_rmse_sweep(
    *,
    args: argparse.Namespace,
    checkpoint: str,
    policy_path: Path,
    valid_mask_checkpoint: Path | None,
    closed_loop_test_id: str,
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    if not bool(getattr(args, "closed_loop_action_rmse_sweep", True)):
        return
    allowed_test_cases = {
        value.strip()
        for value in str(
            getattr(args, "closed_loop_action_rmse_sweep_test_cases", "")
        ).split(",")
        if value.strip()
    }
    if allowed_test_cases and closed_loop_test_id not in allowed_test_cases:
        report["action_rmse_sweep"] = {
            "operation": "so101_action_rmse_sweep",
            "skipped": "test_case_not_allowlisted",
            "closed_loop_test_id": closed_loop_test_id,
        }
        return
    if getattr(args, "closed_loop_start_report_path", None) is None:
        report["action_rmse_sweep_error"] = "missing_closed_loop_start_report_path"
        return
    try:
        sweep = _run_action_rmse_sweep(
            args=args,
            checkpoint=checkpoint,
            policy_path=policy_path,
            valid_mask_checkpoint=valid_mask_checkpoint,
            closed_loop_test_id=closed_loop_test_id,
            output_dir=output_dir,
        )
    except Exception as exc:
        report["action_rmse_sweep_error"] = str(exc)[-1000:]
        return
    report["action_rmse_sweep"] = sweep
    report_path = Path(str(report.get("report_path") or output_dir / "qwen_closed_loop_eval_report.json"))
    try:
        if report_path.exists():
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _run_action_rmse_sweep(
    *,
    args: argparse.Namespace,
    checkpoint: str,
    policy_path: Path,
    valid_mask_checkpoint: Path | None,
    closed_loop_test_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    n_values = _parse_int_csv(str(getattr(args, "closed_loop_action_rmse_sweep_n_action_steps", "")))
    if not n_values:
        return {"operation": "so101_action_rmse_sweep", "skipped": "empty_n_action_steps"}
    y_axis_max = getattr(args, "closed_loop_action_rmse_sweep_y_axis_max", None)
    if y_axis_max is None or not math.isfinite(float(y_axis_max)) or float(y_axis_max) <= 0.0:
        raise ValueError(
            "closed-loop action RMSE sweep requires a positive configured y-axis maximum"
        )
    y_axis_max = float(y_axis_max)
    sweep_dir = output_dir / "action_rmse_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    timeline_mode = str(
        getattr(args, "closed_loop_action_rmse_sweep_timeline_mode", "start_dataset")
    )
    timeline_segments = _action_rmse_timeline_segments(args, timeline_mode=timeline_mode)
    rows: list[dict[str, Any]] = []
    series_by_n: dict[int, list[float]] = {}
    teacher_frames = 0
    phase_segments: list[dict[str, Any]] = []
    for n_action_steps in n_values:
        n_dir = sweep_dir / f"nact{int(n_action_steps):02d}"
        combined_series: list[float] = []
        phase_metrics: dict[str, dict[str, Any]] = {}
        boundary_summaries: list[dict[str, Any]] = []
        run_dirs: dict[str, str] = {}
        transition_segments: list[dict[str, Any]] = []
        final_info: dict[str, Any] = {}
        failed_error: str | None = None
        offset = 0
        for segment in timeline_segments:
            segment_id = str(segment["id"])
            segment_dir = n_dir / segment_id if len(timeline_segments) > 1 else n_dir
            segment_args = _action_rmse_segment_args(args, segment)
            cmd = _closed_loop_sweep_command(
                args=segment_args,
                policy_path=policy_path,
                valid_mask_checkpoint=valid_mask_checkpoint,
                output_dir=segment_dir,
                n_action_steps=int(n_action_steps),
            )
            completed = subprocess.run(
                cmd,
                cwd=args.repo_root,
                env=_runtime_env(args),
                text=True,
                capture_output=True,
                check=False,
            )
            stdout_path = segment_dir / "stdout.json"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(
                json.dumps(
                    {
                        "command": cmd,
                        "returncode": completed.returncode,
                        "stderr_tail": (completed.stderr or "")[-4000:],
                        "stdout_tail": (completed.stdout or "")[-4000:],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            run_dirs[segment_id] = str(segment_dir)
            if completed.returncode != 0:
                failed_error = (completed.stderr or completed.stdout or "")[-1000:]
                break
            sweep_report_path = _closed_loop_eval_report_path(
                segment_dir,
                runner=str(getattr(segment_args, "closed_loop_runner", "")),
            )
            sweep_report = json.loads(sweep_report_path.read_text(encoding="utf-8"))
            first_episode = (sweep_report.get("episodes") or [{}])[0]
            trace_path = _first_trace_path(sweep_report)
            episode_index = _first_dataset_episode_index(sweep_report)
            if episode_index is None:
                failed_error = f"{segment_id}: missing_dataset_episode_index"
                break
            dataset_root = Path(segment["dataset_root"])
            if trace_path is not None:
                series, teacher_count = _teacher_action_rmse_series(
                    dataset_root=dataset_root,
                    episode_index=int(episode_index),
                    rollout_jsonl=trace_path,
                )
            else:
                series, teacher_count = _teacher_action_rmse_series_from_records(
                    dataset_root=dataset_root,
                    episode_index=int(episode_index),
                    records=first_episode.get("records") or [],
                )
            padded_series = list(series[:teacher_count])
            padded_series.extend([float("nan")] * max(0, teacher_count - len(padded_series)))
            combined_series.extend(padded_series)
            summary = _summarize_rmse_series(series)
            phase_metrics[segment_id] = {
                **summary,
                "success": bool(first_episode.get("skill_success", sweep_report.get("success_rate"))),
                "teacher_frames": int(teacher_count),
                "compared_frames": int(len(series)),
                "dataset_root": str(dataset_root),
                "run_dir": str(segment_dir),
            }
            boundary_summaries.append(
                _summarize_requery_boundary(
                    first_episode.get("records") or [],
                    n_action_steps=int(n_action_steps),
                )
            )
            transition_segments.append(
                {
                    "id": segment_id,
                    "start_frame": int(offset),
                    "end_frame_exclusive": int(offset + teacher_count),
                    "teacher_frames": int(teacher_count),
                }
            )
            offset += int(teacher_count)
            final_info = first_episode.get("final_info") or first_episode
        if failed_error is not None:
            rows.append(
                {
                    "n_action_steps": int(n_action_steps),
                    "success": False,
                    "error": failed_error,
                    "run_dir": str(n_dir),
                    "phase_run_dirs": run_dirs,
                }
            )
            continue
        teacher_frames = max(teacher_frames, len(combined_series))
        series_by_n[int(n_action_steps)] = combined_series
        summary = _summarize_rmse_series(combined_series)
        boundary = _merge_requery_boundary_summaries(boundary_summaries)
        if not phase_segments:
            phase_segments = transition_segments
        rows.append(
            {
                "n_action_steps": int(n_action_steps),
                "success": bool(phase_metrics) and all(
                    bool(metric["success"]) for metric in phase_metrics.values()
                ),
                "step0": summary["step0"],
                "mean": summary["mean"],
                "max": summary["max"],
                "last": summary["last"],
                "reference_drift_step0": summary["step0"],
                "reference_drift_mean": summary["mean"],
                "reference_drift_max": summary["max"],
                "reference_drift_last": summary["last"],
                **boundary,
                "phase_metrics": phase_metrics,
                "approach_mean": (phase_metrics.get("approach") or {}).get("mean"),
                "alignment_mean": (phase_metrics.get("alignment") or {}).get("mean"),
                "grip_lift_mean": (phase_metrics.get("grip_lift") or {}).get("mean"),
                "tcp": final_info.get("tcp_to_obj_dist"),
                "lift": final_info.get("lift_height"),
                "run_dir": str(n_dir),
                "phase_run_dirs": run_dirs,
            }
        )
    csv_path = sweep_dir / "action_rmse_sweep.csv"
    _write_action_rmse_sweep_csv(csv_path, rows)
    payload: dict[str, Any] = {
        "operation": "so101_action_rmse_sweep",
        "checkpoint": checkpoint,
        "closed_loop_test_id": closed_loop_test_id,
        "timeline_mode": timeline_mode,
        "phase_segments": phase_segments,
        "dataset_roots": {
            str(segment["id"]): str(segment["dataset_root"])
            for segment in timeline_segments
        },
        "n_action_steps": n_values,
        "y_axis_max": y_axis_max,
        "rows": rows,
        "csv_path": str(csv_path),
    }
    plot_path = sweep_dir / "action_rmse_sweep.png"
    if series_by_n:
        _plot_action_rmse_sweep(
            plot_path=plot_path,
            series_by_n=series_by_n,
            rows=rows,
            teacher_frames=teacher_frames,
            phase_segments=phase_segments,
            y_axis_max=y_axis_max,
        )
        payload["plot_path"] = str(plot_path)
    json_path = sweep_dir / "action_rmse_sweep.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["json_path"] = str(json_path)
    return payload


def _action_rmse_timeline_segments(
    args: argparse.Namespace,
    *,
    timeline_mode: str,
) -> list[dict[str, Any]]:
    if timeline_mode != "phase_chain":
        return [
            {
                "id": "task",
                "dataset_root": str(_closed_loop_sweep_dataset_root(args)),
            }
        ]
    raw_contract = getattr(
        args,
        "closed_loop_action_rmse_sweep_phase_contract_json",
        None,
    ) or getattr(args, "closed_loop_phase_contract_json", None)
    if not raw_contract:
        raise ValueError("phase_chain RMSE timeline requires a phase contract")
    contract = json.loads(str(raw_contract))
    phases = contract.get("phases")
    if contract.get("mode") != "chain" or not isinstance(phases, list) or len(phases) != 3:
        raise ValueError("phase_chain RMSE timeline requires one three-phase chain contract")
    result: list[dict[str, Any]] = []
    for phase in phases:
        report_path = Path(str(phase["reference_report_path"]))
        result.append(
            {
                "id": str(phase["id"]),
                "prompt": str(phase["prompt"]),
                "steps": int(phase["max_steps"]),
                "start_report_path": str(report_path),
                "dataset_root": str(
                    resolve_lerobot_root_for_start_report(
                        report_path,
                        repo_root=args.repo_root,
                    )
                ),
                "phase_contract": {
                    "mode": "primitive",
                    "handoff_mode": "continuous",
                    "phases": [phase],
                },
            }
        )
    return result


def _action_rmse_segment_args(
    args: argparse.Namespace,
    segment: dict[str, Any],
) -> argparse.Namespace:
    if "phase_contract" not in segment:
        return args
    segment_args = argparse.Namespace(**vars(args))
    segment_args.closed_loop_start_report_path = Path(str(segment["start_report_path"]))
    segment_args.closed_loop_phase_contract_json = json.dumps(
        segment["phase_contract"],
        sort_keys=True,
    )
    segment_args.closed_loop_steps = int(segment["steps"])
    segment_args.closed_loop_task_prompt = str(segment["prompt"])
    return segment_args


def _closed_loop_eval_report_path(output_dir: Path, *, runner: str) -> Path:
    names = (
        ("qwen_closed_loop_eval_report.json", "so101_picklift_smolvla_eval_report.json")
        if runner == "qwen_chain"
        else ("so101_picklift_smolvla_eval_report.json", "qwen_closed_loop_eval_report.json")
    )
    for name in names:
        path = output_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"closed-loop eval report not found under {output_dir}")


def _closed_loop_sweep_command(
    *,
    args: argparse.Namespace,
    policy_path: Path,
    valid_mask_checkpoint: Path | None,
    output_dir: Path,
    n_action_steps: int,
) -> list[str]:
    runner = str(getattr(args, "closed_loop_runner", "") or "picklift")
    if runner == "qwen_chain":
        if valid_mask_checkpoint is None:
            raise RuntimeError("qwen_chain RMSE sweep requires a valid-mask checkpoint")
        return _qwen_closed_loop_sweep_command(
            args=args,
            policy_path=policy_path,
            valid_mask_checkpoint=valid_mask_checkpoint,
            output_dir=output_dir,
            n_action_steps=n_action_steps,
        )
    if runner in {"picklift", "auto"}:
        return _picklift_closed_loop_sweep_command(
            args=args,
            policy_path=policy_path,
            valid_mask_checkpoint=valid_mask_checkpoint,
            output_dir=output_dir,
            n_action_steps=n_action_steps,
        )
    raise RuntimeError(f"unsupported closed-loop runner for RMSE sweep: {runner}")


def _qwen_closed_loop_sweep_command(
    *,
    args: argparse.Namespace,
    policy_path: Path,
    valid_mask_checkpoint: Path,
    output_dir: Path,
    n_action_steps: int,
) -> list[str]:
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
        "1",
        "--seed",
        str(args.closed_loop_seed),
        "--start-contract",
        args.closed_loop_start_contract,
        "--device",
        args.policy_device,
        "--max-steps-per-primitive",
        str(args.closed_loop_steps),
        "--policy-n-action-steps",
        str(n_action_steps),
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
        "--record-loop-artifacts",
        "--no-render-loop-media",
    ]
    if args.closed_loop_start_report_path is not None:
        cmd.extend(["--start-report-path", str(args.closed_loop_start_report_path)])
    if args.closed_loop_episode_indices:
        cmd.extend(["--episode-indices", str(args.closed_loop_episode_indices)])
    if args.closed_loop_phase_contract_json:
        cmd.extend(["--phase-contract-json", str(args.closed_loop_phase_contract_json)])
    if args.qwen_plan_json:
        cmd.extend(["--qwen-plan-json", str(args.qwen_plan_json)])
    elif args.qwen_response_json:
        cmd.extend(["--qwen-response-json", str(args.qwen_response_json)])
    elif args.qwen_base_url:
        cmd.extend(["--qwen-base-url", args.qwen_base_url])
    else:
        raise RuntimeError("RMSE sweep requires --qwen-plan-json, --qwen-response-json, or --qwen-base-url")
    if args.qwen_api_key:
        cmd.extend(["--qwen-api-key", args.qwen_api_key])
    if getattr(args, "closed_loop_precondition_plan_json", None):
        cmd.extend(["--precondition-plan-json", str(args.closed_loop_precondition_plan_json)])
    if not args.local_files_only:
        cmd.append("--allow-download")
    return cmd


def _action_rmse_sweep_observation_renderer_json(
    renderer_json: str,
    *,
    render_policy_inference_only: bool,
) -> str:
    renderer = json.loads(renderer_json)
    if render_policy_inference_only:
        renderer["render_policy_inference_only"] = True
    return json.dumps(renderer, sort_keys=True)


def _picklift_closed_loop_sweep_command(
    *,
    args: argparse.Namespace,
    policy_path: Path,
    valid_mask_checkpoint: Path | None,
    output_dir: Path,
    n_action_steps: int,
) -> list[str]:
    cmd = [
        args.python,
        str(args.repo_root / "scripts" / "evaluate_so101_picklift_smolvla_policy.py"),
        "--policy-path",
        str(policy_path),
        "--output-dir",
        str(output_dir),
        "--episodes",
        "1",
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
        str(n_action_steps),
        "--policy-num-steps",
        str(args.policy_num_steps),
        "--sample-input-grid-count",
        str(args.closed_loop_input_grid_count),
        "--torch-seed",
        str(args.torch_seed),
        "--eval-skill-mode",
        args.closed_loop_eval_skill_mode,
        "--no-record-rollout-gif",
        "--subgoal-chain-mode",
        args.closed_loop_subgoal_chain_mode,
        "--fixed-subgoal-chunks",
        str(args.closed_loop_fixed_subgoal_chunks),
        "--valid-mask-threshold",
        str(args.closed_loop_valid_mask_threshold),
        "--valid-mask-consecutive",
        str(args.closed_loop_valid_mask_consecutive),
    ]
    if args.closed_loop_valid_mask_requery_confirmations is not None:
        cmd.extend(
            [
                "--valid-mask-requery-confirmations",
                str(args.closed_loop_valid_mask_requery_confirmations),
            ]
        )
    cmd.extend(
        [
            "--temporal-ensemble" if args.closed_loop_temporal_ensemble else "--no-temporal-ensemble",
            "--temporal-ensemble-decay",
            str(args.closed_loop_temporal_ensemble_decay),
        ]
    )
    if args.closed_loop_start_report_path is not None:
        cmd.extend(["--start-report-path", str(args.closed_loop_start_report_path)])
    if args.closed_loop_episode_indices:
        first_episode_index = str(args.closed_loop_episode_indices).split(",", 1)[0].strip()
        if first_episode_index:
            cmd.extend(["--episode-indices", first_episode_index])
    if args.closed_loop_phase_contract_json:
        cmd.extend(["--phase-contract-json", str(args.closed_loop_phase_contract_json)])
    if args.closed_loop_observation_renderer_json:
        cmd.extend(["--observation-renderer-json", args.closed_loop_observation_renderer_json])
    env_object_color = args.closed_loop_env_object_color or args.qwen_env_object_color
    if env_object_color:
        cmd.extend(["--env-object-color", str(env_object_color)])
    if args.closed_loop_env_config_json:
        cmd.extend(["--env-config-json", args.closed_loop_env_config_json])
    if args.closed_loop_subgoal_sequence:
        cmd.extend(["--subgoal-sequence", args.closed_loop_subgoal_sequence])
    if valid_mask_checkpoint:
        cmd.extend(["--valid-mask-checkpoint", str(valid_mask_checkpoint)])
    if args.closed_loop_task_prompt:
        cmd.extend(["--task-prompt", args.closed_loop_task_prompt])
    if args.local_files_only:
        cmd.append("--local-files-only")
    return cmd


def _closed_loop_sweep_dataset_root(args: argparse.Namespace) -> Path:
    return resolve_lerobot_root_for_start_report(
        args.closed_loop_start_report_path,
        repo_root=args.repo_root,
    )


def _first_trace_path(report: dict[str, Any]) -> Path | None:
    for episode in report.get("episodes") or []:
        value = episode.get("trace_path")
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute():
            return path
        if path.exists():
            return path
        report_path = Path(str(report.get("report_path") or ""))
        base = report_path.parent if report_path.parent != Path(".") else Path.cwd()
        candidate = base / path
        return candidate if candidate.exists() else path
    return None


def _first_dataset_episode_index(report: dict[str, Any]) -> int | None:
    for episode in report.get("episodes") or []:
        state = episode.get("start_contract_state") or {}
        for key in ("dataset_source_episode_index", "dataset_episode_index"):
            value = state.get(key)
            if value is not None:
                return int(value)
        value = episode.get("dataset_episode_index")
        if value is not None:
            return int(value)
        reset_meta = episode.get("reset_meta") or {}
        value = reset_meta.get("source_report_episode_index")
        if value is not None:
            return int(value)
        value = episode.get("episode")
        if value is not None:
            return int(value)
    return None


def _teacher_action_rmse_series(*, dataset_root: Path, episode_index: int, rollout_jsonl: Path) -> tuple[list[float], int]:
    import numpy as np
    import pandas as pd

    files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot parquet files under {dataset_root / 'data'}")
    table = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "frame_index", "action"]) for path in files],
        ignore_index=True,
    )
    episode = table[table["episode_index"] == int(episode_index)].sort_values("frame_index")
    if episode.empty:
        raise ValueError(f"episode_index {episode_index} not found under {dataset_root}")
    teacher = np.stack(episode["action"].to_numpy()).astype(float)[:, :6]
    rollout_actions = []
    for line in rollout_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        action = row.get("action") or (row.get("policy_output") or {}).get("processor_postprocessed_action")
        if action is not None:
            rollout_actions.append([float(value) for value in action[:6]])
    if not rollout_actions:
        raise ValueError(f"no rollout actions found in {rollout_jsonl}")
    rollout = np.asarray(rollout_actions, dtype=float)
    horizon = min(len(teacher), len(rollout))
    rmse = np.sqrt(((rollout[:horizon] - teacher[:horizon]) ** 2).mean(axis=1))
    return [float(value) for value in rmse.tolist()], int(len(teacher))


def _teacher_action_rmse_series_from_records(
    *,
    dataset_root: Path,
    episode_index: int,
    records: list[dict[str, Any]],
) -> tuple[list[float], int]:
    import numpy as np
    import pandas as pd

    files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot parquet files under {dataset_root / 'data'}")
    table = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "frame_index", "action"]) for path in files],
        ignore_index=True,
    )
    episode = table[table["episode_index"] == int(episode_index)].sort_values("frame_index")
    if episode.empty:
        raise ValueError(f"episode_index {episode_index} not found under {dataset_root}")
    teacher = np.stack(episode["action"].to_numpy()).astype(float)[:, :6]
    rollout_actions = []
    for record in records:
        action = record.get("action") or record.get("raw_action")
        if action is not None:
            rollout_actions.append([float(value) for value in action[:6]])
    if not rollout_actions:
        raise ValueError("no rollout record actions found")
    rollout = np.asarray(rollout_actions, dtype=float)
    horizon = min(len(teacher), len(rollout))
    rmse = np.sqrt(((rollout[:horizon] - teacher[:horizon]) ** 2).mean(axis=1))
    return [float(value) for value in rmse.tolist()], int(len(teacher))


def _summarize_rmse_series(series: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in series if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "max": None, "step0": None, "last": None}
    return {
        "mean": float(sum(finite) / len(finite)),
        "max": float(max(finite)),
        "step0": float(finite[0]),
        "last": float(finite[-1]),
    }


def _summarize_requery_boundary(
    records: list[dict[str, Any]],
    *,
    n_action_steps: int,
) -> dict[str, float | int | None]:
    import numpy as np

    boundary: list[float] = []
    non_boundary: list[float] = []
    overlap: list[float] = []
    previous: np.ndarray | None = None
    for index, record in enumerate(records):
        value = record.get("requery_overlap_rmse")
        if isinstance(value, (int, float)):
            overlap.append(float(value))
        action = record.get("action") or record.get("raw_action")
        if action is None:
            continue
        current = np.asarray(action[:6], dtype=float)
        if previous is not None:
            jump = float(np.sqrt(np.mean(np.square(current - previous))))
            is_boundary = bool(record.get("policy_inference", index % max(1, int(n_action_steps)) == 0))
            (boundary if is_boundary else non_boundary).append(jump)
        previous = current
    boundary_mean = float(np.mean(boundary)) if boundary else None
    non_boundary_mean = float(np.mean(non_boundary)) if non_boundary else None
    ratio = None
    if boundary_mean is not None and non_boundary_mean is not None and non_boundary_mean > 0:
        ratio = boundary_mean / non_boundary_mean
    return {
        "requery_boundary_rmse_mean": boundary_mean,
        "requery_boundary_count": len(boundary),
        "non_boundary_step_rmse_mean": non_boundary_mean,
        "non_boundary_step_count": len(non_boundary),
        "requery_boundary_ratio": ratio,
        "requery_overlap_rmse_mean": float(np.mean(overlap)) if overlap else None,
        "requery_overlap_count": len(overlap),
    }


def _merge_requery_boundary_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    def weighted_mean(value_key: str, count_key: str) -> tuple[float | None, int]:
        pairs = [
            (float(summary[value_key]), int(summary[count_key]))
            for summary in summaries
            if summary.get(value_key) is not None and int(summary.get(count_key) or 0) > 0
        ]
        count = sum(pair[1] for pair in pairs)
        if count <= 0:
            return None, 0
        return sum(value * pair_count for value, pair_count in pairs) / count, count

    boundary_mean, boundary_count = weighted_mean(
        "requery_boundary_rmse_mean",
        "requery_boundary_count",
    )
    non_boundary_mean, non_boundary_count = weighted_mean(
        "non_boundary_step_rmse_mean",
        "non_boundary_step_count",
    )
    overlap_mean, overlap_count = weighted_mean(
        "requery_overlap_rmse_mean",
        "requery_overlap_count",
    )
    ratio = None
    if boundary_mean is not None and non_boundary_mean is not None and non_boundary_mean > 0:
        ratio = boundary_mean / non_boundary_mean
    return {
        "requery_boundary_rmse_mean": boundary_mean,
        "requery_boundary_count": boundary_count,
        "non_boundary_step_rmse_mean": non_boundary_mean,
        "non_boundary_step_count": non_boundary_count,
        "requery_boundary_ratio": ratio,
        "requery_overlap_rmse_mean": overlap_mean,
        "requery_overlap_count": overlap_count,
    }


def _write_action_rmse_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "n_action_steps",
        "success",
        "reference_drift_step0",
        "reference_drift_mean",
        "reference_drift_max",
        "reference_drift_last",
        "approach_mean",
        "alignment_mean",
        "grip_lift_mean",
        "requery_boundary_rmse_mean",
        "non_boundary_step_rmse_mean",
        "requery_boundary_ratio",
        "requery_overlap_rmse_mean",
        "tcp",
        "lift",
        "run_dir",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _plot_action_rmse_sweep(
    *,
    plot_path: Path,
    series_by_n: dict[int, list[float]],
    rows: list[dict[str, Any]],
    teacher_frames: int,
    phase_segments: list[dict[str, Any]] | None = None,
    y_axis_max: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8.5), dpi=140)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1.2], height_ratios=[3.0, 1.0], hspace=0.35, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    table_ax = fig.add_subplot(gs[1, :])
    colors = plt.cm.tab10.colors
    for index, n_action_steps in enumerate(sorted(series_by_n)):
        series = series_by_n[n_action_steps]
        color = colors[index % len(colors)]
        ax.plot(list(range(len(series))), series, color=color, linewidth=2, label=f"n_action_steps={n_action_steps}")
        for frame in range(0, max(teacher_frames, len(series)), max(1, int(n_action_steps))):
            ax.axvline(frame, color=color, alpha=0.13, linewidth=0.8)
    for segment_index, segment in enumerate(phase_segments or []):
        start_frame = int(segment["start_frame"])
        if segment_index > 0:
            ax.axvline(start_frame, color="black", linestyle="--", linewidth=1.8, alpha=0.8)
        center = (
            int(segment["start_frame"]) + int(segment["end_frame_exclusive"]) - 1
        ) / 2.0
        ax.text(
            center,
            1.005,
            str(segment["id"]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    title = (
        "Three-phase n_action_steps teacher-reference drift"
        if len(phase_segments or []) > 1
        else "n_action_steps teacher-reference drift"
    )
    fig.suptitle(title, x=0.065, y=0.985, ha="left", fontsize=12, fontweight="bold")
    fig.text(
        0.065,
        0.958,
        "X-axis = combined teacher frame index. Y-axis = RMSE(executed rollout action, source teacher action). "
        f"Y-axis is fixed at 0-{float(y_axis_max):g}. Black dashed lines mark primitive transitions.",
        fontsize=8,
    )
    ax.set_xlabel("combined teacher frame index")
    ax.set_ylabel("teacher-reference drift RMSE")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.0, float(y_axis_max))
    legend_ax.axis("off")
    legend_ax.set_title("Legend / result", loc="left", fontsize=11, fontweight="bold")
    y = 0.96
    for index, row in enumerate(rows):
        n_value = row.get("n_action_steps")
        if n_value not in series_by_n:
            continue
        color = colors[index % len(colors)]
        legend_ax.plot([0.0, 0.18], [y, y], color=color, linewidth=3, transform=legend_ax.transAxes)
        label = (
            f"n_action_steps={n_value}\n"
            f"success={row.get('success')} drift={_fmt_plot_float(row.get('reference_drift_mean'))} "
            f"approach={_fmt_plot_float(row.get('approach_mean'))} "
            f"align={_fmt_plot_float(row.get('alignment_mean'))} "
            f"grip/lift={_fmt_plot_float(row.get('grip_lift_mean'))}"
        )
        legend_ax.text(0.22, y, label, transform=legend_ax.transAxes, va="center", fontsize=8)
        y -= 0.1
    table_ax.axis("off")
    table_rows = [
        [
            row.get("n_action_steps"),
            row.get("success"),
            _fmt_plot_float(row.get("reference_drift_mean")),
            _fmt_plot_float(row.get("approach_mean")),
            _fmt_plot_float(row.get("alignment_mean")),
            _fmt_plot_float(row.get("grip_lift_mean")),
            _fmt_plot_float(row.get("requery_boundary_rmse_mean")),
            _fmt_plot_float(row.get("requery_overlap_rmse_mean")),
            _fmt_plot_float(row.get("tcp")),
            _fmt_plot_float(row.get("lift")),
        ]
        for row in rows
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=[
            "n",
            "success",
            "all RMSE",
            "approach",
            "alignment",
            "grip/lift",
            "requery jump",
            "overlap",
            "tcp",
            "lift",
        ],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.15)
    fig.text(
        0.08,
        0.015,
        "Drift is an open-loop source-teacher reference, not an oracle rollout-state error. Boundary/other isolate executed action discontinuity; overlap compares old-tail vs new-prefix when recorded.",
        fontsize=8,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def _fmt_plot_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.3f}"


def _parse_int_csv(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        number = int(part)
        if number > 0 and number not in result:
            result.append(number)
    return result


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


def _episode_final_is_grasped(episode: dict[str, Any]) -> float | None:
    if not isinstance(episode, dict):
        return None
    if episode.get("final_is_grasped") is not None:
        return float(episode["final_is_grasped"])
    final_info = episode.get("final_info")
    if isinstance(final_info, dict) and final_info.get("is_grasped") is not None:
        return float(final_info["is_grasped"])
    trace_path = Path(str(episode.get("trace_path") or ""))
    if not trace_path.exists():
        return None
    rows = _read_jsonl(trace_path)
    for row in reversed(rows):
        info = row.get("info") if isinstance(row, dict) else None
        if isinstance(info, dict) and info.get("is_grasped") is not None:
            return float(info["is_grasped"])
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
    names = [
        path.name
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name != "last" and (path.name.isdigit() or (path / "pretrained_model").exists())
    ]
    return sorted(names, key=lambda name: (_checkpoint_sort_step(_run_dir_from_checkpoint_root(checkpoints_dir), name), name))


def _validation_already_recorded(run_dir: Path, checkpoint: str) -> bool:
    path = run_dir / "metrics" / "validation_metrics.jsonl"
    started_at = _current_training_started_at_utc(run_dir)
    for row in _read_jsonl(path):
        if not _metric_row_belongs_to_current_training(row, started_at):
            continue
        if _metric_checkpoint_matches(run_dir, row, checkpoint):
            return True
    return False


def _validation_deferred_already_recorded(run_dir: Path, checkpoint: str) -> bool:
    path = run_dir / "metrics" / "monitor_events.jsonl"
    for row in _read_jsonl(path):
        if row.get("kind") == "validation_deferred" and _metric_checkpoint_matches(run_dir, row, checkpoint):
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
    started_at = _current_training_started_at_utc(run_dir)
    closed_loop_rows = [
        row
        for row in _read_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl")
        if str(row.get("test_id", "default")) == test_id
        and _metric_row_belongs_to_current_training(row, started_at)
    ]
    schedule_checkpoint = _checkpoint_schedule_name(run_dir, checkpoint)
    return should_run_closed_loop(
        schedule=SO101TrainingSchedule(
            closed_loop_policy=policy,
            closed_loop_every_epochs=args.closed_loop_every_epochs,
            steps_per_epoch=args.steps_per_epoch,
            stop_on_overfit=args.stop_training_on_overfit,
            overfit_patience_checkpoints=args.overfit_patience_checkpoints,
            overfit_min_delta=args.overfit_min_delta,
        ),
        checkpoint=schedule_checkpoint,
        validation_rows=[
            row
            for row in _read_jsonl(run_dir / "metrics" / "validation_metrics.jsonl")
            if _metric_row_belongs_to_current_training(row, started_at)
        ],
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
    started_at = _current_training_started_at_utc(run_dir)
    for row in _read_jsonl(path):
        if not _metric_row_belongs_to_current_training(row, started_at):
            continue
        if _metric_checkpoint_matches(run_dir, row, checkpoint):
            return True
    return False


def _append_validation_metric(
    run_dir: Path,
    checkpoint: str,
    report: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": _checkpoint_step(run_dir, checkpoint),
        "loss": report["loss_mean"],
        "postprocessed_action_rmse_mean": report.get("postprocessed_action_rmse_mean"),
        "postprocessed_action_global_rmse": report.get("postprocessed_action_global_rmse"),
        "postprocessed_action_rmse_step0_mean": report.get("postprocessed_action_rmse_step0_mean"),
        "postprocessed_action_rmse_max": report.get("postprocessed_action_rmse_max"),
        "postprocessed_action_rmse_frame_count": report.get("postprocessed_action_rmse_frame_count"),
        "checkpoint": checkpoint,
        "split": "validation",
        "batches_evaluated": report.get("batches_evaluated"),
        "batch_size": report.get("batch_size", args.batch_size),
        "samples_seen": report.get("samples_seen"),
        "source": Path(report.get("output_path", "")).name,
    }
    _append_jsonl(run_dir / "metrics" / "validation_metrics.jsonl", row)
    _write_validation_tensorboard(run_dir, row)


def _append_closed_loop_metric(run_dir: Path, checkpoint: str, report: dict[str, Any]) -> None:
    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": _checkpoint_step(run_dir, checkpoint),
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
    write_summary = write_so101_training_loop_test_results(run_dir, row, report)
    row["tensorboard_evidence_status"] = write_summary.get("status")
    row["tensorboard_evidence_scalars"] = write_summary.get("scalars")
    row["tensorboard_evidence_images"] = write_summary.get("images")
    row["tensorboard_evidence_videos"] = write_summary.get("videos")
    row["tensorboard_train_reference"] = write_summary.get("train_reference")
    _append_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl", row)


def write_so101_training_loop_test_results(run_dir: Path, row: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Write the canonical SO101 training-time loop-test evidence package.

    Training loop-test result generation must go through this function. It owns
    the stable TensorBoard tag contract for closed-loop scalars, RMSE sweep
    plots, canonical rollout videos, debug raw-media mirrors, and train
    references. Do not add ad hoc loop-test TensorBoard writers elsewhere.
    """

    write_summary = _write_closed_loop_tensorboard(run_dir, row, report)
    _require_closed_loop_tensorboard_evidence(row, report, write_summary)
    _append_jsonl(run_dir / "metrics" / "closed_loop_tensorboard_writes.jsonl", write_summary)
    return write_summary


def _require_closed_loop_tensorboard_evidence(
    row: dict[str, Any],
    report: dict[str, Any],
    write_summary: dict[str, Any],
) -> None:
    test_id = _safe_id(str(row.get("test_id") or "default"))
    if write_summary.get("status") != "ok":
        raise RuntimeError(f"closed-loop TensorBoard evidence write failed: {write_summary}")

    scalars = set(str(tag) for tag in write_summary.get("scalars") or [])
    images = set(str(tag) for tag in write_summary.get("images") or [])
    videos = set(str(tag) for tag in write_summary.get("videos") or [])
    required: list[str] = [f"closed_loop/{test_id}/success_rate"]
    media_enabled = bool(write_summary.get("media_enabled", True))

    action_rmse_sweep = report.get("action_rmse_sweep")
    if isinstance(action_rmse_sweep, dict) and not action_rmse_sweep.get("skipped"):
        required.append(f"closed_loop/{test_id}/action_rmse_sweep")

    missing = [
        tag
        for tag in required
        if tag not in scalars and tag not in images and tag not in videos
    ]

    if not media_enabled:
        if missing:
            raise RuntimeError(
                "closed-loop TensorBoard evidence incomplete: "
                + ", ".join(missing)
                + f"; summary={write_summary}"
            )
        return

    rollout_layout = str(write_summary.get("rollout_layout") or "per_episode")
    if rollout_layout == "combined":
        combined_tag = f"closed_loop/{test_id}/rollout_combined"
        if combined_tag not in videos:
            missing.append(combined_tag)
    else:
        episode_count = len(report.get("episodes") or [])
        rollout_prefix = f"closed_loop/{test_id}/rollout_episode_"
        rollout_count = sum(1 for tag in videos if tag.startswith(rollout_prefix))
        if episode_count and rollout_count < episode_count:
            missing.append(f"{rollout_prefix}<NNN> ({rollout_count}/{episode_count})")

    train_reference_prefix = f"closed_loop/{test_id}/train_reference_"
    train_reference = write_summary.get("train_reference")
    reference_already_recorded = bool(
        isinstance(train_reference, dict)
        and train_reference.get("frequency") in {"once_per_run", "disabled"}
        and train_reference.get("status") in {"already_recorded", "disabled"}
    )
    if not reference_already_recorded and not any(
        tag.startswith(train_reference_prefix) for tag in videos
    ):
        missing.append(f"{train_reference_prefix}<camera>")

    if missing:
        raise RuntimeError(
            "closed-loop TensorBoard evidence incomplete: "
            + ", ".join(missing)
            + f"; summary={write_summary}"
        )


def replay_so101_training_loop_test_tensorboard(run_dir: Path) -> list[dict[str, Any]]:
    """Re-attach existing loop-test reports to the active TensorBoard logdir."""
    summaries: list[dict[str, Any]] = []
    for row in _read_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl"):
        report_path_value = row.get("report_path")
        if not report_path_value:
            continue
        report_path = Path(str(report_path_value))
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            summary = {
                "checkpoint": row.get("checkpoint"),
                "step": row.get("step"),
                "test_id": row.get("test_id"),
                "status": "failed",
                "error": f"read_report_failed: {exc}",
            }
            _append_jsonl(run_dir / "metrics" / "closed_loop_tensorboard_replays.jsonl", summary)
            summaries.append(summary)
            continue
        report.setdefault("closed_loop_test_id", row.get("test_id", "default"))
        summary = _write_closed_loop_tensorboard(run_dir, row, report)
        summary["replay"] = True
        _append_jsonl(run_dir / "metrics" / "closed_loop_tensorboard_replays.jsonl", summary)
        summaries.append(summary)
    return summaries


def _write_validation_tensorboard(run_dir: Path, row: dict[str, Any]) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return

    loss = row.get("loss")
    if not isinstance(loss, (int, float)):
        return
    step = int(row.get("step") or 0)
    log_dir = _smolvla_tensorboard_log_dir(run_dir)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        writer.add_scalar("val/loss", float(loss), global_step=step)
        writer.add_scalar(IMPORTANT_VAL_LOSS_TAG, float(loss), global_step=step)
        action_rmse = row.get("postprocessed_action_rmse_mean")
        if isinstance(action_rmse, (int, float)):
            writer.add_scalar("val/postprocessed_action_rmse", float(action_rmse), global_step=step)
            writer.add_scalar(IMPORTANT_VAL_POSTPROCESSED_ACTION_RMSE_TAG, float(action_rmse), global_step=step)
        global_rmse = row.get("postprocessed_action_global_rmse")
        if isinstance(global_rmse, (int, float)):
            writer.add_scalar("val/postprocessed_action_global_rmse", float(global_rmse), global_step=step)
        step0_rmse = row.get("postprocessed_action_rmse_step0_mean")
        if isinstance(step0_rmse, (int, float)):
            writer.add_scalar("val/postprocessed_action_rmse_step0", float(step0_rmse), global_step=step)
        max_rmse = row.get("postprocessed_action_rmse_max")
        if isinstance(max_rmse, (int, float)):
            writer.add_scalar("val/postprocessed_action_rmse_max", float(max_rmse), global_step=step)
        frame_count = row.get("postprocessed_action_rmse_frame_count")
        if isinstance(frame_count, (int, float)):
            writer.add_scalar("extra/val/postprocessed_action_rmse_frame_count", float(frame_count), global_step=step)
        batches = row.get("batches_evaluated")
        if isinstance(batches, (int, float)):
            writer.add_scalar("extra/val/batches_evaluated", float(batches), global_step=step)
        samples = row.get("samples_seen")
        if isinstance(samples, (int, float)):
            writer.add_scalar("extra/val/samples_seen", float(samples), global_step=step)


def _smolvla_tensorboard_log_dir(run_dir: Path) -> Path:
    return run_dir / "tensorboard"


def _write_closed_loop_tensorboard(run_dir: Path, row: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    step = int(row.get("step") or 0)
    media_step = _closed_loop_tensorboard_media_step(row, step)
    test_id = _safe_id(str(row.get("test_id") or "default"))
    summary: dict[str, Any] = {
        "step": step,
        "media_step": media_step,
        "checkpoint": row.get("checkpoint"),
        "test_id": test_id,
        "scalars": [],
        "images": [],
        "videos": [],
        "status": "started",
    }
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        summary["status"] = "skipped"
        summary["error"] = f"tensorboard_import_failed: {exc}"
        return summary

    log_dir = _smolvla_tensorboard_log_dir(run_dir)
    media_enabled = _closed_loop_test_case_media_enabled(run_dir, test_id)
    summary["media_enabled"] = media_enabled
    train_reference = (
        _train_reference_write_plan(run_dir, test_id, log_dir=log_dir, report=report)
        if media_enabled
        else {
            "frequency": "disabled",
            "status": "disabled",
            "write": False,
            "tags": [],
        }
    )
    summary["train_reference"] = train_reference
    side_by_side_videos = (
        _closed_loop_policy_camera_side_by_side_videos(report)
        if media_enabled
        else {}
    )
    rollout_layout = _closed_loop_rollout_layout(run_dir, report)
    summary["rollout_layout"] = rollout_layout
    if rollout_layout == "combined":
        combined_video = _combine_closed_loop_episode_videos(side_by_side_videos)
        rollout_videos = {"combined": combined_video} if combined_video is not None else {}
        debug_side_by_side_videos: dict[str, Any] = {}
        side_by_side_videos.clear()
    else:
        rollout_videos = _canonical_closed_loop_rollout_videos(
            side_by_side_videos=side_by_side_videos
        )
        debug_side_by_side_videos = {}
    try:
        with SummaryWriter(log_dir=str(log_dir)) as writer:
            for key in ("success_rate", "grasp_rate", "episodes", "duration_s"):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    tag = f"closed_loop/{test_id}/{key}"
                    writer.add_scalar(tag, float(value), global_step=step)
                    summary["scalars"].append(tag)
                    if key == "success_rate":
                        important_tag = f"{IMPORTANT_CLOSED_LOOP_SUCCESS_RATE_TAG}/{test_id}"
                        writer.add_scalar(important_tag, float(value), global_step=step)
                        summary["scalars"].append(important_tag)
            rmse_plot = _closed_loop_action_rmse_sweep_plot_path(report)
            rmse_sweep = report.get("action_rmse_sweep")
            if isinstance(rmse_sweep, dict):
                for rmse_row in rmse_sweep.get("rows") or []:
                    n_action_steps = _int_or_none(rmse_row.get("n_action_steps"))
                    if n_action_steps is None:
                        continue
                    for field, metric_name in (
                        ("reference_drift_mean", "teacher_reference_drift_rmse"),
                        ("requery_boundary_rmse_mean", "requery_boundary_rmse"),
                        ("non_boundary_step_rmse_mean", "non_boundary_step_rmse"),
                        ("requery_boundary_ratio", "requery_boundary_ratio"),
                        ("requery_overlap_rmse_mean", "requery_overlap_rmse"),
                    ):
                        value = rmse_row.get(field)
                        if not isinstance(value, (int, float)):
                            continue
                        tag = f"closed_loop/{test_id}/{metric_name}/nact_{n_action_steps}"
                        writer.add_scalar(tag, float(value), global_step=step)
                        summary["scalars"].append(tag)
            if rmse_plot is not None:
                image = _read_hwc_image(rmse_plot)
                if image is not None:
                    tag = f"closed_loop/{test_id}/action_rmse_sweep"
                    writer.add_image(
                        tag,
                        image,
                        global_step=step,
                        dataformats="HWC",
                    )
                    summary["images"].append(tag)
            for video_name, video in rollout_videos.items():
                tag = f"closed_loop/{test_id}/rollout_{video_name}"
                writer.add_video(
                    tag,
                    video,
                    global_step=media_step,
                    fps=12,
                )
                summary["videos"].append(tag)
            for camera_name, video in debug_side_by_side_videos.items():
                tag = f"extra/closed_loop/{test_id}/rollout_camera_trace_{camera_name}"
                writer.add_video(
                    tag,
                    video,
                    global_step=media_step,
                    fps=12,
                )
                summary["videos"].append(tag)
            if train_reference["write"]:
                reference_tags = []
                for camera_name, video in _training_reference_camera_side_by_side_videos(
                    run_dir,
                    report=report,
                ).items():
                    tag = f"closed_loop/{test_id}/train_reference_{camera_name}"
                    writer.add_video(
                        tag,
                        video,
                        global_step=media_step,
                        fps=12,
                    )
                    summary["videos"].append(tag)
                    reference_tags.append(tag)
                train_reference["tags"] = reference_tags
                train_reference["status"] = "recorded" if reference_tags else "missing"
        if (
            train_reference["frequency"] == "once_per_run"
            and train_reference["status"] == "recorded"
        ):
            _write_train_reference_marker(
                run_dir,
                test_id,
                log_dir=log_dir,
                tags=train_reference["tags"],
                reference_contract=str(train_reference["reference_contract"]),
            )
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
    return summary


def _closed_loop_tensorboard_media_config(run_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        config = _resolved_training_config_for_run(run_dir, repo_root=repo_root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        config = None
    closed_loop = (
        ((config or {}).get("training_config") or {}).get("closed_loop")
        or (config or {}).get("closed_loop")
        or {}
    )
    media = closed_loop.get("tensorboard_media")
    return dict(media) if isinstance(media, dict) else {}


def _closed_loop_test_case_media_enabled(run_dir: Path, test_id: str) -> bool:
    media_config = _closed_loop_tensorboard_media_config(run_dir)
    render_test_cases = media_config.get("render_test_cases")
    if not isinstance(render_test_cases, list):
        return True
    return str(test_id) in {str(value) for value in render_test_cases}


def _closed_loop_rollout_layout(run_dir: Path, report: dict[str, Any]) -> str:
    phase_contract = report.get("phase_contract")
    is_chain = isinstance(phase_contract, dict) and phase_contract.get("mode") == "chain"
    if not is_chain:
        return "per_episode"
    media_config = _closed_loop_tensorboard_media_config(run_dir)
    layout = media_config.get("chain_rollout_layout")
    return str(layout) if layout in {"per_episode", "combined"} else "per_episode"


def _train_reference_write_plan(
    run_dir: Path,
    test_id: str,
    *,
    log_dir: Path,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    media_config = _closed_loop_tensorboard_media_config(run_dir)
    frequency = str(media_config.get("train_reference_frequency") or "every_checkpoint")
    reference_contract = _training_reference_contract(report)
    plan: dict[str, Any] = {
        "frequency": frequency,
        "reference_contract": reference_contract,
        "status": "pending",
        "write": True,
        "tags": [],
    }
    if frequency == "disabled":
        plan.update(status="disabled", write=False)
        return plan
    if frequency == "every_checkpoint":
        return plan
    if frequency != "once_per_run":
        raise ValueError(f"unsupported train_reference_frequency: {frequency!r}")

    marker = _read_json(_train_reference_marker_path(run_dir, test_id))
    event_files = {
        path.name for path in log_dir.glob("events.out.tfevents.*") if path.is_file()
    }
    marker_events = set(str(value) for value in (marker or {}).get("event_files") or [])
    marker_contract = str((marker or {}).get("reference_contract") or "")
    if marker_contract == reference_contract and marker_events & event_files:
        plan.update(
            status="already_recorded",
            write=False,
            tags=list((marker or {}).get("tags") or []),
        )
        return plan
    existing_tags = _tensorboard_train_reference_tags(log_dir, test_id)
    if existing_tags and reference_contract != "phase_chain_v1":
        _write_train_reference_marker(
            run_dir,
            test_id,
            log_dir=log_dir,
            tags=existing_tags,
            reference_contract=reference_contract,
        )
        plan.update(status="already_recorded", write=False, tags=existing_tags)
    return plan


def _train_reference_marker_path(run_dir: Path, test_id: str) -> Path:
    return run_dir / "metrics" / "train_reference_media" / f"{_safe_id(test_id)}.json"


def _write_train_reference_marker(
    run_dir: Path,
    test_id: str,
    *,
    log_dir: Path,
    tags: list[str],
    reference_contract: str = "single_dataset_v1",
) -> None:
    marker_path = _train_reference_marker_path(run_dir, test_id)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "test_id": test_id,
                "reference_contract": reference_contract,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "event_files": sorted(
                    path.name
                    for path in log_dir.glob("events.out.tfevents.*")
                    if path.is_file()
                ),
                "tags": list(tags),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _training_reference_contract(report: dict[str, Any] | None) -> str:
    phase_contract = (report or {}).get("phase_contract")
    if isinstance(phase_contract, dict) and phase_contract.get("mode") == "chain":
        return "phase_chain_v1"
    return "single_dataset_v1"


def _tensorboard_train_reference_tags(log_dir: Path, test_id: str) -> list[str]:
    if not log_dir.exists():
        return []
    prefix = f"closed_loop/{_safe_id(test_id)}/train_reference_"
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(
            str(log_dir),
            size_guidance={"images": 0, "tensors": 0},
        )
        accumulator.Reload()
        tags = accumulator.Tags()
        candidates = list(tags.get("images") or []) + list(tags.get("tensors") or [])
        return sorted({str(tag) for tag in candidates if str(tag).startswith(prefix)})
    except Exception:
        return []


def _closed_loop_tensorboard_media_step(row: dict[str, Any], checkpoint_step: int) -> int:
    value = _int_or_none(row.get("tensorboard_media_step"))
    return value if value is not None else checkpoint_step


def _closed_loop_action_rmse_sweep_plot_path(report: dict[str, Any]) -> Path | None:
    sweep = report.get("action_rmse_sweep")
    if not isinstance(sweep, dict):
        return None
    value = sweep.get("plot_path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _closed_loop_policy_camera_videos(report: dict[str, Any]) -> dict[str, Any]:
    videos = {}
    n_action_steps = _closed_loop_n_action_steps(report)
    max_frames_per_episode = _closed_loop_media_max_frames_per_episode(report)
    for camera_feature, camera_label in (
        ("observation.images.camera1", "camera1"),
        ("observation.images.camera2", "camera2"),
    ):
        for episode_index, episode_frames in _closed_loop_policy_camera_frames_by_episode(
            report,
            camera_feature,
            max_frames_per_episode=max_frames_per_episode,
            n_action_steps=n_action_steps,
        ).items():
            video = _frames_to_tensorboard_video(episode_frames)
            if video is not None:
                videos[f"{camera_label}_episode_{episode_index:03d}"] = video
    return videos


def _closed_loop_policy_camera_side_by_side_videos(report: dict[str, Any]) -> dict[str, Any]:
    n_action_steps = _closed_loop_n_action_steps(report)
    max_frames_per_episode = _closed_loop_media_max_frames_per_episode(report)
    camera1_by_episode = _closed_loop_policy_camera_frames_by_episode(
        report,
        "observation.images.camera1",
        max_frames_per_episode=max_frames_per_episode,
        n_action_steps=n_action_steps,
    )
    camera2_by_episode = _closed_loop_policy_camera_frames_by_episode(
        report,
        "observation.images.camera2",
        max_frames_per_episode=max_frames_per_episode,
        n_action_steps=n_action_steps,
    )
    videos = {}
    for episode_index in sorted(set(camera1_by_episode) & set(camera2_by_episode)):
        pairs = list(zip(camera1_by_episode[episode_index], camera2_by_episode[episode_index]))
        video = _side_by_side_tensorboard_video(pairs, left_title="camera1 egocentric", right_title="camera2 wrist")
        if video is not None:
            videos[f"camera1_camera2_episode_{episode_index:03d}"] = video
    return videos


def _canonical_closed_loop_rollout_videos(
    *,
    side_by_side_videos: dict[str, Any],
) -> dict[str, Any]:
    return {
        key.replace("camera1_camera2_", ""): value
        for key, value in side_by_side_videos.items()
        if key.startswith("camera1_camera2_episode_")
    }


def _combine_closed_loop_episode_videos(
    side_by_side_videos: dict[str, Any],
) -> Any | None:
    """Concatenate complete episode rollouts into one time-ordered video."""

    try:
        import torch
        import torch.nn.functional as functional

        ordered = [
            video
            for key, video in sorted(
                side_by_side_videos.items(),
                key=lambda item: (
                    _episode_index_from_video_name(item[0])
                    if _episode_index_from_video_name(item[0]) is not None
                    else 1_000_000
                ),
            )
            if getattr(video, "ndim", 0) == 5 and int(video.shape[0]) == 1
        ]
        if not ordered:
            return None

        source_height = int(ordered[0].shape[-2])
        source_width = int(ordered[0].shape[-1])
        sequences = []
        for video in ordered:
            frames = video[0]
            if int(frames.shape[-2]) != source_height or int(frames.shape[-1]) != source_width:
                frames = functional.interpolate(
                    frames,
                    size=(source_height, source_width),
                    mode="nearest",
                )
            sequences.append(frames)
        return torch.cat(sequences, dim=0).unsqueeze(0)
    except Exception:
        return None


def _export_closed_loop_tensorboard_style_gifs(
    report: dict[str, Any],
    output_dir: Path,
    *,
    success_only: bool = True,
    fps: int = 12,
) -> list[dict[str, Any]]:
    """Export review GIFs using the exact TensorBoard rollout rendering path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    success_episodes = {
        int(episode.get("episode"))
        for episode in report.get("episodes") or []
        if (not success_only or bool(episode.get("final_success"))) and _int_or_none(episode.get("episode")) is not None
    }
    exported: list[dict[str, Any]] = []
    for video_name, video in _closed_loop_policy_camera_side_by_side_videos(report).items():
        episode_index = _episode_index_from_video_name(video_name)
        if episode_index is None:
            continue
        if success_only and episode_index not in success_episodes:
            continue
        gif_path = output_dir / f"{video_name}_tensorboard_style.gif"
        if not _write_tensorboard_video_gif(video, gif_path, fps=fps):
            continue
        exported.append(
            {
                "episode": episode_index,
                "video_name": video_name,
                "gif_path": str(gif_path),
                "rendering_contract": "tensorboard_closed_loop_camera1_camera2_side_by_side",
            }
        )
    return exported


def _episode_index_from_video_name(video_name: str) -> int | None:
    marker = "_episode_"
    if marker not in video_name:
        return None
    return _int_or_none(video_name.rsplit(marker, 1)[-1])


def _write_tensorboard_video_gif(video: Any, gif_path: Path, *, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
        import numpy as np

        array = video.detach().cpu().numpy() if hasattr(video, "detach") else np.asarray(video)
        # TensorBoard video shape is (N,T,C,H,W). This exporter writes the first batch item.
        if array.ndim != 5 or array.shape[0] < 1:
            return False
        frames = array[0].transpose(0, 2, 3, 1)
        frames = np.clip(frames, 0, 255).astype("uint8", copy=False)
        imageio.mimsave(gif_path, list(frames), fps=max(1, int(fps)))
        return gif_path.exists()
    except Exception:
        return False


def _closed_loop_n_action_steps(report: dict[str, Any]) -> int:
    config = report.get("policy_rollout_config")
    if isinstance(config, dict):
        value = _int_or_none(config.get("n_action_steps"))
        if value and value > 0:
            return value
    return 15


def _closed_loop_media_max_frames_per_episode(report: dict[str, Any]) -> int:
    counts: list[int] = []
    for episode in report.get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        records = episode.get("records")
        if isinstance(records, list):
            counts.append(len(records))
        steps = _int_or_none(episode.get("steps"))
        if steps is not None:
            counts.append(steps)
    return max(counts) if counts else 160


def _training_reference_camera_videos(run_dir: Path) -> dict[str, Any]:
    dataset_root = _training_reference_dataset_root(run_dir)
    if dataset_root is None:
        return {}
    frames_by_camera = _training_reference_camera_frames_by_episode(
        dataset_root,
        max_episodes=10,
        max_frames_per_episode=_training_reference_media_max_frames_per_episode(run_dir),
    )
    videos = {}
    for camera_label, episodes in frames_by_camera.items():
        for episode_index, frames in episodes.items():
            video = _image_frames_to_tensorboard_video(frames)
            if video is not None:
                videos[f"{camera_label}_episode_{episode_index:03d}"] = video
    return videos


def _training_reference_camera_side_by_side_videos(
    run_dir: Path,
    *,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _training_reference_contract(report) == "phase_chain_v1":
        chain_videos = _phase_chain_reference_camera_side_by_side_videos(report or {})
        if chain_videos:
            return chain_videos
    dataset_root = _training_reference_dataset_root(run_dir)
    if dataset_root is None:
        return {}
    frames_by_camera = _training_reference_camera_frames_by_episode(
        dataset_root,
        max_episodes=10,
        max_frames_per_episode=_training_reference_media_max_frames_per_episode(run_dir),
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


def _phase_chain_reference_camera_side_by_side_videos(
    report: dict[str, Any],
) -> dict[str, Any]:
    episode_specs: list[tuple[int, list[dict[str, Any]]]] = []
    requirements: dict[Path, dict[str, Any]] = {}
    report_contract = report.get("phase_contract")
    report_phases = (
        list(report_contract.get("phases") or [])
        if isinstance(report_contract, dict)
        else []
    )
    for episode in report.get("episodes") or []:
        if not isinstance(episode, dict):
            continue
        output_episode_index = _int_or_none(episode.get("episode"))
        if output_episode_index is None:
            continue
        episode_contract = episode.get("phase_contract")
        phases = (
            list(episode_contract.get("phases") or [])
            if isinstance(episode_contract, dict)
            else report_phases
        )
        phase_specs: list[dict[str, Any]] = []
        fallback_source_index = _int_or_none(
            (episode.get("reset_meta") or {}).get("source_report_episode_index")
        )
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            dataset_root = _phase_reference_dataset_root(
                phase.get("reference_report_path")
            )
            source_episode_index = _int_or_none(
                phase.get("reference_report_episode_index")
            )
            if source_episode_index is None:
                source_episode_index = fallback_source_index
            if dataset_root is None or source_episode_index is None:
                phase_specs = []
                break
            phase_spec = {
                "id": str(phase.get("id") or "phase"),
                "prompt": str(phase.get("prompt") or ""),
                "dataset_root": dataset_root,
                "source_episode_index": source_episode_index,
                "max_steps": max(1, int(phase.get("max_steps") or 1)),
            }
            phase_specs.append(phase_spec)
            requirement = requirements.setdefault(
                dataset_root,
                {"episode_indices": set(), "max_frames_per_episode": 1},
            )
            requirement["episode_indices"].add(source_episode_index)
            requirement["max_frames_per_episode"] = max(
                int(requirement["max_frames_per_episode"]),
                int(phase_spec["max_steps"]),
            )
        if phase_specs:
            episode_specs.append((output_episode_index, phase_specs))
    if not episode_specs:
        return {}

    frames_by_root: dict[Path, dict[str, dict[int, list[tuple[Any, ...]]]]] = {}
    for dataset_root, requirement in requirements.items():
        episode_indices = sorted(int(value) for value in requirement["episode_indices"])
        frames_by_root[dataset_root] = _training_reference_camera_frames_by_episode(
            dataset_root,
            max_episodes=len(episode_indices),
            max_frames_per_episode=int(requirement["max_frames_per_episode"]),
            episode_indices=episode_indices,
        )

    videos: dict[str, Any] = {}
    for output_episode_index, phases in episode_specs:
        frames_by_camera: dict[str, list[tuple[Any, ...]]] = {
            "camera1": [],
            "camera2": [],
        }
        complete = True
        for phase in phases:
            source_frames = frames_by_root.get(phase["dataset_root"]) or {}
            for camera_label in ("camera1", "camera2"):
                phase_frames = (source_frames.get(camera_label) or {}).get(
                    int(phase["source_episode_index"])
                )
                if not phase_frames:
                    complete = False
                    break
                for phase_frame_index, frame_item in enumerate(phase_frames):
                    source, _label, _inference, target, _metadata = (
                        _unpack_video_frame_item(frame_item)
                    )
                    label = (
                        f"teacher chain ep {output_episode_index:03d} | "
                        f"{phase['id']} | frame {phase_frame_index:03d}"
                    )
                    prompt = _short_closed_loop_prompt(phase["prompt"])
                    if prompt:
                        label = f"{label}\nprompt: {prompt}"
                    frames_by_camera[camera_label].append(
                        (
                            source,
                            label,
                            False,
                            target,
                            {
                                "phase": phase["id"],
                                "camera": camera_label,
                                "active_camera": "",
                                "active": False,
                                "termination_reason": "",
                            },
                        )
                    )
            if not complete:
                break
        if not complete:
            continue
        pairs = list(
            zip(
                frames_by_camera["camera1"],
                frames_by_camera["camera2"],
                strict=True,
            )
        )
        video = _side_by_side_tensorboard_video(
            pairs,
            left_title="camera1 egocentric",
            right_title="camera2 wrist",
        )
        if video is not None:
            videos[
                f"camera1_camera2_episode_{output_episode_index:03d}"
            ] = video
    return videos


def _phase_reference_dataset_root(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    dataset_root = path.parent if path.suffix == ".json" else path
    return dataset_root if dataset_root.exists() else None


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


def _training_reference_media_max_frames_per_episode(run_dir: Path) -> int:
    metrics_path = run_dir / "metrics" / "closed_loop_metrics.jsonl"
    if metrics_path.exists():
        max_steps = 0
        for row in _read_jsonl(metrics_path):
            report_path_value = row.get("report_path")
            if not report_path_value:
                continue
            report = _read_json(Path(str(report_path_value)))
            if isinstance(report, dict):
                max_steps = max(max_steps, _closed_loop_media_max_frames_per_episode(report))
        if max_steps > 0:
            return max_steps
    return 160


def _training_reference_camera_frames_by_episode(
    dataset_root: Path,
    *,
    max_episodes: int,
    max_frames_per_episode: int,
    episode_indices: list[int] | None = None,
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
        available_episode_indices = {
            int(value) for value in df["episode_index"].dropna().unique()
        }
        if episode_indices is None:
            selected_episode_indices = sorted(available_episode_indices)[:max_episodes]
        else:
            selected_episode_indices = [
                int(value)
                for value in episode_indices
                if int(value) in available_episode_indices
            ][:max_episodes]
    except Exception:
        return {}
    sidecars = _reference_visual_servo_sidecars(dataset_root)
    result: dict[str, dict[int, list[tuple[Any, str, bool, dict[str, Any] | None]]]] = {
        camera: {} for camera in camera_columns
    }
    for episode_index in selected_episode_indices:
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
) -> dict[int, list[tuple[Path, str, bool, dict[str, Any] | None, dict[str, Any] | None]]]:
    episodes: dict[int, list[tuple[Path, str, bool, dict[str, Any] | None, dict[str, Any] | None]]] = {}
    for episode in report.get("episodes") or []:
        episode_index = _int_or_none(episode.get("episode"))
        if episode_index is None:
            continue
        episode_steps = _int_or_none(episode.get("steps"))
        termination = episode.get("termination") or {}
        termination_reason = str(termination.get("reason") or "") if isinstance(termination, dict) else ""
        trace_path_value = episode.get("trace_path")
        if not trace_path_value:
            continue
        trace_path = Path(str(trace_path_value))
        if not trace_path.exists():
            continue
        frames: list[tuple[Path, str, bool, dict[str, Any] | None, dict[str, Any] | None]] = []
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
                is_inference_frame = _is_closed_loop_inference_frame(row, row_frame, n_action_steps)
                metadata = _closed_loop_frame_metadata(row, camera_feature)
                if (
                    termination_reason
                    in {
                        "valid_mask_stop",
                        "valid_mask_verifier_failed",
                        "valid_mask_verifier_rejected",
                        "env_success",
                        "hard_cap",
                    }
                    and episode_steps is not None
                    and _int_or_none(row_frame) == episode_steps - 1
                ):
                    metadata = dict(metadata or {})
                    metadata["termination_reason"] = termination_reason
                frames.append(
                    (
                        path,
                        _closed_loop_frame_label(row_episode, row_frame, row.get("prompt")),
                        is_inference_frame,
                        _closed_loop_target_overlay(row, camera_feature),
                        metadata,
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


def _closed_loop_frame_metadata(row: dict[str, Any], camera_feature: str) -> dict[str, Any] | None:
    phase = row.get("phase") or row.get("primitive_id") or row.get("subgoal")
    camera = "camera1" if camera_feature.endswith("camera1") else "camera2"
    selection = _closed_loop_servo_selection(row)
    active_camera = selection.get("servo_camera") if isinstance(selection, dict) else None
    termination_reason = str(row.get("termination_reason") or "")
    metadata = {
        "phase": str(phase) if phase else "",
        "camera": camera,
        "active_camera": str(active_camera) if active_camera else "",
        "active": bool(active_camera == camera),
        "termination_reason": termination_reason,
    }
    if (
        not metadata["phase"]
        and not metadata["active_camera"]
        and not metadata["termination_reason"]
    ):
        return None
    return metadata


def _closed_loop_servo_selection(row: dict[str, Any]) -> dict[str, Any] | None:
    policy_output = row.get("policy_output")
    if isinstance(policy_output, dict):
        nested = _closed_loop_servo_selection(policy_output)
        if nested is not None:
            return nested
    prediction = row.get("visual_servo_prediction")
    if isinstance(prediction, dict) and isinstance(prediction.get("servo_selection"), dict):
        return prediction["servo_selection"]
    selection = row.get("servo_selection")
    return selection if isinstance(selection, dict) else None


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
    visible_prob = value.get(f"{camera}_visible_prob", value.get("visible_prob", 1.0))
    try:
        prob = max(0.0, min(1.0, float(visible_prob)))
        return {
            "dx_norm": float(dx),
            "dy_norm": float(dy),
            "label": f"{label} vis={prob:.2f}",
            "color": _probability_tinted_color(color, prob),
            "visible_prob": prob,
        }
    except (TypeError, ValueError):
        return None


def _probability_tinted_color(color: tuple[int, int, int], probability: float) -> tuple[int, int, int]:
    prob = max(0.0, min(1.0, float(probability)))
    floor = 45
    return tuple(max(0, min(255, int(round(floor + prob * (int(channel) - floor))))) for channel in color)


def _is_closed_loop_inference_frame(row: dict[str, Any], frame: Any, n_action_steps: int) -> bool:
    if "policy_inference" in row:
        return bool(row.get("policy_inference"))
    policy_output = row.get("policy_output")
    if isinstance(policy_output, dict):
        hold = policy_output.get("visual_servo_hold")
        if isinstance(hold, dict) and "inference_frame" in hold:
            return bool(hold.get("inference_frame"))
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
            path, label, is_inference_frame, target, metadata = _unpack_video_frame_item(frame_item)
            image = _read_hwc_image(path)
            if image is None:
                continue
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[2] > 3:
                array = array[:, :, :3]
            array = _overlay_closed_loop_frame_label(
                array,
                label,
                inference_frame=is_inference_frame,
                target=target,
                metadata=metadata,
            )
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
            image, label, is_inference_frame, target, metadata = _unpack_video_frame_item(frame_item)
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[2] > 3:
                array = array[:, :, :3]
            array = _overlay_closed_loop_frame_label(
                array,
                label,
                inference_frame=is_inference_frame,
                target=target,
                metadata=metadata,
            )
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
        if _frame_pair_has_terminal_border(frame_pairs[-1]):
            frames.extend(frames[-1].copy() for _ in range(3))
        stacked = np.stack(frames, axis=0)  # T,H,W,C
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0)
    except Exception:
        return None


def _frame_pair_has_terminal_border(frame_pair: tuple[tuple[Any, ...], tuple[Any, ...]]) -> bool:
    for item in frame_pair:
        _source, _label, _inference, _target, metadata = _unpack_video_frame_item(item)
        if metadata and metadata.get("termination_reason") in {
            "valid_mask_stop",
            "env_success",
        }:
            return True
    return False


def _video_frame_item_to_array(frame_item: tuple[Any, ...], *, title: str) -> Any | None:
    try:
        import numpy as np

        source, label, is_inference_frame, target, metadata = _unpack_video_frame_item(frame_item)
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
            metadata=metadata,
        ).astype("uint8", copy=False)
    except Exception:
        return None


def _unpack_video_frame_item(frame_item: tuple[Any, ...]) -> tuple[Any, str, bool, dict[str, Any] | None, dict[str, Any] | None]:
    if len(frame_item) >= 3:
        target = frame_item[3] if len(frame_item) >= 4 and isinstance(frame_item[3], dict) else None
        metadata = frame_item[4] if len(frame_item) >= 5 and isinstance(frame_item[4], dict) else None
        return frame_item[0], str(frame_item[1]), bool(frame_item[2]), target, metadata
    return frame_item[0], str(frame_item[1]), False, None, None


def _overlay_closed_loop_frame_label(
    array: Any,
    label: str,
    *,
    inference_frame: bool = False,
    target: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(np.asarray(array).astype("uint8", copy=False)).convert("RGB")
        draw = ImageDraw.Draw(image)
        termination_reason = str(metadata.get("termination_reason") or "") if metadata else ""
        terminal_border_color = {
            "valid_mask_stop": (255, 0, 0),
            "env_success": (0, 120, 255),
        }.get(termination_reason)
        terminal_border_width = max(5, image.width // 40) if terminal_border_color else 0
        active_border_width = max(4, image.width // 48) if metadata and metadata.get("active") else 0
        phase_color = _phase_border_color(str(metadata.get("phase") or "")) if active_border_width else (255, 190, 60)
        _draw_active_camera_border(draw, image.width, image.height, active_border_width, phase_color)
        border_width = max(3, image.width // 64) if inference_frame else 0
        _draw_inference_border(draw, image.width, image.height, border_width, active_border_width)
        font = ImageFont.load_default()
        _draw_target_overlay(draw, image.width, image.height, target, font=font)
        phase_label = _phase_label(metadata)
        if phase_label:
            label = f"{label}\n{phase_label}"
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=1)
        pad_x = 3
        pad_y = 2
        width = min(image.width, bbox[2] - bbox[0] + 2 * pad_x)
        height = min(image.height, bbox[3] - bbox[1] + 2 * pad_y)
        draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
        draw.multiline_text((pad_x, pad_y), label, fill=(255, 255, 255), font=font, spacing=1)
        _draw_active_camera_border(draw, image.width, image.height, active_border_width, phase_color)
        _draw_inference_border(draw, image.width, image.height, border_width, active_border_width)
        _draw_termination_border(
            draw,
            image.width,
            image.height,
            terminal_border_width,
            terminal_border_color,
        )
        return np.asarray(image)
    except Exception:
        return array


def _draw_active_camera_border(draw: Any, width: int, height: int, border_width: int, color: tuple[int, int, int]) -> None:
    if border_width <= 0:
        return
    for offset in range(border_width):
        draw.rectangle((offset, offset, width - 1 - offset, height - 1 - offset), outline=color)


def _draw_inference_border(draw: Any, width: int, height: int, border_width: int, inset: int) -> None:
    if border_width <= 0:
        return
    for offset in range(inset, inset + border_width):
        draw.rectangle((offset, offset, width - 1 - offset, height - 1 - offset), outline=(0, 255, 0))


def _draw_termination_border(
    draw: Any,
    width: int,
    height: int,
    border_width: int,
    color: tuple[int, int, int] | None,
) -> None:
    if border_width <= 0 or color is None:
        return
    for offset in range(border_width):
        draw.rectangle((offset, offset, width - 1 - offset, height - 1 - offset), outline=color)


def _phase_label(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    phase = str(metadata.get("phase") or "").strip()
    active = str(metadata.get("active_camera") or "").strip()
    termination_reason = str(metadata.get("termination_reason") or "").strip()
    parts = []
    if phase:
        parts.append(f"phase: {phase}")
    if active:
        parts.append(f"active: {active}")
    if termination_reason == "valid_mask_stop":
        parts.append("stop: valid-mask")
    elif termination_reason == "valid_mask_verifier_failed":
        parts.append("stop: verifier-failed")
    elif termination_reason == "valid_mask_verifier_rejected":
        parts.append("stop: verifier-rejected")
    elif termination_reason == "env_success":
        parts.append("stop: env-success")
    elif termination_reason == "hard_cap":
        parts.append("stop: hard-cap")
    return " | ".join(parts)


def _phase_border_color(phase: str) -> tuple[int, int, int]:
    palette = {
        "approach": (255, 140, 0),
        "alignment": (90, 180, 255),
        "grip_lift": (255, 80, 160),
        "move_over_cube_edge": (255, 140, 0),
        "align_fixed_jaw_cube_edge": (90, 180, 255),
        "grip_from_edge_cube": (255, 80, 160),
        "move_and_align_cube_edge": (120, 220, 120),
    }
    return palette.get(phase, (255, 190, 60))


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


def _run_dir_from_checkpoint_root(checkpoints_dir: Path) -> Path:
    checkpoints_dir = Path(checkpoints_dir)
    if checkpoints_dir.name == "checkpoints" and checkpoints_dir.parent.name == "model":
        return checkpoints_dir.parent.parent
    return checkpoints_dir.parent


def _checkpoint_step(run_dir: Path, checkpoint: str) -> int | None:
    numeric = _checkpoint_to_step(checkpoint)
    if numeric is not None:
        return numeric
    state = _read_json(run_dir / "model" / "checkpoints" / "retention_state.json")
    if not isinstance(state, dict):
        state = _read_json(run_dir / "checkpoints" / "retention_state.json")
    if not isinstance(state, dict):
        return None
    alias_to_key = {
        "best_train_loss": "best_train_loss_step",
        "best_val_loss": "best_val_loss_step",
        "best_closed_loop": "best_closed_loop_step",
    }
    value = state.get(alias_to_key.get(str(checkpoint), ""))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _checkpoint_sort_step(run_dir: Path, checkpoint: str) -> int:
    step = _checkpoint_step(run_dir, checkpoint)
    return int(step) if step is not None else 10**12


def _checkpoint_schedule_name(run_dir: Path, checkpoint: str) -> str:
    step = _checkpoint_step(run_dir, checkpoint)
    return f"{int(step):06d}" if step is not None else checkpoint


def _metric_checkpoint_matches(run_dir: Path, row: dict[str, Any], checkpoint: str) -> bool:
    row_checkpoint = str(row.get("checkpoint"))
    if row_checkpoint == str(checkpoint):
        return True
    target_step = _checkpoint_step(run_dir, checkpoint)
    if target_step is None:
        return False
    row_step = row.get("step")
    if isinstance(row_step, int) and row_step == target_step:
        return True
    if isinstance(row_step, str) and row_step.isdigit() and int(row_step) == target_step:
        return True
    return row_checkpoint == f"{int(target_step):06d}"


def _current_training_started_at_utc(run_dir: Path) -> datetime | None:
    summary = _read_json(run_dir / "training_run_summary.json") or {}
    value = summary.get("started_at_utc")
    if not isinstance(value, str):
        active = _read_json(run_dir.parent.parent / "active_training.json") or {}
        value = active.get("started_at_utc")
    if not isinstance(value, str):
        value = _latest_training_command_utc_from_log(run_dir / "logs" / "train.log")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _latest_training_command_utc_from_log(path: Path) -> str | None:
    if not path.exists():
        return None
    latest: str | None = None
    pattern = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2}T[^]]+\+00:00)]\s+\$")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        match = pattern.match(line)
        if match:
            latest = match.group("ts")
    return latest


def _metric_row_belongs_to_current_training(row: dict[str, Any], started_at: datetime | None) -> bool:
    if started_at is None:
        return True
    value = row.get("recorded_at_utc")
    if not isinstance(value, str):
        return False
    try:
        recorded_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    return recorded_at.astimezone(timezone.utc) >= started_at


def _now_local() -> str:
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value)).strip("_") or "default"


if __name__ == "__main__":
    main()
