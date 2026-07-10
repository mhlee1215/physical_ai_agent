#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_resolution_contract import (
    require_dataset_config_256,
    require_so101_image_resolution,
)
from physical_ai_agent.so101_hydra_config import SO101HydraTrainingEntry, load_so101_hydra_training_entry
from physical_ai_agent.so101_training_config_schema import validate_so101_training_config


DEFAULT_ROOT = Path("_workspace/so101_training")
DEFAULT_LOCK = DEFAULT_ROOT / "active_training.json"
DEFAULT_HYDRA_CONFIG = "training/grip_the_cube_v1"
LOCAL_TRAINING_STANDARD_DOC = Path("docs/so101_local_training_standard.md")
LOCAL_TRAINING_STANDARD_NAME = "primitive training with qwen validation v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonical SO101 training launcher. Enforces one active training run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_start_args(subparsers.add_parser("start", help="Start one Lightning training run."))
    _add_common_args(subparsers.add_parser("status", help="Print the active training status."))
    stop_parser = subparsers.add_parser("stop", help="Stop the active training run.")
    _add_common_args(stop_parser)
    stop_parser.add_argument("--timeout-s", type=float, default=20.0)
    args, passthrough = parser.parse_known_args()

    if args.command == "start":
        return start(args, passthrough)
    if args.command == "status":
        current = status(args.lock_file)
        print(json.dumps(current, indent=2, sort_keys=True) if args.json else _human_status(current))
        return 0
    if args.command == "stop":
        return stop(args.lock_file, timeout_s=args.timeout_s, json_output=args.json)
    raise AssertionError(args.command)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")


def _add_start_args(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument(
        "--preset",
        choices=["default", "grip-the-cube-v1-local", "qwen-edge-loopfix-local"],
        help=(
            "Apply a named local SO101 training preset. Presets are shortcuts for "
            "canonical start_so101_training.py flags; do not add separate wrapper scripts. "
            "When neither --preset nor --dataset-config is supplied, the launcher uses "
            "the default preset."
        ),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--training-id",
        help="Stable local ID for this training run. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        help="JSON file defining train/validation LeRobot datasets and training defaults.",
    )
    parser.add_argument(
        "--hydra-config",
        help=(
            "Hydra config name under configs/so101/hydra, for example "
            "`training/grip_the_cube_v1`. It resolves to a Pydantic-validated "
            "configs/so101/training/*.json file."
        ),
    )
    parser.add_argument("--host")
    parser.add_argument("--tensorboard-port", type=int)
    parser.add_argument("--dashboard-port", type=int)
    parser.add_argument("--no-tensorboard", action="store_true", default=None)
    parser.add_argument(
        "--no-tensorboard-tunnel",
        action="store_true",
        default=None,
        help="Do not start a cloudflared quick tunnel for external TensorBoard access.",
    )
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        default=None,
        help="Start the legacy local dashboard. Off by default; use only when explicitly requested.",
    )
    parser.add_argument(
        "--with-gpu-monitor",
        action="store_true",
        default=None,
        help="Start the TensorBoard GPU/system metrics helper. Off by default; use only when explicitly requested.",
    )
    parser.add_argument(
        "--with-progress-monitor",
        action="store_true",
        default=None,
        help="Start the closed-loop progress monitor. Off by default; use only when explicitly requested.",
    )
    parser.add_argument("--no-dashboard", action="store_true", default=None, help="Deprecated no-op: dashboard is off by default.")
    parser.add_argument("--no-gpu-monitor", action="store_true", default=None, help="Deprecated no-op: GPU monitor is off by default.")
    parser.add_argument(
        "--no-progress-monitor",
        action="store_true",
        default=None,
        help="Deprecated no-op: progress monitor is off by default.",
    )
    parser.add_argument(
        "--allow-incomplete-monitoring",
        action="store_true",
        default=None,
        help=(
            "Debug escape hatch: allow dataset-config training to start without "
            "the default TensorBoard, validation, checkpoint, and closed-loop guards."
        ),
    )
    parser.add_argument(
        "--hf-dataset-cache-root",
        type=Path,
        help="Local root for Hugging Face dataset subfolder downloads.",
    )
    parser.add_argument(
        "--skip-hf-dataset-download",
        action="store_true",
        default=None,
        help="Resolve configured HF cache roots without downloading. For debugging only.",
    )
    parser.add_argument(
        "--use-local-dataset-roots",
        action="store_true",
        default=None,
        help="Use dataset root fields from the config directly and ignore configured HF dataset sources.",
    )
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        default=None,
        help="Require configured HF dataset subfolders to already be present in the local HF cache root.",
    )
    parser.add_argument("--gpu-monitor-interval-s", type=float)
    parser.add_argument("--progress-monitor-interval-s", type=int)
    parser.add_argument(
        "--runtime-platform",
        choices=["auto", "macos", "linux"],
        help="Runtime profile for training/eval defaults. auto detects the current OS.",
    )
    parser.add_argument(
        "--training-device",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Default policy.device and Lightning accelerator when not explicitly forwarded.",
    )
    parser.add_argument("--closed-loop-every-epochs", type=int)
    parser.add_argument("--closed-loop-episodes", type=int)
    parser.add_argument("--closed-loop-steps", type=int)
    parser.add_argument("--closed-loop-env-id")
    parser.add_argument(
        "--closed-loop-mujoco-gl",
        choices=["auto", "glfw", "egl", "osmesa"],
        help="MuJoCo backend for closed-loop rollouts. auto uses glfw on macOS and egl on Linux.",
    )
    parser.add_argument(
        "--max-monitored-checkpoints",
        type=int,
        help="Fail fast when --steps/--save_freq would create more monitored checkpoints than this.",
    )
    parser.add_argument("--closed-loop-policy", choices=["off", "periodic", "best_only", "best_or_periodic"])
    parser.add_argument("--closed-loop-runner", choices=["auto", "picklift", "qwen_chain"])
    parser.add_argument("--closed-loop-eval-skill-mode")
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
        help="Action conversion mode for closed-loop evaluator. Defaults come from the selected Hydra launcher config.",
    )
    parser.add_argument("--closed-loop-record-rollout-gif", action="store_true", default=None)
    parser.add_argument("--record-loop-artifacts", action=argparse.BooleanOptionalAction)
    parser.add_argument("--render-loop-media", action=argparse.BooleanOptionalAction)
    parser.add_argument("--loop-artifact-width", type=int)
    parser.add_argument("--loop-artifact-height", type=int)
    parser.add_argument("--loop-artifact-fps", type=int)
    parser.add_argument("--loop-artifact-every-n-steps", type=int)
    parser.add_argument("--qwen-model")
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-api-key")
    parser.add_argument("--qwen-response-json", type=Path)
    parser.add_argument("--qwen-plan-json", type=Path)
    parser.add_argument("--qwen-object")
    parser.add_argument("--qwen-env-object-color")
    parser.add_argument("--closed-loop-subgoal-chain-mode", choices=["off", "fixed", "valid-mask"])
    parser.add_argument("--closed-loop-subgoal-sequence")
    parser.add_argument("--closed-loop-fixed-subgoal-chunks", type=int)
    parser.add_argument("--closed-loop-valid-mask-checkpoint", type=Path)
    parser.add_argument("--closed-loop-valid-mask-threshold", type=float)
    parser.add_argument("--closed-loop-valid-mask-consecutive", type=int)
    parser.add_argument("--closed-loop-policy-n-action-steps", type=int)
    parser.add_argument("--closed-loop-policy-num-steps", type=int)
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        help="Forward validation cadence as steps, e.g. 10 locally or 300 on cloud.",
    )
    parser.add_argument(
        "--validation-interval-epochs",
        type=int,
        help=(
            "Forward validation cadence as epochs. Defaults to 1 so HF/RunPod "
            "training writes val/loss whenever a validation dataset is configured. "
            "Ignored by the trainer when step cadence is also set."
        ),
    )
    parser.add_argument("--replace", action="store_true", help="Stop the active run before starting.")
    parser.add_argument("--dry-run", action="store_true", help="Print the launch plan without starting.")
    parser.add_argument(
        "--python",
        type=Path,
        help="Python executable for SO101 helper scripts.",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to lerobot_train_so101_lightning.py after an optional -- separator.",
    )


def start(args: argparse.Namespace, passthrough: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    passthrough = _apply_start_preset(args, passthrough, repo_root=repo_root)
    hydra_config_name = args.hydra_config or DEFAULT_HYDRA_CONFIG
    hydra_entry = load_so101_hydra_training_entry(hydra_config_name, repo_root=repo_root)
    _apply_hydra_launcher_defaults(args, hydra_entry, repo_root=repo_root)
    use_hydra_training_entry = bool(args.hydra_config) or args.dataset_config is None
    if use_hydra_training_entry:
        if args.dataset_config is not None:
            raise SystemExit("Use either --hydra-config or --dataset-config, not both.")
        args.dataset_config = hydra_entry.training_config_path(repo_root)
        if hydra_entry.training_args and not passthrough and not args.training_args:
            passthrough = ["--", *hydra_entry.training_args]
        args.hydra_config = hydra_config_name
    active = status(args.lock_file)
    if active.get("active"):
        if not args.replace:
            print(
                "Refusing to start: an SO101 training run is already active. "
                "Use `status`, `stop`, or `start --replace`.",
                file=sys.stderr,
            )
            print(_human_status(active), file=sys.stderr)
            return 2
        stop(args.lock_file, timeout_s=20.0)

    run_dir = args.run_dir.resolve()
    training_id = _training_id(args.training_id, run_dir)
    log_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    tensorboard_dir = run_dir / "tensorboard"
    train_output_dir = run_dir / "model"
    train_pid_file = run_dir / "train.pid"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = _load_dataset_config(args.dataset_config, repo_root=repo_root)
    if not args.use_local_dataset_roots:
        dataset_config = _resolve_hf_dataset_downloads(
            dataset_config,
            repo_root=repo_root,
            cache_root=args.hf_dataset_cache_root,
            download=not args.dry_run and not args.skip_hf_dataset_download,
            local_files_only=bool(args.hf_local_files_only),
        )
        dataset_config = _prepare_merged_datasets(
            dataset_config,
            repo_root=repo_root,
            python=args.python,
            merge=not args.dry_run,
        )
    if not args.dry_run:
        require_dataset_config_256(dataset_config, repo_root=repo_root, context="start_so101_training")
    require_so101_image_resolution(
        height=int(args.loop_artifact_height),
        width=int(args.loop_artifact_width),
        context="start_so101_training loop artifact media",
    )
    _ensure_train_grid_bin_sidecars(dataset_config, repo_root=repo_root, build=not args.dry_run)
    training_args = _forwarded_args(args.training_args, passthrough)
    training_args = _with_dataset_config(training_args, dataset_config, runtime_platform=args.runtime_platform)
    training_args = _with_validation_schedule(training_args, args)
    training_args = _with_checkpoint_schedule(training_args, dataset_config, args)
    training_args = _with_aligned_validation_schedule(training_args, dataset_config, args)
    runtime_contract = _runtime_contract(args, training_args)
    training_args = _with_runtime_contract(training_args, runtime_contract)
    training_args = _with_resume_checkpoint(training_args, train_output_dir)
    train_cmd = [
        str(args.python),
        str(repo_root / "scripts" / "lerobot_train_so101_lightning.py"),
        "--tensorboard-log-dir",
        str(tensorboard_dir),
        *_ensure_arg(training_args, "output_dir", str(train_output_dir)),
    ]
    training_run_summary_path = run_dir / "training_run_summary.json"
    train_cmd.extend(["--training-run-summary-path", str(training_run_summary_path)])
    tensorboard_exe = _tensorboard_executable(args.python, repo_root)
    tensorboard_cmd = tensorboard_exe if tensorboard_exe else [str(args.python), "-m", "tensorboard.main"]
    tensorboard_cmd.extend(
        ["--logdir", str(tensorboard_dir), "--host", args.host, "--port", str(args.tensorboard_port)]
    )
    tensorboard_tunnel_cmd = _tensorboard_tunnel_command(args.tensorboard_port)
    dashboard_cmd = [
        str(args.python),
        str(repo_root / "scripts" / "serve_so101_training_dashboard.py"),
        "--run-dir",
        str(run_dir),
        "--host",
        args.host,
        "--port",
        str(args.dashboard_port),
        "--repo-root",
        str(repo_root),
    ]
    gpu_monitor_cmd = [
        str(args.python),
        str(repo_root / "scripts" / "log_gpu_metrics_tensorboard.py"),
        "--log-dir",
        str(tensorboard_dir / "so101_system"),
        "--interval-s",
        str(args.gpu_monitor_interval_s),
        "--backend",
        "auto",
        "--train-pid-file",
        str(train_pid_file),
    ]
    progress_monitor_cmd = _progress_monitor_command(
        args=args,
        repo_root=repo_root,
        run_dir=run_dir,
        train_output_dir=train_output_dir,
        dataset_config=dataset_config,
        training_args=training_args,
        runtime_contract=runtime_contract,
        train_pid_file=train_pid_file,
    )
    post_checkpoint_loop_cmd = _post_checkpoint_loop_command(progress_monitor_cmd)
    post_checkpoint_loop_cmds = _post_checkpoint_loop_commands(
        progress_monitor_cmd=progress_monitor_cmd,
        dataset_config=dataset_config,
    )
    if post_checkpoint_loop_cmds:
        train_cmd.extend(["--post-checkpoint-loop-command-json", json.dumps(post_checkpoint_loop_cmds)])
    cache_build_cmds = _cache_build_commands(args.python, repo_root, dataset_config)
    enable_tensorboard = not args.no_tensorboard
    enable_tensorboard_tunnel = enable_tensorboard and not args.no_tensorboard_tunnel and tensorboard_tunnel_cmd is not None
    enable_dashboard = bool(args.with_dashboard) and not args.no_dashboard
    enable_gpu_monitor = bool(args.with_gpu_monitor) and not args.no_gpu_monitor
    enable_progress_monitor = bool(args.with_progress_monitor) and not args.no_progress_monitor

    launch_plan = {
        "operation": "start_so101_training",
        "training_id": training_id,
        "run_dir": str(run_dir),
        "train_output_dir": str(train_output_dir),
        "lock_file": str(args.lock_file.resolve()),
        "local_training_standard": _local_training_standard(repo_root),
        "train_cmd": train_cmd,
        "dataset_config": dataset_config,
        "hydra_config": args.hydra_config,
        "launcher_defaults_config": hydra_config_name,
        "hydra_entry": hydra_entry.model_dump(mode="json") if hydra_entry is not None else None,
        "tensorboard_cmd": tensorboard_cmd if enable_tensorboard else None,
        "tensorboard_tunnel_cmd": tensorboard_tunnel_cmd if enable_tensorboard_tunnel else None,
        "dashboard_cmd": dashboard_cmd if enable_dashboard else None,
        "gpu_monitor_cmd": gpu_monitor_cmd if enable_gpu_monitor else None,
        "progress_monitor_cmd": progress_monitor_cmd if enable_progress_monitor else None,
        "post_checkpoint_loop_cmd": post_checkpoint_loop_cmd,
        "post_checkpoint_loop_cmds": post_checkpoint_loop_cmds,
        "cache_build_cmds": cache_build_cmds,
        "runtime_contract": runtime_contract,
        "training_run_summary_path": str(training_run_summary_path),
        "tensorboard_url": f"http://127.0.0.1:{args.tensorboard_port}/" if enable_tensorboard else None,
        "mobile_tensorboard_url": _mobile_url(args.tensorboard_port) if enable_tensorboard else None,
        "external_tensorboard_url": None,
        "external_tensorboard_note": (
            "available after start via cloudflared"
            if enable_tensorboard_tunnel
            else ("cloudflared unavailable or --no-tensorboard-tunnel" if enable_tensorboard else None)
        ),
        "clear_tensorboard_old_data": bool(enable_tensorboard),
        "dashboard_url": f"http://127.0.0.1:{args.dashboard_port}/" if enable_dashboard else None,
        "logs": {
            "train": str(log_dir / "train.log"),
            "tensorboard": str(log_dir / "tensorboard.log") if enable_tensorboard else None,
            "tensorboard_tunnel": str(log_dir / "tensorboard_tunnel.log") if enable_tensorboard_tunnel else None,
            "dashboard": str(log_dir / "dashboard.log") if enable_dashboard else None,
            "gpu_monitor": str(log_dir / "gpu_monitor.log") if enable_gpu_monitor else None,
            "progress_monitor": str(log_dir / "progress_monitor.log") if enable_progress_monitor else None,
        },
    }
    _validate_monitoring_contract(
        args=args,
        dataset_config=dataset_config,
        training_args=training_args,
        train_cmd=train_cmd,
        launch_plan=launch_plan,
        runtime_contract=runtime_contract,
    )
    if args.dry_run:
        print(json.dumps(launch_plan, indent=2, sort_keys=True))
        return 0

    _run_cache_builds(cache_build_cmds, log_dir=log_dir, cwd=repo_root)
    if enable_tensorboard:
        _clear_tensorboard_old_data(tensorboard_dir)
    _write_training_run_summary(training_run_summary_path, launch_plan)
    train = _popen(train_cmd, log_dir / "train.log", cwd=repo_root)
    train_pid_file.write_text(str(train.pid) + "\n", encoding="utf-8")
    tensorboard = (
        _popen(tensorboard_cmd, log_dir / "tensorboard.log", cwd=repo_root) if enable_tensorboard else None
    )
    tensorboard_tunnel = (
        _popen(tensorboard_tunnel_cmd, log_dir / "tensorboard_tunnel.log", cwd=repo_root)
        if enable_tensorboard_tunnel and tensorboard_tunnel_cmd is not None
        else None
    )
    external_tensorboard_url = (
        _wait_for_tensorboard_tunnel_url(log_dir / "tensorboard_tunnel.log")
        if tensorboard_tunnel is not None
        else None
    )
    dashboard = _popen(dashboard_cmd, log_dir / "dashboard.log", cwd=repo_root) if enable_dashboard else None
    gpu_monitor = (
        _popen(gpu_monitor_cmd, log_dir / "gpu_monitor.log", cwd=repo_root) if enable_gpu_monitor else None
    )
    progress_monitor = (
        None
        if not enable_progress_monitor or progress_monitor_cmd is None
        else _popen(progress_monitor_cmd, log_dir / "progress_monitor.log", cwd=repo_root)
    )
    record = {
        **launch_plan,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_pid": train.pid,
        "tensorboard_pid": tensorboard.pid if tensorboard else None,
        "tensorboard_tunnel_pid": tensorboard_tunnel.pid if tensorboard_tunnel else None,
        "external_tensorboard_url": external_tensorboard_url,
        "external_tensorboard_note": (
            None
            if external_tensorboard_url
            else launch_plan.get("external_tensorboard_note")
        ),
        "dashboard_pid": dashboard.pid if dashboard else None,
        "gpu_monitor_pid": gpu_monitor.pid if gpu_monitor else None,
        "progress_monitor_pid": progress_monitor.pid if progress_monitor else None,
    }
    _write_json(args.lock_file, record)
    _update_training_registry(repo_root, record)
    current = status(args.lock_file)
    print(json.dumps(current, indent=2, sort_keys=True) if args.json else _human_status(current))
    return 0


def _clear_tensorboard_old_data(tensorboard_dir: Path) -> int:
    if not tensorboard_dir.exists():
        return 0
    removed = 0
    for pattern in ("events.out.tfevents*", "*.profile-empty"):
        for path in tensorboard_dir.rglob(pattern):
            if not path.is_file():
                continue
            path.unlink()
            removed += 1
    return removed


def _apply_hydra_launcher_defaults(
    args: argparse.Namespace,
    entry: SO101HydraTrainingEntry,
    *,
    repo_root: Path,
) -> None:
    defaults = entry.launcher

    path_fields = {
        "run_dir": defaults.run_dir,
        "hf_dataset_cache_root": defaults.hf_dataset_cache_root,
        "closed_loop_valid_mask_checkpoint": defaults.closed_loop_valid_mask_checkpoint,
        "python": defaults.python,
    }
    for name, value in path_fields.items():
        if getattr(args, name, None) is None and value is not None:
            path = Path(value)
            setattr(args, name, path if path.is_absolute() else repo_root / path)

    scalar_fields = {
        "host": defaults.host,
        "tensorboard_port": defaults.tensorboard_port,
        "dashboard_port": defaults.dashboard_port,
        "allow_incomplete_monitoring": defaults.allow_incomplete_monitoring,
        "skip_hf_dataset_download": defaults.skip_hf_dataset_download,
        "use_local_dataset_roots": defaults.use_local_dataset_roots,
        "hf_local_files_only": defaults.hf_local_files_only,
        "gpu_monitor_interval_s": defaults.gpu_monitor_interval_s,
        "progress_monitor_interval_s": defaults.progress_monitor_interval_s,
        "progress_monitor_batch_size": defaults.progress_monitor_batch_size,
        "progress_monitor_validation_max_batches": defaults.progress_monitor_validation_max_batches,
        "runtime_platform": defaults.runtime_platform,
        "training_device": defaults.training_device,
        "closed_loop_every_epochs": defaults.closed_loop_every_epochs,
        "closed_loop_episodes": defaults.closed_loop_episodes,
        "closed_loop_steps": defaults.closed_loop_steps,
        "closed_loop_env_id": defaults.closed_loop_env_id,
        "closed_loop_mujoco_gl": defaults.closed_loop_mujoco_gl,
        "max_monitored_checkpoints": defaults.max_monitored_checkpoints,
        "closed_loop_policy": defaults.closed_loop_policy,
        "closed_loop_runner": defaults.closed_loop_runner,
        "closed_loop_eval_skill_mode": defaults.closed_loop_eval_skill_mode,
        "closed_loop_task_prompt": defaults.closed_loop_task_prompt,
        "closed_loop_action_contract_mode": defaults.closed_loop_action_contract_mode,
        "closed_loop_record_rollout_gif": defaults.closed_loop_record_rollout_gif,
        "record_loop_artifacts": defaults.record_loop_artifacts,
        "render_loop_media": defaults.render_loop_media,
        "loop_artifact_width": defaults.loop_artifact_width,
        "loop_artifact_height": defaults.loop_artifact_height,
        "loop_artifact_fps": defaults.loop_artifact_fps,
        "loop_artifact_every_n_steps": defaults.loop_artifact_every_n_steps,
        "qwen_model": defaults.qwen_model,
        "qwen_base_url": defaults.qwen_base_url,
        "qwen_api_key": defaults.qwen_api_key,
        "qwen_object": defaults.qwen_object,
        "qwen_env_object_color": defaults.qwen_env_object_color,
        "closed_loop_subgoal_chain_mode": defaults.closed_loop_subgoal_chain_mode,
        "closed_loop_subgoal_sequence": defaults.closed_loop_subgoal_sequence,
        "closed_loop_fixed_subgoal_chunks": defaults.closed_loop_fixed_subgoal_chunks,
        "closed_loop_valid_mask_threshold": defaults.closed_loop_valid_mask_threshold,
        "closed_loop_valid_mask_consecutive": defaults.closed_loop_valid_mask_consecutive,
        "closed_loop_policy_n_action_steps": defaults.closed_loop_policy_n_action_steps,
        "closed_loop_policy_num_steps": defaults.closed_loop_policy_num_steps,
        "validation_interval_steps": defaults.validation_interval_steps,
        "validation_interval_epochs": defaults.validation_interval_epochs,
    }
    for name, value in scalar_fields.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)

    if args.no_tensorboard is None:
        args.no_tensorboard = not defaults.tensorboard_enabled
    if args.no_tensorboard_tunnel is None:
        args.no_tensorboard_tunnel = not defaults.tensorboard_tunnel_enabled
    if args.with_dashboard is None:
        args.with_dashboard = defaults.dashboard_enabled
    if args.with_gpu_monitor is None:
        args.with_gpu_monitor = defaults.gpu_monitor_enabled
    if args.with_progress_monitor is None:
        args.with_progress_monitor = defaults.progress_monitor_enabled
    if args.no_dashboard is None:
        args.no_dashboard = False
    if args.no_gpu_monitor is None:
        args.no_gpu_monitor = False
    if args.no_progress_monitor is None:
        args.no_progress_monitor = False


def _apply_start_preset(args: argparse.Namespace, passthrough: list[str], *, repo_root: Path) -> list[str]:
    preset = getattr(args, "preset", None)
    if not preset:
        return passthrough
    if preset == "default":
        args.hydra_config = DEFAULT_HYDRA_CONFIG
        return passthrough
    if preset == "grip-the-cube-v1-local":
        args.hydra_config = DEFAULT_HYDRA_CONFIG
        return passthrough
    if preset == "qwen-edge-loopfix-local":
        args.hydra_config = "training/qwen_edge_loopfix_local"
        return passthrough
    raise SystemExit(f"unknown preset: {args.preset}")


def status(lock_file: Path) -> dict[str, Any]:
    record = _read_json(lock_file) or {"lock_file": str(lock_file.resolve()), "active": False}
    record.setdefault("local_training_standard", _local_training_standard(Path(__file__).resolve().parents[1]))
    train = _process_status(record.get("train_pid"))
    tensorboard = _process_status(record.get("tensorboard_pid"))
    tensorboard_tunnel = _process_status(record.get("tensorboard_tunnel_pid"))
    dashboard = _process_status(record.get("dashboard_pid"))
    gpu_monitor = _process_status(record.get("gpu_monitor_pid"))
    progress_monitor = _process_status(record.get("progress_monitor_pid"))
    record["train"] = train
    record["tensorboard"] = tensorboard
    record["tensorboard_tunnel"] = tensorboard_tunnel
    record["dashboard"] = dashboard
    record["gpu_monitor"] = gpu_monitor
    record["progress_monitor"] = progress_monitor
    record["active"] = any(
        bool(process.get("alive"))
        for process in (train, tensorboard, tensorboard_tunnel, dashboard, gpu_monitor, progress_monitor)
    )
    if record.get("tensorboard_url") and not record.get("mobile_tensorboard_url"):
        port = _url_port(str(record["tensorboard_url"]))
        if port is not None:
            record["mobile_tensorboard_url"] = _mobile_url(port)
    return record


def _write_training_run_summary(path: Path, launch_plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: launch_plan.get(key)
        for key in (
            "operation",
            "training_id",
            "run_dir",
            "train_output_dir",
            "lock_file",
            "local_training_standard",
            "train_cmd",
            "dataset_config",
            "tensorboard_cmd",
            "tensorboard_tunnel_cmd",
            "dashboard_cmd",
            "gpu_monitor_cmd",
            "progress_monitor_cmd",
            "post_checkpoint_loop_cmd",
            "post_checkpoint_loop_cmds",
            "cache_build_cmds",
            "runtime_contract",
            "tensorboard_url",
            "mobile_tensorboard_url",
            "external_tensorboard_url",
            "external_tensorboard_note",
            "dashboard_url",
        )
    }
    summary["training_run_summary_path"] = str(path)
    summary["written_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(path, summary)


def _training_id(value: str | None, run_dir: Path) -> str:
    raw = value or run_dir.name
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip())
    text = text.strip("._-")
    if not text:
        text = run_dir.name
    return text


def _update_training_registry(repo_root: Path, record: dict[str, Any]) -> None:
    registry_path = repo_root / DEFAULT_ROOT / "training_runs_index.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = _read_json(registry_path)
    if not isinstance(registry, dict):
        registry = {"runs": []}
    runs = registry.get("runs")
    if not isinstance(runs, list):
        runs = []
    training_id = str(record.get("training_id") or _training_id(None, Path(str(record.get("run_dir") or "run"))))
    compact = {
        "training_id": training_id,
        "run_dir": record.get("run_dir"),
        "training_run_summary_path": record.get("training_run_summary_path"),
        "dataset_config_name": ((record.get("dataset_config") or {}).get("name") if isinstance(record.get("dataset_config"), dict) else None),
        "started_at_utc": record.get("started_at_utc"),
        "tensorboard_url": record.get("tensorboard_url"),
        "mobile_tensorboard_url": record.get("mobile_tensorboard_url"),
        "external_tensorboard_url": record.get("external_tensorboard_url"),
        "external_tensorboard_note": record.get("external_tensorboard_note"),
        "train_pid": record.get("train_pid"),
        "tensorboard_pid": record.get("tensorboard_pid"),
        "tensorboard_tunnel_pid": record.get("tensorboard_tunnel_pid"),
    }
    runs = [row for row in runs if not (isinstance(row, dict) and row.get("training_id") == training_id)]
    runs.append(compact)
    registry["runs"] = runs
    registry["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(registry_path, registry)


def stop(lock_file: Path, *, timeout_s: float, json_output: bool = False) -> int:
    record = _read_json(lock_file)
    if not record:
        payload = {"active": False, "detail": "no active lock"}
        print(json.dumps(payload, indent=2, sort_keys=True) if json_output else _human_status(payload))
        return 0
    pids = [
        record.get("gpu_monitor_pid"),
        record.get("progress_monitor_pid"),
        record.get("dashboard_pid"),
        record.get("tensorboard_tunnel_pid"),
        record.get("tensorboard_pid"),
        record.get("train_pid"),
    ]
    for pid in pids:
        _terminate(pid)
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if not any(_process_status(pid).get("alive") for pid in pids):
            break
        time.sleep(0.5)
    for pid in pids:
        if _process_status(pid).get("alive"):
            _kill(pid)
    final = status(lock_file)
    final["stopped_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(lock_file, final)
    print(json.dumps(final, indent=2, sort_keys=True) if json_output else _human_status(final))
    return 0


def _human_status(record: dict[str, Any]) -> str:
    lines = [
        f"SO101 training active: {record.get('active')}",
        f"run_dir: {record.get('run_dir', '-')}",
        f"train: {_process_line(record.get('train'))}",
        f"tensorboard: {_process_line(record.get('tensorboard'))}",
        f"tensorboard_tunnel: {_process_line(record.get('tensorboard_tunnel'))}",
        f"dashboard: {_process_line(record.get('dashboard'))}",
        f"gpu_monitor: {_process_line(record.get('gpu_monitor'))}",
        f"progress_monitor: {_process_line(record.get('progress_monitor'))}",
    ]
    if record.get("tensorboard_url"):
        lines.append(f"tensorboard_url: {record['tensorboard_url']}")
    if record.get("mobile_tensorboard_url"):
        lines.append(f"mobile_tensorboard_url: {record['mobile_tensorboard_url']}")
    if record.get("external_tensorboard_url"):
        lines.append(f"external_tensorboard_url: {record['external_tensorboard_url']}")
    elif record.get("external_tensorboard_note"):
        lines.append(f"external_tensorboard_url: unavailable ({record['external_tensorboard_note']})")
    if record.get("dashboard_url"):
        lines.append(f"dashboard_url: {record['dashboard_url']}")
    standard = record.get("local_training_standard")
    if isinstance(standard, dict):
        lines.append(f"local_training_standard: {standard.get('name', '-')}")
        lines.append(f"standard_doc: {standard.get('doc', '-')}")
    logs = record.get("logs") or {}
    if logs.get("train"):
        lines.append(f"train_log: {logs['train']}")
    return "\n".join(lines)


def _mobile_url(port: int) -> str | None:
    host = _lan_ip_address()
    if not host:
        return None
    return f"http://{host}:{int(port)}/"


def _url_port(url: str) -> int | None:
    try:
        tail = url.rsplit(":", 1)[1]
        return int(tail.split("/", 1)[0])
    except (IndexError, ValueError):
        return None


def _lan_ip_address() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        pass
    for interface in ("en0", "en1"):
        try:
            completed = subprocess.run(
                ["ipconfig", "getifaddr", interface],
                check=False,
                text=True,
                capture_output=True,
            )
        except OSError:
            continue
        host = completed.stdout.strip()
        if completed.returncode == 0 and host and not host.startswith("127."):
            return host
    host = _ifconfig_lan_ip()
    if host:
        return host
    try:
        host = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None
    if host.startswith("127."):
        return None
    return host


def _ifconfig_lan_ip() -> str | None:
    try:
        completed = subprocess.run(["ifconfig"], check=False, text=True, capture_output=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    blocks = re.split(r"\n(?=\S)", completed.stdout)
    for block in blocks:
        if "status: active" not in block:
            continue
        match = re.search(r"\binet (192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[0-1])\.\d+\.\d+)\b", block)
        if match:
            return match.group(1)
    return None


def _process_line(process: Any) -> str:
    if not isinstance(process, dict):
        return "-"
    pid = process.get("pid")
    alive = process.get("alive")
    if alive is None:
        return "not started"
    return f"{'alive' if alive else 'stopped'} pid={pid}"


def _forwarded_args(training_args: list[str], passthrough: list[str]) -> list[str]:
    args = [*training_args, *passthrough]
    return args[1:] if args[:1] == ["--"] else args


def _ensure_arg(args: list[str], name: str, value: str) -> list[str]:
    prefix = f"--{name}="
    spaced = f"--{name}"
    if any(arg.startswith(prefix) or arg == spaced for arg in args):
        return args
    return [*args, f"{prefix}{value}"]


def _ensure_boolean_optional_arg(args: list[str], name: str, *, value: bool) -> list[str]:
    flag = f"--{name}"
    no_flag = f"--no-{name}"
    if any(arg == flag or arg == no_flag for arg in args):
        return args
    return [*args, flag if value else no_flag]


def _with_resume_checkpoint(args: list[str], train_output_dir: Path) -> list[str]:
    if not _truthy_arg(args, "resume"):
        return args
    if _arg_present(args, "so101-resume-checkpoint-path"):
        return args
    checkpoint = _latest_resume_checkpoint(train_output_dir / "checkpoints")
    if checkpoint is None:
        return args
    return [*args, f"--so101-resume-checkpoint-path={checkpoint}"]


def _arg_present(args: list[str], name: str) -> bool:
    prefix = f"--{name}="
    flag = f"--{name}"
    return any(arg == flag or arg.startswith(prefix) for arg in args)


def _truthy_arg(args: list[str], name: str) -> bool:
    flag = f"--{name}"
    prefix = f"{flag}="
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                return True
            return str(args[index + 1]).lower() in {"1", "true", "yes", "on"}
        if arg.startswith(prefix):
            return arg.split("=", 1)[1].lower() in {"1", "true", "yes", "on"}
    return False


def _latest_resume_checkpoint(checkpoint_root: Path) -> Path | None:
    last_path = checkpoint_root / "last"
    if last_path.exists():
        return last_path
    if not checkpoint_root.exists():
        return None
    candidates = [path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.name))


def _load_dataset_config(path: Path | None, *, repo_root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path if path.is_absolute() else repo_root / path
    payload = _read_json(resolved)
    if payload is None:
        raise SystemExit(f"Dataset config not found or empty: {resolved}")
    validation_errors = validate_so101_training_config(payload, path=resolved, repo_root=repo_root, strict=False)
    if validation_errors:
        detail = "\n".join(f"- {error}" for error in validation_errors)
        raise SystemExit(f"SO101 training config schema validation failed:\n{detail}")
    payload["config_path"] = str(resolved)
    return payload


def _resolve_hf_dataset_downloads(
    config: dict[str, Any] | None,
    *,
    repo_root: Path,
    cache_root: Path,
    download: bool,
    local_files_only: bool = False,
) -> dict[str, Any] | None:
    if not config:
        return config
    updated = copy.deepcopy(config)
    resolved_cache_root = cache_root if cache_root.is_absolute() else repo_root / cache_root
    downloads: list[dict[str, Any]] = []
    train_datasets = updated.get("train_datasets")
    if isinstance(train_datasets, list) and train_datasets:
        for source_index, dataset in enumerate(train_datasets):
            if not isinstance(dataset, dict):
                raise SystemExit(f"dataset config train_datasets[{source_index}] must be an object")
            resolved_source = _resolve_hf_dataset_source(
                dataset,
                fallback=updated,
                repo_root=repo_root,
                resolved_cache_root=resolved_cache_root,
                dataset_key="train_datasets",
                source_index=source_index,
                download=download,
                local_files_only=local_files_only,
            )
            resolved_root = Path(resolved_source["source"]["root"])
            dataset.update(resolved_source["source"])
            dataset["root"] = str(resolved_root)
            dataset["hf_resolved_root"] = str(resolved_root)
            downloads.append(resolved_source["download_record"])
    for dataset_key in ("train_dataset", "validation_dataset"):
        if dataset_key == "train_dataset" and isinstance(train_datasets, list) and train_datasets:
            continue
        dataset = updated.get(dataset_key)
        if not isinstance(dataset, dict):
            continue
        merge_sources = dataset.get("hf_merge_sources")
        if isinstance(merge_sources, list) and merge_sources:
            resolved_sources = []
            for source_index, source in enumerate(merge_sources):
                if not isinstance(source, dict):
                    raise SystemExit(f"dataset config {dataset_key}.hf_merge_sources[{source_index}] must be an object")
                resolved_source = _resolve_hf_dataset_source(
                    source,
                    fallback=dataset,
                    repo_root=repo_root,
                    resolved_cache_root=resolved_cache_root,
                    dataset_key=dataset_key,
                    source_index=source_index,
                    download=download,
                    local_files_only=local_files_only,
                )
                downloads.append(resolved_source["download_record"])
                resolved_sources.append(resolved_source["source"])
            dataset["hf_resolved_sources"] = resolved_sources
            continue
        hf_repo_id = dataset.get("hf_repo_id") or updated.get("hf_repo_id")
        hf_path = dataset.get("hf_path_in_repo")
        if not hf_repo_id and not hf_path:
            continue
        if not hf_repo_id or not hf_path:
            raise SystemExit(f"dataset config {dataset_key} must define both hf_repo_id and hf_path_in_repo")
        resolved_source = _resolve_hf_dataset_source(
            dataset,
            fallback=updated,
            repo_root=repo_root,
            resolved_cache_root=resolved_cache_root,
            dataset_key=dataset_key,
            source_index=None,
            download=download,
            local_files_only=local_files_only,
        )
        resolved_root = Path(resolved_source["source"]["root"])
        dataset["root"] = str(resolved_root)
        dataset["hf_resolved_root"] = str(resolved_root)
        downloads.append(resolved_source["download_record"])
    if downloads:
        updated["hf_dataset_downloads"] = downloads
    return updated


def _resolve_hf_dataset_source(
    source: dict[str, Any],
    *,
    fallback: dict[str, Any],
    repo_root: Path,
    resolved_cache_root: Path,
    dataset_key: str,
    source_index: int | None,
    download: bool,
    local_files_only: bool,
) -> dict[str, Any]:
    del repo_root
    hf_repo_id = source.get("hf_repo_id") or fallback.get("hf_repo_id")
    hf_path = source.get("hf_path_in_repo")
    if not hf_repo_id or not hf_path:
        if source_index is None:
            label = dataset_key
        elif dataset_key == "train_datasets":
            label = f"train_datasets[{source_index}]"
        else:
            label = f"{dataset_key}.hf_merge_sources[{source_index}]"
        raise SystemExit(f"dataset config {label} must define both hf_repo_id and hf_path_in_repo")
    hf_repo_type = str(source.get("hf_repo_type") or fallback.get("hf_repo_type") or "dataset")
    hf_revision = str(source.get("hf_revision") or fallback.get("hf_revision") or "")
    local_repo_dir = _hf_local_repo_dir(resolved_cache_root, str(hf_repo_id), hf_revision)
    hf_path_str = str(hf_path).strip("/")
    allow_patterns = [f"{hf_path_str}/**"]
    if download:
        download_kwargs: dict[str, Any] = {
            "repo_id": str(hf_repo_id),
            "repo_type": hf_repo_type,
            "allow_patterns": allow_patterns,
            "local_dir": local_repo_dir,
            "local_files_only": local_files_only,
        }
        if hf_revision:
            download_kwargs["revision"] = hf_revision
        _snapshot_download(**download_kwargs)
    resolved_root = local_repo_dir / hf_path_str
    resolved = {
        "name": source.get("name"),
        "repo_id": str(source.get("repo_id") or ""),
        "root": str(resolved_root),
        "hf_repo_id": str(hf_repo_id),
        "hf_repo_type": hf_repo_type,
        "hf_path_in_repo": hf_path_str,
        "hf_revision": hf_revision,
        "expected_episodes": source.get("expected_episodes"),
        "expected_frames": source.get("expected_frames"),
    }
    return {
        "source": {key: value for key, value in resolved.items() if value not in (None, "")},
        "download_record": {
            "dataset_key": dataset_key,
            "source_index": source_index,
            "source_name": source.get("name"),
            "repo_id": str(hf_repo_id),
            "repo_type": hf_repo_type,
            "path_in_repo": hf_path_str,
            "revision": hf_revision,
            "allow_patterns": allow_patterns,
            "local_dir": str(local_repo_dir),
            "resolved_root": str(resolved_root),
            "downloaded": bool(download),
            "local_files_only": bool(local_files_only),
        },
    }


def _prepare_merged_datasets(
    config: dict[str, Any] | None,
    *,
    repo_root: Path,
    python: Path,
    merge: bool,
) -> dict[str, Any] | None:
    if not config:
        return config
    if isinstance(config.get("train_datasets"), list) and config["train_datasets"]:
        return config
    for dataset_key in ("train_dataset", "validation_dataset"):
        config = _prepare_merged_dataset(
            config,
            dataset_key=dataset_key,
            repo_root=repo_root,
            python=python,
            merge=merge,
        )
    return config


def _prepare_merged_dataset(
    config: dict[str, Any],
    *,
    dataset_key: str,
    repo_root: Path,
    python: Path,
    merge: bool,
) -> dict[str, Any]:
    dataset = config.get(dataset_key)
    if not isinstance(dataset, dict):
        return config
    sources = dataset.get("hf_resolved_sources")
    if not isinstance(sources, list) or not sources:
        return config
    if "root" not in dataset or "repo_id" not in dataset:
        raise SystemExit(f"merged {dataset_key} must define root and repo_id")
    output_root = _resolve_root_path(repo_root, Path(str(dataset["root"])))
    shard_roots = [Path(str(source["root"])) for source in sources if isinstance(source, dict) and "root" in source]
    if len(shard_roots) != len(sources):
        raise SystemExit(f"merged {dataset_key} sources must resolve to local roots")
    command = [
        str(python),
        str(repo_root / "scripts" / "merge_so101_lerobot_shards.py"),
        "--output-root",
        str(output_root),
        "--repo-id",
        str(dataset["repo_id"]),
        "--overwrite",
    ]
    for shard_root in shard_roots:
        command.extend(["--shard", str(shard_root)])
    dataset["root"] = str(output_root)
    dataset["merge_command"] = command
    dataset["merged_from"] = [str(path) for path in shard_roots]
    if merge:
        subprocess.run(command, cwd=repo_root, check=True, text=True)
    return config


def _prepare_merged_train_dataset(
    config: dict[str, Any] | None,
    *,
    repo_root: Path,
    python: Path,
    merge: bool,
) -> dict[str, Any] | None:
    return _prepare_merged_datasets(config, repo_root=repo_root, python=python, merge=merge)


def _resolve_root_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _snapshot_download(**kwargs: Any) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Dataset config requires Hugging Face download support. Install `huggingface_hub` "
            "or use --skip-hf-dataset-download only when the resolved roots already exist."
        ) from exc
    return str(snapshot_download(**kwargs))


def _hf_local_repo_dir(cache_root: Path, repo_id: str, revision: str = "") -> Path:
    safe_repo = repo_id.replace("/", "__")
    if revision:
        return cache_root / safe_repo / revision.replace("/", "__")
    return cache_root / safe_repo


def _with_dataset_config(args: list[str], config: dict[str, Any] | None, *, runtime_platform: str = "auto") -> list[str]:
    if not config:
        return args
    train_datasets = _train_dataset_entries(config)
    train = _required_mapping(config, "train_dataset") if not train_datasets else _required_train_dataset_entry(train_datasets[0], 0)
    validation = config.get("validation_dataset") or {}
    if not isinstance(validation, dict):
        raise SystemExit("dataset config validation_dataset must be an object")

    updated = [*args]
    updated = _ensure_arg(updated, "dataset.repo_id", str(train["repo_id"]))
    updated = _ensure_arg(updated, "dataset.root", str(train["root"]))
    if train_datasets:
        updated = _ensure_arg(updated, "train-datasets-json", json.dumps(train_datasets, sort_keys=True))
    else:
        train_sources = train.get("hf_resolved_sources")
        if isinstance(train_sources, list) and train_sources:
            train_source_spans = _dataset_source_spans(train_sources)
        else:
            train_source_spans = []
        if train_source_spans:
            updated = _ensure_arg(
                updated,
                "train-dataset-source-spans-json",
                json.dumps(train_source_spans, sort_keys=True),
            )
    if validation:
        if "repo_id" in validation:
            updated = _ensure_arg(updated, "validation-dataset-repo-id", str(validation["repo_id"]))
        if "root" in validation:
            updated = _ensure_arg(updated, "validation-dataset-root", str(validation["root"]))
        validation_sources = validation.get("hf_resolved_sources")
        if isinstance(validation_sources, list) and validation_sources:
            updated = _ensure_arg(
                updated,
                "validation-datasets-json",
                json.dumps(_with_validation_dataset_cache_dirs(config, validation_sources), sort_keys=True),
            )
    training = config.get("training") or {}
    if not isinstance(training, dict):
        raise SystemExit("dataset config training must be an object")
    for name, cli_name in (
        ("num_workers", "num_workers"),
        ("batch_size", "batch_size"),
        ("policy_repo_id", "policy.path"),
        ("lightning_precision", "lightning-precision"),
        ("checkpoint_retention_policy", "checkpoint-retention-policy"),
    ):
        if name in training:
            value = training[name]
            if name == "num_workers" and runtime_platform == "macos":
                value = 0
            updated = _ensure_arg(updated, cli_name, str(value))
    if "policy_push_to_hub" in training:
        updated = _ensure_arg(updated, "policy.push_to_hub", str(bool(training["policy_push_to_hub"])).lower())
    visual_servo = config.get("visual_servo") or {}
    if not isinstance(visual_servo, dict):
        raise SystemExit("dataset config visual_servo must be an object")
    if "loss_weight" in visual_servo:
        updated = _ensure_arg(updated, "so101-visual-servo-loss-weight", str(visual_servo["loss_weight"]))
    if "hidden_dim" in visual_servo:
        updated = _ensure_arg(updated, "so101-visual-servo-hidden-dim", str(visual_servo["hidden_dim"]))
    action_chunk_consistency = config.get("action_chunk_consistency") or {}
    if not isinstance(action_chunk_consistency, dict):
        raise SystemExit("dataset config action_chunk_consistency must be an object")
    if "steps" in action_chunk_consistency:
        updated = _ensure_arg(
            updated,
            "so101-action-chunk-consistency-steps",
            str(action_chunk_consistency["steps"]),
        )
    if "weight" in action_chunk_consistency:
        updated = _ensure_arg(
            updated,
            "so101-action-chunk-consistency-weight",
            str(action_chunk_consistency["weight"]),
        )
    action_smoothness = config.get("action_smoothness") or {}
    if not isinstance(action_smoothness, dict):
        raise SystemExit("dataset config action_smoothness must be an object")
    if "weight" in action_smoothness:
        updated = _ensure_arg(updated, "so101-action-smoothness-loss-weight", str(action_smoothness["weight"]))
    if "include_gripper" in action_smoothness:
        updated = _ensure_boolean_optional_arg(
            updated,
            "so101-action-smoothness-include-gripper",
            value=bool(action_smoothness["include_gripper"]),
        )
    action_teacher_importance = config.get("action_teacher_importance") or {}
    if not isinstance(action_teacher_importance, dict):
        raise SystemExit("dataset config action_teacher_importance must be an object")
    for name, cli_name in (
        ("delta_weight", "so101-action-delta-loss-weight"),
        ("gripper_transition_weight", "so101-action-gripper-transition-loss-weight"),
        ("terminal_steps", "so101-action-terminal-loss-steps"),
        ("terminal_weight", "so101-action-terminal-loss-weight"),
    ):
        if name in action_teacher_importance:
            updated = _ensure_arg(updated, cli_name, str(action_teacher_importance[name]))
    cache = config.get("predecoded_image_cache") or {}
    if not isinstance(cache, dict):
        raise SystemExit("dataset config predecoded_image_cache must be an object")
    for name, cli_name in (
        ("train", "so101-image-cache-dir"),
        ("validation", "validation-image-cache-dir"),
    ):
        if name == "train" and train_datasets:
            continue
        if name in cache and cache.get(name) not in (None, False, {}):
            updated = _ensure_arg(updated, cli_name, str(_resolve_cache_dir(cache, name)))
    if train.get("grid_bin_sidecar"):
        updated = _ensure_arg(updated, "train-grid-bin-sidecar", str(train["grid_bin_sidecar"]))
    tensorboard = config.get("tensorboard") or {}
    if not isinstance(tensorboard, dict):
        raise SystemExit("dataset config tensorboard must be an object")
    for name, cli_name in (
        ("log_input_images_every_n_steps", "log-input-images-every-n-steps"),
        ("log_input_metadata_every_n_steps", "log-input-metadata-every-n-steps"),
    ):
        if name in tensorboard:
            updated = _ensure_arg(updated, cli_name, str(tensorboard[name]))
    augmentation = config.get("augmentation") or {}
    if not isinstance(augmentation, dict):
        raise SystemExit("dataset config augmentation must be an object")
    for name, cli_name in (
        ("state_jitter_std", "so101-state-jitter-std"),
        ("state_dropout_prob", "so101-state-dropout-prob"),
        ("image_camera_dropout_prob", "so101-image-camera-dropout-prob"),
        ("image_patch_dropout_prob", "so101-image-patch-dropout-prob"),
        ("image_patch_mask_ratio", "so101-image-patch-mask-ratio"),
        ("image_blur_prob", "so101-image-blur-prob"),
        ("image_blur_kernel_size", "so101-image-blur-kernel-size"),
        ("image_motion_blur_prob", "so101-image-motion-blur-prob"),
        ("image_motion_blur_kernel_size", "so101-image-motion-blur-kernel-size"),
        ("image_noise_std", "so101-image-noise-std"),
        ("image_color_jitter_strength", "so101-image-color-jitter-strength"),
        ("image_affine_degrees", "so101-image-affine-degrees"),
        ("image_affine_translate", "so101-image-affine-translate"),
    ):
        if name in augmentation:
            updated = _ensure_arg(updated, cli_name, str(augmentation[name]))
    for name, cli_name in (
        ("state_jitter_arm_only", "so101-state-jitter-arm-only"),
        ("state_dropout_keep_gripper", "so101-state-dropout-keep-gripper"),
        ("image_color_jitter", "so101-image-color-jitter"),
        ("image_sharpness_jitter", "so101-image-sharpness-jitter"),
        ("gpu_image_augmentation", "so101-gpu-image-augmentation"),
    ):
        if name in augmentation:
            updated = _ensure_boolean_optional_arg(updated, cli_name, value=bool(augmentation[name]))
    return updated


def _ensure_train_grid_bin_sidecars(config: dict[str, Any] | None, *, repo_root: Path, build: bool) -> None:
    if not config:
        return
    train_datasets = config.get("train_datasets")
    if isinstance(train_datasets, list) and train_datasets:
        for index, entry in enumerate(train_datasets):
            if not isinstance(entry, dict):
                raise SystemExit(f"dataset config train_datasets[{index}] must be an object")
            _ensure_train_grid_bin_sidecar(entry, repo_root=repo_root, build=build)
        return
    train = config.get("train_dataset")
    if not isinstance(train, dict):
        raise SystemExit("dataset config train_dataset must be an object")
    _ensure_train_grid_bin_sidecar(train, repo_root=repo_root, build=build)


def _ensure_train_grid_bin_sidecar(entry: dict[str, Any], *, repo_root: Path, build: bool) -> None:
    if not entry.get("root"):
        raise SystemExit("train dataset entry must include root before grid-bin balancing")
    sidecar = entry.get("grid_bin_sidecar")
    if sidecar:
        sidecar_path = _resolve_root_path(repo_root, Path(str(sidecar)))
    else:
        root = Path(str(entry["root"]))
        sidecar_path = (
            _resolve_root_path(repo_root, root)
            / "meta"
            / "camera_grid_bins"
            / "observation_images_camera1_4x4_frame0.parquet"
        )
        entry["grid_bin_sidecar"] = _relative_or_absolute(repo_root, sidecar_path)
    if sidecar_path.exists():
        return
    if not build:
        return
    from scripts.build_so101_camera_grid_bins import build_bins

    build_bins(
        dataset_root=_resolve_root_path(repo_root, Path(str(entry["root"]))),
        camera_key="observation.images.camera1",
        grid_size=4,
        frame_index=0,
        min_area=20,
    )
    if not sidecar_path.exists():
        raise SystemExit(f"failed to build required grid-bin sidecar: {sidecar_path}")


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _train_dataset_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("train_datasets")
    if entries is None:
        return []
    if not isinstance(entries, list) or not entries:
        raise SystemExit("dataset config train_datasets must be a non-empty list when provided")
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"dataset config train_datasets[{index}] must be an object")
        required = _required_train_dataset_entry(entry, index)
        result.append(required)
    return _with_train_dataset_cache_dirs(config, result)


def _required_train_dataset_entry(entry: dict[str, Any], index: int) -> dict[str, Any]:
    missing = [name for name in ("repo_id", "root") if not entry.get(name)]
    if missing:
        raise SystemExit(f"dataset config train_datasets[{index}] missing keys: {', '.join(missing)}")
    return dict(entry)


def _with_train_dataset_cache_dirs(config: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache = config.get("predecoded_image_cache") or {}
    if not isinstance(cache, dict):
        raise SystemExit("dataset config predecoded_image_cache must be an object")
    if not cache:
        return entries
    train_cache = cache.get("train")
    if train_cache is None or train_cache is False or (isinstance(train_cache, dict) and not train_cache):
        return [dict(entry) for entry in entries]
    if train_cache is None and not cache.get("default_root") and not cache.get("root_env"):
        return entries
    updated = []
    for index, entry in enumerate(entries):
        item = dict(entry)
        if "image_cache_dir" not in item:
            cache_name = None
            if isinstance(train_cache, dict):
                cache_name = train_cache.get(str(item.get("name") or ""))
            if not cache_name:
                cache_name = str(item.get("name") or Path(str(item["root"])).name)
            item["image_cache_dir"] = str(_resolve_cache_value(cache, str(cache_name)))
        updated.append(item)
    return updated


def _with_validation_dataset_cache_dirs(config: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache = config.get("predecoded_image_cache") or {}
    if not isinstance(cache, dict) or not cache:
        return [dict(entry) for entry in entries]
    validation_cache = cache.get("validation")
    if not isinstance(validation_cache, dict):
        return [dict(entry) for entry in entries]
    updated = []
    for entry in entries:
        item = dict(entry)
        if "image_cache_dir" not in item:
            cache_name = None
            cache_name = validation_cache.get(str(item.get("name") or ""))
            if not cache_name:
                updated.append(item)
                continue
            item["image_cache_dir"] = str(_resolve_cache_value(cache, str(cache_name)))
        updated.append(item)
    return updated


def _dataset_source_spans(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise SystemExit(f"dataset source span {index} must be an object")
        name = source.get("name") or source.get("repo_id") or f"source_{index}"
        length = source.get("expected_frames")
        if length is None:
            root = source.get("root")
            if root:
                info_path = Path(str(root)) / "meta" / "info.json"
                info = _read_json(info_path) or {}
                length = info.get("total_frames")
        if length is None:
            return []
        spans.append({"name": str(name), "length": int(length)})
    return spans


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"dataset config {key} must be an object")
    missing = [name for name in ("repo_id", "root") if name not in value]
    if missing:
        raise SystemExit(f"dataset config {key} missing keys: {', '.join(missing)}")
    return value


def _with_validation_schedule(args: list[str], namespace: argparse.Namespace) -> list[str]:
    if _has_any_arg(
        args,
        "validation-interval-steps",
        "validation-interval-epochs",
        "validation-every-n-train-steps",
    ):
        return args
    if namespace.validation_interval_steps is not None:
        return [*args, f"--validation-interval-steps={int(namespace.validation_interval_steps)}"]
    if _closed_loop_policy_name(namespace) != "off":
        return args
    if namespace.validation_interval_epochs is not None:
        return [*args, f"--validation-interval-epochs={int(namespace.validation_interval_epochs)}"]
    return args


def _with_checkpoint_schedule(
    args: list[str],
    config: dict[str, Any] | None,
    namespace: argparse.Namespace,
) -> list[str]:
    if not config or _has_any_arg(args, "save_freq", "save-freq"):
        return args
    if _closed_loop_policy_name(namespace) == "off":
        return args
    steps_per_epoch = _steps_per_epoch(config, args)
    if steps_per_epoch is None:
        return args
    checkpoint_interval = steps_per_epoch * max(1, int(namespace.closed_loop_every_epochs))
    return [*args, f"--save_freq={checkpoint_interval}"]


def _with_aligned_validation_schedule(
    args: list[str],
    config: dict[str, Any] | None,
    namespace: argparse.Namespace,
) -> list[str]:
    if _has_any_arg(args, "validation-interval-steps", "validation-every-n-train-steps"):
        return args
    if _closed_loop_policy_name(namespace) == "off":
        return args
    save_freq = _positive_int_arg(args, "save_freq") or _positive_int_arg(args, "save-freq")
    if save_freq is None:
        steps_per_epoch = _steps_per_epoch(config or {}, args) if config else None
        if steps_per_epoch is None:
            return args
        save_freq = steps_per_epoch * max(1, int(namespace.closed_loop_every_epochs))
    return [*args, f"--validation-interval-steps={int(save_freq)}"]


def _runtime_contract(namespace: argparse.Namespace, training_args: list[str]) -> dict[str, str]:
    runtime_platform = _runtime_platform(namespace.runtime_platform)
    device = _arg_value(training_args, "policy.device")
    if device not in {"cpu", "mps", "cuda"}:
        device = namespace.training_device
    if device == "auto":
        device = "mps" if runtime_platform == "macos" else "cuda"
    lightning_accelerator = _arg_value(training_args, "lightning-accelerator")
    if not lightning_accelerator or lightning_accelerator == "auto":
        lightning_accelerator = {"mps": "mps", "cuda": "cuda", "cpu": "cpu"}[device]
    lightning_devices = _arg_value(training_args, "lightning-devices") or ("1" if device in {"mps", "cuda"} else "auto")
    mujoco_gl = namespace.closed_loop_mujoco_gl
    if mujoco_gl == "auto":
        mujoco_gl = "glfw" if runtime_platform == "macos" else "egl"
    return {
        "runtime_platform": runtime_platform,
        "training_device": device,
        "lightning_accelerator": lightning_accelerator,
        "lightning_devices": lightning_devices,
        "closed_loop_mujoco_gl": mujoco_gl,
    }


def _local_training_standard(repo_root: Path) -> dict[str, Any]:
    doc = repo_root / LOCAL_TRAINING_STANDARD_DOC
    return {
        "name": LOCAL_TRAINING_STANDARD_NAME,
        "doc": str(doc),
        "summary": [
            "Local SO101/SmolVLA training launches outside the Codex sandbox.",
            "macOS local runtime uses MPS through --runtime-platform macos.",
            "Use dataset-config hf_merge_sources virtual merge declarations.",
            "For macOS local training, prefer --num_workers=0 unless multiprocessing is proven safe.",
            "For Qwen validation v1, scenario=pick_up_cube and execution_policy=qwen_edge_chain.",
        ],
    }


def _runtime_platform(requested: str) -> str:
    if requested in {"macos", "linux"}:
        return requested
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return "linux"


def _with_runtime_contract(args: list[str], contract: dict[str, str]) -> list[str]:
    updated = [*args]
    updated = _ensure_arg(updated, "policy.device", contract["training_device"])
    updated = _ensure_arg(updated, "lightning-accelerator", contract["lightning_accelerator"])
    updated = _ensure_arg(updated, "lightning-devices", contract["lightning_devices"])
    return updated


def _validate_monitoring_contract(
    *,
    args: argparse.Namespace,
    dataset_config: dict[str, Any] | None,
    training_args: list[str],
    train_cmd: list[str],
    launch_plan: dict[str, Any],
    runtime_contract: dict[str, str],
) -> None:
    if not dataset_config or args.allow_incomplete_monitoring:
        return

    errors: list[str] = []
    validation = dataset_config.get("validation_dataset")
    has_virtual_validation = _has_virtual_validation_sources(dataset_config)
    if not isinstance(validation, dict):
        if not has_virtual_validation:
            errors.append("dataset config must define validation_dataset for val/loss.")
    else:
        if not has_virtual_validation:
            for key in ("repo_id", "root"):
                if not validation.get(key):
                    errors.append(f"validation_dataset.{key} is required for val/loss.")

    if args.no_tensorboard:
        errors.append("TensorBoard is disabled; remove --no-tensorboard for monitored training.")
    if launch_plan.get("tensorboard_cmd") is None:
        errors.append("TensorBoard command is missing.")
    if "--tensorboard-log-dir" not in train_cmd:
        errors.append("training command is missing --tensorboard-log-dir.")
    if not _has_any_arg(training_args, "policy.device"):
        errors.append("training command is missing --policy.device for platform-specific execution.")
    if not _has_any_arg(training_args, "lightning-accelerator"):
        errors.append("training command is missing --lightning-accelerator for platform-specific execution.")
    if runtime_contract["runtime_platform"] == "macos" and runtime_contract["training_device"] == "cuda":
        errors.append("macOS runtime profile cannot default to CUDA; use --training-device mps or cpu.")
    if runtime_contract["runtime_platform"] == "linux" and runtime_contract["training_device"] == "mps":
        errors.append("Linux/RunPod runtime profile cannot use MPS; use --training-device cuda or cpu.")

    validation_interval_steps = _validation_interval_steps(dataset_config, training_args)
    if validation_interval_steps <= 0:
        errors.append("validation cadence must be positive; set --validation-interval-epochs or --validation-interval-steps.")
    if not has_virtual_validation and not _has_any_arg(training_args, "validation-dataset-root"):
        errors.append("training command is missing --validation-dataset-root.")
    if not has_virtual_validation and not _has_any_arg(training_args, "validation-dataset-repo-id"):
        errors.append("training command is missing --validation-dataset-repo-id.")

    steps_per_epoch = _steps_per_epoch(dataset_config, training_args)
    save_freq = _positive_int_arg(training_args, "save_freq") or _positive_int_arg(training_args, "save-freq")
    retention_policy = _arg_value(training_args, "checkpoint-retention-policy")
    strict_checkpoint_retention = retention_policy == "best_val_and_closed_loop"
    if save_freq is not None and not strict_checkpoint_retention:
        planned_steps = _positive_int_arg(training_args, "steps")
        max_checkpoints = max(1, int(args.max_monitored_checkpoints))
        if planned_steps is not None:
            planned_checkpoints = (planned_steps + save_freq - 1) // save_freq
            if planned_checkpoints > max_checkpoints:
                errors.append(
                    f"--steps={planned_steps} and --save_freq={save_freq} would create "
                    f"about {planned_checkpoints} checkpoints; expected <= {max_checkpoints}. "
                    "Increase --save_freq, lower --steps, or raise --max-monitored-checkpoints "
                    "only when disk capacity is confirmed."
                )

    loop_test_enabled = (
        launch_plan.get("post_checkpoint_loop_cmd") is not None
        or bool(launch_plan.get("post_checkpoint_loop_cmds"))
        or launch_plan.get("progress_monitor_cmd") is not None
    )
    if not loop_test_enabled:
        if errors:
            detail = "\n".join(f"- {error}" for error in errors)
            raise SystemExit(f"SO101 monitored training contract failed:\n{detail}")
        return

    closed_loop_policy = _closed_loop_policy_name(args)
    if closed_loop_policy == "off":
        errors.append("closed-loop evaluation is disabled; use periodic, best_only, or best_or_periodic.")
    if closed_loop_policy == "best_only":
        errors.append(
            "closed-loop policy best_only can skip validation checkpoints; use periodic or best_or_periodic "
            "so every validation-loss checkpoint also runs closed-loop."
        )
    progress_monitor_cmd = launch_plan.get("progress_monitor_cmd") or []
    post_checkpoint_loop_cmds = launch_plan.get("post_checkpoint_loop_cmds") or []
    loop_test_cmd = (
        post_checkpoint_loop_cmds[0]
        if post_checkpoint_loop_cmds and isinstance(post_checkpoint_loop_cmds[0], list)
        else launch_plan.get("post_checkpoint_loop_cmd")
        or progress_monitor_cmd
    )
    if "--mujoco-gl" not in loop_test_cmd:
        errors.append("loop test command is missing --mujoco-gl for platform-specific closed-loop rendering.")
    if int(args.closed_loop_every_epochs) <= 0:
        errors.append("--closed-loop-every-epochs must be positive.")
    if int(args.closed_loop_episodes) <= 0:
        errors.append("--closed-loop-episodes must be positive.")
    if int(args.closed_loop_steps) <= 0:
        errors.append("--closed-loop-steps must be positive.")

    closed_loop = dataset_config.get("closed_loop") or {}
    closed_loop_runner = _closed_loop_runner(args, dataset_config)
    _validate_closed_loop_test_case_commands(
        dataset_config=dataset_config,
        post_checkpoint_loop_cmds=post_checkpoint_loop_cmds,
        errors=errors,
    )
    if closed_loop_runner != "qwen_chain" and not args.closed_loop_eval_skill_mode and not (
        isinstance(closed_loop, dict) and closed_loop.get("eval_skill_mode")
    ):
        errors.append("closed-loop eval skill mode must be set in config closed_loop.eval_skill_mode or CLI.")
    if not _closed_loop_task_prompt(args, dataset_config):
        errors.append("closed-loop task prompt must be set in config closed_loop.task_prompt or CLI.")
    action_contract_mode = _closed_loop_action_contract_mode(args, dataset_config)
    if (
        closed_loop_runner == "qwen_chain"
        and action_contract_mode not in {"visual_servo_delta_q", "visual_servo_gt_delta_q"}
        and _closed_loop_valid_mask_checkpoint(args, dataset_config) is None
    ):
        errors.append(
            "qwen_chain loop tests require closed_loop.valid_mask_checkpoint or "
            "--closed-loop-valid-mask-checkpoint."
        )

    if steps_per_epoch is None:
        errors.append("training.steps_per_epoch or --steps-per-epoch is required for closed-loop scheduling.")
    if save_freq is None:
        errors.append("checkpoint save cadence is missing; set --save_freq or training.steps_per_epoch.")
    elif steps_per_epoch is not None:
        closed_loop_gap = steps_per_epoch * int(args.closed_loop_every_epochs)
        if save_freq > closed_loop_gap:
            errors.append(
                f"--save_freq={save_freq} is too sparse for closed-loop cadence; "
                f"expected <= {closed_loop_gap}."
            )
        if validation_interval_steps != save_freq:
            errors.append(
                f"validation cadence ({validation_interval_steps} steps) must match checkpoint save cadence "
                f"({save_freq} steps) so every validation loss has a checkpoint for closed-loop evaluation."
            )
        if save_freq != closed_loop_gap:
            errors.append(
                f"checkpoint save cadence ({save_freq} steps) must match closed-loop cadence "
                f"({closed_loop_gap} steps) so every validation-loss checkpoint runs closed-loop."
            )
        planned_steps = _positive_int_arg(training_args, "steps")
        if planned_steps is not None and planned_steps % save_freq != 0:
            errors.append(
                f"training steps ({planned_steps}) must be divisible by checkpoint save cadence "
                f"({save_freq}) so the final checkpoint also has validation and loop-test results."
            )

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"SO101 monitored training contract failed:\n{detail}")


def _validate_closed_loop_test_case_commands(
    *,
    dataset_config: dict[str, Any] | None,
    post_checkpoint_loop_cmds: list[list[str]],
    errors: list[str],
) -> None:
    test_cases = _closed_loop_test_cases(dataset_config)
    if not test_cases or not post_checkpoint_loop_cmds:
        return
    if len(post_checkpoint_loop_cmds) != len(test_cases):
        errors.append(
            f"closed-loop command/test-case count mismatch: commands={len(post_checkpoint_loop_cmds)} "
            f"test_cases={len(test_cases)}"
        )
        return
    for index, (test_case, command) in enumerate(zip(test_cases, post_checkpoint_loop_cmds, strict=True)):
        expected_id = str(test_case.get("id") or test_case.get("name") or "closed_loop")
        actual_id = _arg_value(command, "closed-loop-test-id")
        if actual_id != expected_id:
            errors.append(
                f"closed_loop.test_cases[{index}] id mismatch: command has {actual_id!r}, expected {expected_id!r}."
            )
        expected_report = _closed_loop_start_report_path(test_case)
        if expected_report:
            actual_report = _arg_value(command, "closed-loop-start-report-path")
            if actual_report != expected_report:
                errors.append(
                    f"closed_loop.test_cases[{index}] start report mismatch: command has {actual_report!r}, "
                    f"expected {expected_report!r}."
                )


def _has_virtual_validation_sources(dataset_config: dict[str, Any] | None) -> bool:
    if not dataset_config:
        return False
    validation = dataset_config.get("validation_dataset")
    if not isinstance(validation, dict):
        return False
    for key in ("hf_resolved_sources", "hf_merge_sources"):
        sources = validation.get(key)
        if isinstance(sources, list) and sources:
            return True
    return False


def _validation_interval_steps(config: dict[str, Any], args: list[str]) -> int:
    steps = _positive_int_arg(args, "validation-interval-steps") or _positive_int_arg(
        args, "validation-every-n-train-steps"
    )
    if steps is not None:
        return steps
    epochs = _positive_int_arg(args, "validation-interval-epochs")
    steps_per_epoch = _steps_per_epoch(config, args)
    if epochs is not None and steps_per_epoch is not None:
        return epochs * steps_per_epoch
    return 0


def _steps_per_epoch(config: dict[str, Any], training_args: list[str]) -> int | None:
    value = _positive_int_arg(training_args, "steps-per-epoch")
    if value is not None:
        return value
    training = config.get("training") or {}
    if isinstance(training, dict):
        try:
            parsed = int(training.get("steps_per_epoch") or 0)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
    return None


def _positive_int_arg(args: list[str], name: str) -> int | None:
    value = _arg_value(args, name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _closed_loop_policy_name(args: argparse.Namespace) -> str:
    value = getattr(args, "closed_loop_policy", None)
    if value:
        return str(value)
    raise SystemExit("closed_loop_policy must be set by the selected Hydra launcher config or CLI override.")


def _monitor_validation_dataset(dataset_config: dict[str, Any]) -> dict[str, str] | None:
    validation = dataset_config.get("validation_dataset")
    if isinstance(validation, dict):
        if "root" in validation and "repo_id" in validation:
            return {"root": str(validation["root"]), "repo_id": str(validation["repo_id"])}
        sources = validation.get("hf_resolved_sources")
        if isinstance(sources, list) and sources:
            for source in sources:
                if isinstance(source, dict) and "root" in source and "repo_id" in source:
                    return {"root": str(source["root"]), "repo_id": str(source["repo_id"])}
    train = dataset_config.get("train_dataset")
    if isinstance(train, dict) and "root" in train and "repo_id" in train:
        return {"root": str(train["root"]), "repo_id": str(train["repo_id"])}
    train_datasets = dataset_config.get("train_datasets")
    if isinstance(train_datasets, list) and train_datasets:
        first = train_datasets[0]
        if isinstance(first, dict) and "root" in first and "repo_id" in first:
            return {"root": str(first["root"]), "repo_id": str(first["repo_id"])}
    return None


def _progress_monitor_command(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_dir: Path,
    train_output_dir: Path,
    dataset_config: dict[str, Any] | None,
    training_args: list[str],
    runtime_contract: dict[str, str],
    train_pid_file: Path,
) -> list[str] | None:
    if not dataset_config:
        return None
    validation = _monitor_validation_dataset(dataset_config)
    if validation is None:
        return None
    training = dataset_config.get("training") or {}
    if not isinstance(training, dict):
        training = {}
    batch_size_value = (
        _arg_value(training_args, "batch_size")
        or training.get("batch_size")
        or _required_arg(args, "progress_monitor_batch_size")
    )
    steps_per_epoch_value = training.get("steps_per_epoch") or _arg_value(training_args, "steps_per_epoch")
    if steps_per_epoch_value is None:
        if getattr(args, "allow_incomplete_monitoring", False):
            return None
        raise SystemExit("training.steps_per_epoch or --steps_per_epoch is required for SO101 progress monitor commands.")
    validation_max_batches = training.get("validation_max_batches") or _required_arg(
        args,
        "progress_monitor_validation_max_batches",
    )
    batch_size = int(batch_size_value)
    steps_per_epoch = int(steps_per_epoch_value)
    policy_device = str(_arg_value(training_args, "policy.device") or _required_arg(args, "training_device"))
    if policy_device not in {"auto", "cpu", "mps", "cuda"}:
        policy_device = "auto"
    closed_loop_runner = _closed_loop_runner(args, dataset_config)
    cmd = [
        str(args.python),
        str(repo_root / "scripts" / "monitor_so101_training_dashboard.py"),
        "--run-dir",
        str(run_dir),
        "--interval-s",
        str(args.progress_monitor_interval_s),
        "--policy-device",
        policy_device,
        "--python",
        str(args.python),
        "--repo-root",
        str(repo_root),
        "--checkpoint-root",
        str(train_output_dir / "checkpoints"),
        "--dataset-root",
        str(validation["root"]),
        "--dataset-repo-id",
        str(validation["repo_id"]),
        "--batch-size",
        str(batch_size),
        "--max-batches",
        str(validation_max_batches),
        "--train-pid-file",
        str(train_pid_file),
        "--closed-loop-every-epochs",
        str(args.closed_loop_every_epochs),
        "--steps-per-epoch",
        str(steps_per_epoch),
        "--closed-loop-episodes",
        str(args.closed_loop_episodes),
        "--closed-loop-steps",
        str(args.closed_loop_steps),
        "--closed-loop-env-id",
        _closed_loop_env_id(args, dataset_config),
        "--mujoco-gl",
        runtime_contract["closed_loop_mujoco_gl"],
        "--closed-loop-runner",
        closed_loop_runner,
        "--closed-loop-policy",
        _required_arg(args, "closed_loop_policy"),
        "--closed-loop-eval-skill-mode",
        _closed_loop_eval_skill_mode(args, dataset_config),
        "--closed-loop-subgoal-chain-mode",
        _required_arg(args, "closed_loop_subgoal_chain_mode"),
        "--closed-loop-fixed-subgoal-chunks",
        str(_required_arg(args, "closed_loop_fixed_subgoal_chunks")),
        "--closed-loop-valid-mask-threshold",
        str(_required_arg(args, "closed_loop_valid_mask_threshold")),
        "--closed-loop-valid-mask-consecutive",
        str(_required_arg(args, "closed_loop_valid_mask_consecutive")),
        "--policy-n-action-steps",
        str(_required_arg(args, "closed_loop_policy_n_action_steps")),
        "--policy-num-steps",
        str(_required_arg(args, "closed_loop_policy_num_steps")),
        "--closed-loop-action-contract-mode",
        _closed_loop_action_contract_mode(args, dataset_config),
        "--local-files-only",
    ]
    action_rmse_sweep = ((dataset_config.get("closed_loop") or {}).get("action_rmse_sweep") or {}) if dataset_config else {}
    if isinstance(action_rmse_sweep, dict):
        if "enabled" in action_rmse_sweep:
            cmd.append("--closed-loop-action-rmse-sweep" if action_rmse_sweep.get("enabled") else "--no-closed-loop-action-rmse-sweep")
        n_action_steps = action_rmse_sweep.get("n_action_steps")
        if isinstance(n_action_steps, list) and n_action_steps:
            cmd.extend(
                [
                    "--closed-loop-action-rmse-sweep-n-action-steps",
                    ",".join(str(int(value)) for value in n_action_steps),
                ]
            )
    closed_loop_subgoal_sequence = getattr(args, "closed_loop_subgoal_sequence", None)
    if closed_loop_subgoal_sequence:
        cmd.extend(["--closed-loop-subgoal-sequence", str(closed_loop_subgoal_sequence)])
    closed_loop_valid_mask_checkpoint = _closed_loop_valid_mask_checkpoint(args, dataset_config)
    if closed_loop_valid_mask_checkpoint and _closed_loop_action_contract_mode(args, dataset_config) not in {
        "visual_servo_delta_q",
        "visual_servo_gt_delta_q",
    }:
        cmd.extend(["--closed-loop-valid-mask-checkpoint", str(closed_loop_valid_mask_checkpoint)])
    closed_loop_task_prompt = _closed_loop_task_prompt(args, dataset_config)
    if closed_loop_task_prompt:
        cmd.extend(["--closed-loop-task-prompt", closed_loop_task_prompt])
    closed_loop_env_object_color = _qwen_env_object_color(args, dataset_config)
    if closed_loop_env_object_color:
        cmd.extend(["--closed-loop-env-object-color", closed_loop_env_object_color])
    if args.closed_loop_record_rollout_gif or _closed_loop_record_rollout_gif(dataset_config):
        cmd.append("--closed-loop-record-rollout-gif")
    if args.record_loop_artifacts:
        cmd.extend(
            [
                "--record-loop-artifacts",
                "--render-loop-media" if _required_arg(args, "render_loop_media") else "--no-render-loop-media",
                "--loop-artifact-width",
                str(_required_arg(args, "loop_artifact_width")),
                "--loop-artifact-height",
                str(_required_arg(args, "loop_artifact_height")),
                "--loop-artifact-fps",
                str(_required_arg(args, "loop_artifact_fps")),
                "--loop-artifact-every-n-steps",
                str(_required_arg(args, "loop_artifact_every_n_steps")),
            ]
        )
    else:
        cmd.append("--no-record-loop-artifacts")
    if closed_loop_runner == "qwen_chain":
        cmd.extend(
            [
                "--qwen-model",
                _required_arg(args, "qwen_model"),
                "--qwen-object",
                _qwen_object(args, dataset_config),
                "--qwen-env-object-color",
                _qwen_env_object_color(args, dataset_config),
            ]
        )
        if args.qwen_plan_json:
            cmd.extend(["--qwen-plan-json", str(args.qwen_plan_json)])
        elif args.qwen_response_json:
            cmd.extend(["--qwen-response-json", str(args.qwen_response_json)])
        else:
            qwen_response_json = _qwen_response_json(dataset_config)
            if qwen_response_json:
                cmd.extend(["--qwen-response-json", str(qwen_response_json)])
            elif args.qwen_base_url:
                cmd.extend(["--qwen-base-url", args.qwen_base_url])
        if args.qwen_api_key:
            cmd.extend(["--qwen-api-key", args.qwen_api_key])
    return cmd


def _post_checkpoint_loop_command(progress_monitor_cmd: list[str] | None) -> list[str] | None:
    if not progress_monitor_cmd:
        return None
    cmd = [*progress_monitor_cmd]
    for index, part in enumerate(cmd):
        if part.endswith("monitor_so101_training_dashboard.py"):
            cmd[index] = str(Path(part).with_name("run_so101_training_loop_test.py"))
            break
    cmd.extend(["--iterations", "1"])
    return cmd


def _post_checkpoint_loop_commands(
    *,
    progress_monitor_cmd: list[str] | None,
    dataset_config: dict[str, Any] | None,
) -> list[list[str]]:
    base = _post_checkpoint_loop_command(progress_monitor_cmd)
    if not base:
        return []
    test_cases = _closed_loop_test_cases(dataset_config)
    if not test_cases:
        return [base]
    commands = []
    for test_case in test_cases:
        cmd = _apply_closed_loop_test_case(base, test_case)
        commands.append(cmd)
    return commands


def _closed_loop_test_cases(dataset_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not dataset_config:
        return []
    closed_loop = dataset_config.get("closed_loop") or {}
    if not isinstance(closed_loop, dict):
        return []
    test_cases = closed_loop.get("test_cases")
    source_key = "test_cases"
    if not isinstance(test_cases, list):
        test_cases = closed_loop.get("suites")
        source_key = "suites"
    if not isinstance(test_cases, list):
        return []
    result = []
    for index, test_case in enumerate(test_cases):
        if not isinstance(test_case, dict):
            raise SystemExit(f"closed_loop.{source_key}[{index}] must be an object")
        result.append(dict(test_case))
    return result


_loop_validation_cases = _closed_loop_test_cases


def _apply_closed_loop_test_case(base: list[str], test_case: dict[str, Any]) -> list[str]:
    cmd = [*base]
    test_id = str(test_case.get("id") or test_case.get("name") or "closed_loop")
    cmd = _replace_or_append_arg(cmd, "--closed-loop-test-id", test_id)
    if "episodes" in test_case:
        cmd = _replace_or_append_arg(cmd, "--closed-loop-episodes", str(int(test_case["episodes"])))
    if "steps" in test_case:
        cmd = _replace_or_append_arg(cmd, "--closed-loop-steps", str(int(test_case["steps"])))
    if "seed" in test_case:
        cmd = _replace_or_append_arg(cmd, "--closed-loop-seed", str(int(test_case["seed"])))
    if test_case.get("success_metric"):
        cmd = _replace_or_append_arg(cmd, "--closed-loop-success-metric", str(test_case["success_metric"]))
    if test_case.get("success_threshold") is not None:
        cmd = _replace_or_append_arg(cmd, "--closed-loop-success-threshold", str(float(test_case["success_threshold"])))
    if test_case.get("start_contract"):
        cmd = _replace_or_append_arg(cmd, "--closed-loop-start-contract", str(test_case["start_contract"]))
    start_report_path = _closed_loop_start_report_path(test_case)
    if start_report_path:
        cmd = _replace_or_append_arg(cmd, "--closed-loop-start-report-path", start_report_path)
    if test_case.get("task_prompt"):
        cmd = _replace_or_append_arg(cmd, "--closed-loop-task-prompt", str(test_case["task_prompt"]))
    if test_case.get("qwen_object"):
        cmd = _replace_or_append_arg(cmd, "--qwen-object", str(test_case["qwen_object"]))
    if test_case.get("env_object_color"):
        cmd = _replace_or_append_arg(cmd, "--closed-loop-env-object-color", str(test_case["env_object_color"]))
        cmd = _replace_or_append_arg(cmd, "--qwen-env-object-color", str(test_case["env_object_color"]))
    if test_case.get("plan_json"):
        cmd = _remove_arg_with_value(cmd, "--qwen-response-json")
        cmd = _replace_or_append_arg(cmd, "--qwen-plan-json", str(test_case["plan_json"]))
    elif test_case.get("qwen_response_json"):
        cmd = _remove_arg_with_value(cmd, "--qwen-plan-json")
        cmd = _replace_or_append_arg(cmd, "--qwen-response-json", str(test_case["qwen_response_json"]))
    if test_case.get("precondition_plan_json"):
        cmd = _replace_or_append_arg(cmd, "--closed-loop-precondition-plan-json", str(test_case["precondition_plan_json"]))
    else:
        cmd = _remove_arg_with_value(cmd, "--closed-loop-precondition-plan-json")
    return cmd


def _closed_loop_start_report_path(test_case: dict[str, Any]) -> str | None:
    if test_case.get("start_report_path"):
        return str(test_case["start_report_path"])
    dataset = test_case.get("start_dataset")
    if isinstance(dataset, dict) and dataset.get("root"):
        return str(Path(str(dataset["root"])) / "so101_lerobot_export_report.json")
    return None


def _replace_or_append_arg(cmd: list[str], flag: str, value: str) -> list[str]:
    updated = [*cmd]
    prefix = f"{flag}="
    for index, part in enumerate(updated):
        if part == flag:
            if index + 1 < len(updated):
                updated[index + 1] = value
                return updated
            updated.append(value)
            return updated
        if part.startswith(prefix):
            updated[index] = f"{flag}={value}"
            return updated
    updated.extend([flag, value])
    return updated


def _remove_arg_with_value(cmd: list[str], flag: str) -> list[str]:
    updated: list[str] = []
    skip_next = False
    prefix = f"{flag}="
    for part in cmd:
        if skip_next:
            skip_next = False
            continue
        if part == flag:
            skip_next = True
            continue
        if part.startswith(prefix):
            continue
        updated.append(part)
    return updated


def _closed_loop_eval_skill_mode(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str:
    if args.closed_loop_eval_skill_mode:
        return str(args.closed_loop_eval_skill_mode)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("eval_skill_mode"):
        return str(closed_loop["eval_skill_mode"])
    raise SystemExit("closed_loop.eval_skill_mode or --closed-loop-eval-skill-mode is required.")


def _closed_loop_runner(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str:
    if args.closed_loop_runner != "auto":
        return str(args.closed_loop_runner)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict):
        if closed_loop.get("runner"):
            return str(closed_loop["runner"])
        if closed_loop.get("execution_policy") == "qwen_edge_chain":
            return "qwen_chain"
    if dataset_config.get("execution_policy") == "qwen_edge_chain":
        return "qwen_chain"
    raise SystemExit("closed_loop.runner or --closed-loop-runner is required.")


def _closed_loop_env_id(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str:
    if args.closed_loop_env_id:
        return str(args.closed_loop_env_id)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("env_id"):
        return str(closed_loop["env_id"])
    raise SystemExit("closed_loop.env_id or --closed-loop-env-id is required.")


def _qwen_object(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str:
    if getattr(args, "qwen_object", None):
        return str(args.qwen_object)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("qwen_object"):
        return str(closed_loop["qwen_object"])
    raise SystemExit("closed_loop.qwen_object or --qwen-object is required for qwen_chain.")


def _qwen_env_object_color(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str | None:
    if getattr(args, "qwen_env_object_color", None):
        return str(args.qwen_env_object_color)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("env_object_color"):
        return str(closed_loop["env_object_color"])
    return None


def _closed_loop_valid_mask_checkpoint(args: argparse.Namespace, dataset_config: dict[str, Any]) -> Path | None:
    value = getattr(args, "closed_loop_valid_mask_checkpoint", None)
    if value:
        return Path(value)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("valid_mask_checkpoint"):
        return Path(str(closed_loop["valid_mask_checkpoint"]))
    return None


def _closed_loop_action_contract_mode(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str:
    value = getattr(args, "closed_loop_action_contract_mode", None)
    if value:
        return str(value)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("action_contract_mode"):
        return str(closed_loop["action_contract_mode"])
    raise SystemExit("closed_loop.action_contract_mode or --closed-loop-action-contract-mode is required.")


def _qwen_response_json(dataset_config: dict[str, Any]) -> Path | None:
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("qwen_response_json"):
        return Path(str(closed_loop["qwen_response_json"]))
    return None


def _required_arg(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None:
        raise SystemExit(f"{name} must be set by the selected Hydra launcher config or CLI override.")
    return value


def _closed_loop_task_prompt(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str | None:
    if args.closed_loop_task_prompt:
        return str(args.closed_loop_task_prompt)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("task_prompt"):
        return str(closed_loop["task_prompt"])
    return None


def _closed_loop_record_rollout_gif(dataset_config: dict[str, Any]) -> bool:
    closed_loop = dataset_config.get("closed_loop") or {}
    return bool(isinstance(closed_loop, dict) and closed_loop.get("record_rollout_gif"))


def _arg_value(args: list[str], name: str) -> str | None:
    prefix = f"--{name}="
    spaced = f"--{name}"
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == spaced and index + 1 < len(args):
            return args[index + 1]
    return None


def _has_any_arg(args: list[str], *names: str) -> bool:
    prefixes = tuple(f"--{name}=" for name in names)
    spaced = {f"--{name}" for name in names}
    return any(arg.startswith(prefixes) or arg in spaced for arg in args)


def _cache_build_commands(
    python: Path,
    repo_root: Path,
    config: dict[str, Any] | None,
) -> list[list[str]]:
    if not config:
        return []
    cache = config.get("predecoded_image_cache") or {}
    if not isinstance(cache, dict):
        return []
    commands: list[list[str]] = []
    train_datasets = _train_dataset_entries(config) if config.get("train_datasets") is not None else []
    if train_datasets:
        for dataset in train_datasets:
            if "image_cache_dir" in dataset:
                commands.append(_cache_build_command(python, repo_root, dataset, Path(str(dataset["image_cache_dir"]))))
    else:
        dataset = config.get("train_dataset") or {}
        train_cache = cache.get("train")
        if (
            train_cache not in (None, False, {})
            and isinstance(dataset, dict)
            and "root" in dataset
            and "repo_id" in dataset
        ):
            commands.append(_cache_build_command(python, repo_root, dataset, _resolve_cache_dir(cache, "train")))
    for split, dataset_key in (("validation", "validation_dataset"),):
        dataset = config.get(dataset_key) or {}
        split_cache = cache.get(split)
        if split_cache in (None, False, {}) or not isinstance(dataset, dict):
            continue
        if "root" not in dataset or "repo_id" not in dataset:
            continue
        cache_dir = _resolve_cache_dir(cache, split)
        commands.append(_cache_build_command(python, repo_root, dataset, cache_dir))
    return commands


def _cache_build_command(python: Path, repo_root: Path, dataset: dict[str, Any], cache_dir: Path) -> list[str]:
    return [
        str(python),
        str(repo_root / "scripts" / "build_so101_predecoded_image_cache.py"),
        "--dataset-root",
        str(dataset["root"]),
        "--dataset-repo-id",
        str(dataset["repo_id"]),
        "--cache-dir",
        str(cache_dir),
    ]


def _resolve_cache_dir(cache: dict[str, Any], split: str) -> Path:
    value = cache.get(split)
    if value is None:
        raise SystemExit(f"dataset config predecoded_image_cache missing {split}")
    return _resolve_cache_value(cache, str(value))


def _resolve_cache_value(cache: dict[str, Any], value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return path
    root = None
    root_env = cache.get("root_env")
    if isinstance(root_env, str) and root_env:
        root = os.environ.get(root_env)
    if not root:
        root = str(cache.get("default_root") or "")
    if not root:
        return path
    return Path(root) / path


def _run_cache_builds(commands: list[list[str]], *, log_dir: Path, cwd: Path) -> None:
    for index, command in enumerate(commands):
        log_path = log_dir / f"cache_build_{index}.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] $ {' '.join(command)}\n")
            handle.flush()
            subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True, text=True)


def _tensorboard_executable(python: Path, repo_root: Path) -> list[str] | None:
    candidates = [
        python.resolve().parent / "tensorboard",
        repo_root / ".venv" / "bin" / "tensorboard",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    found = shutil.which("tensorboard")
    return [found] if found else None


def _tensorboard_tunnel_command(port: int) -> list[str] | None:
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        return None
    return [
        cloudflared,
        "tunnel",
        "--url",
        f"http://127.0.0.1:{int(port)}",
        "--no-autoupdate",
    ]


def _wait_for_tensorboard_tunnel_url(log_path: Path, *, timeout_s: float = 12.0) -> str | None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        url = _tensorboard_tunnel_url_from_log(log_path)
        if url:
            return url
        time.sleep(0.5)
    return _tensorboard_tunnel_url_from_log(log_path)


def _tensorboard_tunnel_url_from_log(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"https://[A-Za-z0-9.-]+\.trycloudflare\.com", text)
    return matches[-1] if matches else None


def _popen(cmd: list[str], log_path: Path, *, cwd: Path) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] $ {' '.join(cmd)}\n")
    handle.flush()
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def _process_status(pid: Any) -> dict[str, Any]:
    if pid is None:
        return {"alive": None, "pid": None}
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return {"alive": False, "pid": None}
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return {"alive": False, "pid": pid_int}
    except PermissionError:
        return {"alive": True, "pid": pid_int, "permission": "unknown"}
    return {"alive": True, "pid": pid_int}


def _terminate(pid: Any) -> None:
    _signal(pid, signal.SIGTERM)


def _kill(pid: Any) -> None:
    _signal(pid, signal.SIGKILL)


def _signal(pid: Any, sig: signal.Signals) -> None:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return
    try:
        os.killpg(pid_int, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid_int, sig)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
