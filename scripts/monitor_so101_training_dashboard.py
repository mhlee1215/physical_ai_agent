#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
    parser.add_argument("--closed-loop-episodes", type=int, default=24)
    parser.add_argument("--closed-loop-steps", type=int, default=160)
    parser.add_argument("--closed-loop-seed", type=int, default=98100)
    parser.add_argument("--closed-loop-width", type=int, default=256)
    parser.add_argument("--closed-loop-height", type=int, default=256)
    parser.add_argument(
        "--mujoco-gl",
        choices=["auto", "glfw", "egl", "osmesa"],
        default="auto",
        help="MuJoCo rendering backend for validation rollouts. auto uses glfw on macOS and egl on Linux.",
    )
    parser.add_argument("--closed-loop-task-prompt")
    parser.add_argument(
        "--closed-loop-eval-skill-mode",
        choices=["picklift", "pick_from_top_cube", "pick_and_place_cube"],
        default="picklift",
    )
    parser.add_argument("--closed-loop-record-rollout-gif", action="store_true")
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
                    f"checkpoint {checkpoint} success={closed_loop_report['success_rate']:.3f} "
                    f"grasp={closed_loop_report['grasp_rate']:.3f}"
                ),
                "checkpoint": checkpoint,
                "success_rate": closed_loop_report["success_rate"],
                "grasp_rate": closed_loop_report["grasp_rate"],
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
    output_dir = run_dir / "closed_loop_evals" / (
        f"val24_seed{args.closed_loop_seed}_nact{args.policy_n_action_steps}_{checkpoint}"
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
    if args.closed_loop_valid_mask_checkpoint:
        cmd.extend(["--valid-mask-checkpoint", str(args.closed_loop_valid_mask_checkpoint)])
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
    return json.loads(report_path.read_text(encoding="utf-8"))


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


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 2)


def _should_run_closed_loop(args: argparse.Namespace, run_dir: Path, checkpoint: str) -> bool:
    policy = _closed_loop_policy(args)
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
        closed_loop_rows=_read_jsonl(run_dir / "metrics" / "closed_loop_metrics.jsonl"),
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

    log_dir = run_dir / "tensorboard" / "so101_smolvla"
    step = int(row.get("step") or 0)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for key in ("success_rate", "grasp_rate", "episodes", "duration_s"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f"closed_loop/{key}", float(value), global_step=step)
                if key == "success_rate":
                    writer.add_scalar("important/closed_loop_success_rate", float(value), global_step=step)
        for camera_name, image_path in _first_closed_loop_input_grid_paths(report).items():
            image = _read_hwc_image(Path(image_path))
            if image is not None:
                writer.add_image(
                    f"closed_loop/input_{camera_name}_grid",
                    image,
                    global_step=step,
                    dataformats="HWC",
                )


def _first_closed_loop_input_grid_paths(report: dict[str, Any]) -> dict[str, str]:
    for episode in report.get("episodes") or []:
        paths = episode.get("input_grid_paths")
        if isinstance(paths, dict) and paths:
            return {str(key): str(value) for key, value in paths.items()}
    return {}


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


if __name__ == "__main__":
    main()
