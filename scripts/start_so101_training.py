#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("_workspace/so101_training")
DEFAULT_LOCK = DEFAULT_ROOT / "active_training.json"
DEFAULT_HF_DATASET_CACHE_ROOT = Path("_workspace/hf_datasets")
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
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_ROOT / "runs" / "latest_lightning")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        help="JSON file defining train/validation LeRobot datasets and training defaults.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--dashboard-port", type=int, default=8767)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--no-progress-monitor", action="store_true")
    parser.add_argument(
        "--allow-incomplete-monitoring",
        action="store_true",
        help=(
            "Debug escape hatch: allow dataset-config training to start without "
            "the default TensorBoard, validation, checkpoint, and closed-loop guards."
        ),
    )
    parser.add_argument(
        "--hf-dataset-cache-root",
        type=Path,
        default=DEFAULT_HF_DATASET_CACHE_ROOT,
        help="Local root for Hugging Face dataset subfolder downloads.",
    )
    parser.add_argument(
        "--skip-hf-dataset-download",
        action="store_true",
        help="Resolve configured HF cache roots without downloading. For debugging only.",
    )
    parser.add_argument(
        "--use-local-dataset-roots",
        action="store_true",
        help="Use dataset root fields from the config directly and ignore configured HF dataset sources.",
    )
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        help="Require configured HF dataset subfolders to already be present in the local HF cache root.",
    )
    parser.add_argument("--gpu-monitor-interval-s", type=float, default=5.0)
    parser.add_argument("--progress-monitor-interval-s", type=int, default=600)
    parser.add_argument(
        "--runtime-platform",
        choices=["auto", "macos", "linux"],
        default="auto",
        help="Runtime profile for training/eval defaults. auto detects the current OS.",
    )
    parser.add_argument(
        "--training-device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Default policy.device and Lightning accelerator when not explicitly forwarded.",
    )
    parser.add_argument("--closed-loop-every-epochs", type=int, default=1)
    parser.add_argument("--closed-loop-episodes", type=int, default=10)
    parser.add_argument("--closed-loop-steps", type=int, default=120)
    parser.add_argument(
        "--closed-loop-mujoco-gl",
        choices=["auto", "glfw", "egl", "osmesa"],
        default="auto",
        help="MuJoCo backend for closed-loop rollouts. auto uses glfw on macOS and egl on Linux.",
    )
    parser.add_argument(
        "--max-monitored-checkpoints",
        type=int,
        default=20,
        help="Fail fast when --steps/--save_freq would create more monitored checkpoints than this.",
    )
    parser.add_argument("--closed-loop-policy", choices=["off", "periodic", "best_only", "best_or_periodic"], default="periodic")
    parser.add_argument("--closed-loop-runner", choices=["auto", "picklift", "qwen_chain"], default="auto")
    parser.add_argument(
        "--closed-loop-eval-skill-mode",
        choices=["picklift", "pick_from_top_cube", "pick_and_place_cube"],
        default=None,
    )
    parser.add_argument("--closed-loop-task-prompt")
    parser.add_argument("--closed-loop-record-rollout-gif", action="store_true")
    parser.add_argument("--record-loop-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-loop-media", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loop-artifact-width", type=int, default=128)
    parser.add_argument("--loop-artifact-height", type=int, default=128)
    parser.add_argument("--loop-artifact-fps", type=int, default=12)
    parser.add_argument("--loop-artifact-every-n-steps", type=int, default=1)
    parser.add_argument("--qwen-model", default="qwen3-vl-8b-instruct-mlx")
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-api-key")
    parser.add_argument("--qwen-response-json", type=Path)
    parser.add_argument("--qwen-plan-json", type=Path)
    parser.add_argument("--qwen-object", default="green cube")
    parser.add_argument("--closed-loop-subgoal-chain-mode", choices=["off", "fixed", "valid-mask"], default="off")
    parser.add_argument("--closed-loop-subgoal-sequence")
    parser.add_argument("--closed-loop-fixed-subgoal-chunks", type=int, default=1)
    parser.add_argument("--closed-loop-valid-mask-checkpoint", type=Path)
    parser.add_argument("--closed-loop-valid-mask-threshold", type=float, default=0.5)
    parser.add_argument("--closed-loop-valid-mask-consecutive", type=int, default=2)
    parser.add_argument("--closed-loop-policy-n-action-steps", type=int, default=15)
    parser.add_argument("--closed-loop-policy-num-steps", type=int, default=10)
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        help="Forward validation cadence as steps, e.g. 10 locally or 300 on cloud.",
    )
    parser.add_argument(
        "--validation-interval-epochs",
        type=int,
        default=1,
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
        default=Path(sys.executable),
        help="Python executable for SO101 helper scripts.",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to lerobot_train_so101_lightning.py after an optional -- separator.",
    )


def start(args: argparse.Namespace, passthrough: list[str]) -> int:
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

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
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
    training_args = _forwarded_args(args.training_args, passthrough)
    training_args = _with_dataset_config(training_args, dataset_config, runtime_platform=args.runtime_platform)
    training_args = _with_validation_schedule(training_args, args)
    training_args = _with_checkpoint_schedule(training_args, dataset_config, args)
    runtime_contract = _runtime_contract(args, training_args)
    training_args = _with_runtime_contract(training_args, runtime_contract)
    train_cmd = [
        str(args.python),
        str(repo_root / "scripts" / "lerobot_train_so101_lightning.py"),
        "--tensorboard-log-dir",
        str(tensorboard_dir),
        *_ensure_arg(training_args, "output_dir", str(train_output_dir)),
    ]
    tensorboard_exe = _tensorboard_executable(args.python, repo_root)
    tensorboard_cmd = tensorboard_exe if tensorboard_exe else [str(args.python), "-m", "tensorboard.main"]
    tensorboard_cmd.extend(
        ["--logdir", str(tensorboard_dir), "--host", args.host, "--port", str(args.tensorboard_port)]
    )
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
    cache_build_cmds = _cache_build_commands(args.python, repo_root, dataset_config)

    launch_plan = {
        "operation": "start_so101_training",
        "run_dir": str(run_dir),
        "train_output_dir": str(train_output_dir),
        "lock_file": str(args.lock_file.resolve()),
        "local_training_standard": _local_training_standard(repo_root),
        "train_cmd": train_cmd,
        "dataset_config": dataset_config,
        "tensorboard_cmd": None if args.no_tensorboard else tensorboard_cmd,
        "dashboard_cmd": None if args.no_dashboard else dashboard_cmd,
        "gpu_monitor_cmd": None if args.no_gpu_monitor else gpu_monitor_cmd,
        "progress_monitor_cmd": None if args.no_progress_monitor else progress_monitor_cmd,
        "cache_build_cmds": cache_build_cmds,
        "runtime_contract": runtime_contract,
        "tensorboard_url": None if args.no_tensorboard else f"http://127.0.0.1:{args.tensorboard_port}/",
        "dashboard_url": None if args.no_dashboard else f"http://127.0.0.1:{args.dashboard_port}/",
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
    train = _popen(train_cmd, log_dir / "train.log", cwd=repo_root)
    train_pid_file.write_text(str(train.pid) + "\n", encoding="utf-8")
    tensorboard = None if args.no_tensorboard else _popen(tensorboard_cmd, log_dir / "tensorboard.log", cwd=repo_root)
    dashboard = None if args.no_dashboard else _popen(dashboard_cmd, log_dir / "dashboard.log", cwd=repo_root)
    gpu_monitor = None if args.no_gpu_monitor else _popen(gpu_monitor_cmd, log_dir / "gpu_monitor.log", cwd=repo_root)
    progress_monitor = (
        None
        if args.no_progress_monitor or progress_monitor_cmd is None
        else _popen(progress_monitor_cmd, log_dir / "progress_monitor.log", cwd=repo_root)
    )
    record = {
        **launch_plan,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_pid": train.pid,
        "tensorboard_pid": tensorboard.pid if tensorboard else None,
        "dashboard_pid": dashboard.pid if dashboard else None,
        "gpu_monitor_pid": gpu_monitor.pid if gpu_monitor else None,
        "progress_monitor_pid": progress_monitor.pid if progress_monitor else None,
        "logs": {
            "train": str(log_dir / "train.log"),
            "tensorboard": str(log_dir / "tensorboard.log") if tensorboard else None,
            "dashboard": str(log_dir / "dashboard.log") if dashboard else None,
            "gpu_monitor": str(log_dir / "gpu_monitor.log") if gpu_monitor else None,
            "progress_monitor": str(log_dir / "progress_monitor.log") if progress_monitor else None,
        },
    }
    _write_json(args.lock_file, record)
    current = status(args.lock_file)
    print(json.dumps(current, indent=2, sort_keys=True) if args.json else _human_status(current))
    return 0


def status(lock_file: Path) -> dict[str, Any]:
    record = _read_json(lock_file) or {"lock_file": str(lock_file.resolve()), "active": False}
    record.setdefault("local_training_standard", _local_training_standard(Path(__file__).resolve().parents[1]))
    train = _process_status(record.get("train_pid"))
    tensorboard = _process_status(record.get("tensorboard_pid"))
    dashboard = _process_status(record.get("dashboard_pid"))
    gpu_monitor = _process_status(record.get("gpu_monitor_pid"))
    progress_monitor = _process_status(record.get("progress_monitor_pid"))
    record["train"] = train
    record["tensorboard"] = tensorboard
    record["dashboard"] = dashboard
    record["gpu_monitor"] = gpu_monitor
    record["progress_monitor"] = progress_monitor
    record["active"] = any(
        bool(process.get("alive")) for process in (train, tensorboard, dashboard, gpu_monitor, progress_monitor)
    )
    return record


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
        f"dashboard: {_process_line(record.get('dashboard'))}",
        f"gpu_monitor: {_process_line(record.get('gpu_monitor'))}",
        f"progress_monitor: {_process_line(record.get('progress_monitor'))}",
    ]
    if record.get("tensorboard_url"):
        lines.append(f"tensorboard_url: {record['tensorboard_url']}")
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


def _load_dataset_config(path: Path | None, *, repo_root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path if path.is_absolute() else repo_root / path
    payload = _read_json(resolved)
    if payload is None:
        raise SystemExit(f"Dataset config not found or empty: {resolved}")
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
    if validation:
        if "repo_id" in validation:
            updated = _ensure_arg(updated, "validation-dataset-repo-id", str(validation["repo_id"]))
        if "root" in validation:
            updated = _ensure_arg(updated, "validation-dataset-root", str(validation["root"]))
    training = config.get("training") or {}
    if not isinstance(training, dict):
        raise SystemExit("dataset config training must be an object")
    for name, cli_name in (
        ("num_workers", "num_workers"),
        ("batch_size", "batch_size"),
        ("policy_repo_id", "policy.repo_id"),
        ("lightning_precision", "lightning-precision"),
    ):
        if name in training:
            value = training[name]
            if name == "num_workers" and runtime_platform == "macos":
                value = 0
            updated = _ensure_arg(updated, cli_name, str(value))
    if "policy_push_to_hub" in training:
        updated = _ensure_arg(updated, "policy.push_to_hub", str(bool(training["policy_push_to_hub"])).lower())
    cache = config.get("predecoded_image_cache") or {}
    if not isinstance(cache, dict):
        raise SystemExit("dataset config predecoded_image_cache must be an object")
    for name, cli_name in (
        ("train", "so101-image-cache-dir"),
        ("validation", "validation-image-cache-dir"),
    ):
        if name == "train" and train_datasets:
            continue
        if name in cache:
            updated = _ensure_arg(updated, cli_name, str(_resolve_cache_dir(cache, name)))
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
        ("image_affine_degrees", "so101-image-affine-degrees"),
        ("image_affine_translate", "so101-image-affine-translate"),
    ):
        if name in augmentation:
            updated = _ensure_arg(updated, cli_name, str(augmentation[name]))
    for name, cli_name in (
        ("state_jitter_arm_only", "so101-state-jitter-arm-only"),
        ("state_dropout_keep_gripper", "so101-state-dropout-keep-gripper"),
        ("gpu_image_augmentation", "so101-gpu-image-augmentation"),
    ):
        if name in augmentation:
            updated = _ensure_boolean_optional_arg(updated, cli_name, value=bool(augmentation[name]))
    return updated


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
        return [*args, f"--validation-interval-epochs={int(namespace.closed_loop_every_epochs)}"]
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
    if not isinstance(validation, dict):
        errors.append("dataset config must define validation_dataset for val/loss.")
    else:
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
    if not _has_any_arg(training_args, "validation-dataset-root"):
        errors.append("training command is missing --validation-dataset-root.")
    if not _has_any_arg(training_args, "validation-dataset-repo-id"):
        errors.append("training command is missing --validation-dataset-repo-id.")

    closed_loop_policy = _closed_loop_policy_name(args)
    if closed_loop_policy == "off":
        errors.append("closed-loop evaluation is disabled; use periodic, best_only, or best_or_periodic.")
    if closed_loop_policy == "best_only":
        errors.append(
            "closed-loop policy best_only can skip validation checkpoints; use periodic or best_or_periodic "
            "so every validation-loss checkpoint also runs closed-loop."
        )
    if args.no_progress_monitor:
        errors.append("progress monitor is disabled; it is required for supervised validation and closed-loop metrics.")
    if launch_plan.get("progress_monitor_cmd") is None:
        errors.append("progress monitor command is missing.")
    progress_monitor_cmd = launch_plan.get("progress_monitor_cmd") or []
    if "--mujoco-gl" not in progress_monitor_cmd:
        errors.append("progress monitor command is missing --mujoco-gl for platform-specific closed-loop rendering.")
    if int(args.closed_loop_every_epochs) <= 0:
        errors.append("--closed-loop-every-epochs must be positive.")
    if int(args.closed_loop_episodes) <= 0:
        errors.append("--closed-loop-episodes must be positive.")
    if int(args.closed_loop_steps) <= 0:
        errors.append("--closed-loop-steps must be positive.")

    closed_loop = dataset_config.get("closed_loop") or {}
    closed_loop_runner = _closed_loop_runner(args, dataset_config)
    if closed_loop_runner != "qwen_chain" and not args.closed_loop_eval_skill_mode and not (
        isinstance(closed_loop, dict) and closed_loop.get("eval_skill_mode")
    ):
        errors.append("closed-loop eval skill mode must be set in config closed_loop.eval_skill_mode or CLI.")
    if not _closed_loop_task_prompt(args, dataset_config):
        errors.append("closed-loop task prompt must be set in config closed_loop.task_prompt or CLI.")
    if closed_loop_runner == "qwen_chain" and _closed_loop_valid_mask_checkpoint(args, dataset_config) is None:
        errors.append(
            "qwen_chain loop tests require closed_loop.valid_mask_checkpoint or "
            "--closed-loop-valid-mask-checkpoint."
        )

    steps_per_epoch = _steps_per_epoch(dataset_config, training_args)
    if steps_per_epoch is None:
        errors.append("training.steps_per_epoch or --steps-per-epoch is required for closed-loop scheduling.")
    save_freq = _positive_int_arg(training_args, "save_freq") or _positive_int_arg(training_args, "save-freq")
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

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"SO101 monitored training contract failed:\n{detail}")


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
    return str(getattr(args, "closed_loop_policy", "periodic") or "periodic")


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
    validation = dataset_config.get("validation_dataset") or dataset_config.get("train_dataset") or {}
    if not isinstance(validation, dict) or "root" not in validation or "repo_id" not in validation:
        return None
    training = dataset_config.get("training") or {}
    if not isinstance(training, dict):
        training = {}
    batch_size = int(_arg_value(training_args, "batch_size") or training.get("batch_size") or 4)
    steps_per_epoch = int(training.get("steps_per_epoch") or _arg_value(training_args, "steps_per_epoch") or 1)
    policy_device = str(_arg_value(training_args, "policy.device") or "auto")
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
        str(training.get("validation_max_batches", 16)),
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
        "--mujoco-gl",
        runtime_contract["closed_loop_mujoco_gl"],
        "--closed-loop-runner",
        closed_loop_runner,
        "--closed-loop-policy",
        args.closed_loop_policy,
        "--closed-loop-eval-skill-mode",
        _closed_loop_eval_skill_mode(args, dataset_config),
        "--closed-loop-subgoal-chain-mode",
        getattr(args, "closed_loop_subgoal_chain_mode", "off"),
        "--closed-loop-fixed-subgoal-chunks",
        str(getattr(args, "closed_loop_fixed_subgoal_chunks", 1)),
        "--closed-loop-valid-mask-threshold",
        str(getattr(args, "closed_loop_valid_mask_threshold", 0.5)),
        "--closed-loop-valid-mask-consecutive",
        str(getattr(args, "closed_loop_valid_mask_consecutive", 2)),
        "--policy-n-action-steps",
        str(getattr(args, "closed_loop_policy_n_action_steps", 15)),
        "--policy-num-steps",
        str(getattr(args, "closed_loop_policy_num_steps", 10)),
        "--local-files-only",
    ]
    closed_loop_subgoal_sequence = getattr(args, "closed_loop_subgoal_sequence", None)
    if closed_loop_subgoal_sequence:
        cmd.extend(["--closed-loop-subgoal-sequence", str(closed_loop_subgoal_sequence)])
    closed_loop_valid_mask_checkpoint = _closed_loop_valid_mask_checkpoint(args, dataset_config)
    if closed_loop_valid_mask_checkpoint:
        cmd.extend(["--closed-loop-valid-mask-checkpoint", str(closed_loop_valid_mask_checkpoint)])
    closed_loop_task_prompt = _closed_loop_task_prompt(args, dataset_config)
    if closed_loop_task_prompt:
        cmd.extend(["--closed-loop-task-prompt", closed_loop_task_prompt])
    if args.closed_loop_record_rollout_gif or _closed_loop_record_rollout_gif(dataset_config):
        cmd.append("--closed-loop-record-rollout-gif")
    if args.record_loop_artifacts:
        cmd.extend(
            [
                "--record-loop-artifacts",
                "--render-loop-media" if args.render_loop_media else "--no-render-loop-media",
                "--loop-artifact-width",
                str(args.loop_artifact_width),
                "--loop-artifact-height",
                str(args.loop_artifact_height),
                "--loop-artifact-fps",
                str(args.loop_artifact_fps),
                "--loop-artifact-every-n-steps",
                str(args.loop_artifact_every_n_steps),
            ]
        )
    else:
        cmd.append("--no-record-loop-artifacts")
    if closed_loop_runner == "qwen_chain":
        cmd.extend(["--qwen-model", args.qwen_model, "--qwen-object", args.qwen_object])
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


def _closed_loop_eval_skill_mode(args: argparse.Namespace, dataset_config: dict[str, Any]) -> str:
    if args.closed_loop_eval_skill_mode:
        return str(args.closed_loop_eval_skill_mode)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("eval_skill_mode"):
        return str(closed_loop["eval_skill_mode"])
    return "picklift"


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
    return "picklift"


def _closed_loop_valid_mask_checkpoint(args: argparse.Namespace, dataset_config: dict[str, Any]) -> Path | None:
    value = getattr(args, "closed_loop_valid_mask_checkpoint", None)
    if value:
        return Path(value)
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("valid_mask_checkpoint"):
        return Path(str(closed_loop["valid_mask_checkpoint"]))
    return None


def _qwen_response_json(dataset_config: dict[str, Any]) -> Path | None:
    closed_loop = dataset_config.get("closed_loop") or {}
    if isinstance(closed_loop, dict) and closed_loop.get("qwen_response_json"):
        return Path(str(closed_loop["qwen_response_json"]))
    if (
        isinstance(closed_loop, dict)
        and closed_loop.get("execution_policy") == "qwen_edge_chain"
    ) or dataset_config.get("execution_policy") == "qwen_edge_chain":
        return Path("configs/agent/qwen3_so101_tool_planner_mock_response.json")
    return None


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
        if "train" in cache and isinstance(dataset, dict) and "root" in dataset and "repo_id" in dataset:
            commands.append(_cache_build_command(python, repo_root, dataset, _resolve_cache_dir(cache, "train")))
    for split, dataset_key in (("validation", "validation_dataset"),):
        dataset = config.get(dataset_key) or {}
        if split not in cache or not isinstance(dataset, dict):
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
