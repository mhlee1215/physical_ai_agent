#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        help="Forward validation cadence as steps, e.g. 10 locally or 300 on cloud.",
    )
    parser.add_argument(
        "--validation-interval-epochs",
        type=int,
        help="Forward validation cadence as epochs. Ignored by the trainer when step cadence is also set.",
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
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = _load_dataset_config(args.dataset_config, repo_root=repo_root)
    training_args = _forwarded_args(args.training_args, passthrough)
    training_args = _with_dataset_config(training_args, dataset_config)
    training_args = _with_validation_schedule(training_args, args)
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

    launch_plan = {
        "operation": "start_so101_training",
        "run_dir": str(run_dir),
        "train_output_dir": str(train_output_dir),
        "lock_file": str(args.lock_file.resolve()),
        "train_cmd": train_cmd,
        "dataset_config": dataset_config,
        "tensorboard_cmd": None if args.no_tensorboard else tensorboard_cmd,
        "dashboard_cmd": None if args.no_dashboard else dashboard_cmd,
        "tensorboard_url": None if args.no_tensorboard else f"http://127.0.0.1:{args.tensorboard_port}/",
        "dashboard_url": None if args.no_dashboard else f"http://127.0.0.1:{args.dashboard_port}/",
    }
    if args.dry_run:
        print(json.dumps(launch_plan, indent=2, sort_keys=True))
        return 0

    train = _popen(train_cmd, log_dir / "train.log", cwd=repo_root)
    tensorboard = None if args.no_tensorboard else _popen(tensorboard_cmd, log_dir / "tensorboard.log", cwd=repo_root)
    dashboard = None if args.no_dashboard else _popen(dashboard_cmd, log_dir / "dashboard.log", cwd=repo_root)
    record = {
        **launch_plan,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_pid": train.pid,
        "tensorboard_pid": tensorboard.pid if tensorboard else None,
        "dashboard_pid": dashboard.pid if dashboard else None,
        "logs": {
            "train": str(log_dir / "train.log"),
            "tensorboard": str(log_dir / "tensorboard.log") if tensorboard else None,
            "dashboard": str(log_dir / "dashboard.log") if dashboard else None,
        },
    }
    _write_json(args.lock_file, record)
    current = status(args.lock_file)
    print(json.dumps(current, indent=2, sort_keys=True) if args.json else _human_status(current))
    return 0


def status(lock_file: Path) -> dict[str, Any]:
    record = _read_json(lock_file) or {"lock_file": str(lock_file.resolve()), "active": False}
    train = _process_status(record.get("train_pid"))
    tensorboard = _process_status(record.get("tensorboard_pid"))
    dashboard = _process_status(record.get("dashboard_pid"))
    record["train"] = train
    record["tensorboard"] = tensorboard
    record["dashboard"] = dashboard
    record["active"] = any(
        bool(process.get("alive")) for process in (train, tensorboard, dashboard)
    )
    return record


def stop(lock_file: Path, *, timeout_s: float, json_output: bool = False) -> int:
    record = _read_json(lock_file)
    if not record:
        payload = {"active": False, "detail": "no active lock"}
        print(json.dumps(payload, indent=2, sort_keys=True) if json_output else _human_status(payload))
        return 0
    pids = [record.get("dashboard_pid"), record.get("tensorboard_pid"), record.get("train_pid")]
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
    ]
    if record.get("tensorboard_url"):
        lines.append(f"tensorboard_url: {record['tensorboard_url']}")
    if record.get("dashboard_url"):
        lines.append(f"dashboard_url: {record['dashboard_url']}")
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


def _with_dataset_config(args: list[str], config: dict[str, Any] | None) -> list[str]:
    if not config:
        return args
    train = _required_mapping(config, "train_dataset")
    validation = config.get("validation_dataset") or {}
    if not isinstance(validation, dict):
        raise SystemExit("dataset config validation_dataset must be an object")

    updated = [*args]
    updated = _ensure_arg(updated, "dataset.repo_id", str(train["repo_id"]))
    updated = _ensure_arg(updated, "dataset.root", str(train["root"]))
    if validation:
        if "repo_id" in validation:
            updated = _ensure_arg(updated, "validation-dataset-repo-id", str(validation["repo_id"]))
        if "root" in validation:
            updated = _ensure_arg(updated, "validation-dataset-root", str(validation["root"]))
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
        ("action_dropout_prob", "so101-action-dropout-prob"),
        ("image_camera_dropout_prob", "so101-image-camera-dropout-prob"),
        ("image_patch_dropout_prob", "so101-image-patch-dropout-prob"),
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
    if namespace.validation_interval_epochs is not None:
        return [*args, f"--validation-interval-epochs={int(namespace.validation_interval_epochs)}"]
    return args


def _has_any_arg(args: list[str], *names: str) -> bool:
    prefixes = tuple(f"--{name}=" for name in names)
    spaced = {f"--{name}" for name in names}
    return any(arg.startswith(prefixes) or arg in spaced for arg in args)


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
