#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record periodic SO101 training heartbeat rows.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-s", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--train-pid-file", type=Path)
    args = parser.parse_args()

    iteration = 0
    while True:
        iteration += 1
        record_once(args)
        if args.iterations and iteration >= args.iterations:
            break
        time.sleep(max(1, int(args.interval_s)))


def record_once(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_jsonl(metrics_dir / "training_metrics.jsonl")
    val_rows = _read_jsonl(metrics_dir / "validation_metrics.jsonl")
    closed_rows = _read_jsonl(metrics_dir / "closed_loop_metrics.jsonl")
    checkpoints = _checkpoints(run_dir)
    latest_train = train_rows[-1] if train_rows else None
    latest_val = val_rows[-1] if val_rows else None
    latest_closed = closed_rows[-1] if closed_rows else None
    gpu = _gpu_status()
    train_process = _process_status(args.train_pid_file)

    event = {
        "kind": "ten_minute_progress_check",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checked_at_local": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        "detail": _detail(latest_train, latest_val, checkpoints, train_process, gpu),
        "checkpoint": checkpoints[-1] if checkpoints else None,
        "checkpoint_count": len(checkpoints),
        "latest_train": latest_train,
        "latest_validation": latest_val,
        "latest_closed_loop": latest_closed,
        "gpu": gpu,
        "train_process": train_process,
    }
    _append_jsonl(metrics_dir / "monitor_events.jsonl", event)
    _update_summary(metrics_dir / "loss_summary.json", event)


def _detail(
    latest_train: dict[str, Any] | None,
    latest_val: dict[str, Any] | None,
    checkpoints: list[str],
    train_process: dict[str, Any],
    gpu: dict[str, Any],
) -> str:
    train_text = "train=none"
    if latest_train:
        train_text = f"train_step={latest_train.get('step')} loss={latest_train.get('loss')}"
    val_text = "val=none"
    if latest_val:
        val_text = f"val_step={latest_val.get('step')} loss={latest_val.get('loss')}"
    gpu_text = "gpu=unknown"
    if gpu.get("gpus"):
        first = gpu["gpus"][0]
        gpu_text = (
            f"gpu={first.get('name')} util={first.get('utilization_percent')}% "
            f"mem={first.get('memory_used_mb')}/{first.get('memory_total_mb')}MB"
        )
    return (
        f"{train_text}; {val_text}; checkpoints={len(checkpoints)}; "
        f"train_alive={train_process.get('alive')}; {gpu_text}"
    )


def _update_summary(path: Path, event: dict[str, Any]) -> None:
    summary = _read_json(path) or {}
    latest_train = event.get("latest_train")
    latest_val = event.get("latest_validation")
    latest_closed = event.get("latest_closed_loop")
    if latest_train:
        summary["latest_train_loss"] = latest_train.get("loss")
        summary["latest_train_step"] = latest_train.get("step")
        if latest_train.get("epoch") is not None:
            summary["latest_train_epoch"] = latest_train.get("epoch")
    if latest_val:
        summary["latest_val_loss"] = latest_val.get("loss")
        summary["latest_val_step"] = latest_val.get("step")
        summary["latest_val_checkpoint"] = latest_val.get("checkpoint")
    if latest_closed:
        summary["latest_closed_loop_success_rate"] = latest_closed.get("success_rate")
        summary["latest_closed_loop_grasp_rate"] = latest_closed.get("grasp_rate")
        summary["latest_closed_loop_checkpoint"] = latest_closed.get("checkpoint")
    summary["latest_checkpoint"] = event.get("checkpoint")
    summary["checkpoint_count"] = event.get("checkpoint_count")
    summary["last_heartbeat_local"] = event.get("checked_at_local")
    summary["last_heartbeat_utc"] = event.get("checked_at_utc")
    summary["last_heartbeat_detail"] = event.get("detail")
    summary["last_gpu_status"] = event.get("gpu")
    summary["train_process"] = event.get("train_process")
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _gpu_status() -> dict[str, Any]:
    text = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "name": parts[0],
                "memory_total_mb": _float_or_none(parts[1]),
                "memory_used_mb": _float_or_none(parts[2]),
                "memory_free_mb": _float_or_none(parts[3]),
                "utilization_percent": _float_or_none(parts[4]),
                "temperature_c": _float_or_none(parts[5]),
                "power_w": _float_or_none(parts[6]),
            }
        )
    return {"backend": "NVIDIA CUDA" if rows else "unknown", "gpus": rows}


def _checkpoints(run_dir: Path) -> list[str]:
    root = run_dir / "checkpoints"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


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


if __name__ == "__main__":
    main()
