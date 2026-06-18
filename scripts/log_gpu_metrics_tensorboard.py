#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Log accelerator/system metrics to TensorBoard.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means run forever.")
    parser.add_argument(
        "--backend",
        choices=("auto", "nvidia", "host"),
        default="auto",
        help="auto uses nvidia-smi when available, otherwise host/MPS-safe metrics.",
    )
    parser.add_argument("--train-pid-file", type=Path, help="Optional file containing the training process pid.")
    args = parser.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(args.log_dir))
    sample = 0
    try:
        while True:
            if args.backend in {"auto", "nvidia"} and shutil.which("nvidia-smi"):
                _write_nvidia_metrics(writer, sample)
            elif args.backend == "nvidia":
                writer.add_scalar("system/gpu_monitor_available", 0.0, global_step=sample)
            else:
                _write_host_metrics(writer, sample, args.train_pid_file)
            writer.flush()
            sample += 1
            if args.max_samples > 0 and sample >= args.max_samples:
                break
            time.sleep(max(0.1, float(args.interval_s)))
    finally:
        writer.close()
    return 0


def _write_nvidia_metrics(writer: object, sample: int) -> None:
    rows = _query_nvidia_smi()
    for row in rows:
        prefix = f"system/gpu{row.index}"
        writer.add_scalar(f"{prefix}/util_percent", row.util_percent, global_step=sample)
        writer.add_scalar(f"{prefix}/memory_used_mb", row.memory_used_mb, global_step=sample)
        writer.add_scalar(f"{prefix}/memory_total_mb", row.memory_total_mb, global_step=sample)
        writer.add_scalar(f"{prefix}/memory_used_percent", row.memory_used_percent, global_step=sample)
        if row.power_w is not None:
            writer.add_scalar(f"{prefix}/power_w", row.power_w, global_step=sample)
        if row.index == 0:
            writer.add_scalar("system/gpu_util_percent", row.util_percent, global_step=sample)
            writer.add_scalar("system/gpu_memory_used_mb", row.memory_used_mb, global_step=sample)
            writer.add_scalar("system/gpu_memory_used_percent", row.memory_used_percent, global_step=sample)
            writer.add_scalar("system/gpu_monitor_available", 1.0, global_step=sample)


def _write_host_metrics(writer: object, sample: int, train_pid_file: Path | None) -> None:
    writer.add_scalar("system/gpu_monitor_available", 0.0, global_step=sample)
    writer.add_scalar("system/mps_available", 1.0 if _mps_available() else 0.0, global_step=sample)
    load_1m, load_5m, load_15m = os.getloadavg()
    writer.add_scalar("system/load_avg_1m", load_1m, global_step=sample)
    writer.add_scalar("system/load_avg_5m", load_5m, global_step=sample)
    writer.add_scalar("system/load_avg_15m", load_15m, global_step=sample)
    if train_pid_file:
        pid = _read_pid(train_pid_file)
        rss_mb = _process_rss_mb(pid) if pid else None
        writer.add_scalar("system/train_process_alive", 1.0 if rss_mb is not None else 0.0, global_step=sample)
        if rss_mb is not None:
            writer.add_scalar("system/train_process_rss_mb", rss_mb, global_step=sample)


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _process_rss_mb(pid: int) -> float | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    except Exception:
        return None
    output = output.strip()
    if not output:
        return None
    return float(output) / 1024.0


class GpuRow:
    def __init__(
        self,
        *,
        index: int,
        util_percent: float,
        memory_used_mb: float,
        memory_total_mb: float,
        power_w: float | None,
    ) -> None:
        self.index = index
        self.util_percent = util_percent
        self.memory_used_mb = memory_used_mb
        self.memory_total_mb = memory_total_mb
        self.power_w = power_w

    @property
    def memory_used_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return 100.0 * self.memory_used_mb / self.memory_total_mb


def _query_nvidia_smi() -> list[GpuRow]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[GpuRow] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            GpuRow(
                index=int(parts[0]),
                util_percent=_float(parts[1]),
                memory_used_mb=_float(parts[2]),
                memory_total_mb=_float(parts[3]),
                power_w=None if parts[4] in {"", "[N/A]", "N/A"} else _float(parts[4]),
            )
        )
    return rows


def _float(value: str) -> float:
    return float(value.strip().replace(" W", "").replace(" MiB", "").replace("%", ""))


if __name__ == "__main__":
    raise SystemExit(main())
