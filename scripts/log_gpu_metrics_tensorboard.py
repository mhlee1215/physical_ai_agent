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
    mps_available = _mps_available()
    writer.add_scalar("system/gpu_monitor_available", 0.0, global_step=sample)
    writer.add_scalar("system/accelerator_available", 1.0 if mps_available else 0.0, global_step=sample)
    writer.add_scalar("system/mps_available", 1.0 if mps_available else 0.0, global_step=sample)
    load_1m, load_5m, load_15m = os.getloadavg()
    writer.add_scalar("system/load_avg_1m", load_1m, global_step=sample)
    writer.add_scalar("system/load_avg_5m", load_5m, global_step=sample)
    writer.add_scalar("system/load_avg_15m", load_15m, global_step=sample)
    host_memory = _host_memory()
    for key, value in host_memory.items():
        if value is not None:
            writer.add_scalar(f"system/host_memory_{key}", value, global_step=sample)
    if train_pid_file:
        pid = _read_pid(train_pid_file)
        process_metrics = _process_metrics(pid) if pid else None
        writer.add_scalar(
            "system/train_process_alive",
            1.0 if process_metrics is not None else 0.0,
            global_step=sample,
        )
        if process_metrics is not None:
            for key, value in process_metrics.items():
                writer.add_scalar(f"system/train_process_{key}", value, global_step=sample)


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


def _process_metrics(pid: int) -> dict[str, float] | None:
    try:
        output = subprocess.check_output(["ps", "-o", "pcpu=,pmem=,rss=", "-p", str(pid)], text=True)
    except Exception:
        return None
    parts = output.strip().split()
    if len(parts) < 3:
        return None
    return {
        "cpu_percent": _float(parts[0]),
        "mem_percent": _float(parts[1]),
        "rss_mb": _float(parts[2]) / 1024.0,
    }


def _host_memory() -> dict[str, float | None]:
    if Path("/proc/meminfo").exists():
        return _linux_host_memory()
    if shutil.which("sysctl") and shutil.which("vm_stat"):
        return _macos_host_memory()
    return {}


def _linux_host_memory() -> dict[str, float | None]:
    rows: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows[parts[0].rstrip(":")] = _float(parts[1]) / 1024.0
    total = rows.get("MemTotal")
    available = rows.get("MemAvailable")
    used = total - available if total is not None and available is not None else None
    return _memory_row(total_mb=total, used_mb=used, available_mb=available)


def _macos_host_memory() -> dict[str, float | None]:
    try:
        total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        vm_output = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return {}
    page_size = 4096
    rows: dict[str, int] = {}
    for line in vm_output.splitlines():
        if "page size of" in line:
            parts = line.split()
            for index, part in enumerate(parts):
                if part == "of" and index + 1 < len(parts):
                    try:
                        page_size = int(parts[index + 1])
                    except ValueError:
                        pass
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = value.strip().rstrip(".").replace(".", "")
        if digits.isdigit():
            rows[key.strip()] = int(digits)
    free_pages = rows.get("Pages free", 0)
    inactive_pages = rows.get("Pages inactive", 0)
    speculative_pages = rows.get("Pages speculative", 0)
    available_bytes = (free_pages + inactive_pages + speculative_pages) * page_size
    total_mb = total_bytes / (1024.0 * 1024.0)
    available_mb = available_bytes / (1024.0 * 1024.0)
    used_mb = max(0.0, total_mb - available_mb)
    return _memory_row(total_mb=total_mb, used_mb=used_mb, available_mb=available_mb)


def _memory_row(*, total_mb: float | None, used_mb: float | None, available_mb: float | None) -> dict[str, float | None]:
    used_percent = 100.0 * used_mb / total_mb if total_mb and used_mb is not None else None
    available_percent = 100.0 * available_mb / total_mb if total_mb and available_mb is not None else None
    return {
        "total_mb": total_mb,
        "used_mb": used_mb,
        "available_mb": available_mb,
        "used_percent": used_percent,
        "available_percent": available_percent,
    }


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
