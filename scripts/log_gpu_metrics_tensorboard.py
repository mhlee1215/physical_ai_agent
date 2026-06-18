#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Log nvidia-smi GPU metrics to TensorBoard.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means run forever.")
    args = parser.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(args.log_dir))
    sample = 0
    try:
        while True:
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
            writer.flush()
            sample += 1
            if args.max_samples > 0 and sample >= args.max_samples:
                break
            time.sleep(max(0.1, float(args.interval_s)))
    finally:
        writer.close()
    return 0


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
