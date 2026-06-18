#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune SO101 training checkpoints to stay under quota.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keep", action="append", default=[], help="Checkpoint name to always keep, e.g. 001490.")
    parser.add_argument("--keep-latest-complete", type=int, default=2)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        prune_once(args.run_dir, set(args.keep), args.keep_latest_complete)
        if args.once:
            return
        time.sleep(args.interval_s)


def prune_once(run_dir: Path, keep: set[str], keep_latest_complete: int) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return
    checkpoint_dirs = sorted(path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit())
    complete = [path for path in checkpoint_dirs if _is_complete(path)]
    latest_complete = {path.name for path in complete[-keep_latest_complete:]} if keep_latest_complete > 0 else set()
    keep_names = keep | latest_complete
    for path in checkpoint_dirs:
        if path.name in keep_names:
            continue
        _remove_checkpoint(run_dir, path, keep_names)


def _is_complete(checkpoint_dir: Path) -> bool:
    required = [
        checkpoint_dir / "pretrained_model" / "model.safetensors",
        checkpoint_dir / "pretrained_model" / "train_config.json",
        checkpoint_dir / "training_state" / "training_step.json",
        checkpoint_dir / "training_state" / "optimizer_state.safetensors",
        checkpoint_dir / "training_state" / "scheduler_state.json",
    ]
    return all(path.exists() for path in required)


def _remove_checkpoint(run_dir: Path, checkpoint_dir: Path, keep_names: set[str]) -> None:
    size = _du_bytes(checkpoint_dir)
    shutil.rmtree(checkpoint_dir)
    event = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_dir.name,
        "detail": f"removed checkpoint {checkpoint_dir.name}; freed_estimate_bytes={size}; keep={sorted(keep_names)}",
        "freed_estimate_bytes": size,
        "kind": "checkpoint_pruned",
    }
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "monitor_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _du_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


if __name__ == "__main__":
    main()
