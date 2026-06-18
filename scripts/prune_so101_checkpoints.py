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
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Checkpoint directory. Defaults to run-dir/checkpoints or run-dir/model/checkpoints.",
    )
    parser.add_argument("--keep", action="append", default=[], help="Checkpoint name to always keep, e.g. 001490.")
    parser.add_argument("--keep-best-validation", action="store_true")
    parser.add_argument("--keep-latest-complete", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        prune_once(
            args.run_dir,
            checkpoint_root=args.checkpoint_root,
            keep=set(args.keep),
            keep_latest_complete=args.keep_latest_complete,
            keep_best_validation=args.keep_best_validation,
        )
        if args.once:
            return
        time.sleep(args.interval_s)


def prune_once(
    run_dir: Path,
    *,
    checkpoint_root: Path | None = None,
    keep: set[str],
    keep_latest_complete: int,
    keep_best_validation: bool = False,
) -> None:
    checkpoints_dir = _checkpoint_root(run_dir, checkpoint_root)
    if not checkpoints_dir.exists():
        return
    checkpoint_dirs = sorted(path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit())
    complete = [path for path in checkpoint_dirs if _is_complete(path)]
    latest_complete = {path.name for path in complete[-keep_latest_complete:]} if keep_latest_complete > 0 else set()
    best_validation = {_best_validation_checkpoint(run_dir)} if keep_best_validation else set()
    keep_names = keep | latest_complete | {name for name in best_validation if name}
    for path in checkpoint_dirs:
        if path.name in keep_names:
            continue
        _remove_checkpoint(run_dir, path, keep_names)


def _checkpoint_root(run_dir: Path, checkpoint_root: Path | None) -> Path:
    if checkpoint_root is not None:
        return checkpoint_root
    for candidate in (run_dir / "checkpoints", run_dir / "model" / "checkpoints"):
        if candidate.exists():
            return candidate
    return run_dir / "checkpoints"


def _best_validation_checkpoint(run_dir: Path) -> str | None:
    path = run_dir / "metrics" / "validation_metrics.jsonl"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("checkpoint") is not None and row.get("loss") is not None:
            rows.append(row)
    if not rows:
        return None
    best = min(rows, key=lambda row: float(row["loss"]))
    return str(best["checkpoint"])


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
