#!/usr/bin/env python3

import argparse
import os
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    log_root = run_dir / "tensorboard"
    source = log_root / args.source_run if args.source_run else log_root
    state_path = args.state_path or run_dir / "metrics" / "important_loss_mirror_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _load_seen(state_path)
    while True:
        try:
            count = _mirror_once(
                source=source,
                state_path=state_path,
                seen=seen,
            )
            print(
                "mirrored_new={} seen_train={} seen_val={}".format(
                    count,
                    len(seen.get("train/loss", [])),
                    len(seen.get("val/loss", [])),
                ),
                flush=True,
            )
        except Exception as exc:
            print(f"mirror_error={exc!r}", flush=True)
        if args.once or (args.until_pid is not None and not _pid_exists(args.until_pid)):
            return
        time.sleep(max(1.0, args.interval_s))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror SO101 train/val loss into important/* TensorBoard scalar tags.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", default="")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--until-pid", type=int)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _load_seen(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {"train/loss": [], "val/loss": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"train/loss": [], "val/loss": []}
    return {
        "train/loss": list(data.get("train/loss", [])),
        "val/loss": list(data.get("val/loss", [])),
    }


def _save_seen(path: Path, seen: dict[str, list[int]]) -> None:
    path.write_text(json.dumps(seen, sort_keys=True), encoding="utf-8")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _mirror_once(
    *,
    source: Path,
    state_path: Path,
    seen: dict[str, list[int]],
) -> int:
    accumulator = EventAccumulator(str(source), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    total = 0
    with SummaryWriter(log_dir=str(source)) as writer:
        for source_tag, target_tag in (
            ("train/loss", "important/train_loss"),
            ("val/loss", "important/val_loss"),
        ):
            total += _mirror_scalar_tag(
                accumulator=accumulator,
                tags=tags,
                writer=writer,
                source_tag=source_tag,
                target_tag=target_tag,
                seen=seen,
            )
    _save_seen(state_path, seen)
    return total


def _mirror_scalar_tag(
    *,
    accumulator: EventAccumulator,
    tags: set[str],
    writer: SummaryWriter,
    source_tag: str,
    target_tag: str,
    seen: dict[str, list[int]],
) -> int:
    if source_tag not in tags:
        return 0
    seen_steps = set(int(step) for step in seen.get(source_tag, []))
    if target_tag in tags:
        seen_steps.update(int(event.step) for event in accumulator.Scalars(target_tag))
    total = 0
    for event in accumulator.Scalars(source_tag):
        step = int(event.step)
        if step in seen_steps:
            continue
        writer.add_scalar(target_tag, event.value, global_step=step, walltime=event.wall_time)
        seen_steps.add(step)
        total += 1
    writer.flush()
    seen[source_tag] = sorted(seen_steps)
    return total


if __name__ == "__main__":
    main()
