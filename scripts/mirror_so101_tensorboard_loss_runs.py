#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    log_root = run_dir / "tensorboard"
    source = log_root / args.source_run
    state_path = args.state_path or run_dir / "metrics" / "important_loss_mirror_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _load_seen(state_path)
    while True:
        try:
            count = _mirror_once(
                source=source,
                train_dir=log_root / args.train_run,
                val_dir=log_root / args.val_run,
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
        if args.once or (args.until_pid is not None and not Path(f"/proc/{args.until_pid}").exists()):
            return
        time.sleep(max(1.0, args.interval_s))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror SO101 train/val loss into separate TensorBoard runs for readable colors.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", default="so101_smolvla")
    parser.add_argument("--train-run", default="important_train_loss")
    parser.add_argument("--val-run", default="important_val_loss")
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


def _mirror_once(
    *,
    source: Path,
    train_dir: Path,
    val_dir: Path,
    state_path: Path,
    seen: dict[str, list[int]],
) -> int:
    main = SummaryWriter(log_dir=str(source))
    main.add_custom_scalars({"important_metrics": {"train_val_loss": ["Multiline", ["^important/loss$"]]}})
    main.close()

    accumulator = EventAccumulator(str(source), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    total = 0
    for source_tag, target_dir in (("train/loss", train_dir), ("val/loss", val_dir)):
        if source_tag not in tags:
            continue
        seen_steps = set(int(step) for step in seen.get(source_tag, []))
        writer = SummaryWriter(log_dir=str(target_dir))
        for event in accumulator.Scalars(source_tag):
            step = int(event.step)
            if step in seen_steps:
                continue
            writer.add_scalar("important/loss", event.value, global_step=step, walltime=event.wall_time)
            seen_steps.add(step)
            total += 1
        writer.close()
        seen[source_tag] = sorted(seen_steps)
    _save_seen(state_path, seen)
    return total


if __name__ == "__main__":
    main()
