#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from physical_ai_agent.so101_tensorboard_loss_layout import (
    add_so101_loss_custom_scalars,
    should_log_repeated_scalar_metric,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the canonical SO101 Loss custom-scalars dashboard to a TensorBoard logdir.",
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"TensorBoard log directory does not exist: {log_dir}")

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = sorted(accumulator.Tags().get("scalars", []))

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        add_so101_loss_custom_scalars(writer)
        writer.flush()

    deprecated = [
        tag
        for tag in scalar_tags
        if not should_log_repeated_scalar_metric(tag.rsplit("/", 1)[-1])
    ]
    print(
        json.dumps(
            {
                "log_dir": str(log_dir),
                "custom_scalar_category": "Loss",
                "charts": 5,
                "existing_scalar_tags": len(scalar_tags),
                "excluded_repeated_config_or_debug_tags": deprecated,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
