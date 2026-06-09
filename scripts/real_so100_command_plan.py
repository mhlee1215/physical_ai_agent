#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.safety.so100_action_gate import (
    load_action_chunk_payload,
    load_calibration,
    load_episode_state,
)
from physical_ai_agent.safety.so100_command_adapter import build_so100_command_chunk_plan, write_command_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-actuation SO-100 command plan from a dry action.")
    parser.add_argument("--action", type=Path, required=True, help="SmolVLA dry-run action JSON.")
    parser.add_argument("--episode", type=Path, required=True, help="Observation-only SO-100 episode JSONL.")
    parser.add_argument("--frame-index", type=int, default=25)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument(
        "--adapter-semantics-confirmed",
        action="store_true",
        help="Assert that the raw-action to raw-tick delta mapping has been separately validated.",
    )
    args = parser.parse_args()

    plan = build_so100_command_chunk_plan(
        action_chunk=load_action_chunk_payload(args.action, action_steps=args.action_steps),
        current_state=load_episode_state(args.episode, args.frame_index),
        calibration=load_calibration(args.calibration),
        human_confirmed=args.human_confirmed,
        adapter_semantics_confirmed=args.adapter_semantics_confirmed,
    )
    write_command_plan(plan, args.output)
    print(json.dumps(plan, default=lambda item: item.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
