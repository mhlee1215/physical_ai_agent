#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.safety.so100_action_gate import (
    evaluate_so100_action_safety,
    load_action_payload,
    load_calibration,
    load_episode_state,
    write_safety_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a real SO-100 action candidate without actuation.")
    parser.add_argument("--action", type=Path, required=True, help="SmolVLA dry-run action JSON.")
    parser.add_argument("--episode", type=Path, required=True, help="Observation-only SO-100 episode JSONL.")
    parser.add_argument("--frame-index", type=int, default=25)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument(
        "--allow-unknown-action-semantics",
        action="store_true",
        help="Do not block solely because raw SmolVLA action units are unmapped. This still sends no action.",
    )
    args = parser.parse_args()

    report = evaluate_so100_action_safety(
        action=load_action_payload(args.action),
        current_state=load_episode_state(args.episode, args.frame_index),
        calibration=load_calibration(args.calibration),
        human_confirmed=args.human_confirmed,
        require_known_action_semantics=not args.allow_unknown_action_semantics,
    )
    write_safety_report(report, args.output)
    print(json.dumps(report, default=lambda item: item.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
