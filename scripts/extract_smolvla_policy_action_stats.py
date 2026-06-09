#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.safety.so100_smolvla_metadata_adapter import (
    extract_policy_postprocessor_action_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract authoritative SmolVLA action mean/std stats from a LeRobot policy postprocessor."
    )
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-stats-key")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            extract_policy_postprocessor_action_stats(
                model_id_or_path=args.model,
                output=args.output,
                action_stats_key=args.action_stats_key,
                local_files_only=args.local_files_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
