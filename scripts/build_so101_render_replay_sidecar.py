#!/usr/bin/env python3
"""Build a renderer-independent replay sidecar for an SO101 dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.so101_dataset_generation_schema import load_dataset_generation_recipe
from physical_ai_agent.so101_render_replay import build_render_replay_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-verified-reconstruction",
        action="store_true",
        help=(
            "Reconstruct exact frame state from the recorded frame-0 snapshot and "
            "actions, rejecting any observation-state or final-outcome mismatch."
        ),
    )
    args = parser.parse_args()
    result = build_render_replay_sidecar(
        args.dataset_root,
        recipe=load_dataset_generation_recipe(args.recipe),
        split_name=args.split,
        output_dir=args.output_dir,
        allow_verified_reconstruction=args.allow_verified_reconstruction,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
