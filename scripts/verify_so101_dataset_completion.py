#!/usr/bin/env python3
"""Require registry readiness and live viewer API access for generated datasets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from physical_ai_agent.so101_dataset_registry import require_recipe_training_ready
from physical_ai_agent.so101_dataset_viewer_gate import verify_dataset_viewer_api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SO101_DATASET_VIEWER_URL", "http://127.0.0.1:8768"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--no-restart-viewer",
        action="store_true",
        help="Debug/test only. Normal dataset completion must restart the viewer.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    registry = require_recipe_training_ready(
        repo_root,
        args.recipe,
        splits=args.split or None,
    )
    if not args.no_restart_viewer:
        subprocess.run(
            ["sh", "scripts/launch_so101_dataset_viewer.sh", "restart"],
            cwd=repo_root,
            check=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
    result = verify_dataset_viewer_api(
        args.base_url,
        registry.entries,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "training_ready": True,
                "recipe": str(args.recipe),
                **result.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
