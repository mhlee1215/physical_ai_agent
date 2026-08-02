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
from physical_ai_agent.so101_closed_loop_contract import (
    contract_path_for_start_report,
    load_executable_loop_test_contract,
)
from physical_ai_agent.so101_dataset_generation_schema import (
    DatasetGenerationRecipe,
    load_dataset_generation_recipe,
)
from build_so101_dataset_distribution_report import require_distribution_report


def require_executable_loop_test_contracts(
    repo_root: Path,
    recipe: DatasetGenerationRecipe,
    *,
    splits: list[str] | None = None,
) -> list[str]:
    selected = set(splits or recipe.splits)
    verified: list[str] = []
    for split_name, split_spec in recipe.splits.items():
        if split_name not in selected or split_spec.closed_loop is None:
            continue
        dataset_root = Path(split_spec.output_root)
        if not dataset_root.is_absolute():
            dataset_root = repo_root / dataset_root
        report_path = dataset_root / split_spec.closed_loop.output
        if not report_path.is_file():
            raise FileNotFoundError(
                f"closed-loop start report is missing for {split_name}: {report_path}"
            )
        contract_path = contract_path_for_start_report(report_path)
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"executable loop-test contract is missing for {split_name}: "
                f"{contract_path}"
            )
        contract = load_executable_loop_test_contract(
            contract_path,
            repo_root=repo_root,
            expected_start_report=report_path,
        )
        test_case = contract["test_case"]
        if int(test_case["episodes"]) != int(split_spec.closed_loop.episodes):
            raise ValueError(
                f"loop-test contract episode mismatch for {split_name}: "
                f"{test_case['episodes']} != {split_spec.closed_loop.episodes}"
            )
        configured_root = Path(test_case["start_dataset"]["root"])
        if not configured_root.is_absolute():
            configured_root = repo_root / configured_root
        if configured_root.resolve() != dataset_root.resolve():
            raise ValueError(
                f"loop-test contract dataset mismatch for {split_name}: "
                f"{configured_root} != {dataset_root}"
            )
        verified.append(str(contract_path))
    return verified


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
    recipe = load_dataset_generation_recipe(args.recipe)
    loop_test_contracts: list[str] = []
    if recipe.schema_version >= 2:
        for entry in registry.entries:
            require_distribution_report(
                Path(entry.absolute_root),
                output_dir=recipe.distribution_report.output_dir,
            )
        loop_test_contracts = require_executable_loop_test_contracts(
            repo_root,
            recipe,
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
                "loop_test_contracts": loop_test_contracts,
                **result.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
