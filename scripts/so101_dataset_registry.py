#!/usr/bin/env python3
"""List and validate canonical recipe-backed SO101 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_dataset_registry import (
    DatasetRegistry,
    scan_dataset_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "validate", "training-manifest"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--recipe", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--require-training-ready", action="store_true")
    parser.add_argument("--dataset-id")
    args = parser.parse_args()

    registry = scan_dataset_registry(
        args.repo_root,
        inspect_artifacts=True,
        recipe_paths=args.recipe or None,
    )
    if args.command == "training-manifest":
        if not args.dataset_id:
            parser.error("training-manifest requires --dataset-id")
        manifest = registry.training_manifests.get(args.dataset_id)
        if manifest is None:
            parser.error(f"dataset id is not registered: {args.dataset_id}")
        print(json.dumps(manifest, indent=2))
    elif args.as_json:
        print(json.dumps(registry.to_dict(), indent=2))
    else:
        _print_registry(registry)

    failed = not registry.valid
    if args.require_training_ready:
        failed = failed or not registry.training_ready
    if args.command == "validate" and failed:
        raise SystemExit(1)


def _print_registry(registry: DatasetRegistry) -> None:
    print(
        f"SO101 dataset registry: {len(registry.entries)} splits, "
        f"valid={str(registry.valid).lower()}, training_ready={str(registry.training_ready).lower()}"
    )
    print("DATASET                           SPLIT       STATUS      READY  EPISODES   FRAMES       SIZE")
    for entry in registry.entries:
        print(
            f"{entry.catalog_name[:32]:32}  {entry.split[:10]:10}  {entry.status:10}  "
            f"{_yes_no(entry.training_ready):5}  {_value(entry.episodes):>8}  "
            f"{_value(entry.frames):>7}  {_format_bytes(entry.size_bytes):>9}"
        )
        for error in entry.readiness_errors:
            print(f"  ! {error}")
    for issue in registry.issues:
        print(f"ERROR [{issue.code}] {issue.recipe or '-'}:{issue.split or '-'} {issue.message}")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _value(value: Any) -> str:
    return "-" if value is None else str(value)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return str(value)


if __name__ == "__main__":
    main()
