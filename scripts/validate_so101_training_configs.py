#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from physical_ai_agent.so101_training_config_schema import (
    SCHEMA_PATH,
    TRAINING_CONFIG_DIR,
    validate_so101_training_config_dir,
    validate_so101_training_config_file,
)
from physical_ai_agent.so101_hydra_config import HYDRA_CONFIG_DIR, load_so101_hydra_training_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SO101 training JSON configs.")
    parser.add_argument("configs", nargs="*", type=Path, help="Config files to validate. Defaults to all training configs.")
    parser.add_argument("--config-dir", type=Path, default=TRAINING_CONFIG_DIR)
    parser.add_argument(
        "--hydra-config",
        action="append",
        default=[],
        help="Hydra config name under configs/so101/hydra, e.g. training/grip_the_cube_v2.",
    )
    parser.add_argument(
        "--skip-hydra",
        action="store_true",
        help="Do not validate Hydra training entrypoints.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation results.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.configs:
        results = [validate_so101_training_config_file(path, repo_root=repo_root) for path in args.configs]
    else:
        results = validate_so101_training_config_dir(args.config_dir, repo_root=repo_root)

    payload = {
        "schema": str(SCHEMA_PATH),
        "checked": [
            {
                "path": _display_path(result.path, repo_root),
                "ok": result.ok,
                "errors": result.errors,
            }
            for result in results
        ],
        "hydra_config_dir": str(HYDRA_CONFIG_DIR),
        "hydra_checked": [],
    }
    if not args.skip_hydra:
        hydra_names = args.hydra_config or _discover_hydra_training_configs(repo_root)
        for name in hydra_names:
            item = {"name": name, "ok": True, "errors": []}
            try:
                entry, training_config = load_so101_hydra_training_config(name, repo_root=repo_root)
                item["training_config"] = entry.training_config
                item["training_task"] = training_config.task
            except Exception as exc:  # noqa: BLE001 - validation CLI should report every config error.
                item["ok"] = False
                item["errors"] = [str(exc)]
            payload["hydra_checked"].append(item)
    payload["ok"] = all(item["ok"] for item in payload["checked"])
    payload["ok"] = payload["ok"] and all(item["ok"] for item in payload["hydra_checked"])

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in payload["checked"]:
            status = "ok" if item["ok"] else "FAIL"
            print(f"{status} {item['path']}")
            for error in item["errors"]:
                print(f"  - {error}")
        for item in payload["hydra_checked"]:
            status = "ok" if item["ok"] else "FAIL"
            target = item.get("training_config", "")
            suffix = f" -> {target}" if target else ""
            print(f"{status} hydra:{item['name']}{suffix}")
            for error in item["errors"]:
                print(f"  - {error}")
    return 0 if payload["ok"] else 1


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _discover_hydra_training_configs(repo_root: Path) -> list[str]:
    training_dir = repo_root / HYDRA_CONFIG_DIR / "training"
    if not training_dir.exists():
        return []
    return [f"training/{path.stem}" for path in sorted(training_dir.glob("*.yaml"))]


if __name__ == "__main__":
    raise SystemExit(main())
