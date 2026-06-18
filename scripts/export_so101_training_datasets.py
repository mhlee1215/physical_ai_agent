#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


RECIPE_PATH = Path("configs/so101/training_datasets/export_recipes.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible SO101 LeRobot dataset export recipes.")
    parser.add_argument("--recipes", type=Path, default=RECIPE_PATH)
    parser.add_argument("--only", action="append", default=[], help="Recipe name to run. May be repeated.")
    parser.add_argument("--contract", choices=["dataset_contract", "skill_dataset_contract"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    payload = json.loads(args.recipes.read_text(encoding="utf-8"))
    selected = _select_recipes(payload["recipes"], names=set(args.only), contract=args.contract)
    commands = [_command_for_recipe(args.python, recipe, payload.get("defaults", {}), overwrite=args.overwrite) for recipe in selected]

    if args.dry_run:
        print(json.dumps({"recipes": [recipe["name"] for recipe in selected], "commands": commands}, indent=2))
        return

    env = {**os.environ, "PYTHONPATH": _prepend_pythonpath(os.environ.get("PYTHONPATH", ""), ["src", ".", "scripts"])}
    for recipe, command in zip(selected, commands, strict=True):
        print(f"[so101-export-all] running {recipe['name']} -> {recipe['root']}", flush=True)
        subprocess.run(command, check=True, env=env)


def _select_recipes(recipes: list[dict[str, Any]], *, names: set[str], contract: str | None) -> list[dict[str, Any]]:
    selected = []
    for recipe in recipes:
        if names and recipe["name"] not in names:
            continue
        if contract is not None and recipe.get("contract") != contract:
            continue
        selected.append(recipe)
    missing = sorted(names - {recipe["name"] for recipe in selected})
    if missing:
        raise SystemExit(f"unknown recipe name(s): {', '.join(missing)}")
    return selected


def _command_for_recipe(python: str, recipe: dict[str, Any], defaults: dict[str, Any], *, overwrite: bool) -> list[str]:
    command = [
        python,
        recipe["script"],
        "--root",
        recipe["root"],
        "--repo-id",
        recipe["repo_id"],
        "--episodes",
        str(recipe["episodes"]),
        "--seed",
        str(recipe["seed"]),
        "--fps",
        str(defaults.get("fps", 12)),
        "--width",
        str(defaults.get("width", 256)),
        "--height",
        str(defaults.get("height", 256)),
    ]
    if bool(defaults.get("use_videos", False)):
        command.append("--use-videos")
    if overwrite:
        command.append("--overwrite")
    if not bool(defaults.get("include_camera3_duplicate", True)):
        command.append("--no-camera3-duplicate")
    for key, value in recipe.get("args", {}).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    return command


def _prepend_pythonpath(existing: str, values: list[str]) -> str:
    parts = [*values]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


if __name__ == "__main__":
    main()
