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
    parser.add_argument(
        "--photoreal-preview",
        action="store_true",
        help="After each dataset export, render one high-fidelity SO101 preview sidecar with Blender Cycles.",
    )
    parser.add_argument("--photoreal-width", type=int, default=640)
    parser.add_argument("--photoreal-height", type=int, default=480)
    parser.add_argument("--photoreal-samples", type=int, default=512)
    parser.add_argument("--photoreal-robot-material", choices=("plastic", "matte_pla", "metal"), default="matte_pla")
    parser.add_argument("--photoreal-blender-bin", default="blender")
    parser.add_argument("--photoreal-asset-root", type=Path, default=Path("_workspace/photoreal_assets"))
    args = parser.parse_args()

    payload = json.loads(args.recipes.read_text(encoding="utf-8"))
    selected = _select_recipes(payload["recipes"], names=set(args.only), contract=args.contract)
    commands = [_command_for_recipe(args.python, recipe, payload.get("defaults", {}), overwrite=args.overwrite) for recipe in selected]
    photoreal_commands = [
        _photoreal_command_for_recipe(
            args.python,
            recipe,
            width=args.photoreal_width,
            height=args.photoreal_height,
            samples=args.photoreal_samples,
            robot_material=args.photoreal_robot_material,
            blender_bin=args.photoreal_blender_bin,
            asset_root=args.photoreal_asset_root,
        )
        for recipe in selected
    ]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "recipes": [recipe["name"] for recipe in selected],
                    "commands": commands,
                    "photoreal_preview": bool(args.photoreal_preview),
                    "photoreal_commands": photoreal_commands if args.photoreal_preview else [],
                },
                indent=2,
            )
        )
        return

    env = {**os.environ, "PYTHONPATH": _prepend_pythonpath(os.environ.get("PYTHONPATH", ""), ["src", ".", "scripts"])}
    for recipe, command in zip(selected, commands, strict=True):
        print(f"[so101-export-all] running {recipe['name']} -> {recipe['root']}", flush=True)
        subprocess.run(command, check=True, env=env)
        if args.photoreal_preview:
            photoreal_command = _photoreal_command_for_recipe(
                args.python,
                recipe,
                width=args.photoreal_width,
                height=args.photoreal_height,
                samples=args.photoreal_samples,
                robot_material=args.photoreal_robot_material,
                blender_bin=args.photoreal_blender_bin,
                asset_root=args.photoreal_asset_root,
            )
            print(f"[so101-export-all] photoreal preview -> {recipe['root']}/photoreal_preview", flush=True)
            subprocess.run(photoreal_command, check=True, env=env)


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


def _photoreal_command_for_recipe(
    python: str,
    recipe: dict[str, Any],
    *,
    width: int,
    height: int,
    samples: int,
    robot_material: str,
    blender_bin: str,
    asset_root: Path,
) -> list[str]:
    return [
        python,
        "scripts/render_so101_blender_probe.py",
        "--output-dir",
        str(Path(recipe["root"]) / "photoreal_preview"),
        "--seed",
        str(recipe["seed"]),
        "--warmup-steps",
        "8",
        "--width",
        str(width),
        "--height",
        str(height),
        "--samples",
        str(samples),
        "--denoise",
        "--robot-material",
        robot_material,
        "--blender-bin",
        blender_bin,
        "--asset-root",
        str(asset_root),
    ]


def _prepend_pythonpath(existing: str, values: list[str]) -> str:
    parts = [*values]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


if __name__ == "__main__":
    main()
