#!/usr/bin/env python3
"""Run a complete, reproducible SO101 dataset-generation recipe."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_RECIPE = Path("configs/so101/dataset_generation/grip_the_cube_v2.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    stages = build_stages(recipe, python=args.python, split=args.split, overwrite=args.overwrite)
    if args.dry_run:
        print(json.dumps({"recipe": str(args.recipe), "stages": stages}, indent=2))
        return

    env = {**os.environ, "PYTHONPATH": _prepend_pythonpath(os.environ.get("PYTHONPATH", ""))}
    _run_stages(stages, env=env, workers=args.workers, lookup_cache=Path(recipe["lookup_cache"]))


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if int(recipe.get("schema_version", 0)) != 1:
        raise ValueError("dataset generation recipe schema_version must be 1")
    splits = recipe.get("splits")
    if not isinstance(splits, dict) or not {"train", "validation"}.issubset(splits):
        raise ValueError("recipe must define train and validation splits")
    _validate_unique_seed_ranges(recipe)
    return recipe


def build_stages(
    recipe: dict[str, Any], *, python: str, split: str, overwrite: bool
) -> list[dict[str, Any]]:
    selected = ["train", "validation"] if split == "all" else [split]
    stages: list[dict[str, Any]] = []
    for split_name in selected:
        split_spec = recipe["splits"][split_name]
        shard_roots = []
        for bin_spec in split_spec["bins"]:
            shard_root = Path(str(split_spec["output_root"]) + f"_shard_bin{bin_spec['id']}")
            shard_roots.append(shard_root)
            stages.append(
                {
                    "name": f"export:{split_name}:bin{bin_spec['id']}",
                    "command": _export_command(
                        recipe,
                        split_spec=split_spec,
                        bin_spec=bin_spec,
                        shard_root=shard_root,
                        python=python,
                        overwrite=overwrite,
                    ),
                }
            )
        stages.append(
            {
                "name": f"merge:{split_name}",
                "command": _merge_command(
                    recipe,
                    split_spec=split_spec,
                    shard_roots=shard_roots,
                    python=python,
                    overwrite=overwrite,
                ),
            }
        )
        stages.append(
            {
                "name": f"sidecar:{split_name}",
                "command": _sidecar_command(recipe, split_spec=split_spec, python=python),
            }
        )
        if split_name == "validation":
            stages.append(
                {
                    "name": "closed-loop-starts:validation",
                    "command": _closed_loop_command(recipe, split_spec=split_spec, python=python),
                }
            )
    if split == "all":
        stages.append({"name": "audit:train-vs-validation", "command": _audit_command(recipe, python=python)})
    return stages


def _export_command(
    recipe: dict[str, Any],
    *,
    split_spec: dict[str, Any],
    bin_spec: dict[str, Any],
    shard_root: Path,
    python: str,
    overwrite: bool,
) -> list[str]:
    bin_id = int(bin_spec["id"])
    command = [
        python,
        recipe["exporter"],
        "--root",
        str(shard_root),
        "--repo-id",
        f"{split_spec['repo_id']}-bin{bin_id}",
        "--episodes",
        str(bin_spec["episodes"]),
        "--seed",
        str(bin_spec["seed"]),
        "--grid-balance-target-per-bin",
        str(bin_spec["episodes"]),
        "--grid-balance-bins",
        str(bin_id),
        "--grid-lookup-start-index",
        str(bin_spec["lookup_start_index"]),
        "--grid-lookup-cache",
        recipe["lookup_cache"],
    ]
    for key, value in recipe["common"].items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    if overwrite:
        command.append("--overwrite")
    return command


def _merge_command(
    recipe: dict[str, Any],
    *,
    split_spec: dict[str, Any],
    shard_roots: list[Path],
    python: str,
    overwrite: bool,
) -> list[str]:
    command = [
        python,
        recipe["merge_script"],
        "--output-root",
        split_spec["output_root"],
        "--repo-id",
        split_spec["repo_id"],
    ]
    for root in shard_roots:
        command.extend(["--shard", str(root)])
    if overwrite:
        command.append("--overwrite")
    return command


def _sidecar_command(recipe: dict[str, Any], *, split_spec: dict[str, Any], python: str) -> list[str]:
    sidecar = recipe["sidecar"]
    return [
        python,
        recipe["sidecar_script"],
        "--dataset-root",
        split_spec["output_root"],
        "--camera-key",
        sidecar["camera_key"],
        "--grid-size",
        str(sidecar["grid_size"]),
        "--frame-index",
        str(sidecar["frame_index"]),
        "--min-area",
        str(sidecar["min_area"]),
    ]


def _closed_loop_command(recipe: dict[str, Any], *, split_spec: dict[str, Any], python: str) -> list[str]:
    loop = split_spec["closed_loop"]
    root = Path(split_spec["output_root"])
    return [
        python,
        recipe["closed_loop_script"],
        "--source-report",
        str(root / "so101_lerobot_export_report.json"),
        "--output",
        str(root / loop["output"]),
        "--episodes",
        str(loop["episodes"]),
        "--grid-bins",
        ",".join(str(value) for value in loop["bins"]),
    ]


def _audit_command(recipe: dict[str, Any], *, python: str) -> list[str]:
    train_spec = recipe["splits"]["train"]
    validation_spec = recipe["splits"]["validation"]
    train = train_spec["output_root"]
    validation = validation_spec["output_root"]
    audit = recipe["audit"]
    return [
        python,
        recipe["audit_script"],
        "--train-root",
        train,
        "--validation-root",
        validation,
        "--expected-prompt",
        audit["expected_prompt"],
        "--expected-resolution",
        "x".join(str(value) for value in audit["expected_resolution"]),
        "--expected-train-bins",
        _bin_counts_arg(train_spec),
        "--expected-validation-bins",
        _bin_counts_arg(validation_spec),
        "--expected-terminal-hold-steps",
        str(recipe["common"]["terminal_hold_steps"]),
        "--max-pre-close-alignment-deg",
        str(recipe["common"]["edge_contact_parallel_success_threshold_deg"]),
        "--output",
        str(Path(validation) / "meta" / "split_overlap_audit.json"),
    ]


def _bin_counts_arg(split_spec: dict[str, Any]) -> str:
    return ",".join(f"{row['id']}:{row['episodes']}" for row in split_spec["bins"])


def _validate_unique_seed_ranges(recipe: dict[str, Any]) -> None:
    ranges = []
    for split_name, split_spec in recipe["splits"].items():
        for bin_spec in split_spec["bins"]:
            start = (
                int(bin_spec["seed"])
                + int(bin_spec["id"]) * 100_000
                + int(bin_spec["lookup_start_index"])
            )
            attempts = int(bin_spec["episodes"]) * int(recipe["common"]["max_attempt_multiplier"])
            end = start + attempts - 1
            for other_name, other_start, other_end in ranges:
                if max(start, other_start) <= min(end, other_end):
                    raise ValueError(
                        f"seed ranges overlap: {split_name}/bin{bin_spec['id']} and {other_name}"
                    )
            ranges.append((f"{split_name}/bin{bin_spec['id']}", start, end))


def _run_stages(
    stages: list[dict[str, Any]], *, env: dict[str, str], workers: int, lookup_cache: Path
) -> None:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    index = 0
    while index < len(stages):
        if not stages[index]["name"].startswith("export:"):
            _run_stage(stages[index], env)
            index += 1
            continue
        split_name = stages[index]["name"].split(":", 2)[1]
        end = index
        while end < len(stages) and stages[end]["name"].startswith(f"export:{split_name}:"):
            end += 1
        exports = stages[index:end]
        if not lookup_cache.exists():
            _run_stage(exports.pop(0), env)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(exports) or 1)) as pool:
            futures = [pool.submit(_run_stage, stage, env) for stage in exports]
            for future in futures:
                future.result()
        index = end


def _run_stage(stage: dict[str, Any], env: dict[str, str]) -> None:
    print(f"[so101-dataset] {stage['name']}", flush=True)
    subprocess.run(stage["command"], check=True, env=env)


def _prepend_pythonpath(existing: str) -> str:
    values = ["src", ".", "scripts"]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


if __name__ == "__main__":
    main()
