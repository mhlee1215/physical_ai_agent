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

from physical_ai_agent.so101_dataset_registry import (
    DatasetRegistryError,
    require_recipe_training_ready,
    validate_registered_recipe,
)


DEFAULT_RECIPE = Path("configs/so101/dataset_generation/grip_the_cube_v2.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--confirm-destructive-overwrite",
        action="store_true",
        help="Required with --overwrite for a real run; confirms explicit destructive replacement approval.",
    )
    parser.add_argument(
        "--reuse-complete-shards",
        action="store_true",
        help="Skip export shards whose report already records the requested episode count.",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    try:
        validate_registered_recipe(repo_root, args.recipe)
    except DatasetRegistryError as exc:
        parser.error(str(exc))
    if args.overwrite and not args.dry_run and not args.confirm_destructive_overwrite:
        parser.error("--overwrite requires --confirm-destructive-overwrite for a real run")
    recipe = load_recipe(args.recipe)
    stages = build_stages(recipe, python=args.python, split=args.split, overwrite=args.overwrite)
    if args.dry_run:
        print(json.dumps({"recipe": str(args.recipe), "stages": stages}, indent=2))
        return

    _require_append_only_output_roots(recipe, split=args.split, overwrite=args.overwrite)

    env = {**os.environ, "PYTHONPATH": _prepend_pythonpath(os.environ.get("PYTHONPATH", ""))}
    _run_stages(
        stages,
        env=env,
        workers=args.workers,
        reuse_complete_shards=args.reuse_complete_shards,
    )
    selected_splits = list(recipe["splits"]) if args.split == "all" else [args.split]
    try:
        registry = require_recipe_training_ready(
            repo_root,
            args.recipe,
            splits=selected_splits,
        )
    except DatasetRegistryError as exc:
        raise SystemExit(f"dataset generation finished but training-readiness validation failed:\n{exc}") from exc
    print(
        json.dumps(
            {
                "status": "complete",
                "training_ready": True,
                "recipe": str(args.recipe),
                "datasets": [entry.to_dict() for entry in registry.entries],
            },
            indent=2,
        )
    )


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if int(recipe.get("schema_version", 0)) != 1:
        raise ValueError("dataset generation recipe schema_version must be 1")
    splits = recipe.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("recipe must define at least one split")
    _validate_unique_seed_ranges(recipe)
    return recipe


def _require_append_only_output_roots(
    recipe: dict[str, Any], *, split: str, overwrite: bool
) -> None:
    if overwrite:
        return
    selected = list(recipe["splits"]) if split == "all" else [split]
    existing = [
        Path(str(recipe["splits"][name]["output_root"]))
        for name in selected
        if Path(str(recipe["splits"][name]["output_root"])).exists()
    ]
    if existing:
        roots = "\n".join(f"- {root}" for root in existing)
        raise FileExistsError(
            "append-only dataset generation refuses existing output roots; "
            "create a new versioned recipe/root instead:\n" + roots
        )


def build_stages(
    recipe: dict[str, Any], *, python: str, split: str, overwrite: bool
) -> list[dict[str, Any]]:
    selected = list(recipe["splits"]) if split == "all" else [split]
    missing = [name for name in selected if name not in recipe["splits"]]
    if missing:
        raise ValueError(f"recipe does not define split: {', '.join(missing)}")
    stages: list[dict[str, Any]] = []
    for lookup in recipe.get("lookup_builders", []):
        stages.append(
            {
                "name": f"lookup:{lookup['name']}",
                "command": _lookup_builder_command(recipe, lookup=lookup, python=python),
            }
        )
    for split_name in selected:
        split_spec = recipe["splits"][split_name]
        shard_roots = []
        for bin_spec in split_spec["bins"]:
            shard_name = str(bin_spec.get("shard", f"bin{bin_spec['id']}"))
            shard_root = Path(str(split_spec["output_root"]) + f"_shard_{shard_name}")
            shard_roots.append(shard_root)
            stages.append(
                {
                    "name": f"export:{split_name}:{shard_name}",
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
    if split == "all" and {"train", "validation"}.issubset(recipe["splits"]):
        stages.append({"name": "audit:train-vs-validation", "command": _audit_command(recipe, python=python)})
    if "train" in selected:
        for reference in recipe.get("overlap_audits", []):
            stages.append(
                {
                    "name": f"audit:train-vs-{reference['name']}",
                    "command": _reference_audit_command(recipe, reference=reference, python=python),
                }
            )
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
    shard_name = str(bin_spec.get("shard", f"bin{bin_id}"))
    command = [
        python,
        recipe["exporter"],
        "--root",
        str(shard_root),
        "--repo-id",
        f"{split_spec['repo_id']}-{shard_name}",
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
        split_spec.get("lookup_cache", recipe["lookup_cache"]),
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


def _lookup_builder_command(
    recipe: dict[str, Any], *, lookup: dict[str, Any], python: str
) -> list[str]:
    command = [
        python,
        recipe["lookup_builder_script"],
        "--output",
        lookup["output"],
        "--grid-size",
        str(lookup["grid_size"]),
        "--resolution",
        str(lookup["resolution"]),
        "--x-min",
        str(lookup["x_range"][0]),
        "--x-max",
        str(lookup["x_range"][1]),
        "--y-min",
        str(lookup["y_range"][0]),
        "--y-max",
        str(lookup["y_range"][1]),
        "--bins",
        ",".join(str(value) for value in lookup["bins"]),
    ]
    for source_report in lookup["source_reports"]:
        command.extend(["--source-report", source_report])
    if "candidate_start_index" in lookup:
        command.extend(["--candidate-start-index", str(lookup["candidate_start_index"])])
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
    command = [
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
    if sidecar.get("bin_source"):
        command.extend(["--bin-source", str(sidecar["bin_source"])])
    return command


def _closed_loop_command(recipe: dict[str, Any], *, split_spec: dict[str, Any], python: str) -> list[str]:
    loop = split_spec["closed_loop"]
    root = Path(split_spec["output_root"])
    command = [
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
    for key, flag in (
        ("success_metric", "--success-metric"),
        ("lift_success_height", "--lift-success-height"),
    ):
        if key in loop:
            command.extend([flag, str(loop[key])])
    for source_report in loop.get("exclude_source_reports", []):
        command.extend(["--exclude-source-report", str(source_report)])
    return command


def _audit_command(recipe: dict[str, Any], *, python: str) -> list[str]:
    train_spec = recipe["splits"]["train"]
    validation_spec = recipe["splits"]["validation"]
    train = train_spec["output_root"]
    validation = validation_spec["output_root"]
    audit = recipe["audit"]
    command = [
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
    return _append_lift_audit_args(command, audit)


def _bin_counts_arg(split_spec: dict[str, Any]) -> str:
    counts: dict[int, int] = {}
    for row in split_spec["bins"]:
        bin_id = int(row["id"])
        counts[bin_id] = counts.get(bin_id, 0) + int(row["episodes"])
    return ",".join(f"{bin_id}:{counts[bin_id]}" for bin_id in sorted(counts))


def _reference_audit_command(
    recipe: dict[str, Any], *, reference: dict[str, Any], python: str
) -> list[str]:
    train_spec = recipe["splits"]["train"]
    audit = recipe["audit"]
    command = [
        python,
        recipe["audit_script"],
        "--train-root",
        train_spec["output_root"],
        "--validation-root",
        reference["reference_root"],
        "--expected-prompt",
        audit["expected_prompt"],
        "--expected-resolution",
        "x".join(str(value) for value in audit["expected_resolution"]),
        "--expected-train-bins",
        _bin_counts_arg(train_spec),
        "--expected-validation-bins",
        ",".join(f"{key}:{value}" for key, value in reference["reference_bins"].items()),
        "--expected-terminal-hold-steps",
        str(recipe["common"]["terminal_hold_steps"]),
        "--max-pre-close-alignment-deg",
        str(recipe["common"]["edge_contact_parallel_success_threshold_deg"]),
        "--output",
        str(Path(train_spec["output_root"]) / reference["output"]),
    ]
    return _append_lift_audit_args(command, audit)


def _append_lift_audit_args(command: list[str], audit: dict[str, Any]) -> list[str]:
    for key, flag in (
        ("expected_min_lift_height", "--expected-min-lift-height"),
        ("expected_min_lift_steps", "--expected-min-lift-steps"),
        ("terminal_hold_action_tolerance", "--terminal-hold-action-tolerance"),
    ):
        if key in audit:
            command.extend([flag, str(audit[key])])
    return command


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
            shard_name = str(bin_spec.get("shard", f"bin{bin_spec['id']}"))
            ranges.append((f"{split_name}/{shard_name}", start, end))


def _run_stages(
    stages: list[dict[str, Any]],
    *,
    env: dict[str, str],
    workers: int,
    reuse_complete_shards: bool = False,
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
        if reuse_complete_shards:
            pending = []
            for stage in exports:
                if _export_shard_is_complete(stage):
                    print(f"[so101-dataset] reuse {stage['name']}", flush=True)
                else:
                    pending.append(stage)
            exports = pending
        lookup_cache = None
        if exports:
            command = exports[0]["command"]
            lookup_cache = Path(command[command.index("--grid-lookup-cache") + 1])
        if lookup_cache is not None and not lookup_cache.exists() and exports:
            _run_stage(exports.pop(0), env)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(exports) or 1)) as pool:
            futures = [pool.submit(_run_stage, stage, env) for stage in exports]
            for future in futures:
                future.result()
        index = end


def _export_shard_is_complete(stage: dict[str, Any]) -> bool:
    command = stage["command"]
    root = Path(command[command.index("--root") + 1])
    expected = int(command[command.index("--episodes") + 1])
    report_path = root / "so101_lerobot_export_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(report.get("exported_episodes", -1)) == expected


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
