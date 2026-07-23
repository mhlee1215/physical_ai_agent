#!/usr/bin/env python3
"""Run a complete, reproducible SO101 dataset-generation recipe."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_dataset_generation_schema import load_dataset_generation_recipe
from physical_ai_agent.so101_dataset_registry import (
    DatasetRegistryError,
    require_recipe_training_ready,
    validate_registered_recipe,
)

DEFAULT_RECIPE = Path("configs/so101/dataset_generation/grip_the_cube_v2.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--split", default="all", help="Recipe split name or 'all'.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--confirm-destructive-overwrite",
        action="store_true",
        help=(
            "Required with --overwrite for a real run; confirms explicit destructive "
            "replacement approval."
        ),
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
    stages = build_stages(
        recipe,
        python=args.python,
        split=args.split,
        overwrite=args.overwrite,
        recipe_path=args.recipe,
    )
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
    selected_splits = _selected_split_names(recipe, args.split)
    try:
        registry = require_recipe_training_ready(
            repo_root,
            args.recipe,
            splits=selected_splits,
        )
    except DatasetRegistryError as exc:
        raise SystemExit(
            f"dataset generation finished but training-readiness validation failed:\n{exc}"
        ) from exc
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
    recipe = load_dataset_generation_recipe(path).as_dict()
    _validate_spawn_catalogs(recipe)
    _validate_unique_seed_ranges(recipe)
    return recipe


def _validate_spawn_catalogs(recipe: dict[str, Any]) -> None:
    source = recipe.get("source") or {}
    if source.get("mode") != "from_spawn_catalog":
        return
    expected_yaw = recipe.get("common", {}).get("target_object_yaw_deg")
    expected_qpos = (recipe.get("start_pose") or {}).get("sim_qpos")
    expected_rig = recipe.get("common", {}).get("camera_rig_config")
    expected_rig_sha256 = None
    if expected_rig:
        expected_rig_path = Path(expected_rig)
        if not expected_rig_path.is_file():
            raise FileNotFoundError(f"camera rig config does not exist: {expected_rig_path}")
        expected_rig_sha256 = hashlib.sha256(expected_rig_path.read_bytes()).hexdigest()
    for raw_path in source["catalogs"]:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"spawn catalog does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "so101_spawn_catalog_v1":
            raise ValueError(f"unsupported spawn catalog format: {path}")
        if expected_yaw is not None and (
            payload.get("target_object_yaw_deg") is None
            or not math.isclose(
                float(payload["target_object_yaw_deg"]), float(expected_yaw), abs_tol=1e-9
            )
        ):
            raise ValueError(f"spawn catalog target yaw does not match recipe: {path}")
        if expected_qpos is not None and payload.get("initial_qpos") != expected_qpos:
            raise ValueError(f"spawn catalog initial qpos does not match recipe: {path}")
        if expected_rig is not None and payload.get("camera_rig_config") != expected_rig:
            raise ValueError(f"spawn catalog camera rig does not match recipe: {path}")
        if expected_rig_sha256 is not None and payload.get("camera_rig_sha256") != expected_rig_sha256:
            raise ValueError(f"spawn catalog camera rig checksum does not match recipe: {path}")
        lookup = payload.get("lookup")
        if not isinstance(lookup, dict) or not lookup:
            raise ValueError(f"spawn catalog has no lookup mapping: {path}")
        for bin_id, candidates in lookup.items():
            if not isinstance(candidates, list):
                raise ValueError(f"spawn catalog bin {bin_id} is not a list: {path}")
            if any(
                not isinstance(candidate, list)
                or len(candidate) != 2
                or not all(isinstance(value, (int, float)) for value in candidate)
                for candidate in candidates
            ):
                raise ValueError(
                    f"spawn catalog candidates must contain only [x, y], bin={bin_id}: {path}"
                )


def _require_append_only_output_roots(
    recipe: dict[str, Any], *, split: str, overwrite: bool
) -> None:
    if overwrite:
        return
    selected = _selected_split_names(recipe, split)
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
    recipe: dict[str, Any],
    *,
    python: str,
    split: str,
    overwrite: bool,
    recipe_path: Path = DEFAULT_RECIPE,
) -> list[dict[str, Any]]:
    selected = _selected_split_names(recipe, split)
    stages: list[dict[str, Any]] = []
    generated_selected = [
        name for name in selected if recipe["splits"][name].get("kind", "generated") == "generated"
    ]
    for lookup in recipe.get("lookup_builders", []) if generated_selected else []:
        stages.append(
            {
                "name": f"lookup:{lookup['name']}",
                "command": _lookup_builder_command(recipe, lookup=lookup, python=python),
            }
        )
    for split_name in selected:
        split_spec = recipe["splits"][split_name]
        if split_spec.get("kind", "generated") == "episode_subset":
            stages.append(
                {
                    "name": f"subset:{split_name}",
                    "command": _episode_subset_command(
                        recipe,
                        split_spec=split_spec,
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
            if split_spec.get("closed_loop"):
                stages.append(
                    {
                        "name": f"closed-loop-starts:{split_name}",
                        "command": _closed_loop_command(
                            recipe, split_spec=split_spec, python=python
                        ),
                    }
                )
            continue
        if split_spec.get("kind", "generated") == "render_derivative":
            source_spec = _render_source_spec(recipe, split_spec)
            if split_spec.get("source_dataset_root"):
                stages.append(
                    {
                        "name": f"render-replay:{split_name}",
                        "command": _render_replay_command(
                            recipe,
                            split_name=split_name,
                            split_spec=split_spec,
                            source_spec=source_spec,
                            python=python,
                            recipe_path=recipe_path,
                        ),
                    }
                )
            stages.append(
                {
                    "name": f"render:{split_name}",
                    "command": _render_command(
                        recipe,
                        split_spec=split_spec,
                        source_spec=source_spec,
                        python=python,
                    ),
                }
            )
            render = split_spec["render"]
            if render.get("determinism_probe", True):
                stages.append(
                    {
                        "name": f"render-determinism:{split_name}",
                        "command": _render_command(
                            recipe,
                            split_spec=split_spec,
                            source_spec=source_spec,
                            python=python,
                            determinism_probe=True,
                        ),
                    }
                )
                stages.append(
                    {
                        "name": f"verify-render-determinism:{split_name}",
                        "command": _render_determinism_command(
                            recipe,
                            split_spec=split_spec,
                            python=python,
                        ),
                    }
                )
            stages.append(
                {
                    "name": f"build-derivative:{split_name}",
                    "command": _photoreal_builder_command(
                        recipe,
                        split_spec=split_spec,
                        source_spec=source_spec,
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
            if split_spec.get("closed_loop"):
                stages.append(
                    {
                        "name": f"closed-loop-starts:{split_name}",
                        "command": _closed_loop_command(
                            recipe, split_spec=split_spec, python=python
                        ),
                    }
                )
            continue
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
        replay = recipe.get("render_replay")
        if isinstance(replay, dict) and replay.get("enabled", True):
            stages.append(
                {
                    "name": f"render-replay:{split_name}",
                    "command": [
                        python,
                        recipe["render_replay_script"],
                        "--dataset-root",
                        split_spec["output_root"],
                        "--recipe",
                        str(recipe_path),
                        "--split",
                        split_name,
                    ],
                }
            )
        stages.append(
            {
                "name": f"sidecar:{split_name}",
                "command": _sidecar_command(recipe, split_spec=split_spec, python=python),
            }
        )
        if split_spec.get("closed_loop"):
            stages.append(
                {
                    "name": f"closed-loop-starts:{split_name}",
                    "command": _closed_loop_command(recipe, split_spec=split_spec, python=python),
                }
            )
    if split == "all" and {"train", "validation"}.issubset(recipe["splits"]):
        stages.append(
            {"name": "audit:train-vs-validation", "command": _audit_command(recipe, python=python)}
        )
    if "train" in selected:
        for reference in recipe.get("overlap_audits", []):
            stages.append(
                {
                    "name": f"audit:train-vs-{reference['name']}",
                    "command": _reference_audit_command(recipe, reference=reference, python=python),
                }
            )
    completion_command = [
        python,
        "scripts/verify_so101_dataset_completion.py",
        "--recipe",
        str(recipe_path),
    ]
    for split_name in selected:
        completion_command.extend(["--split", split_name])
    stages.append(
        {
            "name": "completion:registry-viewer",
            "command": completion_command,
        }
    )
    return stages


def _selected_split_names(recipe: dict[str, Any], split: str) -> list[str]:
    if split == "all":
        return list(recipe["splits"])
    if split not in recipe["splits"]:
        raise ValueError(f"recipe does not define split: {split}")
    required = {split}
    selected_spec = recipe["splits"][split]
    if selected_spec.get("kind") == "render_derivative" and selected_spec.get("source_split"):
        required.add(str(selected_spec["source_split"]))
    return [name for name in recipe["splits"] if name in required]


def _render_source_spec(recipe: dict[str, Any], split_spec: dict[str, Any]) -> dict[str, Any]:
    if split_spec.get("source_split"):
        return recipe["splits"][str(split_spec["source_split"])]
    return {
        "output_root": split_spec["source_dataset_root"],
        "expected_episodes": split_spec["expected_episodes"],
        "expected_bins": split_spec.get("expected_bins", {}),
        "render_replay_sidecar": split_spec["render_replay_sidecar"],
    }


def _render_replay_path(source_spec: dict[str, Any], replay: dict[str, Any]) -> Path:
    if source_spec.get("render_replay_sidecar"):
        return Path(str(source_spec["render_replay_sidecar"]))
    return Path(str(source_spec["output_root"])) / replay.get("output_dir", "render_replay")


def _render_replay_command(
    recipe: dict[str, Any],
    *,
    split_name: str,
    split_spec: dict[str, Any],
    source_spec: dict[str, Any],
    python: str,
    recipe_path: Path,
) -> list[str]:
    return [
        python,
        recipe["render_replay_script"],
        "--dataset-root",
        str(source_spec["output_root"]),
        "--recipe",
        str(recipe_path),
        "--split",
        split_name,
        "--output-dir",
        str(_render_replay_path(source_spec, recipe["render_replay"])),
        "--allow-verified-reconstruction",
    ]


def _render_command(
    recipe: dict[str, Any],
    *,
    split_spec: dict[str, Any],
    source_spec: dict[str, Any],
    python: str,
    determinism_probe: bool = False,
) -> list[str]:
    render = split_spec["render"]
    replay = recipe["render_replay"]
    source_root = Path(source_spec["output_root"])
    episodes = int(split_spec.get("expected_episodes") or _expected_split_episodes(source_spec))
    output_dir = _determinism_output_dir(render) if determinism_probe else render["output_dir"]
    command = [
        python,
        "scripts/render_so101_dataset_blender_preview.py",
        "--dataset-root",
        str(source_root),
        "--output-dir",
        output_dir,
        "--episodes",
        "0" if determinism_probe else ",".join(str(index) for index in range(episodes)),
        "--frames",
        "0" if determinism_probe else "all",
        "--camera-keys",
        ",".join(render["camera_keys"]),
        "--render-replay-sidecar",
        str(_render_replay_path(source_spec, replay)),
        "--width",
        str(render["width"]),
        "--height",
        str(render["height"]),
        "--samples",
        str(render["samples"]),
        "--cycles-seed",
        str(render["cycles_seed"]),
        "--lighting-profile",
        render["lighting_profile"],
        "--key-light-power",
        str(render["key_light_power"]),
        "--fill-light-power",
        str(render["fill_light_power"]),
        "--world-strength",
        str(render["world_strength"]),
        "--hdri-rotation-deg",
        str(render["hdri_rotation_deg"]),
        "--exposure",
        str(render["exposure"]),
        "--color-management",
        render["color_management"],
        "--color-look",
        render["color_look"],
        "--gamma",
        str(render["gamma"]),
        "--output-format",
        render["output_format"],
        "--robot-material",
        render["robot_material"],
        "--scene-profile",
        render["scene_profile"],
        "--asset-root",
        render["asset_root"],
        "--blender-bin",
        render["blender_bin"],
        "--blender-batch-size",
        "1" if determinism_probe else str(render["blender_batch_size"]),
    ]
    if render.get("denoise"):
        command.append("--denoise")
    if render.get("skip_existing", True) and not determinism_probe:
        command.append("--skip-existing")
    if render.get("material_profile"):
        command.extend(["--robot-material-config", render["material_profile"]])
    if not render.get("duplicate_camera3_from_camera2", True):
        command.append("--no-duplicate-camera3-from-camera2")
    return command


def _determinism_output_dir(render: dict[str, Any]) -> str:
    return str(
        Path(render["output_dir"]).with_name(Path(render["output_dir"]).name + "_determinism_check")
    )


def _render_determinism_command(
    recipe: dict[str, Any], *, split_spec: dict[str, Any], python: str
) -> list[str]:
    render = split_spec["render"]
    return [
        python,
        recipe["render_determinism_script"],
        "--reference-dir",
        render["output_dir"],
        "--candidate-dir",
        _determinism_output_dir(render),
        "--max-channel-diff",
        str(render["determinism_max_channel_diff"]),
        "--max-changed-pixels",
        str(render["determinism_max_changed_pixels"]),
        "--output",
        str(Path(render["output_dir"]) / "render_determinism_report.json"),
    ]


def _photoreal_builder_command(
    recipe: dict[str, Any],
    *,
    split_spec: dict[str, Any],
    source_spec: dict[str, Any],
    python: str,
    overwrite: bool,
) -> list[str]:
    render = split_spec["render"]
    builder_camera_keys = list(render["camera_keys"])
    if (
        render.get("duplicate_camera3_from_camera2", True)
        and "observation.images.camera3" not in builder_camera_keys
    ):
        builder_camera_keys.append("observation.images.camera3")
    command = [
        python,
        recipe["photoreal_builder_script"],
        "--source-dataset-root",
        source_spec["output_root"],
        "--rendered-dir",
        render["output_dir"],
        "--output-root",
        split_spec["output_root"],
        "--repo-id",
        split_spec["repo_id"],
        "--camera-keys",
        ",".join(builder_camera_keys),
    ]
    if not render.get("duplicate_camera3_from_camera2", True):
        command.append("--no-duplicate-camera3-from-camera2")
    if overwrite:
        command.append("--overwrite")
    elif render.get("skip_existing", True):
        command.append("--skip-existing")
    return command


def _expected_split_episodes(split_spec: dict[str, Any]) -> int:
    return sum(int(row["episodes"]) for row in split_spec.get("bins", []))


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
    ]
    lookup_cache = split_spec.get("lookup_cache") or recipe.get("lookup_cache")
    if lookup_cache:
        command.extend(["--grid-lookup-cache", str(lookup_cache)])
    start_pose = recipe.get("start_pose")
    if start_pose:
        command.append(
            "--initial-qpos=" + ",".join(
                str(value) for value in start_pose["sim_qpos"]
            )
        )
    for key, value in recipe["common"].items():
        if key == "contact_alignment":
            command.extend(_contact_alignment_args(value))
            continue
        if key == "inspection_gates":
            command.extend(_inspection_gate_args(value))
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    if overwrite:
        command.append("--overwrite")
    return command


def _episode_subset_command(
    recipe: dict[str, Any],
    *,
    split_spec: dict[str, Any],
    python: str,
    overwrite: bool,
) -> list[str]:
    subset = split_spec["subset"]
    command = [
        python,
        recipe["subset_script"],
        "--source-root",
        split_spec["source_dataset_root"],
        "--output-root",
        split_spec["output_root"],
        "--repo-id",
        split_spec["repo_id"],
        "--camera-key",
        subset["camera_key"],
        "--edge-mode",
        subset["edge_mode"],
        "--max-angle-deg",
        str(subset["max_angle_deg"]),
    ]
    if subset.get("selection_source_root"):
        command.extend(["--selection-source-root", subset["selection_source_root"]])
    if overwrite:
        command.append("--overwrite")
    return command


def _contact_alignment_args(spec: dict[str, Any]) -> list[str]:
    command = [
        "--edge-contact-parallel-success-threshold-deg",
        str(spec["max_pre_close_error_deg"]),
    ]
    trace = spec.get("camera2_trace")
    if trace is None:
        command.extend(["--close-alignment-gate-mode", "geometry_only"])
        return command
    command.extend(
        [
            "--close-alignment-gate-mode",
            str(trace["mode"]),
            "--pre-close-image-alignment-max-deg",
            str(trace["pre_close_max_deg"]),
            "--close-25-image-alignment-max-deg",
            str(trace["close_25_max_deg"]),
            "--close-50-image-alignment-max-deg",
            str(trace["close_50_max_deg"]),
        ]
    )
    if "close_75_max_deg" in trace:
        command.extend(
            ["--close-75-image-alignment-max-deg", str(trace["close_75_max_deg"])]
        )
    return command


def _inspection_gate_args(gates: list[dict[str, Any]]) -> list[str]:
    by_kind = {str(gate["kind"]): gate for gate in gates}
    geometry = by_kind.get("geometry_contact_alignment")
    if geometry is None:
        return []
    command = [
        "--edge-contact-parallel-success-threshold-deg",
        str(geometry["max_pre_close_error_deg"]),
    ]
    floor_clearance = by_kind.get("gripper_floor_clearance")
    if floor_clearance is not None:
        command.extend(
            [
                "--min-gripper-floor-clearance-m",
                str(floor_clearance["min_clearance_m"]),
            ]
        )
    visual = by_kind.get("camera2_visual_alignment")
    if visual is None:
        return [*command, "--close-alignment-gate-mode", "geometry_only"]
    command.extend(
        [
            "--close-alignment-gate-mode",
            str(visual["mode"]),
            "--pre-close-image-alignment-max-deg",
            str(visual["pre_close_max_deg"]),
            "--close-25-image-alignment-max-deg",
            str(visual["close_25_max_deg"]),
            "--close-50-image-alignment-max-deg",
            str(visual["close_50_max_deg"]),
        ]
    )
    if "close_75_max_deg" in visual:
        command.extend(
            ["--close-75-image-alignment-max-deg", str(visual["close_75_max_deg"])]
        )
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


def _sidecar_command(
    recipe: dict[str, Any], *, split_spec: dict[str, Any], python: str
) -> list[str]:
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


def _closed_loop_command(
    recipe: dict[str, Any], *, split_spec: dict[str, Any], python: str
) -> list[str]:
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
    train_bin_spec = _source_split_spec(recipe, train_spec)
    validation_bin_spec = _source_split_spec(recipe, validation_spec)
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
        _bin_counts_arg(train_bin_spec),
        "--expected-validation-bins",
        _bin_counts_arg(validation_bin_spec),
        "--expected-terminal-hold-steps",
        str(recipe["common"]["terminal_hold_steps"]),
        "--max-pre-close-alignment-deg",
        str(_max_pre_close_alignment_deg(recipe)),
        "--output",
        str(Path(validation) / "meta" / "split_overlap_audit.json"),
    ]
    return _append_lift_audit_args(command, audit)


def _source_split_spec(recipe: dict[str, Any], split_spec: dict[str, Any]) -> dict[str, Any]:
    if split_spec.get("kind") != "render_derivative":
        return split_spec
    return _render_source_spec(recipe, split_spec)


def _bin_counts_arg(split_spec: dict[str, Any]) -> str:
    if split_spec.get("expected_bins"):
        counts = {int(bin_id): int(count) for bin_id, count in split_spec["expected_bins"].items()}
        return ",".join(f"{bin_id}:{counts[bin_id]}" for bin_id in sorted(counts))
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
        str(_max_pre_close_alignment_deg(recipe)),
        "--output",
        str(Path(train_spec["output_root"]) / reference["output"]),
    ]
    return _append_lift_audit_args(command, audit)


def _max_pre_close_alignment_deg(recipe: dict[str, Any]) -> float:
    common = recipe["common"]
    for gate in common.get("inspection_gates", []):
        if gate["kind"] == "geometry_contact_alignment":
            return float(gate["max_pre_close_error_deg"])
    if "contact_alignment" in common:
        return float(common["contact_alignment"]["max_pre_close_error_deg"])
    return float(common["edge_contact_parallel_success_threshold_deg"])


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
        if split_spec.get("kind", "generated") != "generated":
            continue
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
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(exports) or 1)
        ) as pool:
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
