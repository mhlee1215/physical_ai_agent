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
from physical_ai_agent.so101_workspace_spawn_catalog import (
    load_workspace_spawn_catalog,
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

    _require_append_only_output_roots(
        recipe,
        split=args.split,
        overwrite=args.overwrite,
        reuse_complete_splits=args.reuse_complete_shards,
    )

    env = {**os.environ, "PYTHONPATH": _prepend_pythonpath(os.environ.get("PYTHONPATH", ""))}
    _run_stages(
        stages,
        env=env,
        workers=args.workers,
        reuse_complete_shards=args.reuse_complete_shards,
        recipe=recipe,
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
        if payload.get("format") in {
            "so101_workspace_spawn_catalog_v1",
            "so101_workspace_spawn_catalog_v2",
        }:
            catalog = load_workspace_spawn_catalog(path)
            if expected_qpos is not None and (
                len(catalog.home_qpos) != len(expected_qpos)
                or any(
                    not math.isclose(float(left), float(right), abs_tol=5e-6)
                    for left, right in zip(
                        catalog.home_qpos, expected_qpos, strict=True
                    )
                )
            ):
                raise ValueError(
                    f"workspace spawn catalog home qpos does not match recipe: {path}"
                )
            if expected_rig is not None and catalog.camera_rig_config != expected_rig:
                raise ValueError(
                    f"workspace spawn catalog camera rig does not match recipe: {path}"
                )
            common = recipe.get("common", {})
            if (
                common.get("target_object_color") is not None
                and catalog.object_color != common["target_object_color"]
            ):
                raise ValueError(
                    f"workspace spawn catalog object color does not match recipe: {path}"
                )
            common_sizes = [
                float(value)
                for value in str(common.get("object_half_sizes", "")).split(",")
                if value.strip()
            ]
            if common_sizes and common_sizes != [catalog.object_half_size_m]:
                raise ValueError(
                    f"workspace spawn catalog object size does not match recipe: {path}"
                )
            continue
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
        if (
            expected_rig_sha256 is not None
            and payload.get("camera_rig_sha256") != expected_rig_sha256
        ):
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
    recipe: dict[str, Any],
    *,
    split: str,
    overwrite: bool,
    reuse_complete_splits: bool = False,
) -> None:
    if overwrite:
        return
    selected = _selected_split_names(recipe, split)
    existing = []
    for name in selected:
        split_spec = recipe["splits"][name]
        output_root = Path(str(split_spec["output_root"]))
        if not output_root.exists():
            continue
        if (
            reuse_complete_splits
            and split_spec.get("kind", "generated") == "generated"
            and _generated_split_output_is_complete(split_spec)
        ):
            continue
        existing.append(output_root)
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
    if generated_selected:
        source = recipe.get("source") or {}
        for catalog_index, raw_catalog_path in enumerate(source.get("catalogs", [])):
            catalog_path = Path(raw_catalog_path)
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            if payload.get("format") != "so101_workspace_spawn_catalog_v2":
                continue
            stages.append(
                {
                    "name": f"workspace-distribution:{catalog_index}",
                    "command": [
                        python,
                        recipe["workspace_catalog_distribution_report_script"],
                        "--catalog",
                        str(catalog_path),
                        "--output-dir",
                        str(
                            Path(recipe["workspace_catalog_distribution_report_root"])
                            / recipe["name"]
                            / catalog_path.stem
                        ),
                    ],
                }
            )
    for lookup in recipe.get("lookup_builders", []) if generated_selected else []:
        stages.append(
            {
                "name": f"lookup:{lookup['name']}",
                "command": _lookup_builder_command(recipe, lookup=lookup, python=python),
            }
        )
    for split_name in selected:
        split_spec = recipe["splits"][split_name]
        if split_spec.get("kind", "generated") == "phase_subset":
            stages.append(
                {
                    "name": f"phase-subset:{split_name}",
                    "command": _phase_subset_command(
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
                    "command": _sidecar_command(
                        recipe,
                        split_spec=split_spec,
                        python=python,
                    ),
                }
            )
            if split_spec.get("closed_loop"):
                stages.append(
                    {
                        "name": f"closed-loop-starts:{split_name}",
                        "command": _closed_loop_command(
                            recipe,
                            recipe_path=recipe_path,
                            split_name=split_name,
                            split_spec=split_spec,
                            python=python,
                        ),
                    }
                )
            continue
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
                            recipe,
                            recipe_path=recipe_path,
                            split_name=split_name,
                            split_spec=split_spec,
                            python=python,
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
                            recipe_path=recipe_path,
                            split_name=split_name,
                            split_spec=split_spec,
                            source_spec=source_spec,
                            python=python,
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
                            recipe,
                            recipe_path=recipe_path,
                            split_name=split_name,
                            split_spec=split_spec,
                            python=python,
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
                    "command": _closed_loop_command(
                        recipe,
                        recipe_path=recipe_path,
                        split_name=split_name,
                        split_spec=split_spec,
                        python=python,
                    ),
                }
            )
    if int(recipe.get("schema_version", 1)) >= 2:
        for split_name in selected:
            stages.append(
                {
                    "name": f"distribution:{split_name}",
                    "command": _distribution_report_command(
                        recipe,
                        split_name=split_name,
                        split_spec=recipe["splits"][split_name],
                        python=python,
                        recipe_path=recipe_path,
                    ),
                }
            )
    if split == "all" and {"train", "validation"}.issubset(recipe["splits"]):
        audit_command = (
            _phase_subset_audit_command(recipe, python=python)
            if all(
                recipe["splits"][name].get("kind") == "phase_subset"
                for name in ("train", "validation")
            )
            else _audit_command(recipe, python=python)
        )
        stages.append({"name": "audit:train-vs-validation", "command": audit_command})
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
    recipe_path: Path,
    split_name: str,
    split_spec: dict[str, Any],
    source_spec: dict[str, Any],
    python: str,
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
        "--source-width",
        str(render.get("source_width") or render["width"]),
        "--source-height",
        str(render.get("source_height") or render["height"]),
        "--policy-resize",
        render.get("policy_resize", "direct_square_render"),
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
    if render.get("profile_from_camera_rig"):
        camera_rig_config = recipe["common"].get("camera_rig_config")
        if not camera_rig_config:
            raise ValueError(
                "profile_from_camera_rig requires common.camera_rig_config"
            )
        command.extend(["--camera-rig-config", camera_rig_config])
    command.append(
        "--preserve-pinhole-renders"
        if render.get("preserve_pinhole_renders", False)
        else "--no-preserve-pinhole-renders"
    )
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
    ]
    workspace_catalog = recipe["common"].get("workspace_spawn_catalog")
    if workspace_catalog:
        command.extend(
            [
                "--workspace-spawn-catalog",
                str(workspace_catalog),
                "--workspace-spawn-start-index",
                str(bin_spec["lookup_start_index"]),
                "--workspace-spawn-candidate-count",
                str(bin_spec["workspace_candidate_count"]),
            ]
        )
    else:
        command.extend(
            [
                "--grid-balance-target-per-bin",
                str(bin_spec["episodes"]),
                "--grid-balance-bins",
                str(bin_id),
                "--grid-lookup-start-index",
                str(bin_spec["lookup_start_index"]),
            ]
        )
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
        if key == "workspace_spawn_catalog":
            continue
        if key == "include_camera3_duplicate":
            if value is False:
                command.append("--no-camera3-duplicate")
            continue
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


def _distribution_report_command(
    recipe: dict[str, Any],
    *,
    split_name: str,
    split_spec: dict[str, Any],
    python: str,
    recipe_path: Path,
) -> list[str]:
    return [
        python,
        recipe["distribution_report_script"],
        "--dataset-root",
        split_spec["output_root"],
        "--recipe",
        str(recipe_path),
        "--split",
        split_name,
    ]


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


def _phase_subset_command(
    recipe: dict[str, Any],
    *,
    split_spec: dict[str, Any],
    python: str,
    overwrite: bool,
) -> list[str]:
    subset = split_spec["phase_subset"]
    command = [
        python,
        recipe["phase_subset_script"],
        "--output-root",
        split_spec["output_root"],
        "--repo-id",
        split_spec["repo_id"],
        "--phase-id",
        subset["phase_id"],
        "--prompt",
        subset["prompt"],
        "--entry-replay-qpos-rmse-max",
        str(subset["entry_replay_qpos_rmse_max"]),
    ]
    for source in subset["sources"]:
        command.extend(
            [
                "--source-spec",
                json.dumps(source, separators=(",", ":"), sort_keys=True),
            ]
        )
    if subset.get("reconstruct_sim_snapshots", True):
        command.extend(
            [
                "--reconstruct-sim-snapshots",
                "--target-object-color",
                str(recipe["common"]["target_object_color"]),
                "--object-half-sizes",
                str(recipe["common"]["object_half_sizes"]),
                "--camera-rig-config",
                str(recipe["common"]["camera_rig_config"]),
                "--spawn-center",
                f"{recipe['common']['spawn_center_x']},{recipe['common']['spawn_center_y']}",
                "--spawn-min-radius",
                str(recipe["common"]["spawn_min_radius"]),
                "--spawn-max-radius",
                str(recipe["common"]["spawn_max_radius"]),
                "--spawn-angle-half-range-deg",
                str(recipe["common"]["spawn_angle_half_range_deg"]),
            ]
        )
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
    initial_visibility = by_kind.get("initial_target_visibility")
    if initial_visibility is not None:
        camera_names = [
            str(value).removeprefix("observation.images.")
            for value in initial_visibility["camera_keys"]
        ]
        command.extend(
            [
                "--require-initial-target-visible",
                "--initial-target-visibility-cameras",
                ",".join(camera_names),
                "--initial-target-min-area-pixels",
                str(initial_visibility["min_area_pixels"]),
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
    recipe: dict[str, Any],
    *,
    recipe_path: Path,
    split_name: str,
    split_spec: dict[str, Any],
    python: str,
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
        (
            "auto"
            if loop.get("bin_selection", "declared") == "all_visible"
            else ",".join(str(value) for value in loop["bins"])
        ),
    ]
    sidecar = recipe["sidecar"]
    sidecar_stem = (
        f"{sidecar['camera_key'].replace('.', '_')}_"
        f"{sidecar['grid_size']}x{sidecar['grid_size']}_"
        f"frame{sidecar['frame_index']}.parquet"
    )
    command.extend(
        ["--grid-sidecar", str(root / "meta" / "camera_grid_bins" / sidecar_stem)]
    )
    for key, flag in (
        ("success_metric", "--success-metric"),
        ("lift_success_height", "--lift-success-height"),
    ):
        if key in loop:
            command.extend([flag, str(loop[key])])
    for source_report in loop.get("exclude_source_reports", []):
        command.extend(["--exclude-source-report", str(source_report)])
    if int(recipe.get("schema_version", 1)) >= 2:
        common = recipe["common"]
        closed_loop_splits = [
            name
            for name, spec in recipe["splits"].items()
            if isinstance(spec.get("closed_loop"), dict)
        ]
        test_case_id = (
            f"{recipe['name']}_loop_test"
            if len(closed_loop_splits) == 1
            else f"{recipe['name']}_{split_name}_loop_test"
        )
        command.extend(
            [
                "--write-executable-contract",
                "--contract-id",
                test_case_id,
                "--contract-description",
                (
                    f"Held-out {split_name} starts generated by {recipe['name']} "
                    "with the matching camera and environment contract."
                ),
                "--contract-steps",
                str(loop.get("steps", 200)),
                "--contract-seed",
                str(loop.get("seed", 98100)),
                "--task-prompt",
                str(recipe["audit"]["expected_prompt"]),
                "--start-dataset-name",
                Path(split_spec["output_root"]).name,
                "--start-dataset-root",
                str(split_spec["output_root"]),
                "--start-dataset-repo-id",
                str(split_spec["repo_id"]),
                "--source-recipe",
                str(recipe_path),
                "--source-split",
                split_name,
                "--camera-rig-config",
                str(common["camera_rig_config"]),
                "--target-object-color",
                str(common["target_object_color"]),
                "--object-half-sizes",
                str(common["object_half_sizes"]),
                "--spawn-center-x",
                str(common["spawn_center_x"]),
                "--spawn-center-y",
                str(common["spawn_center_y"]),
                "--spawn-min-radius",
                str(common["spawn_min_radius"]),
                "--spawn-max-radius",
                str(common["spawn_max_radius"]),
                "--spawn-angle-half-range-deg",
                str(common["spawn_angle_half_range_deg"]),
            ]
        )
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
        "--expected-terminal-hold-steps",
        str(recipe["common"]["terminal_hold_steps"]),
        "--max-pre-close-alignment-deg",
        str(_max_pre_close_alignment_deg(recipe)),
        "--output",
        str(Path(validation) / "meta" / "split_overlap_audit.json"),
    ]
    if recipe["common"].get("workspace_spawn_catalog"):
        command.extend(
            [
                "--expected-train-episodes",
                str(_split_episode_count(train_bin_spec)),
                "--expected-validation-episodes",
                str(_split_episode_count(validation_bin_spec)),
            ]
        )
    else:
        command.extend(
            [
                "--expected-train-bins",
                _bin_counts_arg(train_bin_spec),
                "--expected-validation-bins",
                _bin_counts_arg(validation_bin_spec),
            ]
        )
    return _append_lift_audit_args(command, audit)


def _phase_subset_audit_command(
    recipe: dict[str, Any],
    *,
    python: str,
) -> list[str]:
    audit = recipe["audit"]
    return [
        python,
        recipe["audit_script"],
        "--train-root",
        recipe["splits"]["train"]["output_root"],
        "--validation-root",
        recipe["splits"]["validation"]["output_root"],
        "--expected-prompt",
        audit["expected_prompt"],
        "--expected-resolution",
        "x".join(str(value) for value in audit["expected_resolution"]),
        "--output",
        str(
            Path(recipe["splits"]["validation"]["output_root"])
            / "meta"
            / "split_overlap_audit.json"
        ),
    ]


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


def _split_episode_count(split_spec: dict[str, Any]) -> int:
    if split_spec.get("expected_episodes") is not None:
        return int(split_spec["expected_episodes"])
    return sum(int(row["episodes"]) for row in split_spec.get("bins", []))


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
        "--expected-validation-bins",
        ",".join(f"{key}:{value}" for key, value in reference["reference_bins"].items()),
        "--expected-terminal-hold-steps",
        str(recipe["common"]["terminal_hold_steps"]),
        "--max-pre-close-alignment-deg",
        str(_max_pre_close_alignment_deg(recipe)),
        "--output",
        str(Path(train_spec["output_root"]) / reference["output"]),
    ]
    if recipe["common"].get("workspace_spawn_catalog"):
        command.extend(
            ["--expected-train-episodes", str(_split_episode_count(train_spec))]
        )
    else:
        command.extend(["--expected-train-bins", _bin_counts_arg(train_spec)])
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
    recipe: dict[str, Any] | None = None,
) -> None:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    index = 0
    while index < len(stages):
        stage_name = str(stages[index]["name"])
        if stage_name.startswith("render:") and workers > 1:
            render_stages = _partition_render_stage(stages[index], workers=workers)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(render_stages)
            ) as pool:
                futures = [pool.submit(_run_stage, stage, env) for stage in render_stages]
                for future in futures:
                    future.result()
            index += 1
            continue
        if not stages[index]["name"].startswith("export:"):
            _run_stage(stages[index], env)
            index += 1
            continue
        split_name = stages[index]["name"].split(":", 2)[1]
        if (
            reuse_complete_shards
            and recipe is not None
            and _generated_split_output_is_complete(recipe["splits"][split_name])
        ):
            print(f"[so101-dataset] reuse complete generated split {split_name}", flush=True)
            index = _skip_generated_split_stages(stages, index=index, split_name=split_name)
            continue
        end = index
        while end < len(stages) and stages[end]["name"].startswith(f"export:{split_name}:"):
            end += 1
        all_exports = list(stages[index:end])
        exports = list(all_exports)
        if reuse_complete_shards:
            pending = []
            for stage in exports:
                if _export_shard_is_complete(stage):
                    print(f"[so101-dataset] reuse {stage['name']}", flush=True)
                else:
                    pending.append(stage)
            exports = pending
        lookup_cache = None
        if exports and "--grid-lookup-cache" in exports[0]["command"]:
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
        if recipe is not None:
            _repair_cross_shard_workspace_spacing(
                all_exports,
                env=env,
                recipe=recipe,
            )
        index = end


def _repair_cross_shard_workspace_spacing(
    export_stages: list[dict[str, Any]],
    *,
    env: dict[str, str],
    recipe: dict[str, Any],
) -> None:
    catalog_path = recipe.get("common", {}).get("workspace_spawn_catalog")
    if not catalog_path or len(export_stages) < 2:
        return
    catalog = load_workspace_spawn_catalog(Path(str(catalog_path)))
    distribution = catalog.continuous_distribution
    minimum_spacing_m = (
        0.0 if distribution is None else float(distribution.minimum_spacing_m)
    )
    if minimum_spacing_m <= 0.0:
        return

    report_paths = [
        Path(_command_arg(stage["command"], "--root"))
        / "so101_lerobot_export_report.json"
        for stage in export_stages
    ]
    repairs_by_shard: dict[int, int] = {}
    max_repairs = max(1, len(export_stages) * 3)
    for _ in range(max_repairs + 1):
        violations = _cross_shard_workspace_spacing_violations(
            report_paths,
            minimum_spacing_m=minimum_spacing_m,
        )
        if not violations:
            print(
                "[so101-dataset] global workspace spacing passed "
                f"minimum_m={minimum_spacing_m:.9f} "
                f"repairs={sum(repairs_by_shard.values())}",
                flush=True,
            )
            return
        if sum(repairs_by_shard.values()) >= max_repairs:
            break

        violation = violations[0]
        repair_index = max(
            int(violation["left_shard_index"]),
            int(violation["right_shard_index"]),
        )
        repairs_by_shard[repair_index] = repairs_by_shard.get(repair_index, 0) + 1
        command = list(export_stages[repair_index]["command"])
        for index, report_path in enumerate(report_paths):
            if index == repair_index:
                continue
            command.extend(
                ["--workspace-spawn-forbidden-report", str(report_path)]
            )
        if "--overwrite" not in command:
            command.append("--overwrite")
        print(
            "[so101-dataset] repairing cross-shard workspace spacing "
            f"shard={repair_index} distance_m={violation['distance_m']:.9f} "
            f"minimum_m={minimum_spacing_m:.9f}",
            flush=True,
        )
        _run_stage(
            {
                "name": f"repair-spacing:{repair_index}",
                "command": command,
            },
            env,
        )

    remaining = _cross_shard_workspace_spacing_violations(
        report_paths,
        minimum_spacing_m=minimum_spacing_m,
    )
    raise RuntimeError(
        "cross-shard workspace spacing repair did not converge: "
        f"remaining={len(remaining)} repairs={repairs_by_shard}"
    )


def _cross_shard_workspace_spacing_violations(
    report_paths: list[Path],
    *,
    minimum_spacing_m: float,
) -> list[dict[str, Any]]:
    shard_positions: list[list[tuple[int, float, float]]] = []
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rows = []
        for episode_index, episode in enumerate(payload.get("episodes") or []):
            workspace_spawn = episode.get("workspace_spawn")
            world_xy = (
                None
                if not isinstance(workspace_spawn, dict)
                else workspace_spawn.get("world_xy_m")
            )
            if not isinstance(world_xy, list) or len(world_xy) != 2:
                raise ValueError(
                    "workspace shard report episode has no world_xy_m: "
                    f"{report_path} episode={episode_index}"
                )
            rows.append((episode_index, float(world_xy[0]), float(world_xy[1])))
        shard_positions.append(rows)

    violations = []
    for left_shard_index, left_rows in enumerate(shard_positions):
        for right_shard_index in range(left_shard_index + 1, len(shard_positions)):
            for left_episode_index, left_x, left_y in left_rows:
                for right_episode_index, right_x, right_y in shard_positions[
                    right_shard_index
                ]:
                    distance_m = math.hypot(left_x - right_x, left_y - right_y)
                    if distance_m >= minimum_spacing_m - 1e-12:
                        continue
                    violations.append(
                        {
                            "distance_m": distance_m,
                            "left_shard_index": left_shard_index,
                            "left_episode_index": left_episode_index,
                            "right_shard_index": right_shard_index,
                            "right_episode_index": right_episode_index,
                        }
                    )
    return sorted(violations, key=lambda row: float(row["distance_m"]))


def _command_arg(command: list[str], flag: str) -> str:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"stage command is missing {flag}") from exc


def _generated_split_output_is_complete(split_spec: dict[str, Any]) -> bool:
    root = Path(str(split_spec["output_root"]))
    report_path = root / "so101_lerobot_export_report.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = sum(int(item["episodes"]) for item in split_spec.get("bins", []))
    if int(report.get("exported_episodes", -1)) != expected:
        return False
    replay_manifest = root / "render_replay" / "manifest.json"
    grid_sidecar = (
        root
        / "meta"
        / "camera_grid_bins"
        / "observation_images_camera1_4x4_frame0.parquet"
    )
    distribution_dir = root / "meta" / "distribution"
    distribution_json = distribution_dir / "distribution.json"
    distribution_markdown = distribution_dir / "distribution.md"
    distribution_html = distribution_dir / "distribution.html"
    if not all(
        path.is_file()
        for path in (distribution_json, distribution_markdown, distribution_html)
    ):
        return False
    try:
        distribution = json.loads(distribution_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if distribution.get("gate", {}).get("status") != "passed":
        return False
    recorded_markdown_sha256 = (
        distribution.get("artifacts", {}).get("markdown_sha256")
    )
    if not recorded_markdown_sha256:
        return False
    actual_markdown_sha256 = hashlib.sha256(distribution_markdown.read_bytes()).hexdigest()
    return (
        replay_manifest.is_file()
        and grid_sidecar.is_file()
        and recorded_markdown_sha256 == actual_markdown_sha256
    )


def _skip_generated_split_stages(
    stages: list[dict[str, Any]],
    *,
    index: int,
    split_name: str,
) -> int:
    while index < len(stages):
        name = str(stages[index]["name"])
        if name.startswith(f"export:{split_name}:") or name in {
            f"merge:{split_name}",
            f"render-replay:{split_name}",
            f"sidecar:{split_name}",
            f"closed-loop-starts:{split_name}",
            f"distribution:{split_name}",
        }:
            index += 1
            continue
        break
    return index


def _partition_render_stage(
    stage: dict[str, Any],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    command = list(stage["command"])
    episodes_index = command.index("--episodes") + 1
    episodes = [value for value in command[episodes_index].split(",") if value]
    worker_count = min(max(1, int(workers)), len(episodes))
    if worker_count == 1:
        return [stage]
    partitions = [episodes[index::worker_count] for index in range(worker_count)]
    result = []
    for worker_index, partition in enumerate(partitions):
        worker_command = list(command)
        worker_command[episodes_index] = ",".join(partition)
        result.append(
            {
                **stage,
                "name": f"{stage['name']}:worker{worker_index}",
                "command": worker_command,
            }
        )
    return result


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
