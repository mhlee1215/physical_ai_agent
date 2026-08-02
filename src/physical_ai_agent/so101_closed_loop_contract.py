from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_training_config_schema import (
    ClosedLoopObservationRendererConfig,
    ClosedLoopTestCaseConfig,
)


CONTRACT_FORMAT = "so101_executable_loop_test_contract_v1"


def contract_path_for_start_report(start_report_path: str | Path) -> Path:
    report = Path(start_report_path)
    return report.with_name(f"{report.stem}.contract.json")


def observation_renderer_from_camera_rig(
    camera_rig_path: str | Path,
    *,
    camera_rig_config: str | None = None,
) -> dict[str, Any]:
    rig_path = Path(camera_rig_path)
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    render = rig.get("render")
    if not isinstance(render, dict):
        raise ValueError(f"camera rig has no render profile: {rig_path}")
    policy_size = int(render.get("policy_size") or 256)
    bevel_mm = render.get("bevel_width_mm_range")
    renderer: dict[str, Any] = {
        "mode": render.get("mode", "blender_cycles_live"),
        "render_policy_inference_only": False,
        "camera_keys": [
            "observation.images.camera1",
            "observation.images.camera2",
        ],
        "width": policy_size,
        "height": policy_size,
        "source_width": int(render.get("source_width") or policy_size),
        "source_height": int(render.get("source_height") or policy_size),
        "policy_resize": render.get("policy_resize", "direct_square_render"),
        "camera_rig_config": camera_rig_config or str(rig_path),
        "profile_from_camera_rig": True,
        "samples": int(render.get("samples") or 256),
        "denoise": bool(render.get("denoise", False)),
        "compute_device_type": render.get("compute_device_type"),
        "cycles_seed": int(render.get("cycles_seed") or 98200),
        "lighting_profile": render.get("lighting_profile", "directional_key_fill_rim_v4"),
        "key_light_power": float(render.get("key_light_power") or 0.0),
        "fill_light_power": float(render.get("fill_light_power") or 0.0),
        "world_strength": float(render.get("world_strength") or 0.0),
        "hdri_rotation_deg": float(
            render.get("hdri_rotation_degrees", render.get("hdri_rotation_deg", 0.0))
        ),
        "exposure": float(render.get("exposure") or 0.0),
        "color_management": render.get("color_management", "AgX"),
        "color_look": render.get("color_look", "AgX - Medium High Contrast"),
        "gamma": float(render.get("gamma") or 1.0),
        "output_format": render.get("output_format", "PNG"),
        "sample_clamp_indirect": float(render.get("sample_clamp_indirect") or 0.0),
        "background_wall": bool(render.get("background_wall", False)),
        "stable_tabletop": bool(render.get("stable_tabletop", True)),
        "scene_profile": render.get("scene_profile", "pbr_workshop_v4"),
        "robot_material": render.get("robot_material", "matte_pla"),
        "material_profile": str(render["material_profile"]),
        "camera_lens": float(render.get("camera_lens_mm", render.get("camera_lens", 48.0))),
        "asset_root": str(
            render.get("photoreal_asset_root", render.get("asset_root", "_workspace/photoreal_assets"))
        ),
        "blender_bin": str(render.get("blender_bin", "blender")),
        "max_mesh_geoms": int(render.get("max_mesh_geoms") or 128),
        # Intermediate pinhole images do not affect policy output and are not retained.
        "preserve_pinhole_renders": False,
        "bevel_segments": int(render.get("bevel_segments") or 3),
    }
    if isinstance(bevel_mm, list) and len(bevel_mm) == 2:
        renderer["bevel_width_range_m"] = [
            float(bevel_mm[0]) / 1000.0,
            float(bevel_mm[1]) / 1000.0,
        ]
    for key in ("lens_distortion", "visual_props", "lights"):
        if render.get(key) is not None:
            renderer[key] = render[key]
    return ClosedLoopObservationRendererConfig.model_validate(renderer).model_dump(
        mode="json",
        exclude_none=True,
    )


def build_executable_loop_test_contract(
    *,
    test_case_id: str,
    description: str,
    episodes: int,
    steps: int,
    seed: int,
    task_prompt: str,
    success_metric: str,
    start_report_path: str,
    start_dataset: dict[str, Any],
    env_config: dict[str, Any],
    observation_renderer: dict[str, Any],
    source_recipe: str,
    source_split: str,
) -> dict[str, Any]:
    test_case = ClosedLoopTestCaseConfig.model_validate(
        {
            "id": test_case_id,
            "description": description,
            "episodes": episodes,
            "steps": steps,
            "seed": seed,
            "start_contract": "dataset_episode_start",
            "task_prompt": task_prompt,
            "env_object_color": env_config["target_object_color"],
            "success_metric": success_metric,
            "start_report_path": start_report_path,
            "start_dataset": start_dataset,
            "env_config": env_config,
        }
    ).model_dump(mode="json", exclude_none=True)
    renderer = ClosedLoopObservationRendererConfig.model_validate(
        observation_renderer
    ).model_dump(mode="json", exclude_none=True)
    return {
        "format": CONTRACT_FORMAT,
        "schema_version": 1,
        "source_recipe": source_recipe,
        "source_split": source_split,
        "test_case": test_case,
        "observation_renderer": renderer,
    }


def load_executable_loop_test_contract(
    contract_path: str | Path,
    *,
    repo_root: Path | None = None,
    expected_start_report: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(contract_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != CONTRACT_FORMAT or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported loop-test contract: {path}")
    test_case = ClosedLoopTestCaseConfig.model_validate(payload.get("test_case"))
    renderer = ClosedLoopObservationRendererConfig.model_validate(
        payload.get("observation_renderer")
    )
    if expected_start_report is not None:
        root = (repo_root or Path.cwd()).resolve()

        def resolved(value: str | Path) -> Path:
            candidate = Path(value)
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        if resolved(str(test_case.start_report_path or "")) != resolved(expected_start_report):
            raise ValueError(
                f"loop-test contract start report mismatch: {path}"
            )
    normalized = dict(payload)
    normalized["test_case"] = test_case.model_dump(mode="json", exclude_none=True)
    normalized["observation_renderer"] = renderer.model_dump(
        mode="json",
        exclude_none=True,
    )
    return normalized


def write_executable_loop_test_contract(
    contract_path: str | Path,
    payload: dict[str, Any],
) -> Path:
    path = Path(contract_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
