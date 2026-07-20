#!/usr/bin/env python3
"""Render selected SO101 LeRobot dataset rows through Blender Cycles Metal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from physical_ai_agent.sim.so101_camera_input import (
    EGOCENTRIC_CAMERA1_POSE,
    _make_camera,
    apply_brown_conrady_distortion,
    brown_conrady_overscan_fovy,
)

try:
    from render_so101_blender_probe import BLENDER_DRIVER, _pbr_paths, _write_photo_tabletop_texture
    from render_so101_mitsuba_probe import _export_mesh_geoms
except ModuleNotFoundError:
    from scripts.render_so101_blender_probe import (
        BLENDER_DRIVER,
        _pbr_paths,
        _write_photo_tabletop_texture,
    )
    from scripts.render_so101_mitsuba_probe import _export_mesh_geoms

BLENDER_BATCH_DRIVER = BLENDER_DRIVER.replace(
    'def main():\n    args = sys.argv[sys.argv.index("--") + 1:]\n    spec_path = Path(args[0])\n    spec = json.loads(spec_path.read_text())',
    'def render_one_spec(spec_path):\n    spec = json.loads(Path(spec_path).read_text())',
).replace(
    '\nif __name__ == "__main__":\n    main()\n',
    '''

def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    manifest_path = Path(args[0])
    manifest = json.loads(manifest_path.read_text())
    for spec_path in manifest["spec_paths"]:
        render_one_spec(spec_path)


if __name__ == "__main__":
    main()
''',
)


class LiveBlenderCyclesPolicyRenderer:
    """Render the current MuJoCo state with the dataset's Blender profile."""

    def __init__(self, *, output_dir: Path, config: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.driver_path = self.output_dir / "blender_driver.py"
        self.driver_path.write_text(BLENDER_DRIVER, encoding="utf-8")
        self.texture_path = _write_photo_tabletop_texture(self.output_dir / "tabletop_texture.png")
        self.asset_root = Path(str(self.config["asset_root"]))
        self.material_profile_path = Path(str(self.config["material_profile"]))
        self.robot_material_config = _load_robot_material_config(self.material_profile_path)
        configured_blender = str(self.config["blender_bin"])
        self.blender_bin = shutil.which(configured_blender) or configured_blender
        self.hdri_path = self.asset_root / "polyhaven" / "studio_small_08_2k.hdr"
        self.table_pbr_dir = self.asset_root / "ambientcg" / "Wood008_1K-JPG"
        self.plastic_pbr_dir = self.asset_root / "ambientcg" / "Plastic013A_1K-JPG"
        self.records: list[dict[str, Any]] = []

    def render(
        self,
        *,
        env: Any,
        mujoco_renderers: dict[str, Any],
        episode: int,
        seed: int,
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import mujoco
        import numpy as np

        frame_dir = self.output_dir / f"episode_{episode:03d}_seed_{seed}" / f"step_{step:04d}"
        mesh_dir = frame_dir / "ply"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        exported = _export_mesh_geoms(
            env.unwrapped.model,
            env.unwrapped.data,
            mesh_dir,
            max_mesh_geoms=int(self.config["max_mesh_geoms"]),
        )
        primitives = _export_primitive_geoms(env.unwrapped.model, env.unwrapped.data)
        camera_specs = _camera_specs_from_mujoco_scene(
            env,
            mujoco_renderers,
            camera_lens=float(self.config["camera_lens"]),
            width=int(self.config["width"]),
            height=int(self.config["height"]),
        )
        distortion_profiles = dict(self.config.get("lens_distortion", {}))
        render_camera_specs = _camera_specs_with_distortion_overscan(
            camera_specs,
            distortion_profiles=distortion_profiles,
            width=int(self.config["width"]),
            height=int(self.config["height"]),
        )
        extension = ".png" if self.config["output_format"] == "PNG" else ".jpg"
        image_paths = {
            camera_key: str(frame_dir / f"{_camera_slug(camera_key)}{extension}")
            for camera_key in self.config["camera_keys"]
        }
        renders = [
            {
                "image_path": str(Path(image_paths[camera_key]).resolve()),
                "camera": render_camera_specs[camera_key],
            }
            for camera_key in self.config["camera_keys"]
        ]
        renders.extend(
            {
                "image_path": str(Path(item["image_path"]).resolve()),
                "camera": dict(item["camera"]),
            }
            for item in self.config.get("extra_renders", [])
        )
        spec_path = frame_dir / "blender_scene_spec.json"
        spec = {
            "width": int(self.config["width"]),
            "height": int(self.config["height"]),
            "samples": int(self.config["samples"]),
            "denoise": bool(self.config["denoise"]),
            "compute_device_type": str(
                self.config.get("compute_device_type", "METAL")
            ),
            "cycles_seed": int(self.config["cycles_seed"]),
            "lighting_profile": str(self.config["lighting_profile"]),
            "key_light_power": float(self.config["key_light_power"]),
            "fill_light_power": float(self.config["fill_light_power"]),
            "world_strength": float(self.config["world_strength"]),
            "hdri_rotation_deg": float(self.config["hdri_rotation_deg"]),
            "exposure": float(self.config["exposure"]),
            "color_management": str(self.config["color_management"]),
            "color_look": str(self.config["color_look"]),
            "gamma": float(self.config["gamma"]),
            "output_format": str(self.config["output_format"]),
            "sample_clamp_indirect": float(self.config["sample_clamp_indirect"]),
            "background_wall": bool(self.config["background_wall"]),
            "stable_tabletop": bool(self.config["stable_tabletop"]),
            "scene_profile": str(self.config["scene_profile"]),
            "visual_props": (
                _visual_props_for_episode(seed)
                if self.config["scene_profile"] == "black_table_clutter"
                else []
            ),
            "robot_material": str(self.config["robot_material"]),
            "robot_material_config": self.robot_material_config,
            "camera_lens": float(self.config["camera_lens"]),
            "texture_path": str(self.texture_path.resolve()),
            "hdri_path": str(self.hdri_path.resolve()) if self.hdri_path.exists() else None,
            "table_pbr": _pbr_paths(self.table_pbr_dir, "Wood008_1K-JPG"),
            "plastic_pbr": _pbr_paths(self.plastic_pbr_dir, "Plastic013A_1K-JPG"),
            "image_path": str(Path(next(iter(image_paths.values()))).resolve()),
            "blender_report_path": str((frame_dir / "blender_device_report.json").resolve()),
            "meshes": [{**item, "path": str(Path(item["path"]).resolve())} for item in exported],
            "primitives": primitives,
            "target_site": None,
            "renders": renders,
        }
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
        rendered = _flush_blender_pending(
            [
                {
                    "spec_path": spec_path,
                    "frame_dir": frame_dir,
                    "mesh_dir": mesh_dir,
                    "camera_specs": camera_specs,
                    "image_paths": image_paths,
                    "rendered_item": {
                        "episode": int(episode),
                        "seed": int(seed),
                        "frame": int(step),
                        "image_paths": image_paths,
                        "mesh_geoms_exported": len(exported),
                        "primitive_geoms_exported": len(primitives),
                    },
                }
            ],
            output_dir=self.output_dir,
            driver_path=self.driver_path,
            blender_bin=self.blender_bin,
            duplicate_camera3_from_camera2=False,
            batch_mode=False,
        )[0]
        camera_postprocess = _apply_lens_distortion_to_images(
            image_paths,
            target_camera_specs=camera_specs,
            render_camera_specs=render_camera_specs,
            distortion_profiles=distortion_profiles,
            preserve_pinhole=bool(self.config.get("preserve_pinhole_renders", False)),
        )
        if camera_postprocess:
            rendered["camera_postprocess"] = camera_postprocess
        pixels = {
            "egocentric_cam": np.asarray(
                Image.open(image_paths["observation.images.camera1"]).convert("RGB")
            ).copy(),
            "wrist_cam": np.asarray(
                Image.open(image_paths["observation.images.camera2"]).convert("RGB")
            ).copy(),
        }
        metadata = {
            **rendered,
            "mode": "blender_cycles_live",
            "policy_inference_step": int(step),
        }
        self.records.append(metadata)
        return pixels, metadata

    def report(self) -> dict[str, Any]:
        render_seconds = [float(record.get("render_seconds", 0.0)) for record in self.records]
        return {
            "mode": "blender_cycles_live",
            "config": self.config,
            "render_count": len(self.records),
            "render_seconds": float(sum(render_seconds)),
            "records": self.records,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SO101 dataset row previews with Blender Cycles Metal.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_dataset_blender_preview"))
    parser.add_argument("--env-id", default="MuJoCoPickLift-v1")
    parser.add_argument(
        "--env-source",
        choices=("gym", "high_contrast_picklift"),
        default="gym",
        help="Environment source. Use high_contrast_picklift for export_so101_teacher_rollouts_lerobot datasets.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--episodes",
        help="Comma-separated episode indices. Overrides --episode when set.",
    )
    parser.add_argument(
        "--frames",
        default="0,22,44,66,87",
        help="Comma-separated frame indices or labels: start, open, grip, final, all.",
    )
    parser.add_argument("--camera-keys", default="observation.images.camera1,observation.images.camera2")
    parser.add_argument("--duplicate-camera3-from-camera2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed-base", type=int, help="Reset seed base. Defaults to seedNNN parsed from dataset root.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--cycles-seed", type=int, default=98200)
    parser.add_argument("--lighting-profile", choices=("studio_small_08", "flat"), default="studio_small_08")
    parser.add_argument("--key-light-power", type=float, default=42.0)
    parser.add_argument("--fill-light-power", type=float, default=5.0)
    parser.add_argument("--world-strength", type=float, default=0.28)
    parser.add_argument("--hdri-rotation-deg", type=float, default=35.0)
    parser.add_argument("--exposure", type=float, default=-1.3)
    parser.add_argument("--color-management", choices=("Filmic", "Standard", "AgX"), default="Filmic")
    parser.add_argument("--color-look", default="Medium High Contrast")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--output-format", choices=("PNG", "JPEG"), default="PNG")
    parser.add_argument(
        "--render-replay-sidecar",
        type=Path,
        help="Read exact frame geoms/cameras from render_replay instead of replaying MuJoCo actions.",
    )
    parser.add_argument("--robot-material", choices=("plastic", "matte_pla", "metal"), default="matte_pla")
    parser.add_argument(
        "--robot-material-config",
        type=Path,
        help="JSON material profile. Schema v2 maps independent robot parts to reusable materials.",
    )
    parser.add_argument(
        "--scene-profile",
        choices=("neutral", "black_table_clutter"),
        default="neutral",
        help="Blender-only tabletop and visual-prop profile. It never changes MuJoCo collisions.",
    )
    parser.add_argument("--camera-lens", type=float, default=48.0)
    parser.add_argument("--asset-root", type=Path, default=Path("_workspace/photoreal_assets"))
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--max-mesh-geoms", type=int, default=128)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing camera PNGs instead of rendering the same frame again.",
    )
    parser.add_argument(
        "--contact-sheet-limit",
        type=int,
        default=240,
        help="Maximum rendered frames to include in the preview contact sheet.",
    )
    parser.add_argument(
        "--blender-batch-size",
        type=int,
        default=1,
        help="Render this many frame scene specs per Blender process.",
    )
    args = parser.parse_args()

    blender_bin = shutil.which(args.blender_bin) or args.blender_bin
    episodes = _parse_int_csv(args.episodes) if args.episodes else [int(args.episode)]
    frame_tokens = [item.strip() for item in args.frames.split(",") if item.strip()]
    result = render_dataset_preview(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        env_id=args.env_id,
        env_source=args.env_source,
        episodes=episodes,
        frame_tokens=frame_tokens,
        camera_keys=[item.strip() for item in args.camera_keys.split(",") if item.strip()],
        duplicate_camera3_from_camera2=args.duplicate_camera3_from_camera2,
        seed_base=args.seed_base,
        width=args.width,
        height=args.height,
        samples=args.samples,
        denoise=args.denoise,
        cycles_seed=args.cycles_seed,
        lighting_profile=args.lighting_profile,
        key_light_power=args.key_light_power,
        fill_light_power=args.fill_light_power,
        world_strength=args.world_strength,
        hdri_rotation_deg=args.hdri_rotation_deg,
        exposure=args.exposure,
        color_management=args.color_management,
        color_look=args.color_look,
        gamma=args.gamma,
        output_format=args.output_format,
        render_replay_sidecar=args.render_replay_sidecar,
        robot_material=args.robot_material,
        robot_material_config_path=args.robot_material_config,
        scene_profile=args.scene_profile,
        camera_lens=args.camera_lens,
        asset_root=args.asset_root,
        blender_bin=blender_bin,
        max_mesh_geoms=args.max_mesh_geoms,
        skip_existing=args.skip_existing,
        contact_sheet_limit=args.contact_sheet_limit,
        blender_batch_size=args.blender_batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def render_dataset_preview(
    *,
    dataset_root: Path,
    output_dir: Path,
    env_id: str,
    env_source: str,
    episodes: list[int],
    frame_tokens: list[str],
    camera_keys: list[str],
    duplicate_camera3_from_camera2: bool,
    seed_base: int | None,
    width: int,
    height: int,
    samples: int,
    denoise: bool,
    cycles_seed: int,
    lighting_profile: str,
    key_light_power: float,
    fill_light_power: float,
    world_strength: float,
    hdri_rotation_deg: float,
    exposure: float,
    color_management: str,
    color_look: str,
    gamma: float,
    output_format: str,
    render_replay_sidecar: Path | None,
    robot_material: str,
    robot_material_config_path: Path | None,
    scene_profile: str,
    camera_lens: float,
    asset_root: Path,
    blender_bin: str,
    max_mesh_geoms: int,
    skip_existing: bool,
    contact_sheet_limit: int,
    blender_batch_size: int,
) -> dict[str, Any]:
    import gymnasium as gym
    import mujoco
    import numpy as np
    import pyarrow.parquet as pq
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    output_dir.mkdir(parents=True, exist_ok=True)
    robot_material_config = _load_robot_material_config(robot_material_config_path)
    episode_summaries = _episode_summaries(dataset_root)
    episode_frames = _resolve_episode_frames(
        dataset_root,
        episodes=episodes,
        frame_tokens=frame_tokens,
        episode_summaries=episode_summaries,
    )
    episode_frame_labels = _resolve_episode_frame_labels(
        dataset_root,
        episodes=episodes,
        frame_tokens=frame_tokens,
        episode_summaries=episode_summaries,
    )
    rows = _dataset_rows_for_replay(dataset_root, episode_frames=episode_frames)
    expected = sum(len(frames) for frames in episode_frames.values())
    selected = [
        row
        for row in rows
        if int(row["frame_index"]) in set(episode_frames[int(row["episode_index"])])
    ]
    if len(selected) != expected:
        found = [(int(row["episode_index"]), int(row["frame_index"])) for row in selected]
        raise ValueError(f"missing requested episode frames: requested={episode_frames}, found={found}")
    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)

    replay_manifest = None
    replay_frames: dict[tuple[int, int], dict[str, Any]] = {}
    if render_replay_sidecar is not None:
        from physical_ai_agent.so101_render_replay import load_render_replay_frames

        replay_manifest, replay_frames = load_render_replay_frames(render_replay_sidecar)
        missing_replay = [
            (episode, frame)
            for episode, frames in episode_frames.items()
            for frame in frames
            if (episode, frame) not in replay_frames
        ]
        if missing_replay:
            raise ValueError(f"render replay sidecar is missing requested frames: {missing_replay[:12]}")

    rendered: list[dict[str, Any]] = []
    driver_path = output_dir / "blender_driver.py"
    driver_path.write_text(BLENDER_BATCH_DRIVER if blender_batch_size > 1 else BLENDER_DRIVER, encoding="utf-8")
    pending_blender: list[dict[str, Any]] = []
    texture_path = _write_photo_tabletop_texture(output_dir / "tabletop_texture.png")
    hdri_path = asset_root / "polyhaven" / "studio_small_08_2k.hdr"
    table_pbr_dir = asset_root / "ambientcg" / "Wood008_1K-JPG"
    plastic_pbr_dir = asset_root / "ambientcg" / "Plastic013A_1K-JPG"
    output_extension = ".png" if output_format == "PNG" else ".jpg"

    env = None if render_replay_sidecar is not None else _make_env(env_id=env_id, env_source=env_source)
    mujoco_renderers: dict[str, Any] = {}
    try:
        if env is not None:
            for camera_name in ("egocentric_cam", "wrist_cam"):
                mujoco_renderers[camera_name] = mujoco.Renderer(env.unwrapped.model, height=height, width=width)
        for episode in sorted(rows_by_episode):
            seed = _seed_for_episode(
                episode,
                seed_base=seed_base,
                dataset_root=dataset_root,
                episode_summaries=episode_summaries,
            )
            if env is not None:
                env.reset(seed=seed)
                unwrapped = env.unwrapped
                actuator_ids = getattr(unwrapped, "_actuator_ids", None)
            else:
                unwrapped = None
                actuator_ids = None
            for row in rows_by_episode[episode]:
                frame_index = int(row["frame_index"])
                state = [float(value) for value in row["observation.state"]]
                action = [float(value) for value in row["action"]]
                replay_frame = replay_frames.get((episode, frame_index))
                if replay_frame is not None:
                    state_error = 0.0
                else:
                    replay_state = unwrapped.data.ctrl[actuator_ids] if actuator_ids is not None else unwrapped.data.ctrl
                    state_error = _state_replay_error(replay_state, state)

                if frame_index in episode_frames[episode]:
                    frame_dir = output_dir / f"episode_{episode:04d}_frame_{frame_index:04d}"
                    image_paths = {
                        camera_key: str(
                            frame_dir
                            / f"episode_{episode:04d}_frame_{frame_index:04d}_{_camera_slug(camera_key)}{output_extension}"
                        )
                        for camera_key in camera_keys
                    }
                    if duplicate_camera3_from_camera2 and "observation.images.camera2" in image_paths:
                        image_paths["observation.images.camera3"] = str(
                            frame_dir / f"episode_{episode:04d}_frame_{frame_index:04d}_camera3{output_extension}"
                        )
                    if skip_existing and all(Path(path).exists() for path in image_paths.values()):
                        rendered.append(
                            {
                                "episode": episode,
                                "seed": seed,
                                "frame": frame_index,
                                "frame_label": _frame_label(episode_frame_labels, episode=episode, frame=frame_index),
                                "timestamp": float(row["timestamp"]),
                                "state": state,
                                "action": action,
                                "replay_state_error": state_error,
                                "image_path": image_paths.get("observation.images.camera1") or next(iter(image_paths.values())),
                                "image_paths": image_paths,
                                "mesh_geoms_exported": 0,
                                "primitive_geoms_exported": 0,
                                "render_seconds": 0.0,
                                "skipped_existing": True,
                            }
                        )
                        if env is not None:
                            env.step(np.asarray(action, dtype=float))
                        continue
                    mesh_dir = frame_dir / "ply"
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    if replay_frame is not None:
                        from physical_ai_agent.so101_render_replay import sidecar_scene_items

                        assert replay_manifest is not None and render_replay_sidecar is not None
                        exported, primitives = sidecar_scene_items(
                            render_replay_sidecar, replay_manifest, replay_frame
                        )
                        camera_specs = replay_frame["camera_specs"]
                    else:
                        mujoco.mj_forward(unwrapped.model, unwrapped.data)
                        exported = _export_mesh_geoms(
                            unwrapped.model,
                            unwrapped.data,
                            mesh_dir,
                            max_mesh_geoms=max_mesh_geoms,
                        )
                        primitives = _export_primitive_geoms(unwrapped.model, unwrapped.data)
                        camera_specs = _camera_specs_from_mujoco_scene(
                            env,
                            mujoco_renderers,
                            camera_lens=camera_lens,
                        )
                    render_specs = []
                    for camera_key in camera_keys:
                        if camera_key not in camera_specs:
                            raise ValueError(f"unsupported camera key: {camera_key}")
                        image_path = Path(image_paths[camera_key])
                        render_specs.append({"image_path": str(image_path.resolve()), "camera": camera_specs[camera_key]})
                        image_paths[camera_key] = str(image_path)
                    blender_report_path = frame_dir / "blender_device_report.json"
                    spec_path = frame_dir / "blender_scene_spec.json"
                    spec = {
                        "width": width,
                        "height": height,
                        "samples": samples,
                        "denoise": denoise,
                        "cycles_seed": int(cycles_seed),
                        "lighting_profile": lighting_profile,
                        "key_light_power": float(key_light_power),
                        "fill_light_power": float(fill_light_power),
                        "world_strength": float(world_strength),
                        "hdri_rotation_deg": float(hdri_rotation_deg),
                        "exposure": float(exposure),
                        "color_management": color_management,
                        "color_look": color_look,
                        "gamma": float(gamma),
                        "output_format": output_format,
                        "sample_clamp_indirect": 0.85,
                        "background_wall": False,
                        "stable_tabletop": True,
                        "scene_profile": scene_profile,
                        "visual_props": _visual_props_for_episode(seed) if scene_profile == "black_table_clutter" else [],
                        "robot_material": robot_material,
                        "robot_material_config": robot_material_config,
                        "camera_lens": camera_lens,
                        "texture_path": str(texture_path.resolve()),
                        "hdri_path": str(hdri_path.resolve()) if hdri_path.exists() else None,
                        "table_pbr": _pbr_paths(table_pbr_dir, "Wood008_1K-JPG"),
                        "plastic_pbr": _pbr_paths(plastic_pbr_dir, "Plastic013A_1K-JPG"),
                        "image_path": str(Path(next(iter(image_paths.values()))).resolve()),
                        "blender_report_path": str(blender_report_path.resolve()),
                        "meshes": [{**item, "path": str(Path(item["path"]).resolve())} for item in exported],
                        "primitives": primitives,
                        "target_site": None,
                        "renders": render_specs,
                    }
                    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
                    pending_blender.append(
                        {
                            "spec_path": spec_path,
                            "frame_dir": frame_dir,
                            "mesh_dir": mesh_dir,
                            "camera_specs": camera_specs,
                            "image_paths": image_paths,
                            "rendered_item": {
                                "episode": episode,
                                "seed": seed,
                                "frame": frame_index,
                                "frame_label": _frame_label(episode_frame_labels, episode=episode, frame=frame_index),
                                "timestamp": float(row["timestamp"]),
                                "state": state,
                                "action": action,
                                "replay_state_error": state_error,
                                "image_path": image_paths.get("observation.images.camera1") or next(iter(image_paths.values())),
                                "image_paths": image_paths,
                                "mesh_geoms_exported": len(exported),
                                "primitive_geoms_exported": len(primitives),
                            },
                        }
                    )
                    if len(pending_blender) >= max(1, blender_batch_size):
                        rendered.extend(
                            _flush_blender_pending(
                                pending_blender,
                                output_dir=output_dir,
                                driver_path=driver_path,
                                blender_bin=blender_bin,
                                duplicate_camera3_from_camera2=duplicate_camera3_from_camera2,
                                batch_mode=blender_batch_size > 1,
                            )
                        )
                        pending_blender.clear()
                if env is not None:
                    env.step(np.asarray(action, dtype=float))
        if pending_blender:
            rendered.extend(
                _flush_blender_pending(
                    pending_blender,
                    output_dir=output_dir,
                    driver_path=driver_path,
                    blender_bin=blender_bin,
                    duplicate_camera3_from_camera2=duplicate_camera3_from_camera2,
                    batch_mode=blender_batch_size > 1,
                )
            )
            pending_blender.clear()
    finally:
        for renderer in mujoco_renderers.values():
            renderer.close()
        if env is not None:
            env.close()
    if len(rendered) != expected:
        raise RuntimeError(f"rendered {len(rendered)} frames, expected {expected}")

    contact_sheet_items = rendered[: max(0, contact_sheet_limit)] if contact_sheet_limit else []
    contact_sheet = (
        _write_contact_sheet(contact_sheet_items, output_dir / "so101_dataset_photoreal_contact_sheet.png")
        if contact_sheet_items
        else None
    )
    report = {
        "dataset_root": str(dataset_root),
        "env_id": env_id,
        "env_source": env_source,
        "episodes": episodes,
        "frame_tokens": frame_tokens,
        "camera_keys": camera_keys,
        "duplicate_camera3_from_camera2": duplicate_camera3_from_camera2,
        "episode_frames": episode_frames,
        "seed_base": seed_base if seed_base is not None else _seed_from_name(dataset_root),
        "renderer": "blender_cycles",
        "robot_material": robot_material,
        "robot_material_config": {
            "path": str(robot_material_config_path),
            "name": robot_material_config["name"],
            "schema_version": robot_material_config["schema_version"],
            "parts": list(robot_material_config["parts"]),
        }
        if robot_material_config is not None
        else None,
        "scene_profile": scene_profile,
        "visual_props_by_episode": {
            str(episode): _visual_props_for_episode(
                _seed_for_episode(
                    episode,
                    seed_base=seed_base,
                    dataset_root=dataset_root,
                    episode_summaries=episode_summaries,
                )
            )
            for episode in episodes
        }
        if scene_profile == "black_table_clutter"
        else {},
        "camera_lens": camera_lens,
        "samples": samples,
        "cycles_seed": int(cycles_seed),
        "denoise": denoise,
        "lighting_profile": lighting_profile,
        "key_light_power": float(key_light_power),
        "fill_light_power": float(fill_light_power),
        "world_strength": float(world_strength),
        "hdri_rotation_deg": float(hdri_rotation_deg),
        "exposure": float(exposure),
        "color_management": color_management,
        "color_look": color_look,
        "gamma": float(gamma),
        "output_format": output_format,
        "render_asset_checksums": _render_asset_checksums(
            texture_path=texture_path,
            hdri_path=hdri_path,
            table_pbr_dir=table_pbr_dir,
            plastic_pbr_dir=plastic_pbr_dir,
            material_profile=robot_material_config_path,
        ),
        "render_replay_sidecar": (
            str(render_replay_sidecar.resolve()) if render_replay_sidecar is not None else None
        ),
        "physics_replayed": render_replay_sidecar is None,
        "width": width,
        "height": height,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "contact_sheet_limit": contact_sheet_limit,
        "renders": rendered,
        "note": (
            "Frame geoms and camera transforms are read directly from render_replay; "
            "MuJoCo physics is not rerun."
            if render_replay_sidecar is not None
            else "Episode state is replayed from source actions after a recorded-seed reset."
        ),
    }
    (output_dir / "so101_dataset_blender_preview_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _render_asset_checksums(
    *,
    texture_path: Path,
    hdri_path: Path,
    table_pbr_dir: Path,
    plastic_pbr_dir: Path,
    material_profile: Path | None,
) -> dict[str, str]:
    paths = [texture_path]
    if material_profile is not None:
        paths.append(material_profile)
    if hdri_path.exists():
        paths.append(hdri_path)
    for directory in (table_pbr_dir, plastic_pbr_dir):
        if directory.exists():
            paths.extend(path for path in sorted(directory.rglob("*")) if path.is_file())
    return {str(path.resolve()): _sha256(path) for path in paths if path.is_file()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flush_blender_pending(
    pending: list[dict[str, Any]],
    *,
    output_dir: Path,
    driver_path: Path,
    blender_bin: str,
    duplicate_camera3_from_camera2: bool,
    batch_mode: bool,
) -> list[dict[str, Any]]:
    if not pending:
        return []
    started = time.perf_counter()
    if batch_mode:
        manifest_path = output_dir / f"blender_batch_{int(started * 1_000_000)}.json"
        manifest_path.write_text(
            json.dumps({"spec_paths": [str(item["spec_path"].resolve()) for item in pending]}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        command = [blender_bin, "--background", "--python", str(driver_path), "--", str(manifest_path)]
        log_path = output_dir / f"blender_batch_{manifest_path.stem.removeprefix('blender_batch_')}.log"
    else:
        command = [blender_bin, "--background", "--python", str(driver_path), "--", str(pending[0]["spec_path"])]
        log_path = pending[0]["frame_dir"] / "blender_render.log"
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    render_seconds = time.perf_counter() - started
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Blender render failed with exit code {completed.returncode}; see {log_path}")

    rendered: list[dict[str, Any]] = []
    seconds_per_item = render_seconds / max(1, len(pending))
    for item in pending:
        image_paths = item["image_paths"]
        camera_specs = item["camera_specs"]
        for camera_key, image_path in image_paths.items():
            if camera_key not in camera_specs:
                continue
            rotation = camera_specs[camera_key].get("rotation_degrees", 0)
            if rotation:
                _rotate_image(Path(image_path), int(rotation))
        if duplicate_camera3_from_camera2 and "observation.images.camera2" in image_paths:
            camera2_path = Path(image_paths["observation.images.camera2"])
            camera3_path = item["frame_dir"] / f"{item['frame_dir'].name}_camera3{camera2_path.suffix}"
            shutil.copyfile(camera2_path, camera3_path)
            image_paths["observation.images.camera3"] = str(camera3_path)
        shutil.rmtree(item["mesh_dir"], ignore_errors=True)
        rendered_item = dict(item["rendered_item"])
        rendered_item["render_seconds"] = seconds_per_item
        rendered.append(rendered_item)
    return rendered


def _dataset_rows_for_replay(dataset_root: Path, *, episode_frames: dict[int, list[int]]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"missing LeRobot parquet data files under {dataset_root}")
    table = pq.read_table(
        [str(path) for path in data_files],
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
    )
    max_frame_by_episode = {int(episode): max(int(frame) for frame in frames) for episode, frames in episode_frames.items()}
    rows = [
        row
        for row in _rows(table.to_pydict())
        if int(row["episode_index"]) in max_frame_by_episode
        and int(row["frame_index"]) <= max_frame_by_episode[int(row["episode_index"])]
    ]
    return sorted(rows, key=lambda row: (int(row["episode_index"]), int(row["frame_index"])))


def _state_replay_error(qpos: Any, state: list[float]) -> float:
    total = 0.0
    for observed, expected in zip(qpos, state, strict=False):
        delta = float(observed) - float(expected)
        total += delta * delta
    return total**0.5


def _export_primitive_geoms(model: Any, data: Any) -> list[dict[str, Any]]:
    import mujoco

    names = {
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    }
    primitives: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        kind = names.get(int(model.geom_type[geom_id]))
        if kind is None:
            continue
        rgba = _geom_rgba(model, geom_id)
        if len(rgba) >= 4 and rgba[3] <= 0.01:
            continue
        name = model.geom(geom_id).name or f"geom_{geom_id:03d}"
        semantic_color = "recorded_cube_color" if "cube" in name or "pick_slot" in name else None
        primitives.append(
            {
                "geom_id": geom_id,
                "name": name,
                "body_name": model.body(int(model.geom_bodyid[geom_id])).name,
                "type": kind,
                "position": [float(value) for value in data.geom_xpos[geom_id]],
                "xmat": [float(value) for value in data.geom_xmat[geom_id]],
                "size": [float(value) for value in model.geom_size[geom_id]],
                "rgba": rgba,
                "semantic_color": semantic_color,
            }
        )
    return primitives


def _geom_rgba(model: Any, geom_id: int) -> list[float]:
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        return [float(value) for value in model.mat_rgba[mat_id]]
    return [float(value) for value in model.geom_rgba[geom_id]]


def _make_env(*, env_id: str, env_source: str) -> Any:
    if env_source == "high_contrast_picklift":
        from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

        return make_high_contrast_picklift_env()
    import gymnasium as gym

    return gym.make(env_id, render_mode=None)


def _write_contact_sheet(rendered: list[dict[str, Any]], output_path: Path) -> Path:
    images = [Image.open(item["image_path"]).convert("RGB") for item in rendered]
    cell_w, cell_h, label_h = 320, 240, 38
    episode_counts: dict[int, int] = {}
    for item in rendered:
        episode_counts[int(item["episode"])] = episode_counts.get(int(item["episode"]), 0) + 1
    columns = max(1, max(episode_counts.values()) if episode_counts else 1)
    if len(images) <= 5:
        columns = len(images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, (cell_h + label_h) * rows), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, (image, item) in enumerate(zip(images, rendered, strict=True)):
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        col = index % columns
        row = index // columns
        x = col * cell_w + (cell_w - image.width) // 2
        y = row * (cell_h + label_h) + label_h + (cell_h - image.height) // 2
        sheet.paste(image, (x, y))
        label = str(item.get("frame_label") or item["frame"])
        draw.text(
            (col * cell_w + 10, row * (cell_h + label_h) + 12),
            f"ep {item['episode']} | {label} | frame {item['frame']}",
            fill=(25, 25, 25),
        )
    sheet.save(output_path)
    return output_path


def _seed_from_name(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        return 0
    return int(match.group(1))


def _visual_props_for_episode(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) ^ 0x5A17)
    anchors = [
        ("workshop_mug", "mug", (0.035, 0.315), (0.13, 0.22, 0.29), 0.58),
        ("steel_bottle", "bottle", (0.535, 0.300), (0.34, 0.37, 0.40), 0.26),
        ("masking_tape", "tape", (0.035, -0.105), (0.68, 0.56, 0.24), 0.72),
        ("bench_screwdriver", "screwdriver", (0.535, -0.085), (0.16, 0.20, 0.24), 0.52),
    ]
    props = []
    for name, kind, (x, y), color, roughness in anchors:
        props.append(
            {
                "name": name,
                "kind": kind,
                "position": [x + rng.uniform(-0.008, 0.008), y + rng.uniform(-0.008, 0.008)],
                "yaw_degrees": rng.uniform(-22.0, 22.0),
                "color": list(color),
                "roughness": roughness,
            }
        )
    return props


def _load_robot_material_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") == 2:
        _validate_robot_material_config_v2(config, path)
        return config
    parts = config.get("parts")
    default_part = config.get("default_part")
    if config.get("schema_version") != 1 or not isinstance(parts, dict) or default_part not in parts:
        raise ValueError(f"invalid robot material config: {path}")
    for name, material in parts.items():
        _validate_material_spec(material, label=f"part {name}", path=path)
    return config


def _validate_robot_material_config_v2(config: dict[str, Any], path: Path) -> None:
    materials = config.get("materials")
    parts = config.get("parts")
    default_material = config.get("default_material")
    if not isinstance(materials, dict) or not materials or default_material not in materials:
        raise ValueError(f"invalid robot material presets: {path}")
    if not isinstance(parts, dict) or not parts:
        raise ValueError(f"invalid robot material parts: {path}")
    for name, material in materials.items():
        _validate_material_spec(material, label=f"material {name}", path=path)
    allowed_selector_keys = {"body_names", "mesh_names", "primitive_names"}
    for name, part in parts.items():
        if not isinstance(part, dict) or part.get("material") not in materials:
            raise ValueError(f"invalid material assignment for part {name}: {path}")
        selectors = part.get("selectors")
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"missing selectors for part {name}: {path}")
        for rule in selectors:
            if not isinstance(rule, dict) or not rule or not set(rule).issubset(allowed_selector_keys):
                raise ValueError(f"invalid selector for part {name}: {path}")
            if any(
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value for value in values)
                for values in rule.values()
            ):
                raise ValueError(f"invalid selector values for part {name}: {path}")


def _validate_material_spec(material: Any, *, label: str, path: Path) -> None:
    if not isinstance(material, dict):
        raise ValueError(f"invalid {label}: {path}")
    color = material.get("base_color")
    if not isinstance(color, list) or len(color) != 3 or not all(0.0 <= float(value) <= 1.0 for value in color):
        raise ValueError(f"invalid base_color for {label}: {path}")
    for key in ("roughness", "metallic"):
        if not 0.0 <= float(material.get(key, -1.0)) <= 1.0:
            raise ValueError(f"invalid {key} for {label}: {path}")


def _rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    count = len(next(iter(columns.values()))) if columns else 0
    return [{key: value[index] for key, value in columns.items()} for index in range(count)]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _episode_summaries(dataset_root: Path) -> dict[int, dict[str, Any]]:
    report_path = dataset_root / "so101_lerobot_export_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes") or []
    return {index: episode for index, episode in enumerate(episodes) if isinstance(episode, dict)}


def _seed_for_episode(
    episode: int,
    *,
    seed_base: int | None,
    dataset_root: Path,
    episode_summaries: dict[int, dict[str, Any]],
) -> int:
    if episode in episode_summaries and "seed" in episode_summaries[episode]:
        return int(episode_summaries[episode]["seed"])
    return (seed_base if seed_base is not None else _seed_from_name(dataset_root)) + int(episode)


def _resolve_episode_frames(
    dataset_root: Path,
    *,
    episodes: list[int],
    frame_tokens: list[str],
    episode_summaries: dict[int, dict[str, Any]],
) -> dict[int, list[int]]:
    ranges = _episode_frame_ranges(dataset_root)
    episode_frames: dict[int, list[int]] = {}
    for episode in episodes:
        if episode not in ranges:
            raise ValueError(f"episode {episode} not found in {dataset_root}")
        first_frame, last_frame = ranges[episode]
        summary = episode_summaries.get(episode, {})
        if len(frame_tokens) == 1 and frame_tokens[0].lower() == "all":
            frames = list(range(first_frame, last_frame + 1))
        else:
            frames = [_resolve_frame_token(token, first_frame=first_frame, last_frame=last_frame, summary=summary) for token in frame_tokens]
        episode_frames[int(episode)] = sorted(dict.fromkeys(frames))
    return episode_frames


def _resolve_episode_frame_labels(
    dataset_root: Path,
    *,
    episodes: list[int],
    frame_tokens: list[str],
    episode_summaries: dict[int, dict[str, Any]],
) -> dict[int, dict[int, str]]:
    ranges = _episode_frame_ranges(dataset_root)
    labels: dict[int, dict[int, str]] = {}
    for episode in episodes:
        first_frame, last_frame = ranges[episode]
        summary = episode_summaries.get(episode, {})
        labels[episode] = {}
        for token in frame_tokens:
            if token.lower() == "all":
                labels[episode].update({frame: str(frame) for frame in range(first_frame, last_frame + 1)})
                labels[episode][first_frame] = "start"
                labels[episode][last_frame] = "final"
                continue
            frame = _resolve_frame_token(token, first_frame=first_frame, last_frame=last_frame, summary=summary)
            labels[episode][frame] = "final" if token.lower() == "last" else token.lower()
    return labels


def _episode_frame_ranges(dataset_root: Path) -> dict[int, tuple[int, int]]:
    import pyarrow.parquet as pq

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"missing LeRobot parquet data files under {dataset_root}")
    table = pq.read_table([str(path) for path in data_files], columns=["episode_index", "frame_index"])
    ranges: dict[int, tuple[int, int]] = {}
    columns = table.to_pydict()
    for episode, frame in zip(columns["episode_index"], columns["frame_index"]):
        episode = int(episode)
        frame = int(frame)
        current = ranges.get(episode)
        ranges[episode] = (frame, frame) if current is None else (min(current[0], frame), max(current[1], frame))
    return ranges


def _resolve_frame_token(token: str, *, first_frame: int, last_frame: int, summary: dict[str, Any]) -> int:
    lowered = token.lower()
    if lowered == "start":
        return first_frame
    if lowered in {"final", "last"}:
        return last_frame
    phase_counts = summary.get("phase_counts") if isinstance(summary, dict) else None
    if lowered == "open" and isinstance(phase_counts, dict):
        return min(last_frame, first_frame + int(phase_counts.get("approach", 0)) + int(phase_counts.get("settle", 0)) - 1)
    if lowered == "grip" and isinstance(phase_counts, dict):
        return min(
            last_frame,
            first_frame
            + int(phase_counts.get("approach", 0))
            + int(phase_counts.get("settle", 0))
            + int(phase_counts.get("close", 0))
            - 1,
        )
    return int(token)


def _frame_label(episode_frame_labels: dict[int, dict[int, str]], *, episode: int, frame: int) -> str:
    return episode_frame_labels.get(int(episode), {}).get(int(frame), str(frame))


def _camera_specs_with_distortion_overscan(
    camera_specs: dict[str, dict[str, Any]],
    *,
    distortion_profiles: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, dict[str, Any]]:
    render_specs = {key: dict(spec) for key, spec in camera_specs.items()}
    for camera_key, profile in distortion_profiles.items():
        if camera_key not in render_specs:
            continue
        if profile.get("model") != "opencv_brown_conrady":
            model_name = profile.get("model")
            raise ValueError(
                f"unsupported lens distortion model for {camera_key}: {model_name}"
            )
        coefficients = tuple(float(value) for value in profile["coefficients"])
        target_fovy = float(camera_specs[camera_key]["fovy"])
        source_fovy = brown_conrady_overscan_fovy(
            width=width,
            height=height,
            target_fovy_degrees=target_fovy,
            coefficients=coefficients,
        )
        render_focal = 0.5 * height / math.tan(math.radians(source_fovy) / 2.0)
        render_specs[camera_key].update(
            {
                "fovy": source_fovy,
                "intrinsics": {
                    "fx": render_focal,
                    "fy": render_focal,
                    "cx": width / 2.0,
                    "cy": height / 2.0,
                    "width": width,
                    "height": height,
                },
                "distortion": None,
                "overscan_for_distortion": dict(profile),
            }
        )
        camera_specs[camera_key]["distortion"] = dict(profile)
    return render_specs


def _apply_lens_distortion_to_images(
    image_paths: dict[str, str],
    *,
    target_camera_specs: dict[str, dict[str, Any]],
    render_camera_specs: dict[str, dict[str, Any]],
    distortion_profiles: dict[str, Any],
    preserve_pinhole: bool,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    records: dict[str, dict[str, Any]] = {}
    for camera_key, profile in distortion_profiles.items():
        image_path_value = image_paths.get(camera_key)
        if image_path_value is None:
            continue
        image_path = Path(image_path_value)
        if preserve_pinhole:
            pinhole_path = image_path.with_name(f"{image_path.stem}_pinhole{image_path.suffix}")
            shutil.copyfile(image_path, pinhole_path)
        else:
            pinhole_path = None
        source = np.asarray(Image.open(image_path).convert("RGB"))
        coefficients = tuple(float(value) for value in profile["coefficients"])
        distorted = apply_brown_conrady_distortion(
            source,
            target_fovy_degrees=float(target_camera_specs[camera_key]["fovy"]),
            source_fovy_degrees=float(render_camera_specs[camera_key]["fovy"]),
            coefficients=coefficients,
        )
        Image.fromarray(distorted).save(image_path)
        records[camera_key] = {
            **dict(profile),
            "target_fovy_degrees": float(target_camera_specs[camera_key]["fovy"]),
            "render_overscan_fovy_degrees": float(render_camera_specs[camera_key]["fovy"]),
            "pinhole_image_path": str(pinhole_path) if pinhole_path else None,
        }
    return records


def _camera_specs_from_mujoco_scene(
    env: Any,
    renderers: dict[str, Any],
    *,
    camera_lens: float,
    width: int = 256,
    height: int = 256,
) -> dict[str, dict[str, Any]]:
    unwrapped = env.unwrapped
    camera1 = _scene_camera_spec(
        unwrapped.model,
        unwrapped.data,
        renderers["egocentric_cam"],
        "egocentric_cam",
    )
    camera1["rotation_degrees"] = (
        0
        if _named_camera_exists(unwrapped.model, "egocentric_cam")
        else int(EGOCENTRIC_CAMERA1_POSE["rotation_degrees"])
    )
    return {
        "observation.images.camera1": _with_explicit_camera_contract(
            camera1, role="egocentric_cam", width=width, height=height
        ),
        "observation.images.camera2": _with_explicit_camera_contract(
            _fixed_mujoco_camera_spec(
                unwrapped.model,
                unwrapped.data,
                "wrist_cam",
            ),
            role="wrist_cam",
            width=width,
            height=height,
        ),
    }


def _with_explicit_camera_contract(
    spec: dict[str, Any], *, role: str, width: int, height: int
) -> dict[str, Any]:
    import math

    import numpy as np

    fovy_rad = math.radians(float(spec["fovy"]))
    focal = 0.5 * float(height) / math.tan(0.5 * fovy_rad)
    forward = np.asarray(spec["forward"], dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.asarray(spec["up"], dtype=float)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4, dtype=float)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = np.asarray(spec["location"], dtype=float)
    return {
        **spec,
        "role": role,
        "world_from_camera": matrix.reshape(-1).tolist(),
        "intrinsics": {
            "fx": focal,
            "fy": focal,
            "cx": float(width) / 2.0,
            "cy": float(height) / 2.0,
            "width": int(width),
            "height": int(height),
        },
        "clip_end": float(spec.get("clip_end", 10.0)),
        "distortion": None,
    }


def _scene_camera_spec(model: Any, data: Any, renderer: Any, camera_name: str) -> dict[str, Any]:
    renderer.update_scene(data, camera=_make_camera(_EnvView(model=model, data=data), camera_name))
    scene = getattr(renderer, "scene", None) or getattr(renderer, "_scene", None)
    if scene is None:
        raise RuntimeError("MuJoCo renderer scene is unavailable")
    camera = scene.camera[0]
    return {
        "mode": "forward_up",
        "location": [float(value) for value in camera.pos],
        "forward": [float(value) for value in camera.forward],
        "up": [float(value) for value in camera.up],
        "fovy": _camera_fovy(model, camera_name),
        "focus_distance": 0.63 if camera_name == "egocentric_cam" else 0.20,
        "aperture_fstop": 10.0,
        "use_dof": False,
        "clip_start": 0.001,
    }


def _fixed_mujoco_camera_spec(model: Any, data: Any, camera_name: str) -> dict[str, Any]:
    camera_id = _camera_id(model, camera_name)
    xmat = [float(value) for value in data.cam_xmat[camera_id]]
    rows = [xmat[0:3], xmat[3:6], xmat[6:9]]
    columns = [
        [rows[0][0], rows[1][0], rows[2][0]],
        [rows[0][1], rows[1][1], rows[2][1]],
        [rows[0][2], rows[1][2], rows[2][2]],
    ]
    forward = [-float(value) for value in columns[2]]
    up = [float(value) for value in columns[1]]
    return {
        "mode": "forward_up",
        "location": [float(value) for value in data.cam_xpos[camera_id]],
        "forward": forward,
        "up": up,
        "fovy": _camera_fovy(model, camera_name),
        "focus_distance": 0.20,
        "aperture_fstop": 10.0,
        "use_dof": False,
        "clip_start": 0.001,
    }


def _camera_fovy(model: Any, camera_name: str) -> float:
    for index in range(model.ncam):
        if model.camera(index).name == camera_name:
            return float(model.cam_fovy[index])
    return float(model.vis.global_.fovy)


def _named_camera_exists(model: Any, camera_name: str) -> bool:
    return any(model.camera(index).name == camera_name for index in range(model.ncam))


def _camera_id(model: Any, camera_name: str) -> int:
    for index in range(model.ncam):
        if model.camera(index).name == camera_name:
            return int(index)
    raise ValueError(f"unknown MuJoCo camera: {camera_name}")


class _EnvView:
    def __init__(self, *, model: Any, data: Any) -> None:
        self.unwrapped = self
        self.model = model
        self.data = data


def _camera_slug(camera_key: str) -> str:
    return camera_key.removeprefix("observation.images.")


def _rotate_image(path: Path, degrees: int) -> None:
    if degrees % 360 == 0:
        return
    image = Image.open(path).convert("RGB")
    if degrees % 360 == 90:
        image = image.transpose(Image.Transpose.ROTATE_270)
    elif degrees % 360 == 180:
        image = image.transpose(Image.Transpose.ROTATE_180)
    elif degrees % 360 == 270:
        image = image.transpose(Image.Transpose.ROTATE_90)
    else:
        image = image.rotate(-degrees, expand=False)
    image.save(path)


if __name__ == "__main__":
    main()
