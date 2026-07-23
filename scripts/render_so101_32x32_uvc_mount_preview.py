#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
from physical_ai_agent.sim.so101_wrist_camera_mount import (
    INTEGRATED_32X32_UVC_PRESET,
    prepare_integrated_32x32_uvc_robot_xml,
)

HOME_QPOS = (0.0, -np.pi / 2.0, np.pi / 2.0, 0.66, np.pi / 2.0, -0.17453)
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an SO101 canary with the integrated 32x32 UVC wrist mount."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mount-stl", type=Path)
    parser.add_argument("--seed", type=int, default=50_000_000)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    if args.mount_stl:
        import os

        os.environ["SO101_32X32_UVC_MOUNT_STL"] = str(args.mount_stl.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset = prepare_integrated_32x32_uvc_robot_xml(args.mount_stl)
    env = make_high_contrast_picklift_env(
        target_object_color="green",
        object_half_sizes=(0.015,),
        wrist_camera_mount_preset=INTEGRATED_32X32_UVC_PRESET,
    )
    renderers: list[Any] = []
    try:
        env.reset(seed=args.seed)
        _set_home_pose(env)
        _set_gripper_materials(env)
        views = _render_views(env, size=args.size, renderers=renderers)
        for name, pixels in views.items():
            Image.fromarray(pixels).save(args.output_dir / f"{name}.png")
        contact_sheet = _make_contact_sheet(views)
        preview_path = args.output_dir / "integrated_32x32_uvc_mount_preview.png"
        contact_sheet.save(preview_path)
        report = {
            "schema_version": 1,
            "status": "review_candidate_not_applied_to_existing_datasets",
            "preset": INTEGRATED_32X32_UVC_PRESET,
            "seed": args.seed,
            "qpos": list(HOME_QPOS),
            "camera1": "egocentric_cam",
            "camera2": "wrist_cam",
            "camera_resolution": [args.size, args.size],
            "camera2_pixel_rotation_degrees": 0,
            "fixed_jaw_visual_mesh": "wrist_cam_mount_32x32_uvc",
            "fixed_jaw_collision_mesh": "wrist_roll_follower_so101_v1",
            "asset_manifest": asset.manifest_path,
            "preview": str(preview_path.resolve()),
        }
        (args.output_dir / "preview_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    finally:
        for renderer in renderers:
            renderer.close()
        env.close()


def _set_home_pose(env: Any) -> None:
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    for name, value in zip(JOINT_NAMES, HOME_QPOS, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _set_gripper_materials(env: Any) -> None:
    import mujoco

    model = env.unwrapped.model
    material_colors = {
        "wrist_roll_follower_so101_v1_material": (0.03, 0.48, 0.10, 1.0),
        "moving_jaw_so101_v1_material": (0.95, 0.97, 1.0, 1.0),
    }
    for name, rgba in material_colors.items():
        material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        model.mat_rgba[material_id] = rgba


def _render_views(
    env: Any,
    *,
    size: int,
    renderers: list[Any],
) -> dict[str, np.ndarray]:
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    output: dict[str, np.ndarray] = {}
    for output_name, camera_name in (
        ("camera1_egocentric", "egocentric_cam"),
        ("camera2_wrist", "wrist_cam"),
    ):
        renderer = mujoco.Renderer(model, height=size, width=size)
        renderers.append(renderer)
        renderer.update_scene(data, camera=_make_camera(env, camera_name))
        output[output_name] = postprocess_camera_frame(camera_name, renderer.render())

    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    close_camera = mujoco.MjvCamera()
    close_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    close_camera.lookat[:] = data.xpos[gripper_id]
    close_camera.distance = 0.24
    close_camera.azimuth = 150.0
    close_camera.elevation = -18.0
    renderer = mujoco.Renderer(model, height=size, width=size)
    renderers.append(renderer)
    renderer.update_scene(data, camera=close_camera)
    output["external_mount"] = renderer.render()
    return output


def _make_contact_sheet(views: dict[str, np.ndarray]) -> Image.Image:
    labels = {
        "camera1_egocentric": "camera1 | egocentric",
        "camera2_wrist": "camera2 | STL lens axis",
        "external_mount": "external | integrated mount",
    }
    tiles = []
    font = ImageFont.load_default(size=18)
    for key in ("camera1_egocentric", "camera2_wrist", "external_mount"):
        image = Image.fromarray(views[key]).convert("RGB")
        tile = Image.new("RGB", (image.width, image.height + 36), (15, 23, 42))
        tile.paste(image, (0, 36))
        ImageDraw.Draw(tile).text((10, 9), labels[key], fill=(241, 245, 249), font=font)
        tiles.append(tile)
    sheet = Image.new(
        "RGB",
        (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)),
        (15, 23, 42),
    )
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    return sheet


if __name__ == "__main__":
    main()
