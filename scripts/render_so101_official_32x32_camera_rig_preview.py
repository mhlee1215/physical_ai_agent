#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_so101_dataset_blender_preview import LiveBlenderCyclesPolicyRenderer
from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

from physical_ai_agent.sim.so101_camera_rig_render_config import (
    FreeEvidenceViewConfig,
    MountedEvidenceViewConfig,
    SO101CameraRigRenderConfig,
    config_sha256,
    load_so101_camera_rig_render_config,
    resolve_repository_path,
)
from physical_ai_agent.sim.so101_overhead_camera_mount import (
    prepare_official_32x32_uvc_camera_rig_xml,
)

DEFAULT_CONFIG_PATH = Path(
    "configs/so101/camera_rigs/official_32x32_uvc_photoreal_v9_white_mount_locked.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the config-defined SO101 camera rig.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--debug-camera-origins",
        action="store_true",
        help="Render red pinhole markers and forward optical-axis rods.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_so101_camera_rig_render_config(config_path)
    _validate_render_dependencies(config)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else resolve_repository_path(config.render.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_expected_outputs(output_dir, config)
    asset = prepare_official_32x32_uvc_camera_rig_xml(rig_config=config)
    env = make_high_contrast_picklift_env(
        target_object_color=config.environment.target_object_color,
        object_half_sizes=config.environment.object_half_sizes_m,
        spawn_center=config.environment.spawn_center_xy_m,
        spawn_min_radius=config.environment.spawn_min_radius_m,
        spawn_max_radius=config.environment.spawn_max_radius_m,
        spawn_angle_half_range_deg=config.environment.spawn_angle_half_range_degrees,
        camera_rig_preset=config.preset,
        camera_rig_config=config,
    )
    mujoco_renderers: dict[str, Any] = {}
    try:
        env.reset(seed=config.environment.seed)
        _set_home_pose(env, config)
        env.unwrapped.model.vis.global_.offwidth = max(
            int(env.unwrapped.model.vis.global_.offwidth), config.render.source_width
        )
        env.unwrapped.model.vis.global_.offheight = max(
            int(env.unwrapped.model.vis.global_.offheight), config.render.source_height
        )
        mujoco_renderers = _make_mujoco_renderers(
            env,
            width=config.render.source_width,
            height=config.render.source_height,
        )
        evidence_renders = _evidence_renders(
            env=env,
            config=config,
            output_dir=output_dir,
        )
        renderer = LiveBlenderCyclesPolicyRenderer(
            output_dir=output_dir / config.render.outputs.photoreal_subdir,
            config=_photoreal_config(
                config,
                evidence_renders=tuple(evidence_renders.values()),
                debug_primitives=(
                    _camera_origin_debug_primitives(env)
                    if args.debug_camera_origins
                    else ()
                ),
            ),
        )
        pixels, metadata = renderer.render(
            env=env,
            mujoco_renderers=mujoco_renderers,
            episode=0,
            seed=config.environment.seed,
            step=0,
        )
        for evidence_path, _camera_spec in evidence_renders.values():
            if not evidence_path.is_file():
                raise RuntimeError(
                    f"Cycles batch did not produce camera-rig evidence: {evidence_path}"
                )
        camera1_policy = _policy_input(
            pixels["egocentric_cam"],
            size=config.render.policy_size,
            mode=config.render.policy_resize,
        )
        camera2_policy = _policy_input(
            pixels["wrist_cam"],
            size=config.render.policy_size,
            mode=config.render.policy_resize,
        )
        camera1_policy_path = output_dir / config.render.outputs.camera1_policy_filename
        camera2_policy_path = output_dir / config.render.outputs.camera2_policy_filename
        Image.fromarray(camera1_policy).save(camera1_policy_path)
        Image.fromarray(camera2_policy).save(camera2_policy_path)
        contact_sheet_path = output_dir / config.render.outputs.contact_sheet.filename
        external_path = evidence_renders["external_scene"][0]
        _write_contact_sheet(
            camera1=pixels["egocentric_cam"],
            camera2=pixels["wrist_cam"],
            external=np.asarray(Image.open(external_path).convert("RGB")),
            output_path=contact_sheet_path,
            config=config,
        )
        report = {
            "schema_version": 1,
            "status": config.status,
            "preset": config.preset,
            "config_path": str(config_path),
            "config_sha256": config_sha256(config_path),
            "validated_config": config.model_dump(mode="json"),
            "seed": config.environment.seed,
            "home_qpos": list(config.robot.home_qpos),
            "camera_contract": config.camera_contract,
            "camera1": {
                "mount": "official overhead 32x32 UVC STL assembly",
                "position_world": list(asset.camera1_position_world),
                "forward_world": list(asset.camera1_forward_world),
                "up_world": list(asset.camera1_up_world),
                "mount_face_center_cad": list(
                    config.camera1.camera_mount_face_center_cad_m
                ),
                "board_contact_center_cad": list(
                    config.camera1.camera_board_contact_center_cad_m
                ),
                "mount_plate_thickness_m": config.camera1.mount_plate_thickness_m,
                "lens_protrusion_m": config.camera1.camera_pinhole_protrusion_m,
                "assembly_mode": config.camera1.assembly_mode,
                "downward_angle_degrees": config.camera1.camera_downward_angle_degrees,
                "fovy_degrees": asset.camera1_fovy_degrees,
                "horizontal_fov_degrees": config.sensor.horizontal_fov_degrees,
                "pixel_rotation_degrees": config.camera1.pixel_postprocess_rotation_degrees,
                "distortion": _distortion_profile(config),
            },
            "camera2": {
                "mount": "official integrated 32x32 UVC fixed-jaw replacement",
                "fovy_degrees": (
                    config.camera2.effective_vertical_fov_degrees
                    or config.sensor.vertical_fov_degrees
                ),
                "sensor_vertical_fov_degrees": config.sensor.vertical_fov_degrees,
                "horizontal_fov_degrees": config.sensor.horizontal_fov_degrees,
                "pixel_rotation_degrees": config.camera2.pixel_postprocess_rotation_degrees,
                "position_gripper": list(config.camera2.camera_position_gripper),
                "mount_face_center_gripper": list(
                    config.camera2.mount_face_center_gripper_m
                ),
                "board_contact_center_gripper": list(
                    config.camera2.board_contact_center_gripper_m
                ),
                "mount_plate_thickness_m": config.camera2.mount_plate_thickness_m,
                "lens_protrusion_m": config.camera2.lens_protrusion_m,
                "assembly_mode": config.camera2.assembly_mode,
                "downward_angle_degrees": config.camera2.optical_downward_angle_degrees,
                "optical_axis_offset_degrees": abs(
                    config.camera2.optical_downward_angle_degrees
                    - config.camera2.mount_downward_angle_degrees
                ),
                "optical_target_gripper": list(config.camera2.optical_target_gripper),
                "distortion": _distortion_profile(config),
            },
            "camera_source_resolution": list(config.sensor.source_resolution),
            "preview_source_resolution": [
                config.render.source_width,
                config.render.source_height,
            ],
            "policy_resolution": [config.render.policy_size, config.render.policy_size],
            "policy_resize": config.render.policy_resize,
            "material_profile": str(
                resolve_repository_path(config.render.material_profile).resolve()
            ),
            "render_mode": "Blender Cycles Metal",
            "samples": config.render.samples,
            "asset_manifest": asset.manifest_path,
            "contact_sheet": str(contact_sheet_path.resolve()),
            "camera1_policy_input": str(camera1_policy_path.resolve()),
            "camera2_policy_input": str(camera2_policy_path.resolve()),
            "evidence_renders": {
                name: str(path.resolve())
                for name, (path, _camera_spec) in evidence_renders.items()
            },
            "live_renderer_metadata": metadata,
            "live_renderer_report": renderer.report(),
            "debug_camera_origins": bool(args.debug_camera_origins),
        }
        report_path = output_dir / config.render.outputs.report_filename
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        for mujoco_renderer in mujoco_renderers.values():
            mujoco_renderer.close()
        env.close()


def _set_home_pose(env: Any, config: SO101CameraRigRenderConfig) -> None:
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    for name, value in zip(
        config.robot.joint_names,
        config.robot.home_qpos,
        strict=True,
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _make_mujoco_renderers(env: Any, *, width: int, height: int) -> dict[str, Any]:
    import mujoco

    return {
        name: mujoco.Renderer(env.unwrapped.model, height=height, width=width)
        for name in ("egocentric_cam", "wrist_cam")
    }


def _remove_stale_expected_outputs(
    output_dir: Path,
    config: SO101CameraRigRenderConfig,
) -> None:
    outputs = config.render.outputs
    expected = {
        output_dir / outputs.camera1_policy_filename,
        output_dir / outputs.camera2_policy_filename,
        output_dir / outputs.report_filename,
        output_dir / outputs.contact_sheet.filename,
        *(output_dir / view.filename for view in config.render.evidence_views),
    }
    frame_dir = (
        output_dir
        / outputs.photoreal_subdir
        / f"episode_000_seed_{config.environment.seed:08d}"
        / "step_0000"
    )
    expected.update(
        frame_dir / filename
        for filename in (
            "camera1.png",
            "camera1_pinhole.png",
            "camera2.png",
            "camera2_pinhole.png",
        )
    )
    for path in expected:
        path.unlink(missing_ok=True)


def _photoreal_config(
    config: SO101CameraRigRenderConfig,
    *,
    evidence_renders: tuple[tuple[Path, dict[str, Any]], ...],
    debug_primitives: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    render = config.render
    return {
        "mode": render.mode,
        "render_policy_inference_only": render.render_policy_inference_only,
        "camera_keys": list(config.camera_contract),
        "width": render.source_width,
        "height": render.source_height,
        "samples": render.samples,
        "denoise": render.denoise,
        "cycles_seed": render.cycles_seed,
        "lighting_profile": render.lighting_profile,
        "key_light_power": render.key_light_power,
        "fill_light_power": render.fill_light_power,
        "world_strength": render.world_strength,
        "hdri_rotation_deg": render.hdri_rotation_degrees,
        "exposure": render.exposure,
        "color_management": render.color_management,
        "color_look": render.color_look,
        "gamma": render.gamma,
        "output_format": render.output_format,
        "sample_clamp_indirect": render.sample_clamp_indirect,
        "background_wall": render.background_wall,
        "stable_tabletop": render.stable_tabletop,
        "scene_profile": render.scene_profile,
        "robot_material": render.robot_material,
        "material_profile": str(resolve_repository_path(render.material_profile)),
        "camera_lens": render.camera_lens_mm,
        "asset_root": str(resolve_repository_path(render.photoreal_asset_root)),
        "blender_bin": render.blender_bin,
        "compute_device_type": render.compute_device_type,
        "max_mesh_geoms": render.max_mesh_geoms,
        "bevel_width_range_m": (
            [value / 1000.0 for value in render.bevel_width_mm_range]
            if render.bevel_width_mm_range is not None
            else None
        ),
        "bevel_segments": render.bevel_segments,
        "visual_props": (
            [
                {
                    "kind": "blend_asset",
                    "name": asset.name,
                    "blend_path": str(resolve_repository_path(asset.blend.path).resolve()),
                    "object_name": asset.object_name,
                    "position": list(asset.position_m),
                    "rotation_euler_degrees": list(asset.rotation_euler_degrees),
                    "scale_xyz": list(asset.scale_xyz),
                }
                for asset in render.scene_assets
            ]
            or None
        ),
        "lights": [light.model_dump(mode="json") for light in render.lights],
        "lens_distortion": {
            camera_key: _distortion_profile(config)
            for camera_key in config.camera_contract
        },
        "preserve_pinhole_renders": render.preserve_pinhole_renders,
        "extra_renders": [
            {
                "image_path": str(image_path.resolve()),
                "camera": camera_spec,
            }
            for image_path, camera_spec in evidence_renders
        ],
        "debug_primitives": [dict(item) for item in debug_primitives],
    }


def _camera_origin_debug_primitives(env: Any) -> tuple[dict[str, Any], ...]:
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    primitives: list[dict[str, Any]] = []
    for index, camera_name in enumerate(("egocentric_cam", "wrist_cam"), start=1):
        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        position = np.asarray(data.cam_xpos[camera_id], dtype=float)
        rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        forward = rotation @ np.array([0.0, 0.0, -1.0])
        forward /= np.linalg.norm(forward)
        length = 0.13 if index == 1 else 0.10
        primitives.extend(
            (
                {
                    "geom_id": -100 - index,
                    "name": f"debug_camera{index}_pinhole_red",
                    "body_name": "world",
                    "type": "sphere",
                    "position": position.tolist(),
                    "xmat": np.eye(3).reshape(-1).tolist(),
                    "size": [0.012 if index == 1 else 0.009, 0.0, 0.0],
                    "rgba": [1.0, 0.01, 0.01, 1.0],
                    "semantic_color": None,
                },
                {
                    "geom_id": -110 - index,
                    "name": f"debug_camera{index}_optical_axis_red",
                    "body_name": "world",
                    "type": "cylinder",
                    "position": (position + forward * length * 0.5).tolist(),
                    "xmat": _matrix_with_z_axis(forward),
                    "size": [0.0035, length * 0.5, 0.0],
                    "rgba": [1.0, 0.01, 0.01, 1.0],
                    "semantic_color": None,
                },
            )
        )
    return tuple(primitives)


def _matrix_with_z_axis(direction: np.ndarray) -> list[float]:
    z_axis = np.asarray(direction, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis)).reshape(-1).tolist()


def _evidence_renders(
    *,
    env: Any,
    config: SO101CameraRigRenderConfig,
    output_dir: Path,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    renders: dict[str, tuple[Path, dict[str, Any]]] = {}
    for view in config.render.evidence_views:
        path = output_dir / view.filename
        if isinstance(view, FreeEvidenceViewConfig):
            camera_spec = _free_camera_spec(
                env,
                width=config.render.source_width,
                height=config.render.source_height,
                lookat=view.lookat_m,
                distance=view.distance_m,
                azimuth=view.azimuth_degrees,
                elevation=view.elevation_degrees,
                fovy=view.fovy_degrees,
                aperture_fstop=view.aperture_fstop,
                use_depth_of_field=view.use_depth_of_field,
                clip_start=view.clip_start_m,
            )
        elif isinstance(view, MountedEvidenceViewConfig):
            camera_spec = _mounted_camera_front_spec(
                env,
                camera_name=view.camera_name,
                width=config.render.source_width,
                height=config.render.source_height,
                distance=view.distance_m,
                fovy=view.fovy_degrees,
                aperture_fstop=view.aperture_fstop,
                use_depth_of_field=view.use_depth_of_field,
                clip_start=view.clip_start_m,
            )
        else:  # pragma: no cover - Pydantic's discriminated union prevents this.
            raise TypeError(f"unsupported evidence view: {type(view)!r}")
        renders[view.name] = (path, camera_spec)
    return renders


def _distortion_profile(config: SO101CameraRigRenderConfig) -> dict[str, Any]:
    return {
        "model": config.sensor.distortion.model,
        "coefficients": list(config.sensor.distortion.coefficients),
        "calibration_status": config.sensor.distortion.calibration_status,
    }


def _free_camera_spec(
    env: Any,
    *,
    width: int,
    height: int,
    lookat: tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
    fovy: float,
    aperture_fstop: float,
    use_depth_of_field: bool,
    clip_start: float,
) -> dict[str, Any]:
    import mujoco

    external_camera = mujoco.MjvCamera()
    external_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    external_camera.lookat[:] = lookat
    external_camera.distance = distance
    external_camera.azimuth = azimuth
    external_camera.elevation = elevation
    scene_renderer = mujoco.Renderer(env.unwrapped.model, height=height, width=width)
    try:
        scene_renderer.update_scene(env.unwrapped.data, camera=external_camera)
        scene = getattr(scene_renderer, "scene", None) or getattr(scene_renderer, "_scene", None)
        camera = scene.camera[0]
        return {
            "mode": "forward_up",
            "location": [float(value) for value in camera.pos],
            "forward": [float(value) for value in camera.forward],
            "up": [float(value) for value in camera.up],
            "fovy": fovy,
            "focus_distance": distance,
            "aperture_fstop": aperture_fstop,
            "use_dof": use_depth_of_field,
            "clip_start": clip_start,
        }
    finally:
        scene_renderer.close()


def _mounted_camera_front_spec(
    env: Any,
    *,
    camera_name: str,
    width: int,
    height: int,
    distance: float,
    fovy: float,
    aperture_fstop: float,
    use_depth_of_field: bool,
    clip_start: float,
) -> dict[str, Any]:
    import mujoco

    model = env.unwrapped.model
    data = env.unwrapped.data
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    target = np.asarray(data.cam_xpos[camera_id])
    optical_forward = rotation @ np.array([0.0, 0.0, -1.0])
    camera_up = rotation @ np.array([0.0, 1.0, 0.0])
    location = target + distance * optical_forward
    return {
        "mode": "forward_up",
        "location": [float(value) for value in location],
        "forward": [float(value) for value in -optical_forward],
        "up": [float(value) for value in camera_up],
        "fovy": fovy,
        "focus_distance": distance,
        "aperture_fstop": aperture_fstop,
        "use_dof": use_depth_of_field,
        "clip_start": clip_start,
        "width": width,
        "height": height,
    }


def _write_contact_sheet(
    *,
    camera1: np.ndarray,
    camera2: np.ndarray,
    external: np.ndarray,
    output_path: Path,
    config: SO101CameraRigRenderConfig,
) -> None:
    sheet_config = config.render.outputs.contact_sheet
    font = ImageFont.load_default(size=sheet_config.font_size)
    camera1_fovy = (
        config.camera1.effective_vertical_fov_degrees
        or config.sensor.vertical_fov_degrees
    )
    camera2_fovy = (
        config.camera2.effective_vertical_fov_degrees
        or config.sensor.vertical_fov_degrees
    )
    entries = (
        (
            "camera1 | overhead | "
            f"H{config.sensor.horizontal_fov_degrees:g}/"
            f"V{camera1_fovy:g} | Brown barrel candidate",
            camera1,
        ),
        (
            "camera2 | wrist | "
            f"H{config.sensor.horizontal_fov_degrees:g}/"
            f"V{camera2_fovy:g} | hardware-centred axis + Brown barrel",
            camera2,
        ),
        ("external | official STL camera rig", external),
    )
    tile_size = sheet_config.tile_size
    header_height = sheet_config.header_height
    tiles: list[Image.Image] = []
    for label, pixels in entries:
        image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile = Image.new(
            "RGB",
            (tile_size, tile_size + header_height),
            sheet_config.background_rgb,
        )
        tile.paste(
            image,
            (
                (tile_size - image.width) // 2,
                header_height + (tile_size - image.height) // 2,
            ),
        )
        ImageDraw.Draw(tile).text(
            (10, 10),
            label,
            fill=sheet_config.foreground_rgb,
            font=font,
        )
        tiles.append(tile)
    sheet = Image.new(
        "RGB",
        (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)),
        sheet_config.background_rgb,
    )
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    sheet.save(output_path)


def _policy_input(
    pixels: np.ndarray,
    *,
    size: int,
    mode: str,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB")
    if mode == "direct_square_render":
        if image.width != image.height:
            raise ValueError("direct_square_render received a non-square image")
        return np.asarray(
            image.resize((size, size), Image.Resampling.LANCZOS)
        ).copy()
    if mode != "center_crop_square_then_resize":
        raise ValueError(f"unsupported policy resize mode: {mode}")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return np.asarray(
        image.crop((left, top, left + side, top + side)).resize(
            (size, size),
            Image.Resampling.LANCZOS,
        )
    ).copy()


def _center_crop_policy_input(pixels: np.ndarray, *, size: int) -> np.ndarray:
    return _policy_input(
        pixels,
        size=size,
        mode="center_crop_square_then_resize",
    )


def _validate_render_dependencies(config: SO101CameraRigRenderConfig) -> None:
    material_profile = resolve_repository_path(config.render.material_profile).resolve()
    if not material_profile.is_file():
        raise FileNotFoundError(f"material profile is missing: {material_profile}")
    actual_material_sha = _sha256_file(material_profile)
    if actual_material_sha != config.render.material_profile_sha256:
        raise ValueError(
            "material profile SHA-256 mismatch: "
            f"expected {config.render.material_profile_sha256}, got {actual_material_sha}"
        )

    for scene_asset in config.render.scene_assets:
        for asset_file in (scene_asset.blend, *scene_asset.dependencies):
            path = resolve_repository_path(asset_file.path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"render asset is missing: {path}")
            actual_sha = _sha256_file(path)
            if actual_sha != asset_file.sha256:
                raise ValueError(
                    f"render asset SHA-256 mismatch for {path}: "
                    f"expected {asset_file.sha256}, got {actual_sha}"
                )

    blender_bin = shutil.which(config.render.blender_bin) or config.render.blender_bin
    completed = subprocess.run(
        [blender_bin, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_version = completed.stdout.splitlines()[0].strip()
    if actual_version != config.render.expected_blender_version:
        raise RuntimeError(
            "Blender version mismatch: "
            f"expected {config.render.expected_blender_version!r}, got {actual_version!r}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
