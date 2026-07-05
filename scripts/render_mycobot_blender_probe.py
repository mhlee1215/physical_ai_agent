#!/usr/bin/env python3
"""Render the current MyCobot MuJoCo state through Blender Cycles Metal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.sim.mycobot_nexus_env import (
    ADAPTIVE_GATE7_TABLE_ARM_QPOS,
    MODEL_PROFILE_280_JN,
    MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    MODEL_PROFILE_320_GRIPPER,
    MyCobotNexusConfig,
    MyCobotNexusEnv,
    TASK_CUBE_POS,
    sample_mycobot_nexus_action,
)

from render_so101_blender_probe import _pbr_paths, _write_comparison, _write_photo_tabletop_texture
from render_so101_mitsuba_probe import _export_mesh_geoms


BLENDER_DRIVER = r'''
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    spec_path = Path(args[0])
    spec = json.loads(spec_path.read_text())

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(spec["samples"])
    scene.cycles.preview_samples = min(64, int(spec["samples"]))
    scene.cycles.use_denoising = bool(spec["denoise"])
    scene.render.resolution_x = int(spec["width"])
    scene.render.resolution_y = int(spec["height"])
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -1.05
    scene.view_settings.gamma = 1.0

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    metal_devices = []
    for device in prefs.devices:
        device.use = device.type == "METAL"
        if device.use:
            metal_devices.append(device.name)
    scene.cycles.device = "GPU" if metal_devices else "CPU"

    floor_mat = make_tabletop_material(spec["table_pbr"])
    bpy.ops.mesh.primitive_plane_add(size=2.8, location=(-0.12, -0.12, -0.002))
    floor = bpy.context.object
    floor.name = "textured_tabletop"
    floor.data.materials.append(floor_mat)

    wall_mat = make_wall_material()
    bpy.ops.mesh.primitive_plane_add(
        size=2.8,
        location=(-0.18, 0.88, 0.72),
        rotation=(math.radians(90.0), 0.0, 0.0),
    )
    wall = bpy.context.object
    wall.name = "matte_background_wall"
    wall.data.materials.append(wall_mat)

    for item in spec["primitives"]:
        if item["name"] == "nexus_floor":
            continue
        add_primitive(item, spec.get("robot_material", "matte_pla"))

    for item in spec["meshes"]:
        bpy.ops.wm.ply_import(filepath=item["path"])
        obj = bpy.context.object
        obj.name = item["name"] or f"mesh_{item['geom_id']:03d}"
        obj.data.materials.append(make_robot_material(item["rgba"], spec.get("robot_material", "matte_pla")))
        bpy.ops.object.shade_smooth()
        weighted = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
        weighted.keep_sharp = True

    tcp = spec.get("tcp_site")
    if tcp:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48,
            ring_count=24,
            radius=max(float(tcp["radius"]), 0.007),
            location=tcp["position"],
        )
        marker = bpy.context.object
        marker.name = "mycobot_tcp_site"
        marker.data.materials.append(make_tcp_material())
        bpy.ops.object.shade_smooth()

    add_area_light("softbox_key", (0.16, -0.70, 0.95), 54.0, 0.82)
    add_area_light("left_fill", (-0.72, 0.08, 0.54), 8.0, 0.68)
    add_area_light("rim", (-0.34, -0.42, 0.72), 6.0, 0.38)
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    configure_world(world, spec.get("hdri_path"))

    camera = bpy.data.cameras.new("camera")
    camera_obj = bpy.data.objects.new("camera", camera)
    bpy.context.collection.objects.link(camera_obj)
    camera_obj.location = (0.52, -0.96, 0.58)
    look_at(camera_obj, Vector((-0.12, -0.09, 0.19)))
    camera.lens = 36
    camera.dof.use_dof = True
    camera.dof.focus_distance = 0.86
    camera.dof.aperture_fstop = 8.0
    scene.camera = camera_obj

    scene.render.filepath = spec["image_path"]
    bpy.ops.render.render(write_still=True)

    report = {
        "blender_version": bpy.app.version_string,
        "cycles_device": scene.cycles.device,
        "compute_device_type": prefs.compute_device_type,
        "metal_devices": metal_devices,
    }
    Path(spec["blender_report_path"]).write_text(json.dumps(report, indent=2, sort_keys=True))


def add_primitive(item, robot_material):
    kind = item["type"]
    location = item["position"]
    rotation = matrix_to_euler(item["xmat"])
    name = item["name"] or f"primitive_{item['geom_id']:03d}"
    size = [float(value) for value in item["size"]]
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location, rotation=rotation)
        obj = bpy.context.object
        obj.scale = (size[0], size[1], size[2])
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=size[0], location=location, rotation=rotation)
        obj = bpy.context.object
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=size[0], depth=2.0 * size[1], location=location, rotation=rotation)
        obj = bpy.context.object
    else:
        return
    obj.name = name
    obj.data.materials.append(make_primitive_material(name, item["rgba"], robot_material))
    bpy.ops.object.shade_smooth()


def matrix_to_euler(values):
    rows = [values[0:3], values[3:6], values[6:9]]
    return Matrix(rows).to_euler()


def make_robot_material(rgba, robot_material):
    mat = bpy.data.materials.new("mycobot_robot_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    rgb = material_color(rgba)
    if robot_material == "metal":
        bsdf.inputs["Base Color"].default_value = (0.80, 0.79, 0.75, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.75
        bsdf.inputs["Roughness"].default_value = 0.32
    else:
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.78 if robot_material == "matte_pla" else 0.55
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 92.0
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.58
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.012 if robot_material == "metal" else 0.020
    bump.inputs["Distance"].default_value = 0.0014
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_primitive_material(name, rgba, robot_material):
    mat = bpy.data.materials.new(f"{name}_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if name == "task_cube":
        rgb = (0.08, 0.52, 0.25)
        roughness = 0.62
    elif name == "nexus_work_mat":
        rgb = (0.075, 0.075, 0.070)
        roughness = 0.88
    elif "finger_pad" in name:
        rgb = (0.020, 0.020, 0.020)
        roughness = 0.72
    elif "palm" in name:
        rgb = (0.12, 0.12, 0.11)
        roughness = 0.58
    else:
        color = material_color(rgba)
        rgb = (color[0], color[1], color[2])
        roughness = 0.76 if robot_material == "matte_pla" else 0.48
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if robot_material == "metal" and name not in {"task_cube", "nexus_work_mat"} and "finger_pad" not in name:
        bsdf.inputs["Metallic"].default_value = 0.55
        bsdf.inputs["Roughness"].default_value = 0.34
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 74.0 if name != "nexus_work_mat" else 40.0
    noise.inputs["Detail"].default_value = 9.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.010 if name == "task_cube" else 0.018
    bump.inputs["Distance"].default_value = 0.0012 if name == "task_cube" else 0.0028
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_tcp_material():
    mat = bpy.data.materials.new("tcp_marker_blue")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.1, 0.36, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.38
    return mat


def make_tabletop_material(table_pbr):
    mat = bpy.data.materials.new("textured_tabletop")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.55, 1.55, 1.0)
    coord = nodes.new("ShaderNodeTexCoord")
    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    if table_pbr.get("color"):
        tex = image_texture(nodes, table_pbr["color"])
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.34, 0.31, 0.25, 1.0)
    if table_pbr.get("roughness"):
        rough_tex = image_texture(nodes, table_pbr["roughness"], colorspace="Non-Color")
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.74
    return mat


def make_wall_material():
    mat = bpy.data.materials.new("warm_matte_wall")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.42, 0.41, 0.38, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.88
    return mat


def configure_world(world, hdri_path):
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg = nodes.get("Background")
    if not hdri_path:
        bg.inputs["Color"].default_value = (0.012, 0.013, 0.015, 1.0)
        bg.inputs["Strength"].default_value = 0.32
        return
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(42.0)
    coord = nodes.new("ShaderNodeTexCoord")
    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.62


def image_texture(nodes, path, colorspace="sRGB"):
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.extension = "REPEAT"
    tex.interpolation = "Linear"
    tex.image.colorspace_settings.name = colorspace
    return tex


def material_color(rgba):
    rgb = [float(value) for value in rgba[:3]]
    if max(rgb) < 0.2:
        return [0.03, 0.03, 0.03]
    if abs(rgb[0] - 0.5) < 0.05 and abs(rgb[1] - 0.5) < 0.05:
        return [0.78, 0.77, 0.72]
    return rgb


def add_area_light(name, location, power, size):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = power
    light.size = size
    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, Vector((-0.12, -0.09, 0.12)))


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MyCobot through Blender Cycles Metal.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("_vendor/mycobot_mujoco"),
        help="Local clone of https://github.com/elephantrobotics/mycobot_mujoco for the 280-jn profile.",
    )
    parser.add_argument(
        "--official-gripper-root",
        type=Path,
        default=Path(os.environ.get("MYCOBOT_ROS2_ROOT", "_vendor/mycobot_ros2")),
        help="Local clone of elephantrobotics/mycobot_ros2 for the default adaptive gripper profile.",
    )
    parser.add_argument(
        "--model-profile",
        choices=(MODEL_PROFILE_280_JN, MODEL_PROFILE_320_GRIPPER, MODEL_PROFILE_320_ADAPTIVE_GRIPPER),
        default=MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
        help=(
            "Robot/gripper source profile. The default MyCobot visual profile is "
            "320-m5-2022-adaptive-gripper. 280-jn is only for explicit legacy checks."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mycobot_blender_probe"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument(
        "--pose-preset",
        choices=("sample", "adaptive-table"),
        default="adaptive-table",
        help="State to render after reset. adaptive-table uses the validated adaptive gripper table pose.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--robot-material", choices=("plastic", "matte_pla", "metal"), default="matte_pla")
    parser.add_argument("--render-asset-root", type=Path, default=Path("_workspace/photoreal_assets"))
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--max-mesh-geoms", type=int, default=128)
    parser.add_argument("--mujoco-reference", type=Path)
    args = parser.parse_args()

    blender_bin = shutil.which(args.blender_bin) or args.blender_bin
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = render_mycobot_blender_probe(
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        model_profile=args.model_profile,
        output_dir=args.output_dir,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        pose_preset=args.pose_preset,
        width=args.width,
        height=args.height,
        samples=args.samples,
        denoise=args.denoise,
        robot_material=args.robot_material,
        render_asset_root=args.render_asset_root,
        blender_bin=blender_bin,
        max_mesh_geoms=args.max_mesh_geoms,
        mujoco_reference=args.mujoco_reference,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def render_mycobot_blender_probe(
    *,
    asset_root: Path,
    official_gripper_root: Path | None,
    model_profile: str,
    output_dir: Path,
    seed: int,
    warmup_steps: int,
    pose_preset: str,
    width: int,
    height: int,
    samples: int,
    denoise: bool,
    robot_material: str,
    render_asset_root: Path,
    blender_bin: str,
    max_mesh_geoms: int,
    mujoco_reference: Path | None,
) -> dict[str, Any]:
    mesh_dir = output_dir / "ply"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_path = _write_photo_tabletop_texture(output_dir / "tabletop_texture.png")
    hdri_path = render_asset_root / "polyhaven" / "studio_small_08_2k.hdr"
    table_pbr_dir = render_asset_root / "ambientcg" / "Wood008_1K-JPG"

    env = MyCobotNexusEnv(
        MyCobotNexusConfig(
            asset_root=asset_root,
            work_dir=output_dir / "mycobot_scene",
            official_gripper_root=official_gripper_root,
            model_profile=model_profile,
            width=width,
            height=height,
        )
    )
    try:
        env.reset(seed=seed)
        if pose_preset == "adaptive-table":
            _set_adaptive_table_pose(env)
        else:
            for step in range(warmup_steps):
                env.step(sample_mycobot_nexus_action(step, warmup_steps))
        model, data = env.model, env.data
        exported = _export_mesh_geoms(model, data, mesh_dir, max_mesh_geoms=max_mesh_geoms)
        primitives = _export_primitive_geoms(model, data)
        tcp_site = _site(model, data, "mycobot_tcp_site")
    finally:
        env.close()

    driver_path = output_dir / "blender_driver.py"
    spec_path = output_dir / "blender_scene_spec.json"
    image_path = output_dir / "mycobot_blender_cycles_metal.png"
    blender_report_path = output_dir / "blender_device_report.json"
    driver_path.write_text(BLENDER_DRIVER, encoding="utf-8")
    spec = {
        "width": width,
        "height": height,
        "samples": samples,
        "denoise": denoise,
        "robot_material": robot_material,
        "texture_path": str(texture_path.resolve()),
        "hdri_path": str(hdri_path.resolve()) if hdri_path.exists() else None,
        "table_pbr": _pbr_paths(table_pbr_dir, "Wood008_1K-JPG"),
        "image_path": str(image_path.resolve()),
        "blender_report_path": str(blender_report_path.resolve()),
        "meshes": [{**item, "path": str(Path(item["path"]).resolve())} for item in exported],
        "primitives": primitives,
        "tcp_site": tcp_site,
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

    command = [blender_bin, "--background", "--python", str(driver_path), "--", str(spec_path)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    render_seconds = time.perf_counter() - started
    log_path = output_dir / "blender_render.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Blender render failed with exit code {completed.returncode}; see {log_path}")

    comparison_path = _write_comparison(mujoco_reference, image_path, output_dir / "mujoco_vs_blender.png")
    device_report = json.loads(blender_report_path.read_text(encoding="utf-8"))
    report = {
        "renderer": "blender_cycles",
        "platform": "mycobot",
        "acceleration": "Metal" if device_report.get("metal_devices") else "CPU",
        "blender_report": device_report,
        "image_path": str(image_path),
        "comparison_path": str(comparison_path) if comparison_path else None,
        "asset_root": str(asset_root),
        "official_gripper_root": str(official_gripper_root) if official_gripper_root else None,
        "model_profile": model_profile,
        "mesh_dir": str(mesh_dir),
        "mesh_geoms_exported": len(exported),
        "primitive_geoms_exported": len(primitives),
        "mesh_format": "binary_little_endian_ply",
        "texture_path": str(texture_path),
        "hdri_path": str(hdri_path) if hdri_path.exists() else None,
        "table_pbr_dir": str(table_pbr_dir) if table_pbr_dir.exists() else None,
        "samples": samples,
        "render_seconds": render_seconds,
        "denoise": denoise,
        "robot_material": robot_material,
        "width": width,
        "height": height,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "pose_preset": pose_preset,
        "log_path": str(log_path),
    }
    (output_dir / "blender_probe_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _set_adaptive_table_pose(env: MyCobotNexusEnv) -> None:
    if env.config.model_profile != MODEL_PROFILE_320_ADAPTIVE_GRIPPER:
        raise ValueError("--pose-preset adaptive-table requires --model-profile 320-m5-2022-adaptive-gripper")
    for qpos_index, value in zip(env._qpos_indices, ADAPTIVE_GATE7_TABLE_ARM_QPOS, strict=True):
        env.data.qpos[qpos_index] = float(value)
    for actuator_index, value in zip(env._arm_actuator_indices, ADAPTIVE_GATE7_TABLE_ARM_QPOS, strict=True):
        env.data.ctrl[actuator_index] = float(value)
    env._set_gripper(command=0.25)
    env._mujoco.mj_forward(env.model, env.data)
    pad_midpoint = env._finger_pad_midpoint()
    cube_position = [float(pad_midpoint[0]), float(pad_midpoint[1]), float(TASK_CUBE_POS[2])]
    for axis, value in enumerate(cube_position):
        env.data.qpos[env._cube_freejoint_qpos_index + axis] = float(value)
    qvel_start = env._cube_freejoint_qvel_index
    env.data.qvel[qvel_start:qvel_start + 6] = 0.0
    env._cube_initial_pos = list(cube_position)
    env._mujoco.mj_forward(env.model, env.data)


def _export_primitive_geoms(model: Any, data: Any) -> list[dict[str, Any]]:
    import mujoco

    names = {
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    }
    primitives: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        kind = names.get(geom_type)
        if kind is None or kind == "plane":
            continue
        name = model.geom(geom_id).name or f"geom_{geom_id:03d}"
        rgba = _geom_rgba(model, geom_id)
        if len(rgba) >= 4 and rgba[3] <= 0.01:
            continue
        primitives.append(
            {
                "geom_id": geom_id,
                "name": name,
                "type": kind,
                "position": [float(value) for value in data.geom_xpos[geom_id]],
                "xmat": [float(value) for value in data.geom_xmat[geom_id]],
                "size": [float(value) for value in model.geom_size[geom_id]],
                "rgba": rgba,
            }
        )
    return primitives


def _geom_rgba(model: Any, geom_id: int) -> list[float]:
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        return [float(value) for value in model.mat_rgba[mat_id]]
    return [float(value) for value in model.geom_rgba[geom_id]]


def _site(model: Any, data: Any, name: str) -> dict[str, Any] | None:
    for site_id in range(model.nsite):
        if model.site(site_id).name == name:
            return {
                "position": [float(value) for value in data.site_xpos[site_id]],
                "radius": float(model.site_size[site_id][0]),
                "rgba": [float(value) for value in model.site_rgba[site_id]],
            }
    return None


if __name__ == "__main__":
    main()
