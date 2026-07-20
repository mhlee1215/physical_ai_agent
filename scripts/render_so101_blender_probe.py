#!/usr/bin/env python3
"""Render the current SO101 MuJoCo state through Blender Cycles Metal."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action

try:
    from render_so101_mitsuba_probe import _export_mesh_geoms, _target_site
except ModuleNotFoundError:
    from scripts.render_so101_mitsuba_probe import _export_mesh_geoms, _target_site


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
    scene.cycles.seed = int(spec.get("cycles_seed", 0))
    if hasattr(scene.cycles, "use_animated_seed"):
        scene.cycles.use_animated_seed = False
    if hasattr(scene.cycles, "sample_clamp_direct"):
        scene.cycles.sample_clamp_direct = float(spec.get("sample_clamp_direct", 0.0))
    if hasattr(scene.cycles, "sample_clamp_indirect"):
        scene.cycles.sample_clamp_indirect = float(spec.get("sample_clamp_indirect", 1.25))
    scene.render.resolution_x = int(spec["width"])
    scene.render.resolution_y = int(spec["height"])
    scene.view_settings.view_transform = str(spec.get("color_management", "Filmic"))
    scene.view_settings.look = str(spec.get("color_look", "Medium High Contrast"))
    scene.view_settings.exposure = float(spec.get("exposure", -1.30))
    scene.view_settings.gamma = float(spec.get("gamma", 1.0))
    scene.render.image_settings.file_format = str(spec.get("output_format", "PNG"))
    scene.render.image_settings.color_mode = "RGB"
    if scene.render.image_settings.file_format == "JPEG":
        scene.render.image_settings.quality = 95

    prefs = bpy.context.preferences.addons["cycles"].preferences
    compute_device_type = str(spec.get("compute_device_type", "METAL"))
    prefs.compute_device_type = compute_device_type
    prefs.get_devices()
    selected_devices = []
    for device in prefs.devices:
        device.use = device.type == compute_device_type
        if device.use:
            selected_devices.append(device.name)
    scene.cycles.device = "GPU" if selected_devices else "CPU"

    scene_profile = spec.get("scene_profile", "neutral")
    floor_mat = (
        make_black_tabletop_material()
        if scene_profile == "black_table_clutter"
        else make_stable_tabletop_material() if spec.get("stable_tabletop") else make_tabletop_material(spec["table_pbr"])
    )
    if scene_profile == "black_table_clutter":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.28, 0.10, -0.018))
        bpy.context.object.scale = (0.72, 0.62, 0.018)
    else:
        bpy.ops.mesh.primitive_plane_add(size=2.5, location=(0.0, 0.0, -0.002))
    floor = bpy.context.object
    floor.name = "tabletop"
    floor.data.materials.append(floor_mat)

    for item in spec.get("visual_props", []):
        add_visual_prop(item)

    if spec.get("background_wall", True):
        wall_mat = make_wall_material()
        bpy.ops.mesh.primitive_plane_add(
            size=2.5,
            location=(0.0, 0.78, 0.62),
            rotation=(math.radians(90.0), 0.0, 0.0),
        )
        wall = bpy.context.object
        wall.name = "matte_background_wall"
        wall.data.materials.append(wall_mat)

    for item in spec["meshes"]:
        bpy.ops.wm.ply_import(filepath=item["path"])
        obj = bpy.context.object
        obj.name = item["name"] or f"mesh_{item['geom_id']:03d}"
        if item.get("position") is not None:
            obj.location = tuple(float(value) for value in item["position"])
        if item.get("quaternion_wxyz") is not None:
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = tuple(float(value) for value in item["quaternion_wxyz"])
        obj.data.materials.append(
            make_robot_material(
                item,
                spec["plastic_pbr"],
                spec.get("robot_material", "plastic"),
                spec.get("robot_material_config"),
            )
        )
        bpy.ops.object.shade_smooth()
        weighted = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
        weighted.keep_sharp = True

    for item in spec.get("primitives", []):
        add_primitive(item, spec.get("robot_material_config"))

    target = spec.get("target_site")
    if target:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64,
            ring_count=32,
            radius=max(float(target["radius"]), 0.018),
            location=target["position"],
        )
        sphere = bpy.context.object
        sphere.name = "reach_target"
        sphere.data.materials.append(make_target_material())
        bpy.ops.object.shade_smooth()

    add_area_light("softbox", (0.04, -0.48, 0.88), float(spec.get("key_light_power", 42.0)), 0.78)
    add_area_light("fill", (-0.55, 0.34, 0.50), float(spec.get("fill_light_power", 5.0)), 0.62)
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    configure_world(
        world,
        spec.get("hdri_path") if spec.get("lighting_profile", "studio_small_08") != "flat" else None,
        strength=float(spec.get("world_strength", 0.28)),
        rotation_deg=float(spec.get("hdri_rotation_deg", 35.0)),
    )

    renders = spec.get("renders") or [
        {
            "image_path": spec["image_path"],
            "camera": {
                "mode": "look_at",
                "location": [0.48, -0.50, 0.31],
                "target": [0.12, 0.015, 0.075],
                "lens": float(spec.get("camera_lens", 48)),
                "focus_distance": 0.56,
            },
        }
    ]
    camera = bpy.data.cameras.new("camera")
    camera_obj = bpy.data.objects.new("camera", camera)
    bpy.context.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    for render in renders:
        configure_camera(camera_obj, camera, render["camera"], default_lens=float(spec.get("camera_lens", 48)))
        scene.render.filepath = render["image_path"]
        bpy.ops.render.render(write_still=True)

    report = {
        "blender_version": bpy.app.version_string,
        "cycles_device": scene.cycles.device,
        "compute_device_type": prefs.compute_device_type,
        "metal_devices": selected_devices if compute_device_type == "METAL" else [],
        "selected_devices": selected_devices,
    }
    Path(spec["blender_report_path"]).write_text(json.dumps(report, indent=2, sort_keys=True))


def make_robot_material(item, plastic_pbr, robot_material, robot_material_config):
    rgba = item["rgba"]
    rgb = material_color(rgba)
    mat = bpy.data.materials.new("robot_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    configured = configured_material_spec(item, robot_material_config, allow_default=True)
    if configured:
        color = configured["base_color"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = float(configured["roughness"])
        bsdf.inputs["Metallic"].default_value = float(configured["metallic"])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = float(configured.get("noise_scale", 100.0))
        noise.inputs["Detail"].default_value = 8.0
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = float(configured.get("bump_strength", 0.015))
        bump.inputs["Distance"].default_value = 0.0012
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        return mat
    is_yellow_body = rgb[0] > 0.5 and rgb[1] > 0.25
    if robot_material == "metal" and is_yellow_body:
        rgb = [0.86, 0.70, 0.38]
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.26
    elif robot_material == "matte_pla" and is_yellow_body:
        rgb = [0.68, 0.46, 0.11]
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.84
        bsdf.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.30
    else:
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.58 if rgb[0] > 0.5 else 0.46
        bsdf.inputs["Metallic"].default_value = 0.0
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 85.0 if rgb[0] > 0.5 else 55.0
    noise.inputs["Detail"].default_value = 11.0
    noise.inputs["Roughness"].default_value = 0.56
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[1].position = 1.0
    if rgb[0] > 0.5:
        ramp.color_ramp.elements[0].color = (max(rgb[0] - 0.16, 0.0), max(rgb[1] - 0.13, 0.0), max(rgb[2] - 0.06, 0.0), 1.0)
        ramp.color_ramp.elements[1].color = (min(rgb[0] + 0.08, 1.0), min(rgb[1] + 0.07, 1.0), min(rgb[2] + 0.04, 1.0), 1.0)
    else:
        ramp.color_ramp.elements[0].color = (0.01, 0.01, 0.01, 1.0)
        ramp.color_ramp.elements[1].color = (0.12, 0.12, 0.11, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.024 if rgb[0] > 0.5 else 0.014
    bump.inputs["Distance"].default_value = 0.0018
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    if robot_material == "metal" and is_yellow_body:
        rough_mix = nodes.new("ShaderNodeTexNoise")
        rough_mix.inputs["Scale"].default_value = 18.0
        rough_mix.inputs["Detail"].default_value = 8.0
        rough_ramp = nodes.new("ShaderNodeValToRGB")
        rough_ramp.color_ramp.elements[0].position = 0.15
        rough_ramp.color_ramp.elements[0].color = (0.18, 0.18, 0.18, 1.0)
        rough_ramp.color_ramp.elements[1].position = 1.0
        rough_ramp.color_ramp.elements[1].color = (0.42, 0.42, 0.42, 1.0)
        links.new(rough_mix.outputs["Fac"], rough_ramp.inputs["Fac"])
        links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])
    elif robot_material == "matte_pla" and is_yellow_body:
        pla_noise = nodes.new("ShaderNodeTexNoise")
        pla_noise.inputs["Scale"].default_value = 145.0
        pla_noise.inputs["Detail"].default_value = 14.0
        pla_noise.inputs["Roughness"].default_value = 0.62
        pla_ramp = nodes.new("ShaderNodeValToRGB")
        pla_ramp.color_ramp.elements[0].position = 0.12
        pla_ramp.color_ramp.elements[0].color = (0.54, 0.36, 0.08, 1.0)
        pla_ramp.color_ramp.elements[1].position = 1.0
        pla_ramp.color_ramp.elements[1].color = (0.76, 0.53, 0.16, 1.0)
        pla_bump = nodes.new("ShaderNodeBump")
        pla_bump.inputs["Strength"].default_value = 0.020
        pla_bump.inputs["Distance"].default_value = 0.0012
        links.new(pla_noise.outputs["Fac"], pla_ramp.inputs["Fac"])
        links.new(pla_ramp.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(pla_noise.outputs["Fac"], pla_bump.inputs["Height"])
        links.new(pla_bump.outputs["Normal"], bsdf.inputs["Normal"])
    elif plastic_pbr.get("roughness"):
        rough_tex = image_texture(nodes, plastic_pbr["roughness"], colorspace="Non-Color")
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    if robot_material == "matte_pla" and is_yellow_body:
        pass
    elif plastic_pbr.get("normal"):
        normal_tex = image_texture(nodes, plastic_pbr["normal"], colorspace="Non-Color")
        normal = nodes.new("ShaderNodeNormalMap")
        normal.inputs["Strength"].default_value = 0.10 if rgb[0] > 0.5 else 0.055
        links.new(normal_tex.outputs["Color"], normal.inputs["Color"])
        links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_target_material():
    mat = bpy.data.materials.new("target_orange")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (1.0, 0.42, 0.04, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.44
    return mat


def add_primitive(item, robot_material_config):
    if len(item.get("rgba", [])) >= 4 and float(item["rgba"][3]) <= 0.01:
        return
    kind = item["type"]
    location = item["position"]
    rotation = matrix_to_euler(item["xmat"])
    name = item["name"] or f"primitive_{item['geom_id']:03d}"
    size = [float(value) for value in item["size"]]
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location, rotation=rotation)
        obj = bpy.context.object
        obj.scale = (size[0], size[1], size[2])
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=size[0], depth=2.0 * size[1], location=location, rotation=rotation)
        obj = bpy.context.object
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=size[0], location=location, rotation=rotation)
        obj = bpy.context.object
    else:
        return
    obj.name = name
    obj.data.materials.append(make_primitive_material(item, robot_material_config))
    bpy.ops.object.shade_smooth()


def matrix_to_euler(values):
    rows = [values[0:3], values[3:6], values[6:9]]
    return Matrix(rows).to_euler()


def make_primitive_material(item, robot_material_config):
    name = item["name"]
    rgba = item["rgba"]
    semantic_color = item.get("semantic_color")
    mat = bpy.data.materials.new(f"{name}_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    configured = configured_material_spec(item, robot_material_config, allow_default=False)
    if configured:
        rgb = configured["base_color"]
        roughness = float(configured["roughness"])
    elif semantic_color is not None or "cube" in name or "pick_slot" in name:
        # Object color is part of the dataset/prompt contract.
        rgb = tuple(float(value) for value in rgba[:3])
        roughness = 0.58
    elif "pad" in name:
        rgb = (0.025, 0.025, 0.024)
        roughness = 0.76
    else:
        color = material_color(rgba)
        rgb = (color[0], color[1], color[2])
        roughness = 0.66
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Metallic"].default_value = float(configured["metallic"]) if configured else 0.0
    bsdf.inputs["Roughness"].default_value = roughness
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 84.0
    noise.inputs["Detail"].default_value = 10.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.010
    bump.inputs["Distance"].default_value = 0.0012
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def configured_material_spec(item, config, allow_default):
    if not config:
        return None
    if config.get("schema_version") == 2:
        for part in config.get("parts", {}).values():
            if any(selector_rule_matches(item, rule) for rule in part.get("selectors", [])):
                return config["materials"][part["material"]]
        if allow_default:
            return config["materials"][config["default_material"]]
        return None
    selector_fields = {
        "body_names": "body_name",
        "mesh_names": "mesh_name",
        "primitive_names": "name",
    }
    for part_name, selector in config.get("selectors", {}).items():
        if any(item.get(item_key) in selector.get(selector_key, []) for selector_key, item_key in selector_fields.items()):
            return config["parts"][part_name]
    if allow_default:
        return config["parts"][config["default_part"]]
    return None


def selector_rule_matches(item, rule):
    selector_fields = {
        "body_names": "body_name",
        "mesh_names": "mesh_name",
        "primitive_names": "name",
    }
    populated = False
    for selector_key, item_key in selector_fields.items():
        if selector_key not in rule:
            continue
        populated = True
        if item.get(item_key) not in rule[selector_key]:
            return False
    return populated


def make_tabletop_material(table_pbr):
    mat = bpy.data.materials.new("textured_tabletop")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.65, 1.65, 1.0)
    coord = nodes.new("ShaderNodeTexCoord")
    mat.node_tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    if table_pbr.get("color"):
        tex = image_texture(nodes, table_pbr["color"])
        mat.node_tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.34, 0.31, 0.25, 1.0)
    if table_pbr.get("roughness"):
        rough_tex = image_texture(nodes, table_pbr["roughness"], colorspace="Non-Color")
        mat.node_tree.links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        mat.node_tree.links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.72
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 44.0
    noise.inputs["Detail"].default_value = 10.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.018
    bump.inputs["Distance"].default_value = 0.004
    if table_pbr.get("normal"):
        normal_tex = image_texture(nodes, table_pbr["normal"], colorspace="Non-Color")
        normal = nodes.new("ShaderNodeNormalMap")
        normal.inputs["Strength"].default_value = 0.18
        mat.node_tree.links.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])
        mat.node_tree.links.new(normal_tex.outputs["Color"], normal.inputs["Color"])
        mat.node_tree.links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        mat.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        mat.node_tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_stable_tabletop_material():
    mat = bpy.data.materials.new("stable_training_tabletop")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.52, 0.52, 0.49, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.82
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def make_black_tabletop_material():
    mat = bpy.data.materials.new("black_workbench")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.025, 0.028, 0.032, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.72
    bsdf.inputs["Metallic"].default_value = 0.0
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 115.0
    noise.inputs["Detail"].default_value = 4.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.025
    bump.inputs["Distance"].default_value = 0.0008
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def add_visual_prop(item):
    kind = item["kind"]
    x, y = item["position"]
    yaw = math.radians(float(item.get("yaw_degrees", 0.0)))
    color = item["color"]
    material = make_simple_material(f"{item['name']}_material", color, float(item.get("roughness", 0.48)))

    def finish(obj, name, mat=material):
        obj.name = name
        obj.data.materials.append(mat)
        bpy.ops.object.shade_smooth()
        return obj

    if kind == "mug":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.034, depth=0.070, location=(x, y, 0.035))
        finish(bpy.context.object, item["name"])
        dark = make_simple_material(f"{item['name']}_inside", (0.008, 0.009, 0.010), 0.82)
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.027, depth=0.002, location=(x, y, 0.071))
        finish(bpy.context.object, f"{item['name']}_inside", dark)
        handle_x, handle_y = x + 0.038 * math.cos(yaw), y + 0.038 * math.sin(yaw)
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.022,
            minor_radius=0.006,
            major_segments=48,
            minor_segments=16,
            location=(handle_x, handle_y, 0.040),
            rotation=(math.radians(90.0), 0.0, yaw),
        )
        finish(bpy.context.object, f"{item['name']}_handle")
    elif kind == "bottle":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.028, depth=0.095, location=(x, y, 0.0475))
        finish(bpy.context.object, item["name"])
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.016, depth=0.024, location=(x, y, 0.107))
        finish(bpy.context.object, f"{item['name']}_neck")
        cap = make_simple_material(f"{item['name']}_cap_material", (0.06, 0.065, 0.07), 0.62)
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.017, depth=0.012, location=(x, y, 0.125))
        finish(bpy.context.object, f"{item['name']}_cap", cap)
    elif kind == "tape":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.030,
            minor_radius=0.010,
            major_segments=64,
            minor_segments=20,
            location=(x, y, 0.011),
            rotation=(0.0, 0.0, yaw),
        )
        finish(bpy.context.object, item["name"])
    elif kind == "screwdriver":
        direction = Vector((math.cos(yaw), math.sin(yaw), 0.0))
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.012,
            depth=0.075,
            location=Vector((x, y, 0.016)) - direction * 0.040,
            rotation=(0.0, math.radians(90.0), yaw),
        )
        finish(bpy.context.object, f"{item['name']}_handle")
        steel = make_simple_material(f"{item['name']}_steel", (0.30, 0.32, 0.34), 0.28, metallic=0.85)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=24,
            radius=0.004,
            depth=0.105,
            location=Vector((x, y, 0.016)) + direction * 0.050,
            rotation=(0.0, math.radians(90.0), yaw),
        )
        finish(bpy.context.object, f"{item['name']}_shaft", steel)


def make_simple_material(name, color, roughness, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def make_wall_material():
    mat = bpy.data.materials.new("warm_matte_wall")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.43, 0.42, 0.39, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.86
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 32.0
    noise.inputs["Detail"].default_value = 6.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.012
    bump.inputs["Distance"].default_value = 0.006
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def configure_world(world, hdri_path, strength=0.28, rotation_deg=35.0):
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg = nodes.get("Background")
    if not hdri_path:
        bg.inputs["Color"].default_value = (0.012, 0.013, 0.015, 1.0)
        bg.inputs["Strength"].default_value = strength
        return
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_deg)
    coord = nodes.new("ShaderNodeTexCoord")
    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = max(strength, 0.0)


def image_texture(nodes, path, colorspace="sRGB"):
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.extension = "REPEAT"
    tex.interpolation = "Linear"
    tex.image.colorspace_settings.name = colorspace
    return tex


def material_color(rgba):
    rgb = [float(value) for value in rgba[:3]]
    if rgb[0] > 0.8 and rgb[1] > 0.6 and rgb[2] < 0.25:
        return [0.64, 0.40, 0.075]
    if max(rgb) < 0.2:
        return [0.025, 0.026, 0.026]
    return rgb


def add_area_light(name, location, power, size):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = power
    light.size = size
    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, Vector((0.12, 0.02, 0.05)))


def configure_camera(camera_obj, camera, camera_spec, default_lens):
    mode = camera_spec.get("mode", "look_at")
    if mode == "matrix":
        camera_obj.location = camera_spec["location"]
        xmat = camera_spec["xmat"]
        rows = [xmat[0:3], xmat[3:6], xmat[6:9]]
        columns = [
            [rows[0][0], rows[1][0], rows[2][0]],
            [rows[0][1], rows[1][1], rows[2][1]],
            [rows[0][2], rows[1][2], rows[2][2]],
        ]
        direction = -Vector(columns[2])
        look_at(camera_obj, camera_obj.location + direction)
    elif mode == "forward_up":
        camera_obj.location = camera_spec["location"]
        forward = Vector(camera_spec["forward"]).normalized()
        up = Vector(camera_spec["up"]).normalized()
        z_axis = -forward
        x_axis = up.cross(z_axis).normalized()
        y_axis = z_axis.cross(x_axis).normalized()
        rotation = Matrix(
            (
                (x_axis.x, y_axis.x, z_axis.x),
                (x_axis.y, y_axis.y, z_axis.y),
                (x_axis.z, y_axis.z, z_axis.z),
            )
        )
        camera_obj.rotation_euler = rotation.to_euler()
    elif mode == "spherical":
        lookat = Vector(camera_spec["lookat"])
        distance = float(camera_spec["distance"])
        azimuth = math.radians(float(camera_spec["azimuth"]))
        elevation = math.radians(float(camera_spec["elevation"]))
        location = Vector(
            (
                lookat.x + distance * math.cos(elevation) * math.cos(azimuth),
                lookat.y + distance * math.cos(elevation) * math.sin(azimuth),
                lookat.z - distance * math.sin(elevation),
            )
        )
        camera_obj.location = location
        look_at(camera_obj, lookat)
    else:
        camera_obj.location = camera_spec["location"]
        look_at(camera_obj, Vector(camera_spec["target"]))
    if camera_spec.get("fovy") is not None:
        camera.sensor_fit = "VERTICAL"
        camera.angle = math.radians(float(camera_spec["fovy"]))
    else:
        camera.lens = float(camera_spec.get("lens", default_lens))
    camera.clip_start = float(camera_spec.get("clip_start", 0.001))
    camera.clip_end = float(camera_spec.get("clip_end", 100.0))
    camera.dof.use_dof = bool(camera_spec.get("use_dof", True))
    camera.dof.focus_distance = float(camera_spec.get("focus_distance", 0.56))
    camera.dof.aperture_fstop = float(camera_spec.get("aperture_fstop", 8.0))


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SO101 through Blender Cycles Metal.")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_blender_probe"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--robot-material", choices=("plastic", "matte_pla", "metal"), default="plastic")
    parser.add_argument("--asset-root", type=Path, default=Path("_workspace/photoreal_assets"))
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--max-mesh-geoms", type=int, default=128)
    parser.add_argument(
        "--mujoco-reference",
        type=Path,
        default=Path("_workspace/so101_realistic_render_probe_v2/enhanced_scene.png"),
    )
    args = parser.parse_args()

    blender_bin = shutil.which(args.blender_bin) or args.blender_bin
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = render_blender_probe(
        env_id=args.env_id,
        output_dir=args.output_dir,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        width=args.width,
        height=args.height,
        samples=args.samples,
        denoise=args.denoise,
        robot_material=args.robot_material,
        asset_root=args.asset_root,
        blender_bin=blender_bin,
        max_mesh_geoms=args.max_mesh_geoms,
        mujoco_reference=args.mujoco_reference,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def render_blender_probe(
    *,
    env_id: str,
    output_dir: Path,
    seed: int,
    warmup_steps: int,
    width: int,
    height: int,
    samples: int,
    denoise: bool,
    robot_material: str,
    asset_root: Path,
    blender_bin: str,
    max_mesh_geoms: int,
    mujoco_reference: Path | None,
) -> dict[str, Any]:
    import gymnasium as gym
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    mesh_dir = output_dir / "ply"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_path = _write_photo_tabletop_texture(output_dir / "tabletop_texture.png")
    hdri_path = asset_root / "polyhaven" / "studio_small_08_2k.hdr"
    table_pbr_dir = asset_root / "ambientcg" / "Wood008_1K-JPG"
    plastic_pbr_dir = asset_root / "ambientcg" / "Plastic013A_1K-JPG"

    env = gym.make(env_id, render_mode=None)
    try:
        env.reset(seed=seed)
        for step in range(warmup_steps):
            env.step(sample_action(env.action_space, step / max(1, warmup_steps - 1)))
        model, data = env.unwrapped.model, env.unwrapped.data
        exported = _export_mesh_geoms(model, data, mesh_dir, max_mesh_geoms=max_mesh_geoms)
        target = _target_site(model, data)
    finally:
        env.close()

    driver_path = output_dir / "blender_driver.py"
    spec_path = output_dir / "blender_scene_spec.json"
    image_path = output_dir / "so101_blender_cycles_metal.png"
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
        "plastic_pbr": _pbr_paths(plastic_pbr_dir, "Plastic013A_1K-JPG"),
        "image_path": str(image_path.resolve()),
        "blender_report_path": str(blender_report_path.resolve()),
        "meshes": [{**item, "path": str(Path(item["path"]).resolve())} for item in exported],
        "target_site": target,
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
        "env_id": env_id,
        "renderer": "blender_cycles",
        "acceleration": "Metal" if device_report.get("metal_devices") else "CPU",
        "blender_report": device_report,
        "image_path": str(image_path),
        "comparison_path": str(comparison_path) if comparison_path else None,
        "mesh_dir": str(mesh_dir),
        "mesh_geoms_exported": len(exported),
        "mesh_format": "binary_little_endian_ply",
        "texture_path": str(texture_path),
        "hdri_path": str(hdri_path) if hdri_path.exists() else None,
        "table_pbr_dir": str(table_pbr_dir) if table_pbr_dir.exists() else None,
        "plastic_pbr_dir": str(plastic_pbr_dir) if plastic_pbr_dir.exists() else None,
        "samples": samples,
        "render_seconds": render_seconds,
        "denoise": denoise,
        "robot_material": robot_material,
        "width": width,
        "height": height,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "log_path": str(log_path),
    }
    (output_dir / "blender_probe_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _write_comparison(reference: Path | None, render_path: Path, output_path: Path) -> Path | None:
    if reference is None or not reference.exists():
        return None
    pairs = [("MuJoCo enhanced", reference), ("Blender Cycles Metal", render_path)]
    cell_w, cell_h, label_h = 640, 480, 38
    sheet = Image.new("RGB", (cell_w * 2, cell_h + label_h), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(pairs):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = index * cell_w + (cell_w - image.width) // 2
        y = label_h + (cell_h - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index * cell_w + 14, 12), label, fill=(25, 25, 25))
    sheet.save(output_path)
    return output_path


def _pbr_paths(directory: Path, stem: str) -> dict[str, str | None]:
    if not directory.exists():
        return {"color": None, "roughness": None, "normal": None, "displacement": None}
    paths = {
        "color": directory / f"{stem}_Color.jpg",
        "roughness": directory / f"{stem}_Roughness.jpg",
        "normal": directory / f"{stem}_NormalGL.jpg",
        "displacement": directory / f"{stem}_Displacement.jpg",
    }
    return {key: str(path.resolve()) if path.exists() else None for key, path in paths.items()}


def _write_photo_tabletop_texture(path: Path, size: int = 1024) -> Path:
    if path.exists():
        return path
    import numpy as np

    image = Image.new("RGB", (size, size), (105, 101, 92))
    pixels = image.load()
    rng = np.random.default_rng(240704)
    base_noise = rng.normal(0.0, 4.0, (size, size))
    scratch_mask = rng.random((size, size))
    for y in range(size):
        long_grain = 7.0 * np.sin(y / 47.0) + 3.5 * np.sin(y / 141.0)
        for x in range(size):
            tile = 4 if ((x // 256) + (y // 256)) % 2 == 0 else -3
            scratch = -18 if scratch_mask[y, x] > 0.9985 else 0
            value = int(np.clip(110 + long_grain + tile + base_noise[y, x] + scratch, 62, 160))
            pixels[x, y] = (value, int(value * 0.95), int(value * 0.84))
    draw = ImageDraw.Draw(image, "RGBA")
    for line in range(0, size, 256):
        draw.line((line, 0, line, size), fill=(55, 48, 40, 32), width=1)
        draw.line((0, line, size, line), fill=(55, 48, 40, 26), width=1)
    for _ in range(90):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        length = int(rng.integers(12, 90))
        alpha = int(rng.integers(12, 38))
        draw.line((x, y, min(size, x + length), y + int(rng.integers(-2, 3))), fill=(245, 238, 220, alpha), width=1)
    image.save(path)
    return path


if __name__ == "__main__":
    main()
