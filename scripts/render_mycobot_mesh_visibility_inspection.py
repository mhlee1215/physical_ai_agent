#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402


VISIBILITY_MODES = (
    "mesh_only",
    "mesh_markers",
    "pads_markers",
    "collision_only",
    "all",
    "audit_mesh",
    "audit_all",
    "audit_connectors",
    "skeleton_only",
    "gripper_mesh",
    "gripper_audit",
)


BODY_MARKERS = {
    "g_base": "1 1 0 1",
    "base": "1 1 0 1",
    "joint1": "1 0.2 0.2 1",
    "joint2": "1 0.55 0 1",
    "joint3": "1 0.9 0 1",
    "joint4": "0.2 1 0.2 1",
    "joint5": "0.1 0.8 1 1",
    "joint6": "0.2 0.25 1 1",
    "joint6_flange": "0.75 0.2 1 1",
    "link1": "1 0.2 0.2 1",
    "link2": "1 0.55 0 1",
    "link3": "1 0.9 0 1",
    "link4": "0.2 1 0.2 1",
    "link5": "0.1 0.8 1 1",
    "link6": "0.75 0.2 1 1",
    "gripper_base": "1 1 0 1",
    "gripper_left3": "0 0.85 0 1",
    "gripper_left1": "0 1 0.25 1",
    "gripper_left2": "0 0.55 0 1",
    "gripper_right3": "0 0.2 1 1",
    "gripper_right1": "0.1 0.45 1 1",
    "gripper_right2": "0 0 0.8 1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render isolated myCobot mesh visibility diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_visibility_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--model-profile", default=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--camera-search",
        choices=("fast", "exhaustive"),
        default="fast",
        help="Use a small deterministic camera set by default; exhaustive keeps the broad search for debugging.",
    )
    parser.add_argument(
        "--visibility-modes",
        default=",".join(VISIBILITY_MODES),
        help=(
            "Comma-separated layers to render: mesh_only, mesh_markers, pads_markers, "
            "collision_only, all, audit_mesh, audit_all, audit_connectors, "
            "skeleton_only, gripper_mesh, gripper_audit."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    visibility_modes = _parse_visibility_modes(args.visibility_modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _install_isolated_scene_override()
    env = nexus.MyCobotNexusEnv(
        nexus.MyCobotNexusConfig(
            asset_root=args.asset_root,
            work_dir=args.output_dir,
            official_gripper_root=args.official_gripper_root,
            model_profile=args.model_profile,
            width=args.width,
            height=args.height,
            teacher_grasp_attachment_enabled=False,
        )
    )
    renderer = None
    try:
        renderer = env._mujoco.Renderer(env.model, height=args.height, width=args.width)
        rgba_snapshot = _rgba_snapshot(env)
        env.reset(seed=1)
        gate7 = nexus._adaptive_gate7_arm_qpos(args.model_profile)
        neutral = tuple(0.0 for _ in gate7)
        poses = [
            ("neutral_open", neutral, 1.0),
            ("gate7_open", gate7, 1.0),
            ("gate7_mid", gate7, -0.1),
            ("gate7_closed", gate7, -0.7),
        ]
        frames: list[Path] = []
        reports: list[dict[str, Any]] = []
        for label, arm_qpos, gripper_command in poses:
            nexus._set_adaptive_gate_arm_pose(env, arm_qpos)
            env._set_gripper(command=gripper_command)
            env._mujoco.mj_forward(env.model, env.data)
            pose_report = _pose_report(env, label, gripper_command)
            mode_reports = []
            for visibility_mode in visibility_modes:
                _apply_visibility_mode(env, visibility_mode, rgba_snapshot)
                view_reports = []
                for view_name, target_names, mode in _view_specs(args.model_profile):
                    rgb, camera_meta = _render_best_view(env, renderer, target_names, mode, args.camera_search)
                    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    _draw_label(bgr, label, f"{visibility_mode} | {view_name}", camera_meta, env)
                    path = args.output_dir / f"{label}_{visibility_mode}_{view_name}.png"
                    cv2.imwrite(str(path), bgr)
                    frames.append(path)
                    view_reports.append({
                        "visibility_mode": visibility_mode,
                        "view": view_name,
                        "path": str(path),
                        **camera_meta,
                    })
                mode_reports.append({"visibility_mode": visibility_mode, "views": view_reports})
            pose_report["visibility_modes"] = mode_reports
            reports.append(pose_report)
        sheet_path = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "model_profile": args.model_profile,
            "scene_path": str(env.scene_path),
            "sheet_path": str(sheet_path),
            "frames": [str(path) for path in frames],
            "visibility_modes": visibility_modes,
            "poses": reports,
            "body_connectivity": _body_connectivity(env),
        }
        report_path = args.output_dir / "mesh_visibility_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _install_isolated_scene_override() -> None:
    original_build = nexus.build_mycobot_nexus_scene_model
    if getattr(original_build, "_mesh_visibility_wrapper", False):
        return

    def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> None:
        original_build(*wrapper_args, **wrapper_kwargs)
        scene_path = wrapper_kwargs.get("scene_path") if "scene_path" in wrapper_kwargs else wrapper_args[1]
        tree = ET.parse(scene_path)
        root = tree.getroot()
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for geom in list(root.findall(".//geom")):
            name = geom.attrib.get("name", "")
            mesh = geom.attrib.get("mesh", "")
            parent = parent_map.get(geom)
            if name in {"nexus_floor", "nexus_work_mat"} and parent is not None:
                parent.remove(geom)
                continue
            if name == nexus.TASK_CUBE_GEOM:
                geom.set("rgba", "0 0 0 0")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.attrib.pop("material", None)
            elif name == "left_finger_pad":
                geom.set("rgba", "0 1 0 0.75")
            elif name == "right_finger_pad":
                geom.set("rgba", "0 0.25 1 0.75")
            elif "gripper" in name or "gripper" in mesh:
                geom.set("rgba", "0.05 0.05 0.045 1")
                geom.attrib.pop("material", None)
            elif name.endswith("_visual_0") or mesh:
                geom.set("rgba", "0.78 0.75 0.62 1")
                geom.attrib.pop("material", None)
        for body in root.findall(".//body"):
            name = body.attrib.get("name", "")
            if name in BODY_MARKERS:
                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": f"{name}_origin_marker",
                        "type": "sphere",
                        "pos": "0 0 0",
                        "size": "0.0045",
                        "rgba": BODY_MARKERS[name],
                    },
                )
        _add_debug_connector_geoms(root)
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        global_node = visual.find("global")
        if global_node is None:
            global_node = ET.SubElement(visual, "global")
        global_node.set("offwidth", "1280")
        global_node.set("offheight", "960")
        map_node = visual.find("map")
        if map_node is None:
            map_node = ET.SubElement(visual, "map")
        map_node.set("znear", "0.001")
        map_node.set("zfar", "10")
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    wrapper._mesh_visibility_wrapper = True  # type: ignore[attr-defined]
    nexus.build_mycobot_nexus_scene_model = wrapper


def _add_debug_connector_geoms(root: ET.Element) -> None:
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for child in root.findall(".//body"):
        child_name = child.attrib.get("name", "")
        parent = parent_map.get(child)
        if parent is None or parent.tag != "body":
            continue
        parent_name = parent.attrib.get("name", "")
        if parent_name not in BODY_MARKERS or child_name not in BODY_MARKERS:
            continue
        pos = _float_triplet(child.attrib.get("pos", "0 0 0"))
        if float(np.linalg.norm(np.asarray(pos, dtype=float))) < 1e-6:
            continue
        ET.SubElement(
            parent,
            "geom",
            {
                "name": f"debug_connector_{parent_name}_to_{child_name}",
                "type": "capsule",
                "fromto": f"0 0 0 {pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}",
                "size": "0.004",
                "rgba": "0.95 0.18 0.05 0.82",
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _float_triplet(raw: str) -> tuple[float, float, float]:
    values = [float(value) for value in raw.split()]
    if len(values) != 3:
        raise ValueError(f"expected 3 floats, got: {raw}")
    return values[0], values[1], values[2]


def _parse_visibility_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(modes) - set(VISIBILITY_MODES))
    if unknown:
        raise SystemExit(f"unknown visibility mode(s): {', '.join(unknown)}")
    return modes or list(VISIBILITY_MODES)


def _rgba_snapshot(env: nexus.MyCobotNexusEnv) -> dict[str, np.ndarray]:
    return {
        "geom": env.model.geom_rgba.copy(),
        "site": env.model.site_rgba.copy(),
    }


def _apply_visibility_mode(env: nexus.MyCobotNexusEnv, mode: str, snapshot: dict[str, np.ndarray]) -> None:
    env.model.geom_rgba[:, :] = snapshot["geom"]
    env.model.site_rgba[:, :] = snapshot["site"]
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(env.model.geom_type[geom_id])
        mesh_id = int(env.model.geom_dataid[geom_id])
        is_pad = name in {"left_finger_pad", "right_finger_pad"}
        is_mesh = geom_type == int(env._mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0
        is_collision = not is_mesh and not is_pad and name != nexus.TASK_CUBE_GEOM
        if mode == "mesh_only":
            env.model.geom_rgba[geom_id, 3] = 1.0 if is_mesh else 0.0
        elif mode == "mesh_markers":
            env.model.geom_rgba[geom_id, 3] = 1.0 if is_mesh else 0.0
        elif mode == "pads_markers":
            env.model.geom_rgba[geom_id, 3] = 0.78 if is_pad else 0.08 if is_mesh else 0.0
        elif mode == "collision_only":
            if is_pad:
                env.model.geom_rgba[geom_id, :] = [0.0, 1.0, 0.0, 0.65] if name == "left_finger_pad" else [0.0, 0.25, 1.0, 0.65]
            elif is_collision:
                env.model.geom_rgba[geom_id, :] = [1.0, 0.0, 1.0, 0.45]
            else:
                env.model.geom_rgba[geom_id, 3] = 0.0
        elif mode == "all":
            if is_pad:
                env.model.geom_rgba[geom_id, 3] = 0.72
            elif is_collision:
                env.model.geom_rgba[geom_id, :] = [1.0, 0.0, 1.0, 0.32]
            elif is_mesh:
                env.model.geom_rgba[geom_id, 3] = 1.0
        elif mode == "audit_mesh":
            env.model.geom_rgba[geom_id, 3] = 1.0 if is_mesh else 0.0
        elif mode == "audit_all":
            if name.startswith("debug_connector_"):
                env.model.geom_rgba[geom_id, :] = [0.95, 0.18, 0.05, 0.9]
            elif is_pad:
                env.model.geom_rgba[geom_id, 3] = 0.34
            elif is_collision:
                env.model.geom_rgba[geom_id, :] = [1.0, 0.0, 1.0, 0.18]
            elif is_mesh:
                env.model.geom_rgba[geom_id, 3] = 1.0
        elif mode == "audit_connectors":
            if name.startswith("debug_connector_"):
                env.model.geom_rgba[geom_id, :] = [0.95, 0.18, 0.05, 0.95]
            elif is_pad:
                env.model.geom_rgba[geom_id, 3] = 0.18
            elif is_mesh:
                env.model.geom_rgba[geom_id, 3] = 0.08
            else:
                env.model.geom_rgba[geom_id, 3] = 0.0
        elif mode == "skeleton_only":
            if name.startswith("debug_connector_"):
                env.model.geom_rgba[geom_id, :] = [1.0, 0.05, 0.0, 1.0]
            else:
                env.model.geom_rgba[geom_id, 3] = 0.0
        elif mode == "gripper_mesh":
            if is_mesh and _is_gripper_geom(name):
                env.model.geom_rgba[geom_id, :] = _gripper_geom_color(name, alpha=1.0)
            else:
                env.model.geom_rgba[geom_id, 3] = 0.0
        elif mode == "gripper_audit":
            if name.startswith("debug_connector_"):
                env.model.geom_rgba[geom_id, :] = [1.0, 0.05, 0.0, 0.72]
            elif is_pad:
                env.model.geom_rgba[geom_id, 3] = 0.22
            elif is_mesh and _is_gripper_geom(name):
                env.model.geom_rgba[geom_id, :] = _gripper_geom_color(name, alpha=1.0)
            elif is_mesh:
                env.model.geom_rgba[geom_id, 3] = 0.04
            else:
                env.model.geom_rgba[geom_id, 3] = 0.0
    show_markers = mode in {
        "mesh_markers",
        "pads_markers",
        "all",
        "audit_all",
        "audit_connectors",
        "skeleton_only",
        "gripper_audit",
    }
    env.model.site_rgba[:, 3] = snapshot["site"][:, 3] if show_markers else 0.0
    if mode in {"skeleton_only", "gripper_audit"}:
        env.model.site_size[:] = np.maximum(env.model.site_size, 0.0075)
    elif mode == "gripper_mesh":
        env.model.site_rgba[:, 3] = 0.0


def _is_gripper_geom(name: str) -> bool:
    return name.startswith("gripper_")


def _gripper_geom_color(name: str, *, alpha: float) -> list[float]:
    if "base" in name:
        return [0.1, 0.1, 0.1, alpha]
    if "left1" in name:
        return [0.0, 0.85, 0.1, alpha]
    if "left2" in name:
        return [0.2, 1.0, 0.45, alpha]
    if "left3" in name:
        return [0.0, 0.55, 0.0, alpha]
    if "right1" in name:
        return [0.05, 0.25, 1.0, alpha]
    if "right2" in name:
        return [0.35, 0.55, 1.0, alpha]
    if "right3" in name:
        return [0.0, 0.05, 0.8, alpha]
    return [0.75, 0.75, 0.75, alpha]


def _view_specs(model_profile: str) -> list[tuple[str, tuple[str, ...], str]]:
    arm_names = (
        ("g_base", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint6_flange", "gripper_base")
        if model_profile == nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER
        else ("base", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base")
    )
    gripper_names = (
        "gripper_base",
        "gripper_left3",
        "gripper_left1",
        "gripper_left2",
        "gripper_right3",
        "gripper_right1",
        "gripper_right2",
    )
    return [
        ("full_auto_a", arm_names, "full"),
        ("full_auto_b", arm_names, "full_alt"),
        ("gripper_auto_a", gripper_names, "gripper"),
        ("gripper_auto_b", gripper_names, "gripper_alt"),
    ]


def _render_best_view(
    env: nexus.MyCobotNexusEnv,
    renderer: Any,
    body_names: tuple[str, ...],
    mode: str,
    search_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    target, radius = _target_and_radius(env, body_names)
    candidates = _camera_candidates(mode, radius, search_mode)
    best_rgb = None
    best_meta = None
    for azimuth, elevation, distance in candidates:
        camera = _camera(env, target, distance=distance, azimuth=azimuth, elevation=elevation)
        renderer.update_scene(env.data, camera=camera)
        rgb = renderer.render()
        score = _visibility_score(rgb)
        meta = {
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance,
            "search_mode": search_mode,
            "target": [float(value) for value in target],
            "target_radius": float(radius),
            **score,
        }
        if best_meta is None or meta["visibility_score"] > best_meta["visibility_score"]:
            best_rgb = rgb.copy()
            best_meta = meta
    assert best_rgb is not None and best_meta is not None
    return best_rgb, best_meta


def _camera_candidates(mode: str, radius: float, search_mode: str) -> list[tuple[float, float, float]]:
    radius = max(radius, 0.08)
    if search_mode == "fast":
        if mode == "full":
            raw = (
                (35.0, -25.0, 3.2),
                (110.0, -25.0, 3.2),
                (215.0, -20.0, 3.6),
                (320.0, -20.0, 3.6),
                (70.0, 10.0, 3.8),
            )
        elif mode == "full_alt":
            raw = (
                (45.0, -45.0, 3.8),
                (135.0, -45.0, 3.8),
                (225.0, -35.0, 4.2),
                (315.0, -35.0, 4.2),
                (90.0, 20.0, 4.0),
            )
        elif mode == "gripper":
            raw = (
                (45.0, -25.0, 4.0),
                (90.0, -25.0, 4.0),
                (135.0, -25.0, 4.0),
                (225.0, -20.0, 4.8),
                (315.0, 5.0, 5.2),
            )
        else:
            raw = (
                (30.0, -55.0, 5.0),
                (90.0, -65.0, 5.2),
                (150.0, -55.0, 5.0),
                (240.0, -35.0, 5.8),
                (300.0, -25.0, 5.8),
            )
        return [(azimuth, elevation, min(max(radius * multiplier, 0.16), 1.8)) for azimuth, elevation, multiplier in raw]

    if mode == "full":
        azimuths = (20.0, 60.0, 100.0, 140.0, 200.0, 260.0, 320.0)
        elevations = (-10.0, -25.0, -40.0, 10.0)
        multipliers = (2.6, 3.4, 4.4)
    elif mode == "full_alt":
        azimuths = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
        elevations = (-60.0, -35.0, 0.0, 25.0)
        multipliers = (3.0, 4.2)
    elif mode == "gripper":
        azimuths = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
        elevations = (-15.0, -35.0, 15.0)
        multipliers = (3.0, 4.5, 6.0)
    else:
        azimuths = (30.0, 75.0, 120.0, 165.0, 210.0, 255.0, 300.0)
        elevations = (-75.0, -55.0, -25.0, 0.0)
        multipliers = (3.2, 5.0, 7.0)
    return [
        (azimuth, elevation, min(max(radius * multiplier, 0.16), 1.8))
        for azimuth in azimuths
        for elevation in elevations
        for multiplier in multipliers
    ]


def _target_and_radius(env: nexus.MyCobotNexusEnv, body_names: tuple[str, ...]) -> tuple[np.ndarray, float]:
    points = []
    for name in body_names:
        body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            points.append(np.asarray(env.data.xpos[body_id], dtype=float))
    if not points:
        target = np.asarray(env._tcp_position(), dtype=float)
        return target, 0.12
    arr = np.vstack(points)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    target = (mins + maxs) * 0.5
    radius = float(np.linalg.norm(maxs - mins) * 0.5)
    return target, radius


def _camera(env: nexus.MyCobotNexusEnv, lookat: np.ndarray, *, distance: float, azimuth: float, elevation: float) -> Any:
    camera = env._mujoco.MjvCamera()
    camera.type = env._mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [float(value) for value in lookat]
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    return camera


def _visibility_score(rgb: np.ndarray) -> dict[str, Any]:
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation_pixels = int((hsv[:, :, 1] > 60).sum())
    dark_pixels = int((rgb.mean(axis=2) < 95).sum())
    bright_nonwhite = int(((rgb.mean(axis=2) > 120) & (hsv[:, :, 1] > 25)).sum())
    score = saturation_pixels + dark_pixels // 3 + bright_nonwhite // 2
    return {
        "saturation_pixels": saturation_pixels,
        "dark_pixels": dark_pixels,
        "bright_nonwhite_pixels": bright_nonwhite,
        "visibility_score": int(score),
    }


def _draw_label(
    frame: np.ndarray,
    pose_label: str,
    view_name: str,
    camera_meta: dict[str, Any],
    env: nexus.MyCobotNexusEnv,
) -> None:
    pad = np.asarray(env._finger_pad_midpoint(), dtype=float)
    tcp = np.asarray(env._tcp_position(), dtype=float)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (255, 255, 255), -1)
    cv2.putText(frame, f"{pose_label} | {view_name}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"score={camera_meta['visibility_score']} sat={camera_meta['saturation_pixels']} dark={camera_meta['dark_pixels']} "
        f"az={camera_meta['azimuth']:.0f} el={camera_meta['elevation']:.0f} d={camera_meta['distance']:.2f}",
        (18, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"tcp=({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) pad=({pad[0]:+.3f},{pad[1]:+.3f},{pad[2]:+.3f})",
        (18, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for path in frames:
        img = cv2.imread(str(path))
        tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
    rows = int(np.ceil(len(tiles) / 4))
    sheet = np.full((rows * 360, 4 * 480, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // 4) * 360
        x = (index % 4) * 480
        sheet[y:y + 360, x:x + 480] = tile
    path = output_dir / "mesh_visibility_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def _pose_report(env: nexus.MyCobotNexusEnv, label: str, gripper_command: float) -> dict[str, Any]:
    body_names = sorted(BODY_MARKERS)
    bodies = {}
    for name in body_names:
        body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            bodies[name] = [float(value) for value in env.data.xpos[body_id]]
    return {
        "label": label,
        "gripper_command": gripper_command,
        "tcp_position": env._tcp_position(),
        "pad_midpoint": env._finger_pad_midpoint(),
        "body_positions": bodies,
    }


def _body_connectivity(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    if env.config.model_profile == nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
        edges = [
            ("g_base", "joint1"),
            ("joint1", "joint2"),
            ("joint2", "joint3"),
            ("joint3", "joint4"),
            ("joint4", "joint5"),
            ("joint5", "joint6"),
            ("joint6", "joint6_flange"),
            ("joint6_flange", "gripper_base"),
        ]
    else:
        edges = [
            ("base", "link1"),
            ("link1", "link2"),
            ("link2", "link3"),
            ("link3", "link4"),
            ("link4", "link5"),
            ("link5", "link6"),
            ("link6", "gripper_base"),
        ]
    rows = []
    for parent, child in edges:
        parent_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, parent)
        child_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, child)
        if parent_id < 0 or child_id < 0:
            continue
        parent_pos = np.asarray(env.data.xpos[parent_id], dtype=float)
        child_pos = np.asarray(env.data.xpos[child_id], dtype=float)
        rows.append(
            {
                "parent": parent,
                "child": child,
                "distance": float(np.linalg.norm(child_pos - parent_pos)),
                "parent_position": parent_pos.tolist(),
                "child_position": child_pos.tolist(),
            }
        )
    return rows


if __name__ == "__main__":
    main()
