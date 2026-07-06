#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402
from scripts import render_mycobot_mesh_visibility_inspection as visibility  # noqa: E402
from scripts.render_mycobot_280_gripper_part_audit import GEOM_COLORS, GRIPPER_GEOMS  # noqa: E402

ARM_BODY_NAMES = ("g_base", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint6_flange", "gripper_base")
GRIPPER_BODY_NAMES = ("gripper_base", "gripper_left3", "gripper_left2", "gripper_left1", "gripper_right3", "gripper_right2", "gripper_right1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render 280 gripper holding-cube visual continuity audit.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_cube_hold_audit_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visibility._install_isolated_scene_override()
    env = nexus.MyCobotNexusEnv(
        nexus.MyCobotNexusConfig(
            asset_root=args.asset_root,
            work_dir=args.output_dir,
            official_gripper_root=args.official_gripper_root,
            model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            width=args.width,
            height=args.height,
            teacher_grasp_attachment_enabled=False,
        )
    )
    renderer = None
    try:
        renderer = env._mujoco.Renderer(env.model, height=args.height, width=args.width)
        env.reset(seed=1)
        nexus._set_adaptive_gate_arm_pose(env, nexus._adaptive_gate7_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER))
        env._set_gripper(command=-0.7)
        env._mujoco.mj_forward(env.model, env.data)
        _shrink_cube_for_readable_hold_audit(env)
        cube_pos = _place_cube_between_pads(env)
        env._mujoco.mj_forward(env.model, env.data)
        frames: list[Path] = []
        full_target, full_radius = visibility._target_and_radius(env, ARM_BODY_NAMES + GRIPPER_BODY_NAMES)
        grip_target, grip_radius = visibility._target_and_radius(env, GRIPPER_BODY_NAMES)
        specs = (
            ("full_left_closed_hold", full_target, full_radius, 225.0, -18.0, 3.15),
            ("full_right_closed_hold", full_target, full_radius, 135.0, -18.0, 3.15),
            ("pad_cube_side_a", grip_target, grip_radius, 70.0, -3.0, 1.62),
            ("pad_cube_side_b", grip_target, grip_radius, 250.0, -3.0, 1.62),
            ("pad_cube_oblique", grip_target, grip_radius, 215.0, -20.0, 1.78),
            ("pad_cube_top", grip_target, grip_radius, 90.0, -82.0, 1.95),
        )
        panels = []
        for view, target, radius, azimuth, elevation, multiplier in specs:
            camera = visibility._camera(env, target, distance=min(max(radius * multiplier, 0.2), 1.25), azimuth=azimuth, elevation=elevation)
            _apply_cube_hold_visibility(env)
            renderer.update_scene(env.data, camera=camera)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            _draw_header(bgr, view, cube_pos)
            _draw_legend(bgr)
            out = args.output_dir / f"cube_hold_{view}.png"
            cv2.imwrite(str(out), bgr)
            frames.append(out)
            panels.append({"view": view, "path": str(out)})
        sheet = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(path) for path in frames],
            "panels": panels,
            "cube_position": [float(value) for value in cube_pos],
            "parts": _inventory(env),
            "note": "Visual holding audit: a smaller transparent cube is placed at the midpoint of the corrected 280 fingertip pads so the body, finger links, distal pads, and cube are simultaneously readable.",
        }
        report_path = args.output_dir / "cube_hold_audit_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _place_cube_between_pads(env: nexus.MyCobotNexusEnv) -> np.ndarray:
    left = _geom_pos(env, "left_finger_pad")
    right = _geom_pos(env, "right_finger_pad")
    midpoint = (left + right) * 0.5
    qpos_index = env._cube_freejoint_qpos_index
    env.data.qpos[qpos_index:qpos_index + 3] = midpoint
    env.data.qpos[qpos_index + 3:qpos_index + 7] = _quat_align_x_to_vector(right - left)
    env.data.qvel[env._cube_freejoint_qvel_index:env._cube_freejoint_qvel_index + 6] = 0.0
    env._cube_initial_pos = [float(value) for value in midpoint]
    return midpoint


def _shrink_cube_for_readable_hold_audit(env: nexus.MyCobotNexusEnv) -> None:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {nexus.TASK_CUBE_GEOM}")
    env.model.geom_size[geom_id, :3] = [0.010, 0.010, 0.010]


def _quat_align_x_to_vector(vector: np.ndarray) -> list[float]:
    x_axis = vector / max(float(np.linalg.norm(vector)), 1e-9)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(x_axis, up))) > 0.92:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    y_axis = np.cross(up, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return _quat_from_matrix(rot)


def _quat_from_matrix(rot: np.ndarray) -> list[float]:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        index = int(np.argmax(np.diagonal(rot)))
        if index == 0:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif index == 1:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=float)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-9)
    return [float(value) for value in quat]


def _geom_pos(env: nexus.MyCobotNexusEnv, name: str) -> np.ndarray:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {name}")
    return np.asarray(env.data.geom_xpos[geom_id], dtype=float)


def _apply_cube_hold_visibility(env: nexus.MyCobotNexusEnv) -> None:
    env.model.site_rgba[:, 3] = 0.0
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(env.model.geom_type[geom_id])
        mesh_id = int(env.model.geom_dataid[geom_id])
        is_mesh = geom_type == int(env._mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0
        if name in GRIPPER_GEOMS:
            rgba = list(GEOM_COLORS.get(name, (0.85, 0.85, 0.85, 1.0)))
            if name in {"left_finger_pad", "right_finger_pad"}:
                rgba[3] = 0.86
            elif name in {"gripper_left2_visual_0", "gripper_left3_visual_0", "gripper_right2_visual_0", "gripper_right3_visual_0"}:
                rgba[3] = 0.28
            elif name in {"gripper_left1_visual_0", "gripper_right1_visual_0"}:
                rgba[3] = 0.72
            env.model.geom_rgba[geom_id, :] = rgba
        elif name == nexus.TASK_CUBE_GEOM:
            env.model.geom_rgba[geom_id, :] = [0.95, 0.18, 0.08, 0.62]
        elif name.startswith("debug_connector_"):
            env.model.geom_rgba[geom_id, 3] = 0.0
        elif is_mesh:
            env.model.geom_rgba[geom_id, :] = [0.86, 0.82, 0.64, 0.92]
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _draw_header(frame: np.ndarray, view: str, cube_pos: np.ndarray) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (255, 255, 255), -1)
    cv2.putText(frame, f"280 cube-hold visual audit | closed gripper | {view}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "Small transparent audit cube is centered between opposing corrected fingertip pads; pads/fingers/body are color-coded.", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(frame, f"cube=({cube_pos[0]:+.3f},{cube_pos[1]:+.3f},{cube_pos[2]:+.3f})", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("arm/body", None),
        ("gripper body/base", "gripper_base_visual_0"),
        ("distal finger links", "gripper_left1_visual_0"),
        ("support/prox links", "gripper_left2_visual_0"),
        ("left pad", "left_finger_pad"),
        ("right pad", "right_finger_pad"),
        ("held cube", "cube"),
    ]
    x0 = 18
    y0 = 122
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (75, 75, 75), 1)
    for index, (label, key) in enumerate(entries):
        y = y0 + index * 22
        if key is None:
            rgb = (0.86, 0.82, 0.64)
        elif key == "cube":
            rgb = (0.95, 0.18, 0.08)
        else:
            rgb = GEOM_COLORS.get(key, (1, 1, 1, 1))[:3]
        color = tuple(int(255 * channel) for channel in rgb)
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), (color[2], color[1], color[0]), -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)


def _inventory(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    return [
        {"geom": geom_name, "present": env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0}
        for geom_name in (*GRIPPER_GEOMS, nexus.TASK_CUBE_GEOM)
    ]


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for frame in frames:
        img = cv2.imread(str(frame))
        tiles.append(cv2.resize(img, (560, 420), interpolation=cv2.INTER_AREA))
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * 420, cols * 560, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 420
        x = (index % cols) * 560
        sheet[y:y + 420, x:x + 560] = tile
    path = output_dir / "cube_hold_audit_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


if __name__ == "__main__":
    main()
