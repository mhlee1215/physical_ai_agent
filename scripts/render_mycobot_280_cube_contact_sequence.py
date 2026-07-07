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
ALL_BODY_NAMES = (*ARM_BODY_NAMES, *GRIPPER_BODY_NAMES)
ROBOT_LEFT_PAD = "left_finger_pad"
ROBOT_RIGHT_PAD = "right_finger_pad"
AUDIT_CUBE_HALF_SIZE = 0.0045
SEQUENCE = (
    ("aligned_open", 1.0, "cube aligned; jaws still open"),
    ("near_touch", 0.8, "pads moving toward cube"),
    ("grasp_contact", 0.7, "first compression/contact response"),
    ("hard_contact", 0.0, "deep contact response / solver repulsion zone"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render 280 full-body gripper/cube approach-contact sequence.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_cube_contact_sequence_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--hide-official-gripper-base-shell",
        action="store_true",
        help="Hide only the official 280 gripper_base visual shell; cube/pad/finger physics and visuals remain.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        _size_audit_cube(env, half_size=AUDIT_CUBE_HALF_SIZE)
        env._set_gripper(command=0.8)
        env._mujoco.mj_forward(env.model, env.data)
        cube_pos, cube_quat = _reference_cube_pose(env)
        frames: list[Path] = []
        panels: list[dict[str, Any]] = []
        for state_name, command, description in SEQUENCE:
            env._set_gripper(command=command)
            _set_cube_pose(env, cube_pos, cube_quat)
            env._mujoco.mj_forward(env.model, env.data)
            metrics = _contact_metrics(env, cube_half=AUDIT_CUBE_HALF_SIZE)
            full_target, full_radius = visibility._target_and_radius(env, ALL_BODY_NAMES)
            wrist_target, wrist_radius = visibility._target_and_radius(env, ("joint5", "joint6", "joint6_flange", *GRIPPER_BODY_NAMES))
            pad_target, pad_radius = _pad_target_and_radius(env)
            specs = (
                ("whole_robot", full_target, full_radius, 205.0, -18.0, 3.05),
                ("wrist_gripper_cube", wrist_target, wrist_radius, 215.0, -24.0, 2.45),
                ("pad_cube_gap", pad_target, pad_radius, *_pad_detail_camera(env), 2.75),
            )
            for view_name, target, radius, azimuth, elevation, multiplier in specs:
                camera = visibility._camera(env, target, distance=min(max(radius * multiplier, 0.13), 1.25), azimuth=azimuth, elevation=elevation)
                _apply_visibility(env, hide_gripper_base_shell=args.hide_official_gripper_base_shell)
                renderer.update_scene(env.data, camera=camera)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _draw_header(bgr, state_name, view_name, description, command, metrics, args.hide_official_gripper_base_shell)
                _draw_legend(bgr, hide_gripper_base_shell=args.hide_official_gripper_base_shell)
                out = args.output_dir / f"{state_name}_{view_name}.png"
                cv2.imwrite(str(out), bgr)
                frames.append(out)
                panels.append({"state": state_name, "command": command, "view": view_name, "path": str(out), "metrics": metrics})
        sheet = _write_sheet(args.output_dir, frames, cols=3, tile_size=(640, 480), name="cube_contact_sequence_sheet.png")
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(p) for p in frames],
            "panels": panels,
            "official_gripper_base_shell_hidden": bool(args.hide_official_gripper_base_shell),
            "note": "Static visual sequence: negative surface clearance indicates the contact-response/normal-force regime, not proven raw lift success.",
        }
        (args.output_dir / "cube_contact_sequence_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _restore_cube_collision(env: nexus.MyCobotNexusEnv) -> None:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {nexus.TASK_CUBE_GEOM}")
    env.model.geom_contype[geom_id] = 1
    env.model.geom_conaffinity[geom_id] = 1
    env.model.geom_friction[geom_id, :3] = [60.0, 6.0, 6.0]


def _size_audit_cube(env: nexus.MyCobotNexusEnv, *, half_size: float) -> None:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {nexus.TASK_CUBE_GEOM}")
    env.model.geom_size[geom_id, :3] = [half_size, half_size, half_size]


def _reference_cube_pose(env: nexus.MyCobotNexusEnv) -> tuple[np.ndarray, list[float]]:
    left = _geom_pos(env, ROBOT_LEFT_PAD)
    right = _geom_pos(env, ROBOT_RIGHT_PAD)
    midpoint = (left + right) * 0.5
    return midpoint, _quat_align_x_to_vector(right - left)


def _set_cube_pose(env: nexus.MyCobotNexusEnv, pos: np.ndarray, quat: list[float]) -> None:
    qpos_index = env._cube_freejoint_qpos_index
    env.data.qpos[qpos_index:qpos_index + 3] = pos
    env.data.qpos[qpos_index + 3:qpos_index + 7] = quat
    env.data.qvel[env._cube_freejoint_qvel_index:env._cube_freejoint_qvel_index + 6] = 0.0


def _contact_metrics(env: nexus.MyCobotNexusEnv, *, cube_half: float) -> dict[str, float | str]:
    left = _geom_pos(env, ROBOT_LEFT_PAD)
    right = _geom_pos(env, ROBOT_RIGHT_PAD)
    delta = right - left
    center_distance = float(np.linalg.norm(delta))
    axis = delta / max(center_distance, 1e-9)
    pad_extent_sum = 0.0
    for name in (ROBOT_LEFT_PAD, ROBOT_RIGHT_PAD):
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
        mat = np.asarray(env.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        size = np.asarray(env.model.geom_size[geom_id, :3], dtype=float)
        pad_extent_sum += float(sum(abs(np.dot(axis, mat[:, index])) * size[index] for index in range(3)))
    jaw_surface_gap = center_distance - pad_extent_sum
    cube_width = cube_half * 2.0
    clearance = jaw_surface_gap - cube_width
    if clearance > 0.001:
        regime = "clearance"
    elif clearance > -0.001:
        regime = "touching"
    else:
        regime = "contact_response"
    contacts = _pad_cube_contact_count(env)
    return {"pad_center_mm": center_distance * 1000.0, "jaw_gap_mm": jaw_surface_gap * 1000.0, "cube_width_mm": cube_width * 1000.0, "surface_clearance_mm": clearance * 1000.0, "pad_cube_contacts": contacts, "regime": regime}


def _pad_cube_contact_count(env: nexus.MyCobotNexusEnv) -> int:
    cube_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    pad_ids = {
        env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, ROBOT_LEFT_PAD),
        env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, ROBOT_RIGHT_PAD),
    }
    count = 0
    for index in range(int(env.data.ncon)):
        contact = env.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id in pair and pair.intersection(pad_ids):
            count += 1
    return count


def _quat_align_x_to_vector(vector: np.ndarray) -> list[float]:
    x_axis = vector / max(float(np.linalg.norm(vector)), 1e-9)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(x_axis, up))) > 0.92:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    y_axis = np.cross(up, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    return _quat_from_matrix(np.column_stack([x_axis, y_axis, z_axis]))


def _quat_from_matrix(rot: np.ndarray) -> list[float]:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        quat = np.asarray([0.25 / s, (rot[2, 1] - rot[1, 2]) * s, (rot[0, 2] - rot[2, 0]) * s, (rot[1, 0] - rot[0, 1]) * s])
    else:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0])
    quat = quat / max(float(np.linalg.norm(quat)), 1e-9)
    return [float(value) for value in quat]


def _geom_pos(env: nexus.MyCobotNexusEnv, name: str) -> np.ndarray:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {name}")
    return np.asarray(env.data.geom_xpos[geom_id], dtype=float)


def _pad_target_and_radius(env: nexus.MyCobotNexusEnv) -> tuple[np.ndarray, float]:
    arr = np.vstack([_geom_pos(env, ROBOT_LEFT_PAD), _geom_pos(env, ROBOT_RIGHT_PAD), _geom_pos(env, nexus.TASK_CUBE_GEOM)])
    return arr.mean(axis=0), max(float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)) * 1.6), 0.045)


def _pad_detail_camera(env: nexus.MyCobotNexusEnv) -> tuple[float, float]:
    left_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, ROBOT_LEFT_PAD)
    mat = np.asarray(env.data.geom_xmat[left_id], dtype=float).reshape(3, 3)
    direction = mat[:, 2] + 0.45 * mat[:, 1]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    return float(np.degrees(np.arctan2(direction[1], direction[0]))), float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))


def _apply_visibility(env: nexus.MyCobotNexusEnv, *, hide_gripper_base_shell: bool = False) -> None:
    env.model.site_rgba[:, 3] = 0.0
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(env.model.geom_type[geom_id])
        mesh_id = int(env.model.geom_dataid[geom_id])
        is_mesh = geom_type == int(env._mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0
        if hide_gripper_base_shell and name == "gripper_base_visual_0":
            env.model.geom_rgba[geom_id, 3] = 0.0
            continue
        if name in GRIPPER_GEOMS:
            rgba = list(GEOM_COLORS.get(name, (0.85, 0.85, 0.85, 1.0)))
            if name in {ROBOT_LEFT_PAD, ROBOT_RIGHT_PAD}:
                rgba[3] = 0.72
            env.model.geom_rgba[geom_id, :] = rgba
        elif name == nexus.TASK_CUBE_GEOM:
            env.model.geom_rgba[geom_id, :] = [0.96, 0.18, 0.08, 0.70]
        elif name.startswith("debug_connector_"):
            env.model.geom_rgba[geom_id, :] = [0.95, 0.18, 0.05, 0.72]
        elif is_mesh:
            env.model.geom_rgba[geom_id, :] = [0.86, 0.82, 0.64, 0.92]
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _draw_header(frame: np.ndarray, state: str, view: str, description: str, command: float, metrics: dict[str, float | str], hide_gripper_base_shell: bool = False) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 126), (255, 255, 255), -1)
    cv2.putText(frame, f"280 cube approach/contact | {state} | {view}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, description, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    shell_text = "; official gripper_base shell hidden" if hide_gripper_base_shell else ""
    cv2.putText(frame, f"robot-left pad=green; robot-right pad=blue; cube/pad collision and friction are enabled{shell_text}", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(frame, f"cmd={command:+.2f}; jaw gap={metrics['jaw_gap_mm']:.1f}mm; cube={metrics['cube_width_mm']:.1f}mm; clearance={metrics['surface_clearance_mm']:.1f}mm; contacts={metrics['pad_cube_contacts']}; {metrics['regime']}", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 90, 0) if metrics['regime'] == 'clearance' else (0, 0, 190), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray, *, hide_gripper_base_shell: bool = False) -> None:
    entries = [("arm/body", None), ("robot-left pad", ROBOT_LEFT_PAD), ("robot-right pad", ROBOT_RIGHT_PAD), ("small cube", "cube")]
    if hide_gripper_base_shell:
        entries.insert(1, ("gripper_base hidden", "hidden"))
    x0, y0 = 18, 154
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 245, y0 + 22 * len(entries) + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 245, y0 + 22 * len(entries) + 10), (75, 75, 75), 1)
    for index, (label, key) in enumerate(entries):
        y = y0 + index * 22
        if key is None:
            rgb = (0.86, 0.82, 0.64)
        elif key == "hidden":
            rgb = (0.65, 0.65, 0.65)
        elif key == "cube":
            rgb = (0.96, 0.18, 0.08)
        elif key == "connector":
            rgb = (0.95, 0.18, 0.05)
        else:
            rgb = GEOM_COLORS.get(key, (1, 1, 1, 1))[:3]
        color = tuple(int(255 * c) for c in rgb)
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), (color[2], color[1], color[0]), -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)


def _write_sheet(output_dir: Path, frames: list[Path], *, cols: int, tile_size: tuple[int, int], name: str) -> Path:
    width, height = tile_size
    tiles = [cv2.resize(cv2.imread(str(frame)), (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * height, cols * width, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * height
        x = (index % cols) * width
        sheet[y:y + height, x:x + width] = tile
    path_out = output_dir / name
    cv2.imwrite(str(path_out), sheet)
    return path_out


if __name__ == "__main__":
    main()
