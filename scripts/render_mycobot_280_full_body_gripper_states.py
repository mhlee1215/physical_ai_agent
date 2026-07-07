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
STATE_SPECS = (
    ("open", 1.0, False, "open gripper"),
    ("just_contact", 0.0, False, "closed / just-contact gripper"),
    ("box_between_jaws", 0.9, True, "gripper around small audit cube"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render full-body 280 robot states with corrected gripper pads.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_full_body_gripper_states_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--hide-official-gripper-base-shell",
        action="store_true",
        help="Hide only the official 280 gripper_base visual shell; physics pads/finger links remain visible and active.",
    )
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
        frames: list[Path] = []
        panels: list[dict[str, Any]] = []
        for state_name, command, show_cube, description in STATE_SPECS:
            nexus._set_adaptive_gate_arm_pose(env, nexus._adaptive_gate7_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER))
            env._set_gripper(command=command)
            env._mujoco.mj_forward(env.model, env.data)
            cube_pos = None
            if show_cube:
                _size_audit_cube(env, half_size=0.0045)
                cube_pos = _place_cube_between_pads(env)
                env._mujoco.mj_forward(env.model, env.data)
            else:
                _hide_task_cube(env)
            full_target, full_radius = visibility._target_and_radius(env, ALL_BODY_NAMES)
            wrist_target, wrist_radius = visibility._target_and_radius(env, ("joint5", "joint6", "joint6_flange", *GRIPPER_BODY_NAMES))
            pad_target, pad_radius = _pad_target_and_radius(env)
            specs = (
                ("whole_robot_left", full_target, full_radius, 225.0, -20.0, 3.0),
                ("whole_robot_right", full_target, full_radius, 135.0, -18.0, 3.0),
                ("wrist_gripper", wrist_target, wrist_radius, 215.0, -24.0, 2.45),
                ("pad_detail", pad_target, pad_radius, *_pad_detail_camera(env), 2.55),
            )
            for view_name, target, radius, azimuth, elevation, multiplier in specs:
                camera = visibility._camera(env, target, distance=min(max(radius * multiplier, 0.13), 1.25), azimuth=azimuth, elevation=elevation)
                _apply_visibility(env, show_cube=show_cube, hide_gripper_base_shell=args.hide_official_gripper_base_shell)
                renderer.update_scene(env.data, camera=camera)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _draw_header(bgr, state_name, view_name, description, command, cube_pos, args.hide_official_gripper_base_shell)
                _draw_legend(bgr, show_cube=show_cube, hide_gripper_base_shell=args.hide_official_gripper_base_shell)
                out = args.output_dir / f"{state_name}_{view_name}.png"
                cv2.imwrite(str(out), bgr)
                frames.append(out)
                panels.append({"state": state_name, "view": view_name, "path": str(out)})
        sheet = _write_sheet(args.output_dir, frames, cols=4, tile_size=(560, 420), name="full_body_gripper_states_sheet.png")
        compact = _write_sheet(args.output_dir, [p for p in frames if p.name.endswith("whole_robot_left.png") or p.name.endswith("wrist_gripper.png") or p.name.endswith("pad_detail.png")], cols=3, tile_size=(640, 480), name="full_body_gripper_states_compact.png")
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "compact_sheet_path": str(compact),
            "frames": [str(path) for path in frames],
            "panels": panels,
            "official_gripper_base_shell_hidden": bool(args.hide_official_gripper_base_shell),
            "note": "Full-body visual audit for corrected 280 gripper pads. The box_between_jaws state uses a small audit cube for visibility, not the full task cube size.",
        }
        report_path = args.output_dir / "full_body_gripper_states_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _hide_task_cube(env: nexus.MyCobotNexusEnv) -> None:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if geom_id >= 0:
        env.model.geom_rgba[geom_id, :] = [0.0, 0.0, 0.0, 0.0]
        qpos = env._cube_freejoint_qpos_index
        env.data.qpos[qpos:qpos + 3] = [0.4, 0.4, -0.2]


def _size_audit_cube(env: nexus.MyCobotNexusEnv, *, half_size: float) -> None:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {nexus.TASK_CUBE_GEOM}")
    env.model.geom_size[geom_id, :3] = [half_size, half_size, half_size]


def _place_cube_between_pads(env: nexus.MyCobotNexusEnv) -> np.ndarray:
    left = _geom_pos(env, "left_finger_pad")
    right = _geom_pos(env, "right_finger_pad")
    midpoint = (left + right) * 0.5
    qpos_index = env._cube_freejoint_qpos_index
    env.data.qpos[qpos_index:qpos_index + 3] = midpoint
    env.data.qpos[qpos_index + 3:qpos_index + 7] = _quat_align_x_to_vector(right - left)
    env.data.qvel[env._cube_freejoint_qvel_index:env._cube_freejoint_qvel_index + 6] = 0.0
    return midpoint


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
    points = [_geom_pos(env, "left_finger_pad"), _geom_pos(env, "right_finger_pad")]
    arr = np.vstack(points)
    return arr.mean(axis=0), max(float(np.linalg.norm(arr[1] - arr[0]) * 1.2), 0.04)


def _pad_detail_camera(env: nexus.MyCobotNexusEnv) -> tuple[float, float]:
    left_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    if left_id < 0:
        return 90.0, -70.0
    mat = np.asarray(env.data.geom_xmat[left_id], dtype=float).reshape(3, 3)
    direction = mat[:, 2] + 0.45 * mat[:, 1]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    return float(np.degrees(np.arctan2(direction[1], direction[0]))), float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))


def _apply_visibility(env: nexus.MyCobotNexusEnv, *, show_cube: bool, hide_gripper_base_shell: bool = False) -> None:
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
            if name in {"left_finger_pad", "right_finger_pad"}:
                rgba[3] = 0.72
            elif name in {"gripper_left2_visual_0", "gripper_left3_visual_0", "gripper_right2_visual_0", "gripper_right3_visual_0"}:
                rgba[3] = 0.78
            env.model.geom_rgba[geom_id, :] = rgba
        elif name == nexus.TASK_CUBE_GEOM:
            env.model.geom_rgba[geom_id, :] = [0.96, 0.18, 0.08, 0.68 if show_cube else 0.0]
        elif name.startswith("debug_connector_"):
            env.model.geom_rgba[geom_id, :] = [0.95, 0.18, 0.05, 0.65]
        elif is_mesh:
            env.model.geom_rgba[geom_id, :] = [0.86, 0.82, 0.64, 0.96]
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _draw_header(frame: np.ndarray, state: str, view: str, description: str, command: float, cube_pos: np.ndarray | None, hide_gripper_base_shell: bool = False) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 116), (255, 255, 255), -1)
    cv2.putText(frame, f"280 full body gripper state | {state} | {view}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, description, (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (45, 45, 45), 1, cv2.LINE_AA)
    cube_text = "small audit cube visible" if cube_pos is not None else "cube hidden"
    shell_text = "; official gripper_base shell hidden" if hide_gripper_base_shell else ""
    cv2.putText(frame, f"gripper command={command:+.2f}; robot-left pad=green; robot-right pad=blue; {cube_text}{shell_text}", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray, *, show_cube: bool, hide_gripper_base_shell: bool = False) -> None:
    entries = [("arm/body", None), ("kinematic connector", "connector"), ("gripper base", "gripper_base_visual_0"), ("distal fingers", "gripper_left1_visual_0"), ("support links", "gripper_left2_visual_0"), ("robot-left pad", "left_finger_pad"), ("robot-right pad", "right_finger_pad")]
    if hide_gripper_base_shell:
        entries = [entry for entry in entries if entry[1] != "gripper_base_visual_0"]
    if show_cube:
        entries.append(("small audit cube", "cube"))
    x0, y0 = 18, 144
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (75, 75, 75), 1)
    for index, (label, key) in enumerate(entries):
        y = y0 + index * 22
        if key is None:
            rgb = (0.86, 0.82, 0.64)
        elif key == "cube":
            rgb = (0.96, 0.18, 0.08)
        elif key == "connector":
            rgb = (0.95, 0.18, 0.05)
        else:
            rgb = GEOM_COLORS.get(key, (1, 1, 1, 1))[:3]
        color = tuple(int(255 * channel) for channel in rgb)
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), (color[2], color[1], color[0]), -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)


def _write_sheet(output_dir: Path, frames: list[Path], *, cols: int, tile_size: tuple[int, int], name: str) -> Path:
    tiles = []
    width, height = tile_size
    for frame in frames:
        img = cv2.imread(str(frame))
        tiles.append(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
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
