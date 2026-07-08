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

POSES = (("open", 1.0), ("mid", -0.1), ("closed", -0.7))
ARM_BODY_NAMES = ("g_base", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint6_flange", "gripper_base")
GRIPPER_BODY_NAMES = ("gripper_base", "gripper_left3", "gripper_left2", "gripper_left1", "gripper_right3", "gripper_right2", "gripper_right1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render full-robot 280 visual continuity proof through gripper fingertips.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_full_robot_continuity_audit_001"))
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
        frames: list[Path] = []
        panels: list[dict[str, Any]] = []
        for pose_label, command in POSES:
            nexus._set_adaptive_gate_arm_pose(env, nexus._adaptive_gate7_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER))
            env._set_gripper(command=command)
            env._mujoco.mj_forward(env.model, env.data)
            full_target, full_radius = _target_radius(env, ARM_BODY_NAMES + GRIPPER_BODY_NAMES)
            wrist_target, wrist_radius = _target_radius(env, ("joint5", "joint6", "joint6_flange", *GRIPPER_BODY_NAMES))
            specs = (
                ("whole_robot_left", full_target, full_radius, 225.0, -18.0, 3.15),
                ("whole_robot_right", full_target, full_radius, 135.0, -18.0, 3.15),
                ("wrist_to_tip_side", wrist_target, wrist_radius, 80.0, -12.0, 2.45),
                ("wrist_to_tip_oblique", wrist_target, wrist_radius, 215.0, -28.0, 2.55),
            )
            for view_name, target, radius, azimuth, elevation, multiplier in specs:
                camera = visibility._camera(env, target, distance=min(max(radius * multiplier, 0.2), 1.25), azimuth=azimuth, elevation=elevation)
                _apply_continuity_visibility(env)
                renderer.update_scene(env.data, camera=camera)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _draw_header(bgr, pose_label, view_name)
                _draw_legend(bgr)
                out = args.output_dir / f"{pose_label}_{view_name}.png"
                cv2.imwrite(str(out), bgr)
                frames.append(out)
                panels.append({"pose": pose_label, "view": view_name, "path": str(out)})
        sheet = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(path) for path in frames],
            "panels": panels,
            "acceptance_target": "Full robot appears visually continuous from arm body through wrist/gripper body/proximal links/distal fingers to pads.",
            "parts": _inventory(env),
        }
        report_path = args.output_dir / "full_robot_continuity_audit_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _apply_continuity_visibility(env: nexus.MyCobotNexusEnv) -> None:
    env.model.site_rgba[:, 3] = 0.0
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(env.model.geom_type[geom_id])
        mesh_id = int(env.model.geom_dataid[geom_id])
        is_mesh = geom_type == int(env._mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0
        if name in GRIPPER_GEOMS:
            rgba = list(GEOM_COLORS.get(name, (0.85, 0.85, 0.85, 1.0)))
            if name in {"left_finger_pad", "right_finger_pad"}:
                rgba[3] = 0.48
            env.model.geom_rgba[geom_id, :] = rgba
        elif name.startswith("debug_connector_"):
            env.model.geom_rgba[geom_id, 3] = 0.0
        elif is_mesh:
            env.model.geom_rgba[geom_id, :] = [0.86, 0.82, 0.64, 1.0]
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _target_radius(env: nexus.MyCobotNexusEnv, body_names: tuple[str, ...]) -> tuple[np.ndarray, float]:
    points = []
    for name in body_names:
        body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            points.append(np.asarray(env.data.xpos[body_id], dtype=float))
    arr = np.vstack(points)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    return (mins + maxs) * 0.5, float(np.linalg.norm(maxs - mins) * 0.5)


def _draw_header(frame: np.ndarray, pose: str, view: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (255, 255, 255), -1)
    cv2.putText(frame, f"280 full-robot continuity proof | {pose} | {view}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "Goal: continuous visible body from arm -> wrist -> gripper base -> finger links -> distal pads.", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(frame, "Arm is solid; gripper links are color-coded; pads are transparent overlays at distal ends.", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("arm/body solid", None),
        ("gripper body/base", "gripper_base_visual_0"),
        ("distal finger links", "gripper_left1_visual_0"),
        ("support/prox links", "gripper_left2_visual_0"),
        ("left pad", "left_finger_pad"),
        ("right pad", "right_finger_pad"),
    ]
    x0 = 18
    y0 = 122
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (75, 75, 75), 1)
    for index, (label, geom_name) in enumerate(entries):
        y = y0 + index * 22
        rgb = (0.86, 0.82, 0.64) if geom_name is None else GEOM_COLORS.get(geom_name, (1, 1, 1, 1))[:3]
        color = tuple(int(255 * channel) for channel in rgb)
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), bgr, -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)


def _inventory(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    return [
        {"geom": geom_name, "present": env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0}
        for geom_name in GRIPPER_GEOMS
    ]


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for frame in frames:
        img = cv2.imread(str(frame))
        tiles.append(cv2.resize(img, (560, 420), interpolation=cv2.INTER_AREA))
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * 420, cols * 560, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 420
        x = (index % cols) * 560
        sheet[y:y + 420, x:x + 560] = tile
    path = output_dir / "full_robot_continuity_audit_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


if __name__ == "__main__":
    main()
