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

ARM_ALPHA = 0.58
PAD_ALPHA = 0.58


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render full-body 280 gripper connection audit views.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_gripper_connection_audit_001"))
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
        poses = (
            ("open", 1.0),
            ("mid", -0.1),
            ("closed", -0.7),
        )
        camera_specs = (
            ("full_context_a", "full", 215.0, -18.0, 3.7),
            ("full_context_b", "full", 135.0, -20.0, 3.7),
            ("connection_side", "gripper", 70.0, -18.0, 2.4),
            ("connection_top", "gripper", 120.0, -62.0, 2.7),
        )
        frames: list[Path] = []
        panels: list[dict[str, Any]] = []
        for pose_label, gripper_command in poses:
            nexus._set_adaptive_gate_arm_pose(env, nexus._adaptive_gate7_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER))
            env._set_gripper(command=gripper_command)
            env._mujoco.mj_forward(env.model, env.data)
            full_target, full_radius = visibility._target_and_radius(env, ("g_base", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint6_flange", "gripper_base"))
            grip_target, grip_radius = visibility._target_and_radius(env, ("gripper_base", "gripper_left3", "gripper_left1", "gripper_left2", "gripper_right3", "gripper_right1", "gripper_right2"))
            for view_name, target_kind, azimuth, elevation, multiplier in camera_specs:
                target = full_target if target_kind == "full" else grip_target
                radius = full_radius if target_kind == "full" else grip_radius
                camera = visibility._camera(
                    env,
                    target,
                    distance=min(max(radius * multiplier, 0.18), 1.4),
                    azimuth=azimuth,
                    elevation=elevation,
                )
                _apply_connection_visibility(env)
                renderer.update_scene(env.data, camera=camera)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _draw_header(bgr, pose_label, view_name)
                _draw_legend(bgr)
                path_out = args.output_dir / f"{pose_label}_{view_name}.png"
                cv2.imwrite(str(path_out), bgr)
                frames.append(path_out)
                panels.append({"pose": pose_label, "view": view_name, "path": str(path_out)})
        sheet = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(frame) for frame in frames],
            "panels": panels,
            "parts": _inventory(env),
        }
        report_path = args.output_dir / "gripper_connection_audit_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _apply_connection_visibility(env: nexus.MyCobotNexusEnv) -> None:
    env.model.site_rgba[:, 3] = 0.0
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(env.model.geom_type[geom_id])
        mesh_id = int(env.model.geom_dataid[geom_id])
        is_mesh = geom_type == int(env._mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0
        if name in GRIPPER_GEOMS:
            rgba = list(GEOM_COLORS.get(name, (0.85, 0.85, 0.85, 1.0)))
            if name in {"left_finger_pad", "right_finger_pad"}:
                rgba[3] = PAD_ALPHA
            env.model.geom_rgba[geom_id, :] = rgba
        elif name.startswith("debug_connector_"):
            env.model.geom_rgba[geom_id, 3] = 0.0
        elif is_mesh:
            env.model.geom_rgba[geom_id, :] = [0.82, 0.78, 0.62, ARM_ALPHA]
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _draw_header(frame: np.ndarray, pose_label: str, view_name: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 88), (255, 255, 255), -1)
    cv2.putText(frame, f"280 full-body gripper connection audit | {pose_label} | {view_name}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "Arm is translucent; gripper body/finger links and pads are saturated in one full-body-context render.", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("body/base", "gripper_base_visual_0"),
        ("distal fingers", "gripper_left1_visual_0"),
        ("support/prox links", "gripper_left2_visual_0"),
        ("left pad", "left_finger_pad"),
        ("right pad", "right_finger_pad"),
    ]
    x0 = 18
    y0 = 112
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 250, y0 + 22 * len(entries) + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 250, y0 + 22 * len(entries) + 10), (80, 80, 80), 1)
    for index, (label, geom_name) in enumerate(entries):
        y = y0 + index * 22
        color = tuple(int(255 * channel) for channel in GEOM_COLORS.get(geom_name, (1, 1, 1, 1))[:3])
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), bgr, -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)


def _inventory(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    rows = []
    for geom_name in GRIPPER_GEOMS:
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        rows.append({"geom": geom_name, "present": geom_id >= 0})
    return rows


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for frame in frames:
        img = cv2.imread(str(frame))
        tiles.append(cv2.resize(img, (520, 390), interpolation=cv2.INTER_AREA))
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * 390, cols * 520, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 390
        x = (index % cols) * 520
        sheet[y:y + 390, x:x + 520] = tile
    path_out = output_dir / "gripper_connection_audit_sheet.png"
    cv2.imwrite(str(path_out), sheet)
    return path_out


if __name__ == "__main__":
    main()
