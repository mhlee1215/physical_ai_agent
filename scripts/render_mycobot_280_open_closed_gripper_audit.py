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

BODY_NAMES = (
    "gripper_base",
    "gripper_left3",
    "gripper_left2",
    "gripper_left1",
    "gripper_right3",
    "gripper_right2",
    "gripper_right1",
)
VISUAL_LINKS = {
    "gripper_base_visual_0",
    "gripper_left3_visual_0",
    "gripper_left2_visual_0",
    "gripper_left1_visual_0",
    "gripper_right3_visual_0",
    "gripper_right2_visual_0",
    "gripper_right1_visual_0",
}
DISTAL_LINKS = {"gripper_left1_visual_0", "gripper_right1_visual_0"}
PAD_GEOMS = {"left_finger_pad", "right_finger_pad"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render open/closed 280 adaptive gripper visual audit.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_open_closed_gripper_audit_001"))
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
        frames: list[Path] = []
        rows: list[dict[str, Any]] = []
        pose_specs = (("open", 1.0), ("closed", -0.7))
        panel_specs = (
            ("complete_gripper", "all visual finger links plus collision pads", VISUAL_LINKS | PAD_GEOMS, 0.92, 0.92),
            ("finger_mesh_only", "visual gripper meshes only; no collision pads", VISUAL_LINKS, 0.96, 0.0),
            ("pad_collision_only", "physics collision pads only; visual finger meshes hidden", PAD_GEOMS, 0.0, 0.92),
            ("distal_fingers_and_pads", "distal finger ends plus the actual collision pads", DISTAL_LINKS | PAD_GEOMS, 0.98, 0.98),
        )
        camera_specs = (
            ("front", 135.0, -12.0, 2.45),
            ("side", 45.0, -10.0, 2.30),
            ("top", 90.0, -72.0, 2.55),
        )
        for pose_name, command in pose_specs:
            env._set_gripper(command=command)
            env._mujoco.mj_forward(env.model, env.data)
            target, radius = _gripper_target_and_radius(env)
            for panel_name, description, visible, visual_alpha, pad_alpha in panel_specs:
                for view_name, azimuth, elevation, multiplier in camera_specs:
                    camera = visibility._camera(
                        env,
                        target,
                        distance=min(max(radius * multiplier, 0.16), 0.8),
                        azimuth=azimuth,
                        elevation=elevation,
                    )
                    _apply_visibility(env, visible, visual_alpha=visual_alpha, pad_alpha=pad_alpha)
                    renderer.update_scene(env.data, camera=camera)
                    rgb = renderer.render()
                    frame = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    _draw_header(frame, pose_name, panel_name, view_name, description, command)
                    _draw_legend(frame)
                    out = args.output_dir / f"{pose_name}_{panel_name}_{view_name}.png"
                    cv2.imwrite(str(out), frame)
                    frames.append(out)
                    rows.append({"pose": pose_name, "command": command, "panel": panel_name, "view": view_name, "path": str(out)})
        sheet = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(path) for path in frames],
            "panels": rows,
            "inventory": _inventory(env),
            "note": "Open/closed gripper audit separates visual meshes from physics collision pads so distal finger ends and pads are readable.",
        }
        report_path = args.output_dir / "open_closed_gripper_audit_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _apply_visibility(env: nexus.MyCobotNexusEnv, visible: set[str], *, visual_alpha: float, pad_alpha: float) -> None:
    env.model.geom_rgba[:, 3] = 0.0
    env.model.site_rgba[:, 3] = 0.0
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name not in visible:
            continue
        rgba = list(GEOM_COLORS.get(name, (0.85, 0.82, 0.64, 1.0)))
        if name in PAD_GEOMS:
            rgba[3] = pad_alpha
        elif name in VISUAL_LINKS:
            rgba[3] = visual_alpha
        if name == "gripper_base_visual_0":
            rgba[3] = max(rgba[3], min(visual_alpha, 0.82))
        env.model.geom_rgba[geom_id, :] = rgba


def _draw_header(frame: np.ndarray, pose_name: str, panel_name: str, view_name: str, description: str, command: float) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 116), (255, 255, 255), -1)
    cv2.putText(
        frame,
        f"280 adaptive gripper | {pose_name} | {panel_name} | {view_name}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, description, (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"gripper command={command:+.2f}; pads are physics proxy boxes, finger links are visual meshes",
        (18, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )


def _draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("gripper body/base", "gripper_base_visual_0"),
        ("left distal finger mesh", "gripper_left1_visual_0"),
        ("left support/prox meshes", "gripper_left2_visual_0"),
        ("right distal finger mesh", "gripper_right1_visual_0"),
        ("right support/prox meshes", "gripper_right2_visual_0"),
        ("left collision pad", "left_finger_pad"),
        ("right collision pad", "right_finger_pad"),
    ]
    x0 = 18
    y0 = 148
    width = 366
    height = 22 * len(entries) + 14
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + width, y0 + height), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + width, y0 + height), (65, 65, 65), 1)
    for index, (label, geom_name) in enumerate(entries):
        y = y0 + index * 22
        rgb = GEOM_COLORS.get(geom_name, (1, 1, 1, 1))[:3]
        color = tuple(int(255 * channel) for channel in rgb)
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), (color[2], color[1], color[0]), -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)


def _gripper_target_and_radius(env: nexus.MyCobotNexusEnv) -> tuple[np.ndarray, float]:
    points = []
    for name in BODY_NAMES:
        body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            points.append(np.asarray(env.data.xpos[body_id], dtype=float))
    for name in GRIPPER_GEOMS:
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            points.append(np.asarray(env.data.geom_xpos[geom_id], dtype=float))
    arr = np.vstack(points)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    radius = float(np.linalg.norm(maxs - mins) * 0.5)
    return (mins + maxs) * 0.5, max(radius, 0.07)


def _inventory(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    rows = []
    for geom_name in GRIPPER_GEOMS:
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            rows.append({"geom": geom_name, "present": False})
            continue
        body_id = int(env.model.geom_bodyid[geom_id])
        rows.append(
            {
                "geom": geom_name,
                "present": True,
                "parent_body": env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_BODY, body_id),
                "contype": int(env.model.geom_contype[geom_id]),
                "conaffinity": int(env.model.geom_conaffinity[geom_id]),
                "world_pos": [float(value) for value in env.data.geom_xpos[geom_id]],
            }
        )
    return rows


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for frame in frames:
        img = cv2.imread(str(frame))
        tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
    cols = 6
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * 360, cols * 480, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 360
        x = (index % cols) * 480
        sheet[y:y + 360, x:x + 480] = tile
    path_out = output_dir / "open_closed_gripper_audit_sheet.png"
    cv2.imwrite(str(path_out), sheet)
    return path_out


if __name__ == "__main__":
    main()
