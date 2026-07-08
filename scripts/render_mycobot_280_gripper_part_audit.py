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

GRIPPER_GEOMS = (
    "gripper_base_visual_0",
    "gripper_left3_visual_0",
    "gripper_left1_visual_0",
    "left_finger_pad",
    "gripper_left2_visual_0",
    "gripper_right3_visual_0",
    "gripper_right1_visual_0",
    "right_finger_pad",
    "gripper_right2_visual_0",
)

LABELS = {
    "gripper_base_visual_0": "gripper body/base",
    "gripper_left1_visual_0": "left distal finger",
    "gripper_left2_visual_0": "left support link",
    "gripper_left3_visual_0": "left proximal link",
    "gripper_right1_visual_0": "right distal finger",
    "gripper_right2_visual_0": "right support link",
    "gripper_right3_visual_0": "right proximal link",
    "left_finger_pad": "left contact pad",
    "right_finger_pad": "right contact pad",
}


GEOM_COLORS = {
    "gripper_base_visual_0": (1.0, 0.72, 0.05, 1.0),
    "gripper_left1_visual_0": (0.72, 1.0, 0.56, 0.86),
    "gripper_left2_visual_0": (0.45, 1.0, 0.25, 1.0),
    "gripper_left3_visual_0": (0.0, 0.62, 0.0, 1.0),
    "gripper_right1_visual_0": (0.50, 0.68, 1.0, 0.86),
    "gripper_right2_visual_0": (0.2, 0.72, 1.0, 1.0),
    "gripper_right3_visual_0": (0.0, 0.08, 0.9, 1.0),
    "left_finger_pad": (0.01, 0.12, 0.02, 1.0),
    "right_finger_pad": (0.01, 0.03, 0.18, 1.0),
}

BODY_NAMES = (
    "gripper_base",
    "gripper_left3",
    "gripper_left1",
    "gripper_left2",
    "gripper_right3",
    "gripper_right1",
    "gripper_right2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a part-by-part 280 adaptive gripper visual audit.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_gripper_part_audit_001"))
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
        env._set_gripper(command=1.0)
        env._mujoco.mj_forward(env.model, env.data)
        target, radius = _gripper_target_and_radius(env)
        camera_specs = (
            ("front", 135.0, -14.0, 2.6),
            ("side", 45.0, -16.0, 2.6),
            ("top", 120.0, -68.0, 2.8),
            ("iso", 215.0, -28.0, 3.0),
        )
        frames: list[Path] = []
        rows: list[dict[str, Any]] = []
        for view_name, azimuth, elevation, multiplier in camera_specs:
            camera = visibility._camera(
                env,
                target,
                distance=min(max(radius * multiplier, 0.18), 1.0),
                azimuth=azimuth,
                elevation=elevation,
            )
            frame = _render_panel(env, renderer, camera, "complete_gripper_and_pads", set(GRIPPER_GEOMS), args.output_dir)
            _draw_legend(frame)
            path_out = args.output_dir / f"complete_{view_name}.png"
            cv2.imwrite(str(path_out), frame)
            frames.append(path_out)
            rows.append({"panel": "complete_gripper_and_pads", "view": view_name, "path": str(path_out)})
        for geom_name in GRIPPER_GEOMS:
            camera = visibility._camera(env, target, distance=min(max(radius * 2.6, 0.18), 1.0), azimuth=135.0, elevation=-14.0)
            frame = _render_panel(env, renderer, camera, geom_name, {geom_name}, args.output_dir)
            path_out = args.output_dir / f"part_{geom_name}.png"
            cv2.imwrite(str(path_out), frame)
            frames.append(path_out)
            rows.append({"panel": geom_name, "view": "front", "path": str(path_out)})
        sheet = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(frame) for frame in frames],
            "parts": _part_inventory(env),
            "panels": rows,
        }
        report_path = args.output_dir / "gripper_part_audit_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _render_panel(env: nexus.MyCobotNexusEnv, renderer: Any, camera: Any, label: str, visible_geoms: set[str], output_dir: Path) -> np.ndarray:
    env.model.geom_rgba[:, 3] = 0.0
    env.model.site_rgba[:, 3] = 0.0
    for geom_name in visible_geoms:
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            env.model.geom_rgba[geom_id, :] = GEOM_COLORS.get(geom_name, (0.85, 0.85, 0.85, 1.0))
    renderer.update_scene(env.data, camera=camera)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 82), (255, 255, 255), -1)
    cv2.putText(bgr, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(bgr, "280 adaptive gripper part audit: mesh geoms and fingertip pads isolated", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    return bgr



def _draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("gripper body/base", "gripper_base_visual_0"),
        ("left distal finger", "gripper_left1_visual_0"),
        ("left support/prox links", "gripper_left2_visual_0"),
        ("right distal finger", "gripper_right1_visual_0"),
        ("right support/prox links", "gripper_right2_visual_0"),
        ("left contact pad", "left_finger_pad"),
        ("right contact pad", "right_finger_pad"),
    ]
    x0 = 18
    y0 = 92
    width = 330
    height = 22 * len(entries) + 14
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + width, y0 + height), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + width, y0 + height), (60, 60, 60), 1)
    for index, (label, geom_name) in enumerate(entries):
        y = y0 + index * 22
        color = tuple(int(255 * channel) for channel in GEOM_COLORS.get(geom_name, (1, 1, 1, 1))[:3])
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), bgr, -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)

def _gripper_target_and_radius(env: nexus.MyCobotNexusEnv) -> tuple[np.ndarray, float]:
    points = []
    for name in BODY_NAMES:
        body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            points.append(np.asarray(env.data.xpos[body_id], dtype=float))
    arr = np.vstack(points)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    return (mins + maxs) * 0.5, float(np.linalg.norm(maxs - mins) * 0.5)


def _part_inventory(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
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
                "type": int(env.model.geom_type[geom_id]),
                "world_pos": [float(value) for value in env.data.geom_xpos[geom_id]],
                "size": [float(value) for value in env.model.geom_size[geom_id]],
            }
        )
    return rows


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for frame in frames:
        img = cv2.imread(str(frame))
        tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * 360, cols * 480, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 360
        x = (index % cols) * 480
        sheet[y:y + 360, x:x + 480] = tile
    path_out = output_dir / "gripper_part_audit_sheet.png"
    cv2.imwrite(str(path_out), sheet)
    return path_out


if __name__ == "__main__":
    main()
