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

BASE_GEOM = "g_base_visual_0"
PILLAR_GEOM = "joint1_visual_0"
BASE_BODIES = ("g_base", "joint1", "joint2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render 280 base-to-body connection audit with base highlighted red.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_base_connection_audit_001"))
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
        target, radius = _target_radius(env, BASE_BODIES)
        specs = (
            ("base_front", 210.0, -20.0, 3.4),
            ("base_side", 135.0, -18.0, 3.4),
            ("base_top", 90.0, -70.0, 3.7),
            ("base_low", 225.0, 4.0, 3.2),
        )
        frames: list[Path] = []
        panels: list[dict[str, Any]] = []
        for view, azimuth, elevation, multiplier in specs:
            camera = visibility._camera(env, target, distance=min(max(radius * multiplier, 0.18), 0.8), azimuth=azimuth, elevation=elevation)
            _apply_visibility(env)
            renderer.update_scene(env.data, camera=camera)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            _draw_header(bgr, view)
            _draw_legend(bgr)
            out = args.output_dir / f"{view}.png"
            cv2.imwrite(str(out), bgr)
            frames.append(out)
            panels.append({"view": view, "path": str(out)})
        sheet = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(p) for p in frames],
            "panels": panels,
            "base_geom": BASE_GEOM,
            "pillar_geom": PILLAR_GEOM,
            "note": "The red base mesh is g_base_visual_0. The cyan pillar/body mesh is joint1_visual_0. Orange rods are hidden kinematic connectors/body-frame offsets, not physical manufacturer meshes.",
            "inventory": _inventory(env),
        }
        (args.output_dir / "base_connection_audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _apply_visibility(env: nexus.MyCobotNexusEnv) -> None:
    env.model.site_rgba[:, 3] = 0.0
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(env.model.geom_type[geom_id])
        mesh_id = int(env.model.geom_dataid[geom_id])
        is_mesh = geom_type == int(env._mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0
        if name == BASE_GEOM:
            env.model.geom_rgba[geom_id, :] = [1.0, 0.04, 0.02, 0.98]
        elif name == PILLAR_GEOM:
            env.model.geom_rgba[geom_id, :] = [0.0, 0.82, 1.0, 0.96]
        elif name.startswith("debug_connector_"):
            env.model.geom_rgba[geom_id, :] = [1.0, 0.25, 0.0, 0.78]
        elif is_mesh:
            env.model.geom_rgba[geom_id, :] = [0.86, 0.82, 0.64, 0.20]
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _target_radius(env: nexus.MyCobotNexusEnv, body_names: tuple[str, ...]) -> tuple[np.ndarray, float]:
    points = []
    for body in body_names:
        body_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id >= 0:
            points.append(np.asarray(env.data.xpos[body_id], dtype=float))
    for geom in (BASE_GEOM, PILLAR_GEOM):
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom)
        if geom_id >= 0:
            points.append(np.asarray(env.data.geom_xpos[geom_id], dtype=float))
    arr = np.vstack(points)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    return (mins + maxs) * 0.5, max(float(np.linalg.norm(maxs - mins) * 0.5), 0.08)


def _draw_header(frame: np.ndarray, view: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 116), (255, 255, 255), -1)
    cv2.putText(frame, f"280 base connection audit | {view}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "red = g_base_visual_0 / robot base; cyan = joint1_visual_0 / upright body", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(frame, "orange rods show kinematic body-frame connections; visual mesh gaps are not separate physics bodies", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray) -> None:
    entries = (("robot base g_base", (1.0, 0.04, 0.02)), ("upright joint1 body", (0.0, 0.82, 1.0)), ("kinematic connector", (1.0, 0.25, 0.0)), ("other mesh faded", (0.86, 0.82, 0.64)))
    x0, y0 = 18, 146
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 260, y0 + 22 * len(entries) + 10), (75, 75, 75), 1)
    for index, (label, rgb) in enumerate(entries):
        y = y0 + index * 22
        color = tuple(int(255 * channel) for channel in rgb)
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), (color[2], color[1], color[0]), -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)


def _inventory(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    rows = []
    for geom in (BASE_GEOM, PILLAR_GEOM):
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom)
        if geom_id < 0:
            rows.append({"geom": geom, "present": False})
            continue
        body_id = int(env.model.geom_bodyid[geom_id])
        rows.append({"geom": geom, "present": True, "parent_body": env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_BODY, body_id), "world_pos": [float(v) for v in env.data.geom_xpos[geom_id]]})
    return rows


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for frame in frames:
        img = cv2.imread(str(frame))
        tiles.append(cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))
    cols = 2
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * 480, cols * 640, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 480
        x = (index % cols) * 640
        sheet[y:y + 480, x:x + 640] = tile
    path = output_dir / "base_connection_audit_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


if __name__ == "__main__":
    main()
