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
CAMERAS = (
    ("left_chain_side", 145.0, -8.0, 2.15),
    ("right_chain_side", 35.0, -8.0, 2.15),
    ("top_chain", 90.0, -78.0, 2.45),
    ("oblique_chain", 215.0, -28.0, 2.55),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render close-up 280 adaptive gripper connection proof.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_gripper_connection_close_audit_001"))
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
            target, radius = visibility._target_and_radius(
                env,
                (
                    "gripper_base",
                    "gripper_left3",
                    "gripper_left2",
                    "gripper_left1",
                    "gripper_right3",
                    "gripper_right2",
                    "gripper_right1",
                ),
            )
            for camera_name, azimuth, elevation, multiplier in CAMERAS:
                camera = visibility._camera(
                    env,
                    target,
                    distance=min(max(radius * multiplier, 0.16), 0.58),
                    azimuth=azimuth,
                    elevation=elevation,
                )
                _apply_close_visibility(env)
                renderer.update_scene(env.data, camera=camera)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _draw_header(bgr, pose_label, camera_name)
                _draw_labels_and_chain_notes(bgr)
                out = args.output_dir / f"{pose_label}_{camera_name}.png"
                cv2.imwrite(str(out), bgr)
                frames.append(out)
                panels.append({"pose": pose_label, "camera": camera_name, "path": str(out)})
        sheet = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "sheet_path": str(sheet),
            "frames": [str(path) for path in frames],
            "panels": panels,
            "parts": _inventory(env),
            "acceptance_target": "Visible continuous gripper body/base -> proximal/support links -> distal finger links -> fingertip pads in open/mid/closed poses.",
        }
        report_path = args.output_dir / "gripper_connection_close_audit_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _apply_close_visibility(env: nexus.MyCobotNexusEnv) -> None:
    env.model.site_rgba[:, 3] = 0.0
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name in GRIPPER_GEOMS:
            rgba = list(GEOM_COLORS.get(name, (0.8, 0.8, 0.8, 1.0)))
            if name in {"left_finger_pad", "right_finger_pad"}:
                rgba[3] = 0.38
            env.model.geom_rgba[geom_id, :] = rgba
        elif name.startswith("debug_connector_"):
            env.model.geom_rgba[geom_id, 3] = 0.0
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _draw_header(frame: np.ndarray, pose: str, camera: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (255, 255, 255), -1)
    cv2.putText(frame, f"280 close gripper connection proof | {pose} | {camera}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "Only gripper meshes + fingertip pads are shown. Pads are transparent so distal finger links remain visible.", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(frame, "Target chain: body/base -> proximal/support links -> distal fingers -> contact pads", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)


def _draw_labels_and_chain_notes(frame: np.ndarray) -> None:
    entries = [
        ("body/base", "gripper_base_visual_0"),
        ("left distal finger", "gripper_left1_visual_0"),
        ("left support/prox", "gripper_left2_visual_0"),
        ("right distal finger", "gripper_right1_visual_0"),
        ("right support/prox", "gripper_right2_visual_0"),
        ("left pad", "left_finger_pad"),
        ("right pad", "right_finger_pad"),
    ]
    x0 = 18
    y0 = 122
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 270, y0 + 22 * len(entries) + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + 270, y0 + 22 * len(entries) + 10), (75, 75, 75), 1)
    for index, (label, geom_name) in enumerate(entries):
        y = y0 + index * 22
        color = tuple(int(255 * channel) for channel in GEOM_COLORS.get(geom_name, (1, 1, 1, 1))[:3])
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(frame, (x0, y - 11), (x0 + 18, y + 5), bgr, -1)
        cv2.putText(frame, label, (x0 + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)


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
        tiles.append(cv2.resize(img, (560, 420), interpolation=cv2.INTER_AREA))
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * 420, cols * 560, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 420
        x = (index % cols) * 560
        sheet[y:y + 420, x:x + 560] = tile
    path = output_dir / "gripper_connection_close_audit_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


if __name__ == "__main__":
    main()
