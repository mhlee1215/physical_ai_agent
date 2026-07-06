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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render myCobot mesh/body sanity inspection sheets.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mesh_inspection/mycobot_280_mesh_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--model-profile", default=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _install_marker_scene_override()
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
        reports = []
        for label, arm_qpos, gripper_command in poses:
            nexus._set_adaptive_gate_arm_pose(env, arm_qpos)
            env._set_gripper(command=gripper_command)
            env._mujoco.mj_forward(env.model, env.data)
            reports.append(_pose_report(env, label, gripper_command))
            for view_name, camera in _cameras(env):
                rgb = _render(env, renderer, camera)
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _draw_label(bgr, label, view_name, env)
                path = args.output_dir / f"{label}_{view_name}.png"
                cv2.imwrite(str(path), bgr)
                frames.append(path)
        sheet_path = _write_sheet(args.output_dir, frames)
        report = {
            "status": "passed",
            "model_profile": args.model_profile,
            "scene_path": str(env.scene_path),
            "sheet_path": str(sheet_path),
            "frames": [str(path) for path in frames],
            "poses": reports,
            "body_tree": _body_tree(env.scene_path),
        }
        report_path = args.output_dir / "mesh_inspection_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _install_marker_scene_override() -> None:
    original_build = nexus.build_mycobot_nexus_scene_model
    if getattr(original_build, "_mesh_inspection_wrapper", False):
        return

    def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> None:
        original_build(*wrapper_args, **wrapper_kwargs)
        scene_path = wrapper_kwargs.get("scene_path") if "scene_path" in wrapper_kwargs else wrapper_args[1]
        tree = ET.parse(scene_path)
        root = tree.getroot()
        for geom in root.findall(".//geom"):
            name = geom.attrib.get("name", "")
            mesh = geom.attrib.get("mesh", "")
            if name == nexus.TASK_CUBE_GEOM:
                geom.set("rgba", "1 0.08 0.02 1")
                geom.attrib.pop("material", None)
            elif name == "left_finger_pad":
                geom.set("rgba", "0 1 0 0.65")
            elif name == "right_finger_pad":
                geom.set("rgba", "0 0.3 1 0.65")
            elif "gripper" in name or "gripper" in mesh:
                geom.set("rgba", "0.74 0.69 0.60 1")
                geom.attrib.pop("material", None)
        body_markers = {
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
        for body in root.findall(".//body"):
            name = body.attrib.get("name", "")
            if name in body_markers:
                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": f"{name}_origin_marker",
                        "type": "sphere",
                        "pos": "0 0 0",
                        "size": "0.005",
                        "rgba": body_markers[name],
                    },
                )
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    wrapper._mesh_inspection_wrapper = True  # type: ignore[attr-defined]
    nexus.build_mycobot_nexus_scene_model = wrapper


def _cameras(env: nexus.MyCobotNexusEnv) -> list[tuple[str, Any]]:
    cube = np.asarray(env._cube_position(), dtype=float)
    pad = np.asarray(env._finger_pad_midpoint(), dtype=float)
    tcp = np.asarray(env._tcp_position(), dtype=float)
    gripper_target = (pad * 0.55 + tcp * 0.45).tolist()
    gripper_target[2] += 0.015
    full_target = (cube * 0.2 + tcp * 0.8).tolist()
    full_target[2] += 0.05
    return [
        ("full_front", _camera(env, full_target, distance=0.92, azimuth=145, elevation=-25)),
        ("full_side", _camera(env, full_target, distance=0.92, azimuth=70, elevation=-18)),
        ("gripper_front", _camera(env, gripper_target, distance=0.28, azimuth=120, elevation=-18)),
        ("gripper_top", _camera(env, gripper_target, distance=0.34, azimuth=90, elevation=-72)),
    ]


def _camera(env: nexus.MyCobotNexusEnv, lookat: list[float], *, distance: float, azimuth: float, elevation: float) -> Any:
    camera = env._mujoco.MjvCamera()
    camera.type = env._mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    return camera


def _render(env: nexus.MyCobotNexusEnv, renderer: Any, camera: Any) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera)
    return renderer.render()


def _draw_label(frame: np.ndarray, pose_label: str, view_name: str, env: nexus.MyCobotNexusEnv) -> None:
    pad = np.asarray(env._finger_pad_midpoint(), dtype=float)
    tcp = np.asarray(env._tcp_position(), dtype=float)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (255, 255, 255), -1)
    cv2.putText(frame, f"{pose_label} | {view_name}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"tcp=({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) pad_mid=({pad[0]:+.3f},{pad[1]:+.3f},{pad[2]:+.3f})",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )


def _write_sheet(output_dir: Path, frames: list[Path]) -> Path:
    tiles = []
    for path in frames:
        img = cv2.imread(str(path))
        tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
    rows = 4
    cols = 4
    sheet = np.full((rows * 360, cols * 480, 3), 245, np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * 360
        x = (index % cols) * 480
        sheet[y:y + 360, x:x + 480] = tile
    path = output_dir / "mesh_inspection_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def _pose_report(env: nexus.MyCobotNexusEnv, label: str, gripper_command: float) -> dict[str, Any]:
    bodies = {}
    for name in (
        "gripper_base",
        "gripper_left3",
        "gripper_left1",
        "gripper_left2",
        "gripper_right3",
        "gripper_right1",
        "gripper_right2",
    ):
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


def _body_tree(scene_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(scene_path).getroot()
    rows: list[dict[str, Any]] = []

    def walk(body: ET.Element, parent: str | None) -> None:
        name = body.attrib.get("name", "")
        if "gripper" in name:
            rows.append(
                {
                    "name": name,
                    "parent": parent,
                    "pos": body.attrib.get("pos", ""),
                    "joints": [joint.attrib for joint in body.findall("joint")],
                    "geoms": [geom.attrib.get("name", geom.attrib.get("mesh", "")) for geom in body.findall("geom")],
                }
            )
        for child in body.findall("body"):
            walk(child, name or parent)

    for body in root.findall("./worldbody/body"):
        walk(body, None)
    return rows


if __name__ == "__main__":
    main()
