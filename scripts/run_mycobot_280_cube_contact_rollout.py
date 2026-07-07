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
from scripts.render_mycobot_280_cube_contact_sequence import (  # noqa: E402
    ALL_BODY_NAMES,
    AUDIT_CUBE_HALF_SIZE,
    ROBOT_LEFT_PAD,
    ROBOT_RIGHT_PAD,
    _apply_visibility,
    _contact_metrics,
    _draw_legend,
    _pad_target_and_radius,
    _quat_align_x_to_vector,
    _set_cube_pose,
    _size_audit_cube,
    _write_sheet,
)

ROLL_OUT_STEPS = 100
HOLD_STEPS = 40
START_COMMAND = 1.0
CONTACT_COMMAND = 0.7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real MuJoCo 280 pad/cube contact rollout.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/rollouts/mycobot_280_cube_contact_rollout_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
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
        env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        gate7_arm_qpos = nexus._adaptive_gate7_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
        _size_audit_cube(env, half_size=AUDIT_CUBE_HALF_SIZE)
        env._set_gripper(command=START_COMMAND)
        env._mujoco.mj_forward(env.model, env.data)
        cube_pos, cube_quat = _open_gap_cube_pose(env)
        _set_cube_pose(env, cube_pos, cube_quat)
        env._mujoco.mj_forward(env.model, env.data)

        records: list[dict[str, Any]] = []
        frame_steps = {0, 25, 50, 75, 99, 120, 139}
        frame_paths: list[Path] = []
        total_steps = ROLL_OUT_STEPS + HOLD_STEPS
        for step in range(total_steps):
            if step < ROLL_OUT_STEPS:
                alpha = step / max(ROLL_OUT_STEPS - 1, 1)
                command = START_COMMAND + alpha * (CONTACT_COMMAND - START_COMMAND)
                phase = "close"
            else:
                command = CONTACT_COMMAND
                phase = "hold"
            nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            env._set_gripper(command=command)
            env._mujoco.mj_step(env.model, env.data)
            nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            env._set_gripper(command=command)
            env._mujoco.mj_forward(env.model, env.data)
            record = _rollout_record(env, step=step, phase=phase, command=command, initial_cube_pos=cube_pos)
            records.append(record)
            if step in frame_steps:
                frame_paths.append(_render_frame(env, renderer, args.output_dir, record))

        trace_path = args.output_dir / "mycobot_280_cube_contact_rollout_trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True) + "\n")
        sheet_path = _write_sheet(args.output_dir, frame_paths, cols=3, tile_size=(640, 480), name="cube_contact_rollout_sheet.png")
        first_contact = next((record for record in records if int(record["pad_cube_contacts"]) > 0), None)
        max_contacts = max(int(record["pad_cube_contacts"]) for record in records)
        max_normal_force = max(float(record["pad_cube_normal_force_sum"]) for record in records)
        final = records[-1]
        status = "passed" if first_contact is not None and max_contacts >= 2 and max_normal_force > 0.0 else "failed"
        report = {
            "status": status,
            "rollout": "280 approach-close-contact",
            "teacher_attachment_enabled": False,
            "gravity": [0.0, 0.0, 0.0],
            "gravity_note": "Disabled for this first contact-isolation rollout so the free cube does not fall before jaw contact.",
            "cube_half_size_m": AUDIT_CUBE_HALF_SIZE,
            "start_command": START_COMMAND,
            "contact_command": CONTACT_COMMAND,
            "steps": len(records),
            "first_contact_step": None if first_contact is None else first_contact["step"],
            "max_pad_cube_contacts": max_contacts,
            "max_pad_cube_normal_force_sum": max_normal_force,
            "final_cube_displacement_m": final["cube_displacement_m"],
            "final_pad_cube_contacts": final["pad_cube_contacts"],
            "trace_path": str(trace_path),
            "sheet_path": str(sheet_path),
            "frame_paths": [str(path) for path in frame_paths],
            "scene_path": str(env.scene_path),
        }
        report_path = args.output_dir / "mycobot_280_cube_contact_rollout_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _open_gap_cube_pose(env: nexus.MyCobotNexusEnv) -> tuple[np.ndarray, list[float]]:
    left = _geom_pos(env, ROBOT_LEFT_PAD)
    right = _geom_pos(env, ROBOT_RIGHT_PAD)
    midpoint = (left + right) * 0.5
    return midpoint, _quat_align_x_to_vector(right - left)


def _rollout_record(
    env: nexus.MyCobotNexusEnv,
    *,
    step: int,
    phase: str,
    command: float,
    initial_cube_pos: np.ndarray,
) -> dict[str, Any]:
    metrics = _contact_metrics(env, cube_half=AUDIT_CUBE_HALF_SIZE)
    cube_pos = np.asarray(env._cube_position(), dtype=float)
    normal_force_sum = 0.0
    for contact_index in _pad_cube_contact_indices(env):
        force = np.zeros(6, dtype=float)
        env._mujoco.mj_contactForce(env.model, env.data, contact_index, force)
        normal_force_sum += abs(float(force[0]))
    return {
        "step": int(step),
        "phase": phase,
        "command": float(command),
        "pad_cube_contacts": int(metrics["pad_cube_contacts"]),
        "pad_cube_normal_force_sum": normal_force_sum,
        "jaw_gap_mm": float(metrics["jaw_gap_mm"]),
        "surface_clearance_mm": float(metrics["surface_clearance_mm"]),
        "regime": str(metrics["regime"]),
        "cube_pos": [float(value) for value in cube_pos],
        "cube_displacement_m": float(np.linalg.norm(cube_pos - initial_cube_pos)),
    }


def _pad_cube_contact_indices(env: nexus.MyCobotNexusEnv) -> list[int]:
    cube_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    pad_ids = {
        env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, ROBOT_LEFT_PAD),
        env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, ROBOT_RIGHT_PAD),
    }
    indices: list[int] = []
    for index in range(int(env.data.ncon)):
        pair = {int(env.data.contact[index].geom1), int(env.data.contact[index].geom2)}
        if cube_id in pair and pair.intersection(pad_ids):
            indices.append(index)
    return indices


def _render_frame(env: nexus.MyCobotNexusEnv, renderer: Any, output_dir: Path, record: dict[str, Any]) -> Path:
    target, radius = visibility._target_and_radius(env, ALL_BODY_NAMES)
    pad_target, pad_radius = _pad_target_and_radius(env)
    if int(record["step"]) >= 50:
        target = pad_target
        radius = max(pad_radius, 0.08)
    camera = visibility._camera(
        env,
        target,
        distance=min(max(radius * 2.65, 0.18), 1.15),
        azimuth=215.0,
        elevation=-24.0,
    )
    _apply_visibility(env)
    renderer.update_scene(env.data, camera=camera)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    _draw_rollout_header(bgr, record)
    _draw_legend(bgr)
    out = output_dir / f"step_{int(record['step']):03d}_{record['phase']}.png"
    cv2.imwrite(str(out), bgr)
    return out


def _draw_rollout_header(frame: np.ndarray, record: dict[str, Any]) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 126), (255, 255, 255), -1)
    cv2.putText(frame, f"280 real MuJoCo contact rollout | {record['phase']} | step={record['step']}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "teacher attachment off; cube and pads are collision-enabled; gravity disabled only for contact isolation", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    color = (0, 120, 0) if int(record["pad_cube_contacts"]) > 0 else (0, 0, 190)
    cv2.putText(frame, f"cmd={record['command']:+.3f}; contacts={record['pad_cube_contacts']}; normal-force-sum={record['pad_cube_normal_force_sum']:.4f}; clearance={record['surface_clearance_mm']:.2f}mm", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"cube displacement={record['cube_displacement_m'] * 1000.0:.2f}mm; regime={record['regime']}", (18, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _geom_pos(env: nexus.MyCobotNexusEnv, name: str) -> np.ndarray:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {name}")
    return np.asarray(env.data.geom_xpos[geom_id], dtype=float)


if __name__ == "__main__":
    main()
