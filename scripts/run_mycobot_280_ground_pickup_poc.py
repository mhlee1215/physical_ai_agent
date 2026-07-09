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
    ROBOT_LEFT_PAD,
    ROBOT_RIGHT_PAD,
    _apply_visibility,
    _contact_metrics,
    _draw_legend,
    _quat_align_x_to_vector,
    _set_cube_pose,
    _size_audit_cube,
    _write_sheet,
)

WORLD_GRAVITY = (0.0, 0.0, -9.81)
WORK_MAT_TOP_Z = 0.008
START_COMMAND = 1.0
CONTACT_COMMAND = 0.994
CUBE_HALF_SIZE = 0.019
CUBE_MASS = 0.032
MAT_FRICTION = 4.0
PAD_FRICTION = 320.0
APPROACH_STEPS = 3
CLOSE_STEPS = 18
HOLD_STEPS = 4
LIFT_STEPS = 120

PICKUP_QPOS = np.asarray(
    [
        1.1090467271542868,
        1.9708220975518413,
        1.0366852271155826,
        -3.402165389648016,
        -0.7968371300189514,
        1.487837875579959,
    ],
    dtype=float,
)
APPROACH_QPOS = np.asarray(
    [
        1.1090467271542868,
        1.9708220975518413,
        1.5166852271155826,
        -2.9821653896480163,
        -0.7968371300189514,
        1.287837875579959,
    ],
    dtype=float,
)
LIFT_QPOS = np.asarray(
    [
        1.1090467271542868,
        2.0630720975518413,
        0.9444352271155826,
        -3.279165389648016,
        -0.7968371300189514,
        1.426337875579959,
    ],
    dtype=float,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 280 Pi adaptive gripper raw-contact cube-from-mat pickup POC.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/rollouts/mycobot_280_ground_pickup_poc_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--video-every", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.video_path or (args.output_dir / "mycobot_280_ground_pickup_poc.mp4")
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
    writer = None
    try:
        renderer = env._mujoco.Renderer(env.model, height=args.height, width=args.width)
        env.reset(seed=args.seed)
        env._diagnostic_cube_half_size = CUBE_HALF_SIZE
        _size_audit_cube(env, half_size=CUBE_HALF_SIZE)
        _apply_physics_overrides(env)
        env.model.opt.gravity[:] = WORLD_GRAVITY

        cube_pos, cube_quat = _initial_cube_pose(env)
        env._set_gripper(command=START_COMMAND)
        _set_cube_pose(env, cube_pos, cube_quat)
        env._mujoco.mj_forward(env.model, env.data)
        initial_cube = np.asarray(env._cube_position(), dtype=float)

        writer = _open_video_writer(video_path, args.width, args.height, args.video_fps)
        records: list[dict[str, Any]] = []
        frame_paths: list[Path] = []
        total_steps = APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS + LIFT_STEPS
        key_steps = {0, 1, 2, APPROACH_STEPS + CLOSE_STEPS - 1, APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS - 1, total_steps - 1}
        camera = _camera(env, initial_cube)
        for step in range(total_steps):
            arm, command, phase = _scripted_state(step)
            env.step([*tuple(float(x) for x in arm), float(command)])
            nexus._set_adaptive_gate_arm_pose(env, tuple(float(x) for x in arm))
            env._set_gripper(command=float(command))
            env._mujoco.mj_forward(env.model, env.data)
            record = _record(env, step=step, phase=phase, command=float(command), initial_cube=initial_cube)
            records.append(record)
            if args.video_every > 0 and step % args.video_every == 0:
                writer.write(_render_bgr(env, renderer, record, camera))
            if step in key_steps:
                frame_paths.append(_write_frame(env, renderer, args.output_dir, record, camera))

        writer.release()
        writer = None
        trace_path = args.output_dir / "ground_pickup_trace.jsonl"
        trace_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
        sheet_path = _write_sheet(args.output_dir, frame_paths, cols=3, tile_size=(640, 480), name="ground_pickup_sheet.png")
        lift_records = [r for r in records if r["phase"] == "lift_from_mat"]
        final = records[-1]
        status = "passed" if _passes(records, lift_records, final) else "failed"
        report = {
            "status": status,
            "scenario": "280 raw-contact cube-from-mat approach-close-lift POC",
            "teacher_attachment_enabled": False,
            "object_teleport_during_pickup_lift": False,
            "cube_starts_on_work_mat": True,
            "work_mat_top_z": WORK_MAT_TOP_Z,
            "cube_half_size_m": CUBE_HALF_SIZE,
            "cube_mass_kg": CUBE_MASS,
            "mat_friction": MAT_FRICTION,
            "pad_friction": PAD_FRICTION,
            "start_command": START_COMMAND,
            "contact_command": CONTACT_COMMAND,
            "approach_steps": APPROACH_STEPS,
            "close_steps": CLOSE_STEPS,
            "hold_steps": HOLD_STEPS,
            "lift_steps": LIFT_STEPS,
            "pickup_qpos": [float(x) for x in PICKUP_QPOS],
            "approach_qpos": [float(x) for x in APPROACH_QPOS],
            "final_lift_qpos": [float(x) for x in LIFT_QPOS],
            "first_frame_pad_cube_contacted_pads": records[0]["pad_cube_contacted_pads"],
            "first_contact_step": next((r["step"] for r in records if r["pad_cube_contacted_pads"] > 0), None),
            "final_cube_lift_m": final["cube_lift_m"],
            "final_pad_cube_contacted_pads": final["pad_cube_contacted_pads"],
            "lift_best_sustained_two_pad_steps": _best_sustained_two_pad(lift_records),
            "first_lift_contact_loss_step": _first_contact_loss(lift_records),
            "trace_path": str(trace_path),
            "video_path": str(video_path),
            "sheet_path": str(sheet_path),
            "frame_paths": [str(p) for p in frame_paths],
            "scene_path": str(env.scene_path),
        }
        report_path = args.output_dir / "ground_pickup_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if status == "passed" else 1)
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        env.close()


def _initial_cube_pose(env: nexus.MyCobotNexusEnv) -> tuple[np.ndarray, list[float]]:
    nexus._set_adaptive_gate_arm_pose(env, tuple(float(x) for x in PICKUP_QPOS))
    env._set_gripper(command=0.74)
    env._mujoco.mj_forward(env.model, env.data)
    left = _geom_pos(env, ROBOT_LEFT_PAD)
    right = _geom_pos(env, ROBOT_RIGHT_PAD)
    axis = right - left
    unit_axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    side_axis = np.cross(np.asarray([0.0, 0.0, 1.0]), unit_axis)
    side_axis = side_axis / max(float(np.linalg.norm(side_axis)), 1e-9)
    pos = (left + right) * 0.5
    pos = np.asarray([pos[0], pos[1], WORK_MAT_TOP_Z + CUBE_HALF_SIZE], dtype=float)
    pos = pos + 0.0160 * unit_axis + 0.0110 * side_axis
    return pos, _quat_align_x_to_vector(axis)


def _scripted_state(step: int) -> tuple[np.ndarray, float, str]:
    if step < APPROACH_STEPS:
        alpha = nexus._smoothstep(step / max(APPROACH_STEPS - 1, 1))
        return APPROACH_QPOS + (PICKUP_QPOS - APPROACH_QPOS) * alpha, START_COMMAND, "approach_down_to_cube_on_mat"
    if step < APPROACH_STEPS + CLOSE_STEPS:
        alpha = (step - APPROACH_STEPS) / max(CLOSE_STEPS - 1, 1)
        return PICKUP_QPOS.copy(), START_COMMAND + alpha * (CONTACT_COMMAND - START_COMMAND), "close_on_cube_on_mat"
    if step < APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS:
        return PICKUP_QPOS.copy(), CONTACT_COMMAND, "hold_before_lift"
    alpha = nexus._smoothstep((step - APPROACH_STEPS - CLOSE_STEPS - HOLD_STEPS) / max(LIFT_STEPS - 1, 1))
    return PICKUP_QPOS + (LIFT_QPOS - PICKUP_QPOS) * alpha, CONTACT_COMMAND, "lift_from_mat"


def _passes(records: list[dict[str, Any]], lift_records: list[dict[str, Any]], final: dict[str, Any]) -> bool:
    return (
        records[0]["pad_cube_contacted_pads"] == 0
        and final["cube_lift_m"] >= 0.025
        and final["pad_cube_contacted_pads"] >= 2
        and _best_sustained_two_pad(lift_records) >= 60
    )


def _apply_physics_overrides(env: nexus.MyCobotNexusEnv) -> None:
    mujoco = env._mujoco
    cube_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, nexus.TASK_CUBE_BODY)
    cube_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if cube_body >= 0:
        env.model.body_mass[cube_body] = CUBE_MASS
        inertia = (1.0 / 6.0) * CUBE_MASS * (2.0 * CUBE_HALF_SIZE) ** 2
        env.model.body_inertia[cube_body, :] = [inertia, inertia, inertia]
    if cube_geom >= 0:
        env.model.geom_friction[cube_geom, :3] = [MAT_FRICTION, MAT_FRICTION * 0.1, MAT_FRICTION * 0.1]
        env.model.geom_condim[cube_geom] = 6
    for name in ("nexus_work_mat", "nexus_floor"):
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            env.model.geom_friction[geom_id, :3] = [MAT_FRICTION, MAT_FRICTION * 0.1, MAT_FRICTION * 0.1]
            env.model.geom_condim[geom_id] = 6
    for name in (ROBOT_LEFT_PAD, ROBOT_RIGHT_PAD):
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            env.model.geom_friction[geom_id, :3] = [PAD_FRICTION, PAD_FRICTION * 0.1, PAD_FRICTION * 0.1]


def _record(env: nexus.MyCobotNexusEnv, *, step: int, phase: str, command: float, initial_cube: np.ndarray) -> dict[str, Any]:
    metrics = _contact_metrics(env, cube_half=CUBE_HALF_SIZE)
    cube = np.asarray(env._cube_position(), dtype=float)
    contacted = _pad_contacts(env)
    return {
        "step": int(step),
        "phase": phase,
        "command": float(command),
        "pad_cube_contacts": int(metrics["pad_cube_contacts"]),
        "pad_cube_contacted_pads": len(set(contacted)),
        "contacted_pads": sorted(set(contacted)),
        "jaw_gap_mm": float(metrics["jaw_gap_mm"]),
        "surface_clearance_mm": float(metrics["surface_clearance_mm"]),
        "cube_pos": [float(x) for x in cube],
        "cube_lift_m": float(cube[2] - initial_cube[2]),
    }


def _pad_contacts(env: nexus.MyCobotNexusEnv) -> list[str]:
    mujoco = env._mujoco
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    pads = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_LEFT_PAD): "left",
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_RIGHT_PAD): "right",
    }
    out: list[str] = []
    for index in range(int(env.data.ncon)):
        pair = {int(env.data.contact[index].geom1), int(env.data.contact[index].geom2)}
        if cube_id in pair:
            for pad_id, side in pads.items():
                if pad_id >= 0 and pad_id in pair:
                    out.append(side)
    return out


def _best_sustained_two_pad(records: list[dict[str, Any]]) -> int:
    best = current = 0
    for record in records:
        if int(record["pad_cube_contacted_pads"]) >= 2:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _first_contact_loss(records: list[dict[str, Any]]) -> int | None:
    seen = False
    for record in records:
        if int(record["pad_cube_contacted_pads"]) >= 2:
            seen = True
        elif seen:
            return int(record["step"])
    return None


def _open_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def _camera(env: nexus.MyCobotNexusEnv, initial_cube: np.ndarray) -> Any:
    target = initial_cube + np.asarray([0.0, 0.0, 0.035], dtype=float)
    return visibility._camera(env, target, distance=0.24, azimuth=215.0, elevation=-10.0)


def _render_bgr(env: nexus.MyCobotNexusEnv, renderer: Any, record: dict[str, Any], camera: Any) -> np.ndarray:
    _apply_visibility(env, hide_gripper_base_shell=True)
    cube_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if cube_id >= 0:
        env.model.geom_rgba[cube_id, :] = [1.0, 0.05, 0.0, 1.0]
    mat_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, "nexus_work_mat")
    if mat_id >= 0:
        env.model.geom_rgba[mat_id, :] = [0.25, 0.45, 0.42, 0.92]
    renderer.update_scene(env.data, camera=camera)
    frame = cv2.cvtColor(renderer.render().astype(np.uint8), cv2.COLOR_RGB2BGR)
    _draw_header(frame, record)
    _draw_legend(frame)
    return frame


def _write_frame(env: nexus.MyCobotNexusEnv, renderer: Any, output_dir: Path, record: dict[str, Any], camera: Any) -> Path:
    path = output_dir / f"ground_pickup_step_{record['step']:03d}_{record['phase']}.png"
    cv2.imwrite(str(path), _render_bgr(env, renderer, record, camera))
    return path


def _draw_header(frame: np.ndarray, record: dict[str, Any]) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 126), (255, 255, 255), -1)
    cv2.putText(frame, f"280 raw cube-from-mat pickup | {record['phase']} | step={record['step']}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "cube starts on work mat; terminal pad contact only; teacher attachment off", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    color = (0, 120, 0) if int(record["pad_cube_contacted_pads"]) >= 2 else (0, 0, 190)
    cv2.putText(frame, f"cmd={record['command']:+.3f}; pads={record['pad_cube_contacted_pads']}; contacts={record['pad_cube_contacts']}; lift={record['cube_lift_m']*1000.0:.1f}mm", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"mat top={WORK_MAT_TOP_Z*1000:.1f}mm; cube half={CUBE_HALF_SIZE*1000:.1f}mm", (18, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)


def _geom_pos(env: nexus.MyCobotNexusEnv, name: str) -> np.ndarray:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {name}")
    return np.asarray(env.data.geom_xpos[geom_id], dtype=float)


if __name__ == "__main__":
    main()
