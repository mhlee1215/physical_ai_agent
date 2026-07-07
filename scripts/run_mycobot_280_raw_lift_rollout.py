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

START_COMMAND = 1.0
DEFAULT_CONTACT_COMMAND = 0.7
DEFAULT_ZERO_GRAVITY_CLOSE_STEPS = 100
DEFAULT_GRAVITY_HOLD_STEPS = 60
DEFAULT_LIFT_STEPS = 140
WORLD_GRAVITY = (0.0, 0.0, -9.81)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a raw-contact-only 280 close/hold/lift rollout.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/rollouts/mycobot_280_raw_lift_rollout_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--contact-command", type=float, default=DEFAULT_CONTACT_COMMAND)
    parser.add_argument("--zero-gravity-close-steps", type=int, default=DEFAULT_ZERO_GRAVITY_CLOSE_STEPS)
    parser.add_argument("--gravity-hold-steps", type=int, default=DEFAULT_GRAVITY_HOLD_STEPS)
    parser.add_argument("--lift-steps", type=int, default=DEFAULT_LIFT_STEPS)
    parser.add_argument("--lift-scale", type=float, default=1.0, help="Scale the default Gate 7 -> Gate 8 lift delta.")
    parser.add_argument("--actuated-lift", action="store_true", help="Use env.step arm controls during gravity-on lift instead of direct qpos teleporting.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cube-offset-x", type=float, default=0.0)
    parser.add_argument("--cube-offset-y", type=float, default=0.0)
    parser.add_argument("--cube-offset-z", type=float, default=0.0)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--video-every", type=int, default=0, help="Write one video frame every N simulation steps; 0 disables video.")
    parser.add_argument("--video-fps", type=float, default=30.0)
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
        env.reset(seed=args.seed)
        gate7_arm_qpos = nexus._adaptive_gate7_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        default_gate8_lift_arm_qpos = nexus._adaptive_gate8_lift_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        gate8_lift_arm_qpos = tuple(
            float(start + args.lift_scale * (end - start))
            for start, end in zip(gate7_arm_qpos, default_gate8_lift_arm_qpos, strict=True)
        )
        _size_audit_cube(env, half_size=AUDIT_CUBE_HALF_SIZE)
        env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
        env._set_gripper(command=START_COMMAND)
        env._mujoco.mj_forward(env.model, env.data)
        cube_pos, cube_quat = _open_gap_cube_pose(env)
        cube_pos = cube_pos + np.asarray([args.cube_offset_x, args.cube_offset_y, args.cube_offset_z], dtype=float)
        _set_cube_pose(env, cube_pos, cube_quat)
        env._mujoco.mj_forward(env.model, env.data)

        records: list[dict[str, Any]] = []
        frame_paths: list[Path] = []
        video_writer = _open_video_writer(args.video_path, width=args.width, height=args.height, fps=args.video_fps)
        wrote_video_steps: set[int] = set()
        total_steps = args.zero_gravity_close_steps + args.gravity_hold_steps + args.lift_steps
        lift_start = args.zero_gravity_close_steps + args.gravity_hold_steps
        frame_steps = {
            0,
            max(0, args.zero_gravity_close_steps // 2),
            max(0, args.zero_gravity_close_steps - 1),
            args.zero_gravity_close_steps,
            args.zero_gravity_close_steps + max(0, args.gravity_hold_steps // 2),
            max(0, lift_start - 1),
            lift_start + max(0, args.lift_steps // 4),
            lift_start + max(0, args.lift_steps // 2),
            max(0, total_steps - 1),
        }
        for step in range(total_steps):
            if step < args.zero_gravity_close_steps:
                phase = "close_zero_g"
                alpha = step / max(args.zero_gravity_close_steps - 1, 1)
                command = START_COMMAND + alpha * (args.contact_command - START_COMMAND)
                arm = gate7_arm_qpos
                env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
            elif step < args.zero_gravity_close_steps + args.gravity_hold_steps:
                phase = "gravity_hold"
                command = args.contact_command
                arm = gate7_arm_qpos
                env.model.opt.gravity[:] = WORLD_GRAVITY
            else:
                phase = "lift_gravity_on"
                command = args.contact_command
                lift_step = step - args.zero_gravity_close_steps - args.gravity_hold_steps
                alpha = nexus._smoothstep(lift_step / max(args.lift_steps - 1, 1))
                arm = tuple(nexus._lerp_vector(list(gate7_arm_qpos), list(gate8_lift_arm_qpos), alpha))
                env.model.opt.gravity[:] = WORLD_GRAVITY
            if args.actuated_lift and phase == "lift_gravity_on":
                env.step([*arm, command])
            else:
                nexus._set_adaptive_gate_arm_pose(env, tuple(arm))
                env._set_gripper(command=command)
                env._mujoco.mj_step(env.model, env.data)
                nexus._set_adaptive_gate_arm_pose(env, tuple(arm))
                env._set_gripper(command=command)
                env._mujoco.mj_forward(env.model, env.data)
            record = _rollout_record(env, step=step, phase=phase, command=command, initial_cube_pos=cube_pos)
            records.append(record)
            if args.video_every > 0 and video_writer is not None and step % args.video_every == 0:
                video_writer.write(_render_bgr(env, renderer, record))
                wrote_video_steps.add(step)
            if step in frame_steps:
                frame_paths.append(_render_frame(env, renderer, args.output_dir, record))

        if video_writer is not None:
            final_step = total_steps - 1
            if records and final_step not in wrote_video_steps:
                video_writer.write(_render_bgr(env, renderer, records[-1]))
            video_writer.release()

        trace_path = args.output_dir / "mycobot_280_raw_lift_rollout_trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True) + "\n")
        sheet_path = _write_sheet(args.output_dir, frame_paths, cols=3, tile_size=(640, 480), name="raw_lift_rollout_sheet.png")
        close_records = [record for record in records if record["phase"] == "close_zero_g"]
        hold_records = [record for record in records if record["phase"] == "gravity_hold"]
        lift_records = [record for record in records if record["phase"] == "lift_gravity_on"]
        final = records[-1]
        status = (
            "passed"
            if _best_sustained_two_pad(lift_records) >= 60
            and final["pad_cube_contacted_pads"] >= 2
            and final["cube_lift_m"] >= 0.025
            and final["cube_displacement_m"] < 0.20
            else "failed"
        )
        report = {
            "status": status,
            "rollout": "280 raw contact-only close-hold-lift",
            "teacher_attachment_enabled": False,
            "cube_half_size_m": AUDIT_CUBE_HALF_SIZE,
            "start_command": START_COMMAND,
            "contact_command": args.contact_command,
            "zero_gravity_close_steps": args.zero_gravity_close_steps,
            "gravity_hold_steps": args.gravity_hold_steps,
            "lift_steps": args.lift_steps,
            "lift_scale": args.lift_scale,
            "actuated_lift": bool(args.actuated_lift),
            "seed": args.seed,
            "cube_offset_m": [args.cube_offset_x, args.cube_offset_y, args.cube_offset_z],
            "video_path": None if args.video_path is None else str(args.video_path),
            "gravity_note": "Only the close phase is zero-g to establish raw pad contact before gravity-on hold/lift; no teacher attachment is used.",
            "close_best_sustained_two_pad_steps": _best_sustained_two_pad(close_records),
            "hold_best_sustained_two_pad_steps": _best_sustained_two_pad(hold_records),
            "lift_best_sustained_two_pad_steps": _best_sustained_two_pad(lift_records),
            "lift_two_pad_steps": sum(1 for record in lift_records if int(record["pad_cube_contacted_pads"]) >= 2),
            "first_lift_contact_loss_step": _first_contact_loss(lift_records),
            "final_cube_lift_m": final["cube_lift_m"],
            "final_cube_displacement_m": final["cube_displacement_m"],
            "final_pad_cube_contacted_pads": final["pad_cube_contacted_pads"],
            "initial_z_alignment": records[0]["z_alignment"],
            "pre_lift_z_alignment": records[args.zero_gravity_close_steps + args.gravity_hold_steps - 1]["z_alignment"],
            "final_z_alignment": final["z_alignment"],
            "trace_path": str(trace_path),
            "sheet_path": str(sheet_path),
            "frame_paths": [str(path) for path in frame_paths],
            "scene_path": str(env.scene_path),
        }
        report_path = args.output_dir / "mycobot_280_raw_lift_rollout_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if status == "passed" else 1)
    finally:
        if "video_writer" in locals() and video_writer is not None:
            video_writer.release()
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
    contacts = _pad_cube_contacts(env)
    contacted_pads = sorted({contact["pad"] for contact in contacts})
    normal_force_sum = float(sum(abs(contact["normal_force"]) for contact in contacts))
    return {
        "step": int(step),
        "phase": phase,
        "command": float(command),
        "pad_cube_contacts": int(metrics["pad_cube_contacts"]),
        "pad_cube_contacted_pads": len(contacted_pads),
        "contacted_pads": contacted_pads,
        "pad_cube_normal_force_sum": normal_force_sum,
        "jaw_gap_mm": float(metrics["jaw_gap_mm"]),
        "surface_clearance_mm": float(metrics["surface_clearance_mm"]),
        "regime": str(metrics["regime"]),
        "cube_pos": [float(value) for value in cube_pos],
        "cube_lift_m": float(cube_pos[2] - initial_cube_pos[2]),
        "cube_displacement_m": float(np.linalg.norm(cube_pos - initial_cube_pos)),
        "z_alignment": _z_alignment(env),
    }


def _z_alignment(env: nexus.MyCobotNexusEnv) -> dict[str, float]:
    left = _geom_pos(env, ROBOT_LEFT_PAD)
    right = _geom_pos(env, ROBOT_RIGHT_PAD)
    cube = np.asarray(env._cube_position(), dtype=float)
    pad_mid_z = float((left[2] + right[2]) * 0.5)
    cube_center_z = float(cube[2])
    return {
        "left_pad_z": float(left[2]),
        "right_pad_z": float(right[2]),
        "pad_mid_z": pad_mid_z,
        "cube_center_z": cube_center_z,
        "cube_top_z": float(cube_center_z + AUDIT_CUBE_HALF_SIZE),
        "cube_bottom_z": float(cube_center_z - AUDIT_CUBE_HALF_SIZE),
        "pad_mid_minus_cube_center_m": float(pad_mid_z - cube_center_z),
        "pad_mid_inside_cube_z_span": float(cube_center_z - AUDIT_CUBE_HALF_SIZE) <= pad_mid_z <= float(cube_center_z + AUDIT_CUBE_HALF_SIZE),
    }


def _pad_cube_contacts(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    mujoco = env._mujoco
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    pad_ids = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_LEFT_PAD): "left_finger_pad",
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_RIGHT_PAD): "right_finger_pad",
    }
    contacts: list[dict[str, Any]] = []
    for index in range(int(env.data.ncon)):
        contact = env.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id not in pair:
            continue
        pad_geom_ids = [geom_id for geom_id in pair if geom_id in pad_ids]
        if not pad_geom_ids:
            continue
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(env.model, env.data, index, force)
        contacts.append(
            {
                "pad": pad_ids[pad_geom_ids[0]],
                "normal_force": float(force[0]),
                "distance": float(contact.dist),
            }
        )
    return contacts


def _best_sustained_two_pad(records: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    for record in records:
        if int(record["pad_cube_contacted_pads"]) >= 2:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _first_contact_loss(records: list[dict[str, Any]]) -> int | None:
    seen_contact = False
    for record in records:
        if int(record["pad_cube_contacted_pads"]) >= 2:
            seen_contact = True
        elif seen_contact:
            return int(record["step"])
    return None


def _open_video_writer(path: Path | None, *, width: int, height: int, fps: float) -> cv2.VideoWriter | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def _render_bgr(env: nexus.MyCobotNexusEnv, renderer: Any, record: dict[str, Any]) -> np.ndarray:
    full_target, full_radius = visibility._target_and_radius(env, ALL_BODY_NAMES)
    pad_target, pad_radius = _pad_target_and_radius(env)
    if record["phase"] in {"gravity_hold", "lift_gravity_on"}:
        target = pad_target
        radius = max(pad_radius, 0.085)
    else:
        target = full_target
        radius = full_radius
    camera = visibility._camera(
        env,
        target,
        distance=min(max(radius * 2.75, 0.18), 1.15),
        azimuth=215.0,
        elevation=-24.0,
    )
    _apply_visibility(env)
    renderer.update_scene(env.data, camera=camera)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    _draw_rollout_header(bgr, record)
    _draw_legend(bgr)
    return bgr


def _render_frame(env: nexus.MyCobotNexusEnv, renderer: Any, output_dir: Path, record: dict[str, Any]) -> Path:
    bgr = _render_bgr(env, renderer, record)
    out = output_dir / f"step_{int(record['step']):03d}_{record['phase']}.png"
    cv2.imwrite(str(out), bgr)
    return out


def _draw_rollout_header(frame: np.ndarray, record: dict[str, Any]) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 136), (255, 255, 255), -1)
    cv2.putText(frame, f"280 raw contact lift | {record['phase']} | step={record['step']}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, "teacher attachment off; close is zero-g setup; hold/lift are gravity-on raw contact", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    color = (0, 120, 0) if int(record["pad_cube_contacted_pads"]) >= 2 else (0, 0, 190)
    cv2.putText(frame, f"cmd={record['command']:+.3f}; pads={record['pad_cube_contacted_pads']}; contacts={record['pad_cube_contacts']}; force={record['pad_cube_normal_force_sum']:.3f}; lift={record['cube_lift_m']*1000.0:.2f}mm", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    z = record["z_alignment"]
    cv2.putText(frame, f"z: pad_mid-cube_center={z['pad_mid_minus_cube_center_m']*1000.0:.2f}mm; pad_mid_inside_cube={z['pad_mid_inside_cube_z_span']}", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _geom_pos(env: nexus.MyCobotNexusEnv, name: str) -> np.ndarray:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {name}")
    return np.asarray(env.data.geom_xpos[geom_id], dtype=float)


if __name__ == "__main__":
    main()
