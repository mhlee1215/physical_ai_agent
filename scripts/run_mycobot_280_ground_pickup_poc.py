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
WORK_MAT_CENTER = (-0.12, -0.12, 0.004)
WORK_MAT_HALF_SIZE = (0.46, 0.26, 0.004)
WORK_MAT_TOP_Z = WORK_MAT_CENTER[2] + WORK_MAT_HALF_SIZE[2]
MAT_GUARD_TOLERANCE_M = 0.001
PAD_MAT_GUARD_TOLERANCE_M = 0.001
GRIPPER_VISUAL_MAT_GUARD_TOLERANCE_M = 0.001
MAX_PAD_CUBE_PENETRATION_M = 0.003
START_COMMAND = 1.0
CONTACT_COMMAND = 0.75
CUBE_HALF_SIZE = 0.015
CUBE_MASS = 0.032
MAT_FRICTION = 4.0
PAD_FRICTION = 640.0
PAD_SIZE_OVERRIDE: tuple[float, float, float] | None = None
CUBE_AXIS_OFFSET = 0.0015
CUBE_SIDE_OFFSET = 0.008
APPROACH_STEPS = 20
CLOSE_STEPS = 100
HOLD_STEPS = 30
LIFT_STEPS = 80

PICKUP_QPOS = np.asarray(
    [
        1.6920851246689788,
        2.3802245106776354,
        0.2643304707557536,
        -2.7574344941163096,
        -0.9716523084214069,
        -0.2085629983144582,
    ],
    dtype=float,
)
APPROACH_QPOS = PICKUP_QPOS.copy()
LIFT_QPOS = np.asarray(
    [
        1.6920851246689788,
        2.3002245106776353,
        0.41433047075575357,
        -2.5074344941163096,
        -0.9716523084214069,
        -0.2085629983144582,
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
    parser.add_argument("--approach-steps", type=int, default=APPROACH_STEPS)
    parser.add_argument("--close-steps", type=int, default=CLOSE_STEPS)
    parser.add_argument("--hold-steps", type=int, default=HOLD_STEPS)
    parser.add_argument("--lift-steps", type=int, default=LIFT_STEPS)
    parser.add_argument("--zero-gravity-close", dest="zero_gravity_close", action="store_true", default=True, help="Disable gravity during approach/close only. Gravity is restored for hold/lift.")
    parser.add_argument("--gravity-close", dest="zero_gravity_close", action="store_false", help="Keep normal gravity during approach/close for a stricter exploratory run.")
    parser.add_argument("--post-step-snap", dest="post_step_snap", action="store_true", default=False, help="Legacy debug mode: qpos-snap/forward the arm and gripper after env.step.")
    parser.add_argument("--no-post-step-snap", dest="post_step_snap", action="store_false", help="Preserve MuJoCo contact dynamics after env.step. This is the default.")
    parser.add_argument("--cube-half-size", type=float, default=CUBE_HALF_SIZE)
    parser.add_argument("--cube-mass", type=float, default=CUBE_MASS)
    parser.add_argument("--mat-top-z", type=float, default=None)
    parser.add_argument("--contact-command", type=float, default=CONTACT_COMMAND)
    parser.add_argument("--pad-friction", type=float, default=PAD_FRICTION)
    parser.add_argument("--max-pad-cube-penetration", type=float, default=MAX_PAD_CUBE_PENETRATION_M)
    parser.add_argument("--pad-size", type=str, default=None, help="Experimental contact half-size override as x,y,z meters.")
    parser.add_argument("--cube-axis-offset", type=float, default=CUBE_AXIS_OFFSET)
    parser.add_argument("--cube-side-offset", type=float, default=CUBE_SIDE_OFFSET)
    parser.add_argument("--cube-pos-xy", type=str, default=None, help="Experimental absolute cube x,y placement on the work mat.")
    parser.add_argument("--pickup-qpos", type=str, default=None)
    parser.add_argument("--approach-qpos", type=str, default=None)
    parser.add_argument("--lift-qpos", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    global CUBE_HALF_SIZE, CUBE_MASS, CONTACT_COMMAND, PAD_FRICTION, PAD_SIZE_OVERRIDE, CUBE_AXIS_OFFSET, CUBE_SIDE_OFFSET, PICKUP_QPOS, APPROACH_QPOS, LIFT_QPOS, WORK_MAT_CENTER, WORK_MAT_TOP_Z, APPROACH_STEPS, CLOSE_STEPS, HOLD_STEPS, LIFT_STEPS, MAX_PAD_CUBE_PENETRATION_M
    CUBE_HALF_SIZE = float(args.cube_half_size)
    CUBE_MASS = float(args.cube_mass)
    if args.mat_top_z is not None:
        WORK_MAT_TOP_Z = float(args.mat_top_z)
        WORK_MAT_CENTER = (WORK_MAT_CENTER[0], WORK_MAT_CENTER[1], WORK_MAT_TOP_Z - WORK_MAT_HALF_SIZE[2])
    CONTACT_COMMAND = float(args.contact_command)
    PAD_FRICTION = float(args.pad_friction)
    MAX_PAD_CUBE_PENETRATION_M = float(args.max_pad_cube_penetration)
    PAD_SIZE_OVERRIDE = tuple(float(value.strip()) for value in args.pad_size.split(",")) if args.pad_size else None
    if PAD_SIZE_OVERRIDE is not None and len(PAD_SIZE_OVERRIDE) != 3:
        raise ValueError("--pad-size must have exactly three comma-separated values")
    CUBE_AXIS_OFFSET = float(args.cube_axis_offset)
    CUBE_SIDE_OFFSET = float(args.cube_side_offset)
    APPROACH_STEPS = int(args.approach_steps)
    CLOSE_STEPS = int(args.close_steps)
    HOLD_STEPS = int(args.hold_steps)
    LIFT_STEPS = int(args.lift_steps)
    if args.pickup_qpos:
        PICKUP_QPOS = _parse_qpos(args.pickup_qpos)
    if args.approach_qpos:
        APPROACH_QPOS = _parse_qpos(args.approach_qpos)
    if args.lift_qpos:
        LIFT_QPOS = _parse_qpos(args.lift_qpos)
    _patch_nexus_work_mat_scene_nodes()
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
        if args.cube_pos_xy:
            xy = [float(value.strip()) for value in args.cube_pos_xy.split(",") if value.strip()]
            if len(xy) != 2:
                raise ValueError("--cube-pos-xy must have exactly two comma-separated values")
            cube_pos = np.asarray([xy[0], xy[1], WORK_MAT_TOP_Z + CUBE_HALF_SIZE], dtype=float)
        env._set_gripper(command=START_COMMAND)
        _set_cube_pose(env, cube_pos, cube_quat)
        env._mujoco.mj_forward(env.model, env.data)
        initial_cube = np.asarray(env._cube_position(), dtype=float)
        placement_guard = _cube_mat_guard(initial_cube)
        if not placement_guard["passed"]:
            raise RuntimeError(f"cube does not start fully on the work mat: {placement_guard}")

        writer = _open_video_writer(video_path, args.width, args.height, args.video_fps)
        records: list[dict[str, Any]] = []
        frame_paths: list[Path] = []
        total_steps = APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS + LIFT_STEPS
        key_steps = {0, 1, 2, APPROACH_STEPS + CLOSE_STEPS - 1, APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS - 1, total_steps - 1}
        camera = _camera(env, initial_cube)
        for step in range(total_steps):
            arm, command, phase = _scripted_state(step)
            if args.zero_gravity_close and phase in {"approach_down_to_cube_on_mat", "close_on_cube_on_mat"}:
                env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
            else:
                env.model.opt.gravity[:] = WORLD_GRAVITY
            env.step([*tuple(float(x) for x in arm), float(command)])
            if args.post_step_snap:
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
        completion = _completion_standard(records, lift_records, final, status=status)
        report = {
            "status": status,
            "completion_standard": completion,
            "completion_standard_status": completion["status"],
            "scenario": "280 raw-contact cube-from-mat approach-close-lift POC",
            "modeling_boundary": "Fingertip pads, cube, mat, and floor use MuJoCo contact. Visible gripper geoms are guarded against mat-plane overlap; broader arm/table collision remains visual-only. By default gravity is disabled during approach/close only, then restored for hold/lift.",
            "teacher_attachment_enabled": False,
            "object_teleport_during_pickup_lift": False,
            "zero_gravity_close": bool(args.zero_gravity_close),
            "post_step_snap_enabled": bool(args.post_step_snap),
            "cube_starts_on_work_mat": True,
            "work_mat_top_z": WORK_MAT_TOP_Z,
            "work_mat_center": [float(x) for x in WORK_MAT_CENTER],
            "work_mat_half_size": [float(x) for x in WORK_MAT_HALF_SIZE],
            "initial_cube_mat_guard": placement_guard,
            "cube_half_size_m": CUBE_HALF_SIZE,
            "cube_mass_kg": CUBE_MASS,
            "mat_friction": MAT_FRICTION,
            "pad_friction": PAD_FRICTION,
            "pad_size_override": list(PAD_SIZE_OVERRIDE) if PAD_SIZE_OVERRIDE is not None else None,
            "cube_axis_offset": CUBE_AXIS_OFFSET,
            "cube_side_offset": CUBE_SIDE_OFFSET,
            "cube_pos_xy_override": args.cube_pos_xy,
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
            "cube_bottom_on_or_above_mat_all_steps": all(bool(r["mat_guard"]["bottom_on_or_above_mat"]) for r in records),
            "worst_cube_bottom_minus_mat_top_m": min(float(r["mat_guard"]["cube_bottom_minus_mat_top_m"]) for r in records),
            "pad_mat_guard_passed_all_steps": all(bool(r["pad_mat_guard"]["passed"]) for r in records),
            "worst_pad_mat_penetration_m": min(float(r["pad_mat_guard"]["min_pad_bottom_minus_mat_top_m"]) for r in records),
            "gripper_visual_mat_guard_passed_all_steps": all(bool(r["gripper_visual_mat_guard"]["passed"]) for r in records),
            "worst_gripper_visual_penetration_m": min(float(r["gripper_visual_mat_guard"]["min_gripper_visual_bottom_minus_mat_top_m"]) for r in records),
            "max_pad_cube_penetration_m": max(float(r["pad_cube_contact_depth"]["max_penetration_m"]) for r in records),
            "max_lift_pad_cube_penetration_m": max((float(r["pad_cube_contact_depth"]["max_penetration_m"]) for r in lift_records), default=0.0),
            "mean_pad_cube_penetration_m": float(np.mean([float(r["pad_cube_contact_depth"]["mean_penetration_m"]) for r in records])),
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




def _patch_nexus_work_mat_scene_nodes() -> None:
    original = nexus._nexus_scene_nodes
    if getattr(original, "_ground_pickup_patched", False):
        return

    def patched_nodes() -> list[Any]:
        nodes = original()
        for node in nodes:
            if node.attrib.get("name") == "nexus_work_mat":
                node.attrib["pos"] = _float_sequence(WORK_MAT_CENTER)
                node.attrib["size"] = _float_sequence(WORK_MAT_HALF_SIZE)
            elif node.attrib.get("name") == "nexus_floor":
                floor_z = min(-0.006, WORK_MAT_TOP_Z - 0.12)
                node.attrib["pos"] = f"0 0 {floor_z:.10g}"
        return nodes

    setattr(patched_nodes, "_ground_pickup_patched", True)
    nexus._nexus_scene_nodes = patched_nodes


def _float_sequence(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)

def _parse_qpos(raw: str) -> np.ndarray:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 6:
        raise ValueError(f"expected 6 comma-separated qpos values, got {len(values)}: {raw!r}")
    return np.asarray(values, dtype=float)

def _initial_cube_pose(env: nexus.MyCobotNexusEnv) -> tuple[np.ndarray, list[float]]:
    nexus._set_adaptive_gate_arm_pose(env, tuple(float(x) for x in PICKUP_QPOS))
    env._set_gripper(command=START_COMMAND)
    env._mujoco.mj_forward(env.model, env.data)
    left = _geom_pos(env, ROBOT_LEFT_PAD)
    right = _geom_pos(env, ROBOT_RIGHT_PAD)
    axis = right - left
    unit_axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    side_axis = np.cross(np.asarray([0.0, 0.0, 1.0]), unit_axis)
    side_axis = side_axis / max(float(np.linalg.norm(side_axis)), 1e-9)
    pos = (left + right) * 0.5
    xy_offset = CUBE_AXIS_OFFSET * unit_axis[:2] + CUBE_SIDE_OFFSET * side_axis[:2]
    pos = np.asarray([pos[0] + xy_offset[0], pos[1] + xy_offset[1], WORK_MAT_TOP_Z + CUBE_HALF_SIZE], dtype=float)
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
        bool(records[0]["mat_guard"]["passed"])
        and all(bool(record["mat_guard"]["bottom_on_or_above_mat"]) for record in records)
        and all(bool(record["pad_mat_guard"]["passed"]) for record in records)
        and all(bool(record["gripper_visual_mat_guard"]["passed"]) for record in records)
        and records[0]["pad_cube_contacted_pads"] == 0
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
        env.model.geom_contype[cube_geom] = 1
        env.model.geom_conaffinity[cube_geom] = 3
    mat_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "nexus_work_mat")
    if mat_id >= 0:
        env.model.geom_pos[mat_id, :3] = [float(x) for x in WORK_MAT_CENTER]
        env.model.geom_size[mat_id, :3] = [float(x) for x in WORK_MAT_HALF_SIZE]
    for name in ("nexus_work_mat", "nexus_floor"):
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            env.model.geom_friction[geom_id, :3] = [MAT_FRICTION, MAT_FRICTION * 0.1, MAT_FRICTION * 0.1]
            env.model.geom_condim[geom_id] = 6
            env.model.geom_contype[geom_id] = 1
            env.model.geom_conaffinity[geom_id] = 1
    for name in (ROBOT_LEFT_PAD, ROBOT_RIGHT_PAD):
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            if PAD_SIZE_OVERRIDE is not None:
                env.model.geom_size[geom_id, :3] = [float(x) for x in PAD_SIZE_OVERRIDE]
            env.model.geom_friction[geom_id, :3] = [PAD_FRICTION, PAD_FRICTION * 0.1, PAD_FRICTION * 0.1]
            env.model.geom_contype[geom_id] = 2
            env.model.geom_conaffinity[geom_id] = 2


def _record(env: nexus.MyCobotNexusEnv, *, step: int, phase: str, command: float, initial_cube: np.ndarray) -> dict[str, Any]:
    metrics = _contact_metrics(env, cube_half=CUBE_HALF_SIZE)
    cube = np.asarray(env._cube_position(), dtype=float)
    contacted = _pad_contacts(env)
    pad_cube_depth = _pad_cube_contact_depth(env)
    return {
        "step": int(step),
        "phase": phase,
        "command": float(command),
        "pad_cube_contacts": int(metrics["pad_cube_contacts"]),
        "pad_cube_contacted_pads": len(set(contacted)),
        "pad_cube_contact_depth": pad_cube_depth,
        "contacted_pads": sorted(set(contacted)),
        "jaw_gap_mm": float(metrics["jaw_gap_mm"]),
        "surface_clearance_mm": float(metrics["surface_clearance_mm"]),
        "cube_pos": [float(x) for x in cube],
        "mat_guard": _cube_mat_guard(cube),
        "pad_mat_guard": _pad_mat_guard(env),
        "gripper_visual_mat_guard": _gripper_visual_mat_guard(env),
        "cube_lift_m": float(cube[2] - initial_cube[2]),
    }


def _cube_mat_guard(cube: np.ndarray) -> dict[str, Any]:
    mat_center = np.asarray(WORK_MAT_CENTER, dtype=float)
    mat_half = np.asarray(WORK_MAT_HALF_SIZE, dtype=float)
    cube = np.asarray(cube, dtype=float)
    min_xy = mat_center[:2] - mat_half[:2] + CUBE_HALF_SIZE
    max_xy = mat_center[:2] + mat_half[:2] - CUBE_HALF_SIZE
    bottom = float(cube[2] - CUBE_HALF_SIZE)
    bottom_delta = bottom - WORK_MAT_TOP_Z
    inside_full_footprint = bool(min_xy[0] <= cube[0] <= max_xy[0] and min_xy[1] <= cube[1] <= max_xy[1])
    bottom_on_or_above_mat = bool(bottom_delta >= -MAT_GUARD_TOLERANCE_M)
    bottom_near_mat_top = bool(abs(bottom_delta) <= MAT_GUARD_TOLERANCE_M)
    return {
        "passed": bool(inside_full_footprint and bottom_on_or_above_mat and bottom_near_mat_top),
        "inside_full_footprint": inside_full_footprint,
        "bottom_on_or_above_mat": bottom_on_or_above_mat,
        "bottom_near_mat_top": bottom_near_mat_top,
        "cube_bottom_z": bottom,
        "mat_top_z": WORK_MAT_TOP_Z,
        "cube_bottom_minus_mat_top_m": bottom_delta,
        "footprint_min_xy": [float(x) for x in min_xy],
        "footprint_max_xy": [float(x) for x in max_xy],
    }



def _pad_mat_guard(env: nexus.MyCobotNexusEnv) -> dict[str, Any]:
    checks = []
    for name in (ROBOT_LEFT_PAD, ROBOT_RIGHT_PAD):
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            continue
        center = np.asarray(env.data.geom_xpos[geom_id], dtype=float)
        xmat = np.asarray(env.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        half = np.asarray(env.model.geom_size[geom_id, :3], dtype=float)
        world_half_extents = np.abs(xmat) @ half
        bottom_z = float(center[2] - world_half_extents[2])
        delta = bottom_z - WORK_MAT_TOP_Z
        checks.append({
            "name": name,
            "pad_bottom_z": bottom_z,
            "pad_bottom_minus_mat_top_m": delta,
        })
    min_delta = min((float(item["pad_bottom_minus_mat_top_m"]) for item in checks), default=float("inf"))
    return {
        "passed": bool(min_delta >= -PAD_MAT_GUARD_TOLERANCE_M),
        "min_pad_bottom_minus_mat_top_m": min_delta,
        "mat_top_z": WORK_MAT_TOP_Z,
        "checks": checks,
    }



def _gripper_visual_mat_guard(env: nexus.MyCobotNexusEnv) -> dict[str, Any]:
    checks = []
    for geom_id in range(env.model.ngeom):
        name = env._mujoco.mj_id2name(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not (name.startswith("gripper_") or name in {ROBOT_LEFT_PAD, ROBOT_RIGHT_PAD}):
            continue
        center = np.asarray(env.data.geom_xpos[geom_id], dtype=float)
        xmat = np.asarray(env.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        half = np.asarray(env.model.geom_size[geom_id, :3], dtype=float)
        world_half_extents = np.abs(xmat) @ half
        bottom_z = float(center[2] - world_half_extents[2])
        delta = bottom_z - WORK_MAT_TOP_Z
        checks.append({
            "name": name,
            "bottom_z": bottom_z,
            "bottom_minus_mat_top_m": delta,
        })
    min_delta = min((float(item["bottom_minus_mat_top_m"]) for item in checks), default=float("inf"))
    return {
        "passed": bool(min_delta >= -GRIPPER_VISUAL_MAT_GUARD_TOLERANCE_M),
        "min_gripper_visual_bottom_minus_mat_top_m": min_delta,
        "mat_top_z": WORK_MAT_TOP_Z,
        "checks": checks,
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



def _pad_cube_contact_depth(env: nexus.MyCobotNexusEnv) -> dict[str, Any]:
    mujoco = env._mujoco
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    pad_ids = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_LEFT_PAD): "left",
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_RIGHT_PAD): "right",
    }
    depths: list[float] = []
    checks: list[dict[str, Any]] = []
    for index in range(int(env.data.ncon)):
        contact = env.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id not in pair:
            continue
        for pad_id, side in pad_ids.items():
            if pad_id >= 0 and pad_id in pair:
                dist = float(contact.dist)
                penetration = max(0.0, -dist)
                depths.append(penetration)
                checks.append({"side": side, "dist_m": dist, "penetration_m": penetration})
    return {
        "contact_count": len(depths),
        "max_penetration_m": max(depths, default=0.0),
        "mean_penetration_m": float(np.mean(depths)) if depths else 0.0,
        "checks": checks,
    }


def _completion_standard(records: list[dict[str, Any]], lift_records: list[dict[str, Any]], final: dict[str, Any], *, status: str) -> dict[str, Any]:
    max_penetration = max(float(r["pad_cube_contact_depth"]["max_penetration_m"]) for r in records)
    max_lift_penetration = max((float(r["pad_cube_contact_depth"]["max_penetration_m"]) for r in lift_records), default=0.0)
    checks = {
        "raw_lift_status_passed": status == "passed",
        "teacher_attachment_disabled": True,
        "object_teleport_disabled": True,
        "first_frame_no_pad_contact": int(records[0]["pad_cube_contacted_pads"]) == 0,
        "cube_stays_on_or_above_mat": all(bool(r["mat_guard"]["bottom_on_or_above_mat"]) for r in records),
        "cube_starts_on_mat": bool(records[0]["mat_guard"]["passed"]),
        "pad_mat_guard_all_steps": all(bool(r["pad_mat_guard"]["passed"]) for r in records),
        "gripper_visual_mat_guard_all_steps": all(bool(r["gripper_visual_mat_guard"]["passed"]) for r in records),
        "final_two_pad_contact": int(final["pad_cube_contacted_pads"]) >= 2,
        "final_lift_at_least_25mm": float(final["cube_lift_m"]) >= 0.025,
        "lift_sustained_two_pad_steps_at_least_60": _best_sustained_two_pad(lift_records) >= 60,
        "max_pad_cube_penetration_within_threshold": max_penetration <= MAX_PAD_CUBE_PENETRATION_M,
    }
    return {
        "name": "mycobot_280_base_aligned_surface_pickup_v0",
        "status": "passed" if all(checks.values()) else "failed",
        "thresholds": {
            "required_final_lift_m": 0.025,
            "required_lift_sustained_two_pad_steps": 60,
            "max_pad_cube_penetration_m": MAX_PAD_CUBE_PENETRATION_M,
            "mat_alignment": "work_mat_top_z equals robot base support plane for this scene",
        },
        "checks": checks,
        "max_pad_cube_penetration_m": max_penetration,
        "max_lift_pad_cube_penetration_m": max_lift_penetration,
    }

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
