#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402
from scripts.render_mycobot_280_cube_contact_sequence import (  # noqa: E402
    AUDIT_CUBE_HALF_SIZE,
    _quat_align_x_to_vector,
    _set_cube_pose,
    _size_audit_cube,
)
from scripts.run_mycobot_280_raw_lift_rollout import (  # noqa: E402
    WORLD_GRAVITY,
    _best_sustained_two_pad,
    _first_contact_loss,
    _rollout_record,
)

DEFAULT_CASES = (
    (1, 0.0, 0.0, 0.0),
    (2, 0.0, 0.001, 0.0),
    (3, 0.0, -0.001, 0.0),
    (4, 0.0, 0.0, 0.001),
    (5, 0.0, 0.0, -0.0005),
    (6, 0.001, 0.0, 0.0),
    (7, -0.001, 0.0, 0.0),
    (8, 0.001, 0.001, 0.0),
    (9, -0.001, -0.001, 0.0),
    (10, 0.0, 0.0005, 0.0005),
)

MODEL_SPECS = (
    {
        "name": "280_pi_adaptive",
        "profile": nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        "official_gripper_root_arg": "ros_root",
        "start_command": 1.0,
        "contact_command": -0.15,
    },
    {
        "name": "320_m5_adaptive",
        "profile": nexus.MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
        "official_gripper_root_arg": "ros2_root",
        "start_command": 0.25,
        "contact_command": -0.7,
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a standardized raw-contact-only actuator-lift parity benchmark for myCobot 280 and 320 adaptive grippers."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/rollouts/mycobot_280_320_standardized_raw_lift_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--ros-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--ros2-root", type=Path, default=Path("_vendor/mycobot_ros2"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--zero-gravity-close-steps", type=int, default=100)
    parser.add_argument("--gravity-hold-steps", type=int, default=60)
    parser.add_argument("--lift-steps", type=int, default=140)
    parser.add_argument("--required-lift-two-pad-steps", type=int, default=60)
    parser.add_argument("--required-final-lift", type=float, default=0.025)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for episode, (seed, ox, oy, oz) in enumerate(DEFAULT_CASES):
            row = _run_episode(args, spec, episode, seed, (ox, oy, oz))
            rows.append(row)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "model",
                            "episode",
                            "seed",
                            "status",
                            "lift_two_pad_steps",
                            "final_cube_lift_m",
                            "final_pad_cube_contacted_pads",
                            "first_lift_contact_loss_step",
                        )
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = _summary(args, rows)
    summary_path = args.output_dir / "standardized_280_320_raw_lift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "passed" else 1)


def _run_episode(args: argparse.Namespace, spec: dict[str, Any], episode: int, seed: int, offset: tuple[float, float, float]) -> dict[str, Any]:
    model_name = str(spec["name"])
    case_dir = args.output_dir / model_name / f"episode_{episode:02d}_seed_{seed}"
    gripper_root = getattr(args, str(spec["official_gripper_root_arg"]))
    env = nexus.MyCobotNexusEnv(
        nexus.MyCobotNexusConfig(
            asset_root=args.asset_root,
            work_dir=case_dir,
            official_gripper_root=gripper_root,
            model_profile=str(spec["profile"]),
            width=args.width,
            height=args.height,
            teacher_grasp_attachment_enabled=False,
        )
    )
    try:
        env.reset(seed=seed)
        gate7 = nexus._adaptive_gate7_arm_qpos(str(spec["profile"]))
        gate8 = nexus._adaptive_gate8_lift_arm_qpos(str(spec["profile"]))
        _size_audit_cube(env, half_size=AUDIT_CUBE_HALF_SIZE)
        env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        nexus._set_adaptive_gate_arm_pose(env, gate7)
        env._set_gripper(command=float(spec["start_command"]))
        env._mujoco.mj_forward(env.model, env.data)
        left = _geom_pos(env, "left_finger_pad")
        right = _geom_pos(env, "right_finger_pad")
        cube_pos = (left + right) * 0.5 + np.asarray(offset, dtype=float)
        cube_quat = _quat_align_x_to_vector(right - left)
        _set_cube_pose(env, cube_pos, cube_quat)
        env._mujoco.mj_forward(env.model, env.data)

        records: list[dict[str, Any]] = []
        total_steps = args.zero_gravity_close_steps + args.gravity_hold_steps + args.lift_steps
        lift_start = args.zero_gravity_close_steps + args.gravity_hold_steps
        for step in range(total_steps):
            if step < args.zero_gravity_close_steps:
                phase = "close_zero_g"
                alpha = step / max(args.zero_gravity_close_steps - 1, 1)
                command = float(spec["start_command"]) + alpha * (float(spec["contact_command"]) - float(spec["start_command"]))
                arm = gate7
                env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
            elif step < lift_start:
                phase = "gravity_hold"
                command = float(spec["contact_command"])
                arm = gate7
                env.model.opt.gravity[:] = WORLD_GRAVITY
            else:
                phase = "lift_gravity_on"
                lift_step = step - lift_start
                alpha = nexus._smoothstep(lift_step / max(args.lift_steps - 1, 1))
                arm = tuple(nexus._lerp_vector(list(gate7), list(gate8), alpha))
                command = float(spec["contact_command"])
                env.model.opt.gravity[:] = WORLD_GRAVITY
            env.step([*arm, command])
            records.append(_rollout_record(env, step=step, phase=phase, command=command, initial_cube_pos=cube_pos))

        close_records = [record for record in records if record["phase"] == "close_zero_g"]
        hold_records = [record for record in records if record["phase"] == "gravity_hold"]
        lift_records = [record for record in records if record["phase"] == "lift_gravity_on"]
        final = records[-1]
        status = (
            "passed"
            if _best_sustained_two_pad(lift_records) >= args.required_lift_two_pad_steps
            and final["pad_cube_contacted_pads"] >= 2
            and final["cube_lift_m"] >= args.required_final_lift
            and final["cube_displacement_m"] < 0.20
            else "failed"
        )
        report = {
            "model": model_name,
            "profile": spec["profile"],
            "episode": episode,
            "seed": seed,
            "cube_offset_m": list(offset),
            "status": status,
            "teacher_attachment_enabled": False,
            "actuator_driven_lift": True,
            "start_command": spec["start_command"],
            "contact_command": spec["contact_command"],
            "zero_gravity_close_steps": args.zero_gravity_close_steps,
            "gravity_hold_steps": args.gravity_hold_steps,
            "lift_steps": args.lift_steps,
            "close_best_sustained_two_pad_steps": _best_sustained_two_pad(close_records),
            "hold_best_sustained_two_pad_steps": _best_sustained_two_pad(hold_records),
            "lift_best_sustained_two_pad_steps": _best_sustained_two_pad(lift_records),
            "lift_two_pad_steps": sum(1 for record in lift_records if record["pad_cube_contacted_pads"] >= 2),
            "first_lift_contact_loss_step": _first_contact_loss(lift_records),
            "final_pad_cube_contacted_pads": final["pad_cube_contacted_pads"],
            "final_cube_lift_m": final["cube_lift_m"],
            "final_cube_displacement_m": final["cube_displacement_m"],
            "initial_z_alignment": records[0]["z_alignment"],
            "pre_lift_z_alignment": records[lift_start - 1]["z_alignment"],
            "final_z_alignment": final["z_alignment"],
            "scene_path": str(env.scene_path),
        }
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "standardized_raw_lift_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        env.close()


def _summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for spec in MODEL_SPECS:
        model_rows = [row for row in rows if row["model"] == spec["name"]]
        passed = [row for row in model_rows if row["status"] == "passed"]
        by_model[str(spec["name"])] = {
            "profile": spec["profile"],
            "episode_count": len(model_rows),
            "passed_count": len(passed),
            "success_rate": len(passed) / max(len(model_rows), 1),
            "teacher_attachment_enabled_all_false": all(row["teacher_attachment_enabled"] is False for row in model_rows),
            "actuator_driven_lift_all_true": all(row["actuator_driven_lift"] is True for row in model_rows),
            "min_lift_two_pad_steps": min((row["lift_two_pad_steps"] for row in model_rows), default=0),
            "min_best_sustained_two_pad_steps": min((row["lift_best_sustained_two_pad_steps"] for row in model_rows), default=0),
            "min_final_cube_lift_m": min((row["final_cube_lift_m"] for row in model_rows), default=0.0),
            "max_abs_final_z_delta_m": max((abs(row["final_z_alignment"]["pad_mid_minus_cube_center_m"]) for row in model_rows), default=0.0),
            "start_command": spec["start_command"],
            "contact_command": spec["contact_command"],
        }
    return {
        "status": "passed" if all(value["passed_count"] == value["episode_count"] for value in by_model.values()) else "failed",
        "benchmark": "standardized_raw_contact_only_actuator_lift_280_vs_320",
        "output_dir": str(args.output_dir),
        "cube_half_size_m": AUDIT_CUBE_HALF_SIZE,
        "episode_count_total": len(rows),
        "episodes_per_model": len(DEFAULT_CASES),
        "case_offsets_m": [{"seed": seed, "offset": [ox, oy, oz]} for seed, ox, oy, oz in DEFAULT_CASES],
        "score_rule": "pass if lift_best_sustained_two_pad_steps>=required, final_pad_cube_contacted_pads>=2, final_cube_lift_m>=required, displacement<0.20",
        "required_lift_two_pad_steps": args.required_lift_two_pad_steps,
        "required_final_lift_m": args.required_final_lift,
        "by_model": by_model,
        "rows": rows,
    }


def _geom_pos(env: nexus.MyCobotNexusEnv, name: str) -> np.ndarray:
    geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"missing geom: {name}")
    return np.asarray(env.data.geom_xpos[geom_id], dtype=float).copy()


if __name__ == "__main__":
    main()
