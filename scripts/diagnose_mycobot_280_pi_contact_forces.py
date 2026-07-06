#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402


@dataclass(frozen=True)
class ContactForceSample:
    step: int
    phase: str
    cube_position: list[float]
    cube_lift: float
    gripper_command: float
    gripper_cube_contact_pads: int
    gripper_cube_contacts: int
    contacts: list[dict[str, Any]]


@dataclass(frozen=True)
class ContactForceReport:
    status: str
    output_dir: str
    scene_path: str
    config: dict[str, Any]
    close_best_sustained_contact_steps: int
    lift_best_sustained_contact_steps: int
    lift_two_pad_contact_steps: int
    final_gripper_cube_contact_pads: int
    final_cube_lift: float
    sample_count: int
    samples: list[ContactForceSample]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay 280 Pi adaptive raw Gate 8 and record pad/cube contact "
            "normals and forces around the close-to-lift transition."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/mycobot_280_contact_force_diag"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--close-gripper-command", type=float, default=-0.12)
    parser.add_argument("--placement-gripper-command", type=float, default=1.0)
    parser.add_argument("--pregrasp-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=240)
    parser.add_argument("--lift-steps", type=int, default=120)
    parser.add_argument("--cube-offset-x", type=float, default=nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET[0])
    parser.add_argument("--cube-offset-y", type=float, default=nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET[1])
    parser.add_argument("--cube-offset-z", type=float, default=nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET[2])
    parser.add_argument("--pad-offset-y", type=float, default=0.0)
    parser.add_argument("--pad-offset-z", type=float, default=0.0)
    parser.add_argument("--pad-size-x", type=float, default=nexus.ADAPTIVE_280_FINGER_PAD_SIZE[0])
    parser.add_argument("--pad-size-y", type=float, default=nexus.ADAPTIVE_280_FINGER_PAD_SIZE[1])
    parser.add_argument("--pad-size-z", type=float, default=nexus.ADAPTIVE_280_FINGER_PAD_SIZE[2])
    parser.add_argument("--cube-mass", type=float, default=0.005)
    parser.add_argument("--sample-tail", type=int, default=12)
    parser.add_argument("--sample-lift-head", type=int, default=24)
    return parser


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = build_parser().parse_args()
    report = diagnose_contact_forces(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "contact_force_report.json"
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 1)


def diagnose_contact_forces(args: argparse.Namespace) -> ContactForceReport:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _install_scene_overrides(args)
    _install_geometry_overrides(args)
    env = nexus.MyCobotNexusEnv(
        nexus.MyCobotNexusConfig(
            asset_root=args.asset_root,
            work_dir=args.output_dir,
            official_gripper_root=args.official_gripper_root,
            model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            width=64,
            height=48,
            teacher_grasp_attachment_enabled=False,
        )
    )
    records: list[dict[str, Any]] = []
    samples: list[ContactForceSample] = []
    try:
        env.reset(seed=1)
        gate7_arm_qpos = nexus._adaptive_gate7_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        gate8_lift_arm_qpos = nexus._adaptive_gate8_lift_arm_qpos(nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
        env._set_gripper(command=args.placement_gripper_command)
        env._mujoco.mj_forward(env.model, env.data)
        pad_midpoint = env._finger_pad_midpoint()
        initial_cube_position = [
            float(pad_midpoint[0] + args.cube_offset_x),
            float(pad_midpoint[1] + args.cube_offset_y),
            float(nexus.TASK_CUBE_POS[2] + args.cube_offset_z),
        ]
        for axis, value in enumerate(initial_cube_position):
            env.data.qpos[env._cube_freejoint_qpos_index + axis] = float(value)
        env.data.qvel[env._cube_freejoint_qvel_index:env._cube_freejoint_qvel_index + 6] = 0.0
        env._cube_initial_pos = list(initial_cube_position)
        env._mujoco.mj_forward(env.model, env.data)

        step_index = 0
        for _ in range(args.pregrasp_steps):
            nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            obs, reward, terminated, truncated, info = env.step([*gate7_arm_qpos, args.placement_gripper_command])
            nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            records.append(nexus._phase_record("pregrasp", step_index, obs, reward, terminated, truncated, info))
            step_index += 1

        close_start = step_index
        close_denominator = max(args.close_steps - 1, 1)
        for close_step in range(args.close_steps):
            alpha = close_step / close_denominator
            gripper = args.placement_gripper_command + alpha * (
                args.close_gripper_command - args.placement_gripper_command
            )
            nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            obs, reward, terminated, truncated, info = env.step([*gate7_arm_qpos, gripper])
            nexus._set_adaptive_gate_arm_pose(env, gate7_arm_qpos)
            phase_record = nexus._phase_record("close", step_index, obs, reward, terminated, truncated, info)
            records.append(phase_record)
            if close_step >= max(0, args.close_steps - args.sample_tail):
                samples.append(_contact_sample(env, phase_record))
            step_index += 1

        lift_denominator = max(args.lift_steps - 1, 1)
        for lift_step in range(args.lift_steps):
            alpha = nexus._smoothstep(lift_step / lift_denominator)
            arm = nexus._lerp_vector(list(gate7_arm_qpos), list(gate8_lift_arm_qpos), alpha)
            nexus._set_adaptive_gate_arm_pose(env, tuple(arm))
            obs, reward, terminated, truncated, info = env.step([*arm, args.close_gripper_command])
            nexus._set_adaptive_gate_arm_pose(env, tuple(arm))
            phase_record = nexus._phase_record("lift", step_index, obs, reward, terminated, truncated, info)
            records.append(phase_record)
            if lift_step < args.sample_lift_head:
                samples.append(_contact_sample(env, phase_record))
            step_index += 1

        close_infos = [record["info"] for record in records if record["phase"] == "close"]
        lift_infos = [record["info"] for record in records if record["phase"] == "lift"]
        final_info = records[-1]["info"]
        final_cube = env._cube_position()
        final_lift = float(final_cube[2]) - float(initial_cube_position[2])
        lift_best = nexus._best_sustained_two_pad_contact(lift_infos)
        status = (
            "passed"
            if nexus._best_sustained_two_pad_contact(close_infos) >= 15
            and lift_best >= 30
            and int(final_info.get("gripper_cube_contact_pads", 0)) >= 2
            and final_lift >= 0.025
            else "failed"
        )
        return ContactForceReport(
            status=status,
            output_dir=str(args.output_dir),
            scene_path=str(env.scene_path),
            config={
                "close_gripper_command": args.close_gripper_command,
                "close_steps": args.close_steps,
                "lift_steps": args.lift_steps,
                "cube_offset": [args.cube_offset_x, args.cube_offset_y, args.cube_offset_z],
                "pad_offset_y": args.pad_offset_y,
                "pad_offset_z": args.pad_offset_z,
                "pad_size": [args.pad_size_x, args.pad_size_y, args.pad_size_z],
                "cube_mass": args.cube_mass,
                "close_start_step": close_start,
            },
            close_best_sustained_contact_steps=nexus._best_sustained_two_pad_contact(close_infos),
            lift_best_sustained_contact_steps=lift_best,
            lift_two_pad_contact_steps=sum(1 for info in lift_infos if int(info.get("gripper_cube_contact_pads", 0)) >= 2),
            final_gripper_cube_contact_pads=int(final_info.get("gripper_cube_contact_pads", 0)),
            final_cube_lift=final_lift,
            sample_count=len(samples),
            samples=samples,
        )
    finally:
        env.close()


def _install_geometry_overrides(args: argparse.Namespace) -> None:
    left = nexus.ADAPTIVE_LEFT_FINGER_PAD_POS
    right = nexus.ADAPTIVE_RIGHT_FINGER_PAD_POS
    nexus.ADAPTIVE_280_LEFT_FINGER_PAD_POS = (
        left[0],
        left[1] + args.pad_offset_y,
        left[2] + args.pad_offset_z,
    )
    nexus.ADAPTIVE_280_RIGHT_FINGER_PAD_POS = (
        right[0],
        right[1] + args.pad_offset_y,
        right[2] + args.pad_offset_z,
    )
    nexus.ADAPTIVE_280_FINGER_PAD_SIZE = (args.pad_size_x, args.pad_size_y, args.pad_size_z)
    nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET = (
        args.cube_offset_x,
        args.cube_offset_y,
        args.cube_offset_z,
    )


def _install_scene_overrides(args: argparse.Namespace) -> None:
    original_build = nexus.build_mycobot_nexus_scene_model

    def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> None:
        original_build(*wrapper_args, **wrapper_kwargs)
        scene_path = wrapper_kwargs.get("scene_path") if "scene_path" in wrapper_kwargs else wrapper_args[1]
        tree = ET.parse(scene_path)
        root = tree.getroot()
        for geom in root.findall(".//geom"):
            if geom.attrib.get("name") == nexus.TASK_CUBE_GEOM:
                geom.set("mass", str(args.cube_mass))
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    nexus.build_mycobot_nexus_scene_model = wrapper


def _contact_sample(env: nexus.MyCobotNexusEnv, record: dict[str, Any]) -> ContactForceSample:
    info = record["info"]
    return ContactForceSample(
        step=int(record["step"]),
        phase=str(record["phase"]),
        cube_position=[float(value) for value in info["cube_position"]],
        cube_lift=float(info["cube_lift"]),
        gripper_command=float(info["gripper_command"]),
        gripper_cube_contact_pads=int(info["gripper_cube_contact_pads"]),
        gripper_cube_contacts=int(info["gripper_cube_contacts"]),
        contacts=_pad_cube_contacts(env),
    )


def _pad_cube_contacts(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    mujoco = env._mujoco
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    pad_ids = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"): "left_finger_pad",
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"): "right_finger_pad",
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
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        normal = frame[0]
        world_force = frame.T @ force[:3]
        geom1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        cube_force_world = world_force if int(contact.geom1) == cube_id else -world_force
        world_normal_force = normal * float(force[0])
        contacts.append(
            {
                "pad": pad_ids[pad_geom_ids[0]],
                "geom1": geom1_name,
                "geom2": geom2_name,
                "distance": float(contact.dist),
                "position": [float(value) for value in contact.pos],
                "normal": [float(value) for value in normal],
                "force_components": [float(value) for value in force],
                "world_force": [float(value) for value in world_force],
                "world_force_z": float(world_force[2]),
                "cube_force_world_assuming_geom1": [float(value) for value in cube_force_world],
                "cube_force_world_z_assuming_geom1": float(cube_force_world[2]),
                "world_normal_force": [float(value) for value in world_normal_force],
                "world_normal_force_z": float(world_normal_force[2]),
            }
        )
    return contacts


if __name__ == "__main__":
    main()
