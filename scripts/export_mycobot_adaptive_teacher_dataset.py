#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_ai_agent.sim.mycobot_nexus_env import (  # noqa: E402
    ADAPTIVE_GATE7_TABLE_ARM_QPOS,
    ADAPTIVE_GATE8_LIFT_ARM_QPOS,
    MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
    TASK_CUBE_BODY,
    TASK_CUBE_GEOM,
    TASK_CUBE_POS,
    MyCobotNexusConfig,
    MyCobotNexusEnv,
    _json_safe_info,
    _lerp_vector,
    _smoothstep,
    _write_bmp,
)


JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
    "gripper_controller",
]
NATURAL_READY_ARM_QPOS = (0.0, 0.28, -0.18, 0.16, 0.0, 0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a small myCobot 320 adaptive-gripper teacher dataset from Gate 8."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_teacher_datasets/mycobot_320_adaptive_gate8_10eps"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros2"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pregrasp-steps", type=int, default=48)
    parser.add_argument("--close-steps", type=int, default=84)
    parser.add_argument("--lift-steps", type=int, default=92)
    parser.add_argument("--placement-gripper-command", type=float, default=0.25)
    parser.add_argument("--close-gripper-command", type=float, default=-0.75)
    parser.add_argument("--cube-half-size", type=float, default=0.02)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = export_dataset(
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render_every=args.render_every,
        pregrasp_steps=args.pregrasp_steps,
        close_steps=args.close_steps,
        lift_steps=args.lift_steps,
        placement_gripper_command=args.placement_gripper_command,
        close_gripper_command=args.close_gripper_command,
        cube_half_size=args.cube_half_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["episodes"] != args.episodes or report["failed_episodes"]:
        raise SystemExit(1)


def export_dataset(
    *,
    output_dir: Path,
    episodes: int,
    seed: int,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
    fps: int,
    render_every: int,
    pregrasp_steps: int,
    close_steps: int,
    lift_steps: int,
    placement_gripper_command: float,
    close_gripper_command: float,
    cube_half_size: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "episodes", ignore_errors=True)
    shutil.rmtree(output_dir / "frames", ignore_errors=True)
    shutil.rmtree(output_dir / "scene_cache", ignore_errors=True)
    (output_dir / "episodes").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    episode_summaries = []
    total_frames = 0
    failed_episodes = []
    for episode_index in range(episodes):
        summary = _export_episode(
            output_dir=output_dir,
            episode_index=episode_index,
            seed=seed + episode_index,
            asset_root=asset_root,
            official_gripper_root=official_gripper_root,
            width=width,
            height=height,
            fps=fps,
            render_every=render_every,
            pregrasp_steps=pregrasp_steps,
            close_steps=close_steps,
            lift_steps=lift_steps,
            placement_gripper_command=placement_gripper_command,
            close_gripper_command=close_gripper_command,
            cube_half_size=cube_half_size,
        )
        episode_summaries.append(summary)
        total_frames += int(summary["frames"])
        if not summary["success"]:
            failed_episodes.append(episode_index)
    manifest = {
        "format": "mycobot_jsonl_v1",
        "dataset_id": output_dir.name,
        "robot": "myCobot 320 M5 2022 + adaptive gripper",
        "task": "short_grasp_lift_red_cube",
        "trajectory": "natural_ready_smooth_full_arm_lift",
        "cube_half_size": cube_half_size,
        "success_criteria": {
            "close_best_sustained_contact_steps": 15,
            "lift_best_sustained_contact_steps": 25,
            "final_cube_lift": 0.025,
            "final_gripper_cube_contact_pads": 2,
        },
        "episodes": episodes,
        "frames": total_frames,
        "fps": fps,
        "render_every": render_every,
        "image_mime_type": "image/bmp",
        "joint_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "episode_summaries": episode_summaries,
        "failed_episodes": failed_episodes,
        "viewer": {
            "type": "mycobot_jsonl",
            "serve_script": "scripts/serve_so101_dataset_viewer.py",
            "env": f"MYCOBOT_TEMP_DATASETS={output_dir.name}={output_dir}",
        },
        "notes": (
            "Gate 8 teacher dataset POC. Episodes start from a natural ready pose, "
            "use full-frame 30fps rendering, move continuously through approach, "
            "grasp, and lift, and keep a full-arm camera view; this is not yet LeRobot parquet."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _export_episode(
    *,
    output_dir: Path,
    episode_index: int,
    seed: int,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
    fps: int,
    render_every: int,
    pregrasp_steps: int,
    close_steps: int,
    lift_steps: int,
    placement_gripper_command: float,
    close_gripper_command: float,
    cube_half_size: float,
) -> dict[str, Any]:
    episode_path = output_dir / "episodes" / f"episode_{episode_index:04d}.jsonl"
    frame_dir = output_dir / "frames" / f"episode_{episode_index:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    scene_cache = output_dir / "scene_cache" / f"episode_{episode_index:04d}"
    env = MyCobotNexusEnv(
        MyCobotNexusConfig(
            asset_root=asset_root,
            work_dir=scene_cache,
            official_gripper_root=official_gripper_root,
            model_profile=MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
            width=width,
            height=height,
        )
    )
    _resize_scene_cube(env, cube_half_size)
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        _set_arm_pose(env, ADAPTIVE_GATE7_TABLE_ARM_QPOS)
        env._set_gripper(command=placement_gripper_command)
        env._mujoco.mj_forward(env.model, env.data)
        pad_midpoint = env._finger_pad_midpoint()
        initial_cube_position = [
            float(pad_midpoint[0]),
            float(pad_midpoint[1]),
            float(cube_half_size + 0.008),
        ]
        for axis, value in enumerate(initial_cube_position):
            env.data.qpos[env._cube_freejoint_qpos_index + axis] = float(value)
        qvel_start = env._cube_freejoint_qvel_index
        env.data.qvel[qvel_start:qvel_start + 6] = 0.0
        env._cube_initial_pos = list(initial_cube_position)
        env._mujoco.mj_forward(env.model, env.data)
        _set_arm_pose(env, NATURAL_READY_ARM_QPOS)
        env._set_gripper(command=placement_gripper_command)
        env._mujoco.mj_forward(env.model, env.data)
        step_index = 0
        approach_denominator = max(pregrasp_steps - 1, 1)
        for step in range(pregrasp_steps):
            alpha = _smoothstep(step / approach_denominator)
            arm = _lerp_vector(
                list(NATURAL_READY_ARM_QPOS),
                list(ADAPTIVE_GATE7_TABLE_ARM_QPOS),
                alpha,
            )
            step_index = _append_step(
                env,
                rows,
                output_dir,
                frame_dir,
                episode_index,
                step_index,
                "approach",
                [*arm, placement_gripper_command],
                fps,
                render_every,
            )
        close_denominator = max(close_steps - 1, 1)
        for step in range(close_steps):
            alpha = step / close_denominator
            gripper = placement_gripper_command + alpha * (close_gripper_command - placement_gripper_command)
            step_index = _append_step(
                env,
                rows,
                output_dir,
                frame_dir,
                episode_index,
                step_index,
                "close",
                [*ADAPTIVE_GATE7_TABLE_ARM_QPOS, gripper],
                fps,
                render_every,
            )
        lift_denominator = max(lift_steps - 1, 1)
        for step in range(lift_steps):
            alpha = _smoothstep(step / lift_denominator)
            arm = _lerp_vector(
                list(ADAPTIVE_GATE7_TABLE_ARM_QPOS),
                list(ADAPTIVE_GATE8_LIFT_ARM_QPOS),
                alpha,
            )
            step_index = _append_step(
                env,
                rows,
                output_dir,
                frame_dir,
                episode_index,
                step_index,
                "lift",
                [*arm, close_gripper_command],
                fps,
                render_every,
            )
    finally:
        env.close()
        shutil.rmtree(scene_cache, ignore_errors=True)
    episode_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    close_infos = [row["info"] for row in rows if row["phase"] == "close"]
    lift_infos = [row["info"] for row in rows if row["phase"] == "lift"]
    final = rows[-1]["info"]
    final_lift = float(final["cube_lift"])
    success = (
        _best_sustained_two_pad_contact(close_infos) >= 15
        and _best_sustained_two_pad_contact(lift_infos) >= 25
        and int(final["gripper_cube_contact_pads"]) >= 2
        and final_lift >= 0.025
    )
    return {
        "episode_index": episode_index,
        "path": str(episode_path.relative_to(output_dir)),
        "frames": len(rows),
        "rendered_frames": sum(
            1 for row in rows if row.get("observation", {}).get("images", {}).get("render")
        ),
        "success": success,
        "close_best_sustained_contact_steps": _best_sustained_two_pad_contact(close_infos),
        "lift_best_sustained_contact_steps": _best_sustained_two_pad_contact(lift_infos),
        "final_cube_lift": final_lift,
        "final_gripper_cube_contact_pads": int(final["gripper_cube_contact_pads"]),
        "final_gripper_cube_contacts": int(final["gripper_cube_contacts"]),
    }


def _append_step(
    env: MyCobotNexusEnv,
    rows: list[dict[str, Any]],
    output_dir: Path,
    frame_dir: Path,
    episode_index: int,
    step_index: int,
    phase: str,
    action: list[float],
    fps: int,
    render_every: int,
) -> int:
    if phase == "approach":
        _set_arm_pose(env, tuple(action[:6]))
        env._mujoco.mj_forward(env.model, env.data)
    obs, reward, terminated, truncated, info = env.step(action)
    image = ""
    if step_index % max(1, render_every) == 0:
        image_path = frame_dir / f"frame_{step_index:04d}.bmp"
        _write_bmp(image_path, env.render())
        image = str(image_path.relative_to(output_dir))
    rows.append(
        {
            "episode_index": episode_index,
            "frame_index": step_index,
            "timestamp": step_index / float(fps),
            "phase": phase,
            "task": "short_grasp_lift_red_cube",
            "observation": {"state": obs, "images": {"render": image} if image else {}},
            "action": action,
            "reward": reward,
            "done": bool(terminated or truncated),
            "info": _json_safe_info(info),
        }
    )
    return step_index + 1


def _set_arm_pose(env: MyCobotNexusEnv, qpos: tuple[float, ...]) -> None:
    for qpos_index, value in zip(env._qpos_indices, qpos, strict=True):
        env.data.qpos[qpos_index] = float(value)
    for actuator_index, value in zip(env._arm_actuator_indices, qpos, strict=True):
        env.data.ctrl[actuator_index] = float(value)


def _resize_scene_cube(env: MyCobotNexusEnv, cube_half_size: float) -> None:
    tree = ET.parse(env.scene_path)
    root = tree.getroot()
    cube_body = root.find(f".//body[@name='{TASK_CUBE_BODY}']")
    cube_geom = root.find(f".//geom[@name='{TASK_CUBE_GEOM}']")
    if cube_body is None or cube_geom is None:
        raise RuntimeError("missing task cube body/geom in generated myCobot scene")
    cube_body.set(
        "pos",
        f"{TASK_CUBE_POS[0]} {TASK_CUBE_POS[1]} {cube_half_size + 0.008}",
    )
    cube_geom.set("size", f"{cube_half_size} {cube_half_size} {cube_half_size}")
    cube_geom.set("mass", "0.005")
    tree.write(env.scene_path, encoding="utf-8", xml_declaration=True)
    env.model = env._mujoco.MjModel.from_xml_path(str(env.scene_path))
    env.data = env._mujoco.MjData(env.model)
    env._renderer = None
    cube_joint_id = env._mujoco.mj_name2id(
        env.model,
        env._mujoco.mjtObj.mjOBJ_JOINT,
        "task_cube_freejoint",
    )
    env._cube_freejoint_qpos_index = int(env.model.jnt_qposadr[cube_joint_id])
    env._cube_freejoint_qvel_index = int(env.model.jnt_dofadr[cube_joint_id])
    env._qpos_indices = [
        int(env.model.jnt_qposadr[env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in env._arm_joint_names
    ]
    env._dof_indices = [
        int(env.model.jnt_dofadr[env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in env._arm_joint_names
    ]
    env._arm_actuator_indices = [
        int(env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}"))
        for name in env._arm_joint_names
    ]


def _best_sustained_two_pad_contact(infos: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    for info in infos:
        if int(info.get("gripper_cube_contact_pads", 0)) >= 2:
            current += 1
        else:
            current = 0
        best = max(best, current)
    return best


if __name__ == "__main__":
    main()
