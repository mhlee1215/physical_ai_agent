#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402
from physical_ai_agent.sim.mycobot_nexus_env import _json_safe_info, _write_bmp  # noqa: E402
from scripts.run_mycobot_280_ground_pickup_poc import (  # noqa: E402
    APPROACH_STEPS,
    CLOSE_STEPS,
    CONTACT_COMMAND,
    CUBE_HALF_SIZE,
    CUBE_MASS,
    HOLD_STEPS,
    LIFT_STEPS,
    MAT_FRICTION,
    PAD_FRICTION,
    START_COMMAND,
    WORK_MAT_TOP_Z,
    WORLD_GRAVITY,
    _apply_physics_overrides,
    _best_sustained_two_pad,
    _initial_cube_pose,
    _passes,
    _record,
    _scripted_state,
)
from scripts.render_mycobot_280_cube_contact_sequence import _set_cube_pose, _size_audit_cube  # noqa: E402

JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint7_to_joint6",
    "gripper_controller",
]
TASK = "pick up the cube from the work mat with the myCobot 280 Pi adaptive gripper"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a myCobot 280 Pi adaptive-gripper cube-from-mat teacher POC dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_teacher_datasets/mycobot_280_pi_ground_pickup_poc_10eps"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "episodes", ignore_errors=True)
    shutil.rmtree(output_dir / "frames", ignore_errors=True)
    shutil.rmtree(output_dir / "scene_cache", ignore_errors=True)
    (output_dir / "episodes").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)

    episode_summaries = []
    total_rows = 0
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
        )
        episode_summaries.append(summary)
        total_rows += int(summary["frames"])
        if not summary["success"]:
            failed_episodes.append(episode_index)

    manifest = {
        "format": "mycobot_jsonl_v1",
        "dataset_id": output_dir.name,
        "robot": "myCobot 280 Pi + adaptive gripper",
        "model_profile": nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        "task": TASK,
        "trajectory": "true_cube_from_work_mat_open_align_close_grasp_lift",
        "teacher_attachment_enabled": False,
        "object_teleport_during_pickup_lift": False,
        "cube_starts_on_work_mat": True,
        "cube_half_size_m": CUBE_HALF_SIZE,
        "cube_mass_kg": CUBE_MASS,
        "work_mat_top_z_m": WORK_MAT_TOP_Z,
        "mat_friction": MAT_FRICTION,
        "pad_friction": PAD_FRICTION,
        "success_criteria": {
            "final_cube_lift_m": 0.025,
            "final_gripper_cube_contact_pads": 2,
            "lift_best_sustained_two_pad_steps": 60,
        },
        "episodes": episodes,
        "frames": total_rows,
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
            "Cube-from-mat teacher POC. The cube is placed on the work mat only at episode "
            "initialization; pickup and lift use raw MuJoCo gripper/cube contact with no "
            "teacher attachment or object teleporting during pickup/lift."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
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
) -> dict[str, Any]:
    episode_path = output_dir / "episodes" / f"episode_{episode_index:04d}.jsonl"
    frame_dir = output_dir / "frames" / f"episode_{episode_index:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    scene_cache = output_dir / "scene_cache" / f"episode_{episode_index:04d}"
    env = nexus.MyCobotNexusEnv(
        nexus.MyCobotNexusConfig(
            asset_root=asset_root,
            work_dir=scene_cache,
            official_gripper_root=official_gripper_root,
            model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            width=width,
            height=height,
            teacher_grasp_attachment_enabled=False,
        )
    )
    try:
        env.reset(seed=seed)
        env._diagnostic_cube_half_size = CUBE_HALF_SIZE
        _size_audit_cube(env, half_size=CUBE_HALF_SIZE)
        _apply_physics_overrides(env)
        env.model.opt.gravity[:] = WORLD_GRAVITY
        cube_pos, cube_quat = _initial_cube_pose(env)
        env._set_gripper(command=START_COMMAND)
        _set_cube_pose(env, cube_pos, cube_quat)
        env._mujoco.mj_forward(env.model, env.data)
        initial_cube = env._cube_position()
        rows: list[dict[str, Any]] = []
        total_steps = APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS + LIFT_STEPS
        for step_index in range(total_steps):
            arm, command, phase = _scripted_state(step_index)
            obs, reward, terminated, truncated, info = env.step([*tuple(float(x) for x in arm), float(command)])
            nexus._set_adaptive_gate_arm_pose(env, tuple(float(x) for x in arm))
            env._set_gripper(command=float(command))
            env._mujoco.mj_forward(env.model, env.data)
            record = _record(env, step=step_index, phase=phase, command=float(command), initial_cube=initial_cube)
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
                    "task": TASK,
                    "observation": {"state": obs, "images": {"render": image} if image else {}},
                    "action": [*tuple(float(x) for x in arm), float(command)],
                    "reward": reward,
                    "done": bool(terminated or truncated),
                    "info": {**_json_safe_info(info), "ground_pickup": record},
                }
            )
    finally:
        env.close()
        shutil.rmtree(scene_cache, ignore_errors=True)

    episode_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    records = [row["info"]["ground_pickup"] for row in rows]
    lift_records = [record for record in records if record["phase"] == "lift_from_mat"]
    final = records[-1]
    success = _passes(records, lift_records, final)
    return {
        "episode_index": episode_index,
        "path": str(episode_path.relative_to(output_dir)),
        "frames": len(rows),
        "rendered_frames": sum(1 for row in rows if row["observation"]["images"].get("render")),
        "success": success,
        "first_frame_pad_cube_contacted_pads": records[0]["pad_cube_contacted_pads"],
        "first_contact_step": next((record["step"] for record in records if record["pad_cube_contacted_pads"] > 0), None),
        "final_cube_lift_m": final["cube_lift_m"],
        "final_gripper_cube_contact_pads": final["pad_cube_contacted_pads"],
        "lift_best_sustained_two_pad_steps": _best_sustained_two_pad(lift_records),
    }


if __name__ == "__main__":
    main()
