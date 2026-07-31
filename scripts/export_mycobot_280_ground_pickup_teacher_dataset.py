#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402
from physical_ai_agent.sim.mycobot_nexus_env import _json_safe_info, _write_bmp  # noqa: E402
from scripts.run_mycobot_280_ground_pickup_poc import (  # noqa: E402
    APPROACH_STEPS,
    CLOSE_STEPS,
    CONTACT_COMMAND,
    CUBE_AXIS_OFFSET,
    CUBE_HALF_SIZE,
    CUBE_MASS,
    CUBE_SIDE_OFFSET,
    HOLD_STEPS,
    LIFT_QPOS,
    LIFT_STEPS,
    MAT_FRICTION,
    MAX_PAD_CUBE_PENETRATION_M,
    PAD_FRICTION,
    PICKUP_QPOS,
    POST_LIFT_HOLD_STEPS,
    ROBOT_LEFT_PAD,
    ROBOT_RIGHT_PAD,
    START_COMMAND,
    WORK_MAT_TOP_Z,
    WORLD_GRAVITY,
    _apply_physics_overrides,
    _best_sustained_two_pad,
    _cube_mat_guard,
    _geom_pos,
    _passes,
    _patch_nexus_work_mat_scene_nodes,
    _quat_align_x_to_vector,
    _record,
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
DEFAULT_TRAIN_EPISODES = 50
DEFAULT_VALIDATION_EPISODES = 10
DEFAULT_YAW_MIN = -0.20
DEFAULT_YAW_MAX = 0.20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a deterministic pose-diverse myCobot 280 Pi cube-from-mat teacher dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_pose_diverse_v1"),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Legacy shortcut: export this many train episodes and no validation split.",
    )
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=DEFAULT_VALIDATION_EPISODES)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument(
        "--render-camera-profile",
        choices=("full_robot", "ground_pickup_closeup"),
        default="full_robot",
    )
    parser.add_argument(
        "--image-format",
        choices=("bmp", "png"),
        default="bmp",
        help=(
            "Lossless output format. PNG substantially reduces storage for all-frame "
            "camera-ablation datasets."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--yaw-min", type=float, default=DEFAULT_YAW_MIN)
    parser.add_argument("--yaw-max", type=float, default=DEFAULT_YAW_MAX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_episodes = int(args.episodes) if args.episodes is not None else int(args.train_episodes)
    val_episodes = 0 if args.episodes is not None else int(args.val_episodes)
    report = export_dataset(
        output_dir=args.output_dir,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        seed=args.seed,
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render_every=args.render_every,
        max_attempts=args.max_attempts,
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        render_camera_profile=args.render_camera_profile,
        image_format=args.image_format,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["accepted_episodes"] != report["requested_episodes"] or report["failed_episodes"]:
        raise SystemExit(1)


def export_dataset(
    *,
    output_dir: Path,
    train_episodes: int,
    val_episodes: int,
    seed: int,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
    fps: int,
    render_every: int,
    max_attempts: int,
    yaw_min: float,
    yaw_max: float,
    render_camera_profile: str = "full_robot",
    image_format: str = "bmp",
) -> dict[str, Any]:
    if train_episodes < 0 or val_episodes < 0:
        raise ValueError("episode counts must be non-negative")
    camera_contract = _camera_contract(render_camera_profile, width=width, height=height)
    if image_format not in {"bmp", "png"}:
        raise ValueError(f"unsupported image format: {image_format}")
    requested = train_episodes + val_episodes
    if requested <= 0:
        raise ValueError("at least one train or validation episode is required")
    if max_attempts <= 0:
        max_attempts = max(requested * 3, requested)

    _patch_nexus_work_mat_scene_nodes()
    _prepare_output_root(output_dir)

    split_targets = {"train": train_episodes, "validation": val_episodes}
    split_counts = {"train": 0, "validation": 0}
    split_episode_summaries: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    accepted_summaries: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []
    total_rows = 0
    split_schedule = _split_schedule(train_episodes, val_episodes)

    for attempt_index, yaw_delta in enumerate(_candidate_yaws(max_attempts=max_attempts, yaw_min=yaw_min, yaw_max=yaw_max)):
        if len(accepted_summaries) >= requested:
            break
        split = split_schedule[len(accepted_summaries)]
        split_episode_index = split_counts[split]
        try:
            summary = _export_attempt(
                output_dir=output_dir,
                split=split,
                split_episode_index=split_episode_index,
                global_episode_index=len(accepted_summaries),
                attempt_index=attempt_index,
                seed=seed + attempt_index,
                yaw_delta=float(yaw_delta),
                asset_root=asset_root,
                official_gripper_root=official_gripper_root,
                width=width,
                height=height,
                fps=fps,
                render_every=render_every,
                render_camera_profile=render_camera_profile,
                image_format=image_format,
            )
        except Exception as exc:  # noqa: BLE001
            rejected_attempts.append(
                {
                    "attempt_index": attempt_index,
                    "split_target": split,
                    "seed": seed + attempt_index,
                    "yaw_delta_rad": float(yaw_delta),
                    "success": False,
                    "reason": f"exception: {exc}",
                }
            )
            continue
        if not summary["success"]:
            rejected_attempts.append(_rejection_from_summary(summary))
            continue
        split_counts[split] += 1
        split_episode_summaries[split].append(summary)
        accepted_summaries.append(summary)
        total_rows += int(summary["frames"])

    failed_episodes = [
        {
            "split": split,
            "missing": target - split_counts[split],
            "target": target,
            "accepted": split_counts[split],
        }
        for split, target in split_targets.items()
        if split_counts[split] != target
    ]
    aggregate_metrics = _aggregate_metrics(accepted_summaries, rejected_attempts)
    manifest = {
        "format": "mycobot_jsonl_v1",
        "dataset_id": output_dir.name,
        "robot": "myCobot 280 Pi + adaptive gripper",
        "model_profile": nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        "task": TASK,
        "dataset_kind": "deterministic_pose_diverse_jsonl",
        "generation_mode": "deterministic_pose_diverse_teacher_aligned",
        "randomization_enabled": False,
        "trajectory": "true_cube_from_work_mat_open_align_close_grasp_lift_post_lift_hold",
        "teacher_attachment_enabled": False,
        "object_teleport_during_pickup_lift": False,
        "zero_gravity_close": True,
        "post_step_snap_enabled": False,
        "cube_starts_on_work_mat": True,
        "cube_half_size_m": CUBE_HALF_SIZE,
        "cube_mass_kg": CUBE_MASS,
        "work_mat_top_z_m": WORK_MAT_TOP_Z,
        "mat_friction": MAT_FRICTION,
        "pad_friction": PAD_FRICTION,
        "success_criteria": {
            "final_cube_lift_m": 0.05,
            "final_gripper_cube_contact_pads": 2,
            "lift_best_sustained_two_pad_steps": 60,
            "post_lift_hold_best_sustained_two_pad_steps": 300,
            "post_lift_hold_min_cube_lift_m": 0.045,
            "max_pad_cube_penetration_m": 0.003,
        },
        "pose_generation": {
            "method": "deterministic_van_der_corput_base_yaw_delta",
            "yaw_min_rad": float(yaw_min),
            "yaw_max_rad": float(yaw_max),
            "max_attempts": int(max_attempts),
            "base_pickup_qpos": [float(x) for x in PICKUP_QPOS],
            "base_lift_qpos": [float(x) for x in LIFT_QPOS],
            "cube_axis_offset_m": CUBE_AXIS_OFFSET,
            "cube_side_offset_m": CUBE_SIDE_OFFSET,
        },
        "requested_episodes": requested,
        "accepted_episodes": len(accepted_summaries),
        "episodes": len(accepted_summaries),
        "passed_episodes": len(accepted_summaries),
        "frames": total_rows,
        "aggregate_metrics": aggregate_metrics,
        "splits": {
            split: {
                "requested_episodes": split_targets[split],
                "accepted_episodes": split_counts[split],
                "episode_summaries": split_episode_summaries[split],
            }
            for split in ("train", "validation")
        },
        "rejected_attempts": rejected_attempts,
        "failed_episodes": failed_episodes,
        "fps": fps,
        "render_every": render_every,
        "image_mime_type": f"image/{image_format}",
        "observation_camera": camera_contract,
        "joint_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "viewer": {
            "type": "mycobot_jsonl",
            "serve_script": "scripts/serve_so101_dataset_viewer.py",
            "env": f"MYCOBOT_TEMP_DATASETS={output_dir.name}={output_dir}",
        },
        "notes": (
            "Pose-diverse cube-from-mat teacher dataset POC. Candidate episodes vary the "
            "reachable teacher-aligned base yaw deterministically, derive the cube XY pose "
            "from the visible terminal pads, and include only successful raw-contact pickup "
            "episodes. The cube is placed on the work mat only at episode initialization; "
            "pickup and lift use raw MuJoCo gripper/cube contact with no teacher attachment "
            "or object teleporting during pickup/lift. Fingertip pads, cube, mat, and floor "
            "use MuJoCo contact; visible gripper geoms are guarded against mat-plane overlap, "
            "while broader arm/table collision remains visual-only. Gravity is disabled during "
            "approach/close only and restored for hold/lift/post-lift hold."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _prepare_output_root(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("splits", "episodes", "frames", "_attempt_frames", "scene_cache"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    for split in ("train", "validation"):
        (output_dir / "splits" / split / "episodes").mkdir(parents=True, exist_ok=True)
        (output_dir / "splits" / split / "frames").mkdir(parents=True, exist_ok=True)
    (output_dir / "_attempt_frames").mkdir(exist_ok=True)


def _candidate_yaws(*, max_attempts: int, yaw_min: float, yaw_max: float) -> list[float]:
    if max_attempts <= 1:
        return [(yaw_min + yaw_max) * 0.5]
    span = yaw_max - yaw_min
    return [yaw_min + span * _van_der_corput(index + 1, base=2) for index in range(max_attempts)]


def _van_der_corput(index: int, *, base: int) -> float:
    value = 0.0
    denom = 1.0
    while index:
        index, remainder = divmod(index, base)
        denom *= base
        value += remainder / denom
    return value


def _split_schedule(train_episodes: int, val_episodes: int) -> list[str]:
    total = train_episodes + val_episodes
    if val_episodes <= 0:
        return ["train"] * total
    val_positions = {
        min(total - 1, int(round((index + 0.5) * total / val_episodes - 0.5)))
        for index in range(val_episodes)
    }
    schedule = []
    train_remaining = train_episodes
    val_remaining = val_episodes
    for position in range(total):
        if position in val_positions and val_remaining > 0:
            schedule.append("validation")
            val_remaining -= 1
        elif train_remaining > 0:
            schedule.append("train")
            train_remaining -= 1
        else:
            schedule.append("validation")
            val_remaining -= 1
    return schedule


def _export_attempt(
    *,
    output_dir: Path,
    split: str,
    split_episode_index: int,
    global_episode_index: int,
    attempt_index: int,
    seed: int,
    yaw_delta: float,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
    fps: int,
    render_every: int,
    cube_axis_offset: float = CUBE_AXIS_OFFSET,
    cube_side_offset: float = CUBE_SIDE_OFFSET,
    cube_mass: float = CUBE_MASS,
    cube_friction: float = MAT_FRICTION,
    support_friction: float = MAT_FRICTION,
    pad_friction: float = PAD_FRICTION,
    candidate_metadata: dict[str, Any] | None = None,
    render_camera_profile: str = "full_robot",
    image_format: str = "bmp",
    refresh_model_constants: bool = False,
    lift_scale: float = 1.0,
    contact_solref: tuple[float, float] | None = None,
) -> dict[str, Any]:
    episode_path = output_dir / "splits" / split / "episodes" / f"episode_{split_episode_index:04d}.jsonl"
    final_frame_dir = output_dir / "splits" / split / "frames" / f"episode_{split_episode_index:04d}"
    attempt_frame_dir = output_dir / "_attempt_frames" / f"attempt_{attempt_index:04d}"
    shutil.rmtree(attempt_frame_dir, ignore_errors=True)
    attempt_frame_dir.mkdir(parents=True, exist_ok=True)
    scene_cache = output_dir / "scene_cache" / f"attempt_{attempt_index:04d}"
    pickup_qpos, lift_qpos = _candidate_qpos(yaw_delta)
    if float(lift_scale) <= 0.0:
        raise ValueError("lift_scale must be positive")
    lift_qpos = pickup_qpos + (lift_qpos - pickup_qpos) * float(lift_scale)
    candidate = {
        "yaw_delta_rad": float(yaw_delta),
        "cube_axis_offset_m": float(cube_axis_offset),
        "cube_side_offset_m": float(cube_side_offset),
        "cube_mass_kg": float(cube_mass),
        "cube_friction": float(cube_friction),
        "support_friction": float(support_friction),
        "pad_friction": float(pad_friction),
        **(candidate_metadata or {}),
    }
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
        _apply_candidate_physics(
            env,
            cube_mass=cube_mass,
            cube_friction=cube_friction,
            support_friction=support_friction,
            pad_friction=pad_friction,
            refresh_model_constants=refresh_model_constants,
            contact_solref=contact_solref,
        )
        env.model.opt.gravity[:] = WORLD_GRAVITY
        cube_pos, cube_quat = _initial_cube_pose_for_qpos(
            env,
            pickup_qpos,
            cube_axis_offset=cube_axis_offset,
            cube_side_offset=cube_side_offset,
        )
        env._set_gripper(command=START_COMMAND)
        _set_cube_pose(env, cube_pos, cube_quat)
        env._mujoco.mj_forward(env.model, env.data)
        initial_cube = env._cube_position()
        render_camera = _dataset_render_camera(env, initial_cube, render_camera_profile)
        placement_guard = _cube_mat_guard(initial_cube)
        if not placement_guard["passed"]:
            return _failed_attempt_summary(
                split=split,
                split_episode_index=split_episode_index,
                global_episode_index=global_episode_index,
                attempt_index=attempt_index,
                seed=seed,
                yaw_delta=yaw_delta,
                pickup_qpos=pickup_qpos,
                lift_qpos=lift_qpos,
                cube_pos=initial_cube,
                reason="cube_placement_guard_failed",
                placement_guard=placement_guard,
                candidate=candidate,
            )

        rows: list[dict[str, Any]] = []
        total_steps = APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS + LIFT_STEPS + POST_LIFT_HOLD_STEPS
        contact_stop_command: float | None = None
        for step_index in range(total_steps):
            arm, command, phase = _scripted_state_for_qpos(step_index, pickup_qpos=pickup_qpos, lift_qpos=lift_qpos)
            if contact_stop_command is not None and phase in {"close_on_cube_on_mat", "hold_before_lift", "lift_from_mat", "post_lift_hold"}:
                command = contact_stop_command
            if phase in {"approach_down_to_cube_on_mat", "close_on_cube_on_mat"}:
                env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
            else:
                env.model.opt.gravity[:] = WORLD_GRAVITY
            obs, reward, terminated, truncated, info = env.step([*tuple(float(x) for x in arm), float(command)])
            record = _record(env, step=step_index, phase=phase, command=float(command), initial_cube=initial_cube)
            if contact_stop_command is None and phase == "close_on_cube_on_mat" and int(record["pad_cube_contacted_pads"]) >= 2:
                contact_stop_command = max(-1.0, min(1.0, float(command)))
            image = ""
            if step_index % max(1, render_every) == 0:
                suffix = f".{image_format}"
                final_image_path = final_frame_dir / f"frame_{step_index:04d}{suffix}"
                attempt_image_path = attempt_frame_dir / f"frame_{step_index:04d}{suffix}"
                _write_dataset_image(attempt_image_path, _render_observation(env, render_camera), image_format)
                image = str(final_image_path.relative_to(output_dir))
            rows.append(
                {
                    "episode_index": global_episode_index,
                    "split": split,
                    "split_episode_index": split_episode_index,
                    "attempt_index": attempt_index,
                    "frame_index": step_index,
                    "timestamp": step_index / float(fps),
                    "phase": phase,
                    "task": TASK,
                    "observation": {"state": obs, "images": {"render": image} if image else {}},
                    "action": [*tuple(float(x) for x in arm), float(command)],
                    "reward": reward,
                    "done": bool(terminated or truncated),
                    "info": {
                        **_json_safe_info(info),
                        "ground_pickup": record,
                        "candidate": {
                            **candidate,
                            "pickup_qpos": [float(x) for x in pickup_qpos],
                            "lift_qpos": [float(x) for x in lift_qpos],
                        },
                    },
                }
            )
    finally:
        env.close()
        shutil.rmtree(scene_cache, ignore_errors=True)

    records = [row["info"]["ground_pickup"] for row in rows]
    lift_records = [record for record in records if record["phase"] == "lift_from_mat"]
    post_lift_hold_records = [record for record in records if record["phase"] == "post_lift_hold"]
    final = records[-1]
    max_penetration = max(float(record["pad_cube_contact_depth"]["max_penetration_m"]) for record in records)
    success = (
        _passes(records, lift_records, post_lift_hold_records, final)
        and max_penetration <= MAX_PAD_CUBE_PENETRATION_M
    )
    summary = _episode_summary(
        split=split,
        split_episode_index=split_episode_index,
        global_episode_index=global_episode_index,
        attempt_index=attempt_index,
        seed=seed,
        yaw_delta=yaw_delta,
        pickup_qpos=pickup_qpos,
        lift_qpos=lift_qpos,
        episode_path=episode_path,
        output_dir=output_dir,
        rows=rows,
        records=records,
        lift_records=lift_records,
        post_lift_hold_records=post_lift_hold_records,
        final=final,
        success=success,
        candidate=candidate,
    )
    if success:
        shutil.rmtree(final_frame_dir, ignore_errors=True)
        if attempt_frame_dir.exists():
            shutil.move(str(attempt_frame_dir), str(final_frame_dir))
        episode_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    else:
        shutil.rmtree(attempt_frame_dir, ignore_errors=True)
    return summary


def _camera_contract(profile: str, *, width: int, height: int) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    contract: dict[str, Any] = {
        "profile": profile,
        "resolution_hw": [int(height), int(width)],
    }
    if profile == "full_robot":
        return {
            **contract,
            "mode": "environment_default",
        }
    if profile == "ground_pickup_closeup":
        return {
            **contract,
            "mode": "free_camera",
            "target": "initial_cube_xyz_plus_[0,0,0.035]_m",
            "distance_m": 0.24,
            "azimuth_deg": 215.0,
            "elevation_deg": -10.0,
        }
    raise ValueError(f"unsupported render camera profile: {profile}")


def _write_dataset_image(path: Path, rgb: Any, image_format: str) -> None:
    if image_format == "bmp":
        _write_bmp(path, rgb)
        return
    if image_format != "png":
        raise ValueError(f"unsupported image format: {image_format}")

    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape {array.shape}")
    height, width, _ = array.shape
    rows = b"".join(b"\x00" + np.ascontiguousarray(row).tobytes() for row in array)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = signature + _png_chunk(b"IHDR", ihdr)
    payload += _png_chunk(b"IDAT", zlib.compress(rows, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _dataset_render_camera(env: Any, initial_cube: list[float], profile: str) -> Any | None:
    if profile == "full_robot":
        return None
    if profile != "ground_pickup_closeup":
        raise ValueError(f"unsupported render camera profile: {profile}")
    camera = env._mujoco.MjvCamera()
    camera.type = env._mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(initial_cube, dtype=float) + np.asarray([0.0, 0.0, 0.035])
    camera.distance = 0.24
    camera.azimuth = 215.0
    camera.elevation = -10.0
    return camera


def _render_observation(env: Any, camera: Any | None) -> Any:
    if camera is None:
        return env.render()
    if env._renderer is None:
        env._renderer = env._mujoco.Renderer(
            env.model,
            height=env.config.height,
            width=env.config.width,
        )
    env._renderer.update_scene(env.data, camera=camera)
    return env._renderer.render()


def _candidate_qpos(yaw_delta: float) -> tuple[np.ndarray, np.ndarray]:
    pickup = np.asarray(PICKUP_QPOS, dtype=float).copy()
    lift = np.asarray(LIFT_QPOS, dtype=float).copy()
    pickup[0] += float(yaw_delta)
    lift[0] += float(yaw_delta)
    return pickup, lift


def _initial_cube_pose_for_qpos(
    env: nexus.MyCobotNexusEnv,
    pickup_qpos: np.ndarray,
    *,
    cube_axis_offset: float = CUBE_AXIS_OFFSET,
    cube_side_offset: float = CUBE_SIDE_OFFSET,
) -> tuple[np.ndarray, list[float]]:
    nexus._set_adaptive_gate_arm_pose(env, tuple(float(x) for x in pickup_qpos))
    env._set_gripper(command=START_COMMAND)
    env._mujoco.mj_forward(env.model, env.data)
    left = _geom_pos(env, ROBOT_LEFT_PAD)
    right = _geom_pos(env, ROBOT_RIGHT_PAD)
    axis = right - left
    unit_axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    side_axis = np.cross(np.asarray([0.0, 0.0, 1.0]), unit_axis)
    side_axis = side_axis / max(float(np.linalg.norm(side_axis)), 1e-9)
    pos = (left + right) * 0.5
    xy_offset = float(cube_axis_offset) * unit_axis[:2] + float(cube_side_offset) * side_axis[:2]
    pos = np.asarray([pos[0] + xy_offset[0], pos[1] + xy_offset[1], WORK_MAT_TOP_Z + CUBE_HALF_SIZE], dtype=float)
    return pos, _quat_align_x_to_vector(axis)


def _apply_candidate_physics(
    env: nexus.MyCobotNexusEnv,
    *,
    cube_mass: float,
    cube_friction: float,
    support_friction: float,
    pad_friction: float,
    refresh_model_constants: bool = False,
    contact_solref: tuple[float, float] | None = None,
) -> None:
    mujoco = env._mujoco
    cube_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, nexus.TASK_CUBE_BODY)
    cube_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    if cube_body < 0 or cube_geom < 0:
        raise RuntimeError("randomized dataset scene is missing the required task cube body or geom")
    env.model.body_mass[cube_body] = float(cube_mass)
    inertia = (1.0 / 6.0) * float(cube_mass) * (2.0 * CUBE_HALF_SIZE) ** 2
    env.model.body_inertia[cube_body, :] = [inertia, inertia, inertia]
    env.model.geom_friction[cube_geom, :3] = _friction_triplet(cube_friction)
    if contact_solref is not None:
        env.model.geom_solref[cube_geom, :2] = _solref_pair(contact_solref)
    for name in ("nexus_work_mat", "nexus_floor"):
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"randomized dataset scene is missing required support geom {name!r}")
        env.model.geom_friction[geom_id, :3] = _friction_triplet(support_friction)
    for name in (ROBOT_LEFT_PAD, ROBOT_RIGHT_PAD):
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"randomized dataset scene is missing required fingertip pad {name!r}")
        env.model.geom_friction[geom_id, :3] = _friction_triplet(pad_friction)
        if contact_solref is not None:
            env.model.geom_solref[geom_id, :2] = _solref_pair(contact_solref)
    if refresh_model_constants:
        mujoco.mj_setConst(env.model, env.data)


def _solref_pair(values: tuple[float, float]) -> list[float]:
    if len(values) != 2 or float(values[0]) <= 0.0 or float(values[1]) <= 0.0:
        raise ValueError("contact_solref must contain two positive values")
    return [float(values[0]), float(values[1])]


def _friction_triplet(sliding: float) -> list[float]:
    value = float(sliding)
    if value <= 0.0:
        raise ValueError("friction values must be positive")
    return [value, value * 0.1, value * 0.1]


def _scripted_state_for_qpos(step: int, *, pickup_qpos: np.ndarray, lift_qpos: np.ndarray) -> tuple[np.ndarray, float, str]:
    if step < APPROACH_STEPS:
        return pickup_qpos.copy(), START_COMMAND, "approach_down_to_cube_on_mat"
    if step < APPROACH_STEPS + CLOSE_STEPS:
        alpha = (step - APPROACH_STEPS) / max(CLOSE_STEPS - 1, 1)
        return pickup_qpos.copy(), START_COMMAND + alpha * (CONTACT_COMMAND - START_COMMAND), "close_on_cube_on_mat"
    if step < APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS:
        return pickup_qpos.copy(), CONTACT_COMMAND, "hold_before_lift"
    if step < APPROACH_STEPS + CLOSE_STEPS + HOLD_STEPS + LIFT_STEPS:
        alpha = nexus._smoothstep((step - APPROACH_STEPS - CLOSE_STEPS - HOLD_STEPS) / max(LIFT_STEPS - 1, 1))
        return pickup_qpos + (lift_qpos - pickup_qpos) * alpha, CONTACT_COMMAND, "lift_from_mat"
    return lift_qpos.copy(), CONTACT_COMMAND, "post_lift_hold"


def _failed_attempt_summary(
    *,
    split: str,
    split_episode_index: int,
    global_episode_index: int,
    attempt_index: int,
    seed: int,
    yaw_delta: float,
    pickup_qpos: np.ndarray,
    lift_qpos: np.ndarray,
    cube_pos: Any,
    reason: str,
    placement_guard: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "split_episode_index": split_episode_index,
        "episode_index": global_episode_index,
        "attempt_index": attempt_index,
        "seed": seed,
        "yaw_delta_rad": float(yaw_delta),
        "pickup_qpos": [float(x) for x in pickup_qpos],
        "lift_qpos": [float(x) for x in lift_qpos],
        "initial_cube_xy": [float(x) for x in np.asarray(cube_pos, dtype=float)[:2]],
        "success": False,
        "reason": reason,
        "initial_cube_mat_guard": placement_guard,
        "candidate": candidate,
    }


def _episode_summary(
    *,
    split: str,
    split_episode_index: int,
    global_episode_index: int,
    attempt_index: int,
    seed: int,
    yaw_delta: float,
    pickup_qpos: np.ndarray,
    lift_qpos: np.ndarray,
    episode_path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    lift_records: list[dict[str, Any]],
    post_lift_hold_records: list[dict[str, Any]],
    final: dict[str, Any],
    success: bool,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    action_hash = _trajectory_hash([row["action"] for row in rows])
    initial_cube = np.asarray(records[0]["cube_pos"], dtype=float)
    return {
        "split": split,
        "split_episode_index": split_episode_index,
        "episode_index": global_episode_index,
        "attempt_index": attempt_index,
        "seed": seed,
        "yaw_delta_rad": float(yaw_delta),
        "pickup_qpos": [float(x) for x in pickup_qpos],
        "lift_qpos": [float(x) for x in lift_qpos],
        "initial_cube_xy": [float(x) for x in initial_cube[:2]],
        "path": str(episode_path.relative_to(output_dir)),
        "frames": len(rows),
        "rendered_frames": sum(1 for row in rows if row["observation"]["images"].get("render")),
        "success": success,
        "trajectory_hash": action_hash,
        "candidate": candidate,
        "first_frame_pad_cube_contacted_pads": records[0]["pad_cube_contacted_pads"],
        "first_contact_step": next((record["step"] for record in records if record["pad_cube_contacted_pads"] > 0), None),
        "initial_cube_mat_guard": records[0]["mat_guard"],
        "cube_bottom_on_or_above_mat_all_steps": all(bool(record["mat_guard"]["bottom_on_or_above_mat"]) for record in records),
        "worst_cube_bottom_minus_mat_top_m": min(float(record["mat_guard"]["cube_bottom_minus_mat_top_m"]) for record in records),
        "pad_mat_guard_passed_all_steps": all(bool(record["pad_mat_guard"]["passed"]) for record in records),
        "worst_pad_mat_penetration_m": min(float(record["pad_mat_guard"]["min_pad_bottom_minus_mat_top_m"]) for record in records),
        "gripper_visual_mat_guard_passed_all_steps": all(bool(record["gripper_visual_mat_guard"]["passed"]) for record in records),
        "worst_gripper_visual_penetration_m": min(float(record["gripper_visual_mat_guard"]["min_gripper_visual_bottom_minus_mat_top_m"]) for record in records),
        "final_cube_lift_m": final["cube_lift_m"],
        "final_gripper_cube_contact_pads": final["pad_cube_contacted_pads"],
        "lift_best_sustained_two_pad_steps": _best_sustained_two_pad(lift_records),
        "post_lift_hold_steps": POST_LIFT_HOLD_STEPS,
        "post_lift_hold_best_sustained_two_pad_steps": _best_sustained_two_pad(post_lift_hold_records),
        "post_lift_hold_min_cube_lift_m": min((float(record["cube_lift_m"]) for record in post_lift_hold_records), default=0.0),
        "max_pad_cube_penetration_m": max(float(record["pad_cube_contact_depth"]["max_penetration_m"]) for record in records),
        "max_lift_pad_cube_penetration_m": max((float(record["pad_cube_contact_depth"]["max_penetration_m"]) for record in lift_records), default=0.0),
    }


def _trajectory_hash(actions: list[list[float]]) -> str:
    rounded = [[round(float(value), 7) for value in action] for action in actions]
    payload = json.dumps(rounded, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rejection_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "final_cube_lift_m": summary.get("final_cube_lift_m"),
        "final_gripper_cube_contact_pads": summary.get("final_gripper_cube_contact_pads"),
        "lift_best_sustained_two_pad_steps": summary.get("lift_best_sustained_two_pad_steps"),
        "post_lift_hold_best_sustained_two_pad_steps": summary.get("post_lift_hold_best_sustained_two_pad_steps"),
        "post_lift_hold_min_cube_lift_m": summary.get("post_lift_hold_min_cube_lift_m"),
        "max_pad_cube_penetration_m": summary.get("max_pad_cube_penetration_m"),
    }
    return {
        "attempt_index": summary.get("attempt_index"),
        "split_target": summary.get("split"),
        "seed": summary.get("seed"),
        "yaw_delta_rad": summary.get("yaw_delta_rad"),
        "initial_cube_xy": summary.get("initial_cube_xy"),
        "candidate": summary.get("candidate"),
        "success": False,
        "reason": "success_criteria_failed",
        "checks": checks,
    }


def _aggregate_metrics(summaries: list[dict[str, Any]], rejected_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"passed_episodes": 0, "rejected_attempts": len(rejected_attempts), "pose_coverage": {}}
    xs = [float(summary["initial_cube_xy"][0]) for summary in summaries]
    ys = [float(summary["initial_cube_xy"][1]) for summary in summaries]
    yaws = [float(summary["yaw_delta_rad"]) for summary in summaries]
    hashes = [summary["trajectory_hash"] for summary in summaries]
    candidate_values = [summary.get("candidate", {}) for summary in summaries]
    return {
        "passed_episodes": len(summaries),
        "rejected_attempts": len(rejected_attempts),
        "min_final_cube_lift_m": min(float(summary["final_cube_lift_m"]) for summary in summaries),
        "min_lift_best_sustained_two_pad_steps": min(int(summary["lift_best_sustained_two_pad_steps"]) for summary in summaries),
        "min_post_lift_hold_sustained_two_pad_steps": min(int(summary["post_lift_hold_best_sustained_two_pad_steps"]) for summary in summaries),
        "min_post_lift_hold_cube_lift_m": min(float(summary["post_lift_hold_min_cube_lift_m"]) for summary in summaries),
        "max_pad_cube_penetration_m": max(float(summary["max_pad_cube_penetration_m"]) for summary in summaries),
        "max_lift_pad_cube_penetration_m": max(float(summary["max_lift_pad_cube_penetration_m"]) for summary in summaries),
        "pose_coverage": {
            "unique_pose_count": len({(round(x, 6), round(y, 6)) for x, y in zip(xs, ys)}),
            "x_min_m": min(xs),
            "x_max_m": max(xs),
            "x_span_m": max(xs) - min(xs),
            "y_min_m": min(ys),
            "y_max_m": max(ys),
            "y_span_m": max(ys) - min(ys),
            "yaw_min_rad": min(yaws),
            "yaw_max_rad": max(yaws),
            "yaw_span_rad": max(yaws) - min(yaws),
            "unique_trajectory_hashes": len(set(hashes)),
        },
        "factor_coverage": {
            "cube_mass_kg": _numeric_range(candidate_values, "cube_mass_kg"),
            "cube_friction": _numeric_range(candidate_values, "cube_friction"),
            "support_friction": _numeric_range(candidate_values, "support_friction"),
            "pad_friction": _numeric_range(candidate_values, "pad_friction"),
            "cube_axis_offset_m": _numeric_range(candidate_values, "cube_axis_offset_m"),
            "cube_side_offset_m": _numeric_range(candidate_values, "cube_side_offset_m"),
        },
    }


def _numeric_range(items: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
    if not values:
        return None
    return {"min": min(values), "max": max(values), "span": max(values) - min(values)}


if __name__ == "__main__":
    main()
