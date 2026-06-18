#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
from physical_ai_agent.sim.so101_live_viewer import _cartesian_error_controller_action
from train_so101_wrist_ego_picklift_policy import sweep_until_visible
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _current_qpos,
    _make_policy_renderers,
    _make_teacher_renderers,
    _restore_sim_state,
    _set_qpos,
    _snapshot_sim_state,
    make_high_contrast_picklift_env,
    make_teacher_targets,
    object_visible_to_teacher,
)


TASK = "Grasp the visible cube and lift it up."
SKILL_TASKS = {
    "pick_cube": TASK,
    "move_over_cube": "Move the gripper over the visible cube.",
    "pick_from_top_cube": "From above the visible cube, grasp it and lift it up.",
}
STATE_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export privileged SO101 teacher PickLift rollouts to a local LeRobotDataset."
    )
    parser.add_argument("--root", type=Path, default=Path("_workspace/so101_lerobot/teacher_picklift_smolvla"))
    parser.add_argument("--repo-id", default="physical-ai-agent/so101-picklift-teacher")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--teacher-style", choices=["legacy", "staged"], default="staged")
    parser.add_argument("--approach-steps", type=int, default=34)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--close-steps", type=int, default=42)
    parser.add_argument("--lift-steps", type=int, default=58)
    parser.add_argument("--start-mode", choices=["home", "near-gripper"], default="home")
    parser.add_argument("--near-gripper-joint-std", type=float, default=0.025)
    parser.add_argument(
        "--skill-mode",
        choices=sorted(SKILL_TASKS),
        default="pick_cube",
        help="Export the full pick skill or one of its agentic primitive segments.",
    )
    parser.add_argument(
        "--random-start-joint-std",
        type=float,
        default=0.55,
        help="Joint-space std used for move_over_cube random starts.",
    )
    parser.add_argument(
        "--move-success-tcp-dist",
        type=float,
        default=0.085,
        help="Max TCP-to-object distance accepted for move_over_cube success.",
    )
    parser.add_argument(
        "--move-target-z-offset",
        type=float,
        default=0.075,
        help="Approximate Cartesian z offset from grasp prepose for move_over_cube/pick_from_top starts.",
    )
    parser.add_argument(
        "--closed-gripper-prob",
        type=float,
        default=0.45,
        help="Probability that move_over_cube is exported with the gripper closed.",
    )
    parser.add_argument(
        "--move-gripper-profile",
        choices=["binary", "balanced", "closed"],
        default="balanced",
        help="Gripper qpos sampling for move_over_cube. balanced cycles through closed/intermediate/open gaps.",
    )
    parser.add_argument(
        "--move-min-actual-z",
        type=float,
        default=0.0,
        help="Reject move_over_cube episodes whose final TCP/cube world-z offset is smaller than this.",
    )
    parser.add_argument(
        "--pick-start-joint-std",
        type=float,
        default=0.035,
        help="Joint jitter around the elevated top pose for pick_from_top_cube starts.",
    )
    parser.add_argument(
        "--pick-correction-steps",
        type=int,
        default=18,
        help="Approach/correction steps from elevated top pose to grasp prepose before closing.",
    )
    parser.add_argument(
        "--pick-start-min-abs-y",
        type=float,
        default=0.018,
        help="Minimum absolute world-y offset between pick_from_top start TCP and cube.",
    )
    parser.add_argument(
        "--pick-start-max-abs-y",
        type=float,
        default=0.055,
        help="Maximum absolute world-y offset target between pick_from_top start TCP and cube.",
    )
    parser.add_argument(
        "--pick-start-min-actual-abs-y",
        type=float,
        default=0.015,
        help="Reject pick_from_top episodes whose actual start TCP/cube world-y offset is smaller than this.",
    )
    parser.add_argument(
        "--pick-start-min-actual-z",
        type=float,
        default=0.0,
        help="Reject pick_from_top episodes whose actual start TCP/cube world-z offset is smaller than this.",
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=8,
        help="Maximum candidate seeds to try, as episodes * multiplier.",
    )
    parser.add_argument("--no-camera3-duplicate", action="store_true")
    args = parser.parse_args()

    report = export_teacher_rollouts(
        root=args.root,
        repo_id=args.repo_id,
        episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        use_videos=args.use_videos,
        overwrite=args.overwrite,
        teacher_style=args.teacher_style,
        approach_steps=args.approach_steps,
        settle_steps=args.settle_steps,
        close_steps=args.close_steps,
        lift_steps=args.lift_steps,
        start_mode=args.start_mode,
        near_gripper_joint_std=args.near_gripper_joint_std,
        skill_mode=args.skill_mode,
        random_start_joint_std=args.random_start_joint_std,
        move_success_tcp_dist=args.move_success_tcp_dist,
        move_target_z_offset=args.move_target_z_offset,
        closed_gripper_prob=args.closed_gripper_prob,
        move_gripper_profile=args.move_gripper_profile,
        move_min_actual_z=args.move_min_actual_z,
        pick_start_joint_std=args.pick_start_joint_std,
        pick_correction_steps=args.pick_correction_steps,
        pick_start_min_abs_y=args.pick_start_min_abs_y,
        pick_start_max_abs_y=args.pick_start_max_abs_y,
        pick_start_min_actual_abs_y=args.pick_start_min_actual_abs_y,
        pick_start_min_actual_z=args.pick_start_min_actual_z,
        max_attempt_multiplier=args.max_attempt_multiplier,
        include_camera3_duplicate=not args.no_camera3_duplicate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def export_teacher_rollouts(
    *,
    root: Path,
    repo_id: str,
    episodes: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    use_videos: bool,
    overwrite: bool,
    teacher_style: str = "staged",
    approach_steps: int = 34,
    settle_steps: int = 10,
    close_steps: int = 42,
    lift_steps: int = 58,
    start_mode: str = "home",
    near_gripper_joint_std: float = 0.025,
    skill_mode: str = "pick_cube",
    random_start_joint_std: float = 0.55,
    move_success_tcp_dist: float = 0.085,
    move_target_z_offset: float = 0.075,
    closed_gripper_prob: float = 0.45,
    move_gripper_profile: str = "balanced",
    move_min_actual_z: float = 0.0,
    pick_start_joint_std: float = 0.035,
    pick_correction_steps: int = 18,
    pick_start_min_abs_y: float = 0.018,
    pick_start_max_abs_y: float = 0.055,
    pick_start_min_actual_abs_y: float = 0.015,
    pick_start_min_actual_z: float = 0.0,
    max_attempt_multiplier: int = 8,
    include_camera3_duplicate: bool = True,
) -> dict[str, Any]:
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite or choose a new --root")
        shutil.rmtree(root)

    features = _lerobot_features(
        height=height,
        width=width,
        use_videos=use_videos,
        include_camera3_duplicate=include_camera3_duplicate,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    if skill_mode not in SKILL_TASKS:
        raise ValueError(f"unknown skill_mode: {skill_mode}")
    task = SKILL_TASKS[skill_mode]

    config = WristEgoServoConfig(width=width, height=height)
    env = make_high_contrast_picklift_env()
    policy_renderers = _make_policy_renderers(env, config)
    teacher_renderers = _make_teacher_renderers(env, config)
    action_space_low = np.asarray(env.action_space.low, dtype=np.float32).copy()
    action_space_high = np.asarray(env.action_space.high, dtype=np.float32).copy()
    exported = 0
    attempted = 0
    skipped = []
    episode_summaries = []
    try:
        candidate_seed = seed
        while exported < episodes and attempted < episodes * max_attempt_multiplier:
            attempted += 1
            env.reset(seed=candidate_seed)
            candidate_seed += 1
            teacher_visible = object_visible_to_teacher(env, teacher_renderers, config=config)
            visible, search_steps = sweep_until_visible(env, policy_renderers, max_sweeps=config.max_sweeps)
            teacher_visible = teacher_visible or object_visible_to_teacher(env, teacher_renderers, config=config)
            if not visible:
                skipped.append({"seed": candidate_seed - 1, "reason": "not_visible_after_sweep"})
                continue
            candidates = make_teacher_targets(env)
            if skill_mode in {"move_over_cube", "pick_from_top_cube"}:
                candidates = [
                    candidate
                    for candidate in candidates
                    if str(candidate["meta"].get("mode")) == "overhead"
                ]
            if not candidates:
                skipped.append({"seed": candidate_seed - 1, "reason": "no_successful_teacher_candidate"})
                continue
            best = max(candidates, key=lambda item: float(item["meta"].get("score", -1e9)))
            summary = _write_teacher_episode(
                dataset=dataset,
                env=env,
                renderers=policy_renderers,
                q_open=np.asarray(best["q_open"], dtype=np.float32),
                q_lift=np.asarray(best["q_lift"], dtype=np.float32),
                seed=candidate_seed - 1,
                search_steps=search_steps,
                teacher_visible=teacher_visible,
                best_meta=dict(best["meta"]),
                teacher_style=teacher_style,
                approach_steps=approach_steps,
                settle_steps=settle_steps,
                close_steps=close_steps,
                lift_steps=lift_steps,
                start_mode=start_mode,
                near_gripper_joint_std=near_gripper_joint_std,
                skill_mode=skill_mode,
                task=task,
                episode_index=exported,
                random_start_joint_std=random_start_joint_std,
                move_success_tcp_dist=move_success_tcp_dist,
                move_target_z_offset=move_target_z_offset,
                closed_gripper_prob=closed_gripper_prob,
                move_gripper_profile=move_gripper_profile,
                move_min_actual_z=move_min_actual_z,
                pick_start_joint_std=pick_start_joint_std,
                pick_correction_steps=pick_correction_steps,
                pick_start_min_abs_y=pick_start_min_abs_y,
                pick_start_max_abs_y=pick_start_max_abs_y,
                pick_start_min_actual_abs_y=pick_start_min_actual_abs_y,
                pick_start_min_actual_z=pick_start_min_actual_z,
                include_camera3_duplicate=include_camera3_duplicate,
            )
            if summary["success"]:
                dataset.save_episode()
                exported += 1
                episode_summaries.append(summary)
                print(
                    f"[so101-lerobot] exported {exported}/{episodes} "
                    f"seed={summary['seed']} frames={summary['frames']} "
                    f"mode={summary['best_meta'].get('mode')}",
                    flush=True,
                )
            else:
                dataset.clear_episode_buffer()
                skipped.append({"seed": candidate_seed - 1, "reason": "teacher_replay_failed", **summary})
    finally:
        for renderer in [*policy_renderers.values(), *teacher_renderers.values()]:
            renderer.close()
        env.close()

    dataset.finalize()
    audit = audit_lerobot_dataset(
        root=root,
        repo_id=repo_id,
        features=features,
        action_space_low=action_space_low,
        action_space_high=action_space_high,
    )
    report = {
        "operation": "export_so101_teacher_rollouts_lerobot",
        "root": str(root),
        "repo_id": repo_id,
        "task": task,
        "skill_mode": skill_mode,
        "requested_episodes": episodes,
        "exported_episodes": exported,
        "attempted_seeds": attempted,
        "fps": fps,
        "use_videos": use_videos,
        "teacher_style": teacher_style,
        "teacher_timing": {
            "approach_steps": int(approach_steps),
            "settle_steps": int(settle_steps),
            "close_steps": int(close_steps),
            "lift_steps": int(lift_steps),
            "start_mode": str(start_mode),
            "near_gripper_joint_std": float(near_gripper_joint_std),
            "random_start_joint_std": float(random_start_joint_std),
            "move_success_tcp_dist": float(move_success_tcp_dist),
            "move_target_z_offset": float(move_target_z_offset),
            "closed_gripper_prob": float(closed_gripper_prob),
            "move_gripper_profile": str(move_gripper_profile),
            "move_min_actual_z": float(move_min_actual_z),
            "pick_start_joint_std": float(pick_start_joint_std),
            "pick_correction_steps": int(pick_correction_steps),
            "pick_start_min_abs_y": float(pick_start_min_abs_y),
            "pick_start_max_abs_y": float(pick_start_max_abs_y),
            "pick_start_min_actual_abs_y": float(pick_start_min_actual_abs_y),
            "pick_start_min_actual_z": float(pick_start_min_actual_z),
        },
        "camera3_duplicate": {
            "enabled": bool(include_camera3_duplicate),
            "source": "wrist_cam",
            "reason": "lerobot/smolvla_base expects camera2 to carry the eye-in-hand/wrist view; camera3 duplicates camera2 when requested.",
        },
        "feature_mapping": {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
            **({"observation.images.camera3": "wrist_cam duplicate"} if include_camera3_duplicate else {}),
            "observation.state": "SO101 qpos/control state",
            "action": "SO101 qpos target action",
            "task": task,
        },
        "official_camera_contract": {
            "dataset": "SO101 egocentric+wrist visual-student dataset aligned to the local real-hardware policy cameras",
            "dataset_features": ["observation.images.egocentric_cam", "observation.images.wrist_cam"],
            "rename_map": {
                "observation.images.egocentric_cam": "observation.images.camera1",
                "observation.images.wrist_cam": "observation.images.camera2",
            },
            "local_verification": "Student inputs use egocentric_cam and wrist_cam; top_down is debug-only and must not be fed to SmolVLA.",
        },
        "action_normalization": {
            "producer": "raw SO101 qpos target in simulator action-space units",
            "expected_smolvla_mode": "MEAN_STD from LeRobotDataset stats",
            "manual_scaling_applied": False,
        },
        "episodes": episode_summaries,
        "skipped": skipped,
        "audit": audit,
    }
    report_path = root / "so101_lerobot_export_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _write_teacher_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    q_lift: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    teacher_style: str,
    approach_steps: int,
    settle_steps: int,
    close_steps: int,
    lift_steps: int,
    start_mode: str,
    near_gripper_joint_std: float,
    skill_mode: str,
    task: str,
    episode_index: int,
    random_start_joint_std: float,
    move_success_tcp_dist: float,
    move_target_z_offset: float,
    closed_gripper_prob: float,
    move_gripper_profile: str,
    move_min_actual_z: float,
    pick_start_joint_std: float,
    pick_correction_steps: int,
    pick_start_min_abs_y: float,
    pick_start_max_abs_y: float,
    pick_start_min_actual_abs_y: float,
    pick_start_min_actual_z: float,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    if teacher_style == "legacy":
        return _write_legacy_teacher_episode(
            dataset=dataset,
            env=env,
            renderers=renderers,
            q_open=q_open,
            q_lift=q_lift,
            seed=seed,
            search_steps=search_steps,
            teacher_visible=teacher_visible,
            best_meta=best_meta,
            task=task,
            include_camera3_duplicate=include_camera3_duplicate,
        )

    if skill_mode == "move_over_cube":
        return _write_move_over_cube_episode(
            dataset=dataset,
            env=env,
            renderers=renderers,
            q_open=q_open,
            seed=seed,
            search_steps=search_steps,
            teacher_visible=teacher_visible,
            best_meta=best_meta,
            approach_steps=approach_steps,
            settle_steps=settle_steps,
            episode_index=episode_index,
            random_start_joint_std=random_start_joint_std,
            move_success_tcp_dist=move_success_tcp_dist,
            move_target_z_offset=move_target_z_offset,
            closed_gripper_prob=closed_gripper_prob,
            move_gripper_profile=move_gripper_profile,
            move_min_actual_z=move_min_actual_z,
            task=task,
            include_camera3_duplicate=include_camera3_duplicate,
        )

    if skill_mode == "pick_from_top_cube":
        return _write_pick_from_top_cube_episode(
            dataset=dataset,
            env=env,
            renderers=renderers,
            q_open=q_open,
            seed=seed,
            search_steps=search_steps,
            teacher_visible=teacher_visible,
            best_meta=best_meta,
            close_steps=close_steps,
            lift_steps=lift_steps,
            episode_index=episode_index,
            move_target_z_offset=move_target_z_offset,
            pick_start_joint_std=pick_start_joint_std,
            pick_correction_steps=pick_correction_steps,
            pick_start_min_abs_y=pick_start_min_abs_y,
            pick_start_max_abs_y=pick_start_max_abs_y,
            pick_start_min_actual_abs_y=pick_start_min_actual_abs_y,
            pick_start_min_actual_z=pick_start_min_actual_z,
            task=task,
            include_camera3_duplicate=include_camera3_duplicate,
        )

    return _write_staged_teacher_episode(
        dataset=dataset,
        env=env,
        renderers=renderers,
        q_open=q_open,
        seed=seed,
        search_steps=search_steps,
        teacher_visible=teacher_visible,
        best_meta=best_meta,
        approach_steps=approach_steps,
        settle_steps=settle_steps,
        close_steps=close_steps,
        lift_steps=lift_steps,
        start_mode=start_mode,
        near_gripper_joint_std=near_gripper_joint_std,
        task=task,
        include_camera3_duplicate=include_camera3_duplicate,
    )


def _write_legacy_teacher_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    q_lift: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    info: dict[str, Any] = {}
    frames = 0
    success_step = None
    for step in range(180):
        if step < 58:
            action = q_open.copy()
        elif step < 118:
            action = q_close.copy()
        else:
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=np.float32)
            action[-1] = q_close[-1]
            q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=np.asarray(action, dtype=np.float32),
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        if bool(info.get("success", False)) and success_step is None:
            success_step = step + 1
        if bool(info.get("success", False)) or terminated or truncated:
            break
    return {
        "seed": seed,
        "frames": frames,
        "success": bool(info.get("success", False)),
        "success_step": success_step,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "q_lift": [float(value) for value in q_lift],
        "teacher_style": "legacy",
        "phase_counts": {"approach": 58, "close": 60, "lift": max(0, frames - 118)},
    }


def _write_staged_teacher_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    approach_steps: int,
    settle_steps: int,
    close_steps: int,
    lift_steps: int,
    start_mode: str,
    near_gripper_joint_std: float,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_start = _current_qpos(env).astype(np.float32)
    q_start[-1] = _open_gripper_value(env)
    q_open = np.clip(q_open.astype(np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    if start_mode == "near-gripper":
        q_start = _make_near_gripper_qpos(
            env,
            q_open,
            seed=seed,
            joint_std=near_gripper_joint_std,
        )
        _set_qpos(env, q_start)
    elif start_mode != "home":
        raise ValueError(f"unknown start_mode: {start_mode}")
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    success_step = None
    q_lift = q_close.copy()
    phase_counts = {"approach": 0, "settle": 0, "close": 0, "lift": 0}
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None

    def add_step(action: np.ndarray, phase: str) -> bool:
        nonlocal frames, info, success_step, previous_action
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=action,
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
        previous_action = action.copy()
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        if bool(info.get("success", False)) and success_step is None:
            success_step = frames
        return bool(info.get("success", False)) or bool(terminated) or bool(truncated)

    approach_steps = max(1, int(approach_steps))
    for index in range(approach_steps):
        alpha = (index + 1) / float(approach_steps)
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        action = (1.0 - alpha) * q_start + alpha * q_open
        action[-1] = _open_gripper_value(env)
        if add_step(action, "approach"):
            break

    if not bool(info.get("success", False)):
        for _ in range(max(0, int(settle_steps))):
            if add_step(q_open, "settle"):
                break

    if not bool(info.get("success", False)):
        for _ in range(max(1, int(close_steps))):
            if add_step(q_close, "close"):
                break

    if not bool(info.get("success", False)):
        for _ in range(max(1, int(lift_steps))):
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=np.float32)
            action[-1] = q_close[-1]
            q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            if add_step(q_lift, "lift"):
                break

    return {
        "seed": seed,
        "frames": frames,
        "success": bool(info.get("success", False)),
        "success_step": success_step,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "q_start": [float(value) for value in q_start],
        "q_open": [float(value) for value in q_open],
        "q_lift": [float(value) for value in q_lift],
        "teacher_style": "staged",
        "start_mode": start_mode,
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _write_move_over_cube_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    approach_steps: int,
    settle_steps: int,
    episode_index: int,
    random_start_joint_std: float,
    move_success_tcp_dist: float,
    move_target_z_offset: float,
    closed_gripper_prob: float,
    move_gripper_profile: str,
    move_min_actual_z: float,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_open = np.clip(q_open.astype(np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    q_above = _offset_qpos_by_cartesian(env, q_open, np.asarray([0.0, 0.0, float(move_target_z_offset)]))
    rng = np.random.default_rng(int(seed) + 4242)
    gripper_value = _sample_move_gripper_value(
        env,
        rng=rng,
        episode_index=episode_index,
        profile=move_gripper_profile,
        closed_gripper_prob=closed_gripper_prob,
    )
    q_above[-1] = gripper_value
    q_above_delta = _tcp_to_object_delta_for_qpos(env, q_above)
    q_above_z_offset = float(q_above_delta[2])
    if q_above_z_offset < float(move_min_actual_z):
        return {
            "seed": seed,
            "frames": 0,
            "success": False,
            "success_step": None,
            "search_steps": search_steps,
            "teacher_visible_in_any_camera": bool(teacher_visible),
            "best_meta": best_meta,
            "final_info": dict(env.unwrapped._get_info()),
            "q_start": [float(value) for value in q_above],
            "q_above": [float(value) for value in q_above],
            "q_open": [float(value) for value in q_open],
            "q_above_tcp_to_obj_delta": [float(value) for value in q_above_delta],
            "q_above_z_offset": q_above_z_offset,
            "move_min_actual_z": float(move_min_actual_z),
            "gripper_value": float(gripper_value),
            "gripper_closed": bool(gripper_value <= float(env.action_space.low[-1]) + 1e-5),
            "gripper_profile": str(move_gripper_profile),
            "gripper_bucket": None,
            "teacher_style": "staged_skill",
            "skill_mode": "move_over_cube",
            "phase_counts": {"move": 0, "settle": 0},
            "mean_action_delta": 0.0,
            "max_action_delta": 0.0,
        }
    q_start = _make_random_start_qpos(env, q_above, seed=seed, joint_std=random_start_joint_std)
    q_start[-1] = gripper_value
    _set_qpos(env, q_start)
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    phase_counts = {"move": 0, "settle": 0}
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None

    def add_step(action: np.ndarray, phase: str) -> None:
        nonlocal frames, info, previous_action
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        action[-1] = gripper_value
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=action,
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
        previous_action = action.copy()
        _obs, _reward, _terminated, _truncated, info = env.step(np.asarray(action, dtype=float))

    approach_steps = max(1, int(approach_steps))
    for index in range(approach_steps):
        alpha = (index + 1) / float(approach_steps)
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        add_step((1.0 - alpha) * q_start + alpha * q_above, "move")
    for _ in range(max(0, int(settle_steps))):
        add_step(q_above, "settle")

    tcp_to_obj_dist = float(info.get("tcp_to_obj_dist", 1.0))
    final_tcp_to_obj_delta = _tcp_to_object_delta(env)
    final_z_offset = float(final_tcp_to_obj_delta[2])
    success = tcp_to_obj_dist <= float(move_success_tcp_dist) and final_z_offset >= float(move_min_actual_z)
    return {
        "seed": seed,
        "frames": frames,
        "success": success,
        "success_step": frames if success else None,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": tcp_to_obj_dist,
        },
        "q_start": [float(value) for value in q_start],
        "q_above": [float(value) for value in q_above],
        "q_open": [float(value) for value in q_open],
        "q_above_tcp_to_obj_delta": [float(value) for value in q_above_delta],
        "q_above_z_offset": q_above_z_offset,
        "final_tcp_to_obj_delta": [float(value) for value in final_tcp_to_obj_delta],
        "final_z_offset": final_z_offset,
        "move_min_actual_z": float(move_min_actual_z),
        "gripper_value": float(gripper_value),
        "gripper_closed": bool(gripper_value <= float(env.action_space.low[-1]) + 1e-5),
        "gripper_profile": str(move_gripper_profile),
        "gripper_bucket": int(episode_index % 5) if move_gripper_profile == "balanced" else None,
        "teacher_style": "staged_skill",
        "skill_mode": "move_over_cube",
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _write_pick_from_top_cube_episode(
    *,
    dataset: Any,
    env: Any,
    renderers: dict[str, Any],
    q_open: np.ndarray,
    seed: int,
    search_steps: int,
    teacher_visible: bool,
    best_meta: dict[str, Any],
    close_steps: int,
    lift_steps: int,
    episode_index: int,
    move_target_z_offset: float,
    pick_start_joint_std: float,
    pick_correction_steps: int,
    pick_start_min_abs_y: float,
    pick_start_max_abs_y: float,
    pick_start_min_actual_abs_y: float,
    pick_start_min_actual_z: float,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    q_open = np.clip(q_open.astype(np.float32), env.action_space.low, env.action_space.high)
    q_open[-1] = _open_gripper_value(env)
    q_above = _offset_qpos_by_cartesian(env, q_open, np.asarray([0.0, 0.0, float(move_target_z_offset)]))
    q_start = _make_near_gripper_qpos(env, q_above, seed=seed + 313, joint_std=pick_start_joint_std)
    q_start, start_target_y_offset = _balance_pick_start_y_offset(
        env,
        q_start,
        episode_index=episode_index,
        min_abs_y=pick_start_min_abs_y,
        max_abs_y=pick_start_max_abs_y,
    )
    q_start[-1] = float(env.action_space.low[-1])
    _set_qpos(env, q_start)
    start_tcp_to_obj_delta = _tcp_to_object_delta(env)
    start_abs_y_offset = float(abs(start_tcp_to_obj_delta[1]))
    start_z_offset = float(start_tcp_to_obj_delta[2])
    if start_abs_y_offset < float(pick_start_min_actual_abs_y) or start_z_offset < float(pick_start_min_actual_z):
        return {
            "seed": seed,
            "frames": 0,
            "success": False,
            "success_step": None,
            "search_steps": search_steps,
            "teacher_visible_in_any_camera": bool(teacher_visible),
            "best_meta": best_meta,
            "final_info": dict(env.unwrapped._get_info()),
            "q_start": [float(value) for value in q_start],
            "q_above": [float(value) for value in q_above],
            "q_open": [float(value) for value in q_open],
            "q_lift": [float(value) for value in q_open],
            "start_target_y_offset": float(start_target_y_offset),
            "start_tcp_to_obj_delta": [float(value) for value in start_tcp_to_obj_delta],
            "start_abs_y_offset": start_abs_y_offset,
            "start_z_offset": start_z_offset,
            "pick_start_min_actual_abs_y": float(pick_start_min_actual_abs_y),
            "pick_start_min_actual_z": float(pick_start_min_actual_z),
            "teacher_style": "staged_skill",
            "skill_mode": "pick_from_top_cube",
            "phase_counts": {"correct": 0, "close": 0, "lift": 0},
            "mean_action_delta": 0.0,
            "max_action_delta": 0.0,
        }
    q_close = q_open.copy()
    q_close[-1] = float(env.action_space.low[-1])
    info: dict[str, Any] = env.unwrapped._get_info()
    frames = 0
    success_step = None
    q_lift = q_close.copy()
    phase_counts = {"correct": 0, "close": 0, "lift": 0}
    action_deltas: list[float] = []
    previous_action: np.ndarray | None = None

    def add_step(action: np.ndarray, phase: str) -> bool:
        nonlocal frames, info, success_step, previous_action
        action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
        dataset.add_frame(
            _make_lerobot_frame(
                env=env,
                renderers=renderers,
                action=action,
                task=task,
                include_camera3_duplicate=include_camera3_duplicate,
            )
        )
        frames += 1
        phase_counts[phase] += 1
        if previous_action is not None:
            action_deltas.append(float(np.linalg.norm(action[:5] - previous_action[:5])))
        previous_action = action.copy()
        _obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
        if bool(info.get("success", False)) and success_step is None:
            success_step = frames
        return bool(info.get("success", False)) or bool(terminated) or bool(truncated)

    correction_steps = max(0, int(pick_correction_steps))
    for index in range(correction_steps):
        alpha = (index + 1) / float(max(1, correction_steps))
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * alpha))
        action = (1.0 - alpha) * q_start + alpha * q_open
        action[-1] = _open_gripper_value(env)
        if add_step(action, "correct"):
            break

    if not bool(info.get("success", False)):
        for _ in range(max(1, int(close_steps))):
            if add_step(q_close, "close"):
                break
    if not bool(info.get("success", False)):
        for _ in range(max(1, int(lift_steps))):
            action = np.asarray(_cartesian_error_controller_action(env, np.asarray([0.0, 0.0, 0.12])), dtype=np.float32)
            action[-1] = q_close[-1]
            q_lift = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            if add_step(q_lift, "lift"):
                break

    return {
        "seed": seed,
        "frames": frames,
        "success": bool(info.get("success", False)),
        "success_step": success_step,
        "search_steps": search_steps,
        "teacher_visible_in_any_camera": bool(teacher_visible),
        "best_meta": best_meta,
        "final_info": {
            "is_grasped": bool(info.get("is_grasped", False)),
            "lift_height": float(info.get("lift_height", 0.0)),
            "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        },
        "q_start": [float(value) for value in q_start],
        "q_above": [float(value) for value in q_above],
        "q_open": [float(value) for value in q_open],
        "q_lift": [float(value) for value in q_lift],
        "start_target_y_offset": float(start_target_y_offset),
        "start_tcp_to_obj_delta": [float(value) for value in start_tcp_to_obj_delta],
        "start_abs_y_offset": start_abs_y_offset,
        "start_z_offset": start_z_offset,
        "pick_start_min_actual_abs_y": float(pick_start_min_actual_abs_y),
        "pick_start_min_actual_z": float(pick_start_min_actual_z),
        "teacher_style": "staged_skill",
        "skill_mode": "pick_from_top_cube",
        "phase_counts": phase_counts,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "max_action_delta": float(np.max(action_deltas)) if action_deltas else 0.0,
    }


def _open_gripper_value(env: Any) -> float:
    return float(env.action_space.high[-1])


def _sample_move_gripper_value(
    env: Any,
    *,
    rng: np.random.Generator,
    episode_index: int,
    profile: str,
    closed_gripper_prob: float,
) -> float:
    low = float(env.action_space.low[-1])
    high = _open_gripper_value(env)
    if profile == "closed":
        return low
    if profile == "balanced":
        buckets = np.linspace(low, high, num=5, dtype=np.float32)
        return float(buckets[int(episode_index) % len(buckets)])
    return low if rng.random() < float(np.clip(closed_gripper_prob, 0.0, 1.0)) else high


def _balance_pick_start_y_offset(
    env: Any,
    qpos: np.ndarray,
    *,
    episode_index: int,
    min_abs_y: float,
    max_abs_y: float,
) -> tuple[np.ndarray, float]:
    low = max(0.0, float(min_abs_y))
    high = max(low, float(max_abs_y))
    buckets = np.linspace(low, high, num=5, dtype=np.float32)
    target_abs_y = float(buckets[int(episode_index) % len(buckets)])
    sign = -1.0 if ((int(episode_index) // len(buckets)) % 2) else 1.0
    target_y_offset = sign * target_abs_y
    snapshot = _snapshot_sim_state(env)
    try:
        qpos = np.clip(np.asarray(qpos, dtype=np.float32), env.action_space.low, env.action_space.high)
        _set_qpos(env, qpos)
        current_y_offset = float(_tcp_to_object_delta(env)[1])
        adjusted = _offset_qpos_by_cartesian(env, qpos, np.asarray([0.0, target_y_offset - current_y_offset, 0.0]))
        adjusted[-1] = _open_gripper_value(env)
        return adjusted, target_y_offset
    finally:
        _restore_sim_state(env, snapshot)


def _tcp_to_object_delta(env: Any) -> np.ndarray:
    model = env.unwrapped.model
    data = env.unwrapped.data
    site_id = model.site("gripperframe").id
    obj_geom_id = int(env.unwrapped._obj_geom_id)
    return np.asarray(data.site_xpos[site_id], dtype=float) - np.asarray(data.geom_xpos[obj_geom_id], dtype=float)


def _tcp_to_object_delta_for_qpos(env: Any, qpos: np.ndarray) -> np.ndarray:
    snapshot = _snapshot_sim_state(env)
    try:
        _set_qpos(env, np.clip(np.asarray(qpos, dtype=np.float32), env.action_space.low, env.action_space.high))
        return _tcp_to_object_delta(env)
    finally:
        _restore_sim_state(env, snapshot)


def _make_near_gripper_qpos(env: Any, q_open: np.ndarray, *, seed: int, joint_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 9117)
    target = np.asarray(q_open, dtype=np.float32).copy()
    jitter = rng.normal(0.0, max(0.0, float(joint_std)), size=target.shape).astype(np.float32)
    if jitter.shape[0] >= 6:
        jitter[-1] = 0.0
    target = target + jitter
    target[-1] = _open_gripper_value(env)
    return np.clip(target, env.action_space.low, env.action_space.high).astype(np.float32)


def _offset_qpos_by_cartesian(env: Any, qpos: np.ndarray, offset: np.ndarray, *, steps: int = 10) -> np.ndarray:
    snapshot = _snapshot_sim_state(env)
    try:
        target = np.clip(np.asarray(qpos, dtype=np.float32), env.action_space.low, env.action_space.high)
        gripper_value = float(target[-1])
        _set_qpos(env, target)
        per_step_offset = np.asarray(offset, dtype=float) / float(max(1, int(steps)))
        action = target.copy()
        for _ in range(max(1, int(steps))):
            action = np.asarray(_cartesian_error_controller_action(env, per_step_offset), dtype=np.float32)
            action[-1] = gripper_value
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            _obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
            if terminated or truncated:
                break
        result = _current_qpos(env).astype(np.float32)
        result[-1] = gripper_value
        return np.clip(result, env.action_space.low, env.action_space.high).astype(np.float32)
    finally:
        _restore_sim_state(env, snapshot)


def _make_random_start_qpos(env: Any, q_open: np.ndarray, *, seed: int, joint_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 27183)
    home = _current_qpos(env).astype(np.float32)
    home[-1] = _open_gripper_value(env)
    jitter = rng.normal(0.0, max(0.0, float(joint_std)), size=home.shape).astype(np.float32)
    jitter[-1] = 0.0
    # Blend a home-relative random pose with the target so starts are varied but still reachable.
    target = 0.65 * (home + jitter) + 0.35 * np.asarray(q_open, dtype=np.float32)
    target[-1] = _open_gripper_value(env)
    return np.clip(target, env.action_space.low, env.action_space.high).astype(np.float32)


def _make_lerobot_frame(
    *,
    env: Any,
    renderers: dict[str, Any],
    action: np.ndarray,
    task: str,
    include_camera3_duplicate: bool,
) -> dict[str, Any]:
    wrist = _render_camera(env, renderers["wrist_cam"], "wrist_cam")
    ego = _render_camera(env, renderers["egocentric_cam"], "egocentric_cam")
    frame = {
        "observation.images.camera1": ego,
        "observation.images.camera2": wrist,
        "observation.state": _current_qpos(env).astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    if include_camera3_duplicate:
        frame["observation.images.camera3"] = wrist.copy()
    return frame


def _render_camera(env: Any, renderer: Any, camera_name: str) -> np.ndarray:
    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
    return postprocess_camera_frame(camera_name, renderer.render()).astype(np.uint8)


def _lerobot_features(
    *,
    height: int,
    width: int,
    use_videos: bool,
    include_camera3_duplicate: bool,
) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"
    image_feature = {
        "dtype": image_dtype,
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    features = {
        "observation.images.camera1": dict(image_feature),
        "observation.images.camera2": dict(image_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": STATE_NAMES,
        },
    }
    if include_camera3_duplicate:
        features["observation.images.camera3"] = dict(image_feature)
    return features


def audit_lerobot_dataset(
    *,
    root: Path,
    repo_id: str,
    features: dict[str, dict[str, Any]],
    action_space_low: np.ndarray,
    action_space_high: np.ndarray,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    sample = dataset[0]
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    action_values = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    state_values = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    action_low = action_values.min(axis=0)
    action_high = action_values.max(axis=0)
    state_low = state_values.min(axis=0)
    state_high = state_values.max(axis=0)
    required_keys = {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "action",
        "task",
    }
    if "observation.images.camera3" in features:
        required_keys.add("observation.images.camera3")
    sample_keys = set(sample.keys())
    audit = {
        "status": "passed",
        "dataset_len": len(dataset),
        "num_episodes": dataset.num_episodes,
        "fps": dataset.fps,
        "features": dataset.features,
        "sample_keys": sorted(sample_keys),
        "missing_required_keys": sorted(required_keys - sample_keys),
        "sample_shapes": {
            key: list(np.asarray(sample[key]).shape)
            for key in required_keys
            if key in sample and key != "task"
        },
        "task_sample": sample.get("task"),
        "stats_path": str(stats_path),
        "stats_keys": sorted(stats.keys()),
        "action_min": [float(value) for value in action_low],
        "action_max": [float(value) for value in action_high],
        "action_space_low": [float(value) for value in action_space_low],
        "action_space_high": [float(value) for value in action_space_high],
        "action_within_space": bool(
            np.all(action_low >= action_space_low - 1e-5)
            and np.all(action_high <= action_space_high + 1e-5)
        ),
        "state_min": [float(value) for value in state_low],
        "state_max": [float(value) for value in state_high],
        "state_within_space": bool(
            np.all(state_low >= action_space_low - 1e-5)
            and np.all(state_high <= action_space_high + 1e-5)
        ),
        "declared_features": features,
    }
    if audit["missing_required_keys"]:
        audit["status"] = "failed"
    if list(action_values.shape[1:]) != [6] or list(state_values.shape[1:]) != [6]:
        audit["status"] = "failed"
    if "action" not in stats or "observation.state" not in stats:
        audit["status"] = "failed"
    if not audit["action_within_space"] or not audit["state_within_space"]:
        audit["status"] = "failed"
    audit_path = root / "so101_lerobot_audit.json"
    audit["audit_path"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


if __name__ == "__main__":
    main()
