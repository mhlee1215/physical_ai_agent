#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from physical_ai_agent.policies.smolvla_real import (
    _build_batch_for_policy,
    _load_pretrained_policy,
    _policy_device_metadata,
    _tensor_to_float_list,
)
from physical_ai_agent.policies.so101_valid_mask import (
    execution_horizon_from_valid_probs,
    load_valid_mask_head,
)
from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
from train_so101_wrist_ego_picklift_policy import sweep_until_visible
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _current_qpos,
    _make_policy_renderers,
    _restore_sim_state,
    _set_qpos,
    make_teacher_targets,
    make_high_contrast_picklift_env,
)
from export_so101_teacher_rollouts_lerobot import (
    _balance_pick_start_y_offset,
    _make_near_gripper_qpos,
    _offset_qpos_by_cartesian,
    _tcp_to_object_delta,
)
from export_so101_pickplace_teacher_rollouts_lerobot import _make_pickplace_env


TASK = "Grasp the visible cube and lift it up."
PICK_FROM_TOP_TASK = "From above the visible cube, grasp it and lift it up."
PICK_AND_PLACE_TASK = "Pick up the small red cube and place it on the blue circle."
POLICY_DISPLAY_IMAGE_FEATURE_MAPPING = {
    "observation.images.camera1": "egocentric_cam",
    "observation.images.camera2": "wrist_cam",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a SmolVLA policy path in the SO101 PickLift simulator."
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/so101_smolvla_eval/picklift"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=79000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--action-alpha", type=float, default=1.0)
    parser.add_argument("--max-arm-delta", type=float, default=0.0)
    parser.add_argument("--max-gripper-delta", type=float, default=0.0)
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-num-steps", type=int, default=None)
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--env-object-color", default=None)
    parser.add_argument(
        "--start-report-path",
        type=Path,
        help="LeRobot export report whose episode sim_snapshot entries define closed-loop start states.",
    )
    parser.add_argument(
        "--eval-skill-mode",
        choices=["picklift", "pick_from_top_cube", "pick_and_place_cube", "grip_the_cube_v1"],
        default="picklift",
    )
    parser.add_argument("--pick-start-min-actual-z", type=float, default=0.05)
    parser.add_argument("--pick-start-min-actual-abs-y", type=float, default=0.015)
    parser.add_argument("--pick-start-max-actual-abs-y", type=float, default=0.065)
    parser.add_argument("--pick-start-z-offset", type=float, default=0.7)
    parser.add_argument("--pick-start-joint-std", type=float, default=0.035)
    parser.add_argument("--pick-start-max-attempts", type=int, default=40)
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-rollout-gif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--sample-input-grid-count", type=int, default=16)
    parser.add_argument(
        "--subgoal-chain-mode",
        choices=["off", "fixed", "valid-mask"],
        default="off",
        help="Optional subgoal termination mode. Baseline policy behavior is mode=off.",
    )
    parser.add_argument(
        "--subgoal-sequence",
        default=None,
        help="Comma-separated subgoal names, e.g. move_over_cube,pick_from_top_cube.",
    )
    parser.add_argument("--fixed-subgoal-chunks", type=int, default=1)
    parser.add_argument("--valid-mask-checkpoint", type=Path)
    parser.add_argument("--valid-mask-threshold", type=float, default=0.5)
    parser.add_argument("--valid-mask-consecutive", type=int, default=2)
    parser.add_argument(
        "--use-policy-processors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the saved LeRobot policy_preprocessor/policy_postprocessor pipeline for inference.",
    )
    parser.add_argument(
        "--torch-seed",
        type=int,
        default=None,
        help="Seed torch before each episode to make SmolVLA flow-sampling noise reproducible.",
    )
    args = parser.parse_args()

    report = evaluate_smolvla_picklift(
        policy_path=args.policy_path,
        output_dir=args.output_dir,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        local_files_only=args.local_files_only,
        width=args.width,
        height=args.height,
        action_alpha=args.action_alpha,
        max_arm_delta=args.max_arm_delta,
        max_gripper_delta=args.max_gripper_delta,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_num_steps=args.policy_num_steps,
        task_prompt=args.task_prompt,
        env_object_color=args.env_object_color,
        start_report_path=args.start_report_path,
        eval_skill_mode=args.eval_skill_mode,
        pick_start_min_actual_z=args.pick_start_min_actual_z,
        pick_start_min_actual_abs_y=args.pick_start_min_actual_abs_y,
        pick_start_max_actual_abs_y=args.pick_start_max_actual_abs_y,
        pick_start_z_offset=args.pick_start_z_offset,
        pick_start_joint_std=args.pick_start_joint_std,
        pick_start_max_attempts=args.pick_start_max_attempts,
        sweep=args.sweep,
        record_rollout_gif=args.record_rollout_gif,
        gif_fps=args.gif_fps,
        sample_input_grid_count=args.sample_input_grid_count,
        subgoal_chain_mode=args.subgoal_chain_mode,
        subgoal_sequence=args.subgoal_sequence,
        fixed_subgoal_chunks=args.fixed_subgoal_chunks,
        valid_mask_checkpoint=args.valid_mask_checkpoint,
        valid_mask_threshold=args.valid_mask_threshold,
        valid_mask_consecutive=args.valid_mask_consecutive,
        use_policy_processors=args.use_policy_processors,
        torch_seed=args.torch_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate_smolvla_picklift(
    *,
    policy_path: str,
    output_dir: Path,
    episodes: int,
    steps: int,
    seed: int,
    device: str,
    local_files_only: bool,
    width: int,
    height: int,
    action_alpha: float = 1.0,
    max_arm_delta: float = 0.0,
    max_gripper_delta: float = 0.0,
    policy_n_action_steps: int | None = None,
    policy_num_steps: int | None = None,
    task_prompt: str | None = None,
    env_object_color: str | None = None,
    start_report_path: Path | None = None,
    eval_skill_mode: str = "picklift",
    pick_start_min_actual_z: float = 0.05,
    pick_start_min_actual_abs_y: float = 0.015,
    pick_start_max_actual_abs_y: float = 0.065,
    pick_start_z_offset: float = 0.7,
    pick_start_joint_std: float = 0.035,
    pick_start_max_attempts: int = 40,
    sweep: bool = True,
    record_rollout_gif: bool = False,
    gif_fps: int = 12,
    sample_input_grid_count: int = 16,
    subgoal_chain_mode: str = "off",
    subgoal_sequence: str | None = None,
    fixed_subgoal_chunks: int = 1,
    valid_mask_checkpoint: Path | None = None,
    valid_mask_threshold: float = 0.5,
    valid_mask_consecutive: int = 2,
    use_policy_processors: bool = True,
    torch_seed: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    policy = _load_pretrained_policy(
        model_id=policy_path,
        local_files_only=local_files_only,
        device=device,
    )
    _override_policy_rollout_config(
        policy,
        n_action_steps=policy_n_action_steps,
        num_steps=policy_num_steps,
    )
    preprocessor, postprocessor = _load_policy_processors(policy, policy_path) if use_policy_processors else (None, None)
    valid_mask_head = None
    if subgoal_chain_mode == "valid-mask":
        if valid_mask_checkpoint is None:
            raise ValueError("--valid-mask-checkpoint is required when --subgoal-chain-mode=valid-mask")
        selected_device = str(_policy_device_metadata(policy).get("device_selected") or getattr(policy.config, "device", "cpu"))
        valid_mask_head = load_valid_mask_head(valid_mask_checkpoint, device=selected_device)
    start_report_episodes = _load_start_report_episodes(start_report_path)
    start_report_object_color = _start_report_object_color(start_report_episodes)
    resolved_env_object_color = env_object_color or start_report_object_color
    config = WristEgoServoConfig(width=width, height=height)
    env = _make_eval_env(eval_skill_mode, target_object_color=resolved_env_object_color)
    renderers = _make_policy_renderers(env, config)
    rows = []
    resolved_task_prompt = task_prompt or _default_task_prompt(eval_skill_mode)
    try:
        for episode in range(episodes):
            if torch_seed is not None:
                _set_torch_seed(int(torch_seed) + episode)
            reset_meta = _reset_episode(
                env=env,
                episode=episode,
                seed=seed + episode,
                eval_skill_mode=eval_skill_mode,
                start_report_episode=(
                    start_report_episodes[episode % len(start_report_episodes)]
                    if start_report_episodes
                    else None
                ),
                pick_start_min_actual_z=pick_start_min_actual_z,
                pick_start_min_actual_abs_y=pick_start_min_actual_abs_y,
                pick_start_max_actual_abs_y=pick_start_max_actual_abs_y,
                pick_start_z_offset=pick_start_z_offset,
                pick_start_joint_std=pick_start_joint_std,
                pick_start_max_attempts=pick_start_max_attempts,
            )
            if reset_meta.get("dropped"):
                rows.append(
                    {
                        "episode": episode,
                        "seed": seed + episode,
                        "success": False,
                        "skill_success": False,
                        "dropped": True,
                        "drop_reason": reset_meta.get("drop_reason"),
                        "reset_meta": reset_meta,
                        "search_steps": 0,
                        "steps": 0,
                    }
                )
                continue
            should_sweep = bool(sweep and eval_skill_mode == "picklift")
            if should_sweep:
                visible, search_steps = sweep_until_visible(env, renderers, max_sweeps=config.max_sweeps)
            else:
                visible, search_steps = True, 0
            if not visible:
                rows.append(
                    {
                        "episode": episode,
                        "seed": seed + episode,
                        "success": False,
                        "dropped": True,
                        "search_steps": search_steps,
                        "steps": 0,
                    }
                )
                continue
            rows.append(
                _run_episode(
                    env=env,
                    renderers=renderers,
                    policy=policy,
                    episode=episode,
                    seed=seed + episode,
                    max_steps=steps,
                    search_steps=search_steps,
                    action_alpha=action_alpha,
                    max_arm_delta=max_arm_delta,
                    max_gripper_delta=max_gripper_delta,
                    output_dir=output_dir,
                    record_rollout_gif=record_rollout_gif,
                    gif_fps=gif_fps,
                    sample_input_grid_count=sample_input_grid_count,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_prompt=resolved_task_prompt,
                    eval_skill_mode=eval_skill_mode,
                    reset_meta=reset_meta,
                    lift_success_height=pick_start_min_actual_z,
                    subgoal_chain_mode=subgoal_chain_mode,
                    subgoal_sequence=_resolve_subgoal_sequence(subgoal_sequence, eval_skill_mode),
                    fixed_subgoal_chunks=fixed_subgoal_chunks,
                    valid_mask_head=valid_mask_head,
                    valid_mask_threshold=valid_mask_threshold,
                    valid_mask_consecutive=valid_mask_consecutive,
                )
            )
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    report = {
        "operation": "evaluate_so101_picklift_smolvla_policy",
        "policy_path": policy_path,
        "runtime_inputs": ["egocentric_cam", "wrist_cam", "joint_positions", "task"],
        "runtime_excludes": ["top_down", "camera_calibration", "object_pose", "mujoco_jacobian"],
        "action_filter": {
            "action_alpha": float(action_alpha),
            "max_arm_delta": float(max_arm_delta),
            "max_gripper_delta": float(max_gripper_delta),
        },
        "pre_rollout_sweep": bool(sweep),
        "eval_skill_mode": eval_skill_mode,
        "task_prompt": resolved_task_prompt,
        "subgoal_chain": {
            "mode": subgoal_chain_mode,
            "sequence": _resolve_subgoal_sequence(subgoal_sequence, eval_skill_mode),
            "fixed_subgoal_chunks": int(fixed_subgoal_chunks),
            "valid_mask_checkpoint": str(valid_mask_checkpoint) if valid_mask_checkpoint else None,
            "valid_mask_threshold": float(valid_mask_threshold),
            "valid_mask_consecutive": int(valid_mask_consecutive),
        },
        "use_policy_processors": bool(preprocessor is not None and postprocessor is not None),
        "torch_seed": torch_seed,
        "env_config": {
            "object_shape": "cube",
            "object_color": resolved_env_object_color,
            "source": "start_report_path" if start_report_path is not None else "evaluator_default",
        },
        "start_report_path": str(start_report_path) if start_report_path is not None else None,
        "policy_rollout_config": _policy_rollout_config(policy),
        "feature_mapping": {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
            "observation.images.camera3": "wrist_cam duplicate when requested by policy",
            "observation.state": "SO101 qpos/control state",
            "action": (
                "SO101 qpos target action via saved policy_preprocessor/policy_postprocessor"
                if preprocessor is not None and postprocessor is not None
                else "SO101 raw policy output interpreted as qpos target"
            ),
            "task": resolved_task_prompt,
        },
        "episodes": rows,
        "success_rate": float(np.mean([row.get("skill_success", row["success"]) for row in rows])) if rows else 0.0,
        "env_success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "grasp_rate": float(np.mean([row.get("final_is_grasped", 0.0) > 0.5 for row in rows])) if rows else 0.0,
        "place_rate": float(np.mean([row.get("final_is_obj_placed", False) for row in rows])) if rows else 0.0,
        "duration_s": round(perf_counter() - started, 4),
        "device": _policy_device_metadata(policy),
    }
    report_path = output_dir / "so101_picklift_smolvla_eval_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_start_report_episodes(start_report_path: Path | None) -> list[dict[str, Any]]:
    if start_report_path is None:
        return []
    report = json.loads(Path(start_report_path).read_text(encoding="utf-8"))
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"closed-loop start report has no episodes: {start_report_path}")
    missing = [
        index
        for index, episode in enumerate(episodes)
        if not isinstance(episode, dict) or not isinstance(episode.get("sim_snapshot"), dict)
    ]
    if missing:
        raise ValueError(f"closed-loop start report episodes missing sim_snapshot: {missing[:8]}")
    return [dict(episode) for episode in episodes]


def _start_report_object_color(start_report_episodes: list[dict[str, Any]]) -> str | None:
    colors = {
        str(episode.get("object_color") or "").strip().lower()
        for episode in start_report_episodes
        if episode.get("object_color")
    }
    colors.discard("")
    if not colors:
        return None
    if len(colors) > 1:
        raise ValueError(f"closed-loop start report mixes object colors: {sorted(colors)}")
    return next(iter(colors))


def _restore_report_start_state(env: Any, episode: dict[str, Any]) -> dict[str, Any]:
    snapshot = episode.get("sim_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("closed-loop start episode missing sim_snapshot")
    restored = {
        key: np.asarray(snapshot[key], dtype=float)
        for key in ("qpos", "qvel", "ctrl")
        if key in snapshot
    }
    missing = {"qpos", "qvel", "ctrl"} - set(restored)
    if missing:
        raise ValueError(f"closed-loop start sim_snapshot missing fields: {sorted(missing)}")
    _restore_sim_state(env, restored)
    return {
        "mode": "start_report_snapshot",
        "source_episode_seed": episode.get("seed"),
        "source_episode_task": episode.get("task"),
        "source_episode_object_color": episode.get("object_color"),
        "source_episode_grid_bin": episode.get("grid_balance_bin", episode.get("desired_grid_bin")),
        "source_episode_success": episode.get("success"),
    }


def _make_eval_env(eval_skill_mode: str, *, target_object_color: str | None = None) -> Any:
    if eval_skill_mode == "pick_and_place_cube":
        return _make_pickplace_env()
    return make_high_contrast_picklift_env(target_object_color=target_object_color)


def _default_task_prompt(eval_skill_mode: str) -> str:
    if eval_skill_mode == "grip_the_cube_v1":
        return "grip the green cube and lift"
    if eval_skill_mode == "pick_from_top_cube":
        return PICK_FROM_TOP_TASK
    if eval_skill_mode == "pick_and_place_cube":
        return PICK_AND_PLACE_TASK
    return TASK


def _run_episode(
    *,
    env: Any,
    renderers: dict[str, Any],
    policy: Any,
    episode: int,
    seed: int,
    max_steps: int,
    search_steps: int,
    action_alpha: float,
    max_arm_delta: float,
    max_gripper_delta: float,
    output_dir: Path,
    record_rollout_gif: bool,
    gif_fps: int,
    sample_input_grid_count: int,
    preprocessor: Any | None,
    postprocessor: Any | None,
    task_prompt: str,
    eval_skill_mode: str,
    reset_meta: dict[str, Any],
    lift_success_height: float,
    subgoal_chain_mode: str,
    subgoal_sequence: list[str],
    fixed_subgoal_chunks: int,
    valid_mask_head: Any | None,
    valid_mask_threshold: float,
    valid_mask_consecutive: int,
) -> dict[str, Any]:
    records = []
    frames = []
    camera_samples: dict[str, list[np.ndarray]] = {"camera1": [], "camera2": []}
    info = env.unwrapped._get_info()
    image_feature_mapping = dict(POLICY_DISPLAY_IMAGE_FEATURE_MAPPING)
    trace_path = output_dir / "traces" / f"episode_{episode:03d}_seed_{seed}_policy_inputs.jsonl"
    trace_rows: list[dict[str, Any]] = []
    sample_every = max(1, max_steps // max(1, int(sample_input_grid_count)))
    active_chain = subgoal_chain_mode != "off" and bool(subgoal_sequence)
    subgoal_index = 0
    horizon_remaining = 0
    horizon_reason = "baseline"
    pending_advance = False
    n_action_steps = int(getattr(getattr(policy, "config", None), "n_action_steps", 15) or 15)
    if hasattr(policy, "reset"):
        policy.reset()
    for step in range(max_steps):
        if record_rollout_gif:
            frames.append(_render_rollout_frame(env, renderers))
        camera_pixels = _render_policy_cameras(env, renderers)
        if sample_input_grid_count > 0 and step % sample_every == 0:
            _append_camera_samples(camera_samples, camera_pixels, max_samples=sample_input_grid_count)
        if active_chain and horizon_remaining <= 0:
            if pending_advance and subgoal_index < len(subgoal_sequence) - 1:
                subgoal_index += 1
                if hasattr(policy, "reset"):
                    policy.reset()
            horizon_remaining, horizon_reason = _next_subgoal_horizon(
                mode=subgoal_chain_mode,
                policy=policy,
                preprocessor=preprocessor,
                qpos=_current_qpos(env).astype(float),
                camera_pixels=camera_pixels,
                task_prompt=_subgoal_prompt(subgoal_sequence[subgoal_index], task_prompt),
                valid_mask_head=valid_mask_head,
                max_horizon=n_action_steps,
                fixed_subgoal_chunks=fixed_subgoal_chunks,
                valid_mask_threshold=valid_mask_threshold,
                valid_mask_consecutive=valid_mask_consecutive,
            )
            pending_advance = horizon_reason in {"fixed_subgoal_stop", "valid_mask_stop"}
        current_subgoal = subgoal_sequence[subgoal_index] if active_chain else eval_skill_mode
        current_task_prompt = _subgoal_prompt(current_subgoal, task_prompt)
        if preprocessor is not None and postprocessor is not None:
            raw_action = _predict_action_with_processors(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                qpos=_current_qpos(env).astype(float),
                camera_pixels=camera_pixels,
                task_prompt=current_task_prompt,
            )
            image_feature_mapping = dict(POLICY_DISPLAY_IMAGE_FEATURE_MAPPING)
        else:
            batch, image_feature_mapping = _build_batch_for_policy(
                policy,
                _current_qpos(env).astype(float).tolist(),
                camera_pixels,
                instruction=current_task_prompt,
                local_files_only=True,
            )
            image_feature_mapping = {
                key: value
                for key, value in image_feature_mapping.items()
                if key in POLICY_DISPLAY_IMAGE_FEATURE_MAPPING
            }
            raw_action = policy.select_action(batch)
        media_paths = _write_policy_trace_images(
            camera_pixels=camera_pixels,
            output_dir=output_dir,
            episode=episode,
            seed=seed,
            step=step,
            enabled=record_rollout_gif,
        )
        if media_paths:
            trace_rows.append(
                {
                    "episode": int(episode),
                    "global_step": int(step),
                    "prompt": current_task_prompt,
                    "image_feature_mapping": dict(POLICY_DISPLAY_IMAGE_FEATURE_MAPPING),
                    "media": {"policy_input_images": media_paths},
                }
            )
        action = np.asarray(_tensor_to_float_list(raw_action)[:6], dtype=float)
        if action.shape[0] < 6:
            action = np.pad(action, (0, 6 - action.shape[0]))
        raw_action_values = action.copy()
        action = _filter_absolute_qpos_action(
            env=env,
            action=action,
            action_alpha=action_alpha,
            max_arm_delta=max_arm_delta,
            max_gripper_delta=max_gripper_delta,
        )
        action = np.clip(action, env.action_space.low, env.action_space.high)
        _obs, _reward, terminated, truncated, info = env.step(action)
        records.append(
            {
                "step": step,
                "action": [float(value) for value in action],
                "raw_action": [float(value) for value in raw_action_values],
                "success": bool(info.get("success", False)),
                "is_grasped": float(info.get("is_grasped", 0.0)),
                "is_obj_placed": bool(info.get("is_obj_placed", False)),
                "lift_height": float(info.get("lift_height", 0.0)),
                "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
                "obj_to_target_dist": float(info.get("obj_to_target_dist", 0.0)),
                "subgoal_chain_mode": subgoal_chain_mode,
                "subgoal": current_subgoal,
                "subgoal_index": subgoal_index,
                "subgoal_horizon_remaining": int(horizon_remaining),
                "subgoal_horizon_reason": horizon_reason,
            }
        )
        if active_chain:
            horizon_remaining -= 1
        if bool(info.get("success", False)) or terminated or truncated:
            break
    gif_path = None
    mp4_path = None
    if record_rollout_gif and frames:
        gif_path, mp4_path = _write_rollout_media(
            frames=frames,
            output_dir=output_dir,
            episode=episode,
            seed=seed,
            fps=gif_fps,
        )
    input_grid_paths = _write_policy_input_grids(
        samples=camera_samples,
        output_dir=output_dir,
        episode=episode,
        seed=seed,
    )
    written_trace_path = _write_policy_trace(trace_path, trace_rows)
    final_is_grasped = float(info.get("is_grasped", 0.0))
    final_lift_height = float(info.get("lift_height", 0.0))
    final_is_obj_placed = bool(info.get("is_obj_placed", False))
    final_obj_to_target_dist = float(info.get("obj_to_target_dist", 1.0))
    if eval_skill_mode == "pick_and_place_cube":
        skill_success = bool(final_is_obj_placed or (bool(info.get("success", False)) and final_obj_to_target_dist <= 0.035))
    else:
        skill_success = bool(final_is_grasped > 0.5 and final_lift_height >= float(lift_success_height))
    return {
        "episode": episode,
        "seed": seed,
        "eval_skill_mode": eval_skill_mode,
        "task_prompt": task_prompt,
        "subgoal_chain": {
            "mode": subgoal_chain_mode,
            "sequence": subgoal_sequence,
            "final_subgoal_index": subgoal_index,
        },
        "reset_meta": reset_meta,
        "search_steps": search_steps,
        "steps": len(records),
        "success": bool(info.get("success", False)),
        "skill_success": skill_success,
        "final_is_grasped": final_is_grasped,
        "final_is_obj_placed": final_is_obj_placed,
        "final_lift_height": final_lift_height,
        "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        "final_obj_to_target_dist": final_obj_to_target_dist,
        "image_feature_mapping": image_feature_mapping,
        "trace_path": written_trace_path,
        "rollout_gif": gif_path,
        "rollout_mp4": mp4_path,
        "input_grid_paths": input_grid_paths,
        "records": records,
    }


def _filter_absolute_qpos_action(
    *,
    env: Any,
    action: np.ndarray,
    action_alpha: float,
    max_arm_delta: float,
    max_gripper_delta: float,
) -> np.ndarray:
    current = _current_qpos(env).astype(float)
    target = np.asarray(action, dtype=float).copy()
    alpha = float(np.clip(action_alpha, 0.0, 1.0))
    if alpha < 1.0:
        target = current + alpha * (target - current)
    if max_arm_delta > 0:
        arm_delta = np.clip(target[:5] - current[:5], -float(max_arm_delta), float(max_arm_delta))
        target[:5] = current[:5] + arm_delta
    if max_gripper_delta > 0:
        gripper_delta = float(np.clip(target[-1] - current[-1], -float(max_gripper_delta), float(max_gripper_delta)))
        target[-1] = current[-1] + gripper_delta
    return target


def _resolve_subgoal_sequence(sequence: str | None, eval_skill_mode: str) -> list[str]:
    if sequence:
        items = [item.strip() for item in sequence.split(",") if item.strip()]
        if items:
            return items
    return [eval_skill_mode]


def _subgoal_prompt(subgoal: str, fallback: str) -> str:
    prompts = {
        "move_over_cube": "Move the gripper above the visible cube.",
        "pick_from_top_cube": PICK_FROM_TOP_TASK,
        "picklift": TASK,
        "pick_and_place_cube": PICK_AND_PLACE_TASK,
    }
    return prompts.get(subgoal, fallback)


def _next_subgoal_horizon(
    *,
    mode: str,
    policy: Any,
    preprocessor: Any | None,
    qpos: np.ndarray,
    camera_pixels: dict[str, np.ndarray],
    task_prompt: str,
    valid_mask_head: Any | None,
    max_horizon: int,
    fixed_subgoal_chunks: int,
    valid_mask_threshold: float,
    valid_mask_consecutive: int,
) -> tuple[int, str]:
    if mode == "fixed":
        return max(1, int(max_horizon) * max(1, int(fixed_subgoal_chunks))), "fixed_subgoal_stop"
    if mode != "valid-mask":
        return max(1, int(max_horizon)), "baseline"
    if valid_mask_head is None:
        raise ValueError("valid_mask_head is required for valid-mask subgoal chaining")
    state_for_head, action_chunk = _predict_action_chunk_for_valid_mask(
        policy=policy,
        preprocessor=preprocessor,
        qpos=qpos,
        camera_pixels=camera_pixels,
        task_prompt=task_prompt,
    )
    valid_probs = valid_mask_head.predict_valid_probs(state_for_head, action_chunk)
    if hasattr(policy, "reset"):
        policy.reset()
    return execution_horizon_from_valid_probs(
        valid_probs[0],
        max_horizon=max_horizon,
        threshold=valid_mask_threshold,
        consecutive=valid_mask_consecutive,
    )


def _predict_action_chunk_for_valid_mask(
    *,
    policy: Any,
    preprocessor: Any | None,
    qpos: np.ndarray,
    camera_pixels: dict[str, np.ndarray],
    task_prompt: str,
) -> tuple[Any, Any]:
    if preprocessor is not None:
        observation = {
            "observation.state": np.asarray(qpos, dtype=np.float32),
            "observation.images.camera1": np.asarray(camera_pixels["egocentric_cam"], dtype=np.uint8),
            "observation.images.camera2": np.asarray(camera_pixels["wrist_cam"], dtype=np.uint8),
            "observation.images.camera3": np.asarray(camera_pixels["wrist_cam"], dtype=np.uint8),
            "task": task_prompt,
        }
        try:
            batch = preprocessor(observation)
        except Exception:
            batch = None
        if batch is not None:
            with torch.inference_mode():
                return batch["observation.state"], policy.predict_action_chunk(batch)
    batch, _mapping = _build_batch_for_policy(
        policy,
        np.asarray(qpos, dtype=float).tolist(),
        camera_pixels,
        instruction=task_prompt,
        local_files_only=True,
    )
    with torch.inference_mode():
        return batch["observation.state"], policy.predict_action_chunk(batch)


def _reset_episode(
    *,
    env: Any,
    episode: int,
    seed: int,
    eval_skill_mode: str,
    start_report_episode: dict[str, Any] | None,
    pick_start_min_actual_z: float,
    pick_start_min_actual_abs_y: float,
    pick_start_max_actual_abs_y: float,
    pick_start_z_offset: float,
    pick_start_joint_std: float,
    pick_start_max_attempts: int,
) -> dict[str, Any]:
    if start_report_episode is not None:
        env.reset(seed=int(start_report_episode.get("seed") or seed))
        meta = _restore_report_start_state(env, start_report_episode)
        meta["reset_seed"] = int(start_report_episode.get("seed") or seed)
        return meta
    if eval_skill_mode in {"picklift", "grip_the_cube_v1"}:
        env.reset(seed=seed)
        return {"mode": eval_skill_mode, "reset_seed": seed}
    if eval_skill_mode == "pick_and_place_cube":
        env.reset(seed=seed)
        return {"mode": "pick_and_place_cube", "reset_seed": seed}
    if eval_skill_mode != "pick_from_top_cube":
        raise ValueError(f"Unsupported eval_skill_mode: {eval_skill_mode}")
    for attempt in range(max(1, int(pick_start_max_attempts))):
        reset_seed = int(seed) + attempt * 1009
        env.reset(seed=reset_seed)
        targets = [target for target in make_teacher_targets(env) if target.get("meta", {}).get("mode") == "overhead"]
        if not targets:
            targets = make_teacher_targets(env)
        if not targets:
            continue
        best = max(targets, key=lambda target: float(target.get("meta", {}).get("score", 0.0)))
        q_open = np.asarray(best["q_open"], dtype=np.float32)
        q_above = _offset_qpos_by_cartesian(
            env,
            q_open,
            np.asarray([0.0, 0.0, float(pick_start_z_offset)], dtype=float),
        )
        q_start = _make_near_gripper_qpos(
            env,
            q_above,
            seed=reset_seed + 313,
            joint_std=float(pick_start_joint_std),
        )
        q_start, target_y_offset = _balance_pick_start_y_offset(
            env,
            q_start,
            episode_index=episode + attempt,
            min_abs_y=float(pick_start_min_actual_abs_y),
            max_abs_y=float(pick_start_max_actual_abs_y),
        )
        q_start[-1] = float(env.action_space.low[-1])
        _set_qpos(env, q_start)
        tcp_delta = _tcp_to_object_delta(env)
        actual_z = float(tcp_delta[2])
        actual_abs_y = abs(float(tcp_delta[1]))
        if actual_z >= float(pick_start_min_actual_z) and actual_abs_y >= float(pick_start_min_actual_abs_y):
            return {
                "mode": "pick_from_top_cube",
                "reset_seed": reset_seed,
                "attempt": attempt,
                "teacher_candidate_meta": best.get("meta", {}),
                "target_y_offset": float(target_y_offset),
                "tcp_to_object_delta": [float(value) for value in tcp_delta],
                "start_min_actual_z": float(pick_start_min_actual_z),
                "start_min_actual_abs_y": float(pick_start_min_actual_abs_y),
                "start_gripper": float(q_start[-1]),
            }
    return {
        "mode": "pick_from_top_cube",
        "reset_seed": seed,
        "dropped": True,
        "drop_reason": "could_not_construct_pick_from_top_start",
        "attempts": max(1, int(pick_start_max_attempts)),
    }


def _load_policy_processors(policy: Any, policy_path: str):
    policy_dir = Path(policy_path)
    preprocessor_config = policy_dir / "policy_preprocessor.json"
    postprocessor_config = policy_dir / "policy_postprocessor.json"
    if not preprocessor_config.exists() or not postprocessor_config.exists():
        return None, None
    from lerobot.policies.factory import make_pre_post_processors

    selected_device = str(_policy_device_metadata(policy).get("device_selected") or getattr(policy.config, "device", "cpu"))
    if hasattr(policy.config, "device"):
        policy.config.device = selected_device
    return make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={"device_processor": {"device": selected_device}},
    )


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_action_with_processors(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    qpos: np.ndarray,
    camera_pixels: dict[str, np.ndarray],
    task_prompt: str,
):
    try:
        from lerobot.utils.control_utils import predict_action
    except ModuleNotFoundError:
        from lerobot.common.control_utils import predict_action

    selected_device = str(_policy_device_metadata(policy).get("device_selected") or getattr(policy.config, "device", "cpu"))
    observation = {
        "observation.state": np.asarray(qpos, dtype=np.float32),
        "observation.images.camera1": np.asarray(camera_pixels["egocentric_cam"], dtype=np.uint8),
        "observation.images.camera2": np.asarray(camera_pixels["wrist_cam"], dtype=np.uint8),
        "observation.images.camera3": np.asarray(camera_pixels["wrist_cam"], dtype=np.uint8),
    }
    return predict_action(
        observation=observation,
        policy=policy,
        device=torch.device(selected_device),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task=task_prompt,
        robot_type="so101",
    )


def _override_policy_rollout_config(
    policy: Any,
    *,
    n_action_steps: int | None,
    num_steps: int | None,
) -> None:
    config = getattr(policy, "config", None)
    if config is None:
        return
    if n_action_steps is not None:
        if n_action_steps < 1:
            raise ValueError(f"policy_n_action_steps must be positive, got {n_action_steps}")
        chunk_size = getattr(config, "chunk_size", None)
        if chunk_size is not None and n_action_steps > int(chunk_size):
            raise ValueError(f"policy_n_action_steps={n_action_steps} exceeds chunk_size={chunk_size}")
        config.n_action_steps = int(n_action_steps)
    if num_steps is not None:
        if num_steps < 1:
            raise ValueError(f"policy_num_steps must be positive, got {num_steps}")
        config.num_steps = int(num_steps)
    if hasattr(policy, "reset"):
        policy.reset()


def _policy_rollout_config(policy: Any) -> dict[str, Any]:
    config = getattr(policy, "config", None)
    return {
        "chunk_size": getattr(config, "chunk_size", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "num_steps": getattr(config, "num_steps", None),
    }


def _render_policy_cameras(env: Any, renderers: dict[str, Any]) -> dict[str, np.ndarray]:
    pixels = {}
    for camera_name in ("egocentric_cam", "wrist_cam"):
        renderer = renderers[camera_name]
        renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
        pixels[camera_name] = postprocess_camera_frame(camera_name, renderer.render()).astype(np.uint8)
    return pixels


def _render_rollout_frame(env: Any, renderers: dict[str, Any]) -> np.ndarray:
    renderer = renderers.get("scene_3d") or renderers.get("top_down")
    if renderer is None:
        renderer = renderers["egocentric_cam"]
        camera_name = "egocentric_cam"
    else:
        camera_name = "scene_3d" if "scene_3d" in renderers else "top_down"
    renderer.update_scene(env.unwrapped.data, camera=_make_camera(env, camera_name))
    return renderer.render().astype(np.uint8)


def _append_camera_samples(
    samples: dict[str, list[np.ndarray]],
    camera_pixels: dict[str, np.ndarray],
    *,
    max_samples: int,
) -> None:
    mapping = {"camera1": "egocentric_cam", "camera2": "wrist_cam"}
    for output_name, source_name in mapping.items():
        if len(samples[output_name]) >= max_samples:
            continue
        image = camera_pixels.get(source_name)
        if image is not None:
            samples[output_name].append(np.asarray(image, dtype=np.uint8).copy())


def _write_policy_input_grids(
    *,
    samples: dict[str, list[np.ndarray]],
    output_dir: Path,
    episode: int,
    seed: int,
) -> dict[str, str]:
    grids_dir = output_dir / "input_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for camera_name, images in samples.items():
        if not images:
            continue
        grid = _make_hwc_grid(images)
        path = grids_dir / f"episode_{episode:03d}_seed_{seed}_{camera_name}_grid.png"
        _write_png(path, grid)
        paths[camera_name] = str(path)
    return paths


def _write_policy_trace_images(
    *,
    camera_pixels: dict[str, np.ndarray],
    output_dir: Path,
    episode: int,
    seed: int,
    step: int,
    enabled: bool,
) -> dict[str, str]:
    if not enabled:
        return {}
    frame_dir = output_dir / "policy_input_frames" / f"episode_{episode:03d}_seed_{seed}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for camera_name in ("egocentric_cam", "wrist_cam"):
        image = camera_pixels.get(camera_name)
        if image is None:
            continue
        path = frame_dir / f"step_{step:04d}_{camera_name}.png"
        _write_png(path, np.asarray(image, dtype=np.uint8))
        paths[camera_name] = str(path)
    return paths


def _write_policy_trace(trace_path: Path, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return str(trace_path)


def _make_hwc_grid(images: list[np.ndarray]) -> np.ndarray:
    count = len(images)
    rows = max(1, int(round(count**0.5)))
    cols = int((count + rows - 1) // rows)
    h, w, c = images[0].shape
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    return grid


def _write_png(path: Path, image: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(path)


def _write_rollout_media(
    *,
    frames: list[np.ndarray],
    output_dir: Path,
    episode: int,
    seed: int,
    fps: int,
) -> tuple[str, str]:
    import imageio.v2 as imageio

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    gif_path = videos_dir / f"smolvla_policy_only_episode_{episode:03d}_seed_{seed}_rollout.gif"
    mp4_path = videos_dir / f"smolvla_policy_only_episode_{episode:03d}_seed_{seed}_rollout.mp4"
    imageio.mimsave(gif_path, frames, fps=fps)
    imageio.mimsave(mp4_path, frames, fps=fps)
    return str(gif_path), str(mp4_path)


if __name__ == "__main__":
    main()
