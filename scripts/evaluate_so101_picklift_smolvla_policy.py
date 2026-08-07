#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    update_valid_mask_requery_stop,
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
    _current_jaw_cube_face_normal_error_deg,
    _make_near_gripper_qpos,
    _offset_qpos_by_cartesian,
    _static_finger_edge_error,
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
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Blend overlapping postprocessed action chunks using ACT-style exponential temporal ensembling.",
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.01,
        help="Exponential decay k used for temporal-ensemble weights exp(-k * age_index).",
    )
    parser.add_argument(
        "--action-chunk-inference-mode",
        choices=["policy_queue", "temporal_ensemble", "rtc"],
        default=None,
        help=(
            "Explicit chunk execution mode. New training runs provide this from "
            "closed_loop.inference.mode."
        ),
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=["ZEROS", "ONES", "LINEAR", "EXP"],
    )
    parser.add_argument("--rtc-max-guidance-weight", type=float)
    parser.add_argument("--rtc-execution-horizon", type=int)
    parser.add_argument("--rtc-inference-delay", type=int)
    parser.add_argument("--rtc-debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rtc-debug-maxlen", type=int)
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--env-object-color", default=None)
    parser.add_argument(
        "--env-config-json",
        help="JSON environment contract for closed-loop replay, including object geometry and camera rig.",
    )
    parser.add_argument(
        "--start-report-path",
        type=Path,
        help="LeRobot export report whose episode sim_snapshot entries define closed-loop start states.",
    )
    parser.add_argument(
        "--episode-indices",
        help="Comma-separated report episode indices selected in deterministic order.",
    )
    parser.add_argument(
        "--phase-contract-json",
        help="JSON primitive/chain phase prompts, caps, reference reports, and verifier thresholds.",
    )
    parser.add_argument(
        "--observation-renderer-json",
        help="JSON closed-loop observation renderer contract. Omit to use MuJoCo policy cameras.",
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
    parser.add_argument("--valid-mask-requery-confirmations", type=int)
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
        temporal_ensemble=args.temporal_ensemble,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        action_chunk_inference_mode=args.action_chunk_inference_mode,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_inference_delay=args.rtc_inference_delay,
        rtc_debug=args.rtc_debug,
        rtc_debug_maxlen=args.rtc_debug_maxlen,
        task_prompt=args.task_prompt,
        env_object_color=args.env_object_color,
        env_config=(json.loads(args.env_config_json) if args.env_config_json else None),
        start_report_path=args.start_report_path,
        episode_indices=_parse_episode_indices(args.episode_indices),
        phase_contract=(json.loads(args.phase_contract_json) if args.phase_contract_json else None),
        observation_renderer_config=(
            json.loads(args.observation_renderer_json) if args.observation_renderer_json else None
        ),
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
        valid_mask_requery_confirmations=args.valid_mask_requery_confirmations,
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
    temporal_ensemble: bool = False,
    temporal_ensemble_decay: float = 0.01,
    action_chunk_inference_mode: str | None = None,
    rtc_prefix_attention_schedule: str | None = None,
    rtc_max_guidance_weight: float | None = None,
    rtc_execution_horizon: int | None = None,
    rtc_inference_delay: int | None = None,
    rtc_debug: bool | None = None,
    rtc_debug_maxlen: int | None = None,
    task_prompt: str | None = None,
    env_object_color: str | None = None,
    env_config: dict[str, Any] | None = None,
    start_report_path: Path | None = None,
    episode_indices: list[int] | None = None,
    phase_contract: dict[str, Any] | None = None,
    observation_renderer_config: dict[str, Any] | None = None,
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
    valid_mask_requery_confirmations: int | None = None,
    use_policy_processors: bool = True,
    torch_seed: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if temporal_ensemble_decay < 0:
        raise ValueError(f"temporal_ensemble_decay must be non-negative, got {temporal_ensemble_decay}")
    resolved_inference_mode = _resolve_action_chunk_inference_mode(
        action_chunk_inference_mode,
        legacy_temporal_ensemble=temporal_ensemble,
    )
    temporal_ensemble = resolved_inference_mode == "temporal_ensemble"
    rtc_settings = _validate_rtc_settings(
        mode=resolved_inference_mode,
        prefix_attention_schedule=rtc_prefix_attention_schedule,
        max_guidance_weight=rtc_max_guidance_weight,
        execution_horizon=rtc_execution_horizon,
        inference_delay=rtc_inference_delay,
        debug=rtc_debug,
        debug_maxlen=rtc_debug_maxlen,
    )
    if phase_contract is not None and subgoal_chain_mode != "valid-mask":
        raise ValueError("phase-contract rollout requires --subgoal-chain-mode=valid-mask")
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
    rtc_policy_config = _configure_policy_rtc(
        policy,
        mode=resolved_inference_mode,
        settings=rtc_settings,
    )
    preprocessor, postprocessor = _load_policy_processors(policy, policy_path) if use_policy_processors else (None, None)
    if resolved_inference_mode == "rtc" and (preprocessor is None or postprocessor is None):
        raise ValueError("RTC rollout requires the saved policy preprocessor and postprocessor")
    valid_mask_head = None
    if subgoal_chain_mode == "valid-mask":
        if valid_mask_checkpoint is None:
            raise ValueError("--valid-mask-checkpoint is required when --subgoal-chain-mode=valid-mask")
        if valid_mask_requery_confirmations is None or int(valid_mask_requery_confirmations) < 1:
            raise ValueError(
                "--valid-mask-requery-confirmations must be positive when --subgoal-chain-mode=valid-mask"
            )
        if preprocessor is None or postprocessor is None:
            raise ValueError("valid-mask rollout requires the saved policy processors")
        selected_device = str(_policy_device_metadata(policy).get("device_selected") or getattr(policy.config, "device", "cpu"))
        valid_mask_head = load_valid_mask_head(valid_mask_checkpoint, device=selected_device)
    start_report_episodes = _load_start_report_episodes(
        start_report_path,
        episode_indices=episode_indices,
        limit=episodes,
    )
    if start_report_episodes and len(start_report_episodes) != int(episodes):
        raise ValueError(
            "selected start-report episode count must match --episodes: "
            f"selected={len(start_report_episodes)} episodes={episodes}"
        )
    resolved_phase_contract = _load_phase_contract_episodes(
        phase_contract,
        episode_indices=episode_indices,
        episodes=episodes,
    )
    if resolved_phase_contract is not None:
        _validate_phase_episode_alignment(
            start_report_episodes=start_report_episodes,
            phase_contract=resolved_phase_contract,
        )
    start_report_object_color = _start_report_object_color(start_report_episodes)
    resolved_env_object_color = env_object_color or start_report_object_color
    config = WristEgoServoConfig(width=width, height=height)
    env = _make_eval_env(
        eval_skill_mode,
        target_object_color=resolved_env_object_color,
        env_config=env_config,
    )
    renderers = _make_policy_renderers(env, config)
    live_observation_renderer = _make_live_observation_renderer(
        output_dir=output_dir,
        width=width,
        height=height,
        config=observation_renderer_config,
    )
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
            episode_phase_contract = _phase_contract_for_episode(
                resolved_phase_contract,
                episode=episode,
            )
            rows.append(
                _run_episode(
                    env=env,
                    renderers=renderers,
                    policy=policy,
                    episode=episode,
                    seed=seed + episode,
                    max_steps=_phase_contract_episode_max_steps(
                        episode_phase_contract,
                        configured_max_steps=steps,
                    ),
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
                    valid_mask_requery_confirmations=int(valid_mask_requery_confirmations or 0),
                    temporal_ensemble=temporal_ensemble,
                    temporal_ensemble_decay=temporal_ensemble_decay,
                    action_chunk_inference_mode=resolved_inference_mode,
                    rtc_policy_config=rtc_policy_config,
                    rtc_settings=rtc_settings,
                    live_observation_renderer=live_observation_renderer,
                    phase_contract=episode_phase_contract,
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
            "valid_mask_requery_confirmations": valid_mask_requery_confirmations,
        },
        "use_policy_processors": bool(preprocessor is not None and postprocessor is not None),
        "torch_seed": torch_seed,
        "env_config": {
            "object_shape": "cube",
            "object_color": resolved_env_object_color,
            "source": "start_report_path" if start_report_path is not None else "evaluator_default",
            "contract": env_config,
        },
        "start_report_path": str(start_report_path) if start_report_path is not None else None,
        "episode_indices": episode_indices,
        "phase_contract": _phase_contract_report(resolved_phase_contract),
        "observation_renderer": (
            live_observation_renderer.report()
            if live_observation_renderer is not None
            else {"mode": "mujoco"}
        ),
        "policy_rollout_config": _policy_rollout_config(policy),
        "action_chunk_inference": _action_chunk_inference_report(
            mode=resolved_inference_mode,
            temporal_ensemble_decay=temporal_ensemble_decay,
            rtc_settings=rtc_settings,
        ),
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


def _parse_episode_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    indices = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not indices:
        raise ValueError("--episode-indices must contain at least one index")
    if any(index < 0 for index in indices):
        raise ValueError("--episode-indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError("--episode-indices must not contain duplicates")
    return indices


def _load_start_report_episodes(
    start_report_path: Path | None,
    *,
    episode_indices: list[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
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
    if episode_indices is not None:
        out_of_range = [index for index in episode_indices if index >= len(episodes)]
        if out_of_range:
            raise ValueError(
                f"closed-loop report episode indices out of range for {start_report_path}: "
                f"{out_of_range[:8]} >= {len(episodes)}"
            )
        selected = [(index, episodes[index]) for index in episode_indices]
    else:
        count = len(episodes) if limit is None else min(len(episodes), int(limit))
        selected = list(enumerate(episodes[:count]))
    return [
        {
            **dict(episode),
            "_report_episode_index": int(index),
            "_report_path": str(start_report_path),
        }
        for index, episode in selected
    ]


def _load_phase_contract_episodes(
    phase_contract: dict[str, Any] | None,
    *,
    episode_indices: list[int] | None,
    episodes: int,
) -> dict[str, Any] | None:
    if phase_contract is None:
        return None
    contract = dict(phase_contract)
    phases = contract.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("phase contract must contain a non-empty phases list")
    loaded_phases: list[dict[str, Any]] = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError(f"phase contract phases[{index}] must be an object")
        report_path = phase.get("reference_report_path")
        if not report_path:
            raise ValueError(f"phase contract phases[{index}] is missing reference_report_path")
        reference_episodes = _load_start_report_episodes(
            Path(str(report_path)),
            episode_indices=episode_indices,
            limit=episodes,
        )
        if len(reference_episodes) != int(episodes):
            raise ValueError(
                f"phase {phase.get('id', index)!r} selected {len(reference_episodes)} reference "
                f"episodes, expected {episodes}"
            )
        reference_frame_counts = [
            _reference_episode_frame_count(reference_episode)
            for reference_episode in reference_episodes
        ]
        multiplier = phase.get("reference_length_multiplier")
        if multiplier is not None:
            resolved_caps = [
                int(math.ceil(frame_count * float(multiplier)))
                for frame_count in reference_frame_counts
            ]
            declared_cap = int(phase["max_steps"])
            mismatches = [
                {
                    "episode": int(reference_episode["_report_episode_index"]),
                    "reference_frames": int(frame_count),
                    "resolved_max_steps": int(resolved_cap),
                    "declared_max_steps": declared_cap,
                }
                for reference_episode, frame_count, resolved_cap in zip(
                    reference_episodes,
                    reference_frame_counts,
                    resolved_caps,
                    strict=True,
                )
                if int(resolved_cap) != declared_cap
            ]
            if mismatches:
                raise ValueError(
                    f"phase {phase.get('id', index)!r} max_steps does not match "
                    f"ceil(reference_frames * reference_length_multiplier): {mismatches[:4]}"
                )
        loaded_phases.append(
            {
                **dict(phase),
                "_reference_episodes": reference_episodes,
                "_reference_frame_counts": reference_frame_counts,
            }
        )
    contract["phases"] = loaded_phases
    return contract


def _reference_episode_frame_count(episode: dict[str, Any]) -> int:
    try:
        frame_count = int(episode["frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("phase reference episode is missing a positive frames count") from exc
    if frame_count <= 0:
        raise ValueError(f"phase reference episode frames must be positive, got {frame_count}")
    phase_counts = episode.get("phase_counts")
    if isinstance(phase_counts, dict) and phase_counts:
        counted = sum(int(value) for value in phase_counts.values())
        if counted != frame_count:
            raise ValueError(
                "phase reference episode frames disagrees with phase_counts: "
                f"frames={frame_count} phase_counts={counted}"
            )
    return frame_count


def _episode_source_identity(episode: dict[str, Any]) -> dict[str, Any]:
    provenance = episode.get("source_provenance")
    source_episode = provenance.get("episode_index") if isinstance(provenance, dict) else None
    forced_spawn_xy = episode.get("forced_spawn_xy")
    return {
        "source_episode_index": source_episode,
        "seed": episode.get("seed"),
        "forced_spawn_xy": (
            [round(float(value), 9) for value in forced_spawn_xy]
            if isinstance(forced_spawn_xy, list)
            else None
        ),
        "object_color": episode.get("object_color"),
        "object_shape": episode.get("object_shape"),
    }


def _validate_phase_episode_alignment(
    *,
    start_report_episodes: list[dict[str, Any]],
    phase_contract: dict[str, Any],
) -> None:
    phases = phase_contract["phases"]
    if not start_report_episodes:
        raise ValueError("phase-contract rollout requires start-report episodes")
    first_phase_episodes = phases[0]["_reference_episodes"]
    for episode_index, start_episode in enumerate(start_report_episodes):
        expected = _episode_source_identity(start_episode)
        first = _episode_source_identity(first_phase_episodes[episode_index])
        if first != expected:
            raise ValueError(
                f"phase start report is not aligned at rollout episode {episode_index}: "
                f"start={expected} phase={first}"
            )
        for phase in phases[1:]:
            candidate = _episode_source_identity(phase["_reference_episodes"][episode_index])
            if candidate != expected:
                raise ValueError(
                    f"phase {phase.get('id')!r} is not aligned at rollout episode {episode_index}: "
                    f"start={expected} phase={candidate}"
                )


def _phase_contract_for_episode(
    phase_contract: dict[str, Any] | None,
    *,
    episode: int,
) -> dict[str, Any] | None:
    if phase_contract is None:
        return None
    result = {key: value for key, value in phase_contract.items() if key != "phases"}
    result["phases"] = [
        {
            **{
                key: value
                for key, value in phase.items()
                if key not in {"_reference_episodes", "_reference_frame_counts"}
            },
            "_reference_episode": phase["_reference_episodes"][episode],
            "reference_steps": int(phase["_reference_frame_counts"][episode]),
        }
        for phase in phase_contract["phases"]
    ]
    return result


def _phase_contract_report(phase_contract: dict[str, Any] | None) -> dict[str, Any] | None:
    if phase_contract is None:
        return None
    result = {key: value for key, value in phase_contract.items() if key != "phases"}
    result["phases"] = [
        {
            key: value
            for key, value in phase.items()
            if key not in {"_reference_episodes", "_reference_frame_counts"}
        }
        for phase in phase_contract["phases"]
    ]
    return result


def _phase_contract_episode_max_steps(
    phase_contract: dict[str, Any] | None,
    *,
    configured_max_steps: int,
) -> int:
    if phase_contract is None:
        return int(configured_max_steps)
    phase_max_steps = sum(int(phase["max_steps"]) for phase in phase_contract["phases"])
    if int(configured_max_steps) != phase_max_steps:
        raise ValueError(
            "closed-loop steps must equal the sum of resolved phase caps: "
            f"configured={configured_max_steps} phase_sum={phase_max_steps}"
        )
    return phase_max_steps


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
    expected_shapes = {
        "qpos": np.asarray(env.unwrapped.data.qpos).shape,
        "qvel": np.asarray(env.unwrapped.data.qvel).shape,
        "ctrl": np.asarray(env.unwrapped.data.ctrl).shape,
    }
    mismatched = {
        key: {"snapshot": list(restored[key].shape), "env": list(expected_shapes[key])}
        for key in restored
        if restored[key].shape != expected_shapes[key]
    }
    if mismatched:
        raise ValueError(
            "closed-loop start snapshot does not match evaluator environment contract: "
            f"{mismatched}"
        )
    _restore_sim_state(env, restored)
    return {
        "mode": "start_report_snapshot",
        "source_episode_seed": episode.get("seed"),
        "source_report_path": episode.get("_report_path"),
        "source_report_episode_index": episode.get("_report_episode_index"),
        "source_episode_task": episode.get("task"),
        "source_episode_object_color": episode.get("object_color"),
        "source_episode_grid_bin": episode.get("grid_balance_bin", episode.get("desired_grid_bin")),
        "source_episode_success": episode.get("success"),
    }


def _make_eval_env(
    eval_skill_mode: str,
    *,
    target_object_color: str | None = None,
    env_config: dict[str, Any] | None = None,
) -> Any:
    if eval_skill_mode == "pick_and_place_cube":
        if env_config is not None:
            raise ValueError("pick_and_place_cube does not support the SO101 picklift env_config contract")
        return _make_pickplace_env()
    if env_config is None:
        return make_high_contrast_picklift_env(target_object_color=target_object_color)

    configured_color = str(env_config["target_object_color"]).strip().lower()
    if target_object_color is not None and configured_color != str(target_object_color).strip().lower():
        raise ValueError(
            "closed-loop target object color disagrees with env_config: "
            f"target={target_object_color!r} env_config={configured_color!r}"
        )
    camera_rig_path = Path(str(env_config["camera_rig_config"])).expanduser().resolve()
    from physical_ai_agent.sim.so101_camera_rig_render_config import load_so101_camera_rig_render_config

    camera_rig = load_so101_camera_rig_render_config(camera_rig_path)
    half_sizes = tuple(float(value) for value in env_config["object_half_sizes"])
    spawn_center_values = env_config["spawn_center"]
    if len(spawn_center_values) != 2:
        raise ValueError("env_config.spawn_center must contain exactly two values")
    return make_high_contrast_picklift_env(
        target_object_color=configured_color,
        object_half_sizes=half_sizes,
        spawn_center=(float(spawn_center_values[0]), float(spawn_center_values[1])),
        spawn_min_radius=float(env_config["spawn_min_radius"]),
        spawn_max_radius=float(env_config["spawn_max_radius"]),
        spawn_angle_half_range_deg=float(env_config["spawn_angle_half_range_deg"]),
        camera_rig_preset=camera_rig.preset,
        camera_rig_config=camera_rig,
    )


def _default_task_prompt(eval_skill_mode: str) -> str:
    if eval_skill_mode == "grip_the_cube_v1":
        return "grip the green cube and lift"
    if eval_skill_mode == "pick_from_top_cube":
        return PICK_FROM_TOP_TASK
    if eval_skill_mode == "pick_and_place_cube":
        return PICK_AND_PLACE_TASK
    return TASK


def _evaluate_phase_verifier(
    *,
    env: Any,
    phase: dict[str, Any],
    hold_streak: int,
) -> dict[str, Any]:
    phase_id = str(phase["id"])
    verifier = phase["verifier"]
    reference_episode = phase["_reference_episode"]
    current_qpos = _current_qpos(env).astype(float)
    open_error = abs(float(env.action_space.high[-1]) - float(current_qpos[-1]))
    if phase_id == "approach":
        reference_qpos = reference_episode.get("phase_end_observation_state")
        if not isinstance(reference_qpos, list):
            raise ValueError("approach reference episode is missing phase_end_observation_state")
        snapshot = {
            "qpos": np.asarray(env.unwrapped.data.qpos, dtype=float).copy(),
            "qvel": np.asarray(env.unwrapped.data.qvel, dtype=float).copy(),
            "ctrl": np.asarray(env.unwrapped.data.ctrl, dtype=float).copy(),
        }
        try:
            _set_qpos(env, np.asarray(reference_qpos, dtype=float))
            reference_tcp_delta = _tcp_to_object_delta(env).astype(float)
        finally:
            _restore_sim_state(env, snapshot)
        current_tcp_delta = _tcp_to_object_delta(env).astype(float)
        tcp_error = float(np.linalg.norm(current_tcp_delta - reference_tcp_delta))
        passed = (
            tcp_error <= float(verifier["tcp_position_tolerance_m"])
            and open_error <= float(verifier["gripper_open_tolerance_rad"])
        )
        metrics = {
            "tcp_position_error_m": tcp_error,
            "tcp_position_tolerance_m": float(verifier["tcp_position_tolerance_m"]),
            "gripper_open_error_rad": open_error,
            "gripper_open_tolerance_rad": float(verifier["gripper_open_tolerance_rad"]),
            "current_tcp_to_object_delta": current_tcp_delta.tolist(),
            "reference_tcp_to_object_delta": reference_tcp_delta.tolist(),
        }
    elif phase_id == "alignment":
        best_meta = reference_episode.get("best_meta")
        if not isinstance(best_meta, dict):
            raise ValueError("alignment reference episode is missing best_meta")
        edge_error = _static_finger_edge_error(env, best_meta)
        jaw_angle_error = float(_current_jaw_cube_face_normal_error_deg(env, best_meta))
        passed = (
            float(edge_error["xy_error"]) <= float(verifier["edge_xy_tolerance_m"])
            and jaw_angle_error <= float(verifier["jaw_angle_tolerance_deg"])
            and open_error <= float(verifier["gripper_open_tolerance_rad"])
        )
        metrics = {
            "edge_xy_error_m": float(edge_error["xy_error"]),
            "edge_xy_tolerance_m": float(verifier["edge_xy_tolerance_m"]),
            "jaw_angle_error_deg": jaw_angle_error,
            "jaw_angle_tolerance_deg": float(verifier["jaw_angle_tolerance_deg"]),
            "gripper_open_error_rad": open_error,
            "gripper_open_tolerance_rad": float(verifier["gripper_open_tolerance_rad"]),
        }
    elif phase_id == "grip_lift":
        info = env.unwrapped._get_info()
        grasped = float(info.get("is_grasped", 0.0)) > 0.5
        lift_height = float(info.get("lift_height", 0.0))
        required_hold = int(verifier["hold_steps"])
        passed = (
            grasped
            and lift_height >= float(verifier["lift_height_m"])
            and int(hold_streak) >= required_hold
        )
        metrics = {
            "grasped": grasped,
            "lift_height_m": lift_height,
            "lift_height_threshold_m": float(verifier["lift_height_m"]),
            "hold_steps": int(hold_streak),
            "required_hold_steps": required_hold,
        }
    else:
        raise ValueError(f"unsupported phase verifier: {phase_id!r}")
    return {
        "passed": bool(passed),
        "kind": phase_id,
        "metrics": metrics,
    }


def _phase_result(
    *,
    phase: dict[str, Any],
    start_step: int,
    end_step: int,
    verifier_result: dict[str, Any],
    termination_reason: str,
) -> dict[str, Any]:
    reference_episode = phase["_reference_episode"]
    return {
        "id": str(phase["id"]),
        "prompt": str(phase["prompt"]),
        "start_step": int(start_step),
        "end_step_exclusive": int(end_step),
        "executed_steps": max(0, int(end_step) - int(start_step)),
        "max_steps": int(phase["max_steps"]),
        "reference_steps": int(phase["reference_steps"]),
        "reference_length_multiplier": (
            float(phase["reference_length_multiplier"])
            if phase.get("reference_length_multiplier") is not None
            else None
        ),
        "termination_reason": termination_reason,
        "verifier": verifier_result,
        "reference_report_path": str(phase["reference_report_path"]),
        "reference_report_episode_index": reference_episode.get("_report_episode_index"),
        "source_identity": _episode_source_identity(reference_episode),
    }


def _phase_transition_outcome(
    *,
    verifier_passed: bool,
    final_phase: bool,
) -> dict[str, str]:
    if not verifier_passed:
        return {"action": "continue", "reason": "valid_mask_verifier_rejected"}
    if final_phase:
        return {"action": "complete", "reason": "env_success"}
    return {"action": "advance", "reason": "env_success"}


def _phase_contract_episode_report(
    phase_contract: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if phase_contract is None:
        return None
    result = {key: value for key, value in phase_contract.items() if key != "phases"}
    result["phases"] = []
    for phase in phase_contract["phases"]:
        reference_episode = phase["_reference_episode"]
        result["phases"].append(
            {
                **{
                    key: value
                    for key, value in phase.items()
                    if key != "_reference_episode"
                },
                "reference_report_episode_index": reference_episode.get("_report_episode_index"),
                "source_identity": _episode_source_identity(reference_episode),
            }
        )
    return result


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
    valid_mask_requery_confirmations: int,
    temporal_ensemble: bool,
    temporal_ensemble_decay: float,
    action_chunk_inference_mode: str,
    rtc_policy_config: Any | None,
    rtc_settings: dict[str, Any] | None,
    live_observation_renderer: Any | None,
    phase_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    records = []
    frames = []
    camera_samples: dict[str, list[np.ndarray]] = {"camera1": [], "camera2": []}
    info = env.unwrapped._get_info()
    image_feature_mapping = dict(POLICY_DISPLAY_IMAGE_FEATURE_MAPPING)
    trace_path = output_dir / "traces" / f"episode_{episode:03d}_seed_{seed}_policy_inputs.jsonl"
    trace_rows: list[dict[str, Any]] = []
    sample_every = max(1, max_steps // max(1, int(sample_input_grid_count)))
    valid_mask_mode = subgoal_chain_mode == "valid-mask"
    phase_specs = list(phase_contract.get("phases") or []) if phase_contract is not None else []
    phase_contract_mode = bool(phase_specs)
    if phase_contract_mode:
        subgoal_sequence = [str(phase["id"]) for phase in phase_specs]
    active_chain = phase_contract_mode or (subgoal_chain_mode != "off" and bool(subgoal_sequence))
    subgoal_index = 0
    phase_start_step = 0
    phase_results: list[dict[str, Any]] = []
    phase_stop_proposals: list[dict[str, Any]] = []
    phase_hold_streak = 0
    phase_contract_success = False
    hard_cap_scope = "chain"
    horizon_remaining = 0
    horizon_reason = "baseline"
    pending_advance = False
    n_action_steps = int(getattr(getattr(policy, "config", None), "n_action_steps", 15) or 15)
    temporal_chunks: list[tuple[int, np.ndarray]] = []
    rtc_action_queue = (
        _make_rtc_action_queue(rtc_policy_config)
        if action_chunk_inference_mode == "rtc"
        else None
    )
    next_policy_inference_step = 0
    valid_mask_stop_streak = 0
    valid_mask_stop_after_step: int | None = None
    termination_reason = "hard_cap"
    last_camera_pixels: dict[str, np.ndarray] | None = None
    observation_render_records: list[dict[str, Any]] = []
    if hasattr(policy, "reset"):
        policy.reset()
    for step in range(max_steps):
        if phase_contract_mode and horizon_remaining <= 0 and pending_advance:
            phase = phase_specs[subgoal_index]
            verifier_result = _evaluate_phase_verifier(
                env=env,
                phase=phase,
                hold_streak=phase_hold_streak,
            )
            transition = _phase_transition_outcome(
                verifier_passed=bool(verifier_result["passed"]),
                final_phase=subgoal_index == len(phase_specs) - 1,
            )
            phase_stop_proposals.append(
                {
                    "phase": str(phase["id"]),
                    "step": int(step),
                    "accepted": bool(verifier_result["passed"]),
                    "reason": str(transition["reason"]),
                    "verifier": verifier_result,
                }
            )
            if transition["action"] == "continue":
                pending_advance = False
                horizon_remaining = 0
                temporal_chunks.clear()
                _clear_rtc_action_queue(rtc_action_queue)
                next_policy_inference_step = step
                valid_mask_stop_streak = 0
                valid_mask_stop_after_step = None
                if hasattr(policy, "reset"):
                    policy.reset()
            else:
                phase_results.append(
                    _phase_result(
                        phase=phase,
                        start_step=phase_start_step,
                        end_step=step,
                        verifier_result=verifier_result,
                        termination_reason=str(transition["reason"]),
                    )
                )
            if transition["action"] == "complete":
                termination_reason = str(transition["reason"])
                phase_contract_success = True
                break
            if transition["action"] == "advance" and phase_contract.get("handoff_mode") == "oracle_reset":
                _restore_report_start_state(
                    env,
                    phase_specs[subgoal_index + 1]["_reference_episode"],
                )
                info = env.unwrapped._get_info()
            if transition["action"] == "advance":
                subgoal_index += 1
                phase_start_step = step
                phase_hold_streak = 0
                horizon_remaining = 0
                pending_advance = False
                temporal_chunks.clear()
                _clear_rtc_action_queue(rtc_action_queue)
                next_policy_inference_step = step
                valid_mask_stop_streak = 0
                if hasattr(policy, "reset"):
                    policy.reset()
        if phase_contract_mode:
            phase_cap = int(phase_specs[subgoal_index]["max_steps"])
            if step - phase_start_step >= phase_cap:
                phase = phase_specs[subgoal_index]
                verifier_result = _evaluate_phase_verifier(
                    env=env,
                    phase=phase,
                    hold_streak=phase_hold_streak,
                )
                transition = _phase_transition_outcome(
                    verifier_passed=bool(verifier_result["passed"]),
                    final_phase=subgoal_index == len(phase_specs) - 1,
                )
                if transition["action"] == "continue":
                    termination_reason = "hard_cap"
                    hard_cap_scope = f"phase:{phase['id']}"
                    phase_results.append(
                        _phase_result(
                            phase=phase,
                            start_step=phase_start_step,
                            end_step=step,
                            verifier_result=verifier_result,
                            termination_reason="hard_cap",
                        )
                    )
                    break
                phase_results.append(
                    _phase_result(
                        phase=phase,
                        start_step=phase_start_step,
                        end_step=step,
                        verifier_result=verifier_result,
                        termination_reason=str(transition["reason"]),
                    )
                )
                if transition["action"] == "complete":
                    termination_reason = str(transition["reason"])
                    phase_contract_success = True
                    break
                if phase_contract.get("handoff_mode") == "oracle_reset":
                    _restore_report_start_state(
                        env,
                        phase_specs[subgoal_index + 1]["_reference_episode"],
                    )
                    info = env.unwrapped._get_info()
                subgoal_index += 1
                phase_start_step = step
                phase_hold_streak = 0
                horizon_remaining = 0
                pending_advance = False
                temporal_chunks.clear()
                _clear_rtc_action_queue(rtc_action_queue)
                next_policy_inference_step = step
                valid_mask_stop_streak = 0
                if hasattr(policy, "reset"):
                    policy.reset()
        live_render_due = _live_observation_render_due(
            render_policy_inference_only=bool(
                getattr(live_observation_renderer, "render_policy_inference_only", True)
            ),
            has_last_camera_pixels=last_camera_pixels is not None,
            step=step,
            next_policy_inference_step=next_policy_inference_step,
            has_temporal_chunks=(
                not rtc_action_queue.empty()
                if rtc_action_queue is not None
                else bool(temporal_chunks)
            ),
            subgoal_advance_due=bool(active_chain and horizon_remaining <= 0 and pending_advance),
        )
        live_render_record = None
        if live_observation_renderer is not None:
            if live_render_due:
                camera_pixels, live_render_record = live_observation_renderer.render(
                    env=env,
                    mujoco_renderers=renderers,
                    episode=episode,
                    seed=int(reset_meta.get("reset_seed") or seed),
                    step=step,
                )
                last_camera_pixels = camera_pixels
                observation_render_records.append(live_render_record)
            else:
                assert last_camera_pixels is not None
                camera_pixels = last_camera_pixels
            if record_rollout_gif and live_render_record is not None:
                frames.append(_render_policy_camera_pair(camera_pixels))
        else:
            if record_rollout_gif:
                frames.append(_render_rollout_frame(env, renderers))
            camera_pixels = _render_policy_cameras(env, renderers)
        if sample_input_grid_count > 0 and (
            live_observation_renderer is None and step % sample_every == 0
            or live_observation_renderer is not None and live_render_record is not None
        ):
            _append_camera_samples(camera_samples, camera_pixels, max_samples=sample_input_grid_count)
        if active_chain and not phase_contract_mode and horizon_remaining <= 0:
            if pending_advance and subgoal_index < len(subgoal_sequence) - 1:
                subgoal_index += 1
                temporal_chunks.clear()
                _clear_rtc_action_queue(rtc_action_queue)
                valid_mask_stop_streak = 0
                if hasattr(policy, "reset"):
                    policy.reset()
            if not valid_mask_mode:
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
        current_task_prompt = (
            str(phase_specs[subgoal_index]["prompt"])
            if phase_contract_mode
            else _subgoal_prompt(current_subgoal, task_prompt)
        )
        policy_inference = False
        temporal_source_count = 1
        requery_overlap_rmse = None
        valid_mask_probs: list[float] | None = None
        valid_mask_predicted_horizon: int | None = None
        valid_mask_horizon_reason: str | None = None
        rtc_guidance_applied = False
        rtc_leftover_steps = 0
        if (
            temporal_ensemble or valid_mask_mode or rtc_action_queue is not None
        ) and preprocessor is not None and postprocessor is not None:
            has_chunk_buffer = (
                not rtc_action_queue.empty()
                if rtc_action_queue is not None
                else bool(temporal_chunks)
            )
            policy_inference = step >= next_policy_inference_step or not has_chunk_buffer
            if policy_inference:
                rtc_previous_leftover = (
                    rtc_action_queue.get_left_over()
                    if rtc_action_queue is not None
                    else None
                )
                rtc_previous_processed = (
                    rtc_action_queue.get_processed_left_over()
                    if rtc_action_queue is not None
                    else None
                )
                rtc_leftover_steps = (
                    int(rtc_previous_leftover.shape[0])
                    if rtc_previous_leftover is not None
                    else 0
                )
                rtc_guidance_applied = rtc_leftover_steps > 0
                new_chunk, state_for_mask, raw_chunk_for_mask = _predict_action_chunk_with_processors_and_inputs(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    qpos=_current_qpos(env).astype(float),
                    camera_pixels=camera_pixels,
                    task_prompt=current_task_prompt,
                    policy_predict_kwargs=(
                        _rtc_policy_predict_kwargs(
                            previous_leftover=rtc_previous_leftover,
                            settings=rtc_settings,
                        )
                        if rtc_action_queue is not None
                        else None
                    ),
                )
                next_policy_inference_step = step + n_action_steps
                if valid_mask_mode:
                    valid_probs_tensor = valid_mask_head.predict_valid_probs(state_for_mask, raw_chunk_for_mask)
                    valid_mask_probs = [float(value) for value in valid_probs_tensor[0].detach().float().cpu().tolist()]
                    valid_mask_predicted_horizon, valid_mask_horizon_reason = execution_horizon_from_valid_probs(
                        valid_probs_tensor[0],
                        max_horizon=n_action_steps,
                        threshold=valid_mask_threshold,
                        consecutive=valid_mask_consecutive,
                    )
                    horizon_reason = valid_mask_horizon_reason
                    predicted_stop = valid_mask_horizon_reason == "valid_mask_stop"
                    valid_mask_stop_streak, stop_confirmed = update_valid_mask_requery_stop(
                        valid_mask_stop_streak,
                        predicted_stop=predicted_stop,
                        required_confirmations=valid_mask_requery_confirmations,
                    )
                    if predicted_stop:
                        next_policy_inference_step = step + valid_mask_predicted_horizon
                        if rtc_action_queue is None:
                            new_chunk = new_chunk[:valid_mask_predicted_horizon]
                    if active_chain:
                        horizon_remaining = int(valid_mask_predicted_horizon)
                        pending_advance = stop_confirmed if phase_contract_mode else predicted_stop
                    final_subgoal = not active_chain or subgoal_index == len(subgoal_sequence) - 1
                    if final_subgoal and stop_confirmed and not phase_contract_mode:
                        valid_mask_stop_after_step = step + int(valid_mask_predicted_horizon)
                if rtc_action_queue is not None:
                    requery_overlap_rmse = _rtc_processed_overlap_rmse(
                        previous_leftover=rtc_previous_processed,
                        new_chunk=new_chunk,
                        max_horizon=n_action_steps,
                    )
                    rtc_action_queue.merge(
                        original_actions=_rtc_unbatched_chunk(raw_chunk_for_mask),
                        processed_actions=torch.as_tensor(new_chunk),
                        real_delay=int((rtc_settings or {})["inference_delay"]),
                    )
                elif temporal_chunks:
                    previous_start, previous_chunk = temporal_chunks[-1]
                    requery_overlap_rmse = _action_chunk_overlap_rmse(
                        previous_start=previous_start,
                        previous_chunk=previous_chunk,
                        new_start=step,
                        new_chunk=new_chunk,
                        max_horizon=n_action_steps,
                    )
                if rtc_action_queue is None:
                    temporal_chunks.append((step, new_chunk))
            if rtc_action_queue is not None:
                raw_action = rtc_action_queue.get()
                if raw_action is None:
                    raise RuntimeError(f"RTC action queue is empty at environment step {step}")
            else:
                temporal_chunks = [
                    (start, chunk)
                    for start, chunk in temporal_chunks
                    if step < start + int(chunk.shape[0])
                ]
            if temporal_ensemble:
                raw_action, temporal_source_count = _temporal_ensemble_action(
                    temporal_chunks,
                    step=step,
                    decay=temporal_ensemble_decay,
                )
            elif rtc_action_queue is None:
                chunk_start, current_chunk = temporal_chunks[-1]
                raw_action = current_chunk[step - chunk_start]
            image_feature_mapping = dict(POLICY_DISPLAY_IMAGE_FEATURE_MAPPING)
        elif preprocessor is not None and postprocessor is not None:
            policy_inference = step % n_action_steps == 0
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
            enabled=bool(
                record_rollout_gif
                and (live_observation_renderer is None or live_render_record is not None)
            ),
        )
        if media_paths:
            trace_rows.append(
                {
                    "episode": int(episode),
                    "global_step": int(step),
                    "prompt": current_task_prompt,
                    "phase": current_subgoal,
                    "phase_index": int(subgoal_index),
                    "policy_inference": bool(policy_inference),
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
        if phase_contract_mode and phase_specs[subgoal_index]["id"] == "grip_lift":
            verifier = phase_specs[subgoal_index]["verifier"]
            lift_height = float(info.get("lift_height", 0.0))
            if (
                float(info.get("is_grasped", 0.0)) > 0.5
                and lift_height >= float(verifier["lift_height_m"])
            ):
                phase_hold_streak += 1
            else:
                phase_hold_streak = 0
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
                "phase_contract_mode": bool(phase_contract_mode),
                "phase_elapsed_steps": int(step - phase_start_step + 1),
                "phase_max_steps": (
                    int(phase_specs[subgoal_index]["max_steps"])
                    if phase_contract_mode
                    else None
                ),
                "phase_hold_streak": int(phase_hold_streak),
                "policy_inference": bool(policy_inference),
                "action_chunk_inference_mode": action_chunk_inference_mode,
                "temporal_ensemble_enabled": bool(temporal_ensemble),
                "temporal_ensemble_source_count": int(temporal_source_count),
                "rtc_guidance_applied": bool(rtc_guidance_applied),
                "rtc_leftover_steps": int(rtc_leftover_steps),
                "rtc_execution_horizon": (
                    int(rtc_settings["execution_horizon"])
                    if rtc_settings is not None
                    else None
                ),
                "rtc_inference_delay": (
                    int(rtc_settings["inference_delay"])
                    if rtc_settings is not None
                    else None
                ),
                "requery_overlap_rmse": requery_overlap_rmse,
                "valid_mask_probs": valid_mask_probs,
                "valid_mask_predicted_horizon": valid_mask_predicted_horizon,
                "valid_mask_horizon_reason": valid_mask_horizon_reason,
                "valid_mask_stop_streak": int(valid_mask_stop_streak),
                "valid_mask_requery_confirmations": int(valid_mask_requery_confirmations),
            }
        )
        if active_chain:
            horizon_remaining -= 1
        if not phase_contract_mode and bool(info.get("success", False)):
            termination_reason = "env_success"
            break
        if not phase_contract_mode and terminated:
            termination_reason = "env_terminated"
            break
        if not phase_contract_mode and truncated:
            termination_reason = "env_truncated"
            break
        if (
            not phase_contract_mode
            and valid_mask_stop_after_step is not None
            and step + 1 >= valid_mask_stop_after_step
        ):
            termination_reason = "valid_mask_stop"
            break
    else:
        hard_cap_scope = "chain"
        if phase_contract_mode and phase_specs:
            verifier_result = _evaluate_phase_verifier(
                env=env,
                phase=phase_specs[subgoal_index],
                hold_streak=phase_hold_streak,
            )
            final_phase = subgoal_index == len(phase_specs) - 1
            verifier_passed = bool(verifier_result["passed"])
            if final_phase and verifier_passed:
                termination_reason = "env_success"
                phase_contract_success = True
            phase_results.append(
                _phase_result(
                    phase=phase_specs[subgoal_index],
                    start_step=phase_start_step,
                    end_step=max_steps,
                    verifier_result=verifier_result,
                    termination_reason=(
                        "env_success"
                        if final_phase and verifier_passed
                        else "hard_cap"
                    ),
                )
            )
    if live_observation_renderer is not None and record_rollout_gif and records:
        final_step = len(records) - 1
        last_render_step = int(observation_render_records[-1]["policy_inference_step"])
        if last_render_step != final_step:
            final_pixels, final_render_record = live_observation_renderer.render(
                env=env,
                mujoco_renderers=renderers,
                episode=episode,
                seed=int(reset_meta.get("reset_seed") or seed),
                step=final_step,
            )
            observation_render_records.append(final_render_record)
            frames.append(_render_policy_camera_pair(final_pixels))
            final_media_paths = _write_policy_trace_images(
                camera_pixels=final_pixels,
                output_dir=output_dir,
                episode=episode,
                seed=seed,
                step=final_step,
                enabled=True,
            )
            trace_rows.append(
                {
                    "episode": int(episode),
                    "global_step": int(final_step),
                    "prompt": current_task_prompt,
                    "phase": current_subgoal,
                    "phase_index": int(subgoal_index),
                    "policy_inference": False,
                    "visualization_only": True,
                    "termination_reason": termination_reason,
                    "image_feature_mapping": dict(POLICY_DISPLAY_IMAGE_FEATURE_MAPPING),
                    "media": {"policy_input_images": final_media_paths},
                }
            )
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
    if phase_contract_mode:
        skill_success = bool(phase_contract_success)
    elif eval_skill_mode == "pick_and_place_cube":
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
        "phase_contract": _phase_contract_episode_report(phase_contract),
        "phase_results": phase_results,
        "phase_stop_proposals": phase_stop_proposals,
        "action_chunk_inference": _action_chunk_inference_report(
            mode=action_chunk_inference_mode,
            temporal_ensemble_decay=temporal_ensemble_decay,
            rtc_settings=rtc_settings,
        ),
        "reset_meta": reset_meta,
        "search_steps": search_steps,
        "steps": len(records),
        "termination": {
            "reason": termination_reason,
            "hard_cap_steps": int(max_steps),
            "hard_cap_scope": hard_cap_scope if termination_reason == "hard_cap" else None,
            "valid_mask_enabled": bool(valid_mask_mode),
            "valid_mask_requery_confirmations": int(valid_mask_requery_confirmations),
            "final_valid_mask_stop_streak": int(valid_mask_stop_streak),
        },
        "success": bool(phase_contract_success) if phase_contract_mode else bool(info.get("success", False)),
        "raw_env_success": bool(info.get("success", False)),
        "skill_success": skill_success,
        "final_is_grasped": final_is_grasped,
        "final_is_obj_placed": final_is_obj_placed,
        "final_lift_height": final_lift_height,
        "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        "final_obj_to_target_dist": final_obj_to_target_dist,
        "image_feature_mapping": image_feature_mapping,
        "observation_renderer": {
            "mode": "blender_cycles_live" if live_observation_renderer is not None else "mujoco",
            "render_policy_inference_only": (
                bool(getattr(live_observation_renderer, "render_policy_inference_only", True))
                if live_observation_renderer is not None
                else False
            ),
            "frame_cadence": (
                "policy_inference_only"
                if live_observation_renderer is not None
                and bool(getattr(live_observation_renderer, "render_policy_inference_only", True))
                else "every_environment_step"
            ),
            "render_count": len(observation_render_records),
            "render_seconds": float(
                sum(float(record.get("render_seconds", 0.0)) for record in observation_render_records)
            ),
        },
        "trace_path": written_trace_path,
        "rollout_gif": gif_path,
        "rollout_mp4": mp4_path,
        "input_grid_paths": input_grid_paths,
        "records": records,
    }


def _live_observation_render_due(
    *,
    render_policy_inference_only: bool,
    has_last_camera_pixels: bool,
    step: int,
    next_policy_inference_step: int,
    has_temporal_chunks: bool,
    subgoal_advance_due: bool,
) -> bool:
    if not render_policy_inference_only:
        return True
    return bool(
        not has_last_camera_pixels
        or step >= next_policy_inference_step
        or not has_temporal_chunks
        or subgoal_advance_due
    )


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


def _predict_action_chunk_with_processors(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    qpos: np.ndarray,
    camera_pixels: dict[str, np.ndarray],
    task_prompt: str,
) -> np.ndarray:
    processed_chunk, _state, _raw_chunk = _predict_action_chunk_with_processors_and_inputs(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        qpos=qpos,
        camera_pixels=camera_pixels,
        task_prompt=task_prompt,
    )
    return processed_chunk


def _predict_action_chunk_with_processors_and_inputs(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    qpos: np.ndarray,
    camera_pixels: dict[str, np.ndarray],
    task_prompt: str,
    policy_predict_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, Any, Any]:
    try:
        from lerobot.utils.control_utils import prepare_observation_for_inference
    except ModuleNotFoundError:
        from lerobot.common.control_utils import prepare_observation_for_inference

    selected_device = str(
        _policy_device_metadata(policy).get("device_selected")
        or getattr(policy.config, "device", "cpu")
    )
    observation = {
        "observation.state": np.asarray(qpos, dtype=np.float32),
        "observation.images.camera1": np.asarray(camera_pixels["egocentric_cam"], dtype=np.uint8),
        "observation.images.camera2": np.asarray(camera_pixels["wrist_cam"], dtype=np.uint8),
        "observation.images.camera3": np.asarray(camera_pixels["wrist_cam"], dtype=np.uint8),
    }
    observation = prepare_observation_for_inference(
        observation,
        torch.device(selected_device),
        task_prompt,
        "so101",
    )
    batch = preprocessor(observation)
    inference_context = torch.no_grad if policy_predict_kwargs else torch.inference_mode
    with inference_context():
        if policy_predict_kwargs:
            raw_chunk = policy.predict_action_chunk(batch, **policy_predict_kwargs)
        else:
            raw_chunk = policy.predict_action_chunk(batch)
    try:
        processed_chunk = postprocessor(raw_chunk)
    except Exception:  # Some LeRobot processor versions only accept one action at a time.
        chunk = torch.as_tensor(raw_chunk)
        if chunk.ndim != 3:
            raise
        pieces = [postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])]
        processed_chunk = torch.stack([torch.as_tensor(piece) for piece in pieces], dim=1)
    chunk_array = np.asarray(torch.as_tensor(processed_chunk).detach().float().cpu())
    if chunk_array.ndim == 3 and chunk_array.shape[0] == 1:
        chunk_array = chunk_array[0]
    if chunk_array.ndim != 2:
        raise ValueError(f"postprocessed action chunk must have shape [T, A], got {chunk_array.shape}")
    return chunk_array[:, :6].astype(float, copy=False), batch["observation.state"], raw_chunk


def _resolve_action_chunk_inference_mode(
    requested_mode: str | None,
    *,
    legacy_temporal_ensemble: bool,
) -> str:
    if requested_mode is None:
        return "temporal_ensemble" if legacy_temporal_ensemble else "policy_queue"
    if requested_mode not in {"policy_queue", "temporal_ensemble", "rtc"}:
        raise ValueError(f"unsupported action chunk inference mode: {requested_mode}")
    if legacy_temporal_ensemble and requested_mode != "temporal_ensemble":
        raise ValueError(
            "--temporal-ensemble conflicts with "
            f"--action-chunk-inference-mode={requested_mode}"
        )
    return requested_mode


def _validate_rtc_settings(
    *,
    mode: str,
    prefix_attention_schedule: str | None,
    max_guidance_weight: float | None,
    execution_horizon: int | None,
    inference_delay: int | None,
    debug: bool | None,
    debug_maxlen: int | None,
) -> dict[str, Any] | None:
    values = {
        "prefix_attention_schedule": prefix_attention_schedule,
        "max_guidance_weight": max_guidance_weight,
        "execution_horizon": execution_horizon,
        "inference_delay": inference_delay,
        "debug": debug,
        "debug_maxlen": debug_maxlen,
    }
    if mode != "rtc":
        return None
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError("RTC mode requires explicit settings: " + ", ".join(missing))
    if prefix_attention_schedule not in {"ZEROS", "ONES", "LINEAR", "EXP"}:
        raise ValueError(
            f"unsupported RTC prefix attention schedule: {prefix_attention_schedule}"
        )
    if float(max_guidance_weight) <= 0:
        raise ValueError("rtc_max_guidance_weight must be positive")
    if int(execution_horizon) < 1:
        raise ValueError("rtc_execution_horizon must be positive")
    if int(inference_delay) < 0:
        raise ValueError("rtc_inference_delay must be non-negative")
    if int(debug_maxlen) < 1:
        raise ValueError("rtc_debug_maxlen must be positive")
    return {
        "prefix_attention_schedule": str(prefix_attention_schedule),
        "max_guidance_weight": float(max_guidance_weight),
        "execution_horizon": int(execution_horizon),
        "inference_delay": int(inference_delay),
        "debug": bool(debug),
        "debug_maxlen": int(debug_maxlen),
    }


def _configure_policy_rtc(
    policy: Any,
    *,
    mode: str,
    settings: dict[str, Any] | None,
) -> Any | None:
    if mode != "rtc":
        return None
    if settings is None:
        raise ValueError("RTC settings are required")
    config = getattr(policy, "config", None)
    if config is None or not hasattr(policy, "init_rtc_processor"):
        raise ValueError("loaded policy does not support LeRobot RTC inference")
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc import RTCConfig

    rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule[
            settings["prefix_attention_schedule"]
        ],
        max_guidance_weight=float(settings["max_guidance_weight"]),
        execution_horizon=int(settings["execution_horizon"]),
        debug=bool(settings["debug"]),
        debug_maxlen=int(settings["debug_maxlen"]),
    )
    config.rtc_config = rtc_config
    policy.init_rtc_processor()
    return rtc_config


def _make_rtc_action_queue(rtc_policy_config: Any) -> Any:
    from lerobot.policies.rtc import ActionQueue

    return ActionQueue(rtc_policy_config)


def _clear_rtc_action_queue(action_queue: Any | None) -> None:
    if action_queue is not None:
        action_queue.clear()


def _rtc_policy_predict_kwargs(
    *,
    previous_leftover: Any | None,
    settings: dict[str, Any] | None,
) -> dict[str, Any]:
    if settings is None:
        raise ValueError("RTC settings are required")
    return {
        "prev_chunk_left_over": previous_leftover,
        "inference_delay": int(settings["inference_delay"]),
        "execution_horizon": int(settings["execution_horizon"]),
    }


def _rtc_unbatched_chunk(raw_chunk: Any) -> torch.Tensor:
    chunk = torch.as_tensor(raw_chunk).detach()
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2:
        raise ValueError(
            f"RTC raw action chunk must have shape [T, A], got {tuple(chunk.shape)}"
        )
    return chunk


def _rtc_processed_overlap_rmse(
    *,
    previous_leftover: Any | None,
    new_chunk: np.ndarray,
    max_horizon: int,
) -> float | None:
    if previous_leftover is None:
        return None
    previous = np.asarray(torch.as_tensor(previous_leftover).detach().float().cpu())
    horizon = min(int(max_horizon), int(previous.shape[0]), int(new_chunk.shape[0]))
    if horizon <= 0:
        return None
    delta = previous[:horizon] - np.asarray(new_chunk[:horizon], dtype=float)
    return float(np.sqrt(np.mean(np.square(delta))))


def _action_chunk_inference_report(
    *,
    mode: str,
    temporal_ensemble_decay: float,
    rtc_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "temporal_ensemble_decay": (
            float(temporal_ensemble_decay)
            if mode == "temporal_ensemble"
            else None
        ),
        "rtc": dict(rtc_settings) if rtc_settings is not None else None,
        "reference": (
            "LeRobot RTCProcessor raw-chunk prefix guidance"
            if mode == "rtc"
            else None
        ),
    }


def _temporal_ensemble_action(
    chunks: list[tuple[int, np.ndarray]],
    *,
    step: int,
    decay: float,
) -> tuple[np.ndarray, int]:
    candidates = [
        np.asarray(chunk[int(step) - int(start)], dtype=float)
        for start, chunk in chunks
        if int(start) <= int(step) < int(start) + int(chunk.shape[0])
    ]
    if not candidates:
        raise ValueError(f"no temporal-ensemble action covers step {step}")
    weights = np.exp(-float(decay) * np.arange(len(candidates), dtype=float))
    weights /= weights.sum()
    action = np.sum(np.stack(candidates, axis=0) * weights[:, None], axis=0)
    return action, len(candidates)


def _action_chunk_overlap_rmse(
    *,
    previous_start: int,
    previous_chunk: np.ndarray,
    new_start: int,
    new_chunk: np.ndarray,
    max_horizon: int,
) -> float | None:
    previous_offset = int(new_start) - int(previous_start)
    if previous_offset < 0 or previous_offset >= int(previous_chunk.shape[0]):
        return None
    horizon = min(
        max(0, int(max_horizon)),
        int(previous_chunk.shape[0]) - previous_offset,
        int(new_chunk.shape[0]),
    )
    if horizon <= 0:
        return None
    delta = previous_chunk[previous_offset : previous_offset + horizon] - new_chunk[:horizon]
    return float(np.sqrt(np.mean(np.square(delta))))


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


def _make_live_observation_renderer(
    *,
    output_dir: Path,
    width: int,
    height: int,
    config: dict[str, Any] | None,
) -> Any | None:
    if config is None or config.get("mode") == "mujoco":
        return None
    if config.get("mode") != "blender_cycles_live":
        raise ValueError(f"unsupported observation renderer mode: {config.get('mode')!r}")
    if int(config["width"]) != int(width) or int(config["height"]) != int(height):
        raise ValueError(
            "live observation renderer resolution must match closed-loop policy resolution: "
            f"renderer={config['width']}x{config['height']} policy={width}x{height}"
        )
    if not isinstance(config.get("render_policy_inference_only"), bool):
        raise ValueError("blender_cycles_live requires boolean render_policy_inference_only")
    expected_cameras = ["observation.images.camera1", "observation.images.camera2"]
    if list(config.get("camera_keys") or []) != expected_cameras:
        raise ValueError(f"blender_cycles_live camera_keys must be {expected_cameras}")
    from render_so101_dataset_blender_preview import LiveBlenderCyclesPolicyRenderer

    return LiveBlenderCyclesPolicyRenderer(
        output_dir=output_dir / "photoreal_policy_inputs",
        config=config,
    )


def _render_policy_camera_pair(camera_pixels: dict[str, np.ndarray]) -> np.ndarray:
    camera1 = np.asarray(camera_pixels["egocentric_cam"], dtype=np.uint8)
    camera2 = np.asarray(camera_pixels["wrist_cam"], dtype=np.uint8)
    if camera1.shape[0] != camera2.shape[0]:
        raise ValueError(f"policy camera heights differ: {camera1.shape} vs {camera2.shape}")
    return np.concatenate([camera1, camera2], axis=1)


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
