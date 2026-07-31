#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from scripts.validate_mycobot280_training_dataset import validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or run a myCobot 280 SmolVLA closed-loop simulation evaluation."
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-repo-id")
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=530)
    parser.add_argument("--seed-start", type=int, default=91000)
    parser.add_argument("--torch-seed", type=int, default=20260731)
    parser.add_argument("--yaw-min", type=float, default=-0.20)
    parser.add_argument("--yaw-max", type=float, default=0.20)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--render-camera-profile",
        choices=("full_robot", "ground_pickup_closeup"),
        default=None,
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--record-representative-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-policy", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        report = build_eval_report(
            policy_path=Path(args.policy_path),
            config_path=args.config,
            output_dir=args.output_dir,
            episodes=args.episodes,
            dry_run=True,
            require_policy=args.require_policy,
            render_camera_profile=args.render_camera_profile,
        )
    else:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for closed-loop execution")
        report = evaluate_closed_loop(
            policy_path=args.policy_path,
            config_path=args.config,
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            dataset_repo_id=args.dataset_repo_id,
            asset_root=args.asset_root,
            official_gripper_root=args.official_gripper_root,
            episodes=args.episodes,
            steps=args.steps,
            seed_start=args.seed_start,
            torch_seed=args.torch_seed,
            yaw_min=args.yaw_min,
            yaw_max=args.yaw_max,
            width=args.width,
            height=args.height,
            render_camera_profile=args.render_camera_profile,
            device=args.device,
            local_files_only=args.local_files_only,
            record_representative_frames=args.record_representative_frames,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "eval_report.json"
    report["report_path"] = str(report_path.resolve())
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] in {"planned", "blocked", "completed"} else 1)


def build_eval_report(
    *,
    policy_path: Path,
    config_path: Path,
    output_dir: Path,
    episodes: int | None,
    dry_run: bool,
    require_policy: bool,
    render_camera_profile: str | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved_camera_profile = _resolve_render_camera_profile(config, render_camera_profile)
    validation = validate_config(config_path=config_path, require_present=False)
    closed_loop = config["closed_loop_stub"]
    requested_episodes = int(episodes if episodes is not None else closed_loop.get("episodes", 3))
    policy_exists = policy_path.exists()
    status = "planned" if dry_run or policy_exists else "blocked"
    if require_policy and not policy_exists:
        status = "blocked"
    return {
        "operation": "evaluate_mycobot280_smolvla_policy",
        "status": status,
        "policy_path": str(policy_path),
        "policy_exists": policy_exists,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "episodes": requested_episodes,
        "validation_status": validation["status"],
        "blocker": None if status == "planned" else "policy checkpoint does not exist yet; run tiny smoke/fine-tune first",
        "scenario": config["scenario"],
        "task_prompt": config["task_prompt"],
        "metrics": closed_loop["metrics"],
        "comparison_rows": closed_loop["future_comparison_rows"],
        "planned_execution": {
            "mode": "policy_only_closed_loop_sim",
            "robot": config["robot"]["name"],
            "state_dim": config["robot"]["state_dim"],
            "action_dim": config["robot"]["action_dim"],
            "render_camera_profile": resolved_camera_profile,
            "success_source": "myCobot 280 ground-pickup contact/lift/hold verifier",
        },
        "claim_boundary": (
            "Closed-loop evaluation plan only; no policy rollout was executed by this report."
        ),
    }



def evaluate_closed_loop(
    *,
    policy_path: str,
    config_path: Path,
    output_dir: Path,
    dataset_root: Path,
    dataset_repo_id: str | None,
    asset_root: Path,
    official_gripper_root: Path,
    episodes: int | None,
    steps: int,
    seed_start: int,
    torch_seed: int,
    yaw_min: float,
    yaw_max: float,
    width: int,
    height: int,
    render_camera_profile: str | None,
    device: str,
    local_files_only: bool,
    record_representative_frames: bool,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    closed_loop = config["closed_loop_stub"]
    resolved_camera_profile = _resolve_render_camera_profile(config, render_camera_profile)
    episode_count = int(episodes if episodes is not None else closed_loop.get("episodes", 3))
    if episode_count <= 0:
        raise ValueError("--episodes must be positive")
    resolved_repo_id = dataset_repo_id or config["lerobot_conversion"]["repo_id"]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    policy, preprocessor, postprocessor, policy_metadata = _load_policy_runtime(
        policy_path=policy_path,
        dataset_root=dataset_root.resolve(),
        dataset_repo_id=resolved_repo_id,
        device=device,
        local_files_only=local_files_only,
    )
    schedule = [
        {
            "episode": index,
            "seed": int(seed_start) + index,
            "torch_seed": int(torch_seed) + index,
            "yaw_delta_rad": yaw,
        }
        for index, yaw in enumerate(_yaw_schedule(episode_count, yaw_min, yaw_max))
    ]
    episode_summaries = _execute_schedule(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        policy_metadata=policy_metadata,
        task_prompt=config["task_prompt"],
        output_dir=output_dir,
        asset_root=asset_root.resolve(),
        official_gripper_root=official_gripper_root.resolve(),
        schedule=schedule,
        steps=steps,
        width=width,
        height=height,
        render_camera_profile=resolved_camera_profile,
        record_representative_frames=record_representative_frames,
    )
    return {
        "operation": "evaluate_mycobot280_smolvla_policy",
        "status": "completed",
        "mode": "policy_only_closed_loop_sim",
        "policy_path": policy_path,
        "config_path": str(config_path.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_repo_id": resolved_repo_id,
        "output_dir": str(output_dir),
        "task_prompt": config["task_prompt"],
        "episodes_requested": episode_count,
        "steps_per_episode": int(steps),
        "schedule": schedule,
        "episode_summaries": episode_summaries,
        "aggregate": _aggregate_episode_summaries(episode_summaries),
        "policy_runtime": policy_metadata,
        "environment": {
            "object_physics": "fixed",
            "teacher_attachment_enabled": False,
            "object_teleport_during_rollout": False,
            "gravity_schedule": "zero for steps 0-119; normal for steps 120-529",
            "state_input": "7D arm qpos plus gripper command",
            "camera_input": (
                f"observation.images.camera1 rendered at {width}x{height} RGB"
            ),
            "render_camera_profile": resolved_camera_profile,
            "excluded_policy_inputs": ["cube pose", "contact metrics", "MuJoCo state"],
        },
        "duration_s": round(perf_counter() - started, 4),
        "claim_boundary": (
            "This report executes policy-only closed-loop simulation. A pilot or small "
            "episode count is engineering evidence, not a publication-level success estimate."
        ),
    }


def _load_policy_runtime(
    *,
    policy_path: str,
    dataset_root: Path,
    dataset_repo_id: str,
    device: str,
    local_files_only: bool,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    from physical_ai_agent.policies.mycobot280_smolvla_contract import (
        make_mycobot280_pre_post_processors,
    )
    from physical_ai_agent.policies.smolvla_real import (
        _load_pretrained_policy,
        _policy_device_metadata,
    )

    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = _load_pretrained_policy(
        model_id=policy_path,
        local_files_only=local_files_only,
        device=device,
    )
    device_metadata = _policy_device_metadata(policy)
    selected_device = str(device_metadata.get("device_selected") or device)
    policy.config.device = selected_device
    policy.to(selected_device)
    policy.eval()
    preprocessor, postprocessor, contract = make_mycobot280_pre_post_processors(
        policy=policy,
        dataset_meta=metadata,
        policy_path=policy_path,
        selected_device=selected_device,
    )
    return policy, preprocessor, postprocessor, {
        "device": device_metadata,
        "contract": contract,
        "n_action_steps": int(getattr(policy.config, "n_action_steps", 1)),
        "chunk_size": int(getattr(policy.config, "chunk_size", 1)),
    }


def _execute_schedule(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    policy_metadata: dict[str, Any],
    task_prompt: str,
    output_dir: Path,
    asset_root: Path,
    official_gripper_root: Path,
    schedule: list[dict[str, Any]],
    steps: int,
    width: int,
    height: int,
    render_camera_profile: str,
    record_representative_frames: bool,
) -> list[dict[str, Any]]:
    from scripts.export_mycobot_280_ground_pickup_teacher_dataset import _make_env
    from scripts.run_mycobot_280_ground_pickup_poc import (
        CUBE_HALF_SIZE,
        _apply_physics_overrides,
        _patch_nexus_work_mat_scene_nodes,
    )
    from scripts.render_mycobot_280_cube_contact_sequence import _size_audit_cube

    _patch_nexus_work_mat_scene_nodes()
    env = _make_env(
        asset_root=asset_root,
        official_gripper_root=official_gripper_root,
        work_dir=output_dir / "scene",
        width=width,
        height=height,
    )
    summaries: list[dict[str, Any]] = []
    try:
        for item in schedule:
            env.reset(seed=int(item["seed"]))
            env._diagnostic_cube_half_size = CUBE_HALF_SIZE
            _size_audit_cube(env, half_size=CUBE_HALF_SIZE)
            _apply_physics_overrides(env)
            summaries.append(
                _run_episode(
                    env=env,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    policy_metadata=policy_metadata,
                    task_prompt=task_prompt,
                    output_dir=output_dir,
                    episode=int(item["episode"]),
                    seed=int(item["seed"]),
                    torch_seed=int(item["torch_seed"]),
                    yaw_delta=float(item["yaw_delta_rad"]),
                    steps=steps,
                    render_camera_profile=render_camera_profile,
                    record_frames=record_representative_frames and int(item["episode"]) == 0,
                )
            )
    finally:
        env.close()
    return summaries


def _run_episode(
    *,
    env: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    policy_metadata: dict[str, Any],
    task_prompt: str,
    output_dir: Path,
    episode: int,
    seed: int,
    torch_seed: int,
    yaw_delta: float,
    steps: int,
    render_camera_profile: str,
    record_frames: bool,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from scripts.export_mycobot_280_ground_pickup_teacher_dataset import (
        _candidate_qpos,
        _dataset_render_camera,
        _initial_cube_pose_for_qpos,
        _render_observation,
        _scripted_state_for_qpos,
    )
    from scripts.run_mycobot_280_ground_pickup_poc import (
        APPROACH_STEPS,
        CLOSE_STEPS,
        CUBE_HALF_SIZE,
        POST_LIFT_HOLD_STEPS,
        START_COMMAND,
        WORLD_GRAVITY,
        _best_sustained_two_pad,
        _cube_mat_guard,
        _record,
    )
    from scripts.render_mycobot_280_cube_contact_sequence import _set_cube_pose

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    if hasattr(policy, "reset"):
        policy.reset()

    pickup_qpos, lift_qpos = _candidate_qpos(yaw_delta)
    cube_pos, cube_quat = _initial_cube_pose_for_qpos(env, pickup_qpos)
    env._set_gripper(command=START_COMMAND)
    _set_cube_pose(env, cube_pos, cube_quat)
    env._mujoco.mj_forward(env.model, env.data)
    initial_cube = np.asarray(env._cube_position(), dtype=float)
    env._cube_initial_pos = [float(value) for value in initial_cube]
    render_camera = _dataset_render_camera(env, initial_cube, render_camera_profile)
    placement_guard = _cube_mat_guard(initial_cube)
    if not placement_guard["passed"]:
        return _runtime_failure_summary(
            episode=episode,
            seed=seed,
            torch_seed=torch_seed,
            yaw_delta=yaw_delta,
            reason="cube_placement_guard_failed",
            detail=json.dumps(placement_guard, sort_keys=True),
        )

    initial_setup_record = _record(
        env,
        step=-1,
        phase="pre_policy_setup",
        command=START_COMMAND,
        initial_cube=initial_cube,
    )
    observation = env._observation(gripper=START_COMMAND)
    records: list[dict[str, Any]] = []
    trace_path = output_dir / "traces" / f"episode_{episode:03d}_seed_{seed}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    episode_frame_dir = output_dir / "representative_frames" / f"episode_{episode:03d}"
    if record_frames:
        episode_frame_dir.mkdir(parents=True, exist_ok=True)
    key_steps = {0, APPROACH_STEPS + CLOSE_STEPS - 1, 149, 229, steps - 1}
    trace_file = trace_path.open("w", encoding="utf-8")
    inference_error: str | None = None
    clipped_values = 0
    terminated_ignored = 0
    try:
        for step in range(steps):
            _arm, _command, phase = _scripted_state_for_qpos(
                step,
                pickup_qpos=pickup_qpos,
                lift_qpos=lift_qpos,
            )
            if step < APPROACH_STEPS + CLOSE_STEPS:
                env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
            else:
                env.model.opt.gravity[:] = WORLD_GRAVITY
            pixels = _render_observation(env, render_camera)
            if record_frames and step in key_steps:
                _write_rgb_image(pixels, episode_frame_dir / f"step_{step:04d}_{phase}.png")
            try:
                raw_action = _predict_action(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    policy_metadata=policy_metadata,
                    observation=observation[:7],
                    pixels=pixels,
                    task_prompt=task_prompt,
                )
                action, clipped = _clip_policy_action(
                    raw_action,
                    arm_low=env._low,
                    arm_high=env._high,
                )
            except Exception as exc:  # noqa: BLE001
                inference_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:800]
                break
            clipped_values += clipped
            observation, reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                terminated_ignored += 1
            record = _record(
                env,
                step=step,
                phase=phase,
                command=float(action[-1]),
                initial_cube=initial_cube,
            )
            records.append(record)
            trace_file.write(
                json.dumps(
                    {
                        "step": step,
                        "phase": phase,
                        "observation_state": [float(value) for value in observation[:7]],
                        "action": action,
                        "reward": float(reward),
                        "terminated_ignored": bool(terminated),
                        "truncated_ignored": bool(truncated),
                        "ground_pickup": record,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        trace_file.close()

    if record_frames:
        _write_rgb_image(
            _render_observation(env, render_camera), episode_frame_dir / "final.png"
        )
    if inference_error is not None or len(records) != steps:
        return _runtime_failure_summary(
            episode=episode,
            seed=seed,
            torch_seed=torch_seed,
            yaw_delta=yaw_delta,
            reason="policy_inference_failed" if inference_error else "incomplete_rollout",
            detail=inference_error or f"completed {len(records)} of {steps} steps",
            trace_path=str(trace_path),
            steps=len(records),
        )

    lift_records = [record for record in records if record["phase"] == "lift_from_mat"]
    post_records = [record for record in records if record["phase"] == "post_lift_hold"]
    final = records[-1]
    max_penetration = max(
        float(record["pad_cube_contact_depth"]["max_penetration_m"])
        for record in records
    )
    lift_contact_steps = _best_sustained_two_pad(lift_records)
    post_contact_steps = _best_sustained_two_pad(post_records)
    post_min_lift = min(
        (float(record["cube_lift_m"]) for record in post_records),
        default=0.0,
    )
    guard_records = [initial_setup_record, *records]
    failed_gates = _failed_gates(
        placement_guard=placement_guard,
        cube_mat_guard_passed=all(
            bool(record["mat_guard"]["bottom_on_or_above_mat"]) for record in guard_records
        ),
        pad_mat_guard_passed=all(
            bool(record["pad_mat_guard"]["passed"]) for record in guard_records
        ),
        gripper_visual_mat_guard_passed=all(
            bool(record["gripper_visual_mat_guard"]["passed"]) for record in guard_records
        ),
        initial_contact_pads=int(initial_setup_record["pad_cube_contacted_pads"]),
        max_penetration=max_penetration,
        final_lift=float(final["cube_lift_m"]),
        final_contact_pads=int(final["pad_cube_contacted_pads"]),
        lift_contact_steps=lift_contact_steps,
        post_contact_steps=post_contact_steps,
        post_min_lift=post_min_lift,
        post_window_complete=len(post_records) >= min(POST_LIFT_HOLD_STEPS, steps),
    )
    success = not failed_gates
    failure_reason = "passed" if success else failed_gates[0]
    return {
        "episode": episode,
        "seed": seed,
        "torch_seed": torch_seed,
        "yaw_delta_rad": yaw_delta,
        "success": success,
        "failed_gates": failed_gates,
        "failure_reason": failure_reason,
        "steps": len(records),
        "initial_cube_position": [float(value) for value in initial_cube],
        "initial_pad_cube_contacted_pads": int(initial_setup_record["pad_cube_contacted_pads"]),
        "final_cube_position": [float(value) for value in final["cube_pos"]],
        "final_cube_lift_m": float(final["cube_lift_m"]),
        "final_pad_cube_contacted_pads": int(final["pad_cube_contacted_pads"]),
        "lift_best_sustained_two_pad_steps": int(lift_contact_steps),
        "post_lift_hold_best_sustained_two_pad_steps": int(post_contact_steps),
        "post_lift_hold_min_cube_lift_m": float(post_min_lift),
        "max_pad_cube_penetration_m": float(max_penetration),
        "clipped_action_values": int(clipped_values),
        "terminated_or_truncated_signals_ignored": int(terminated_ignored),
        "trace_path": str(trace_path),
        "representative_frame_dir": str(episode_frame_dir) if record_frames else None,
    }


def _predict_action(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    policy_metadata: dict[str, Any],
    observation: list[float],
    pixels: Any,
    task_prompt: str,
) -> list[float]:
    import numpy as np
    import torch

    try:
        from lerobot.utils.control_utils import predict_action
    except ModuleNotFoundError:
        from lerobot.common.control_utils import predict_action

    selected_device = str(
        policy_metadata["device"].get("device_selected")
        or getattr(policy.config, "device", "cpu")
    )
    action = predict_action(
        observation={
            "observation.state": np.asarray(observation, dtype=np.float32),
            "observation.images.camera1": np.asarray(pixels, dtype=np.uint8),
        },
        policy=policy,
        device=torch.device(selected_device),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task=task_prompt,
        robot_type="mycobot_280_pi_adaptive_gripper",
    )
    return [float(value) for value in torch.as_tensor(action).detach().cpu().reshape(-1)]


def _clip_policy_action(
    action: list[float],
    *,
    arm_low: Any,
    arm_high: Any,
) -> tuple[list[float], int]:
    if len(action) != 7:
        raise ValueError(f"myCobot policy must return exactly 7 actions, got {len(action)}")
    if not all(math.isfinite(float(value)) for value in action):
        raise ValueError("myCobot policy returned a non-finite action")
    clipped: list[float] = []
    clipped_values = 0
    for value, low, high in zip(action[:6], arm_low, arm_high, strict=True):
        bounded = max(float(low), min(float(high), float(value)))
        clipped_values += int(bounded != float(value))
        clipped.append(bounded)
    gripper = max(-1.0, min(1.0, float(action[6])))
    clipped_values += int(gripper != float(action[6]))
    return [*clipped, gripper], clipped_values


def _failure_reason(
    *,
    success: bool,
    placement_guard: dict[str, Any],
    max_penetration: float,
    final_lift: float,
    final_contact_pads: int,
    lift_contact_steps: int,
    post_contact_steps: int,
    post_min_lift: float,
) -> str:
    if success:
        return "passed"
    failed_gates = _failed_gates(
        placement_guard=placement_guard,
        max_penetration=max_penetration,
        final_lift=final_lift,
        final_contact_pads=final_contact_pads,
        lift_contact_steps=lift_contact_steps,
        post_contact_steps=post_contact_steps,
        post_min_lift=post_min_lift,
    )
    return failed_gates[0] if failed_gates else "verifier_failed"


def _failed_gates(
    *,
    placement_guard: dict[str, Any],
    max_penetration: float,
    final_lift: float,
    final_contact_pads: int,
    lift_contact_steps: int,
    post_contact_steps: int,
    post_min_lift: float,
    cube_mat_guard_passed: bool = True,
    pad_mat_guard_passed: bool = True,
    gripper_visual_mat_guard_passed: bool = True,
    initial_contact_pads: int = 0,
    post_window_complete: bool = True,
) -> list[str]:
    failures: list[str] = []
    if not placement_guard.get("passed", False):
        failures.append("cube_placement_guard_failed")
    if not cube_mat_guard_passed:
        failures.append("cube_mat_clearance_guard_failed")
    if not pad_mat_guard_passed:
        failures.append("pad_mat_clearance_guard_failed")
    if not gripper_visual_mat_guard_passed:
        failures.append("gripper_visual_mat_clearance_guard_failed")
    if initial_contact_pads != 0:
        failures.append("initial_pad_cube_contact_present")
    if max_penetration > 0.003:
        failures.append("max_pad_cube_penetration_exceeded")
    if final_lift < 0.05:
        failures.append("final_cube_lift_below_threshold")
    if final_contact_pads < 2:
        failures.append("final_two_pad_contact_missing")
    if lift_contact_steps < 60:
        failures.append("lift_two_pad_contact_too_short")
    if not post_window_complete:
        failures.append("post_lift_hold_window_incomplete")
    if post_contact_steps < 300:
        failures.append("post_lift_two_pad_contact_too_short")
    if post_min_lift < 0.045:
        failures.append("post_lift_height_below_threshold")
    return failures


def _runtime_failure_summary(
    *,
    episode: int,
    seed: int,
    torch_seed: int,
    yaw_delta: float,
    reason: str,
    detail: str,
    trace_path: str | None = None,
    steps: int = 0,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "seed": seed,
        "torch_seed": torch_seed,
        "yaw_delta_rad": yaw_delta,
        "success": False,
        "failure_reason": reason,
        "failure_detail": detail,
        "steps": steps,
        "trace_path": trace_path,
    }


def _aggregate_episode_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row.get("success")) for row in rows)
    valid_lifts = [
        float(row["final_cube_lift_m"])
        for row in rows
        if row.get("final_cube_lift_m") is not None
    ]
    penetrations = [
        float(row["max_pad_cube_penetration_m"])
        for row in rows
        if row.get("max_pad_cube_penetration_m") is not None
    ]
    failure_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("failure_reason", "unknown"))
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return {
        "episodes": len(rows),
        "successful_episodes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "mean_final_cube_lift_m": (
            sum(valid_lifts) / len(valid_lifts) if valid_lifts else None
        ),
        "max_pad_cube_penetration_m": max(penetrations) if penetrations else None,
        "failure_reason_counts": failure_counts,
    }


def _resolve_render_camera_profile(
    config: dict[str, Any],
    override: str | None,
) -> str:
    closed_loop = config.get("closed_loop_stub", {})
    configured = (
        closed_loop.get("render_camera_profile")
        if isinstance(closed_loop, dict)
        else None
    )
    profile = override or configured or "full_robot"
    if profile not in {"full_robot", "ground_pickup_closeup"}:
        raise ValueError(f"unsupported render camera profile: {profile}")
    return str(profile)


def _yaw_schedule(episodes: int, yaw_min: float, yaw_max: float) -> list[float]:
    if episodes <= 0:
        return []
    if episodes == 1:
        return [(float(yaw_min) + float(yaw_max)) / 2.0]
    return [
        float(yaw_min) + (float(yaw_max) - float(yaw_min)) * index / (episodes - 1)
        for index in range(episodes)
    ]


def _write_rgb_image(pixels: Any, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


if __name__ == "__main__":
    main()
