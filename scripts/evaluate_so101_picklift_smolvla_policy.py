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
from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame
from train_so101_wrist_ego_picklift_policy import sweep_until_visible
from train_so101_wrist_ego_visual_servo import (
    WristEgoServoConfig,
    _current_qpos,
    _make_policy_renderers,
    make_high_contrast_picklift_env,
)


TASK = "Grasp the visible cube and lift it up."


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
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-rollout-gif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gif-fps", type=int, default=12)
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
        sweep=args.sweep,
        record_rollout_gif=args.record_rollout_gif,
        gif_fps=args.gif_fps,
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
    sweep: bool = True,
    record_rollout_gif: bool = False,
    gif_fps: int = 12,
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
    config = WristEgoServoConfig(width=width, height=height)
    env = make_high_contrast_picklift_env()
    renderers = _make_policy_renderers(env, config)
    rows = []
    try:
        for episode in range(episodes):
            if torch_seed is not None:
                _set_torch_seed(int(torch_seed) + episode)
            env.reset(seed=seed + episode)
            if sweep:
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
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
            )
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()
    report = {
        "operation": "evaluate_so101_picklift_smolvla_policy",
        "policy_path": policy_path,
        "runtime_inputs": ["top_down", "wrist_cam", "joint_positions", "task"],
        "runtime_excludes": ["egocentric_cam", "camera_calibration", "object_pose", "mujoco_jacobian"],
        "action_filter": {
            "action_alpha": float(action_alpha),
            "max_arm_delta": float(max_arm_delta),
            "max_gripper_delta": float(max_gripper_delta),
        },
        "pre_rollout_sweep": bool(sweep),
        "use_policy_processors": bool(preprocessor is not None and postprocessor is not None),
        "torch_seed": torch_seed,
        "policy_rollout_config": _policy_rollout_config(policy),
        "feature_mapping": {
            "observation.images.camera1": "top_down",
            "observation.images.camera2": "wrist_cam",
            "observation.images.camera3": "wrist_cam duplicate when requested by policy",
            "observation.state": "SO101 qpos/control state",
            "action": (
                "SO101 qpos target action via saved policy_preprocessor/policy_postprocessor"
                if preprocessor is not None and postprocessor is not None
                else "SO101 raw policy output interpreted as qpos target"
            ),
            "task": TASK,
        },
        "episodes": rows,
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "grasp_rate": float(np.mean([row.get("final_is_grasped", 0.0) > 0.5 for row in rows])) if rows else 0.0,
        "duration_s": round(perf_counter() - started, 4),
        "device": _policy_device_metadata(policy),
    }
    report_path = output_dir / "so101_picklift_smolvla_eval_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


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
    preprocessor: Any | None,
    postprocessor: Any | None,
) -> dict[str, Any]:
    records = []
    frames = []
    info = env.unwrapped._get_info()
    image_feature_mapping = {}
    if hasattr(policy, "reset"):
        policy.reset()
    for step in range(max_steps):
        if record_rollout_gif:
            frames.append(_render_rollout_frame(env, renderers))
        camera_pixels = _render_policy_cameras(env, renderers)
        if preprocessor is not None and postprocessor is not None:
            raw_action = _predict_action_with_processors(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                qpos=_current_qpos(env).astype(float),
                camera_pixels=camera_pixels,
            )
            image_feature_mapping = {
                "observation.images.camera1": "top_down",
                "observation.images.camera2": "wrist_cam",
                "observation.images.camera3": "wrist_cam",
            }
        else:
            batch, image_feature_mapping = _build_batch_for_policy(
                policy,
                _current_qpos(env).astype(float).tolist(),
                camera_pixels,
                instruction=TASK,
                local_files_only=True,
            )
            raw_action = policy.select_action(batch)
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
                "lift_height": float(info.get("lift_height", 0.0)),
                "tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
            }
        )
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
    return {
        "episode": episode,
        "seed": seed,
        "search_steps": search_steps,
        "steps": len(records),
        "success": bool(info.get("success", False)),
        "final_is_grasped": float(info.get("is_grasped", 0.0)),
        "final_lift_height": float(info.get("lift_height", 0.0)),
        "final_tcp_to_obj_dist": float(info.get("tcp_to_obj_dist", 0.0)),
        "image_feature_mapping": image_feature_mapping,
        "rollout_gif": gif_path,
        "rollout_mp4": mp4_path,
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
):
    from lerobot.utils.control_utils import predict_action

    selected_device = str(_policy_device_metadata(policy).get("device_selected") or getattr(policy.config, "device", "cpu"))
    observation = {
        "observation.state": np.asarray(qpos, dtype=np.float32),
        "observation.images.camera1": np.asarray(camera_pixels["top_down"], dtype=np.uint8),
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
        task=TASK,
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
    for camera_name in ("top_down", "wrist_cam"):
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
