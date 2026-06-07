#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.policies.lerobot_policy_runner import LeRobotPolicyRunner, load_lerobot_policy_runner


DEFAULT_TASK_INSTRUCTIONS = {
    "PushCube-v1": "Push the cube to the goal.",
    "StackCube-v1": "Stack the cube on top of the other cube.",
    "PullCube-v1": "Pull the cube to the goal.",
    "LiftPegUpright-v1": "Lift the peg upright.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a SmolVLA checkpoint on ManiSkill3 through LeRobot processors.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="PushCube-v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--obs-mode", default="rgb+state")
    parser.add_argument("--instruction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-success-percent", type=float, default=None)
    args = parser.parse_args()

    import gymnasium as gym
    import mani_skill  # noqa: F401

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = load_lerobot_policy_runner(
        args.checkpoint,
        device=args.device,
        local_files_only=True,
    )
    instruction = args.instruction or DEFAULT_TASK_INSTRUCTIONS.get(args.env_id, "Perform the ManiSkill task.")
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        control_mode=args.control_mode,
        num_envs=1,
        max_episode_steps=args.max_steps,
    )

    records: list[dict[str, Any]] = []
    successes: list[bool] = []
    started_at = time.time()
    try:
        for episode in range(args.episodes):
            obs, _info = env.reset(seed=args.seed + episode)
            runner.policy.reset()
            episode_success = False
            step_count = 0
            for step in range(args.max_steps):
                observation = build_lerobot_observation(runner, obs, instruction=instruction)
                action_tensor = runner.select_action(observation)
                action = clip_to_action_space(env, tensor_to_float_list(action_tensor))
                obs, reward, terminated, truncated, info = env.step(action)
                step_success = info_success(info)
                episode_success = episode_success or step_success
                step_count = step + 1
                records.append(
                    {
                        "episode": episode,
                        "seed": args.seed + episode,
                        "step": step,
                        "reward": float(np.asarray(reward).reshape(-1)[0]),
                        "success": step_success,
                        "action_norm": float(np.linalg.norm(np.asarray(action))),
                    }
                )
                if bool1(terminated) or bool1(truncated):
                    break
            successes.append(episode_success)
            print(f"episode={episode} success={episode_success} steps={step_count}", flush=True)
    finally:
        env.close()

    success_rate = float(sum(successes) / len(successes)) if successes else 0.0
    reference = args.reference_success_percent
    metrics = {
        "env_id": args.env_id,
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "success_count": int(sum(successes)),
        "success_rate": success_rate,
        "success_percent": success_rate * 100.0,
        "reference_success_percent": reference,
        "delta_vs_reference_pp": (success_rate * 100.0 - reference) if reference is not None else None,
        "runner": "LeRobotPolicyRunner",
        "preprocessor_applied": True,
        "postprocessor_applied": True,
        "postprocessor_steps": [type(step).__name__ for step in runner.postprocessor.steps],
        "duration_s": time.time() - started_at,
    }
    (output_dir / "trace.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def build_lerobot_observation(runner: LeRobotPolicyRunner, obs: Any, *, instruction: str) -> dict[str, Any]:
    config = runner.policy.config
    state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else 0
    state = np.zeros((1, state_dim), dtype=np.float32)
    numeric = extract_robot_qpos(obs, limit=max(1, state_dim))
    if not numeric:
        numeric = flatten_numeric_observation(obs, limit=max(1, state_dim))
    if numeric and state_dim:
        source = np.asarray(numeric[:state_dim], dtype=np.float32)
        state[0, : len(source)] = source

    observation: dict[str, Any] = {
        "observation.state": state,
        "task": [instruction],
    }
    images = extract_rgb_images(obs)
    zero_image = np.zeros((1, 3, 128, 128), dtype=np.float32)
    base_camera = images.get("base_camera")
    observation["observation.images.base_camera"] = (
        np.expand_dims(image_to_lerobot_chw(base_camera), axis=0) if base_camera is not None else zero_image
    )
    for image_key in getattr(config, "image_features", {}):
        if str(image_key).endswith("camera1"):
            continue
        observation[str(image_key)] = zero_image
    return observation


def flatten_numeric_observation(obs: Any, *, limit: int) -> list[float]:
    values: list[float] = []

    def visit(value: Any) -> None:
        if len(values) >= limit:
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                if key in {"sensor_data", "sensor_param"}:
                    continue
                visit(value[key])
                if len(values) >= limit:
                    return
            return
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            for item in array[: max(0, limit - len(values))]:
                values.append(float(item))
        except Exception:
            return

    visit(obs.get("state", obs) if isinstance(obs, dict) else obs)
    return values


def extract_robot_qpos(obs: Any, *, limit: int) -> list[float]:
    if not isinstance(obs, dict):
        return []
    agent = obs.get("agent")
    if not isinstance(agent, dict) or "qpos" not in agent:
        return []
    value = agent["qpos"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape(-1, array.shape[-1])[0]
    return [float(item) for item in array.reshape(-1)[:limit]]


def extract_rgb_images(obs: Any) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    if not isinstance(obs, dict):
        return images
    sensor_data = obs.get("sensor_data", {})
    if not isinstance(sensor_data, dict):
        return images
    for camera_name, camera_data in sensor_data.items():
        if not isinstance(camera_data, dict) or "rgb" not in camera_data:
            continue
        value = camera_data["rgb"]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == 5:
            array = array[0, 0]
        elif array.ndim == 4:
            array = array[0]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        images[str(camera_name)] = array[..., :3]
    return images


def image_to_lerobot_chw(image: np.ndarray) -> np.ndarray:
    if image.shape[0] != 128 or image.shape[1] != 128:
        image = resize_nearest(image, height=128, width=128)
    return np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))


def resize_nearest(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return image[y_idx][:, x_idx]


def tensor_to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [float(item) for item in np.asarray(value, dtype=np.float32).reshape(-1).tolist()]


def clip_to_action_space(env: Any, action: list[float]) -> Any:
    action_space = env.action_space
    size = int(np.prod(action_space.shape))
    values = action[:size] if len(action) >= size else action + [0.0] * (size - len(action))
    array = np.asarray(values, dtype=getattr(action_space, "dtype", np.float32)).reshape(action_space.shape)
    return np.clip(array, action_space.low, action_space.high)


def info_success(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    return bool1(info.get("success", False))


def bool1(value: Any) -> bool:
    if hasattr(value, "detach"):
        return bool(value.detach().cpu().numpy().reshape(-1)[0])
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
