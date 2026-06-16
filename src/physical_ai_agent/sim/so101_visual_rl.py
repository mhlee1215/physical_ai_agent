from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


@dataclass(frozen=True)
class SO101VisualRLConfig:
    env_id: str = DEFAULT_SO101_ENV_ID
    camera_name: str = "wrist_cam"
    width: int = 128
    height: int = 128
    include_state: bool = True
    channel_first: bool = True


class SO101VisualObservationWrapper:
    """Gymnasium wrapper that exposes rendered camera frames for visual RL."""

    def __init__(self, env: Any, config: SO101VisualRLConfig) -> None:
        import gymnasium as gym
        import mujoco
        import numpy as np

        self.env = env
        self.config = config
        self._gym = gym
        self._np = np
        self._renderer = mujoco.Renderer(
            env.unwrapped.model,
            height=config.height,
            width=config.width,
        )
        image_shape = (
            (3, config.height, config.width)
            if config.channel_first
            else (config.height, config.width, 3)
        )
        spaces: dict[str, Any] = {
            "image": gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
        }
        if config.include_state:
            spaces["state"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=env.observation_space.shape,
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = env.action_space

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._visual_observation(obs), info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._visual_observation(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        return self._render_pixels()

    def close(self) -> None:
        self._renderer.close()
        self.env.close()

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def _visual_observation(self, state_obs: Any) -> dict[str, Any]:
        image = self._render_pixels()
        if self.config.channel_first:
            image = image.transpose(2, 0, 1)
        observation = {"image": image}
        if self.config.include_state:
            observation["state"] = self._np.asarray(state_obs, dtype=self._np.float32).reshape(-1)
        return observation

    def _render_pixels(self) -> Any:
        from physical_ai_agent.sim.so101_camera_input import _make_camera

        camera = _make_camera(self.env, self.config.camera_name)
        self._renderer.update_scene(self.env.unwrapped.data, camera=camera)
        return self._renderer.render()


def make_so101_visual_rl_env(config: SO101VisualRLConfig) -> SO101VisualObservationWrapper:
    try:
        import gymnasium as gym
        import so101_nexus_mujoco  # noqa: F401 - registers MuJoCo env ids.
    except ModuleNotFoundError as exc:
        raise RuntimeError("SO101 visual RL requires gymnasium and so101-nexus-mujoco") from exc

    env = gym.make(config.env_id, render_mode=None)
    return SO101VisualObservationWrapper(env, config)


def run_visual_rl_smoke(
    *,
    output_dir: Path,
    config: SO101VisualRLConfig,
    steps: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    env = make_so101_visual_rl_env(config)
    records = []
    try:
        obs, info = env.reset(seed=seed)
        for step in range(steps):
            image_hwc = _image_hwc(obs["image"], channel_first=config.channel_first)
            image_path = frames_dir / f"visual_obs_{step:03d}.png"
            Image.fromarray(image_hwc).save(image_path)
            action = sample_action(env.action_space, step / max(1, steps - 1))
            obs, reward, terminated, truncated, info = env.step(action)
            records.append(
                {
                    "step": step,
                    "image_path": str(image_path),
                    "image_shape": list(image_hwc.shape),
                    "state_shape": list(obs["state"].shape) if "state" in obs else [],
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _json_safe_info(info),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    manifest = {
        "operation": "so101_visual_rl_smoke",
        "config": asdict(config),
        "steps": len(records),
        "observation_space": {
            key: {
                "shape": list(space.shape),
                "dtype": str(space.dtype),
            }
            for key, space in env.observation_space.spaces.items()
        },
        "action_space_shape": list(env.action_space.shape),
        "records": records,
    }
    manifest_path = output_dir / "visual_rl_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def _image_hwc(image: Any, *, channel_first: bool) -> Any:
    import numpy as np

    array = np.asarray(image, dtype=np.uint8)
    if channel_first:
        return array.transpose(1, 2, 0)
    return array


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in info.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
