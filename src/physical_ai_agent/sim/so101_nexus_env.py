from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SO101_ENV_ID = "MuJoCoReach-v1"


@dataclass(frozen=True)
class SO101Step:
    step: int
    observation: list[float]
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class SO101Rollout:
    env_id: str
    steps: list[SO101Step]
    success: bool
    total_reward: float
    trace_path: str
    frame_path: str
    gif_path: str
    metrics_path: str


class SO101NexusEnv:
    def __init__(self, env_id: str = DEFAULT_SO101_ENV_ID, render_mode: str | None = None) -> None:
        try:
            importlib.import_module("so101_nexus_mujoco")
            gym = importlib.import_module("gymnasium")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("SO101-Nexus MuJoCo and Gymnasium are required") from exc

        self.env_id = env_id
        self._gym = gym
        self.env = gym.make(env_id, render_mode=render_mode)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_dim(self) -> int:
        return int(self.action_space.shape[0])

    def reset(self, seed: int = 0) -> tuple[list[float], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return _as_float_list(obs), info

    def step(self, action: list[float]) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return _as_float_list(obs), float(reward), bool(terminated), bool(truncated), dict(info)

    def close(self) -> None:
        self.env.close()


def sample_action(action_space, fraction: float) -> list[float]:
    """Deterministic smooth action inside the action bounds."""
    import math

    low = action_space.low
    high = action_space.high
    center = (low + high) / 2.0
    radius = (high - low) * 0.18
    values = []
    for index, (base, amp) in enumerate(zip(center, radius, strict=True)):
        values.append(float(base + amp * math.sin(fraction * 2.0 * math.pi + index * 0.7)))
    return values


def rollout_so101(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    steps: int = 48,
    seed: int = 0,
) -> SO101Rollout:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "so101_rollout.jsonl"
    frame_path = output_dir / "so101_rollout.png"
    gif_path = output_dir / "so101_rollout.gif"
    metrics_path = output_dir / "so101_metrics.json"

    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    obs, _info = env.reset(seed=seed)
    records: list[SO101Step] = []
    total_reward = 0.0
    try:
        for step in range(steps):
            action = sample_action(env.action_space, step / max(1, steps - 1))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            records.append(
                SO101Step(
                    step=step,
                    observation=obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=_json_safe_info(info),
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    with trace_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    _write_so101_visualization(records, frame_path, gif_path)
    success = bool(records) and all(_is_finite(record.observation) for record in records)
    metrics = {
        "env_id": env_id,
        "steps": len(records),
        "success": success,
        "total_reward": total_reward,
        "avg_reward": total_reward / len(records) if records else 0.0,
        "observation_dim": len(records[-1].observation) if records else 0,
        "action_dim": len(records[-1].action) if records else 0,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return SO101Rollout(
        env_id=env_id,
        steps=records,
        success=success,
        total_reward=total_reward,
        trace_path=str(trace_path),
        frame_path=str(frame_path),
        gif_path=str(gif_path),
        metrics_path=str(metrics_path),
    )


def _write_so101_visualization(records: list[SO101Step], frame_path: Path, gif_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 520, 320
    frames = []

    def draw(record: SO101Step) -> Image.Image:
        im = Image.new("RGB", (width, height), (247, 247, 242))
        d = ImageDraw.Draw(im)
        d.text((14, 12), f"SO101-Nexus {record.step:03d} | reward {record.reward:.3f}", fill=(20, 20, 20))
        d.text((14, 34), "Observation bars", fill=(65, 65, 65))
        obs = record.observation[: min(12, len(record.observation))]
        for index, value in enumerate(obs):
            x0 = 20 + index * 40
            y_mid = 158
            d.line((x0, y_mid - 70, x0, y_mid + 70), fill=(210, 210, 205))
            bar = max(-1.0, min(1.0, float(value))) * 65
            color = (40, 120, 220) if bar >= 0 else (230, 120, 60)
            y0, y1 = sorted((y_mid, y_mid - bar))
            d.rectangle((x0 - 9, y0, x0 + 9, y1), fill=color)
            d.text((x0 - 12, 232), str(index), fill=(90, 90, 90))
        d.text((14, 262), "Action", fill=(65, 65, 65))
        for index, value in enumerate(record.action):
            x0 = 80 + index * 60
            y0 = 286
            d.rectangle((x0, y0 - 8, x0 + 44, y0 + 8), outline=(160, 160, 160))
            fill = int(22 * max(-1.0, min(1.0, abs(float(value)))))
            d.rectangle((x0 + 22 - fill, y0 - 7, x0 + 22 + fill, y0 + 7), fill=(60, 155, 100))
            d.text((x0, y0 + 14), f"a{index}", fill=(90, 90, 90))
        return im

    for record in records[:: max(1, len(records) // 18)]:
        frames.append(draw(record))
    if records:
        frames.append(draw(records[-1]))
    if not frames:
        frames.append(Image.new("RGB", (width, height), (247, 247, 242)))
    frames[-1].save(frame_path)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0)


def _as_float_list(obs: object) -> list[float]:
    import numpy as np

    if isinstance(obs, dict):
        values: list[float] = []
        for key in sorted(obs):
            values.extend(np.asarray(obs[key], dtype=float).reshape(-1).tolist())
        return values
    return np.asarray(obs, dtype=float).reshape(-1).tolist()


def _is_finite(values: list[float]) -> bool:
    import math

    return all(math.isfinite(value) for value in values)


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in info.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
