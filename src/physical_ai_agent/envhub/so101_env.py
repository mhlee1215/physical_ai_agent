from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SO101EnvConfig:
    env_id: str = "MuJoCoReach-v1"


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg: SO101EnvConfig | None = None):
    import gymnasium as gym
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    env_id = (cfg or SO101EnvConfig()).env_id

    def make_single():
        return gym.make(env_id, render_mode=None)

    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return {"so101_nexus": {env_id: vector_cls([make_single for _ in range(n_envs)])}}

