from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, sample_action


@dataclass(frozen=True)
class LiveViewerConfig:
    env_id: str = DEFAULT_SO101_ENV_ID
    fps: float = 30.0
    seed: int = 0
    max_steps: int | None = None


def run_live_viewer(config: LiveViewerConfig) -> int:
    import gymnasium as gym
    import mujoco.viewer
    import so101_nexus_mujoco  # noqa: F401 - registers Gymnasium env ids.

    env = gym.make(config.env_id, render_mode=None)
    env.reset(seed=config.seed)
    sleep_s = 1.0 / max(1.0, config.fps)
    step = 0

    try:
        with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
            while viewer.is_running():
                if config.max_steps is not None and step >= config.max_steps:
                    break
                action = sample_action(env.action_space, (step % 120) / 119.0)
                env.step(action)
                viewer.sync()
                step += 1
                time.sleep(sleep_s)
    finally:
        env.close()
    return step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a live MuJoCo viewer for SO101-Nexus.")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional finite step count for smoke checks or scripted demos.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    steps = run_live_viewer(
        LiveViewerConfig(
            env_id=args.env_id,
            fps=args.fps,
            seed=args.seed,
            max_steps=args.max_steps,
        )
    )
    print(f"SO101 live viewer closed after {steps} steps")


if __name__ == "__main__":
    main()
