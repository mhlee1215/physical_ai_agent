from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable, Protocol

from physical_ai_agent.sim.tiny_mujoco_env import TinyMujocoEnv


class Policy(Protocol):
    def act(self, observation: object) -> list[float]:
        ...


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_index: int
    steps: int
    success: bool
    total_reward: float
    wall_time_s: float
    avg_step_latency_s: float
    final_distance_from_origin: float
    trace_path: str
    frame_path: str


@dataclass(frozen=True)
class EvalMetrics:
    episodes: int
    success_rate: float
    avg_episode_length: float
    avg_total_reward: float
    avg_step_latency_s: float
    episode_metrics: list[EpisodeMetrics]


def run_episode(
    env: TinyMujocoEnv,
    policy: Policy,
    output_dir: Path,
    episode_index: int = 0,
) -> EpisodeMetrics:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"episode_{episode_index:03d}.jsonl"
    frame_path = output_dir / f"episode_{episode_index:03d}_final.ppm"

    observation = env.reset()
    done = False
    total_reward = 0.0
    step_latencies: list[float] = []
    final_info = {"finite_state": False, "distance_from_origin": 0.0}
    started_at = perf_counter()

    with trace_path.open("w", encoding="utf-8") as trace_file:
        while not done:
            step_started_at = perf_counter()
            action = policy.act(observation)
            next_observation, reward, done, info = env.step(action)
            latency = perf_counter() - step_started_at
            step_latencies.append(latency)
            total_reward += reward
            final_info = info

            trace_file.write(
                json.dumps(
                    {
                        "step": observation.step,
                        "observation": asdict(observation),
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "latency_s": latency,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            observation = next_observation

    env.write_frame_ppm(str(frame_path))
    steps = len(step_latencies)
    success = bool(final_info.get("finite_state")) and steps == env.config.episode_steps
    return EpisodeMetrics(
        episode_index=episode_index,
        steps=steps,
        success=success,
        total_reward=total_reward,
        wall_time_s=perf_counter() - started_at,
        avg_step_latency_s=mean(step_latencies) if step_latencies else 0.0,
        final_distance_from_origin=float(final_info.get("distance_from_origin", 0.0)),
        trace_path=str(trace_path),
        frame_path=str(frame_path),
    )


def run_eval(
    env_factory: Callable[[], TinyMujocoEnv],
    policy_factory: Callable[[TinyMujocoEnv], Policy],
    output_dir: Path,
    episodes: int,
) -> EvalMetrics:
    episode_metrics = []
    for episode_index in range(episodes):
        env = env_factory()
        policy = policy_factory(env)
        episode_metrics.append(
            run_episode(
                env=env,
                policy=policy,
                output_dir=output_dir,
                episode_index=episode_index,
            )
        )

    metrics = EvalMetrics(
        episodes=episodes,
        success_rate=mean([1.0 if item.success else 0.0 for item in episode_metrics]),
        avg_episode_length=mean([item.steps for item in episode_metrics]),
        avg_total_reward=mean([item.total_reward for item in episode_metrics]),
        avg_step_latency_s=mean([item.avg_step_latency_s for item in episode_metrics]),
        episode_metrics=episode_metrics,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_summary_markdown(metrics), encoding="utf-8")
    return metrics


def _summary_markdown(metrics: EvalMetrics) -> str:
    return "\n".join(
        [
            "# Checkpoint 02-04 Evaluation Summary",
            "",
            f"- Episodes: {metrics.episodes}",
            f"- Success rate: {metrics.success_rate:.3f}",
            f"- Average episode length: {metrics.avg_episode_length:.1f}",
            f"- Average total reward: {metrics.avg_total_reward:.6f}",
            f"- Average step latency: {metrics.avg_step_latency_s:.6f}s",
            "",
        ]
    )
