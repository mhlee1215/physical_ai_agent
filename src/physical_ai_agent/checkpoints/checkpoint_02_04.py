from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from physical_ai_agent.evaluation.runner import run_eval
from physical_ai_agent.policies.random_policy import RandomPolicy, RandomPolicyConfig
from physical_ai_agent.sim.tiny_mujoco_env import TinyMujocoConfig, TinyMujocoEnv


@dataclass(frozen=True)
class Checkpoint0204Report:
    checkpoint: str
    status: str
    python: str
    platform: str
    output_dir: str
    duration_s: float
    checks: dict[str, bool]
    metrics: dict[str, object]
    artifacts: dict[str, str]


def run_checkpoint(output_dir: Path, episodes: int, episode_steps: int, seed: int) -> Checkpoint0204Report:
    started_at = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    def env_factory() -> TinyMujocoEnv:
        return TinyMujocoEnv(TinyMujocoConfig(episode_steps=episode_steps, seed=seed))

    def policy_factory(env: TinyMujocoEnv) -> RandomPolicy:
        return RandomPolicy(RandomPolicyConfig(action_dim=env.action_dim, seed=seed, scale=1.0))

    metrics = run_eval(
        env_factory=env_factory,
        policy_factory=policy_factory,
        output_dir=output_dir,
        episodes=episodes,
    )
    first_episode = metrics.episode_metrics[0]
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    trace_path = Path(first_episode.trace_path)
    frame_path = Path(first_episode.frame_path)

    checks = {
        "cp02_random_policy_episode_completed": first_episode.success
        and first_episode.steps == episode_steps,
        "cp03_trace_saved": trace_path.exists() and trace_path.stat().st_size > 0,
        "cp03_frame_saved": frame_path.exists() and frame_path.stat().st_size > 0,
        "cp03_metrics_saved": metrics_path.exists() and metrics_path.stat().st_size > 0,
        "cp04_success_rate_computed": 0.0 <= metrics.success_rate <= 1.0,
        "cp04_summary_saved": summary_path.exists() and summary_path.stat().st_size > 0,
    }
    status = "passed" if all(checks.values()) else "failed"
    report = Checkpoint0204Report(
        checkpoint="checkpoint_02_03_04_random_policy_eval",
        status=status,
        python=platform.python_version(),
        platform=platform.platform(),
        output_dir=str(output_dir),
        duration_s=round(perf_counter() - started_at, 4),
        checks=checks,
        metrics={
            "episodes": metrics.episodes,
            "success_rate": metrics.success_rate,
            "avg_episode_length": metrics.avg_episode_length,
            "avg_total_reward": metrics.avg_total_reward,
            "avg_step_latency_s": metrics.avg_step_latency_s,
        },
        artifacts={
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "first_trace": str(trace_path),
            "first_frame": str(frame_path),
        },
    )
    (output_dir / "checkpoint_report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 02-04 random policy evaluator.")
    parser.add_argument(
        "--output-dir",
        default="_workspace/checkpoints/checkpoint_02_04",
        help="Directory for checkpoint artifacts.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(
        output_dir=Path(args.output_dir),
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(f"{report.checkpoint}: {report.status}")
        print(f"output_dir={report.output_dir}")
        for name, passed in report.checks.items():
            print(f"- {'PASS' if passed else 'FAIL'} {name}")
        print(
            "metrics="
            f"episodes:{report.metrics['episodes']} "
            f"success_rate:{report.metrics['success_rate']:.3f} "
            f"avg_episode_length:{report.metrics['avg_episode_length']:.1f}"
        )

    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()

