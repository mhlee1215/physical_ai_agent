from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from physical_ai_agent.data.so101_demo_dataset import write_demo_dataset
from physical_ai_agent.envhub.so101_env import SO101EnvConfig, make_env
from physical_ai_agent.policies.smolvla_dry_run import SmolVLADryRunBridge
from physical_ai_agent.policies.so101_action_chunk import SO101ActionChunkConfig, SO101CenterActionChunkPolicy
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv, rollout_so101


@dataclass(frozen=True)
class Checkpoint0713Report:
    checkpoint: str
    status: str
    python: str
    platform: str
    output_dir: str
    checks: dict[str, bool]
    metrics: dict[str, object]
    artifacts: dict[str, str]


def run_checkpoint(output_dir: Path, env_id: str = DEFAULT_SO101_ENV_ID, steps: int = 48) -> Checkpoint0713Report:
    started_at = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    obs, _info = env.reset(seed=0)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _step_info = env.step(action)
    action_low = [float(value) for value in env.action_space.low]
    action_high = [float(value) for value in env.action_space.high]
    action_dim = env.action_dim
    env.close()

    rollout = rollout_so101(output_dir=output_dir / "rollout", env_id=env_id, steps=steps, seed=0)

    envhub = make_env(n_envs=1, use_async_envs=False, cfg=SO101EnvConfig(env_id=env_id))
    vector_env = envhub["so101_nexus"][env_id]
    try:
        vector_obs, vector_info = vector_env.reset(seed=0)
        vector_action = vector_env.action_space.sample()
        vector_step = vector_env.step(vector_action)
    finally:
        vector_env.close()

    policy = SO101CenterActionChunkPolicy(
        SO101ActionChunkConfig(action_dim=action_dim, chunk_size=6, low=action_low, high=action_high)
    )
    chunk = policy.action_chunk(obs, "hold the SO-101 arm near center")

    dry_bridge = SmolVLADryRunBridge()
    dry_input = dry_bridge.build_input(next_obs, "reach the target with the SO-101 arm", action_dim)
    dry_chunk = dry_bridge.dry_action_chunk(dry_input, chunk_size=6)

    smolvla_rollout_dir = output_dir / "smolvla_dry_rollout"
    smolvla_rollout = _run_dry_chunk_rollout(
        output_dir=smolvla_rollout_dir,
        env_id=env_id,
        actions=dry_chunk.actions,
    )

    dataset_paths = write_demo_dataset(
        output_dir=output_dir / "demo_dataset",
        steps=rollout.steps,
        task=f"{env_id}: demonstration from deterministic SO101-Nexus rollout",
    )

    checks = {
        "cp07_so101_env_reset_step": len(obs) > 0 and len(next_obs) > 0 and isinstance(reward, float),
        "cp08_rollout_trace_saved": Path(rollout.trace_path).exists() and Path(rollout.trace_path).stat().st_size > 0,
        "cp08_rollout_visualization_saved": Path(rollout.frame_path).exists()
        and Path(rollout.gif_path).exists(),
        "cp09_lerobot_make_env_surface": "so101_nexus" in envhub
        and env_id in envhub["so101_nexus"]
        and vector_obs is not None
        and len(vector_step) == 5,
        "cp10_so101_action_chunk_policy": chunk.chunk_size == 6
        and all(len(item) == action_dim for item in chunk.actions),
        "cp11_smolvla_dry_input_mapping": dry_bridge.adapter.ready
        and len(dry_input.state) > 0
        and dry_input.action_dim == action_dim,
        "cp12_smolvla_dry_rollout_visualized": Path(smolvla_rollout["trace"]).exists()
        and Path(smolvla_rollout["frame"]).exists(),
        "cp13_demo_dataset_written": Path(dataset_paths["episodes"]).exists()
        and Path(dataset_paths["metadata"]).exists(),
    }
    artifacts = {
        "rollout_trace": rollout.trace_path,
        "rollout_frame": rollout.frame_path,
        "rollout_gif": rollout.gif_path,
        "rollout_metrics": rollout.metrics_path,
        "smolvla_dry_trace": smolvla_rollout["trace"],
        "smolvla_dry_frame": smolvla_rollout["frame"],
        "smolvla_dry_gif": smolvla_rollout["gif"],
        "demo_dataset_episodes": dataset_paths["episodes"],
        "demo_dataset_metadata": dataset_paths["metadata"],
        "checkpoint_report": str(output_dir / "checkpoint_report.json"),
    }
    metrics = {
        "env_id": env_id,
        "duration_s": round(perf_counter() - started_at, 4),
        "observation_dim": len(obs),
        "action_dim": action_dim,
        "rollout_steps": len(rollout.steps),
        "rollout_success": rollout.success,
        "smolvla_dry_chunk_size": dry_chunk.chunk_size,
        "smolvla_note": dry_chunk.metadata["note"],
    }
    report = Checkpoint0713Report(
        checkpoint="checkpoint_07_13_so101_nexus_smolvla_dry_dataset",
        status="passed" if all(checks.values()) else "failed",
        python=platform.python_version(),
        platform=platform.platform(),
        output_dir=str(output_dir),
        checks=checks,
        metrics=metrics,
        artifacts=artifacts,
    )
    (output_dir / "checkpoint_report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _run_dry_chunk_rollout(output_dir: Path, env_id: str, actions: list[list[float]]) -> dict[str, str]:
    import json
    from physical_ai_agent.sim.so101_nexus_env import SO101Step, _write_so101_visualization

    output_dir.mkdir(parents=True, exist_ok=True)
    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    obs, _info = env.reset(seed=1)
    records = []
    try:
        for index, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            records.append(
                SO101Step(
                    step=index,
                    observation=obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info={key: str(value) for key, value in info.items()},
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    trace_path = output_dir / "smolvla_dry_rollout.jsonl"
    frame_path = output_dir / "smolvla_dry_rollout.png"
    gif_path = output_dir / "smolvla_dry_rollout.gif"
    with trace_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    _write_so101_visualization(records, frame_path, gif_path)
    return {"trace": str(trace_path), "frame": str(frame_path), "gif": str(gif_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 07-13 SO101-Nexus simulation pipeline.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_07_13")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(output_dir=Path(args.output_dir), env_id=args.env_id, steps=args.steps)
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(f"{report.checkpoint}: {report.status}")
        print(f"output_dir={report.output_dir}")
        for name, passed in report.checks.items():
            print(f"- {'PASS' if passed else 'FAIL'} {name}")
        print(
            "metrics="
            f"env:{report.metrics['env_id']} "
            f"steps:{report.metrics['rollout_steps']} "
            f"obs_dim:{report.metrics['observation_dim']} "
            f"action_dim:{report.metrics['action_dim']}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()

