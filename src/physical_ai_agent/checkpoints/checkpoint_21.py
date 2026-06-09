from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from physical_ai_agent.agent_core.planner import RuleBasedSO101Planner
from physical_ai_agent.agent_core.verifier import SO101SimulationStateVerifier
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv, sample_action


@dataclass(frozen=True)
class Checkpoint21Report:
    checkpoint: str
    status: str
    python: str
    platform: str
    output_dir: str
    checks: dict[str, bool]
    metrics: dict[str, object]
    artifacts: dict[str, str]


def run_checkpoint(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    task: str = "reach_target",
) -> Checkpoint21Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "verification_decision.json"
    report_path = output_dir / "checkpoint_report.json"
    plan = RuleBasedSO101Planner().plan(task=task, env_id=env_id)
    subgoal = plan.subgoals[0]
    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    try:
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(sample_action(env.action_space, 0.0))
    finally:
        env.close()
    decision = SO101SimulationStateVerifier().verify(subgoal, info)
    decision_path.write_text(
        json.dumps(
            {
                "subgoal": asdict(subgoal),
                "decision": asdict(decision),
                "observation": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": {key: str(value) for key, value in info.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checks = {
        "cp21_sim_step_executed": bool(obs) and isinstance(reward, float),
        "cp21_verifier_decision_saved": decision_path.exists() and decision_path.stat().st_size > 0,
        "cp21_metric_from_sim_state": decision.metric_name == "tcp_to_target_dist",
        "cp21_decision_is_boolean": isinstance(decision.passed, bool),
    }
    artifacts = {
        "verification_decision": str(decision_path),
        "checkpoint_report": str(report_path),
    }
    metrics = {
        "env_id": env_id,
        "task": task,
        "subgoal": subgoal.name,
        "metric_name": decision.metric_name,
        "metric_value": decision.metric_value,
        "threshold": decision.threshold,
        "passed": decision.passed,
    }
    report = Checkpoint21Report(
        checkpoint="checkpoint_21_simulation_state_verifier",
        status="passed" if all(checks.values()) else "failed",
        python=platform.python_version(),
        platform=platform.platform(),
        output_dir=str(output_dir),
        checks=checks,
        metrics=metrics,
        artifacts=artifacts,
    )
    report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 21 SO101 simulation-state verifier.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_21")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--task", default="reach_target")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(output_dir=Path(args.output_dir), env_id=args.env_id, task=args.task)
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
            f"metric:{report.metrics['metric_name']}={report.metrics['metric_value']}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
