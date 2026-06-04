from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from physical_ai_agent.agent_core.planner import RuleBasedSO101Planner, write_plan
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID


@dataclass(frozen=True)
class Checkpoint20Report:
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
) -> Checkpoint20Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = RuleBasedSO101Planner().plan(task=task, env_id=env_id)
    plan_path = output_dir / "rule_based_plan.json"
    report_path = output_dir / "checkpoint_report.json"
    write_plan(plan, plan_path)
    checks = {
        "cp20_plan_saved": plan_path.exists() and plan_path.stat().st_size > 0,
        "cp20_subgoals_created": len(plan.subgoals) >= 3,
        "cp20_subgoals_have_thresholds": all(subgoal.threshold > 0 for subgoal in plan.subgoals),
        "cp20_retry_budget_recorded": all(subgoal.retry_budget >= 1 for subgoal in plan.subgoals),
    }
    artifacts = {
        "plan": str(plan_path),
        "checkpoint_report": str(report_path),
    }
    metrics = {
        "env_id": env_id,
        "task": plan.task,
        "subgoal_names": [subgoal.name for subgoal in plan.subgoals],
        "subgoal_count": len(plan.subgoals),
    }
    report = Checkpoint20Report(
        checkpoint="checkpoint_20_rule_based_planner",
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
    parser = argparse.ArgumentParser(description="Checkpoint 20 rule-based SO101 planner.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_20")
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
            f"subgoals:{','.join(report.metrics['subgoal_names'])}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
