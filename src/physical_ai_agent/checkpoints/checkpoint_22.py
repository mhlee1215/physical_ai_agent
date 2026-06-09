from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from physical_ai_agent.agent_core.so101_agentic_eval import run_agentic_retry
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID


@dataclass(frozen=True)
class Checkpoint22Report:
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
) -> Checkpoint22Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_agentic_retry(output_dir=output_dir / "agentic_retry", env_id=env_id, task=task)
    report_path = output_dir / "checkpoint_report.json"
    trace_text = Path(result.trace_path).read_text(encoding="utf-8")
    checks = {
        "cp22_agentic_trace_saved": Path(result.trace_path).exists()
        and Path(result.trace_path).stat().st_size > 0,
        "cp22_plan_saved": Path(result.plan_path).exists() and Path(result.plan_path).stat().st_size > 0,
        "cp22_retry_once_after_failure": result.retry_events >= 1 and '"retry": true' in trace_text,
        "cp22_verifier_decisions_recorded": '"verifier"' in trace_text,
    }
    artifacts = {
        "plan": result.plan_path,
        "agentic_trace": result.trace_path,
        "checkpoint_report": str(report_path),
    }
    metrics = {
        "env_id": env_id,
        "task": task,
        "steps": result.steps,
        "success": result.success,
        "final_distance": result.final_distance,
        "retry_events": result.retry_events,
        "passed_subgoals": result.passed_subgoals,
        "total_subgoals": result.total_subgoals,
    }
    report = Checkpoint22Report(
        checkpoint="checkpoint_22_agentic_retry_loop",
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
    parser = argparse.ArgumentParser(description="Checkpoint 22 SO101 agentic retry loop.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_22")
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
            f"retries:{report.metrics['retry_events']} "
            f"subgoals:{report.metrics['passed_subgoals']}/{report.metrics['total_subgoals']}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
