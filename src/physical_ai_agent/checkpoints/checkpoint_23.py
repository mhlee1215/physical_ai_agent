from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from physical_ai_agent.agent_core.so101_agentic_eval import run_policy_vs_agentic_comparison
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID


@dataclass(frozen=True)
class Checkpoint23Report:
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
    policy_steps: int = 8,
) -> Checkpoint23Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = run_policy_vs_agentic_comparison(
        output_dir=output_dir / "comparison",
        env_id=env_id,
        task=task,
        policy_steps=policy_steps,
    )
    report_path = output_dir / "checkpoint_report.json"
    markdown_text = Path(comparison.markdown_path).read_text(encoding="utf-8")
    checks = {
        "cp23_policy_only_trace_saved": Path(comparison.policy_only.trace_path).exists()
        and Path(comparison.policy_only.trace_path).stat().st_size > 0,
        "cp23_agentic_retry_trace_saved": Path(comparison.agentic_retry.trace_path).exists()
        and Path(comparison.agentic_retry.trace_path).stat().st_size > 0,
        "cp23_metrics_saved": Path(comparison.metrics_path).exists()
        and Path(comparison.metrics_path).stat().st_size > 0,
        "cp23_markdown_report_saved": Path(comparison.markdown_path).exists()
        and "policy_only" in markdown_text
        and "agentic_retry" in markdown_text,
        "cp23_results_compared": comparison.policy_only.mode == "policy_only"
        and comparison.agentic_retry.mode == "agentic_retry",
    }
    artifacts = {
        "comparison_metrics": comparison.metrics_path,
        "comparison_markdown": comparison.markdown_path,
        "policy_trace": comparison.policy_only.trace_path,
        "agentic_plan": comparison.agentic_retry.plan_path,
        "agentic_trace": comparison.agentic_retry.trace_path,
        "checkpoint_report": str(report_path),
    }
    metrics = {
        "env_id": env_id,
        "task": task,
        "policy_only_success": comparison.policy_only.success,
        "agentic_retry_success": comparison.agentic_retry.success,
        "policy_only_final_distance": comparison.policy_only.final_distance,
        "agentic_retry_final_distance": comparison.agentic_retry.final_distance,
        "agentic_retry_events": comparison.agentic_retry.retry_events,
        "agentic_passed_subgoals": comparison.agentic_retry.passed_subgoals,
        "agentic_total_subgoals": comparison.agentic_retry.total_subgoals,
    }
    report = Checkpoint23Report(
        checkpoint="checkpoint_23_first_comparison_report",
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
    parser = argparse.ArgumentParser(description="Checkpoint 23 policy-only vs agentic-retry report.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_23")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--task", default="reach_target")
    parser.add_argument("--policy-steps", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(
        output_dir=Path(args.output_dir),
        env_id=args.env_id,
        task=args.task,
        policy_steps=args.policy_steps,
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
            f"env:{report.metrics['env_id']} "
            f"policy_success:{report.metrics['policy_only_success']} "
            f"agentic_success:{report.metrics['agentic_retry_success']} "
            f"retries:{report.metrics['agentic_retry_events']}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
