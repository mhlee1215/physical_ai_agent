from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EpisodeVerification:
    task_group: str
    task_id: int
    episode_index: int
    passed: bool
    reason: str


@dataclass(frozen=True)
class RetryPlan:
    task_group: str
    failed_task_ids: list[int]
    failed_episodes: int
    total_episodes: int
    retry_budget: int
    verifier: str


@dataclass(frozen=True)
class AgenticRetryMetrics:
    task_group: str
    baseline_success_rate: float
    retry_success_rate: float
    success_once_rate: float
    recovery_success_rate: float
    total_episodes: int
    failed_episodes: int
    recovered_episodes: int
    retry_budget: int


def load_eval_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_eval_info(data: dict[str, Any], task_group: str | None = None) -> list[EpisodeVerification]:
    decisions: list[EpisodeVerification] = []
    for item in data.get("per_task", []):
        group = str(item.get("task_group", ""))
        if task_group and group != task_group:
            continue
        task_id = int(item.get("task_id"))
        successes = item.get("metrics", {}).get("successes", [])
        for episode_index, success in enumerate(successes):
            passed = bool(success)
            reason = "benchmark success flag true" if passed else "benchmark success flag false"
            decisions.append(
                EpisodeVerification(
                    task_group=group,
                    task_id=task_id,
                    episode_index=episode_index,
                    passed=passed,
                    reason=reason,
                )
            )
    return decisions


def build_retry_plan(data: dict[str, Any], task_group: str, retry_budget: int = 1) -> RetryPlan:
    decisions = verify_eval_info(data, task_group=task_group)
    failed_task_ids = sorted({decision.task_id for decision in decisions if not decision.passed})
    return RetryPlan(
        task_group=task_group,
        failed_task_ids=failed_task_ids,
        failed_episodes=sum(1 for decision in decisions if not decision.passed),
        total_episodes=len(decisions),
        retry_budget=retry_budget,
        verifier="libero_benchmark_success_flag",
    )


def aggregate_retry_metrics(
    baseline_data: dict[str, Any],
    retry_data: dict[str, Any],
    task_group: str,
    retry_budget: int = 1,
) -> tuple[AgenticRetryMetrics, list[dict[str, Any]]]:
    baseline = _episode_success_map(baseline_data, task_group)
    retry = _episode_success_map(retry_data, task_group)
    trace: list[dict[str, Any]] = []
    recovered = 0
    failed = 0
    success_once = 0
    retry_successes = sum(1 for passed in retry.values() if passed)

    for key in sorted(baseline):
        baseline_passed = baseline[key]
        retry_passed = retry.get(key, False)
        attempted_retry = not baseline_passed and key in retry
        if not baseline_passed:
            failed += 1
        if attempted_retry and retry_passed:
            recovered += 1
        if baseline_passed or (attempted_retry and retry_passed):
            success_once += 1
        task_id, episode_index = key
        trace.append(
            {
                "task_group": task_group,
                "task_id": task_id,
                "episode_index": episode_index,
                "baseline_success": baseline_passed,
                "retry_attempted": attempted_retry,
                "retry_success": retry_passed if attempted_retry else None,
                "success_once": baseline_passed or (attempted_retry and retry_passed),
                "verifier": {
                    "name": "libero_benchmark_success_flag",
                    "passed": baseline_passed,
                },
                "retry_policy": "retry_failed_task_episode_index_once",
            }
        )

    total = len(baseline)
    metrics = AgenticRetryMetrics(
        task_group=task_group,
        baseline_success_rate=_pct(sum(1 for passed in baseline.values() if passed), total),
        retry_success_rate=_pct(retry_successes, len(retry)),
        success_once_rate=_pct(success_once, total),
        recovery_success_rate=_pct(recovered, failed),
        total_episodes=total,
        failed_episodes=failed,
        recovered_episodes=recovered,
        retry_budget=retry_budget,
    )
    return metrics, trace


def retry_task_ids_arg(plan: RetryPlan) -> str:
    return "[" + ",".join(str(task_id) for task_id in plan.failed_task_ids) + "]"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def comparison_markdown(metrics: AgenticRetryMetrics, baseline_eval: Path, retry_eval: Path) -> str:
    return "\n".join(
        [
            "# LIBERO Agentic Retry Probe",
            "",
            f"- task_group: `{metrics.task_group}`",
            f"- total_episodes: `{metrics.total_episodes}`",
            f"- retry_budget: `{metrics.retry_budget}`",
            f"- baseline_eval: `{baseline_eval}`",
            f"- retry_eval: `{retry_eval}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| baseline_success_rate | {metrics.baseline_success_rate:.2f} |",
            f"| retry_success_rate | {metrics.retry_success_rate:.2f} |",
            f"| success_once_rate | {metrics.success_once_rate:.2f} |",
            f"| recovery_success_rate | {metrics.recovery_success_rate:.2f} |",
            f"| failed_episodes | {metrics.failed_episodes} |",
            f"| recovered_episodes | {metrics.recovered_episodes} |",
            "",
            "## Semantics",
            "",
            "- `baseline_success_rate` is the policy-only benchmark success flag.",
            "- `success_once_rate` counts an episode as successful if the baseline passed or a retry for the same task/episode index passed.",
            "- This first wrapper uses the LIBERO benchmark success flag as the verifier. It is a basic retry wrapper, not yet a subgoal-level environment intervention.",
            "",
        ]
    )


def _episode_success_map(data: dict[str, Any], task_group: str) -> dict[tuple[int, int], bool]:
    result: dict[tuple[int, int], bool] = {}
    for item in data.get("per_task", []):
        if str(item.get("task_group", "")) != task_group:
            continue
        task_id = int(item.get("task_id"))
        successes = item.get("metrics", {}).get("successes", [])
        for episode_index, success in enumerate(successes):
            result[(task_id, episode_index)] = bool(success)
    return result


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and aggregate a basic LIBERO agentic retry wrapper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("baseline_eval_info", type=Path)
    plan_parser.add_argument("--task-group", required=True)
    plan_parser.add_argument("--retry-budget", type=int, default=1)
    plan_parser.add_argument("--output-json", type=Path, required=True)
    plan_parser.add_argument("--print-task-ids", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("baseline_eval_info", type=Path)
    aggregate_parser.add_argument("retry_eval_info", type=Path)
    aggregate_parser.add_argument("--task-group", required=True)
    aggregate_parser.add_argument("--retry-budget", type=int, default=1)
    aggregate_parser.add_argument("--output-json", type=Path, required=True)
    aggregate_parser.add_argument("--output-jsonl", type=Path, required=True)
    aggregate_parser.add_argument("--output-md", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        plan = build_retry_plan(load_eval_info(args.baseline_eval_info), args.task_group, args.retry_budget)
        write_json(args.output_json, asdict(plan))
        if args.print_task_ids:
            print(retry_task_ids_arg(plan))
    elif args.command == "aggregate":
        metrics, trace = aggregate_retry_metrics(
            load_eval_info(args.baseline_eval_info),
            load_eval_info(args.retry_eval_info),
            args.task_group,
            args.retry_budget,
        )
        write_json(args.output_json, asdict(metrics))
        write_jsonl(args.output_jsonl, trace)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            comparison_markdown(metrics, args.baseline_eval_info, args.retry_eval_info),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
