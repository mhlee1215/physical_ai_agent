#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}


@dataclass(frozen=True)
class ConditionData:
    trace: dict[tuple[int, int], dict[str, object]]
    success_once_rate: float
    retry_attempts: int
    retry_eval_seconds: float


@dataclass(frozen=True)
class SuiteSeedRow:
    suite: str
    seed: int
    episodes: int
    baseline_success_rate: float
    best_single_condition: str
    best_single_success_rate: float
    portfolio_success_rate: float
    portfolio_delta: float
    failed_episodes: int
    recovered_episodes: int
    conditions: dict[str, float]
    baseline_eval_seconds: float
    portfolio_retry_attempts: int
    portfolio_total_attempts: int
    portfolio_extra_environment_resets: int
    portfolio_eval_seconds: float
    portfolio_success_once_per_attempt: float
    portfolio_success_once_per_eval_minute: float
    action_step_count_available: bool
    action_step_count_source: str

    @property
    def recovery_rate(self) -> float:
        return pct(self.recovered_episodes, self.failed_episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a four-suite LIBERO retry portfolio report across seeds.")
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--remaining-root", type=Path, action="append", default=[])
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    rows.extend(load_suite_rows("libero_10", args.long_root))
    for root in args.remaining_root:
        for suite_dir in sorted(path for path in root.glob("libero_*") if path.is_dir()):
            rows.extend(load_suite_rows(suite_dir.name, suite_dir))

    rows = sorted(rows, key=lambda row: (row.seed, suite_sort_key(row.suite)))
    if not rows:
        raise ValueError("No agentic retry traces found")

    payload = {
        "long_root": str(args.long_root),
        "remaining_roots": [str(root) for root in args.remaining_root],
        "rows": [serialize_row(row) for row in rows],
        "suite_summary": summarize_by_suite(rows),
        "seed_macro_summary": summarize_by_seed(rows),
        "overall_macro_summary": summarize_overall(rows),
    }
    args.output_md.write_text(render_markdown(payload, rows), encoding="utf-8")
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={args.output_md}")
    print(f"summary={args.output_json}")


def load_suite_rows(suite: str, suite_root: Path) -> list[SuiteSeedRow]:
    by_seed: dict[int, dict[str, ConditionData]] = defaultdict(dict)
    for trace_path in sorted(suite_root.glob("*_seed*/agentic/agentic_retry_trace.jsonl")):
        condition_seed = trace_path.parent.parent.name
        condition, seed_text = condition_seed.rsplit("_seed", 1)
        seed = int(seed_text)
        metrics = json.loads((trace_path.parent / "agentic_retry_metrics.json").read_text(encoding="utf-8"))
        retry_eval = load_json(trace_path.parent.parent / "retry" / "eval_logs" / "eval_info.json")
        by_seed[seed][condition] = ConditionData(
            trace=load_trace(trace_path),
            success_once_rate=float(metrics["success_once_rate"]),
            retry_attempts=count_episodes(retry_eval, suite),
            retry_eval_seconds=eval_seconds(retry_eval),
        )

    rows = []
    for seed, conditions in sorted(by_seed.items()):
        if not conditions:
            continue
        keys = sorted(next(iter(conditions.values())).trace)
        baseline_eval = load_json(suite_root / f"baseline_seed{seed}" / "eval_logs" / "eval_info.json")
        baseline_eval_seconds = eval_seconds(baseline_eval)
        baseline_successes = 0
        portfolio_successes = 0
        failed = 0
        recovered = 0
        for key in keys:
            first_trace = next(iter(conditions.values())).trace
            baseline_success = bool(first_trace[key]["baseline_success"])
            retry_success = any(bool(data.trace[key]["retry_success"]) for data in conditions.values()) if not baseline_success else False
            baseline_successes += int(baseline_success)
            if not baseline_success:
                failed += 1
                recovered += int(retry_success)
            portfolio_successes += int(baseline_success or retry_success)

        condition_rates = {condition: data.success_once_rate for condition, data in sorted(conditions.items())}
        best_condition, best_rate = max(condition_rates.items(), key=lambda item: item[1])
        baseline_rate = pct(baseline_successes, len(keys))
        portfolio_rate = pct(portfolio_successes, len(keys))
        portfolio_retry_attempts = sum(data.retry_attempts for data in conditions.values())
        portfolio_total_attempts = len(keys) + portfolio_retry_attempts
        portfolio_eval_seconds = baseline_eval_seconds + sum(data.retry_eval_seconds for data in conditions.values())
        rows.append(
            SuiteSeedRow(
                suite=suite,
                seed=seed,
                episodes=len(keys),
                baseline_success_rate=baseline_rate,
                best_single_condition=best_condition,
                best_single_success_rate=best_rate,
                portfolio_success_rate=portfolio_rate,
                portfolio_delta=portfolio_rate - baseline_rate,
                failed_episodes=failed,
                recovered_episodes=recovered,
                conditions=condition_rates,
                baseline_eval_seconds=baseline_eval_seconds,
                portfolio_retry_attempts=portfolio_retry_attempts,
                portfolio_total_attempts=portfolio_total_attempts,
                portfolio_extra_environment_resets=portfolio_retry_attempts,
                portfolio_eval_seconds=portfolio_eval_seconds,
                portfolio_success_once_per_attempt=ratio(portfolio_successes, portfolio_total_attempts),
                portfolio_success_once_per_eval_minute=ratio(portfolio_successes, portfolio_eval_seconds / 60.0),
                action_step_count_available=False,
                action_step_count_source="not_recorded_in_lerobot_eval_info",
            )
        )
    return rows


def load_trace(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            result[(int(item["task_id"]), int(item["episode_index"]))] = item
    return result


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_by_suite(rows: list[SuiteSeedRow]) -> dict[str, dict[str, float]]:
    output = {}
    for suite in sorted({row.suite for row in rows}, key=suite_sort_key):
        selected = [row for row in rows if row.suite == suite]
        output[suite] = summarize_rows(selected)
    return output


def summarize_by_seed(rows: list[SuiteSeedRow]) -> dict[str, dict[str, float]]:
    output = {}
    for seed in sorted({row.seed for row in rows}):
        selected = [row for row in rows if row.seed == seed]
        output[str(seed)] = {
            "suites": float(len(selected)),
            "baseline_macro": mean(row.baseline_success_rate for row in selected),
            "best_single_macro": mean(row.best_single_success_rate for row in selected),
            "portfolio_macro": mean(row.portfolio_success_rate for row in selected),
            "portfolio_delta_macro": mean(row.portfolio_delta for row in selected),
        }
    return output


def summarize_overall(rows: list[SuiteSeedRow]) -> dict[str, float]:
    return summarize_rows(rows)


def summarize_rows(rows: list[SuiteSeedRow]) -> dict[str, float]:
    return {
        "runs": float(len(rows)),
        "baseline_mean": mean(row.baseline_success_rate for row in rows),
        "baseline_std": pstdev(row.baseline_success_rate for row in rows) if len(rows) > 1 else 0.0,
        "best_single_mean": mean(row.best_single_success_rate for row in rows),
        "best_single_std": pstdev(row.best_single_success_rate for row in rows) if len(rows) > 1 else 0.0,
        "portfolio_mean": mean(row.portfolio_success_rate for row in rows),
        "portfolio_std": pstdev(row.portfolio_success_rate for row in rows) if len(rows) > 1 else 0.0,
        "portfolio_delta_mean": mean(row.portfolio_delta for row in rows),
        "portfolio_delta_std": pstdev(row.portfolio_delta for row in rows) if len(rows) > 1 else 0.0,
        "recovery_mean": mean(row.recovery_rate for row in rows),
        "recovery_std": pstdev(row.recovery_rate for row in rows) if len(rows) > 1 else 0.0,
        "portfolio_total_attempts_mean": mean(row.portfolio_total_attempts for row in rows),
        "portfolio_eval_seconds_mean": mean(row.portfolio_eval_seconds for row in rows),
        "portfolio_success_once_per_attempt_mean": mean(row.portfolio_success_once_per_attempt for row in rows),
        "portfolio_success_once_per_eval_minute_mean": mean(row.portfolio_success_once_per_eval_minute for row in rows),
    }


def render_markdown(payload: dict[str, object], rows: list[SuiteSeedRow]) -> str:
    suite_summary = payload["suite_summary"]  # type: ignore[assignment]
    seed_summary = payload["seed_macro_summary"]  # type: ignore[assignment]
    overall = payload["overall_macro_summary"]  # type: ignore[assignment]

    lines = [
        "# LIBERO Four-Suite Agentic Retry Portfolio Report",
        "",
        f"- long_root: `{payload['long_root']}`",
        f"- remaining_roots: `{', '.join(payload['remaining_roots'])}`",
        "- portfolio_budget2: baseline success or any retry condition succeeds after a baseline failure",
        "- protocol: episode-level retry budget, not in-episode replanning",
        "",
        "## Per-Suite Seed Results",
        "",
        "| Suite | Seed | Episodes | Baseline | Best single retry | Portfolio budget2 | Delta | Recovery | Portfolio attempts | Success/attempt | Success/eval min | Conditions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        conditions = ", ".join(f"{name}={rate:.2f}" for name, rate in sorted(row.conditions.items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    SUITE_LABELS.get(row.suite, row.suite),
                    str(row.seed),
                    str(row.episodes),
                    f"{row.baseline_success_rate:.2f}",
                    f"{row.best_single_success_rate:.2f} (`{row.best_single_condition}`)",
                    f"{row.portfolio_success_rate:.2f}",
                    f"{row.portfolio_delta:+.2f}",
                    f"{row.recovered_episodes}/{row.failed_episodes} ({row.recovery_rate:.2f})",
                    str(row.portfolio_total_attempts),
                    f"{row.portfolio_success_once_per_attempt:.4f}",
                    f"{row.portfolio_success_once_per_eval_minute:.2f}",
                    f"`{conditions}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Suite Summary", ""])
    lines.append("| Suite | Runs | Baseline | Best single | Portfolio budget2 | Delta | Recovery | Attempts mean | Success/attempt | Success/eval min |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for suite, item in suite_summary.items():  # type: ignore[union-attr]
        lines.append(summary_line(SUITE_LABELS.get(suite, suite), item))

    lines.extend(["", "## Seed Macro Summary", ""])
    lines.append("| Seed | Suites | Baseline macro | Best single macro | Portfolio macro | Delta |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for seed, item in seed_summary.items():  # type: ignore[union-attr]
        lines.append(
            f"| {seed} | {int(item['suites'])} | {item['baseline_macro']:.2f} | "
            f"{item['best_single_macro']:.2f} | {item['portfolio_macro']:.2f} | {item['portfolio_delta_macro']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Overall Macro Summary",
            "",
            "| Runs | Baseline | Best single | Portfolio budget2 | Delta | Recovery | Attempts mean | Success/attempt | Success/eval min |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            summary_line(str(int(overall["runs"])), overall, include_label=False),  # type: ignore[index]
            "",
            "## Interpretation Guardrail",
            "",
            "- Compare `portfolio_budget2` against retry-budget controls, not against policy-only alone.",
            "- `Portfolio attempts` counts all baseline episodes plus all retry episodes actually evaluated for both retry conditions.",
            "- The benchmark success flag remains the final success metric; retry traces only decide whether to rerun failed task/episode indexes.",
            "- A strong blind-retry result means this is currently evidence for retry-budget scaling more than evidence for intelligent failure diagnosis.",
            "- Per-episode action-step counts are not recorded in the current LeRobot `eval_info.json`; action-step-normalized metrics require an instrumented rollout path.",
            "",
        ]
    )
    return "\n".join(lines)


def summary_line(label: str, item: dict[str, float], include_label: bool = True) -> str:
    cells = []
    if include_label:
        cells.append(label)
    cells.extend(
        [
            str(int(item["runs"])),
            f"{item['baseline_mean']:.2f} +/- {item['baseline_std']:.2f}",
            f"{item['best_single_mean']:.2f} +/- {item['best_single_std']:.2f}",
            f"{item['portfolio_mean']:.2f} +/- {item['portfolio_std']:.2f}",
            f"{item['portfolio_delta_mean']:+.2f} +/- {item['portfolio_delta_std']:.2f}",
            f"{item['recovery_mean']:.2f} +/- {item['recovery_std']:.2f}",
            f"{item['portfolio_total_attempts_mean']:.2f}",
            f"{item['portfolio_success_once_per_attempt_mean']:.4f}",
            f"{item['portfolio_success_once_per_eval_minute_mean']:.2f}",
        ]
    )
    return "| " + " | ".join(cells) + " |"


def serialize_row(row: SuiteSeedRow) -> dict[str, object]:
    return row.__dict__ | {"recovery_rate": row.recovery_rate}


def suite_sort_key(suite: str) -> int:
    order = {"libero_goal": 0, "libero_object": 1, "libero_spatial": 2, "libero_10": 3}
    return order.get(suite, 99)


def pct(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else 100.0 * numerator / denominator


def ratio(numerator: int, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def eval_seconds(data: dict[str, object]) -> float:
    overall = data.get("overall", {})
    if not isinstance(overall, dict):
        return 0.0
    try:
        return float(overall.get("eval_s", 0.0))
    except (TypeError, ValueError):
        return 0.0


def count_episodes(data: dict[str, object], suite: str) -> int:
    total = 0
    for item in data.get("per_task", []):
        if not isinstance(item, dict) or str(item.get("task_group", "")) != suite:
            continue
        metrics = item.get("metrics", {})
        if isinstance(metrics, dict):
            total += len(metrics.get("successes", []))
    return total


if __name__ == "__main__":
    main()
