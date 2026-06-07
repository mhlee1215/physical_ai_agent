#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuitePortfolio:
    suite: str
    baseline_success_rate: float
    best_single_condition: str
    best_single_success_rate: float
    portfolio_success_rate: float
    portfolio_delta: float
    failed_episodes: int
    portfolio_recovered_episodes: int
    conditions: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LIBERO portfolio retry report from suite-level retry traces.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    suites = [build_suite(path) for path in sorted(args.root.glob("libero_*")) if path.is_dir()]
    md = render_markdown(args.root, suites)
    payload = {
        "root": str(args.root),
        "suites": [suite.__dict__ for suite in suites],
        "macro_avg": macro_average(suites),
    }

    output_md = args.output_md or args.root / "agentic_retry_portfolio_report.md"
    output_json = args.output_json or args.root / "agentic_retry_portfolio_summary.json"
    output_md.write_text(md, encoding="utf-8")
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={output_md}")
    print(f"summary={output_json}")


def build_suite(suite_dir: Path) -> SuitePortfolio:
    condition_traces: dict[str, dict[tuple[int, int], dict[str, object]]] = {}
    condition_rates: dict[str, float] = {}
    baseline_rate = 0.0
    for trace_path in sorted(suite_dir.glob("*_seed*/agentic/agentic_retry_trace.jsonl")):
        condition = trace_path.parent.parent.name.rsplit("_seed", 1)[0]
        trace = load_trace(trace_path)
        condition_traces[condition] = trace
        metrics_path = trace_path.parent / "agentic_retry_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        condition_rates[condition] = float(metrics["success_once_rate"])
        baseline_rate = float(metrics["baseline_success_rate"])

    if not condition_traces:
        raise ValueError(f"No condition traces found in {suite_dir}")

    keys = sorted(next(iter(condition_traces.values())))
    baseline_successes = 0
    portfolio_successes = 0
    failed = 0
    recovered = 0
    for key in keys:
        first_item = next(iter(condition_traces.values()))[key]
        baseline_success = bool(first_item["baseline_success"])
        retry_success = any(bool(trace[key]["retry_success"]) for trace in condition_traces.values()) if not baseline_success else False
        baseline_successes += int(baseline_success)
        if not baseline_success:
            failed += 1
            recovered += int(retry_success)
        portfolio_successes += int(baseline_success or retry_success)

    portfolio_rate = pct(portfolio_successes, len(keys))
    best_condition, best_rate = max(condition_rates.items(), key=lambda item: item[1])
    return SuitePortfolio(
        suite=suite_dir.name,
        baseline_success_rate=baseline_rate,
        best_single_condition=best_condition,
        best_single_success_rate=best_rate,
        portfolio_success_rate=portfolio_rate,
        portfolio_delta=portfolio_rate - baseline_rate,
        failed_episodes=failed,
        portfolio_recovered_episodes=recovered,
        conditions=condition_rates,
    )


def load_trace(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    result: dict[tuple[int, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            result[(int(item["task_id"]), int(item["episode_index"]))] = item
    return result


def macro_average(suites: list[SuitePortfolio]) -> dict[str, float]:
    if not suites:
        return {}
    return {
        "baseline_success_rate": sum(suite.baseline_success_rate for suite in suites) / len(suites),
        "best_single_success_rate": sum(suite.best_single_success_rate for suite in suites) / len(suites),
        "portfolio_success_rate": sum(suite.portfolio_success_rate for suite in suites) / len(suites),
        "portfolio_delta": sum(suite.portfolio_delta for suite in suites) / len(suites),
    }


def render_markdown(root: Path, suites: list[SuitePortfolio]) -> str:
    lines = [
        "# LIBERO Agentic Retry Portfolio Report",
        "",
        f"- root: `{root}`",
        "- portfolio_budget2: baseline success or any retry condition succeeds",
        "",
        "## Suite Results",
        "",
        "| Suite | Baseline | Best single | Portfolio budget2 | Delta vs baseline | Recovered | Conditions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for suite in suites:
        conditions = ", ".join(f"{name}={rate:.2f}" for name, rate in sorted(suite.conditions.items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    suite.suite,
                    f"{suite.baseline_success_rate:.2f}",
                    f"{suite.best_single_success_rate:.2f} ({suite.best_single_condition})",
                    f"{suite.portfolio_success_rate:.2f}",
                    f"{suite.portfolio_delta:+.2f}",
                    f"{suite.portfolio_recovered_episodes}/{suite.failed_episodes}",
                    f"`{conditions}`",
                ]
            )
            + " |"
        )

    avg = macro_average(suites)
    lines.extend(
        [
            "",
            "## Macro Average",
            "",
            "| Baseline | Best single | Portfolio budget2 | Delta vs baseline |",
            "| ---: | ---: | ---: | ---: |",
            (
                f"| {avg.get('baseline_success_rate', 0.0):.2f} | "
                f"{avg.get('best_single_success_rate', 0.0):.2f} | "
                f"{avg.get('portfolio_success_rate', 0.0):.2f} | "
                f"{avg.get('portfolio_delta', 0.0):+.2f} |"
            ),
            "",
            "## Interpretation Guardrail",
            "",
            "- This report summarizes seed-1000 remaining-suite probes unless the input root contains more seeds.",
            "- `portfolio_budget2` spends two retry attempts after baseline failure, so compare it against retry-budget controls, not policy-only alone.",
            "- These are episode-level retries, not in-episode replanning.",
            "",
        ]
    )
    return "\n".join(lines)


def pct(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else 100.0 * numerator / denominator


if __name__ == "__main__":
    main()
