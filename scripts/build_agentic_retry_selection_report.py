#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


CONDITIONS = ("blind_new_seed", "alternate_steps10")


@dataclass(frozen=True)
class EpisodeKey:
    seed: int
    task_id: int
    episode_index: int


@dataclass(frozen=True)
class SelectionRow:
    selector: str
    seed: int
    total_episodes: int
    baseline_successes: int
    selected_successes: int
    failed_episodes: int
    recovered_episodes: int

    @property
    def baseline_rate(self) -> float:
        return pct(self.baseline_successes, self.total_episodes)

    @property
    def success_once_rate(self) -> float:
        return pct(self.selected_successes, self.total_episodes)

    @property
    def delta(self) -> float:
        return self.success_once_rate - self.baseline_rate

    @property
    def recovery_rate(self) -> float:
        return pct(self.recovered_episodes, self.failed_episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task-guided retry selection analysis from retry traces.")
    parser.add_argument("series_root", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    traces = load_traces(args.series_root)
    seeds = sorted({key.seed for key in traces[CONDITIONS[0]]})
    rows = build_rows(traces, seeds)
    task_maps = {seed: learn_task_policy(traces, [s for s in seeds if s != seed]) for seed in seeds}
    oracle_maps = {seed: learn_task_policy(traces, [seed]) for seed in seeds}

    output_md = args.output_md or args.series_root / "agentic_retry_selection_report.md"
    output_json = args.output_json or args.series_root / "agentic_retry_selection_summary.json"
    payload = {
        "series_root": str(args.series_root),
        "rows": [row.__dict__ | {
            "baseline_rate": row.baseline_rate,
            "success_once_rate": row.success_once_rate,
            "delta": row.delta,
            "recovery_rate": row.recovery_rate,
        } for row in rows],
        "summary": summarize(rows),
        "leave_one_seed_out_task_maps": task_maps,
        "same_seed_task_oracle_maps": oracle_maps,
    }

    output_md.write_text(render_markdown(args.series_root, rows, task_maps, oracle_maps), encoding="utf-8")
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={output_md}")
    print(f"summary={output_json}")


def load_traces(series_root: Path) -> dict[str, dict[EpisodeKey, dict[str, object]]]:
    result: dict[str, dict[EpisodeKey, dict[str, object]]] = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for path in sorted(series_root.glob(f"{condition}_seed*/agentic/agentic_retry_trace.jsonl")):
            seed = int(path.parent.parent.name.rsplit("_seed", 1)[1])
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    key = EpisodeKey(seed=seed, task_id=int(item["task_id"]), episode_index=int(item["episode_index"]))
                    result[condition][key] = item
    return result


def build_rows(traces: dict[str, dict[EpisodeKey, dict[str, object]]], seeds: list[int]) -> list[SelectionRow]:
    rows: list[SelectionRow] = []
    for seed in seeds:
        for condition in CONDITIONS:
            rows.append(score_selector(condition, seed, traces, lambda _task_id, c=condition: c))
        loso_map = learn_task_policy(traces, [s for s in seeds if s != seed])
        rows.append(score_selector("task_guided_loso", seed, traces, lambda task_id, m=loso_map: m.get(str(task_id), "blind_new_seed")))
        oracle_map = learn_task_policy(traces, [seed])
        rows.append(score_selector("task_oracle_same_seed", seed, traces, lambda task_id, m=oracle_map: m.get(str(task_id), "blind_new_seed")))
        rows.append(score_portfolio_budget2(seed, traces))
    return rows


def score_selector(
    name: str,
    seed: int,
    traces: dict[str, dict[EpisodeKey, dict[str, object]]],
    selector: object,
) -> SelectionRow:
    baseline = traces[CONDITIONS[0]]
    keys = sorted((key for key in baseline if key.seed == seed), key=lambda key: (key.task_id, key.episode_index))
    baseline_successes = 0
    selected_successes = 0
    failed = 0
    recovered = 0
    for key in keys:
        baseline_success = bool(baseline[key]["baseline_success"])
        condition = selector(key.task_id)  # type: ignore[operator]
        retry_success = bool(traces[condition][key]["retry_success"]) if not baseline_success else False
        baseline_successes += int(baseline_success)
        if not baseline_success:
            failed += 1
            recovered += int(retry_success)
        selected_successes += int(baseline_success or retry_success)
    return SelectionRow(
        selector=name,
        seed=seed,
        total_episodes=len(keys),
        baseline_successes=baseline_successes,
        selected_successes=selected_successes,
        failed_episodes=failed,
        recovered_episodes=recovered,
    )


def score_portfolio_budget2(seed: int, traces: dict[str, dict[EpisodeKey, dict[str, object]]]) -> SelectionRow:
    baseline = traces[CONDITIONS[0]]
    keys = sorted((key for key in baseline if key.seed == seed), key=lambda key: (key.task_id, key.episode_index))
    baseline_successes = 0
    selected_successes = 0
    failed = 0
    recovered = 0
    for key in keys:
        baseline_success = bool(baseline[key]["baseline_success"])
        retry_success = any(bool(traces[condition][key]["retry_success"]) for condition in CONDITIONS) if not baseline_success else False
        baseline_successes += int(baseline_success)
        if not baseline_success:
            failed += 1
            recovered += int(retry_success)
        selected_successes += int(baseline_success or retry_success)
    return SelectionRow(
        selector="portfolio_budget2",
        seed=seed,
        total_episodes=len(keys),
        baseline_successes=baseline_successes,
        selected_successes=selected_successes,
        failed_episodes=failed,
        recovered_episodes=recovered,
    )


def learn_task_policy(traces: dict[str, dict[EpisodeKey, dict[str, object]]], train_seeds: Iterable[int]) -> dict[str, str]:
    train_seed_set = set(train_seeds)
    task_condition_counts: dict[int, dict[str, list[int]]] = defaultdict(lambda: {condition: [0, 0] for condition in CONDITIONS})
    baseline = traces[CONDITIONS[0]]
    for key, item in baseline.items():
        if key.seed not in train_seed_set or bool(item["baseline_success"]):
            continue
        for condition in CONDITIONS:
            task_condition_counts[key.task_id][condition][1] += 1
            task_condition_counts[key.task_id][condition][0] += int(bool(traces[condition][key]["retry_success"]))

    policy: dict[str, str] = {}
    for task_id, counts in sorted(task_condition_counts.items()):
        rates = {
            condition: (counts[condition][0] / counts[condition][1] if counts[condition][1] else 0.0)
            for condition in CONDITIONS
        }
        if rates["alternate_steps10"] > rates["blind_new_seed"]:
            policy[str(task_id)] = "alternate_steps10"
        else:
            policy[str(task_id)] = "blind_new_seed"
    return policy


def summarize(rows: list[SelectionRow]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for selector in sorted({row.selector for row in rows}):
        selected = [row for row in rows if row.selector == selector]
        output[selector] = {
            "runs": float(len(selected)),
            "baseline_mean": mean(row.baseline_rate for row in selected),
            "baseline_std": pstdev(row.baseline_rate for row in selected) if len(selected) > 1 else 0.0,
            "success_once_mean": mean(row.success_once_rate for row in selected),
            "success_once_std": pstdev(row.success_once_rate for row in selected) if len(selected) > 1 else 0.0,
            "delta_mean": mean(row.delta for row in selected),
            "delta_std": pstdev(row.delta for row in selected) if len(selected) > 1 else 0.0,
            "recovery_mean": mean(row.recovery_rate for row in selected),
            "recovery_std": pstdev(row.recovery_rate for row in selected) if len(selected) > 1 else 0.0,
        }
    return output


def render_markdown(
    series_root: Path,
    rows: list[SelectionRow],
    task_maps: dict[int, dict[str, str]],
    oracle_maps: dict[int, dict[str, str]],
) -> str:
    lines = [
        "# LIBERO Agentic Retry Selection Report",
        "",
        f"- series_root: `{series_root}`",
        "- selector_input: `task_id` plus baseline failure from the benchmark success verifier",
        "",
        "## Per-Seed Results",
        "",
        "| Selector | Seed | Episodes | Baseline | Success once | Delta | Recovery | Recovered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.selector,
                    str(row.seed),
                    str(row.total_episodes),
                    f"{row.baseline_rate:.2f}",
                    f"{row.success_once_rate:.2f}",
                    f"{row.delta:+.2f}",
                    f"{row.recovery_rate:.2f}",
                    f"{row.recovered_episodes}/{row.failed_episodes}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Summary", ""])
    lines.append("| Selector | Runs | Baseline mean | Success-once mean | Delta mean | Recovery mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for selector, item in summarize(rows).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    selector,
                    str(int(item["runs"])),
                    f"{item['baseline_mean']:.2f} +/- {item['baseline_std']:.2f}",
                    f"{item['success_once_mean']:.2f} +/- {item['success_once_std']:.2f}",
                    f"{item['delta_mean']:+.2f} +/- {item['delta_std']:.2f}",
                    f"{item['recovery_mean']:.2f} +/- {item['recovery_std']:.2f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Leave-One-Seed-Out Task Policies", ""])
    lines.append("| Held-out seed | Task policy |")
    lines.append("| ---: | --- |")
    for seed, mapping in sorted(task_maps.items()):
        lines.append(f"| {seed} | `{json.dumps(mapping, sort_keys=True)}` |")

    lines.extend(["", "## Same-Seed Task Oracle Policies", ""])
    lines.append("| Seed | Task policy |")
    lines.append("| ---: | --- |")
    for seed, mapping in sorted(oracle_maps.items()):
        lines.append(f"| {seed} | `{json.dumps(mapping, sort_keys=True)}` |")

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- `task_guided_loso` is cross-validated by seed, but it still uses task identity rather than visual failure diagnosis.",
            "- `portfolio_budget2` is deployable if the evaluation protocol allows two retries after baseline failure.",
            "- `task_oracle_same_seed` is an upper bound, not a deployable policy.",
            "- A paper-grade agentic wrapper still needs a verifier or planner signal richer than task id alone.",
            "",
        ]
    )
    return "\n".join(lines)


def pct(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else 100.0 * numerator / denominator


if __name__ == "__main__":
    main()
