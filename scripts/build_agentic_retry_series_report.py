#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


@dataclass(frozen=True)
class SeriesRow:
    condition: str
    base_seed: int
    retry_seed: int
    run_dir: str
    baseline_success_rate: float
    retry_success_rate: float
    success_once_rate: float
    recovery_success_rate: float
    total_episodes: int
    failed_episodes: int
    recovered_episodes: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a summary report for LIBERO agentic retry series runs.")
    parser.add_argument("series_root", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.series_root)
    md = render_markdown(args.series_root, rows)
    payload = {
        "series_root": str(args.series_root),
        "runs": [row.__dict__ for row in rows],
        "conditions": summarize_conditions(rows),
    }

    output_md = args.output_md or args.series_root / "agentic_retry_series_report.md"
    output_json = args.output_json or args.series_root / "agentic_retry_series_summary.json"
    output_md.write_text(md, encoding="utf-8")
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={output_md}")
    print(f"summary={output_json}")


def load_rows(series_root: Path) -> list[SeriesRow]:
    manifest_path = series_root / "series_manifest.jsonl"
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            manifest[str(item["run_dir"])] = item

    rows: list[SeriesRow] = []
    for metrics_path in sorted(series_root.glob("*/agentic/agentic_retry_metrics.json")):
        run_dir = metrics_path.parent.parent.name
        meta = manifest.get(run_dir, {})
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            SeriesRow(
                condition=str(meta.get("condition", infer_condition(run_dir))),
                base_seed=int(meta.get("base_seed", infer_seed(run_dir))),
                retry_seed=int(meta.get("retry_seed", -1)),
                run_dir=run_dir,
                baseline_success_rate=float(metrics.get("baseline_success_rate", 0.0)),
                retry_success_rate=float(metrics.get("retry_success_rate", 0.0)),
                success_once_rate=float(metrics.get("success_once_rate", 0.0)),
                recovery_success_rate=float(metrics.get("recovery_success_rate", 0.0)),
                total_episodes=int(metrics.get("total_episodes", 0)),
                failed_episodes=int(metrics.get("failed_episodes", 0)),
                recovered_episodes=int(metrics.get("recovered_episodes", 0)),
            )
        )
    return rows


def infer_condition(run_dir: str) -> str:
    if "_seed" in run_dir:
        return run_dir.rsplit("_seed", 1)[0]
    return run_dir


def infer_seed(run_dir: str) -> int:
    if "_seed" not in run_dir:
        return -1
    suffix = run_dir.rsplit("_seed", 1)[1]
    digits = ""
    for char in suffix:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else -1


def summarize_conditions(rows: list[SeriesRow]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for condition in sorted({row.condition for row in rows}):
        selected = [row for row in rows if row.condition == condition]
        baselines = [row.baseline_success_rate for row in selected]
        success_once = [row.success_once_rate for row in selected]
        deltas = [row.success_once_rate - row.baseline_success_rate for row in selected]
        recoveries = [row.recovery_success_rate for row in selected]
        summaries[condition] = {
            "runs": float(len(selected)),
            "baseline_mean": mean(baselines),
            "baseline_std": pstdev(baselines) if len(baselines) > 1 else 0.0,
            "success_once_mean": mean(success_once),
            "success_once_std": pstdev(success_once) if len(success_once) > 1 else 0.0,
            "delta_mean": mean(deltas),
            "delta_std": pstdev(deltas) if len(deltas) > 1 else 0.0,
            "recovery_mean": mean(recoveries),
            "recovery_std": pstdev(recoveries) if len(recoveries) > 1 else 0.0,
        }
    return summaries


def render_markdown(series_root: Path, rows: list[SeriesRow]) -> str:
    lines = [
        "# LIBERO Agentic Retry Series Report",
        "",
        f"- series_root: `{series_root}`",
        f"- completed_runs: `{len(rows)}`",
        "",
        "## Per-Run Results",
        "",
        "| Condition | Base seed | Retry seed | Episodes | Baseline | Success once | Delta | Recovery | Recovered | Run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        delta = row.success_once_rate - row.baseline_success_rate
        lines.append(
            "| "
            + " | ".join(
                [
                    row.condition,
                    str(row.base_seed),
                    str(row.retry_seed),
                    str(row.total_episodes),
                    f"{row.baseline_success_rate:.2f}",
                    f"{row.success_once_rate:.2f}",
                    f"{delta:+.2f}",
                    f"{row.recovery_success_rate:.2f}",
                    f"{row.recovered_episodes}/{row.failed_episodes}",
                    f"`{row.run_dir}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Condition Summary", ""])
    lines.append("| Condition | Runs | Baseline mean | Success-once mean | Delta mean | Recovery mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for condition, summary in summarize_conditions(rows).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(int(summary["runs"])),
                    f"{summary['baseline_mean']:.2f} +/- {summary['baseline_std']:.2f}",
                    f"{summary['success_once_mean']:.2f} +/- {summary['success_once_std']:.2f}",
                    f"{summary['delta_mean']:+.2f} +/- {summary['delta_std']:.2f}",
                    f"{summary['recovery_mean']:.2f} +/- {summary['recovery_std']:.2f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- `blind_new_seed` is the control for retry budget without changing the policy action horizon.",
            "- `alternate_steps10` tests whether changing the retry action horizon recovers additional failed episodes.",
            "- This remains an episode-level retry wrapper. It is not yet a subgoal-level in-episode replanning controller.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
