#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_LONG_PC_SUCCESS = 75.0
DEFAULT_REFERENCE_LONG_PC_SUCCESS = 77.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def number_values(rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    l2_values = number_values(rows, ("risk1b_diversity_metrics", "mean_normalized_pairwise_l2"))
    cosine_values = number_values(rows, ("risk1b_diversity_metrics", "mean_pairwise_cosine_distance"))
    min_l2_values = number_values(rows, ("risk1b_diversity_metrics", "min_pairwise_l2"))
    score_spreads = number_values(rows, ("score_spread",))
    pc_success_values = number_values(rows, ("pc_success",))
    selected = [row.get("selected_candidate_id") for row in rows if row.get("selected_candidate_id")]
    score_sources = [row.get("score_source") for row in rows if row.get("score_source")]
    non_baseline = [
        value
        for value in selected
        if isinstance(value, str) and value not in {"candidate_00", "candidate_00_policy_only", "policy_only"}
    ]
    policy_generated_rows = [
        row
        for row in rows
        if row.get("risk1b_provenance") in {"policy_generated", "actual_policy_generated", "external_vlm_json_policy_generated"}
    ]
    return {
        "row_count": len(rows),
        "suites": sorted({str(row.get("suite")) for row in rows if row.get("suite") is not None}),
        "task_ids": sorted({int(row["task_id"]) for row in rows if isinstance(row.get("task_id"), int)}),
        "seeds": sorted({int(row["seed"]) for row in rows if isinstance(row.get("seed"), int)}),
        "risk1b_policy_generated_rows": len(policy_generated_rows),
        "risk1b_mean_normalized_pairwise_l2": mean_or_none(l2_values),
        "risk1b_mean_pairwise_cosine_distance": mean_or_none(cosine_values),
        "risk1b_mean_min_pairwise_l2": mean_or_none(min_l2_values),
        "risk1c_mean_score_spread": mean_or_none(score_spreads),
        "risk1c_selected_candidate_counts": dict(Counter(str(value) for value in selected)),
        "risk1c_score_source_counts": dict(Counter(str(value) for value in score_sources)),
        "risk1c_non_baseline_selection_rate": round(len(non_baseline) / len(selected), 6) if selected else None,
        "smoke_pc_success_mean": mean_or_none(pc_success_values),
        "smoke_pc_success_values": pc_success_values,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {val}" for key, val in value.items()) or "n/a"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "n/a"
    return str(value)


def build_markdown(args: argparse.Namespace, summary: dict[str, Any]) -> str:
    experiment_label = args.experiment_label
    lines = [
        "# Risk1-B/C LIBERO Long-Horizon Comparison",
        "",
        f"- Generated: `{now_iso()}`",
        f"- Experiment: `{experiment_label}`",
        f"- Result root: `{args.risk1bc_root}`",
        f"- Rows: `{summary['row_count']}`; suites: `{fmt(summary['suites'])}`; task_ids: `{fmt(summary['task_ids'])}`; seeds: `{fmt(summary['seeds'])}`",
        "- Boundary: Risk1-B/C metrics are shallow OSMesa first-action-chunk candidate-generation/selector diagnostics, not EGL benchmark/deployment evidence.",
        "- Boundary: smoke `pc_success` is reported separately and must not be compared as a full closed-loop benchmark unless a full-horizon success protocol is explicitly run.",
        "",
        "| Evidence class | Our SmolVLA baseline | Reference paper number | Risk1-B/C alternative-goal experiment | Claim boundary |",
        "| --- | --- | --- | --- | --- |",
        (
            "| LIBERO long-horizon benchmark success | "
            f"{args.baseline_libero10_pc_success:.1f}% (`libero_10`, local baseline report) | "
            f"{args.reference_libero10_pc_success:.1f}% (`libero_10`, reference/paper comparator) | "
            f"smoke pc_success mean {fmt(summary['smoke_pc_success_mean'])}; values {fmt(summary['smoke_pc_success_values'])} | "
            "Experiment smoke is first-chunk diagnostic, not a fair full benchmark success comparison. |"
        ),
        (
            "| Risk1-B candidate diversity | n/a | n/a | "
            f"mean normalized L2 {fmt(summary['risk1b_mean_normalized_pairwise_l2'])}; "
            f"mean cosine distance {fmt(summary['risk1b_mean_pairwise_cosine_distance'])}; "
            f"mean min pairwise L2 {fmt(summary['risk1b_mean_min_pairwise_l2'])}; "
            f"policy-generated rows {summary['risk1b_policy_generated_rows']}/{summary['row_count']} | "
            "Candidate-generation diagnostic only; does not imply task success. |"
        ),
        (
            "| Risk1-C selector signal | n/a | n/a | "
            f"mean score spread {fmt(summary['risk1c_mean_score_spread'])}; "
            f"selected candidates {fmt(summary['risk1c_selected_candidate_counts'])}; "
            f"score sources {fmt(summary['risk1c_score_source_counts'])}; "
            f"non-baseline selection rate {fmt(summary['risk1c_non_baseline_selection_rate'])} | "
            "Selector proxy signal only; candidate selected by proxy is not proven better without full/fair rollout evidence. |"
        ),
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build side-by-side Risk1-B/C LIBERO long-horizon comparison tables.")
    parser.add_argument("--risk1bc-root", required=True, help="Root containing results.jsonl from the Risk1-B/C run.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-label", default="Risk1-B/C alternative-goal shallow OSMesa diagnostic")
    parser.add_argument("--baseline-libero10-pc-success", type=float, default=DEFAULT_BASELINE_LONG_PC_SUCCESS)
    parser.add_argument("--reference-libero10-pc-success", type=float, default=DEFAULT_REFERENCE_LONG_PC_SUCCESS)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.risk1bc_root)
    results_path = root / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"missing results.jsonl: {results_path}")
    rows = load_jsonl(results_path)
    summary = aggregate(rows)
    payload = {
        "timestamp": now_iso(),
        "experiment_label": args.experiment_label,
        "risk1bc_root": str(root),
        "baseline_libero10_pc_success": args.baseline_libero10_pc_success,
        "reference_libero10_pc_success": args.reference_libero10_pc_success,
        "summary": summary,
        "claim_boundary": (
            "Risk1-B/C is first-action-chunk alternative-goal candidate-generation/selector evidence. "
            "Do not promote it to EGL benchmark/deployment or full closed-loop success evidence."
        ),
    }
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "risk1bc_libero10_comparison_table.json"
    md_path = output_dir / "risk1bc_libero10_comparison_table.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(args, summary), encoding="utf-8")
    result = {"json": str(json_path), "markdown": str(md_path), "summary": summary}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"json={json_path}")
        print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
