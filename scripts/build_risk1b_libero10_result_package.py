#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_LIBERO10_PC_SUCCESS = 75.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Risk1-B LIBERO-10 paper-table result package.")
    parser.add_argument("--summary", type=Path, required=True, help="Full LIBERO-10 run summary.json")
    parser.add_argument("--ablation-summary", type=Path, default=None, help="Optional candidate_ablation_summary.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-pc-success", type=float, default=BASELINE_LIBERO10_PC_SUCCESS)
    parser.add_argument("--json", action="store_true")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_package(
    *,
    summary: dict[str, Any],
    baseline_pc_success: float = BASELINE_LIBERO10_PC_SUCCESS,
    ablation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_episodes = int(summary.get("total_episodes") or 0)
    pc_success = summary.get("pc_success")
    pc_success_value = float(pc_success) if pc_success is not None else None
    delta = None if pc_success_value is None else pc_success_value - float(baseline_pc_success)
    rows = [row for row in summary.get("rows", []) if isinstance(row, dict)]
    weak_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("pc_success") if row.get("pc_success") is not None else 101.0),
            int(row.get("task_id") or 0),
        ),
    )[:3]
    fallback_rows = summary.get("fallback_rows") or []
    invalid_qwen_rows = summary.get("invalid_qwen_rows") or []
    full_table_ready = (
        total_episodes == 100
        and pc_success_value is not None
        and not fallback_rows
        and not invalid_qwen_rows
    )
    verdict = "PASS" if full_table_ready and pc_success_value >= baseline_pc_success else "NEGATIVE_OR_INCOMPLETE"
    if not full_table_ready:
        verdict = "INCOMPLETE_OR_BLOCKED"
    package = {
        "baseline": {
            "method": "SmolVLA baseline",
            "pc_success": baseline_pc_success,
            "episodes": 100,
        },
        "risk1b": {
            "method": "Risk1-B alternative-goal selector",
            "pc_success": pc_success_value,
            "episodes": total_episodes,
            "delta_vs_baseline_pp": delta,
            "status": summary.get("status"),
            "lane": summary.get("lane"),
            "fallback_rows": fallback_rows,
            "invalid_qwen_rows": invalid_qwen_rows,
        },
        "verdict": verdict,
        "paper_table_ready": full_table_ready,
        "allowed_claims": allowed_claims(full_table_ready, pc_success_value, baseline_pc_success, summary),
        "disallowed_claims": disallowed_claims(summary),
        "weak_rows": weak_rows,
        "ablation": summarize_ablation(ablation_summary) if ablation_summary else None,
    }
    return package


def allowed_claims(
    full_table_ready: bool,
    pc_success: float | None,
    baseline_pc_success: float,
    summary: dict[str, Any],
) -> list[str]:
    claims = []
    if full_table_ready:
        claims.append(
            f"Full LIBERO-10 100-episode comparison completed in lane: {summary.get('lane', 'unknown')}"
        )
        if pc_success is not None and pc_success >= baseline_pc_success:
            claims.append("Risk1-B exceeded the internal SmolVLA LIBERO-10 baseline in this lane.")
        elif pc_success is not None:
            claims.append("Risk1-B did not exceed the internal SmolVLA LIBERO-10 baseline in this lane.")
    else:
        claims.append("Run is not full paper-table evidence because coverage or Qwen gate requirements are incomplete.")
    return claims


def disallowed_claims(summary: dict[str, Any]) -> list[str]:
    claims = [
        "Do not claim EGL/deployment benchmark evidence from a shallow OSMesa lane.",
        "Do not count deterministic fallback rows as Qwen-only paper evidence.",
    ]
    if summary.get("total_episodes") != 100:
        claims.append("Do not compare against the 100-episode baseline as a complete table row.")
    return claims


def summarize_ablation(ablation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ablation:
        return None
    by_candidate = ablation.get("by_candidate") if isinstance(ablation.get("by_candidate"), dict) else {}
    baseline = by_candidate.get("candidate_00_policy_only", {})
    candidate_01 = by_candidate.get("candidate_01", {})
    baseline_pc = baseline.get("pc_success")
    candidate_01_pc = candidate_01.get("pc_success")
    return {
        "tasks": ablation.get("tasks"),
        "candidate_00_pc_success": baseline_pc,
        "candidate_01_pc_success": candidate_01_pc,
        "candidate_01_delta_vs_candidate_00_pp": (
            None
            if baseline_pc is None or candidate_01_pc is None
            else float(candidate_01_pc) - float(baseline_pc)
        ),
        "summary": "candidate_01_better"
        if baseline_pc is not None and candidate_01_pc is not None and float(candidate_01_pc) > float(baseline_pc)
        else "candidate_01_not_better_or_unknown",
    }


def render_markdown(package: dict[str, Any]) -> str:
    baseline = package["baseline"]
    risk1b = package["risk1b"]
    lines = [
        "# Risk1-B LIBERO-10 Result Package",
        "",
        "## Paper Table",
        "",
        "| Method | Success | Episodes | Delta vs SmolVLA | Notes |",
        "|---|---:|---:|---:|---|",
        f"| {baseline['method']} | {baseline['pc_success']:.1f}% | {baseline['episodes']} | 0.0pp | Internal baseline |",
        "| {method} | {pc} | {episodes} | {delta} | {lane} |".format(
            method=risk1b["method"],
            pc="NA" if risk1b["pc_success"] is None else f"{risk1b['pc_success']:.1f}%",
            episodes=risk1b["episodes"],
            delta="NA" if risk1b["delta_vs_baseline_pp"] is None else f"{risk1b['delta_vs_baseline_pp']:+.1f}pp",
            lane=risk1b.get("lane") or "unknown lane",
        ),
        "",
        f"Verdict: `{package['verdict']}`",
        f"Paper-table-ready: `{str(package['paper_table_ready']).lower()}`",
        "",
        "## Weak Rows",
        "",
        "| Task | Success | Episodes | Success rate | Notes |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in package["weak_rows"]:
        lines.append(
            "| {task} | {success} | {episodes} | {pc} | {status} |".format(
                task=row.get("task_id"),
                success=row.get("success_count", "NA"),
                episodes=row.get("n_episodes", "NA"),
                pc="NA" if row.get("pc_success") is None else f"{float(row['pc_success']):.1f}%",
                status=row.get("status", ""),
            )
        )
    if package.get("ablation"):
        ablation = package["ablation"]
        lines.extend(
            [
                "",
                "## Candidate Ablation",
                "",
                f"- tasks: `{ablation.get('tasks')}`",
                f"- candidate_00 success: `{ablation.get('candidate_00_pc_success')}`",
                f"- candidate_01 success: `{ablation.get('candidate_01_pc_success')}`",
                f"- candidate_01 delta vs candidate_00: `{ablation.get('candidate_01_delta_vs_candidate_00_pp')}`",
                f"- summary: `{ablation.get('summary')}`",
            ]
        )
    lines.extend(["", "## Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in package["allowed_claims"])
    lines.extend(["", "## Disallowed Claims", ""])
    lines.extend(f"- {claim}" for claim in package["disallowed_claims"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = load_json(args.summary)
    ablation = load_json(args.ablation_summary) if args.ablation_summary else None
    package = build_package(
        summary=summary,
        baseline_pc_success=args.baseline_pc_success,
        ablation_summary=ablation,
    )
    output_dir = args.output_dir or args.summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "risk1b_libero10_result_package.json"
    md_path = output_dir / "risk1b_libero10_result_package.md"
    json_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(package), encoding="utf-8")
    if args.json:
        print(json.dumps(package, indent=2, sort_keys=True))
    else:
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
