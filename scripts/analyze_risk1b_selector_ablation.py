#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CANDIDATES = [
    "candidate_00_policy_only",
    "candidate_01",
    "candidate_02",
    "candidate_03",
    "candidate_04",
    "candidate_05",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze downloaded Risk1-B weak-task candidate ablation artifacts. "
            "This is an offline preflight for selector work; it does not rerun SmolVLA."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="candidate ablation root")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = analyze_ablation(args.root)
    json_path = args.root / "selector_ablation_analysis.json"
    md_path = args.root / "selector_ablation_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps({"report": str(json_path), "markdown": str(md_path)}, indent=2, sort_keys=True))
    else:
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    return 0


def analyze_ablation(root: Path) -> dict[str, Any]:
    rows = []
    by_task: dict[int, list[dict[str, Any]]] = {}
    for task in discover_tasks(root):
        task_rows = []
        for candidate_id in CANDIDATES:
            row = read_candidate_task_row(root, candidate_id, task)
            if row is None:
                continue
            rows.append(row)
            task_rows.append(row)
        if task_rows:
            by_task[task] = task_rows

    task_summaries = []
    for task, task_rows in sorted(by_task.items()):
        baseline = next((row for row in task_rows if row["candidate_id"] == "candidate_00_policy_only"), None)
        best = max(task_rows, key=lambda row: (row["success_count"], -row["candidate_order"]))
        proxy_rows = [row for row in task_rows if row.get("old_progress_proxy_score") is not None]
        old_proxy_pick = max(
            proxy_rows,
            key=lambda row: (float(row["old_progress_proxy_score"]), row["candidate_id"]),
        ) if proxy_rows else None
        task_summaries.append(
            {
                "task_id": task,
                "baseline_success_count": baseline["success_count"] if baseline else None,
                "best_candidate_id": best["candidate_id"],
                "best_success_count": best["success_count"],
                "best_delta_vs_baseline": (
                    best["success_count"] - baseline["success_count"] if baseline else None
                ),
                "old_proxy_pick": old_proxy_pick["candidate_id"] if old_proxy_pick else None,
                "old_proxy_pick_success_count": old_proxy_pick["success_count"] if old_proxy_pick else None,
                "old_proxy_matches_best": bool(old_proxy_pick and old_proxy_pick["candidate_id"] == best["candidate_id"]),
                "pairwise_focus": build_pairwise_focus_rows(baseline, task_rows),
                "rows": task_rows,
            }
        )

    baseline_total = sum(row["success_count"] for row in rows if row["candidate_id"] == "candidate_00_policy_only")
    baseline_n = sum(row["n_episodes"] for row in rows if row["candidate_id"] == "candidate_00_policy_only")
    oracle_total = sum(summary["best_success_count"] for summary in task_summaries)
    oracle_n = sum(
        next((row["n_episodes"] for row in summary["rows"] if row["candidate_id"] == "candidate_00_policy_only"), 0)
        for summary in task_summaries
    )
    old_proxy_total = sum(summary["old_proxy_pick_success_count"] or 0 for summary in task_summaries)
    old_proxy_n = oracle_n
    return {
        "root": str(root),
        "status": "completed" if task_summaries else "missing_rows",
        "task_count": len(task_summaries),
        "candidate_count": len(CANDIDATES),
        "baseline": percent_payload(baseline_total, baseline_n),
        "oracle_best_per_task": percent_payload(oracle_total, oracle_n),
        "old_first_action_proxy_pick": percent_payload(old_proxy_total, old_proxy_n),
        "old_proxy_match_count": sum(1 for summary in task_summaries if summary["old_proxy_matches_best"]),
        "pairwise_focus_summary": summarize_pairwise_focus(task_summaries),
        "task_summaries": task_summaries,
        "preflight_limitations": [
            "Downloaded artifacts do not include raw candidate action chunks, so the patched chunk-level selector cannot be exactly replayed from this run.",
            "Use this report to validate failure modes and oracle upper bound before spending RunPod time.",
        ],
    }


def discover_tasks(root: Path) -> list[int]:
    tasks = set()
    for path in root.glob("candidate_*/*/eval_logs/eval_info.json"):
        name = path.parents[1].name
        if name.startswith("libero_10_task") and "_seed" in name:
            task_text = name.removeprefix("libero_10_task").split("_seed", 1)[0]
            try:
                tasks.add(int(task_text))
            except ValueError:
                continue
    return sorted(tasks)


def read_candidate_task_row(root: Path, candidate_id: str, task: int) -> dict[str, Any] | None:
    eval_path = root / candidate_id / f"libero_10_task{task}_seed1201" / "eval_logs" / "eval_info.json"
    if not eval_path.exists():
        return None
    successes = read_successes(eval_path, task)
    proxy_score, proxy_details = read_old_proxy(root, candidate_id, task)
    return {
        "candidate_id": candidate_id,
        "candidate_order": CANDIDATES.index(candidate_id) if candidate_id in CANDIDATES else 999,
        "task_id": task,
        "success_count": sum(successes),
        "n_episodes": len(successes),
        "pc_success": 100.0 * sum(successes) / len(successes) if successes else 0.0,
        "successes": successes,
        "old_progress_proxy_score": proxy_score,
        "old_progress_proxy_details": proxy_details,
        "eval_info_path": str(eval_path),
    }


def build_pairwise_focus_rows(
    baseline: dict[str, Any] | None,
    task_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    baseline_successes = list(baseline.get("successes") or [])
    focus_rows = []
    for row in task_rows:
        candidate_id = row["candidate_id"]
        if candidate_id == "candidate_00_policy_only":
            continue
        candidate_successes = list(row.get("successes") or [])
        n = min(len(baseline_successes), len(candidate_successes))
        both_success = candidate_only = baseline_only = both_fail = 0
        candidate_only_episodes: list[int] = []
        baseline_only_episodes: list[int] = []
        both_fail_episodes: list[int] = []
        for index in range(n):
            baseline_success = bool(baseline_successes[index])
            candidate_success = bool(candidate_successes[index])
            if baseline_success and candidate_success:
                both_success += 1
            elif candidate_success:
                candidate_only += 1
                candidate_only_episodes.append(index)
            elif baseline_success:
                baseline_only += 1
                baseline_only_episodes.append(index)
            else:
                both_fail += 1
                both_fail_episodes.append(index)
        informative = candidate_only + baseline_only + both_fail
        focus_rows.append(
            {
                "candidate_id": candidate_id,
                "episode_count": n,
                "both_success_excluded": both_success,
                "informative_episode_count": informative,
                "candidate_only_success": candidate_only,
                "baseline_only_success": baseline_only,
                "both_fail": both_fail,
                "candidate_advantage_on_informative": candidate_only - baseline_only,
                "candidate_only_episodes": candidate_only_episodes,
                "baseline_only_episodes": baseline_only_episodes,
                "both_fail_episodes": both_fail_episodes,
            }
        )
    return focus_rows


def summarize_pairwise_focus(task_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, dict[str, int]] = {}
    for summary in task_summaries:
        for row in summary.get("pairwise_focus", []):
            target = by_candidate.setdefault(
                row["candidate_id"],
                {
                    "informative_episode_count": 0,
                    "candidate_only_success": 0,
                    "baseline_only_success": 0,
                    "both_fail": 0,
                    "both_success_excluded": 0,
                    "candidate_advantage_on_informative": 0,
                },
            )
            for key in target:
                target[key] += int(row.get(key, 0))
    return by_candidate


def read_successes(eval_path: Path, task: int) -> list[bool]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    for item in payload.get("per_task", []):
        if item.get("task_group") == "libero_10" and item.get("task_id") == task:
            return [bool(value) for value in item.get("metrics", {}).get("successes", [])]
    return []


def read_old_proxy(root: Path, candidate_id: str, task: int) -> tuple[float | None, dict[str, Any] | None]:
    trace_path = root / candidate_id / f"libero_10_task{task}_seed1201" / "benchmark_trace.jsonl"
    if not trace_path.exists():
        return None, None
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        ita = record.get("ita")
        if not isinstance(ita, dict):
            continue
        for candidate in ita.get("candidates", []):
            if candidate.get("candidate_id") == candidate_id:
                return candidate.get("progress_proxy_score"), candidate.get("progress_proxy_details")
    return None, None


def percent_payload(success_count: int, n_episodes: int) -> dict[str, Any]:
    return {
        "success_count": success_count,
        "n_episodes": n_episodes,
        "pc_success": 100.0 * success_count / n_episodes if n_episodes else 0.0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Risk1-B Selector Ablation Analysis",
        "",
        f"Root: `{report['root']}`",
        "",
        "| Selector | Success |",
        "|---|---:|",
        f"| Baseline candidate_00 | {format_percent(report['baseline'])} |",
        f"| Old first-action proxy pick | {format_percent(report['old_first_action_proxy_pick'])} |",
        f"| Oracle best per task | {format_percent(report['oracle_best_per_task'])} |",
        "",
        f"Old proxy matched the best candidate on {report['old_proxy_match_count']}/{report['task_count']} tasks.",
        "",
        "| Task | Baseline | Old proxy pick | Best candidate |",
        "|---:|---:|---:|---:|",
    ]
    for summary in report["task_summaries"]:
        lines.append(
            "| "
            f"{summary['task_id']} | "
            f"{summary['baseline_success_count']}/10 | "
            f"{summary['old_proxy_pick']} {summary['old_proxy_pick_success_count']}/10 | "
            f"{summary['best_candidate_id']} {summary['best_success_count']}/10 |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Focus Episodes",
            "",
            "Both-success episodes are excluded because they do not help distinguish selector choices.",
            "",
            "| Candidate | Informative | Candidate-only wins | Baseline-only wins | Both fail | Net advantage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id, row in sorted(report["pairwise_focus_summary"].items()):
        lines.append(
            "| "
            f"{candidate_id} | "
            f"{row['informative_episode_count']} | "
            f"{row['candidate_only_success']} | "
            f"{row['baseline_only_success']} | "
            f"{row['both_fail']} | "
            f"{row['candidate_advantage_on_informative']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Raw candidate action chunks were not saved in the downloaded run, so patched chunk-level selector replay is not exact.",
            "- Next run should rely on `candidate_score_table` traces added by the selector patch.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_percent(payload: dict[str, Any]) -> str:
    return f"{payload['success_count']}/{payload['n_episodes']} = {payload['pc_success']:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
