#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_candidate_memory(*, reports: list[Path], output: Path, markdown: Path | None = None) -> dict[str, Any]:
    candidates = []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidates.extend(_extract_candidates(report=report, report_path=report_path))
    ranked = sorted(candidates, key=_candidate_key)
    best = ranked[0] if ranked else None
    latest = _latest_report_candidates(candidates, reports[-1] if reports else None)
    latest_best = sorted(latest, key=_candidate_key)[0] if latest else None
    memory = {
        "operation": "real_so100_agentic_candidate_memory",
        "status": "passed" if candidates else "blocked",
        "source_reports": [str(path) for path in reports],
        "policy_camera_indexes": _first_non_null([candidate.get("policy_camera_indexes") for candidate in candidates]),
        "observer_camera_indexes": _first_non_null([candidate.get("observer_camera_indexes") for candidate in candidates]) or [],
        "observer_camera_status": _first_non_null([candidate.get("observer_camera_status") for candidate in candidates]) or "unknown",
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "candidates": candidates,
        "ranked_candidates": ranked,
        "best_candidate": best,
        "latest_best_candidate": latest_best,
        "regression_from_best": _regression(best, latest_best),
        "next_agentic_layer_step": _next_step(best, latest_best),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(memory, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(memory), encoding="utf-8")
    memory["json_path"] = str(output)
    memory["markdown_path"] = str(md_path)
    return memory


def _extract_candidates(*, report: dict[str, Any], report_path: Path) -> list[dict[str, Any]]:
    result = []
    for candidate in report.get("ranked_candidates") or report.get("candidates") or []:
        score = _score(candidate)
        if score is None:
            continue
        result.append(
            {
                "source_report": str(report_path),
                "source_operation": report.get("operation"),
                "source_prompt_profile": report.get("prompt_profile"),
                "source_projection": report.get("operation") == "real_so100_projection_analysis",
                "candidate_index": candidate.get("candidate_index"),
                "prompt": candidate.get("prompt"),
                "action_path": candidate.get("action_path"),
                "execute_gate_path": candidate.get("execute_gate_path"),
                "policy_camera_indexes": report.get("policy_camera_indexes"),
                "observer_camera_indexes": report.get("observer_camera_indexes", []),
                "observer_camera_status": report.get("observer_camera_status"),
                "send_action_called": False,
                "policy_actions_executed": False,
                "physical_robot_motion": False,
                "task_success_claim_allowed": False,
                "score": score,
            }
        )
    return result


def _score(candidate: dict[str, Any]) -> dict[str, Any] | None:
    raw = candidate.get("score") or candidate.get("projection")
    if not raw:
        return None
    penalty = raw.get("range_penalty_score", raw.get("projection_penalty_score"))
    violations = raw.get("range_violation_count")
    total = raw.get("total_range_excess_raw_ticks", raw.get("total_raw_distortion"))
    max_excess = raw.get("max_range_excess_raw_ticks", raw.get("max_raw_distortion"))
    if penalty is None or violations is None:
        return None
    return {
        "ready_for_execution": bool(raw.get("ready_for_execution", False)) and int(violations) == 0,
        "penalty_score": round(float(penalty), 4),
        "range_violation_count": int(violations),
        "total_excess_raw_ticks": round(float(total or 0.0), 4),
        "max_excess_raw_ticks": round(float(max_excess or 0.0), 4),
        "joint_excess_raw_ticks": raw.get("violation_joint_excess_raw_ticks") or _projection_joint_excess(raw),
        "joint_violation_counts": raw.get("violation_joint_counts") or _projection_joint_counts(raw),
    }


def render_markdown(memory: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Agentic Candidate Memory",
        "",
        f"- Status: `{memory['status']}`",
        f"- Observer cameras: `{memory['observer_camera_indexes']}` (`{memory['observer_camera_status']}`)",
        f"- Actuation enabled: `{memory['actuation_enabled']}`",
        f"- Physical robot motion: `{memory['physical_robot_motion']}`",
        f"- Task success claim allowed: `{memory['task_success_claim_allowed']}`",
        "",
        "## Best So Far",
        "",
    ]
    best = memory.get("best_candidate")
    if best:
        score = best["score"]
        lines.extend(
            [
                f"- Source: `{best['source_report']}`",
                f"- Candidate: `{int(best['candidate_index']):02d}`",
                f"- Prompt: {best['prompt']}",
                f"- Penalty score: `{score['penalty_score']}`",
                f"- Range violations: `{score['range_violation_count']}`",
                f"- Max excess raw ticks: `{score['max_excess_raw_ticks']}`",
                "",
            ]
        )
    latest = memory.get("latest_best_candidate")
    if latest:
        score = latest["score"]
        lines.extend(
            [
                "## Latest Best",
                "",
                f"- Source: `{latest['source_report']}`",
                f"- Candidate: `{int(latest['candidate_index']):02d}`",
                f"- Prompt: {latest['prompt']}",
                f"- Penalty score: `{score['penalty_score']}`",
                f"- Range violations: `{score['range_violation_count']}`",
                f"- Regression from best: `{memory['regression_from_best']['is_regression']}`",
                "",
            ]
        )
    lines.extend(["## Ranking", ""])
    for candidate in memory.get("ranked_candidates") or []:
        score = candidate["score"]
        lines.append(
            f"- `{score['penalty_score']}` / violations `{score['range_violation_count']}`: "
            f"{candidate['prompt']} (`{candidate['source_report']}` candidate `{int(candidate['candidate_index']):02d}`)"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            f"- Type: `{memory['next_agentic_layer_step']['type']}`",
            f"- Reason: {memory['next_agentic_layer_step']['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_key(candidate: dict[str, Any]) -> tuple[int, float, int, float, int, str]:
    score = candidate["score"]
    return (
        0 if score["ready_for_execution"] else 1,
        float(score["penalty_score"]),
        int(score["range_violation_count"]),
        float(score["max_excess_raw_ticks"]),
        int(candidate.get("candidate_index") or 0),
        str(candidate.get("source_report")),
    )


def _latest_report_candidates(candidates: list[dict[str, Any]], latest_report: Path | None) -> list[dict[str, Any]]:
    if latest_report is None:
        return []
    latest = str(latest_report)
    return [candidate for candidate in candidates if candidate.get("source_report") == latest]


def _regression(best: dict[str, Any] | None, latest: dict[str, Any] | None) -> dict[str, Any]:
    if not best or not latest:
        return {"is_regression": False, "penalty_delta": None}
    delta = float(latest["score"]["penalty_score"]) - float(best["score"]["penalty_score"])
    return {
        "is_regression": delta > 0,
        "penalty_delta": round(delta, 4),
        "best_source_report": best["source_report"],
        "latest_source_report": latest["source_report"],
    }


def _next_step(best: dict[str, Any] | None, latest: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {"type": "run_proposal_sweep", "reason": "No comparable candidates are available."}
    if best["score"]["ready_for_execution"]:
        return {
            "type": "hold_for_observer_camera_3_and_execution_gates",
            "reason": "A no-actuation candidate is range-ready; physical execution still requires observer evidence and confirmation.",
            "selected_prompt": best["prompt"],
        }
    regression = _regression(best, latest)
    if regression["is_regression"]:
        return {
            "type": "reuse_best_historical_prompt_family",
            "reason": "The latest attempt regressed against the best historical no-actuation candidate; continue from the best-so-far prompt family instead of the latest profile.",
            "selected_prompt": best["prompt"],
            "selected_source_report": best["source_report"],
        }
    return {
        "type": "continue_from_latest_best_prompt_family",
        "reason": "The latest attempt matches the best available no-actuation candidate; use it as the next feedback source.",
        "selected_prompt": best["prompt"],
        "selected_source_report": best["source_report"],
    }


def _projection_joint_excess(raw: dict[str, Any]) -> dict[str, float]:
    return {
        str(joint): round(float(distortion.get("total_raw_distortion", 0.0) or 0.0), 4)
        for joint, distortion in (raw.get("joint_distortion") or {}).items()
        if float(distortion.get("total_raw_distortion", 0.0) or 0.0) > 0
    }


def _projection_joint_counts(raw: dict[str, Any]) -> dict[str, int]:
    return {
        str(joint): int(distortion.get("violation_count", 0) or 0)
        for joint, distortion in (raw.get("joint_distortion") or {}).items()
        if int(distortion.get("violation_count", 0) or 0) > 0
    }


def _first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build best-so-far candidate memory for real SO-100 agentic SmolVLA sweeps.")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_candidate_memory(reports=args.report, output=args.output, markdown=args.markdown), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
