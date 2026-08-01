#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PENETRATION_GATE = "max_pad_cube_penetration_exceeded"


def audit_reports(
    report_specs: list[tuple[str, Path]],
    *,
    threshold_m: float = 0.003,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    all_episodes: list[dict[str, Any]] = []
    for label, report_path in report_specs:
        if label in rows:
            raise ValueError(f"duplicate report label: {label}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audited = []
        for episode in report.get("episode_summaries", []):
            failed_gates = [str(item) for item in episode.get("failed_gates", [])]
            if PENETRATION_GATE not in failed_gates:
                continue
            trace_path = Path(str(episode["trace_path"]))
            trace = audit_trace(trace_path, threshold_m=threshold_m)
            reported_peak = float(episode["max_pad_cube_penetration_m"])
            if abs(reported_peak - trace["peak_penetration_m"]) > 1e-9:
                raise ValueError(
                    f"trace/report penetration mismatch for {label} seed={episode.get('seed')}"
                )
            item = {
                "label": label,
                "episode": int(episode["episode"]),
                "seed": int(episode["seed"]),
                "penetration_only_failure": set(failed_gates) == {PENETRATION_GATE},
                "failed_gates": failed_gates,
                "trace_path": str(trace_path),
                **trace,
            }
            audited.append(item)
            all_episodes.append(item)
        summaries = report.get("episode_summaries", [])
        rows[label] = _summarize_row(
            audited,
            episodes=len(summaries),
            strict_successes=sum(bool(item.get("success")) for item in summaries),
        )

    return {
        "operation": "audit_mycobot280_penetration_failures",
        "status": "passed",
        "penetration_threshold_m": threshold_m,
        "report_count": len(rows),
        "rows": rows,
        "aggregate": _summarize_row(
            all_episodes,
            episodes=sum(int(row["episodes"]) for row in rows.values()),
            strict_successes=sum(int(row["strict_successes"]) for row in rows.values()),
        ),
        "scope": {
            "included_contact_pair": (
                "myCobot adaptive-gripper pad geoms against cube geom"
            ),
            "excluded_contact_pairs": [
                "cube-table or cube-mat",
                "robot self-collision",
                "non-pad gripper geoms against cube",
            ],
            "available_trace_evidence": [
                "contact side",
                "rollout phase",
                "threshold-crossing steps",
                "longest consecutive crossing",
                "peak penetration",
            ],
            "not_logged": ["contact force", "contact impulse"],
        },
        "claim_boundary": (
            "A penetration-only failure is functional pickup/lift/hold success that failed "
            "the conservative pad-cube contact-quality gate. This audit characterizes saved "
            "policy-only traces; it does not validate a recovery action or agentic improvement."
        ),
    }


def audit_trace(trace_path: Path, *, threshold_m: float = 0.003) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = []
    for record in records:
        pickup = record.get("ground_pickup") or {}
        depth = pickup.get("pad_cube_contact_depth") or {}
        checks = list(depth.get("checks") or [])
        samples.append(
            {
                "step": int(record.get("step", pickup.get("step", -1))),
                "phase": str(record.get("phase", pickup.get("phase", "unknown"))),
                "penetration_m": float(depth.get("max_penetration_m", 0.0)),
                "checks": checks,
            }
        )
    if not samples:
        raise ValueError(f"trace has no records: {trace_path}")

    peak = max(samples, key=lambda item: item["penetration_m"])
    crossing = [item for item in samples if item["penetration_m"] > threshold_m]
    phase_counts = Counter(item["phase"] for item in crossing)
    peak_depth = float(peak["penetration_m"])
    peak_sides = sorted(
        {
            str(check.get("side", "unknown"))
            for check in peak["checks"]
            if abs(float(check.get("penetration_m", 0.0)) - peak_depth) <= 1e-12
        }
    )
    runs = _consecutive_runs([int(item["step"]) for item in crossing])
    return {
        "trace_steps": len(samples),
        "peak_penetration_m": peak_depth,
        "peak_excess_over_gate_m": max(0.0, peak_depth - threshold_m),
        "peak_step": int(peak["step"]),
        "peak_phase": str(peak["phase"]),
        "peak_sides": peak_sides,
        "above_gate_steps": len(crossing),
        "first_above_gate_step": int(crossing[0]["step"]) if crossing else None,
        "last_above_gate_step": int(crossing[-1]["step"]) if crossing else None,
        "longest_above_gate_run_steps": max((len(run) for run in runs), default=0),
        "above_gate_phase_counts": dict(sorted(phase_counts.items())),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# myCobot 280 Penetration-Failure Audit",
        "",
        (
            "Strict pad-cube penetration threshold: "
            f"**{audit['penetration_threshold_m'] * 1000:.1f} mm**."
        ),
        "",
        (
            "| Evaluation | Episodes | Strict | Penetration gate failures | "
            "Penetration-only | Mean peak (mm) | Median steps over gate | "
            "Longest run | Peak side |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        ),
    ]
    for label, row in audit["rows"].items():
        lines.append(
            f"| {label} | {row['episodes']} | {row['strict_successes']} | "
            f"{row['penetration_gate_failures']} | {row['penetration_only_failures']} | "
            f"{row['mean_peak_penetration_mm']:.3f} | "
            f"{row['median_above_gate_steps']:.1f} | "
            f"{row['max_longest_above_gate_run_steps']} | "
            f"{_format_counts(row['peak_side_counts'])} |"
        )
    scope = audit["scope"]
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- The gate covers adaptive-gripper **pad-cube** contacts only.",
            (
                "- Cube-table/mat penetration and robot self-collision are not being "
                "mislabeled by this metric."
            ),
            (
                "- Penetration-only means pickup/lift/hold passed, but strict contact "
                "quality did not."
            ),
            (
                "- Saved traces identify side, phase, magnitude, and duration in "
                "simulation steps."
            ),
            (
                f"- They do not log {', '.join(scope['not_logged'])}; those need evaluator "
                "instrumentation before force-based diagnosis."
            ),
            "",
            f"> {audit['claim_boundary']}",
            "",
        ]
    )
    return "\n".join(lines)


def _summarize_row(
    episodes_with_penetration: list[dict[str, Any]],
    *,
    episodes: int,
    strict_successes: int,
) -> dict[str, Any]:
    side_counts = Counter(
        side
        for item in episodes_with_penetration
        for side in (item.get("peak_sides") or ["unknown"])
    )
    return {
        "episodes": episodes,
        "strict_successes": strict_successes,
        "penetration_gate_failures": len(episodes_with_penetration),
        "penetration_only_failures": sum(
            bool(item.get("penetration_only_failure"))
            for item in episodes_with_penetration
        ),
        "mean_peak_penetration_mm": _mean(
            [
                item["peak_penetration_m"] * 1000
                for item in episodes_with_penetration
            ]
        ),
        "mean_peak_excess_over_gate_mm": _mean(
            [
                item["peak_excess_over_gate_m"] * 1000
                for item in episodes_with_penetration
            ]
        ),
        "median_above_gate_steps": _median(
            [item["above_gate_steps"] for item in episodes_with_penetration]
        ),
        "max_longest_above_gate_run_steps": max(
            (
                item["longest_above_gate_run_steps"]
                for item in episodes_with_penetration
            ),
            default=0,
        ),
        "peak_side_counts": dict(sorted(side_counts.items())),
        "peak_phase_counts": dict(
            sorted(
                Counter(
                    item["peak_phase"] for item in episodes_with_penetration
                ).items()
            )
        ),
        "audited_episodes": episodes_with_penetration,
    }


def _consecutive_runs(steps: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for step in steps:
        if not runs or step != runs[-1][-1] + 1:
            runs.append([step])
        else:
            runs[-1].append(step)
    return runs


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _format_counts(values: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items()) or "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit myCobot 280 pad-cube penetration failures from saved traces."
    )
    parser.add_argument(
        "--report",
        nargs=2,
        action="append",
        metavar=("LABEL", "EVAL_REPORT"),
        required=True,
    )
    parser.add_argument("--threshold-mm", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = audit_reports(
        [(label, Path(path)) for label, path in args.report],
        threshold_m=args.threshold_mm / 1000.0,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
