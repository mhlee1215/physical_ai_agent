#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class ConditionRow:
    condition: str
    root: str
    success: bool
    pc_success: float
    action_step_count: int
    verifier_trigger_count: int
    intervention_count: int
    environment_resets: int
    eval_seconds: float

    @property
    def success_per_action_step(self) -> float:
        return _ratio(1 if self.success else 0, self.action_step_count)

    @property
    def success_per_eval_minute(self) -> float:
        return _ratio(1 if self.success else 0, self.eval_seconds / 60.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny LIBERO in-episode intervention ablation report.")
    parser.add_argument("--condition", action="append", default=[], help="NAME=ROOT")
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = [load_condition(item) for item in args.condition]
    if not rows:
        raise ValueError("At least one --condition NAME=ROOT is required")
    payload = {
        "rows": [serialize_row(row) for row in rows],
        "summary": summarize(rows),
    }
    args.output_md.write_text(render_markdown(rows, payload), encoding="utf-8")
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={args.output_md}")
    print(f"summary={args.output_json}")


def load_condition(spec: str) -> ConditionRow:
    if "=" not in spec:
        raise ValueError(f"Condition must be NAME=ROOT, got {spec!r}")
    name, root_text = spec.split("=", 1)
    root = Path(root_text)
    eval_info = json.loads((root / "eval_logs" / "eval_info.json").read_text(encoding="utf-8"))
    summary = load_rollout_summary(root / "in_episode_trace.jsonl")
    overall = eval_info.get("overall", {})
    pc_success = float(overall.get("pc_success", 0.0))
    return ConditionRow(
        condition=name,
        root=str(root),
        success=bool(summary.get("success", pc_success > 0.0)),
        pc_success=pc_success,
        action_step_count=int(summary["action_step_count"]),
        verifier_trigger_count=int(summary["verifier_trigger_count"]),
        intervention_count=int(summary["intervention_count"]),
        environment_resets=1,
        eval_seconds=float(overall.get("eval_s", 0.0)),
    )


def load_rollout_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("event") == "rollout_summary":
                return item
    raise ValueError(f"No rollout_summary found in {path}")


def summarize(rows: list[ConditionRow]) -> dict[str, float]:
    return {
        "conditions": float(len(rows)),
        "success_rate": 100.0 * mean(1 if row.success else 0 for row in rows),
        "mean_action_step_count": mean(row.action_step_count for row in rows),
        "mean_eval_seconds": mean(row.eval_seconds for row in rows),
        "mean_success_per_action_step": mean(row.success_per_action_step for row in rows),
        "mean_success_per_eval_minute": mean(row.success_per_eval_minute for row in rows),
    }


def render_markdown(rows: list[ConditionRow], payload: dict[str, Any]) -> str:
    lines = [
        "# LIBERO In-Episode Intervention Ablation Report",
        "",
        "## Conditions",
        "",
        "| Condition | Success | PC success | Action steps | Triggers | Interventions | Resets | Eval seconds | Success/action step | Success/eval min |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.condition,
                    str(row.success).lower(),
                    f"{row.pc_success:.2f}",
                    str(row.action_step_count),
                    str(row.verifier_trigger_count),
                    str(row.intervention_count),
                    str(row.environment_resets),
                    f"{row.eval_seconds:.4f}",
                    f"{row.success_per_action_step:.6f}",
                    f"{row.success_per_eval_minute:.6f}",
                ]
            )
            + " |"
        )
    summary = payload["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Conditions | Success rate | Mean action steps | Mean eval seconds | Mean success/action step | Mean success/eval min |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {int(summary['conditions'])} | {summary['success_rate']:.2f} | "
                f"{summary['mean_action_step_count']:.2f} | {summary['mean_eval_seconds']:.4f} | "
                f"{summary['mean_success_per_action_step']:.6f} | {summary['mean_success_per_eval_minute']:.6f} |"
            ),
            "",
            "## Claim Boundary",
            "",
            "- This is a tiny same-task smoke ablation, not a paper-scale benchmark.",
            "- A non-trivial intervention is useful only if it improves success, action-step cost, or eval-time cost against the no-op hook under fixed task/seed budget.",
            "",
        ]
    )
    return "\n".join(lines)


def serialize_row(row: ConditionRow) -> dict[str, Any]:
    return row.__dict__ | {
        "success_per_action_step": row.success_per_action_step,
        "success_per_eval_minute": row.success_per_eval_minute,
    }


def _ratio(numerator: int, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


if __name__ == "__main__":
    main()
