from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Subgoal:
    index: int
    name: str
    instruction: str
    success_metric: str
    threshold: float
    max_steps: int
    retry_budget: int = 1


@dataclass(frozen=True)
class Plan:
    task: str
    env_id: str
    subgoals: list[Subgoal]


class RuleBasedSO101Planner:
    """Small deterministic planner for the first SO101 agentic wrapper."""

    def plan(self, task: str, env_id: str) -> Plan:
        normalized = task.strip() or "reach_target"
        subgoals = [
            Subgoal(
                index=0,
                name="stabilize_arm",
                instruction=f"{normalized}: settle the arm and observe the target.",
                success_metric="tcp_to_target_dist",
                threshold=0.20,
                max_steps=2,
            ),
            Subgoal(
                index=1,
                name="approach_target",
                instruction=f"{normalized}: move the wrist toward the target.",
                success_metric="tcp_to_target_dist",
                threshold=0.16,
                max_steps=3,
            ),
            Subgoal(
                index=2,
                name="finish_reach",
                instruction=f"{normalized}: finish close to the target.",
                success_metric="tcp_to_target_dist",
                threshold=0.12,
                max_steps=3,
            ),
        ]
        return Plan(task=normalized, env_id=env_id, subgoals=subgoals)


def write_plan(plan: Plan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True), encoding="utf-8")
