#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from physical_ai_agent.agent_core.libero_in_episode import (
    ScaleActionIntervention,
    StagnationVerifier,
    run_in_episode_rollout,
    write_result,
)


class ToyStagnationEnv:
    """Small deterministic stand-in for a LIBERO step loop."""

    def __init__(self) -> None:
        self.step_index = 0
        self.progress = 0.0
        self.last_action = 0.0
        self.recovered = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, float], dict[str, Any]]:
        self.step_index = 0
        self.progress = 0.0
        self.last_action = 0.0
        self.recovered = False
        return {"progress": self.progress}, {"seed": seed}

    def step(self, action: Any) -> tuple[dict[str, float], float, bool, bool, dict[str, Any]]:
        self.step_index += 1
        self.last_action = float(action)
        if abs(self.last_action) <= 0.2:
            self.recovered = True
        if self.step_index <= 4:
            # Force a stagnation window so the verifier has something real to catch.
            self.progress = 0.0
        elif self.recovered:
            self.progress += 0.3
        else:
            self.progress += max(0.0, 0.3 - abs(self.last_action))
        success = self.progress >= 0.5
        terminated = success or self.step_index >= 8
        reward = self.progress
        return (
            {"progress": self.progress},
            reward,
            terminated,
            False,
            {
                "is_success": success,
                "progress": self.progress,
                "step_index": self.step_index,
                "last_action": self.last_action,
            },
        )


class ConstantPolicy:
    def __init__(self, action: float = 0.6) -> None:
        self.action = action

    def reset(self) -> None:
        return None

    def select_action(self, observation: Any) -> float:
        return self.action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-dependency in-episode instrumentation smoke test.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/libero_in_episode_smoke"))
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_in_episode_rollout(
        env=ToyStagnationEnv(),
        policy=ConstantPolicy(action=0.6),
        verifier=StagnationVerifier(metric_name="progress", window=3, min_delta=1e-6),
        intervention=ScaleActionIntervention(scale=0.25, intervention_type="scale_next_action"),
        output_jsonl=args.output_dir / "in_episode_trace.jsonl",
        task_group="toy_libero_goal",
        task_id=0,
        seed=args.seed,
        max_steps=8,
        max_interventions=1,
    )
    write_result(
        result,
        output_json=args.output_dir / "in_episode_metrics.json",
        output_md=args.output_dir / "in_episode_report.md",
    )
    print(f"metrics={args.output_dir / 'in_episode_metrics.json'}")
    print(f"report={args.output_dir / 'in_episode_report.md'}")
    print(f"trace={args.output_dir / 'in_episode_trace.jsonl'}")


if __name__ == "__main__":
    main()
