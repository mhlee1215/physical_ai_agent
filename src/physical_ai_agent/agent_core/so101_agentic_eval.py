from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from physical_ai_agent.agent_core.planner import Plan, RuleBasedSO101Planner, Subgoal, write_plan
from physical_ai_agent.agent_core.verifier import SO101SimulationStateVerifier, VerificationDecision
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv, sample_action


@dataclass(frozen=True)
class PolicyOnlyResult:
    mode: str
    env_id: str
    task: str
    steps: int
    success: bool
    final_distance: float
    total_reward: float
    trace_path: str


@dataclass(frozen=True)
class AgenticRetryResult:
    mode: str
    env_id: str
    task: str
    steps: int
    success: bool
    final_distance: float
    total_reward: float
    retry_events: int
    passed_subgoals: int
    total_subgoals: int
    plan_path: str
    trace_path: str


@dataclass(frozen=True)
class ComparisonReport:
    env_id: str
    task: str
    policy_only: PolicyOnlyResult
    agentic_retry: AgenticRetryResult
    markdown_path: str
    metrics_path: str


def run_policy_only(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    task: str = "reach_target",
    steps: int = 8,
    seed: int = 0,
) -> PolicyOnlyResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "policy_only_trace.jsonl"
    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    records = []
    try:
        for step in range(steps):
            action = sample_action(env.action_space, step / max(1, steps - 1))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_info = info
            records.append(
                {
                    "mode": "policy_only",
                    "step": step,
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": _json_safe_info(info),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    _write_jsonl(trace_path, records)
    return PolicyOnlyResult(
        mode="policy_only",
        env_id=env_id,
        task=task,
        steps=len(records),
        success=_info_success(final_info),
        final_distance=_info_distance(final_info),
        total_reward=total_reward,
        trace_path=str(trace_path),
    )


def run_agentic_retry(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    task: str = "reach_target",
    seed: int = 0,
) -> AgenticRetryResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    planner = RuleBasedSO101Planner()
    verifier = SO101SimulationStateVerifier()
    plan = planner.plan(task=task, env_id=env_id)
    plan_path = output_dir / "agentic_plan.json"
    trace_path = output_dir / "agentic_retry_trace.jsonl"
    write_plan(plan, plan_path)

    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    obs, _info = env.reset(seed=seed)
    step_index = 0
    total_reward = 0.0
    retry_events = 0
    passed_subgoals = 0
    final_info: dict[str, Any] = {}
    records = []
    try:
        for subgoal in plan.subgoals:
            decision: VerificationDecision | None = None
            for attempt in range(subgoal.retry_budget + 1):
                started_at = perf_counter()
                segment = _execute_subgoal(
                    env=env,
                    subgoal=subgoal,
                    observation=obs,
                    start_step=step_index,
                    attempt=attempt,
                )
                obs = segment["observation"]
                step_index = int(segment["next_step"])
                total_reward += float(segment["total_reward"])
                final_info = dict(segment["final_info"])
                decision = verifier.verify(subgoal, final_info)
                if attempt > 0:
                    retry_events += 1
                records.append(
                    {
                        "mode": "agentic_retry",
                        "subgoal": asdict(subgoal),
                        "attempt": attempt,
                        "retry": attempt > 0,
                        "segment_steps": segment["records"],
                        "verifier": asdict(decision),
                        "latency_s": perf_counter() - started_at,
                    }
                )
                if decision.passed:
                    passed_subgoals += 1
                    break
                if attempt >= subgoal.retry_budget:
                    break
    finally:
        env.close()
    _write_jsonl(trace_path, records)
    return AgenticRetryResult(
        mode="agentic_retry",
        env_id=env_id,
        task=task,
        steps=step_index,
        success=passed_subgoals == len(plan.subgoals),
        final_distance=_info_distance(final_info),
        total_reward=total_reward,
        retry_events=retry_events,
        passed_subgoals=passed_subgoals,
        total_subgoals=len(plan.subgoals),
        plan_path=str(plan_path),
        trace_path=str(trace_path),
    )


def run_policy_vs_agentic_comparison(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    task: str = "reach_target",
    policy_steps: int = 8,
    seed: int = 0,
) -> ComparisonReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_only = run_policy_only(
        output_dir=output_dir / "policy_only",
        env_id=env_id,
        task=task,
        steps=policy_steps,
        seed=seed,
    )
    agentic_retry = run_agentic_retry(
        output_dir=output_dir / "agentic_retry",
        env_id=env_id,
        task=task,
        seed=seed,
    )
    markdown_path = output_dir / "comparison_report.md"
    metrics_path = output_dir / "comparison_metrics.json"
    report = ComparisonReport(
        env_id=env_id,
        task=task,
        policy_only=policy_only,
        agentic_retry=agentic_retry,
        markdown_path=str(markdown_path),
        metrics_path=str(metrics_path),
    )
    metrics_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_comparison_markdown(report), encoding="utf-8")
    return report


def _execute_subgoal(
    env: SO101NexusEnv,
    subgoal: Subgoal,
    observation: list[float],
    start_step: int,
    attempt: int,
) -> dict[str, Any]:
    records = []
    obs = observation
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    for offset in range(subgoal.max_steps):
        step = start_step + offset
        action = _subgoal_action(env, subgoal, step, attempt)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_info = info
        records.append(
            {
                "step": step,
                "observation": obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": _json_safe_info(info),
            }
        )
        if terminated or truncated:
            break
    return {
        "observation": obs,
        "records": records,
        "next_step": start_step + len(records),
        "total_reward": total_reward,
        "final_info": final_info,
    }


def _subgoal_action(env: SO101NexusEnv, subgoal: Subgoal, step: int, attempt: int) -> list[float]:
    fraction = ((step + subgoal.index * 11) % 120) / 119.0
    action = sample_action(env.action_space, fraction)
    if attempt == 0:
        return action
    damping = 0.5
    return [float(value) * damping for value in action]


def _comparison_markdown(report: ComparisonReport) -> str:
    policy = report.policy_only
    agentic = report.agentic_retry
    distance_delta = policy.final_distance - agentic.final_distance
    reward_delta = agentic.total_reward - policy.total_reward
    rows = [
        "| Mode | Success | Steps | Final distance | Total reward | Retry events |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| policy_only | {policy.success} | {policy.steps} | "
            f"{policy.final_distance:.6f} | {policy.total_reward:.6f} | 0 |"
        ),
        (
            f"| agentic_retry | {agentic.success} | {agentic.steps} | "
            f"{agentic.final_distance:.6f} | {agentic.total_reward:.6f} | {agentic.retry_events} |"
        ),
    ]
    return "\n".join(
        [
            "# CP23 Policy vs Agentic Retry Comparison",
            "",
            f"- Environment: `{report.env_id}`",
            f"- Task: `{report.task}`",
            f"- Final-distance delta, positive means agentic is closer: `{distance_delta:.6f}`",
            f"- Total-reward delta, positive means agentic accumulated more reward: `{reward_delta:.6f}`",
            f"- Agentic subgoals passed: `{agentic.passed_subgoals}/{agentic.total_subgoals}`",
            "",
            "## Results",
            "",
            *rows,
            "",
            "## Artifacts",
            "",
            f"- Policy trace: `{policy.trace_path}`",
            f"- Agentic plan: `{agentic.plan_path}`",
            f"- Agentic trace: `{agentic.trace_path}`",
            "",
        ]
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _info_success(info: dict[str, Any]) -> bool:
    value = info.get("success", False)
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _info_distance(info: dict[str, Any]) -> float:
    try:
        return float(info.get("tcp_to_target_dist", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in info.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
