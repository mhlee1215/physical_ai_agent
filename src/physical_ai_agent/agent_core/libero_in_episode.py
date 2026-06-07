from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol


class StepEnv(Protocol):
    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        ...


class StepPolicy(Protocol):
    def reset(self) -> None:
        ...

    def select_action(self, observation: Any) -> Any:
        ...


@dataclass(frozen=True)
class VerifierTrigger:
    triggered: bool
    reason: str
    trigger_step: int | None = None
    metric_name: str = "none"
    metric_value: float = 0.0
    threshold: float = 0.0


@dataclass(frozen=True)
class StepRecord:
    step: int
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    verifier_triggered: bool
    verifier_reason: str
    intervention_type: str | None
    observation_summary: dict[str, Any]
    info: dict[str, Any]


@dataclass(frozen=True)
class InEpisodeRolloutResult:
    task_group: str
    task_id: int
    seed: int | None
    success: bool
    action_step_count: int
    verifier_trigger_count: int
    intervention_count: int
    total_reward: float
    eval_seconds: float
    environment_resets: int
    terminated: bool
    truncated: bool
    trace_path: str

    @property
    def success_per_action_step(self) -> float:
        return _ratio(1 if self.success else 0, self.action_step_count)


class StagnationVerifier:
    """Online verifier that fires when scalar progress stops changing."""

    def __init__(self, metric_name: str, window: int = 4, min_delta: float = 1e-4) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.metric_name = metric_name
        self.window = window
        self.min_delta = min_delta
        self._values: list[float] = []

    def observe(self, step: int, info: dict[str, Any]) -> VerifierTrigger:
        value = _safe_float(info.get(self.metric_name))
        if value is None:
            return VerifierTrigger(
                triggered=False,
                reason=f"{self.metric_name} missing",
                trigger_step=None,
                metric_name=self.metric_name,
            )
        self._values.append(value)
        if len(self._values) < self.window:
            return VerifierTrigger(
                triggered=False,
                reason=f"collecting {self.metric_name} window",
                trigger_step=None,
                metric_name=self.metric_name,
                metric_value=value,
                threshold=self.min_delta,
            )
        recent = self._values[-self.window :]
        if abs(recent[0] - recent[-1]) <= self.min_delta:
            return VerifierTrigger(
                triggered=True,
                reason=f"{self.metric_name} stagnated over {self.window} steps",
                trigger_step=step,
                metric_name=self.metric_name,
                metric_value=abs(recent[0] - recent[-1]),
                threshold=self.min_delta,
            )
        return VerifierTrigger(
            triggered=False,
            reason=f"{self.metric_name} changed",
            trigger_step=None,
            metric_name=self.metric_name,
            metric_value=abs(recent[0] - recent[-1]),
            threshold=self.min_delta,
        )


class ScaleActionIntervention:
    """Simple bounded intervention that rescales the next action in-episode."""

    def __init__(self, scale: float = 0.5, intervention_type: str = "scale_next_action") -> None:
        self.scale = scale
        self.intervention_type = intervention_type

    def apply(self, action: Any) -> tuple[Any, str]:
        if isinstance(action, (int, float)):
            return float(action) * self.scale, self.intervention_type
        if isinstance(action, list):
            return [_scale_value(value, self.scale) for value in action], self.intervention_type
        if isinstance(action, tuple):
            return tuple(_scale_value(value, self.scale) for value in action), self.intervention_type
        return action, self.intervention_type


def run_in_episode_rollout(
    *,
    env: StepEnv,
    policy: StepPolicy,
    verifier: StagnationVerifier,
    intervention: ScaleActionIntervention,
    output_jsonl: Path,
    task_group: str,
    task_id: int,
    seed: int | None,
    max_steps: int,
    max_interventions: int = 1,
) -> InEpisodeRolloutResult:
    started_at = perf_counter()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    policy.reset()
    observation, _reset_info = env.reset(seed=seed)
    records: list[StepRecord] = []
    total_reward = 0.0
    intervention_count = 0
    verifier_trigger_count = 0
    pending_intervention = False
    success = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        action = policy.select_action(observation)
        intervention_type = None
        if pending_intervention and intervention_count < max_interventions:
            action, intervention_type = intervention.apply(action)
            intervention_count += 1
            pending_intervention = False

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        success = _info_success(info)
        trigger = verifier.observe(step, info)
        if trigger.triggered and not (terminated or truncated) and intervention_count < max_interventions:
            verifier_trigger_count += 1
            pending_intervention = True

        records.append(
            StepRecord(
                step=step,
                action=_jsonable(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                success=success,
                verifier_triggered=trigger.triggered,
                verifier_reason=trigger.reason,
                intervention_type=intervention_type,
                observation_summary=_summarize_observation(observation),
                info=_jsonable_dict(info),
            )
        )
        if terminated or truncated:
            break

    write_jsonl(output_jsonl, [asdict(record) for record in records])
    return InEpisodeRolloutResult(
        task_group=task_group,
        task_id=task_id,
        seed=seed,
        success=success,
        action_step_count=len(records),
        verifier_trigger_count=verifier_trigger_count,
        intervention_count=intervention_count,
        total_reward=total_reward,
        eval_seconds=perf_counter() - started_at,
        environment_resets=1,
        terminated=bool(terminated),
        truncated=bool(truncated),
        trace_path=str(output_jsonl),
    )


def write_result(result: InEpisodeRolloutResult, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result) | {"success_per_action_step": result.success_per_action_step}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: InEpisodeRolloutResult) -> str:
    return "\n".join(
        [
            "# LIBERO In-Episode Instrumented Rollout",
            "",
            f"- task_group: `{result.task_group}`",
            f"- task_id: `{result.task_id}`",
            f"- seed: `{result.seed}`",
            f"- trace_path: `{result.trace_path}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| success | {str(result.success).lower()} |",
            f"| action_step_count | {result.action_step_count} |",
            f"| verifier_trigger_count | {result.verifier_trigger_count} |",
            f"| intervention_count | {result.intervention_count} |",
            f"| environment_resets | {result.environment_resets} |",
            f"| eval_seconds | {result.eval_seconds:.6f} |",
            f"| success_per_action_step | {result.success_per_action_step:.6f} |",
            "",
            "## Semantics",
            "",
            "- This is an in-episode instrumentation contract: verifier triggers and interventions occur before terminal reset.",
            "- Final task success must still come from the benchmark/environment success flag.",
            "",
        ]
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _info_success(info: dict[str, Any]) -> bool:
    return bool(info.get("is_success", info.get("success", False)))


def _ratio(numerator: int, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scale_value(value: Any, scale: float) -> Any:
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return value


def _summarize_observation(observation: Any) -> dict[str, Any]:
    if isinstance(observation, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in observation)}
    if isinstance(observation, (list, tuple)):
        return {"type": type(observation).__name__, "length": len(observation)}
    return {"type": type(observation).__name__}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return _jsonable_dict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _jsonable_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(item) for key, item in value.items()}
