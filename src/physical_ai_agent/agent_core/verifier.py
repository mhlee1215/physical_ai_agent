from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from physical_ai_agent.agent_core.planner import Subgoal


@dataclass(frozen=True)
class VerificationDecision:
    subgoal_name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    reason: str


class SO101SimulationStateVerifier:
    def verify(self, subgoal: Subgoal, info: dict[str, Any]) -> VerificationDecision:
        success = _as_bool(info.get("success", False))
        metric_value = _as_float(info.get(subgoal.success_metric), default=math.inf)
        passed = success or metric_value <= subgoal.threshold
        if success:
            reason = "environment reported success"
        elif passed:
            reason = f"{subgoal.success_metric} <= {subgoal.threshold:.3f}"
        elif math.isfinite(metric_value):
            reason = f"{subgoal.success_metric}={metric_value:.6f} above threshold"
        else:
            reason = f"{subgoal.success_metric} missing or non-finite"
        return VerificationDecision(
            subgoal_name=subgoal.name,
            passed=passed,
            metric_name=subgoal.success_metric,
            metric_value=metric_value,
            threshold=subgoal.threshold,
            reason=reason,
        )


def _as_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
