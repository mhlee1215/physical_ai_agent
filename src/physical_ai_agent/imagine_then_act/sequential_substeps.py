"""Conservative sequential-substep verifier utilities for LIBERO probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubstepVerifierSpec:
    substep_id: str
    prompt: str
    verifier_type: str
    target_object_key: str
    receptacle_object_key: str | None = None
    distance_threshold: float = 0.08
    progress_threshold: float = 0.015
    eef_distance_threshold: float | None = None
    eef_progress_threshold: float | None = None
    pass_on_progress: bool = False
    required: bool = True
    max_attempts: int = 2
    chunk_steps: int = 15


@dataclass(frozen=True)
class VerifierDecision:
    status: str
    reason: str
    score: float | None
    details: dict[str, Any]


def normalize_substep_plan(payload: dict[str, Any]) -> list[SubstepVerifierSpec]:
    records = payload.get("substeps") or payload.get("sequential_substeps") or []
    if not isinstance(records, list):
        raise ValueError("sequential substep plan must contain a substeps list")
    substeps: list[SubstepVerifierSpec] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"substep {index} must be an object")
        substep_id = str(record.get("substep_id") or record.get("id") or f"substep_{index + 1:02d}")
        prompt = str(record.get("prompt") or record.get("goal") or "").strip()
        verifier = record.get("verifier") if isinstance(record.get("verifier"), dict) else record
        verifier_type = str(verifier.get("type") or verifier.get("verifier_type") or "object_in_target")
        target_object_key = str(verifier.get("target_object_key") or "").strip()
        receptacle_object_key = verifier.get("receptacle_object_key")
        if receptacle_object_key is not None:
            receptacle_object_key = str(receptacle_object_key).strip()
        if not prompt:
            raise ValueError(f"{substep_id} missing prompt")
        if not target_object_key:
            raise ValueError(f"{substep_id} missing target_object_key")
        if verifier_type == "object_in_target" and not receptacle_object_key:
            raise ValueError(f"{substep_id} object_in_target verifier requires receptacle_object_key")
        substeps.append(
            SubstepVerifierSpec(
                substep_id=substep_id,
                prompt=prompt,
                verifier_type=verifier_type,
                target_object_key=target_object_key,
                receptacle_object_key=receptacle_object_key,
                distance_threshold=float(verifier.get("distance_threshold", record.get("distance_threshold", 0.08))),
                progress_threshold=float(verifier.get("progress_threshold", record.get("progress_threshold", 0.015))),
                eef_distance_threshold=_optional_float(
                    verifier.get("eef_distance_threshold", record.get("eef_distance_threshold"))
                ),
                eef_progress_threshold=_optional_float(
                    verifier.get("eef_progress_threshold", record.get("eef_progress_threshold"))
                ),
                pass_on_progress=bool(verifier.get("pass_on_progress", record.get("pass_on_progress", False))),
                required=bool(record.get("required", True)),
                max_attempts=max(1, int(record.get("max_attempts", 2))),
                chunk_steps=max(1, int(record.get("chunk_steps", 15))),
            )
        )
    if not substeps:
        raise ValueError("sequential substep plan is empty")
    return substeps


def verify_substep_completion(
    spec: SubstepVerifierSpec,
    semantic_state: dict[str, Any],
    *,
    previous_semantic_state: dict[str, Any] | None = None,
) -> VerifierDecision:
    if spec.verifier_type == "object_in_target":
        return verify_object_in_target(spec, semantic_state, previous_semantic_state=previous_semantic_state)
    if spec.verifier_type == "eef_near_object":
        return verify_eef_near_object(spec, semantic_state, previous_semantic_state=previous_semantic_state)
    if spec.verifier_type == "object_lifted":
        return verify_object_lifted(spec, semantic_state, previous_semantic_state=previous_semantic_state)
    return VerifierDecision(
        status="unknown",
        reason=f"unsupported_verifier_type:{spec.verifier_type}",
        score=None,
        details={"substep_id": spec.substep_id, "verifier_type": spec.verifier_type},
    )


def verify_eef_near_object(
    spec: SubstepVerifierSpec,
    semantic_state: dict[str, Any],
    *,
    previous_semantic_state: dict[str, Any] | None = None,
) -> VerifierDecision:
    distance = _number(semantic_state.get("eef_to_target_dist"))
    previous_distance = (
        _number(previous_semantic_state.get("eef_to_target_dist")) if previous_semantic_state is not None else None
    )
    threshold = spec.eef_distance_threshold if spec.eef_distance_threshold is not None else spec.distance_threshold
    progress_threshold = (
        spec.eef_progress_threshold if spec.eef_progress_threshold is not None else spec.progress_threshold
    )
    details = {
        "substep_id": spec.substep_id,
        "verifier_type": spec.verifier_type,
        "target_object_key": spec.target_object_key,
        "eef_to_target_dist": distance,
        "previous_eef_to_target_dist": previous_distance,
        "eef_distance_threshold": threshold,
        "eef_progress_threshold": progress_threshold,
        "pass_on_progress": spec.pass_on_progress,
    }
    if distance is None:
        return VerifierDecision(
            status="unknown",
            reason="eef_or_object_position_unavailable",
            score=None,
            details=details,
        )
    if distance <= threshold:
        return VerifierDecision(
            status="pass",
            reason=f"eef_to_target_dist_{distance:.6f}_le_{threshold:.6f}",
            score=1.0,
            details=details,
        )
    if previous_distance is not None:
        progress = previous_distance - distance
        details["eef_distance_progress"] = progress
        if progress >= progress_threshold:
            status = "pass" if spec.pass_on_progress else "unknown"
            score = 0.5 if spec.pass_on_progress else 0.25
            return VerifierDecision(
                status=status,
                reason=f"eef_target_progress_{progress:.6f}_ge_{progress_threshold:.6f}",
                score=score,
                details=details,
            )
    return VerifierDecision(
        status="unknown",
        reason=f"eef_not_near_dist_{distance:.6f}_gt_{threshold:.6f}",
        score=0.0,
        details=details,
    )


def verify_object_lifted(
    spec: SubstepVerifierSpec,
    semantic_state: dict[str, Any],
    *,
    previous_semantic_state: dict[str, Any] | None = None,
) -> VerifierDecision:
    target_pos = semantic_state.get("target_pos")
    previous_pos = previous_semantic_state.get("target_pos") if previous_semantic_state is not None else None
    z = _position_z(target_pos)
    previous_z = _position_z(previous_pos)
    threshold = spec.progress_threshold
    details = {
        "substep_id": spec.substep_id,
        "verifier_type": spec.verifier_type,
        "target_object_key": spec.target_object_key,
        "target_z": z,
        "previous_target_z": previous_z,
        "lift_progress_threshold": threshold,
        "pass_on_progress": spec.pass_on_progress,
    }
    if z is None:
        return VerifierDecision(
            status="unknown",
            reason="object_position_unavailable",
            score=None,
            details=details,
        )
    if previous_z is not None:
        lift = z - previous_z
        details["lift_progress"] = lift
        if lift >= threshold:
            return VerifierDecision(
                status="pass",
                reason=f"target_lift_progress_{lift:.6f}_ge_{threshold:.6f}",
                score=0.8,
                details=details,
            )
    return VerifierDecision(
        status="unknown",
        reason="target_not_lifted",
        score=0.0,
        details=details,
    )


def verify_object_in_target(
    spec: SubstepVerifierSpec,
    semantic_state: dict[str, Any],
    *,
    previous_semantic_state: dict[str, Any] | None = None,
) -> VerifierDecision:
    distance = _number(semantic_state.get("target_to_receptacle_dist"))
    details = {
        "substep_id": spec.substep_id,
        "verifier_type": spec.verifier_type,
        "target_object_key": spec.target_object_key,
        "receptacle_object_key": spec.receptacle_object_key,
        "distance_threshold": spec.distance_threshold,
        "target_to_receptacle_dist": distance,
        "pass_policy": "distance_threshold_or_relaxed_progress"
        if spec.pass_on_progress
        else "high_precision_only_distance_le_threshold",
        "pass_on_progress": spec.pass_on_progress,
    }
    if distance is None:
        return VerifierDecision(
            status="unknown",
            reason="object_or_receptacle_position_unavailable",
            score=None,
            details=details,
        )
    if distance <= spec.distance_threshold:
        return VerifierDecision(
            status="pass",
            reason=f"target_to_receptacle_dist_{distance:.6f}_le_{spec.distance_threshold:.6f}",
            score=1.0,
            details=details,
        )
    previous_distance = None
    previous_eef_distance = None
    if previous_semantic_state is not None:
        previous_distance = _number(previous_semantic_state.get("target_to_receptacle_dist"))
        previous_eef_distance = _number(previous_semantic_state.get("eef_to_target_dist"))
    details["previous_target_to_receptacle_dist"] = previous_distance
    eef_distance = _number(semantic_state.get("eef_to_target_dist"))
    details["eef_to_target_dist"] = eef_distance
    details["previous_eef_to_target_dist"] = previous_eef_distance
    if spec.eef_distance_threshold is not None:
        details["eef_distance_threshold"] = spec.eef_distance_threshold
    if spec.eef_progress_threshold is not None:
        details["eef_progress_threshold"] = spec.eef_progress_threshold
    if spec.pass_on_progress and eef_distance is not None and spec.eef_distance_threshold is not None:
        if eef_distance <= spec.eef_distance_threshold:
            return VerifierDecision(
                status="pass",
                reason=f"eef_to_target_dist_{eef_distance:.6f}_le_{spec.eef_distance_threshold:.6f}",
                score=0.7,
                details=details,
            )
    if previous_distance is not None:
        progress = previous_distance - distance
        details["distance_progress"] = progress
        if progress >= spec.progress_threshold:
            if spec.pass_on_progress:
                return VerifierDecision(
                    status="pass",
                    reason=(
                        f"target_receptacle_progress_{progress:.6f}_ge_"
                        f"{spec.progress_threshold:.6f}"
                    ),
                    score=0.6,
                    details=details,
                )
            return VerifierDecision(
                status="unknown",
                reason=(
                    f"progress_without_completion_{progress:.6f}_ge_"
                    f"{spec.progress_threshold:.6f}"
                ),
                score=0.5,
                details=details,
            )
    if (
        spec.pass_on_progress
        and eef_distance is not None
        and previous_eef_distance is not None
        and spec.eef_progress_threshold is not None
    ):
        eef_progress = previous_eef_distance - eef_distance
        details["eef_distance_progress"] = eef_progress
        if eef_progress >= spec.eef_progress_threshold:
            return VerifierDecision(
                status="pass",
                reason=f"eef_target_progress_{eef_progress:.6f}_ge_{spec.eef_progress_threshold:.6f}",
                score=0.4,
                details=details,
            )
    return VerifierDecision(
        status="unknown",
        reason=f"not_complete_dist_{distance:.6f}_gt_{spec.distance_threshold:.6f}",
        score=0.0,
        details=details,
    )


def should_retry_or_advance(
    decision: VerifierDecision,
    *,
    attempt: int,
    max_attempts: int,
    required: bool,
) -> dict[str, Any]:
    if decision.status == "pass":
        return {"next": "advance", "terminal": False, "counts_as_pass": True}
    if attempt < max_attempts:
        return {"next": "retry", "terminal": False, "counts_as_pass": False}
    if required:
        return {
            "next": "advance_with_required_unmet",
            "terminal": False,
            "counts_as_pass": False,
            "warning": "required substep exhausted attempts without verifier pass",
        }
    return {
        "next": "advance_optional_unmet",
        "terminal": False,
        "counts_as_pass": False,
        "warning": "optional substep exhausted attempts without verifier pass",
    }


def task0_sequential_substep_plan(*, relaxed_progress: bool = False) -> dict[str, Any]:
    source = "task0_handauthored_relaxed_progress_pilot" if relaxed_progress else "task0_handauthored_conservative_pilot"
    false_positive_policy = (
        "relaxed diagnostic: progress/near-target can advance substeps, but full task success still requires LIBERO success"
        if relaxed_progress
        else "verifier pass requires object-target distance under threshold; progress alone is UNKNOWN"
    )
    distance_threshold = 0.12 if relaxed_progress else 0.08
    progress_threshold = 0.01 if relaxed_progress else 0.015
    max_attempts = 3 if relaxed_progress else 2
    return {
        "schema": "physical_ai_agent.sequential_substeps.v1",
        "suite": "libero_10",
        "task_id": 0,
        "task_description": "put both the alphabet soup and the tomato sauce in the basket",
        "source": source,
        "false_positive_policy": false_positive_policy,
        "substeps": [
            {
                "substep_id": "task0_step01_alphabet_soup_to_basket",
                "prompt": "Put the alphabet soup in the basket.",
                "required": True,
                "max_attempts": max_attempts,
                "chunk_steps": 15,
                "verifier": {
                    "type": "object_in_target",
                    "target_object_key": "alphabet_soup_1_pos",
                    "receptacle_object_key": "basket_1_pos",
                    "distance_threshold": distance_threshold,
                    "progress_threshold": progress_threshold,
                    "eef_distance_threshold": 0.20,
                    "eef_progress_threshold": 0.03,
                    "pass_on_progress": relaxed_progress,
                },
            },
            {
                "substep_id": "task0_step02_tomato_sauce_to_basket",
                "prompt": "Put the tomato sauce in the basket.",
                "required": True,
                "max_attempts": max_attempts,
                "chunk_steps": 15,
                "verifier": {
                    "type": "object_in_target",
                    "target_object_key": "tomato_sauce_1_pos",
                    "receptacle_object_key": "basket_1_pos",
                    "distance_threshold": distance_threshold,
                    "progress_threshold": progress_threshold,
                    "eef_distance_threshold": 0.20,
                    "eef_progress_threshold": 0.03,
                    "pass_on_progress": relaxed_progress,
                },
            },
        ],
    }


def task0_primitive_relaxed_substep_plan(
    *,
    chunk_steps: int = 15,
    max_attempts: int = 3,
    lift_progress_threshold: float = 0.015,
    place_pass_on_progress: bool = True,
) -> dict[str, Any]:
    def reach(substep_id: str, prompt: str, target: str) -> dict[str, Any]:
        return {
            "substep_id": substep_id,
            "prompt": prompt,
            "required": True,
            "max_attempts": max_attempts,
            "chunk_steps": chunk_steps,
            "verifier": {
                "type": "eef_near_object",
                "target_object_key": target,
                "eef_distance_threshold": 0.18,
                "eef_progress_threshold": 0.025,
                "pass_on_progress": True,
            },
        }

    def lift(substep_id: str, prompt: str, target: str) -> dict[str, Any]:
        return {
            "substep_id": substep_id,
            "prompt": prompt,
            "required": True,
            "max_attempts": max_attempts,
            "chunk_steps": chunk_steps,
            "verifier": {
                "type": "object_lifted",
                "target_object_key": target,
                "progress_threshold": lift_progress_threshold,
                "pass_on_progress": True,
            },
        }

    def place(substep_id: str, prompt: str, target: str) -> dict[str, Any]:
        return {
            "substep_id": substep_id,
            "prompt": prompt,
            "required": True,
            "max_attempts": max_attempts,
            "chunk_steps": chunk_steps,
            "verifier": {
                "type": "object_in_target",
                "target_object_key": target,
                "receptacle_object_key": "basket_1_pos",
                "distance_threshold": 0.12,
                "progress_threshold": 0.01,
                "eef_distance_threshold": 0.20,
                "eef_progress_threshold": 0.03,
                "pass_on_progress": place_pass_on_progress,
            },
        }

    return {
        "schema": "physical_ai_agent.sequential_substeps.v1",
        "suite": "libero_10",
        "task_id": 0,
        "task_description": "put both the alphabet soup and the tomato sauce in the basket",
        "source": "task0_handauthored_primitive_relaxed_round2",
        "false_positive_policy": (
            "primitive diagnostic: reach and lift can advance on local progress; "
            "full task success still requires LIBERO success"
        ),
        "substeps": [
            reach(
                "task0_step01_reach_alphabet_soup",
                "Move the gripper directly above the alphabet soup.",
                "alphabet_soup_1_pos",
            ),
            lift(
                "task0_step02_grasp_lift_alphabet_soup",
                "Close the gripper on the alphabet soup and lift it slightly.",
                "alphabet_soup_1_pos",
            ),
            place(
                "task0_step03_place_alphabet_soup_in_basket",
                "Move the alphabet soup over the basket and release it.",
                "alphabet_soup_1_pos",
            ),
            reach(
                "task0_step04_reach_tomato_sauce",
                "Move the gripper directly above the tomato sauce.",
                "tomato_sauce_1_pos",
            ),
            lift(
                "task0_step05_grasp_lift_tomato_sauce",
                "Close the gripper on the tomato sauce and lift it slightly.",
                "tomato_sauce_1_pos",
            ),
            place(
                "task0_step06_place_tomato_sauce_in_basket",
                "Move the tomato sauce over the basket and release it.",
                "tomato_sauce_1_pos",
            ),
        ],
    }


def task0_push_only_substep_plan(*, chunk_steps: int = 45, max_attempts: int = 6) -> dict[str, Any]:
    def push(substep_id: str, prompt: str, target: str) -> dict[str, Any]:
        return {
            "substep_id": substep_id,
            "prompt": prompt,
            "required": True,
            "max_attempts": max_attempts,
            "chunk_steps": chunk_steps,
            "verifier": {
                "type": "object_in_target",
                "target_object_key": target,
                "receptacle_object_key": "basket_1_pos",
                "distance_threshold": 0.12,
                "progress_threshold": 0.01,
                "eef_distance_threshold": 0.20,
                "eef_progress_threshold": 0.03,
                "pass_on_progress": False,
            },
        }

    return {
        "schema": "physical_ai_agent.sequential_substeps.v1",
        "suite": "libero_10",
        "task_id": 0,
        "task_description": "put both the alphabet soup and the tomato sauce in the basket",
        "source": "task0_handauthored_push_only_round8",
        "false_positive_policy": (
            "push-only diagnostic: object-to-basket distance must cross threshold; "
            "eef progress alone cannot advance the substep"
        ),
        "substeps": [
            push(
                "task0_step01_push_alphabet_soup_to_basket",
                "Push the alphabet soup along the table into the basket.",
                "alphabet_soup_1_pos",
            ),
            push(
                "task0_step02_push_tomato_sauce_to_basket",
                "Push the tomato sauce along the table into the basket.",
                "tomato_sauce_1_pos",
            ),
        ],
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _position_z(value: Any) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3 and isinstance(value[2], (int, float)):
        return float(value[2])
    return None
