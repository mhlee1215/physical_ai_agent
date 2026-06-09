from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER

SUPPORTED_ACTION_SEMANTICS = {"absolute_joint_position", "joint_delta"}
SUPPORTED_GRIPPER_SEMANTICS = {"higher_raw_opens", "higher_raw_closes"}
SUPPORTED_COMMAND_UNITS = {"feetech_raw_ticks", "lerobot_so100_position"}
DEFAULT_POLICY_POSTPROCESSOR_CONFIG = "policy_postprocessor.json"


@dataclass(frozen=True)
class SmolVLAActionMetadata:
    model_id: str
    action_dim: int
    action_normalization: str
    action_stats_available: bool
    output_is_normalized: bool
    chunk_size: int | None
    n_action_steps: int | None
    action_semantics: str | None
    joint_order: list[str] | None
    gripper_semantics: str | None
    stats_source: str | None
    selected_action_stats_key: str | None
    available_action_stats_keys: list[str]
    command_units: str | None
    blockers: list[str]


@dataclass(frozen=True)
class SO100SmolVLAJointTarget:
    joint: str
    normalized_action_value: float
    unnormalized_action_value: float
    target_command_value: float
    command_units: str
    write_normalize: bool
    target_raw: float
    range_min: float | None
    range_max: float | None
    clipped_by_calibrated_range: bool
    raw_target_in_calibrated_range: bool | None


@dataclass(frozen=True)
class SO100SmolVLACommandStep:
    step_index: int
    joint_targets: list[SO100SmolVLAJointTarget]


@dataclass(frozen=True)
class SO100SmolVLACommandChunkPlan:
    status: str
    ready_for_execution: bool
    action_adapter: str
    model_id: str
    action_chunk_steps: int
    action_dim: int
    expected_action_dim: int
    action_normalization: str
    action_semantics: str | None
    joint_order: list[str] | None
    gripper_semantics: str | None
    command_units: str
    step_plans: list[SO100SmolVLACommandStep]
    blockers: list[str]
    notes: list[str]


def inspect_smolvla_action_metadata(
    *,
    config: dict[str, Any],
    model_id: str,
    stats: dict[str, Any] | None = None,
    action_semantics: str | None = None,
    joint_order: list[str] | None = None,
    gripper_semantics: str | None = None,
    command_units: str | None = None,
) -> SmolVLAActionMetadata:
    output_features = config.get("output_features") or {}
    action_feature = output_features.get("action") or {}
    action_shape = action_feature.get("shape") or []
    action_dim = int(action_shape[0]) if len(action_shape) == 1 else 0
    normalization_mapping = config.get("normalization_mapping") or {}
    action_normalization = str(normalization_mapping.get("ACTION", "UNKNOWN")).upper()
    action_stats = _extract_action_mean_std(stats, expected_dim=action_dim)
    action_stats_available = action_stats is not None
    available_action_stats_keys = _action_stats_keys(stats, expected_dim=action_dim)
    output_is_normalized = action_normalization not in {"IDENTITY", "NONE"}
    blockers: list[str] = []

    if action_dim != len(SO100_JOINT_ORDER):
        blockers.append(f"SmolVLA action dim is {action_dim}; SO-100 follower expects {len(SO100_JOINT_ORDER)}.")
    if output_is_normalized and not action_stats_available:
        blockers.append(f"Action normalization is {action_normalization}, but action mean/std stats are unavailable.")
    if action_semantics not in SUPPORTED_ACTION_SEMANTICS:
        blockers.append("Action semantics must be explicitly confirmed as absolute_joint_position or joint_delta.")
    if joint_order != SO100_JOINT_ORDER:
        blockers.append(f"Joint order must match SO-100 follower order: {SO100_JOINT_ORDER}.")
    if gripper_semantics not in SUPPORTED_GRIPPER_SEMANTICS:
        blockers.append("Gripper semantics must be explicitly confirmed as higher_raw_opens or higher_raw_closes.")
    if command_units not in SUPPORTED_COMMAND_UNITS:
        blockers.append(
            "Command units must be explicitly confirmed as feetech_raw_ticks or lerobot_so100_position before writing SO-100 Goal_Position."
        )

    return SmolVLAActionMetadata(
        model_id=model_id,
        action_dim=action_dim,
        action_normalization=action_normalization,
        action_stats_available=action_stats_available,
        output_is_normalized=output_is_normalized,
        chunk_size=_optional_int(config.get("chunk_size")),
        n_action_steps=_optional_int(config.get("n_action_steps")),
        action_semantics=action_semantics,
        joint_order=joint_order,
        gripper_semantics=gripper_semantics,
        stats_source=_optional_str((stats or {}).get("_source")),
        selected_action_stats_key=_optional_str((stats or {}).get("_selected_action_stats_key")),
        available_action_stats_keys=available_action_stats_keys,
        command_units=command_units,
        blockers=blockers,
    )


def build_so100_smolvla_metadata_command_chunk_plan(
    *,
    action_chunk: list[list[float]],
    current_state: dict[str, Any],
    calibration: dict[str, Any] | None,
    config: dict[str, Any],
    model_id: str,
    stats: dict[str, Any] | None,
    action_semantics: str | None,
    joint_order: list[str] | None,
    gripper_semantics: str | None,
    command_units: str | None = None,
) -> SO100SmolVLACommandChunkPlan:
    metadata = inspect_smolvla_action_metadata(
        config=config,
        model_id=model_id,
        stats=stats,
        action_semantics=action_semantics,
        joint_order=joint_order,
        gripper_semantics=gripper_semantics,
        command_units=command_units,
    )
    blockers = list(metadata.blockers)
    action_stats = _extract_action_mean_std(stats, expected_dim=metadata.action_dim)
    step_plans: list[SO100SmolVLACommandStep] = []

    if not action_chunk:
        blockers.append("No action chunk steps were provided.")
    wrong_dim = [
        index for index, action in enumerate(action_chunk) if len(action) != len(SO100_JOINT_ORDER)
    ]
    if wrong_dim:
        blockers.append(f"Chunk contains wrong-dimension actions at step indexes: {wrong_dim}.")

    if not blockers and action_stats is not None and action_semantics in SUPPORTED_ACTION_SEMANTICS:
        mean, std = action_stats
        simulated_state = {joint: _float_or_nan(current_state.get(joint)) for joint in SO100_JOINT_ORDER}
        for step_index, action in enumerate(action_chunk):
            joint_targets: list[SO100SmolVLAJointTarget] = []
            for joint_index, joint in enumerate(SO100_JOINT_ORDER):
                normalized_value = float(action[joint_index])
                unnormalized_value = normalized_value * std[joint_index] + mean[joint_index]
                current = simulated_state[joint]
                if action_semantics == "absolute_joint_position":
                    target_command = unnormalized_value
                else:
                    target_command = current + unnormalized_value

                joint_calibration = (calibration or {}).get(joint, {})
                range_min = _optional_float(joint_calibration.get("range_min"))
                range_max = _optional_float(joint_calibration.get("range_max"))
                clipped_by_range = False
                write_normalize = command_units == "lerobot_so100_position"
                target_raw = (
                    _lerobot_so100_position_to_raw(joint=joint, value=target_command, calibration=joint_calibration)
                    if command_units == "lerobot_so100_position"
                    else target_command
                )
                raw_target_in_range = None
                if range_min is not None and range_max is not None and math.isfinite(target_raw):
                    raw_target_in_range = range_min <= target_raw <= range_max
                    if command_units == "feetech_raw_ticks":
                        bounded = _clip(target_raw, range_min, range_max)
                        clipped_by_range = bounded != target_raw
                        target_raw = bounded
                        target_command = bounded
                    elif not raw_target_in_range:
                        blockers.append(
                            f"Step {step_index} joint {joint} maps to raw target {target_raw:.4f} outside calibrated range [{range_min:.4f}, {range_max:.4f}]."
                        )
                simulated_state[joint] = target_command if command_units == "lerobot_so100_position" else target_raw
                joint_targets.append(
                    SO100SmolVLAJointTarget(
                        joint=joint,
                        normalized_action_value=normalized_value,
                        unnormalized_action_value=unnormalized_value,
                        target_command_value=target_command,
                        command_units=command_units or "unconfirmed",
                        write_normalize=write_normalize,
                        target_raw=target_raw,
                        range_min=range_min,
                        range_max=range_max,
                        clipped_by_calibrated_range=clipped_by_range,
                        raw_target_in_calibrated_range=raw_target_in_range,
                    )
                )
            step_plans.append(SO100SmolVLACommandStep(step_index=step_index, joint_targets=joint_targets))

    non_finite = [
        target.joint
        for step in step_plans
        for target in step.joint_targets
        if not math.isfinite(target.target_raw)
    ]
    if non_finite:
        blockers.append(f"Command plan contains non-finite targets for joints: {sorted(set(non_finite))}.")

    ready = bool(step_plans) and not blockers
    return SO100SmolVLACommandChunkPlan(
        status="passed" if ready else "blocked",
        ready_for_execution=ready,
        action_adapter="smolvla_metadata_unnormalize_to_so100_follower_v1",
        model_id=model_id,
        action_chunk_steps=len(action_chunk),
        action_dim=metadata.action_dim,
        expected_action_dim=len(SO100_JOINT_ORDER),
        action_normalization=metadata.action_normalization,
        action_semantics=action_semantics,
        joint_order=joint_order,
        gripper_semantics=gripper_semantics,
        command_units=command_units or "unconfirmed",
        step_plans=step_plans,
        blockers=blockers,
        notes=[
            "SmolVLA ACTION outputs are interpreted only after checkpoint normalization metadata is inspected.",
            "MEAN_STD actions require authoritative action mean/std stats before unnormalization.",
            "SO-100 execution requires explicit follower joint order, gripper semantics, command units, and calibrated-range checks.",
        ],
    )


def load_smolvla_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_action_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_policy_postprocessor_action_stats(
    *,
    model_id_or_path: str,
    output: Path | None = None,
    action_stats_key: str | None = None,
    local_files_only: bool = False,
    config_filename: str = DEFAULT_POLICY_POSTPROCESSOR_CONFIG,
) -> dict[str, Any]:
    config, base_path, config_path = _load_policy_postprocessor_config(
        model_id_or_path=model_id_or_path,
        config_filename=config_filename,
        local_files_only=local_files_only,
    )
    unnormalizer = _find_unnormalizer_step(config)
    state_file = unnormalizer.get("state_file")
    if not state_file:
        raise ValueError(f"{config_filename} does not reference an unnormalizer state_file")
    state_path = _resolve_postprocessor_state_file(
        model_id_or_path=model_id_or_path,
        base_path=base_path,
        state_file=str(state_file),
        local_files_only=local_files_only,
    )
    from safetensors.torch import load_file

    state = load_file(str(state_path))
    candidates = _extract_action_stats_candidates(state)
    if not candidates:
        raise ValueError(f"No action mean/std stats found in {state_path}")
    selected_key = _select_action_stats_key(candidates, action_stats_key=action_stats_key)
    payload = {
        "status": "passed",
        "model_id_or_path": model_id_or_path,
        "config_filename": config_filename,
        "config_path": str(config_path),
        "state_file": str(state_file),
        "state_path": str(state_path),
        "available_action_stats_keys": sorted(candidates),
        "selected_action_stats_key": selected_key,
        "action": candidates[selected_key],
        "_source": "lerobot_policy_postprocessor",
        "_selected_action_stats_key": selected_key,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _extract_action_mean_std(
    stats: dict[str, Any] | None,
    *,
    expected_dim: int,
) -> tuple[list[float], list[float]] | None:
    if not stats:
        return None
    action_stats = stats.get("action") if isinstance(stats, dict) else None
    if not isinstance(action_stats, dict):
        return None
    mean = _stats_vector(action_stats.get("mean"))
    std = _stats_vector(action_stats.get("std"))
    if mean is None or std is None:
        return None
    if len(mean) != expected_dim or len(std) != expected_dim:
        return None
    if any(not math.isfinite(value) for value in mean + std):
        return None
    if any(value == 0 for value in std):
        return None
    return mean, std


def _action_stats_keys(stats: dict[str, Any] | None, *, expected_dim: int) -> list[str]:
    if not stats:
        return []
    keys = []
    for key, value in stats.items():
        if not isinstance(value, dict):
            continue
        if _extract_action_mean_std({"action": value}, expected_dim=expected_dim) is not None:
            keys.append(str(key))
    if _extract_action_mean_std(stats, expected_dim=expected_dim) is not None and "action" not in keys:
        keys.append("action")
    return sorted(keys)


def _load_policy_postprocessor_config(
    *,
    model_id_or_path: str,
    config_filename: str,
    local_files_only: bool,
) -> tuple[dict[str, Any], Path | None, Path]:
    model_path = Path(model_id_or_path)
    if model_path.is_dir():
        config_path = model_path / config_filename
        return json.loads(config_path.read_text(encoding="utf-8")), model_path, config_path
    if model_path.is_file():
        return json.loads(model_path.read_text(encoding="utf-8")), model_path.parent, model_path
    from huggingface_hub import hf_hub_download

    config_path = Path(
        hf_hub_download(
            repo_id=model_id_or_path,
            filename=config_filename,
            repo_type="model",
            local_files_only=local_files_only,
        )
    )
    return json.loads(config_path.read_text(encoding="utf-8")), config_path.parent, config_path


def _find_unnormalizer_step(config: dict[str, Any]) -> dict[str, Any]:
    for step in config.get("steps", []):
        registry_name = str(step.get("registry_name") or "")
        class_name = str(step.get("class") or "")
        if "unnormalizer" in registry_name.lower() or "unnormalizer" in class_name.lower():
            return step
    raise ValueError("policy postprocessor does not contain an unnormalizer step")


def _resolve_postprocessor_state_file(
    *,
    model_id_or_path: str,
    base_path: Path | None,
    state_file: str,
    local_files_only: bool,
) -> Path:
    if base_path is not None and (base_path / state_file).exists():
        return base_path / state_file
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=model_id_or_path,
            filename=state_file,
            repo_type="model",
            local_files_only=local_files_only,
        )
    )


def _extract_action_stats_candidates(state: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    candidates: dict[str, dict[str, list[float]]] = {}
    for key, tensor in state.items():
        if not key.endswith(".action.mean"):
            continue
        prefix = key.removesuffix(".action.mean")
        std_key = f"{prefix}.action.std"
        if std_key not in state:
            continue
        candidates[prefix] = {
            "mean": [float(item) for item in tensor.detach().cpu().tolist()],
            "std": [float(item) for item in state[std_key].detach().cpu().tolist()],
        }
    return candidates


def _select_action_stats_key(candidates: dict[str, dict[str, list[float]]], *, action_stats_key: str | None) -> str:
    if action_stats_key:
        if action_stats_key not in candidates:
            raise ValueError(
                f"requested action_stats_key={action_stats_key!r} not found; available={sorted(candidates)}"
            )
        return action_stats_key
    if len(candidates) == 1:
        return next(iter(candidates))
    if "so100.buffer" in candidates:
        return "so100.buffer"
    raise ValueError(f"multiple action stats candidates found; pass --action-stats-key. available={sorted(candidates)}")


def _stats_vector(value: Any) -> list[float] | None:
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if not isinstance(value, list):
        return None
    return [float(item) for item in _flatten(value)]


def _flatten(value: list[Any]) -> list[Any]:
    out: list[Any] = []
    for item in value:
        if isinstance(item, list):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _lerobot_so100_position_to_raw(*, joint: str, value: float, calibration: dict[str, Any]) -> float:
    range_min = _optional_float(calibration.get("range_min"))
    range_max = _optional_float(calibration.get("range_max"))
    if range_min is None or range_max is None:
        return math.nan
    if joint == "gripper":
        return (float(value) / 100.0) * (range_max - range_min) + range_min
    mid = (range_min + range_max) / 2.0
    max_res = 4095.0
    return float(value) * max_res / 360.0 + mid


def raw_to_lerobot_so100_position(*, joint: str, raw_value: float, calibration: dict[str, Any]) -> float:
    range_min = _optional_float(calibration.get("range_min"))
    range_max = _optional_float(calibration.get("range_max"))
    if range_min is None or range_max is None:
        return math.nan
    if joint == "gripper":
        return (float(raw_value) - range_min) * 100.0 / (range_max - range_min)
    mid = (range_min + range_max) / 2.0
    max_res = 4095.0
    return (float(raw_value) - mid) * 360.0 / max_res


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
