#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER, load_calibration
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import (
    _lerobot_so100_position_to_raw,
    extract_policy_postprocessor_action_stats,
    load_action_stats,
)


DEFAULT_MODEL_CACHE = Path(
    "/Users/minhaeng/.cache/huggingface/hub/models--lerobot--smolvla_base/"
    "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)


def audit_processor_adapter(
    *,
    smolvla_report: Path,
    action_chunk: Path,
    execute_gate: Path | None,
    action_stats: Path,
    calibration: Path,
    model_path: Path,
    output: Path,
) -> dict[str, Any]:
    report_payload = _load_json(smolvla_report)
    action_payload = _load_json(action_chunk)
    execute_payload = _load_json(execute_gate) if execute_gate is not None else None
    stats_payload = load_action_stats(action_stats)
    official_stats = extract_policy_postprocessor_action_stats(
        model_id_or_path=str(model_path),
        local_files_only=True,
        action_stats_key="so100.buffer",
    )
    raw_chunk = action_payload.get("raw_action_chunk")
    if not isinstance(raw_chunk, list) or not raw_chunk:
        raise ValueError(f"{action_chunk} does not contain raw_action_chunk")
    action_tensor = torch.tensor(raw_chunk, dtype=torch.float32)

    saved_mean, saved_std = _stats_mean_std(stats_payload)
    official_mean, official_std = _stats_mean_std(official_stats)
    manual_postprocessed = action_tensor * saved_std + saved_mean
    official_direct = action_tensor * official_std + official_mean
    official_pipeline = _try_official_pipeline(
        model_path=model_path,
        action_tensor=action_tensor,
        mean=official_mean,
        std=official_std,
    )

    calibration_payload = load_calibration(calibration)
    adapter_raw = _postprocessed_to_raw_targets(
        postprocessed=manual_postprocessed,
        calibration=calibration_payload or {},
    )
    gate_comparison = _compare_execute_gate(
        execute_payload=execute_payload,
        manual_postprocessed=manual_postprocessed,
        adapter_raw=adapter_raw,
    )
    preproc_audit = _preprocessing_audit(report_payload=report_payload, model_path=model_path)
    stats_comparison = _compare_tensors(saved_mean, official_mean) | {
        f"std_{key}": value for key, value in _compare_tensors(saved_std, official_std).items()
    }

    result = {
        "status": _overall_status(
            stats_comparison=stats_comparison,
            official_pipeline=official_pipeline,
            gate_comparison=gate_comparison,
            preproc_audit=preproc_audit,
        ),
        "operation": "real_so100_smolvla_processor_adapter_audit",
        "inputs": {
            "smolvla_report": str(smolvla_report),
            "action_chunk": str(action_chunk),
            "execute_gate": str(execute_gate) if execute_gate is not None else None,
            "action_stats": str(action_stats),
            "calibration": str(calibration),
            "model_path": str(model_path),
        },
        "preprocessing": preproc_audit,
        "postprocessing": {
            "normalization_mode": "ACTION MEAN_STD",
            "manual_formula": "postprocessed_action = normalized_action * std + mean",
            "official_formula_source": "lerobot.processor.normalize_processor.UnnormalizerProcessorStep",
            "selected_action_stats_key": stats_payload.get("_selected_action_stats_key"),
            "official_selected_action_stats_key": official_stats.get("_selected_action_stats_key"),
            "stats_match_saved_vs_official": stats_comparison,
            "official_pipeline_match": official_pipeline,
            "postprocessed_summary": _tensor_summary(manual_postprocessed),
        },
        "action_adapter": {
            "joint_order": SO100_JOINT_ORDER,
            "assumed_action_semantics": "absolute_joint_position",
            "assumed_command_units": "lerobot_so100_position",
            "raw_conversion": "non-gripper: value_deg * 4095 / 360 + calibration_mid; gripper: percent over calibrated range",
            "adapter_raw_summary": _tensor_summary(adapter_raw),
            "execute_gate_comparison": gate_comparison,
            "remaining_risk": [
                "The saved postprocessor proves action unnormalization, not physical SO-100 command semantics.",
                "The real adapter still assumes postprocessed SO-100 actions are absolute LeRobot SO-100 positions.",
                "The local checkpoint does not include a saved policy_preprocessor.json, so state normalization cannot be replayed from a saved processor artifact.",
                "The model config expects three image feature names; current real setup supplies two cameras and duplicates one context image into the third feature.",
            ],
        },
        "conclusion": _conclusion(
            preproc_audit=preproc_audit,
            stats_comparison=stats_comparison,
            official_pipeline=official_pipeline,
            gate_comparison=gate_comparison,
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stats_mean_std(payload: dict[str, Any] | None) -> tuple[torch.Tensor, torch.Tensor]:
    if not payload or not isinstance(payload.get("action"), dict):
        raise ValueError("action stats payload must contain action.mean and action.std")
    mean = torch.tensor(_flatten(payload["action"]["mean"]), dtype=torch.float32)
    std = torch.tensor(_flatten(payload["action"]["std"]), dtype=torch.float32)
    return mean, std


def _flatten(value: Any) -> list[float]:
    if isinstance(value, dict) and "data" in value:
        return _flatten(value["data"])
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def _try_official_pipeline(
    *,
    model_path: Path,
    action_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, Any]:
    try:
        from lerobot.policies.smolvla import processor_smolvla  # noqa: F401 - registers steps.
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.processor.pipeline import DataProcessorPipeline

        expected = action_tensor * std + mean
        saved_pipeline = DataProcessorPipeline.from_pretrained(
            model_path,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        saved_output = saved_pipeline(action_tensor)
        saved_diff = (saved_output.detach().cpu().to(torch.float32) - expected).abs()

        override_pipeline = DataProcessorPipeline.from_pretrained(
            model_path,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            overrides={
                "unnormalizer_processor": {
                    "stats": {
                        "action": {
                            "mean": mean.detach().cpu().tolist(),
                            "std": std.detach().cpu().tolist(),
                        }
                    }
                }
            },
        )
        override_output = override_pipeline(action_tensor)
        override_diff = (override_output.detach().cpu().to(torch.float32) - expected).abs()
        saved_matches = float(saved_diff.max().item()) < 1e-6
        override_matches = float(override_diff.max().item()) < 1e-6
        return {
            "status": "passed" if override_matches else "mismatch",
            "loaded_policy_postprocessor": True,
            "saved_pipeline_without_stats_override": {
                "matches_direct_formula": saved_matches,
                "max_abs_error_vs_direct_formula": round(float(saved_diff.max().item()), 9),
                "mean_abs_error_vs_direct_formula": round(float(saved_diff.mean().item()), 9),
                "interpretation": (
                    "Saved local postprocessor state is dataset-prefixed, e.g. so100.buffer.action.mean, "
                    "so direct loading does not bind stats to the generic action key."
                ),
            },
            "pipeline_with_so100_buffer_stats_override": {
                "matches_direct_formula": override_matches,
                "max_abs_error_vs_direct_formula": round(float(override_diff.max().item()), 9),
                "mean_abs_error_vs_direct_formula": round(float(override_diff.mean().item()), 9),
            },
            "output_shape": list(override_output.shape),
        }
    except Exception as exc:  # noqa: BLE001 - audit should preserve incompatibilities.
        return {
            "status": "blocked",
            "loaded_policy_postprocessor": False,
            "blocker": f"{type(exc).__name__}: {exc}"[:1200],
        }


def _postprocessed_to_raw_targets(*, postprocessed: torch.Tensor, calibration: dict[str, Any]) -> torch.Tensor:
    rows: list[list[float]] = []
    for row in postprocessed.tolist():
        raw_row = []
        for joint, value in zip(SO100_JOINT_ORDER, row, strict=True):
            raw_row.append(_lerobot_so100_position_to_raw(joint=joint, value=float(value), calibration=calibration[joint]))
        rows.append(raw_row)
    return torch.tensor(rows, dtype=torch.float32)


def _compare_execute_gate(
    *,
    execute_payload: dict[str, Any] | None,
    manual_postprocessed: torch.Tensor,
    adapter_raw: torch.Tensor,
) -> dict[str, Any]:
    if execute_payload is None:
        return {"status": "skipped", "reason": "no execute_gate provided"}
    step_plans = ((execute_payload.get("dry_plan") or {}).get("step_plans") or [])
    if not step_plans:
        return {"status": "blocked", "blocker": "execute_gate has no dry_plan.step_plans"}
    gate_post: list[list[float]] = []
    gate_raw: list[list[float]] = []
    for step in step_plans:
        targets = step.get("joint_targets") or []
        gate_post.append([float(target["unnormalized_action_value"]) for target in targets])
        gate_raw.append([float(target["target_raw"]) for target in targets])
    gate_post_tensor = torch.tensor(gate_post, dtype=torch.float32)
    gate_raw_tensor = torch.tensor(gate_raw, dtype=torch.float32)
    post_diff = (gate_post_tensor - manual_postprocessed[: gate_post_tensor.shape[0]]).abs()
    raw_diff = (gate_raw_tensor - adapter_raw[: gate_raw_tensor.shape[0]]).abs()
    return {
        "status": "passed" if float(post_diff.max()) < 1e-4 and float(raw_diff.max()) < 1e-3 else "mismatch",
        "gate_status": execute_payload.get("status"),
        "gate_ready_for_execution": (execute_payload.get("dry_plan") or {}).get("ready_for_execution"),
        "max_abs_error_unnormalized_action": round(float(post_diff.max().item()), 9),
        "mean_abs_error_unnormalized_action": round(float(post_diff.mean().item()), 9),
        "max_abs_error_target_raw": round(float(raw_diff.max().item()), 9),
        "mean_abs_error_target_raw": round(float(raw_diff.mean().item()), 9),
    }


def _preprocessing_audit(*, report_payload: dict[str, Any], model_path: Path) -> dict[str, Any]:
    config = _load_json(model_path / "config.json") or {}
    batch_audit = report_payload.get("batch_audit") or {}
    image_keys = [key for key in batch_audit if key.startswith("observation.images.")]
    state = batch_audit.get("observation.state") or {}
    token_shape = report_payload.get("language_token_shape")
    local_preprocessor_exists = (model_path / "policy_preprocessor.json").exists()
    image_scale_ok = all(
        0.0 <= float((batch_audit[key] or {}).get("min", math.nan))
        and float((batch_audit[key] or {}).get("max", math.nan)) <= 1.0
        for key in image_keys
    )
    return {
        "policy_preprocessor_json_exists": local_preprocessor_exists,
        "image_feature_keys_present": image_keys,
        "image_input_scale_0_to_1_before_policy_internal_siglip_normalization": image_scale_ok,
        "policy_internal_image_resize_imgs_with_padding": config.get("resize_imgs_with_padding"),
        "policy_internal_image_normalization": "[0,1] -> [-1,1] inside SmolVLAPolicy.prepare_images",
        "language_token_shape": token_shape,
        "language_token_count": report_payload.get("language_token_count"),
        "state_units": report_payload.get("policy_state_units"),
        "state_tensor_summary": state,
        "model_normalization_mapping": config.get("normalization_mapping"),
        "model_image_features": sorted((config.get("input_features") or {}).keys()),
        "status": "attention" if not local_preprocessor_exists else "passed",
        "attention": "No saved policy_preprocessor.json is present in the local checkpoint; input state normalization is manual/current-code, not replayed from a saved processor artifact.",
    }


def _compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    diff = (left - right).abs()
    return {
        "shape": list(left.shape),
        "max_abs_error": round(float(diff.max().item()), 9),
        "mean_abs_error": round(float(diff.mean().item()), 9),
        "allclose": bool(torch.allclose(left, right, rtol=1e-6, atol=1e-6)),
    }


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    numeric = tensor.detach().cpu().to(torch.float32)
    return {
        "shape": list(numeric.shape),
        "min": round(float(numeric.min().item()), 6),
        "max": round(float(numeric.max().item()), 6),
        "mean": round(float(numeric.mean().item()), 6),
        "std": round(float(numeric.std(unbiased=False).item()), 6),
    }


def _overall_status(
    *,
    stats_comparison: dict[str, Any],
    official_pipeline: dict[str, Any],
    gate_comparison: dict[str, Any],
    preproc_audit: dict[str, Any],
) -> str:
    if not stats_comparison.get("allclose") or not stats_comparison.get("std_allclose"):
        return "failed"
    if official_pipeline.get("status") != "passed":
        return "attention"
    if gate_comparison.get("status") not in {"passed", "skipped"}:
        return "failed"
    if preproc_audit.get("status") == "attention":
        return "attention"
    return "passed"


def _conclusion(
    *,
    preproc_audit: dict[str, Any],
    stats_comparison: dict[str, Any],
    official_pipeline: dict[str, Any],
    gate_comparison: dict[str, Any],
) -> list[str]:
    lines = []
    if stats_comparison.get("allclose") and stats_comparison.get("std_allclose"):
        lines.append("Postprocessing stats are identical to the official local SmolVLA policy_postprocessor so100.buffer stats.")
    override = official_pipeline.get("pipeline_with_so100_buffer_stats_override") or {}
    saved = official_pipeline.get("saved_pipeline_without_stats_override") or {}
    if override.get("matches_direct_formula"):
        lines.append("LeRobot policy_postprocessor matches direct mean/std unnormalization when so100.buffer stats are explicitly bound to action stats.")
    if saved.get("matches_direct_formula") is False:
        lines.append("Directly loading the saved local policy_postprocessor is not enough because its stats are dataset-prefixed rather than bound to the generic action key.")
    if gate_comparison.get("status") == "passed":
        lines.append("The saved execute gate unnormalized actions and raw target conversion match the audited adapter formulas.")
    if preproc_audit.get("status") == "attention":
        lines.append("The remaining unresolved input-side risk is state/preprocessor replay: no saved policy_preprocessor.json exists in the local checkpoint.")
    lines.append("The highest-risk remaining assumption is action semantics/unit mapping, not image preprocessing or saved action postprocessing.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit real SO-100 SmolVLA preprocessing and adapter/postprocessor contracts.")
    parser.add_argument("--smolvla-report", type=Path, required=True)
    parser.add_argument("--action-chunk", type=Path, required=True)
    parser.add_argument("--execute-gate", type=Path)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_processor_adapter(
                smolvla_report=args.smolvla_report,
                action_chunk=args.action_chunk,
                execute_gate=args.execute_gate,
                action_stats=args.action_stats,
                calibration=args.calibration,
                model_path=args.model_path,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
