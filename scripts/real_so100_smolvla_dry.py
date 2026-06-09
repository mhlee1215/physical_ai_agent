#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from physical_ai_agent.safety.so100_action_gate import load_calibration
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import raw_to_lerobot_so100_position

SO100_STATE_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

PROMPT_WIRING_NOTE = (
    "Dry-run tokenizes the instruction with the SmolVLA VLM tokenizer and feeds those language "
    "tokens alongside real camera images and SO-100 state. Action remains non-executable until "
    "safety clipping and human confirmation gates are added."
)


def _load_episode_record(path: Path, frame_index: int) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["frame_index"]) == frame_index:
            return record
    raise ValueError(f"frame_index={frame_index} not found in {path}")


def _read_rgb(path: Path):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def run_dry_inference(
    *,
    episode: Path,
    frame_index: int,
    output_dir: Path,
    instruction: str,
    model_id: str,
    local_files_only: bool,
    wrist_camera_index: str = "0",
    egocentric_camera_index: str = "1",
    observer_camera_indexes: list[str] | None = None,
    action_steps: int = 10,
    calibration: Path | None = None,
    state_units: str = "raw_ticks",
    device: str = "auto",
) -> dict[str, Any]:
    from physical_ai_agent.policies.smolvla_real import (
        _build_batch_for_policy,
        _clip_action,
        _load_pretrained_policy,
        _policy_device_metadata,
        _tensor_to_float_list,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if action_steps < 1:
        raise ValueError(f"action_steps must be positive, got {action_steps}")
    report_path = output_dir / "smolvla_dry_report.json"
    action_path = output_dir / "smolvla_action_chunk.json"
    blocker_path = output_dir / "smolvla_dry_blocker.md"
    started = perf_counter()
    record = _load_episode_record(episode, frame_index)
    state_dict = record["observation"]["state"]
    missing_state = [name for name in SO100_STATE_ORDER if name not in state_dict]
    if missing_state:
        raise ValueError(f"Episode record is missing SO-100 state keys: {missing_state}")
    calibration_payload = load_calibration(calibration)
    state, policy_state_units = _policy_state_from_record(
        state_dict=state_dict,
        calibration=calibration_payload,
        state_units=state_units,
    )
    image_paths = record["observation"]["images"]
    missing_images = [
        index
        for index in [wrist_camera_index, egocentric_camera_index]
        if index not in image_paths or not image_paths[index]
    ]
    if missing_images:
        raise ValueError(f"Episode record is missing SmolVLA camera image indexes: {missing_images}")
    camera_pixels = {
        "wrist_cam": _read_rgb(Path(image_paths[wrist_camera_index])),
        "egocentric_cam": _read_rgb(Path(image_paths[egocentric_camera_index])),
    }
    observer_camera_indexes = observer_camera_indexes or []
    result: dict[str, Any] = {
        "status": "blocked",
        "model_id": model_id,
        "local_files_only": local_files_only,
        "device_requested": device,
        "device_selected": "not_loaded",
        "device_probe": {},
        "device_fallback_reason": None,
        "episode": str(episode),
        "frame_index": frame_index,
        "instruction": instruction,
        "instruction_tokenized": False,
        "prompt_wiring_status": "not_attempted",
        "prompt_wiring_note": PROMPT_WIRING_NOTE,
        "actuation_enabled": False,
        "policy_actions_executed": False,
        "send_action_called": False,
        "camera_source_mapping": {
            wrist_camera_index: "wrist_cam",
            egocentric_camera_index: "egocentric_cam",
        },
        "policy_camera_indexes": [wrist_camera_index, egocentric_camera_index],
        "observer_camera_indexes": observer_camera_indexes,
        "observer_camera_role": "codex_debug_only_not_smolvla_input",
        "episode_state_units": state_units,
        "policy_state_units": policy_state_units,
        "calibration": str(calibration) if calibration else None,
        "requested_action_steps": action_steps,
        "report_path": str(report_path),
        "action_path": str(action_path),
        "blocker_path": str(blocker_path),
    }

    try:
        policy = _load_pretrained_policy(
            model_id=model_id,
            local_files_only=local_files_only,
            device=device,
        )
        result.update(_policy_device_metadata(policy))
        batch, image_feature_mapping = _build_batch_for_policy(
            policy,
            state,
            camera_pixels,
            instruction=instruction,
            local_files_only=local_files_only,
        )
        batch_audit = _summarize_policy_batch(batch)
        input_image_audit = {
            name: _summarize_numpy_image(pixels)
            for name, pixels in camera_pixels.items()
        }
        language_tokens = batch["observation.language.tokens"]
        language_attention_mask = batch["observation.language.attention_mask"]
        language_token_count = int(language_attention_mask.detach().cpu().sum().item())
        raw_action_chunk = policy.predict_action_chunk(batch)
        predicted_chunk_size = int(raw_action_chunk.shape[1])
        if action_steps > predicted_chunk_size:
            raise ValueError(f"action_steps must be in [1, {predicted_chunk_size}], got {action_steps}")
        action_chunk = [
            _tensor_to_float_list(raw_action_chunk[:, index, :])
            for index in range(action_steps)
        ]
        first_action = action_chunk[0]
        action_dim = len(first_action)
        clipped_chunk = [_clip_action(action, action_dim) for action in action_chunk]
        action_payload = {
            "instruction": instruction,
            "instruction_tokenized": True,
            "language_token_shape": list(language_tokens.shape),
            "language_token_count": language_token_count,
            "raw_action": first_action,
            "raw_action_legacy_note": "First step only, preserved for compatibility. Use raw_action_chunk for execution planning.",
            "raw_action_chunk": action_chunk,
            "first_action": first_action,
            "raw_action_dim": action_dim,
            "raw_action_chunk_steps": len(action_chunk),
            "predicted_chunk_size": predicted_chunk_size,
            "planned_action_steps": len(action_chunk),
            "executed_action_steps": len(action_chunk),
            "action_chunk_semantics": "SmolVLA predicts an action chunk; real execution must consume chunk steps, not one isolated action.",
            "safe_to_execute": False,
            "note": "Dry-run only. This action was not sent to SO-100.",
            "prompt_wiring_note": PROMPT_WIRING_NOTE,
        }
        action_path.write_text(json.dumps(action_payload, indent=2, sort_keys=True), encoding="utf-8")
        blocker_path.write_text("", encoding="utf-8")
        result.update(
            {
                "status": "passed",
                "raw_action_dim": action_dim,
                "raw_action_chunk_steps": len(action_chunk),
                "predicted_chunk_size": predicted_chunk_size,
                "planned_action_steps": len(action_chunk),
                "executed_action_steps": len(action_chunk),
                "action_preview": clipped_chunk[: min(2, len(clipped_chunk))],
                "image_feature_mapping": image_feature_mapping,
                "input_image_audit": input_image_audit,
                "batch_audit": batch_audit,
                "instruction_tokenized": True,
                "language_token_shape": list(language_tokens.shape),
                "language_token_count": language_token_count,
                "prompt_wiring_status": "tokenized_instruction",
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve model/runtime blocker detail.
        blocker = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1200]
        blocker_path.write_text(
            "\n".join(
                [
                    "# Real SO-100 SmolVLA Dry-Run Blocker",
                    "",
                    f"- Model id: `{model_id}`",
                    f"- Local files only: `{local_files_only}`",
                    f"- Blocker: {blocker}",
                    "",
                    "No robot action was sent.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        result["blocker"] = blocker

    result["duration_s"] = round(perf_counter() - started, 4)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _policy_state_from_record(
    *,
    state_dict: dict[str, Any],
    calibration: dict[str, Any] | None,
    state_units: str,
) -> tuple[list[float], str]:
    if state_units == "lerobot_so100_position":
        return [float(state_dict[name]) for name in SO100_STATE_ORDER], "lerobot_so100_position"
    if calibration:
        return [
            raw_to_lerobot_so100_position(
                joint=name,
                raw_value=float(state_dict[name]),
                calibration=calibration.get(name, {}),
            )
            for name in SO100_STATE_ORDER
        ], "lerobot_so100_position"
    return [float(state_dict[name]) for name in SO100_STATE_ORDER], "raw_ticks_unconverted"


def _summarize_policy_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _summarize_tensor(value)
        for key, value in sorted(batch.items())
        if hasattr(value, "detach")
    }


def _summarize_tensor(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().cpu()
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    if tensor.numel() == 0:
        return summary
    if str(tensor.dtype).startswith("torch.bool"):
        numeric = tensor.to(dtype=torch.float32)
    elif str(tensor.dtype).startswith("torch.int") or str(tensor.dtype).startswith("torch.long"):
        numeric = tensor.to(dtype=torch.float32)
    else:
        numeric = tensor.to(dtype=torch.float32)
    summary.update(
        {
            "min": round(float(numeric.min().item()), 6),
            "max": round(float(numeric.max().item()), 6),
            "mean": round(float(numeric.mean().item()), 6),
            "std": round(float(numeric.std(unbiased=False).item()), 6),
            "nonzero_ratio": round(float((numeric != 0).float().mean().item()), 6),
        }
    )
    return summary


def _summarize_numpy_image(pixels: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(pixels)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size == 0:
        return summary
    numeric = array.astype("float32")
    summary.update(
        {
            "min": round(float(numeric.min()), 6),
            "max": round(float(numeric.max()), 6),
            "mean": round(float(numeric.mean()), 6),
            "std": round(float(numeric.std()), 6),
            "nonzero_ratio": round(float((numeric != 0).mean()), 6),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run SmolVLA on real SO-100 observation frames.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick up the green Android figure.")
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--wrist-camera-index", default="0")
    parser.add_argument("--egocentric-camera-index", default="1")
    parser.add_argument("--observer-camera-index", action="append", default=[])
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--state-units", default="raw_ticks", choices=["raw_ticks", "lerobot_so100_position"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    print(
        json.dumps(
            run_dry_inference(
                episode=args.episode,
                frame_index=args.frame_index,
                output_dir=args.output_dir,
                instruction=args.instruction,
                model_id=args.model_id,
                local_files_only=not args.allow_download,
                wrist_camera_index=args.wrist_camera_index,
                egocentric_camera_index=args.egocentric_camera_index,
                observer_camera_indexes=args.observer_camera_index,
                action_steps=args.action_steps,
                calibration=args.calibration,
                state_units=args.state_units,
                device=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
