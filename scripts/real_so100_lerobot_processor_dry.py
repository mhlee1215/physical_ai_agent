#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from physical_ai_agent.safety.so100_action_gate import load_calibration
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import raw_to_lerobot_so100_position
from scripts.real_so100_smolvla_dry import SO100_STATE_ORDER


DEFAULT_LOCAL_MODEL = "/Users/minhaeng/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"


def run_lerobot_processor_dry(
    *,
    episode: Path,
    frame_index: int,
    output_dir: Path,
    instruction: str,
    model_id: str,
    policy_type: str,
    local_files_only: bool,
    device: str,
    calibration: Path,
    state_units: str,
    top_camera_index: str,
    wrist_camera_index: str,
    camera_top_name: str,
    camera_wrist_name: str,
    action_steps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "lerobot_processor_dry_report.json"
    action_path = output_dir / "smolvla_action_chunk.json"
    blocker_path = output_dir / "lerobot_processor_dry_blocker.md"
    started = perf_counter()
    report: dict[str, Any] = {
        "status": "blocked",
        "operation": "real_so100_lerobot_processor_dry",
        "method": "legalaspro_style_preprocessor_select_action_postprocessor",
        "episode": str(episode),
        "frame_index": frame_index,
        "instruction": instruction,
        "model_id": model_id,
        "policy_type": policy_type,
        "local_files_only": local_files_only,
        "device_requested": device,
        "calibration": str(calibration),
        "state_units": state_units,
        "camera_source_mapping": {
            top_camera_index: camera_top_name,
            wrist_camera_index: camera_wrist_name,
        },
        "policy_camera_indexes": [top_camera_index, wrist_camera_index],
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "requested_action_steps": action_steps,
        "report_path": str(report_path),
        "action_path": str(action_path),
        "blocker_path": str(blocker_path),
    }
    try:
        record = _load_episode_record(episode, frame_index)
        observation, obs_audit = _build_raw_observation(
            record=record,
            instruction=instruction,
            calibration_path=calibration,
            state_units=state_units,
            top_camera_index=top_camera_index,
            wrist_camera_index=wrist_camera_index,
            camera_top_name=camera_top_name,
            camera_wrist_name=camera_wrist_name,
        )
        runner, runner_audit = _load_runner(
            model_id=model_id,
            policy_type=policy_type,
            local_files_only=local_files_only,
            device=device,
        )
        runner.policy.reset()
        actions = []
        per_step_shapes = []
        for _index in range(action_steps):
            action = runner.select_action(observation)
            action_array = _to_numpy(action).reshape(-1).astype("float32")
            actions.append([float(item) for item in action_array.tolist()])
            per_step_shapes.append(list(action_array.shape))
        first_action = actions[0] if actions else []
        action_payload = {
            "instruction": instruction,
            "instruction_tokenized": "handled_by_lerobot_preprocessor",
            "raw_action": first_action,
            "raw_action_legacy_note": "First step only, preserved for compatibility. Use raw_action_chunk for execution planning.",
            "raw_action_chunk": actions,
            "first_action": first_action,
            "raw_action_dim": len(first_action),
            "raw_action_chunk_steps": len(actions),
            "planned_action_steps": len(actions),
            "executed_action_steps": len(actions),
            "action_chunk_semantics": (
                "legalaspro-style path: repeated policy.select_action calls with saved LeRobot "
                "preprocessor and postprocessor applied."
            ),
            "safe_to_execute": False,
            "note": "Dry-run only. This action was not sent to SO-100.",
        }
        action_path.write_text(json.dumps(action_payload, indent=2, sort_keys=True), encoding="utf-8")
        blocker_path.write_text("", encoding="utf-8")
        report.update(
            {
                "status": "passed",
                "runner_audit": runner_audit,
                "observation_audit": obs_audit,
                "raw_action_dim": len(first_action),
                "raw_action_chunk_steps": len(actions),
                "planned_action_steps": len(actions),
                "executed_action_steps": len(actions),
                "per_step_action_shapes": per_step_shapes,
                "action_preview": actions[:2],
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve model/runtime/processor blocker detail.
        blocker = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1600]
        trace = traceback.format_exc()
        blocker_path.write_text(
            "\n".join(
                [
                    "# Real SO-100 LeRobot Processor Dry-Run Blocker",
                    "",
                    f"- Model id: `{model_id}`",
                    f"- Policy type: `{policy_type}`",
                    f"- Local files only: `{local_files_only}`",
                    f"- Blocker: {blocker}",
                    "",
                    "No robot action was sent.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        report["blocker"] = blocker
        report["traceback"] = trace

    report["duration_s"] = round(perf_counter() - started, 4)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_runner(*, model_id: str, policy_type: str, local_files_only: bool, device: str):
    from physical_ai_agent.policies.lerobot_policy_runner import load_lerobot_policy_runner

    selected_device = _select_device(device)
    runner = load_lerobot_policy_runner(
        model_id,
        device=selected_device,
        policy_type=policy_type,
        local_files_only=local_files_only,
    )
    audit = {
        "device_selected": selected_device,
        "policy_config_device": str(getattr(runner.policy.config, "device", "unknown")),
        "processor_source": getattr(runner, "processor_source", "unknown"),
        "preprocessor_steps": [type(step).__name__ for step in getattr(runner.preprocessor, "steps", [])],
        "postprocessor_steps": [type(step).__name__ for step in getattr(runner.postprocessor, "steps", [])],
    }
    return runner, audit


def _select_device(device: str) -> str:
    import torch

    normalized = device.lower()
    if normalized == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if normalized == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return normalized


def _build_raw_observation(
    *,
    record: dict[str, Any],
    instruction: str,
    calibration_path: Path,
    state_units: str,
    top_camera_index: str,
    wrist_camera_index: str,
    camera_top_name: str,
    camera_wrist_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_dict = record["observation"]["state"]
    images = record["observation"]["images"]
    missing_state = [name for name in SO100_STATE_ORDER if name not in state_dict]
    if missing_state:
        raise ValueError(f"Episode record is missing SO-100 state keys: {missing_state}")
    missing_images = [
        index
        for index in [top_camera_index, wrist_camera_index]
        if index not in images or not images[index]
    ]
    if missing_images:
        raise ValueError(f"Episode record is missing policy camera image indexes: {missing_images}")

    calibration = load_calibration(calibration_path)
    state = _state_from_record(state_dict=state_dict, calibration=calibration, state_units=state_units)
    top_rgb = _read_rgb_uint8(Path(images[top_camera_index]))
    wrist_rgb = _read_rgb_uint8(Path(images[wrist_camera_index]))
    top_tensor = _image_to_lerobot_tensor(top_rgb)
    wrist_tensor = _image_to_lerobot_tensor(wrist_rgb)
    state_tensor = _vector_to_lerobot_tensor(state)
    observation = {
        "observation.state": state_tensor,
        f"observation.images.{camera_top_name}": top_tensor,
        f"observation.images.{camera_wrist_name}": wrist_tensor,
        "task": instruction,
    }
    return observation, {
        "state_shape": list(state.shape),
        "state_min": round(float(state.min()), 6),
        "state_max": round(float(state.max()), 6),
        "state_mean": round(float(state.mean()), 6),
        "state_units": state_units,
        "state_tensor": _tensor_summary(state_tensor),
        f"observation.images.{camera_top_name}": _tensor_summary(top_tensor),
        f"observation.images.{camera_wrist_name}": _tensor_summary(wrist_tensor),
        "task": instruction,
    }


def _state_from_record(*, state_dict: dict[str, Any], calibration: dict[str, Any] | None, state_units: str) -> np.ndarray:
    if state_units == "lerobot_so100_position":
        return np.asarray([float(state_dict[name]) for name in SO100_STATE_ORDER], dtype="float32")
    if state_units == "raw_ticks":
        if calibration is None:
            raise ValueError("--calibration is required when --state-units raw_ticks")
        return np.asarray(
            [
                raw_to_lerobot_so100_position(
                    joint=name,
                    raw_value=float(state_dict[name]),
                    calibration=calibration.get(name, {}),
                )
                for name in SO100_STATE_ORDER
            ],
            dtype="float32",
        )
    if state_units == "raw_ticks_unconverted":
        return np.asarray([float(state_dict[name]) for name in SO100_STATE_ORDER], dtype="float32")
    raise ValueError(f"Unsupported state_units={state_units!r}")


def _read_rgb_uint8(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    numeric = array.astype("float32")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": round(float(numeric.min()), 6),
        "max": round(float(numeric.max()), 6),
        "mean": round(float(numeric.mean()), 6),
    }


def _image_to_lerobot_tensor(array: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(array.copy()).float().div(255.0).permute(2, 0, 1).contiguous()


def _vector_to_lerobot_tensor(array: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(array.copy()).to(dtype=torch.float32).contiguous()


def _tensor_summary(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().cpu().to(dtype=value.dtype)
    numeric = tensor.to(dtype=tensor.dtype if str(tensor.dtype).startswith("torch.float") else torch.float32)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": round(float(numeric.min().item()), 6),
        "max": round(float(numeric.max().item()), 6),
        "mean": round(float(numeric.mean().item()), 6),
    }


def _to_numpy(value: Any) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_episode_record(path: Path, frame_index: int) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["frame_index"]) == frame_index:
            return record
    raise ValueError(f"frame_index={frame_index} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run real SO-100 through LeRobot saved pre/postprocessors, legalaspro-style.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick the green figure.")
    parser.add_argument("--model-id", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--policy-type", default="smolvla")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--state-units",
        default="raw_ticks",
        choices=["raw_ticks", "lerobot_so100_position", "raw_ticks_unconverted"],
    )
    parser.add_argument("--top-camera-index", default="1")
    parser.add_argument("--wrist-camera-index", default="0")
    parser.add_argument("--camera-top-name", default="camera1")
    parser.add_argument("--camera-wrist-name", default="camera2")
    parser.add_argument("--action-steps", type=int, default=15)
    args = parser.parse_args()
    print(
        json.dumps(
            run_lerobot_processor_dry(
                episode=args.episode,
                frame_index=args.frame_index,
                output_dir=args.output_dir,
                instruction=args.instruction,
                model_id=args.model_id,
                policy_type=args.policy_type,
                local_files_only=not args.allow_download,
                device=args.device,
                calibration=args.calibration,
                state_units=args.state_units,
                top_camera_index=args.top_camera_index,
                wrist_camera_index=args.wrist_camera_index,
                camera_top_name=args.camera_top_name,
                camera_wrist_name=args.camera_wrist_name,
                action_steps=args.action_steps,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
