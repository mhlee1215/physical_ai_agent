#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.utils import ACTION, build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

from scripts.real_so100_lerobot_processor_dry import DEFAULT_LOCAL_MODEL, _load_runner
from scripts.real_so100_motor_state_snapshot import read_motor_state_snapshot
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose
from scripts.real_so100_micro_step import _capture_visual, _probe_motion_video, _record_motion_video, _start_motion_video


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION_DIR = Path("_workspace/real_so100/calibration")
DEFAULT_ROBOT_ID = "so100_local"


def run_lerobot_native_task(
    *,
    output_dir: Path,
    instruction: str,
    steps: int,
    port: str,
    calibration_dir: Path,
    robot_id: str,
    model_id: str,
    policy_type: str,
    allow_download: bool,
    device: str,
    chunk_size: int | None,
    n_action_steps: int | None,
    num_steps: int | None,
    execute: bool,
    human_confirmed: bool,
    max_relative_target: float | None,
    use_degrees: bool,
    step_settle_seconds: float,
    visual_camera_index: int,
    video_fps: float,
    home_pose: Path,
    save_inference_inputs: bool,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report: dict[str, Any] = {
        "operation": "real_so100_lerobot_native_task",
        "started_at": started_at,
        "instruction": instruction,
        "steps_requested": steps,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "port": port,
        "calibration_dir": str(calibration_dir),
        "robot_id": robot_id,
        "model_id": model_id,
        "policy_type": policy_type,
        "device_requested": device,
        "chunk_size_requested": chunk_size,
        "n_action_steps_requested": n_action_steps,
        "num_steps_requested": num_steps,
        "policy_camera_indexes": [0, 1],
        "camera_mapping": {
            "camera1": "opencv index 1 wide context",
            "camera2": "opencv index 0 wrist",
        },
        "visual_camera_index": visual_camera_index,
        "save_inference_inputs": save_inference_inputs,
        "max_relative_target": max_relative_target,
        "use_degrees": use_degrees,
        "step_settle_seconds": step_settle_seconds,
        "per_step_torque_policy": "keep torque on until final home return",
        "send_action_called": False,
        "policy_actions_executed": False,
        "post_task_torque_disabled": False,
        "status": "ready" if execute else "dry_run",
        "blockers": [],
    }
    if execute and not human_confirmed:
        report["blockers"].append("Human confirmation flag is required for execution.")
        report["status"] = "blocked"
        _write_json(output_dir / "summary.json", report)
        return report
    if report["blockers"] or not execute:
        _write_json(output_dir / "summary.json", report)
        return report

    selected_device = _select_device(device)
    runner = None
    robot = None
    video_capture = None
    video_writer = None
    video_result = None
    physical_motion_sent = False
    home_return: dict[str, Any] | None = None
    final_snapshot: dict[str, Any] | None = None
    before_image_path = None
    try:
        load_started = time.perf_counter()
        runner, runner_audit = _load_runner(
            model_id=model_id,
            policy_type=policy_type,
            local_files_only=not allow_download,
            device=selected_device,
        )
        _override_policy_rollout_config(
            runner.policy,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            num_steps=num_steps,
        )
        report["runner_audit"] = dict(runner_audit, loaded_once=True)
        report["policy_rollout_config"] = _policy_rollout_config(runner.policy)
        report["runner_load_duration_s"] = round(time.perf_counter() - load_started, 4)
        runner.policy.reset()

        robot_cfg = SO100FollowerConfig(
            port=port,
            id=robot_id,
            calibration_dir=calibration_dir,
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=False,
            use_degrees=use_degrees,
            cameras={
                "camera1": OpenCVCameraConfig(index_or_path=1, width=1920, height=1080, fps=30),
                "camera2": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30),
            },
        )
        robot = SO100Follower(robot_cfg)
        robot.connect(calibrate=False)
        report["robot_observation_features"] = _jsonable_features(robot.observation_features)
        report["robot_action_features"] = _jsonable_features(robot.action_features)

        action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=False)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=False)
        ds_features = {**action_features, **obs_features}
        report["dataset_action_names"] = ds_features[ACTION]["names"]
        report["dataset_feature_keys"] = sorted(ds_features)
        report["robot_connected"] = True

        before_visual = _capture_visual(
            camera_index=visual_camera_index,
            output_dir=output_dir / "visual",
            label="before",
            before_path=None,
        )
        before_image_path = Path(before_visual["image_path"])
        report["visual_check"] = {"before": before_visual}
        video_capture, video_writer, video_result = _start_motion_video(
            camera_index=visual_camera_index,
            output_dir=output_dir / "visual",
            fps=video_fps,
        )
        report["motion_video"] = video_result

        executed_steps: list[dict[str, Any]] = []
        action_queue_refills: list[dict[str, Any]] = []
        last_saved_inference_images: dict[str, Path] | None = None
        for step_index in range(steps):
            observation = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=observation,
                ds_features=ds_features,
                device=torch.device(selected_device),
                task=instruction,
                robot_type="so100_follower",
            )
            processed = runner.preprocessor(obs_frame)
            processed = _move_tensors_to_policy_device(processed, runner.policy)
            queue_before = _action_queue_length(runner.policy)
            with torch.inference_mode():
                policy_action = runner.policy.select_action(processed)
            queue_after_select = _action_queue_length(runner.policy)
            queue_refilled = queue_before == 0
            if queue_refilled:
                inference_input_record = None
                if save_inference_inputs:
                    inference_input_record = _save_inference_input_images(
                        observation=observation,
                        output_dir=output_dir / "inference_inputs",
                        step_index=step_index,
                        previous_paths=last_saved_inference_images,
                    )
                    last_saved_inference_images = {
                        camera_key: Path(item["image_path"])
                        for camera_key, item in inference_input_record["images"].items()
                    }
                action_queue_refills.append(
                    {
                        "step_index": step_index,
                        "queue_before_select": queue_before,
                        "queue_after_select": queue_after_select,
                        "n_action_steps": getattr(getattr(runner.policy, "config", None), "n_action_steps", None),
                        "chunk_size": getattr(getattr(runner.policy, "config", None), "chunk_size", None),
                        "inference_input": inference_input_record,
                    }
                )
            postprocessed = runner.postprocessor(policy_action)
            robot_action = make_robot_action(postprocessed, ds_features)
            sent_action = robot.send_action(robot_action)
            physical_motion_sent = True
            report["send_action_called"] = True
            report["policy_actions_executed"] = True
            executed_steps.append(
                {
                    "step_index": step_index,
                    "observation_state": {
                        key: float(value)
                        for key, value in observation.items()
                        if key.endswith(".pos") and isinstance(value, (int, float))
                    },
                    "policy_action": _tensor_to_list(postprocessed),
                    "robot_action": robot_action,
                    "sent_action": sent_action,
                    "queue_before_select": queue_before,
                    "queue_after_select": queue_after_select,
                    "queue_refilled_this_step": queue_refilled,
                }
            )
            if video_capture is not None and video_writer is not None and video_result is not None:
                _record_motion_video(
                    capture=video_capture,
                    writer=video_writer,
                    result=video_result,
                    duration_seconds=step_settle_seconds,
                    fps=video_fps,
                )
            else:
                time.sleep(step_settle_seconds)

        report["executed_steps"] = executed_steps
        report["action_queue_refills"] = action_queue_refills
        report["executed_action_steps"] = len(executed_steps)
        if executed_steps:
            report["first_sent_action"] = executed_steps[0]["sent_action"]
            report["last_sent_action"] = executed_steps[-1]["sent_action"]
        after_visual = _capture_visual(
            camera_index=visual_camera_index,
            output_dir=output_dir / "visual",
            label="after",
            before_path=before_image_path,
        )
        report.setdefault("visual_check", {})["after"] = after_visual
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - hardware report should preserve exact failure.
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        if video_writer is not None:
            video_writer.release()
        if video_capture is not None:
            video_capture.release()
        if isinstance(report.get("motion_video"), dict):
            report["motion_video"].update(_probe_motion_video(Path(report["motion_video"]["path"])))
        try:
            if robot is not None and robot.is_connected:
                robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            report["robot_disconnect_error"] = repr(exc)

        if execute and physical_motion_sent:
            home_return = move_to_home_pose(
                port=port,
                calibration=calibration_dir / f"{robot_id}.json",
                home_pose=home_pose,
                output=output_dir / "task_home_return" / "report.json",
                execute=True,
                human_confirmed=human_confirmed,
                workspace_clear_confirmed=human_confirmed,
                max_abs_delta_raw=80.0,
                step_settle_seconds=step_settle_seconds,
                camera_index=visual_camera_index,
                visual_output_dir=output_dir / "task_home_return" / "visual",
                record_video=True,
                video_fps=video_fps,
            )
            report["task_home_return"] = {
                "path": str(output_dir / "task_home_return" / "report.json"),
                "status": home_return.get("status"),
                "post_task_torque_disabled": home_return.get("post_task_torque_disabled"),
                "executed_action_steps": home_return.get("executed_action_steps"),
            }
            report["post_task_torque_disabled"] = bool(home_return.get("post_task_torque_disabled"))

        final_snapshot = read_motor_state_snapshot(
            port=port,
            calibration=calibration_dir / f"{robot_id}.json",
            output=output_dir / "post_task_motor_state_snapshot.json",
        )
        report["final_motor_snapshot"] = str(output_dir / "post_task_motor_state_snapshot.json")
        report["final_torque"] = {
            name: state.get("Torque_Enable")
            for name, state in (final_snapshot or {}).get("motors", {}).items()
        }
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(output_dir / "summary.json", report)
    return report


def _select_device(device: str) -> str:
    normalized = device.lower()
    if normalized == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if normalized == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return normalized


def _move_tensors_to_policy_device(payload: dict[str, Any], policy: Any) -> dict[str, Any]:
    device = getattr(getattr(policy, "config", None), "device", None)
    if device is None:
        return payload
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in payload.items()}


def _override_policy_rollout_config(
    policy: Any,
    *,
    chunk_size: int | None,
    n_action_steps: int | None,
    num_steps: int | None,
) -> None:
    config = getattr(policy, "config", None)
    if config is None:
        return
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        config.chunk_size = int(chunk_size)
    if n_action_steps is not None:
        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
        chunk_size = getattr(config, "chunk_size", None)
        if chunk_size is not None and n_action_steps > int(chunk_size):
            raise ValueError(f"n_action_steps={n_action_steps} exceeds chunk_size={chunk_size}")
        config.n_action_steps = int(n_action_steps)
    if num_steps is not None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        config.num_steps = int(num_steps)
    if hasattr(policy, "reset"):
        policy.reset()


def _policy_rollout_config(policy: Any) -> dict[str, Any]:
    config = getattr(policy, "config", None)
    return {
        "chunk_size": getattr(config, "chunk_size", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "num_steps": getattr(config, "num_steps", None),
    }


def _action_queue_length(policy: Any) -> int | None:
    try:
        from lerobot.policies.utils import ACTION

        queue = getattr(policy, "_queues", {}).get(ACTION)
        return len(queue)
    except Exception:  # noqa: BLE001
        return None


def _save_inference_input_images(
    *,
    observation: dict[str, Any],
    output_dir: Path,
    step_index: int,
    previous_paths: dict[str, Path] | None,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    step_dir = output_dir / f"step_{step_index:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Any] = {}
    for camera_key in sorted(key for key, value in observation.items() if _looks_like_image(value)):
        array = np.asarray(observation[camera_key])
        image_path = step_dir / f"{camera_key}.jpg"
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if array.ndim == 3 and array.shape[-1] == 3 else array
        cv2.imwrite(str(image_path), bgr)
        previous_path = (previous_paths or {}).get(camera_key)
        diff = _image_diff_from_path(array, previous_path) if previous_path is not None else None
        images[camera_key] = {
            "image_path": str(image_path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "mean": round(float(array.astype("float32").mean()), 6),
            "min": round(float(array.min()), 6),
            "max": round(float(array.max()), 6),
            "diff_from_previous_inference": diff,
        }
    return {
        "step_index": step_index,
        "image_count": len(images),
        "images": images,
    }


def _looks_like_image(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) == 3 and int(shape[-1]) in {1, 3, 4}


def _image_diff_from_path(array_rgb: Any, previous_path: Path | None) -> dict[str, Any] | None:
    if previous_path is None or not previous_path.exists():
        return None
    import cv2
    import numpy as np

    previous_bgr = cv2.imread(str(previous_path), cv2.IMREAD_COLOR)
    if previous_bgr is None:
        return {"previous_path": str(previous_path), "readable": False}
    previous_rgb = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2RGB)
    current = np.asarray(array_rgb)
    if previous_rgb.shape != current.shape:
        return {
            "previous_path": str(previous_path),
            "readable": True,
            "shape_mismatch": [list(previous_rgb.shape), list(current.shape)],
        }
    diff = np.abs(previous_rgb.astype("int16") - current.astype("int16"))
    return {
        "previous_path": str(previous_path),
        "readable": True,
        "mean_absdiff": round(float(diff.mean()), 4),
        "changed_pixel_ratio_gt_10": round(float((diff > 10).mean()), 6),
    }


def _tensor_to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in value]


def _jsonable_features(features: dict[str, Any]) -> dict[str, Any]:
    return {key: list(value) if isinstance(value, tuple) else str(value) for key, value in features.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmolVLA on real SO-100 through LeRobot-native robot APIs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick the green figure.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--model-id", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--policy-type", default="smolvla")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--max-relative-target", type=float)
    parser.add_argument("--use-degrees", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--visual-camera-index", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--home-pose", type=Path, default=DEFAULT_HOME_POSE)
    parser.add_argument(
        "--save-inference-inputs",
        action="store_true",
        help="Save camera images at each action-queue refill, i.e. each fresh SmolVLA inference.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_lerobot_native_task(
                output_dir=args.output_dir,
                instruction=args.instruction,
                steps=args.steps,
                port=args.port,
                calibration_dir=args.calibration_dir,
                robot_id=args.robot_id,
                model_id=args.model_id,
                policy_type=args.policy_type,
                allow_download=args.allow_download,
                device=args.device,
                chunk_size=args.chunk_size,
                n_action_steps=args.n_action_steps,
                num_steps=args.num_steps,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                max_relative_target=args.max_relative_target,
                use_degrees=args.use_degrees,
                step_settle_seconds=args.step_settle_seconds,
                visual_camera_index=args.visual_camera_index,
                video_fps=args.video_fps,
                home_pose=args.home_pose,
                save_inference_inputs=args.save_inference_inputs,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
