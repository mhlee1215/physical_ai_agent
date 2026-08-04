#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from physical_ai_agent.policies.lerobot_policy_runner import load_lerobot_policy_runner
from physical_ai_agent.real_so100.sim_policy_bridge import (
    JOINT_ORDER,
    clamp_hardware_positions,
    hardware_position_limits_from_calibration,
    hardware_positions_to_sim_qpos,
    sim_qpos_to_hardware_positions,
)


def run_chunk(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute or not args.human_confirmed or not args.workspace_clear_confirmed:
        raise RuntimeError("--execute, --human-confirmed, and --workspace-clear-confirmed are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "operation": "real_so101_smolvla_bounded_chunk",
        "prompt": args.prompt,
        "checkpoint": str(args.checkpoint),
        "device": "mps",
        "camera_contract": {"camera1": "opencv_0_egocentric", "camera2": "opencv_1_wrist"},
        "policy_image_size": [256, 256],
        "policy_image_resize_mode": args.image_resize_mode,
        "max_steps": args.steps,
        "n_action_steps": args.n_action_steps,
        "max_relative_target": args.max_relative_target,
        "human_direct_observer": True,
        "send_action_called": False,
        "policy_actions_executed": False,
        "post_task_torque_disabled": False,
        "status": "starting",
    }
    robot = None
    start_positions: dict[str, float] | None = None
    start_raw: dict[str, float] | None = None
    motion_sent = False
    frames: list[Image.Image] = []
    try:
        runner = load_lerobot_policy_runner(str(args.checkpoint), device="mps", local_files_only=True)
        runner.policy.config.n_action_steps = args.n_action_steps
        runner.policy.reset()
        report["model_parameter_devices"] = sorted({str(parameter.device) for parameter in runner.policy.parameters()})
        report["processor_source"] = runner.processor_source

        robot = SO101Follower(
            SO101FollowerConfig(
                port=args.port,
                id=args.robot_id,
                calibration_dir=args.calibration_dir,
                disable_torque_on_disconnect=True,
                use_degrees=True,
                max_relative_target=args.max_relative_target,
                cameras={
                    "camera1": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30),
                    "camera2": OpenCVCameraConfig(index_or_path=1, width=1920, height=1080, fps=30),
                },
            )
        )
        calibration_path = args.calibration_dir / f"{args.robot_id}.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        hardware_limits = hardware_position_limits_from_calibration(calibration)
        report["hardware_position_limits"] = {
            joint: {"min": limits[0], "max": limits[1]} for joint, limits in hardware_limits.items()
        }
        if args.expected_start_sim_qpos is None and not args.allow_unverified_start_state:
            raise RuntimeError(
                "real policy execution requires --expected-start-sim-qpos; "
                "use --allow-unverified-start-state only for an explicitly labeled diagnostic"
            )
        if args.expected_start_sim_qpos is not None:
            expected_start_hardware = sim_qpos_to_hardware_positions(args.expected_start_sim_qpos)
            _safe_expected_start, expected_start_clip = clamp_hardware_positions(
                expected_start_hardware, hardware_limits
            )
            report["expected_start_sim_qpos"] = args.expected_start_sim_qpos
            report["expected_start_hardware"] = expected_start_hardware
            report["expected_start_calibration_clip"] = expected_start_clip
            if expected_start_clip:
                raise RuntimeError(
                    "expected training start is outside the active hardware calibration: "
                    f"{expected_start_clip}"
                )
        start_raw, range_report = _safe_connect_at_current_pose(
            robot,
            calibration_path=calibration_path,
            range_tolerance_raw=args.start_range_tolerance_raw,
        )
        report["start_calibration_range"] = range_report
        first_observation = robot.get_observation()
        start_positions = _positions(first_observation)
        report["readback_before_raw"] = start_raw
        report["readback_before_hardware"] = start_positions
        if args.expected_start_sim_qpos is not None:
            start_contract = _start_state_contract_report(
                start_positions,
                args.expected_start_sim_qpos,
                max_arm_error_degrees=args.start_state_max_arm_error_degrees,
                max_gripper_error_percent=args.start_state_max_gripper_error_percent,
            )
            report["start_state_contract"] = start_contract
            if not start_contract["passed"]:
                raise RuntimeError(f"hardware start does not match the training start: {start_contract}")
        for camera in ("camera1", "camera2"):
            _policy_image(first_observation[camera], resize_mode=args.image_resize_mode).save(
                args.output_dir / f"policy_input_start_{camera}.png"
            )

        executed = []
        for step_index in range(args.steps):
            observation = first_observation if step_index == 0 else robot.get_observation()
            current = _positions(observation)
            policy_observation = _policy_observation(
                observation,
                args.prompt,
                image_resize_mode=args.image_resize_mode,
            )
            trace = runner.select_action_with_trace(policy_observation)
            fresh_inference = _is_fresh_inference_step(step_index, args.n_action_steps)
            sim_action = torch.as_tensor(trace["action"]).detach().cpu().reshape(-1).tolist()
            requested = sim_qpos_to_hardware_positions(sim_action)
            calibration_clipped, calibration_clip = clamp_hardware_positions(requested, hardware_limits)
            relative_bounded = {
                joint: min(
                    current[joint] + args.max_relative_target,
                    max(current[joint] - args.max_relative_target, calibration_clipped[joint]),
                )
                for joint in JOINT_ORDER
            }
            bounded, post_relative_calibration_clip = clamp_hardware_positions(relative_bounded, hardware_limits)
            robot_action = {f"{joint}.pos": bounded[joint] for joint in JOINT_ORDER}
            sent = robot.send_action(robot_action)
            motion_sent = True
            report["send_action_called"] = True
            report["policy_actions_executed"] = True
            executed.append(
                {
                    "step_index": step_index,
                    "fresh_inference": fresh_inference,
                    "state_hardware": current,
                    "state_sim_qpos": hardware_positions_to_sim_qpos(current),
                    "policy_action_sim_qpos": sim_action,
                    "requested_hardware": requested,
                    "calibration_clipped_hardware": calibration_clipped,
                    "calibration_clip": calibration_clip,
                    "post_relative_calibration_clip": post_relative_calibration_clip,
                    "bounded_hardware": bounded,
                    "sent_action": sent,
                }
            )
            frames.append(
                _annotated_frame(
                    observation,
                    step_index,
                    args.prompt,
                    fresh_inference=fresh_inference,
                    image_resize_mode=args.image_resize_mode,
                )
            )
            time.sleep(args.step_settle_seconds)

        report["executed_steps"] = executed
        report["executed_action_steps"] = len(executed)
        report["inference_count"] = sum(int(step["fresh_inference"]) for step in executed)
        report["readback_after_policy_raw"] = {
            key: float(value) for key, value in robot.bus.sync_read("Present_Position", normalize=False).items()
        }
        report["status"] = "policy_chunk_executed"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        if robot is not None and robot.is_connected:
            if motion_sent and start_positions is not None:
                try:
                    report["return_to_start"] = _return_to_start(
                        robot,
                        start_positions,
                        max_delta=args.max_relative_target,
                        settle=args.step_settle_seconds,
                        tolerance=args.return_tolerance,
                        max_steps=args.return_max_steps,
                        hardware_limits=hardware_limits,
                    )
                except Exception as exc:  # noqa: BLE001
                    report["return_to_start"] = {"status": "failed", "error": repr(exc)}
            try:
                report["readback_final_raw"] = {
                    key: float(value) for key, value in robot.bus.sync_read("Present_Position", normalize=False).items()
                }
            except Exception as exc:  # noqa: BLE001
                report["final_readback_error"] = repr(exc)
            robot.disconnect()
            report["post_task_torque_disabled"] = True
        if frames:
            gif = args.output_dir / "camera1_camera2_rollout.gif"
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1000 // 12, loop=0)
            report["rollout_gif"] = str(gif)
        report["start_raw_return_target"] = start_raw
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _positions(observation: dict[str, Any]) -> dict[str, float]:
    return {joint: float(observation[f"{joint}.pos"]) for joint in JOINT_ORDER}


def _start_state_contract_report(
    actual_hardware: dict[str, float],
    expected_sim_qpos: list[float],
    *,
    max_arm_error_degrees: float,
    max_gripper_error_percent: float,
) -> dict[str, Any]:
    expected_hardware = sim_qpos_to_hardware_positions(expected_sim_qpos)
    errors = {joint: abs(float(actual_hardware[joint]) - expected_hardware[joint]) for joint in JOINT_ORDER}
    violations = [
        joint
        for joint in JOINT_ORDER[:-1]
        if errors[joint] > max_arm_error_degrees
    ]
    if errors["gripper"] > max_gripper_error_percent:
        violations.append("gripper")
    return {
        "passed": not violations,
        "expected_hardware": expected_hardware,
        "actual_hardware": actual_hardware,
        "absolute_errors": errors,
        "max_arm_error_degrees": max_arm_error_degrees,
        "max_gripper_error_percent": max_gripper_error_percent,
        "violations": violations,
    }


def _parse_sim_qpos(value: str) -> list[float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected JSON array with six qpos values") from exc
    if not isinstance(parsed, list) or len(parsed) != len(JOINT_ORDER):
        raise argparse.ArgumentTypeError(f"expected JSON array with {len(JOINT_ORDER)} qpos values")
    try:
        qpos = [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("qpos values must be numeric") from exc
    if not all(np.isfinite(qpos)):
        raise argparse.ArgumentTypeError("qpos values must be finite")
    return qpos


def _safe_connect_at_current_pose(
    robot: SO101Follower,
    *,
    calibration_path: Path,
    range_tolerance_raw: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    robot.bus.connect()
    raw = {key: float(value) for key, value in robot.bus.sync_read("Present_Position", normalize=False).items()}
    range_report = _calibration_range_report(calibration_path, raw, tolerance_raw=range_tolerance_raw)
    if not range_report["safe"]:
        raise RuntimeError(f"start pose outside calibration tolerance: {range_report['violations']}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    preload_raw = {
        joint: int(
            round(
                min(
                    float(calibration[joint]["range_max"]),
                    max(float(calibration[joint]["range_min"]), raw[joint]),
                )
            )
        )
        for joint in JOINT_ORDER
    }
    range_report["preloaded_goal_raw"] = preload_raw
    range_report["preload_clipped_joints"] = [joint for joint in JOINT_ORDER if preload_raw[joint] != round(raw[joint])]
    robot.bus.sync_write("Goal_Position", preload_raw, normalize=False)
    for camera in robot.cameras.values():
        camera.connect()
    robot.configure()
    return raw, range_report


def _calibration_range_report(path: Path, positions: dict[str, float], *, tolerance_raw: float) -> dict[str, Any]:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    joints = {}
    violations = []
    for joint in JOINT_ORDER:
        value = float(positions[joint])
        lower = float(calibration[joint]["range_min"])
        upper = float(calibration[joint]["range_max"])
        outside = max(lower - value, value - upper, 0.0)
        joints[joint] = {"raw": value, "range_min": lower, "range_max": upper, "outside_raw": outside}
        if outside > tolerance_raw:
            violations.append(joint)
    return {"safe": not violations, "tolerance_raw": tolerance_raw, "violations": violations, "joints": joints}


def _is_fresh_inference_step(step_index: int, n_action_steps: int) -> bool:
    if n_action_steps <= 0:
        raise ValueError("n_action_steps must be positive")
    return step_index % n_action_steps == 0


def _policy_observation(
    observation: dict[str, Any],
    prompt: str,
    *,
    image_resize_mode: str = "center_crop",
) -> dict[str, Any]:
    return {
        "observation.state": torch.tensor(hardware_positions_to_sim_qpos(_positions(observation)), dtype=torch.float32),
        "observation.images.camera1": _image_tensor(observation["camera1"], resize_mode=image_resize_mode),
        "observation.images.camera2": _image_tensor(observation["camera2"], resize_mode=image_resize_mode),
        "task": prompt,
    }


def _policy_image(array: Any, *, resize_mode: str = "center_crop") -> Image.Image:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    if resize_mode == "center_crop":
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    elif resize_mode != "stretch":
        raise ValueError(f"unsupported image resize mode: {resize_mode}")
    return image.resize((256, 256), Image.Resampling.LANCZOS)


def _image_tensor(array: Any, *, resize_mode: str = "center_crop") -> torch.Tensor:
    image = _policy_image(array, resize_mode=resize_mode)
    return torch.from_numpy(np.asarray(image).copy()).float().div(255.0).permute(2, 0, 1).contiguous()


def _annotated_frame(
    observation: dict[str, Any],
    step_index: int,
    prompt: str,
    *,
    fresh_inference: bool,
    image_resize_mode: str = "center_crop",
) -> Image.Image:
    images = [
        _policy_image(observation[key], resize_mode=image_resize_mode) for key in ("camera1", "camera2")
    ]
    canvas = Image.new("RGB", (512, 292), "black")
    canvas.paste(images[0], (0, 36)); canvas.paste(images[1], (256, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"episode 0 | frame {step_index:03d} | inference {'yes' if fresh_inference else 'no'}", fill="lime")
    if fresh_inference:
        draw.rectangle((1, 1, 510, 290), outline="lime", width=4)
    draw.text((8, 274), prompt, fill="white")
    return canvas


def _return_to_start(
    robot: SO101Follower,
    target: dict[str, float],
    *,
    max_delta: float,
    settle: float,
    tolerance: float,
    max_steps: int,
    hardware_limits: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    safe_target, target_clip = clamp_hardware_positions(target, hardware_limits)
    steps = []
    for index in range(max_steps):
        current = _positions(robot.get_observation())
        if max(abs(safe_target[joint] - current[joint]) for joint in JOINT_ORDER) <= tolerance:
            return {
                "status": "passed",
                "steps": steps,
                "final_hardware": current,
                "requested_target": target,
                "safe_target": safe_target,
                "target_clip": target_clip,
            }
        relative_command = {
            joint: min(current[joint] + max_delta, max(current[joint] - max_delta, safe_target[joint]))
            for joint in JOINT_ORDER
        }
        command, command_clip = clamp_hardware_positions(relative_command, hardware_limits)
        sent = robot.send_action({f"{joint}.pos": command[joint] for joint in JOINT_ORDER})
        steps.append({"step_index": index, "sent_action": sent, "calibration_clip": command_clip})
        time.sleep(settle)
    raise RuntimeError(f"return-to-start exceeded {max_steps} bounded steps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", default="/dev/cu.usbmodem5AE60824791")
    parser.add_argument("--robot-id", default="so100_local")
    parser.add_argument("--calibration-dir", type=Path, default=Path("_workspace/real_so100/calibration"))
    parser.add_argument("--prompt", default="grip the green cube and lift")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--n-action-steps", type=int, default=15)
    parser.add_argument("--max-relative-target", type=float, default=2.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--start-range-tolerance-raw", type=float, default=10.0)
    parser.add_argument("--return-tolerance", type=float, default=1.5)
    parser.add_argument("--return-max-steps", type=int, default=200)
    parser.add_argument("--image-resize-mode", choices=("center_crop", "stretch"), default="center_crop")
    parser.add_argument("--expected-start-sim-qpos", type=_parse_sim_qpos)
    parser.add_argument("--start-state-max-arm-error-degrees", type=float, default=5.0)
    parser.add_argument("--start-state-max-gripper-error-percent", type=float, default=10.0)
    parser.add_argument("--allow-unverified-start-state", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_chunk(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
