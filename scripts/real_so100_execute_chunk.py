#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import (
    SO100_JOINT_ORDER,
    load_action_chunk_payload,
    load_calibration,
)
from physical_ai_agent.safety.so100_command_adapter import (
    build_so100_command_chunk_plan,
)
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import (
    build_so100_smolvla_metadata_command_chunk_plan,
    load_action_stats,
    load_smolvla_config,
)
from scripts.real_so100_micro_step import (
    _capture_visual,
    _make_so100_bus,
    _probe_motion_video,
    _record_motion_video,
    _start_motion_video,
)


def execute_action_chunk(
    *,
    port: str,
    action: Path,
    output: Path,
    calibration: Path | None,
    execute: bool,
    human_confirmed: bool,
    experimental_adapter_confirmed: bool,
    action_steps: int,
    delta_scale_raw_ticks: float,
    max_abs_delta_raw: float,
    step_settle_seconds: float,
    camera_index: int | None,
    visual_output_dir: Path | None,
    record_video: bool,
    video_fps: float,
    metadata_config: Path | None = None,
    action_stats: Path | None = None,
    action_semantics: str | None = None,
    gripper_semantics: str | None = None,
    command_units: str | None = None,
    confirm_so100_joint_order: bool = False,
    allow_deprecated_raw_tick_scaling: bool = False,
) -> dict[str, Any]:
    if action_steps < 1:
        raise ValueError(f"action_steps must be positive, got {action_steps}")
    if delta_scale_raw_ticks <= 0:
        raise ValueError(f"delta_scale_raw_ticks must be positive, got {delta_scale_raw_ticks}")
    if max_abs_delta_raw <= 0:
        raise ValueError(f"max_abs_delta_raw must be positive, got {max_abs_delta_raw}")
    if step_settle_seconds < 0:
        raise ValueError(f"step_settle_seconds must be non-negative, got {step_settle_seconds}")

    action_chunk = load_action_chunk_payload(action, action_steps=action_steps)
    calibration_payload = load_calibration(calibration)
    scale = {joint: delta_scale_raw_ticks for joint in SO100_JOINT_ORDER}
    max_delta = {joint: max_abs_delta_raw for joint in SO100_JOINT_ORDER}
    config_payload = load_smolvla_config(metadata_config) if metadata_config is not None else None
    stats_payload = load_action_stats(action_stats)
    confirmed_joint_order = SO100_JOINT_ORDER if confirm_so100_joint_order else None
    if config_payload is not None:
        dry_plan = build_so100_smolvla_metadata_command_chunk_plan(
            action_chunk=action_chunk,
            current_state={joint: 0 for joint in SO100_JOINT_ORDER},
            calibration=calibration_payload,
            config=config_payload,
            model_id="lerobot/smolvla_base",
            stats=stats_payload,
            action_semantics=action_semantics,
            joint_order=confirmed_joint_order,
            gripper_semantics=gripper_semantics,
            command_units=command_units,
        )
    else:
        dry_plan = build_so100_command_chunk_plan(
            action_chunk=action_chunk,
            current_state={joint: 0 for joint in SO100_JOINT_ORDER},
            calibration=calibration_payload,
            human_confirmed=human_confirmed,
            adapter_semantics_confirmed=experimental_adapter_confirmed,
            delta_scale_raw_ticks=scale,
            max_delta_raw_ticks=max_delta,
        )
    blockers: list[str] = []
    if not allow_deprecated_raw_tick_scaling and config_payload is None:
        blockers.append(
            "Deprecated raw tick scaling is disabled. Provide --metadata-config, --action-stats, "
            "--action-semantics, --gripper-semantics, and --confirm-so100-joint-order before executing SmolVLA chunks."
        )
    if config_payload is not None and not dry_plan.ready_for_execution:
        blockers.extend(dry_plan.blockers)
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if allow_deprecated_raw_tick_scaling and not experimental_adapter_confirmed:
        blockers.append("Experimental adapter confirmation is required before executing SmolVLA raw chunks.")
    if execute:
        if not record_video:
            blockers.append("Executed real robot chunk movements must pass --record-video.")
        if camera_index is None:
            blockers.append("Executed real robot chunk movements must pass --camera-index.")
        if visual_output_dir is None:
            blockers.append("Executed real robot chunk movements must pass --visual-output-dir.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_execute_smolvla_action_chunk",
        "port": port,
        "action_path": str(action),
        "calibration": str(calibration) if calibration else None,
        "action_chunk_steps": len(action_chunk),
        "requested_action_steps": action_steps,
        "delta_scale_raw_ticks": delta_scale_raw_ticks,
        "max_abs_delta_raw": max_abs_delta_raw,
        "allow_deprecated_raw_tick_scaling": allow_deprecated_raw_tick_scaling,
        "metadata_config": str(metadata_config) if metadata_config else None,
        "action_stats": str(action_stats) if action_stats else None,
        "action_semantics": action_semantics,
        "gripper_semantics": gripper_semantics,
        "command_units": command_units,
        "confirm_so100_joint_order": confirm_so100_joint_order,
        "step_settle_seconds": step_settle_seconds,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "experimental_adapter_confirmed": experimental_adapter_confirmed,
        "send_action_called": False,
        "policy_actions_executed": False,
        "writes_intended": execute,
        "disconnect_disable_torque": True,
        "camera_index": camera_index,
        "record_video_requested": record_video,
        "dry_plan_schema": type(dry_plan).__name__,
        "dry_plan": asdict(dry_plan),
        "blockers": blockers,
        "status": "blocked" if blockers else "ready",
    }

    if not execute:
        report["status"] = "dry_run"
        report["notes"] = ["No serial write was attempted. Re-run with --execute plus confirmations to actuate."]
        _write_json(output, report)
        return report
    if blockers:
        report["notes"] = ["Execution was requested but blocked before connecting to the robot."]
        _write_json(output, report)
        return report

    bus, _motors = _make_so100_bus(port)
    video_capture = None
    video_writer = None
    video_result: dict[str, Any] | None = None
    try:
        before_image_path = None
        if camera_index is not None and visual_output_dir is not None:
            before_visual = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="before",
                before_path=None,
            )
            before_image_path = Path(before_visual["image_path"])
            report["visual_check"] = {"before": before_visual}

        bus.connect(handshake=True)
        before_state = {joint: float(value) for joint, value in bus.sync_read("Present_Position", normalize=False).items()}
        report["readback_before_raw"] = before_state
        if config_payload is not None:
            chunk_plan = build_so100_smolvla_metadata_command_chunk_plan(
                action_chunk=action_chunk,
                current_state=before_state,
                calibration=calibration_payload,
                config=config_payload,
                model_id="lerobot/smolvla_base",
                stats=stats_payload,
                action_semantics=action_semantics,
                joint_order=confirmed_joint_order,
                gripper_semantics=gripper_semantics,
                command_units=command_units,
            )
            if not chunk_plan.ready_for_execution:
                raise RuntimeError(f"metadata adapter blocked execution: {chunk_plan.blockers}")
        else:
            chunk_plan = build_so100_command_chunk_plan(
                action_chunk=action_chunk,
                current_state=before_state,
                calibration=calibration_payload,
                human_confirmed=human_confirmed,
                adapter_semantics_confirmed=experimental_adapter_confirmed,
                delta_scale_raw_ticks=scale,
                max_delta_raw_ticks=max_delta,
            )
        report["execution_plan"] = asdict(chunk_plan)

        if record_video and camera_index is not None and visual_output_dir is not None:
            video_capture, video_writer, video_result = _start_motion_video(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                fps=video_fps,
            )
            report["motion_video"] = video_result

        executed_steps: list[dict[str, Any]] = []
        for index, step_plan in enumerate(chunk_plan.step_plans):
            target = _target_from_step_plan(step_plan)
            if set(target) != set(SO100_JOINT_ORDER):
                raise RuntimeError(f"step {index} did not produce all joint targets")
            write_normalize = _write_normalize_from_step_plan(step_plan)
            bus.sync_write("Goal_Position", target, normalize=write_normalize, num_retry=3)
            report["send_action_called"] = True
            executed_steps.append(
                {
                    "step_index": index,
                    "target_command": target,
                    "write_normalize": write_normalize,
                    "target_raw_estimate": _raw_estimate_from_step_plan(step_plan),
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

        after_state = {joint: float(value) for joint, value in bus.sync_read("Present_Position", normalize=False).items()}
        report["executed_steps"] = executed_steps
        report["executed_action_steps"] = len(executed_steps)
        report["readback_after_raw"] = after_state
        report["observed_delta_raw"] = {
            joint: round(after_state[joint] - before_state[joint], 4)
            for joint in SO100_JOINT_ORDER
        }
        report["policy_actions_executed"] = True
        if camera_index is not None and visual_output_dir is not None:
            after_visual = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
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
            if bus.is_connected:
                bus.disconnect(disable_torque=True)
                report["disconnect_disable_torque"] = True
                report["post_task_torque_disabled"] = True
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)
            report["post_task_torque_disabled"] = False

    _write_json(output, report)
    return report


def _target_from_step_plan(step_plan: Any) -> dict[str, float | int]:
    if hasattr(step_plan, "joint_targets"):
        return {
            joint_target.joint: (
                float(joint_target.target_command_value)
                if getattr(joint_target, "write_normalize", False)
                else int(round(joint_target.target_raw))
            )
            for joint_target in step_plan.joint_targets
            if math.isfinite(joint_target.target_raw)
        }
    return {
        joint_plan.joint: int(round(joint_plan.target_raw))
        for joint_plan in step_plan.joint_plans
        if math.isfinite(joint_plan.target_raw)
    }


def _write_normalize_from_step_plan(step_plan: Any) -> bool:
    if hasattr(step_plan, "joint_targets"):
        return any(bool(getattr(joint_target, "write_normalize", False)) for joint_target in step_plan.joint_targets)
    return False


def _raw_estimate_from_step_plan(step_plan: Any) -> dict[str, float]:
    if hasattr(step_plan, "joint_targets"):
        return {
            joint_target.joint: float(joint_target.target_raw)
            for joint_target in step_plan.joint_targets
            if math.isfinite(joint_target.target_raw)
        }
    return {
        joint_plan.joint: float(joint_plan.target_raw)
        for joint_plan in step_plan.joint_plans
        if math.isfinite(joint_plan.target_raw)
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a limited SO-100 SmolVLA action chunk with camera evidence.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--experimental-adapter-confirmed", action="store_true")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--delta-scale-raw-ticks", type=float, default=2.0)
    parser.add_argument("--max-abs-delta-raw", type=float, default=4.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.15)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--metadata-config", type=Path)
    parser.add_argument("--action-stats", type=Path)
    parser.add_argument("--action-semantics", choices=["absolute_joint_position", "joint_delta"])
    parser.add_argument("--gripper-semantics", choices=["higher_raw_opens", "higher_raw_closes"])
    parser.add_argument("--command-units", choices=["feetech_raw_ticks", "lerobot_so100_position"])
    parser.add_argument("--confirm-so100-joint-order", action="store_true")
    parser.add_argument("--allow-deprecated-raw-tick-scaling", action="store_true")
    args = parser.parse_args()

    print(
        json.dumps(
            execute_action_chunk(
                port=args.port,
                action=args.action,
                output=args.output,
                calibration=args.calibration,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                experimental_adapter_confirmed=args.experimental_adapter_confirmed,
                action_steps=args.action_steps,
                delta_scale_raw_ticks=args.delta_scale_raw_ticks,
                max_abs_delta_raw=args.max_abs_delta_raw,
                step_settle_seconds=args.step_settle_seconds,
                camera_index=args.camera_index,
                visual_output_dir=args.visual_output_dir,
                record_video=args.record_video,
                video_fps=args.video_fps,
                metadata_config=args.metadata_config,
                action_stats=args.action_stats,
                action_semantics=args.action_semantics,
                gripper_semantics=args.gripper_semantics,
                command_units=args.command_units,
                confirm_so100_joint_order=args.confirm_so100_joint_order,
                allow_deprecated_raw_tick_scaling=args.allow_deprecated_raw_tick_scaling,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
