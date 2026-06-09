#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_micro_step import (
    _capture_visual,
    _make_so100_bus,
    _probe_motion_video,
    _record_motion_video,
    _start_motion_video,
)


def execute_transition_packet(
    *,
    packet: Path,
    output: Path,
    port: str,
    execute: bool = False,
    human_confirmed: bool = False,
    workspace_clear_confirmed: bool = False,
    observer_camera_index: int = 3,
    visual_output_dir: Path | None = None,
    record_video: bool = False,
    video_fps: float = 12.0,
    step_settle_seconds: float = 0.15,
) -> dict[str, Any]:
    payload = json.loads(packet.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") or []
    blockers = _blockers(
        payload=payload,
        chunks=chunks,
        execute=execute,
        human_confirmed=human_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
        observer_camera_index=observer_camera_index,
        visual_output_dir=visual_output_dir,
        record_video=record_video,
    )
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_execute_transition_packet",
        "packet": str(packet),
        "port": port,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "observer_camera_index": observer_camera_index,
        "visual_output_dir": str(visual_output_dir) if visual_output_dir else None,
        "record_video_requested": record_video,
        "video_fps": video_fps,
        "step_settle_seconds": step_settle_seconds,
        "packet_status": payload.get("status"),
        "packet_execution_ready": bool(payload.get("execution_ready")),
        "transition_chunk_count": len(chunks),
        "transition_step_count": sum(len(chunk.get("steps") or []) for chunk in chunks),
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "writes_intended": execute,
        "disconnect_disable_torque": True,
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
    try:
        before_image_path = None
        if visual_output_dir is not None:
            before_visual = _capture_visual(
                camera_index=observer_camera_index,
                output_dir=visual_output_dir,
                label="before",
                before_path=None,
            )
            before_image_path = Path(before_visual["image_path"])
            report["visual_check"] = {"before": before_visual}
        bus.connect(handshake=True)
        before_state = {joint: float(value) for joint, value in bus.sync_read("Present_Position", normalize=False).items()}
        report["readback_before_raw"] = before_state
        if record_video and visual_output_dir is not None:
            video_capture, video_writer, video_result = _start_motion_video(
                camera_index=observer_camera_index,
                output_dir=visual_output_dir,
                fps=video_fps,
            )
            report["motion_video"] = video_result
        executed_steps = []
        for chunk in chunks:
            for step in chunk.get("steps") or []:
                target = {joint: float(step["target_command"][joint]) for joint in SO100_JOINT_ORDER}
                bus.sync_write("Goal_Position", target, normalize=bool(step.get("write_normalize", True)), num_retry=3)
                report["send_action_called"] = True
                executed_steps.append(
                    {
                        "chunk_index": int(chunk.get("chunk_index", 0)),
                        "step_index": int(step.get("step_index", 0)),
                        "step_index_in_chunk": int(step.get("step_index_in_chunk", 0)),
                        "target_command": target,
                        "write_normalize": bool(step.get("write_normalize", True)),
                        "target_raw_estimate": step.get("target_raw_estimate"),
                    }
                )
                if video_capture is not None and video_writer is not None and isinstance(report.get("motion_video"), dict):
                    _record_motion_video(
                        capture=video_capture,
                        writer=video_writer,
                        result=report["motion_video"],
                        duration_seconds=step_settle_seconds,
                        fps=video_fps,
                    )
                else:
                    time.sleep(step_settle_seconds)
        after_state = {joint: float(value) for joint, value in bus.sync_read("Present_Position", normalize=False).items()}
        report["executed_steps"] = executed_steps
        report["executed_action_steps"] = len(executed_steps)
        report["readback_after_raw"] = after_state
        report["observed_delta_raw"] = {joint: round(after_state[joint] - before_state[joint], 4) for joint in SO100_JOINT_ORDER}
        report["policy_actions_executed"] = True
        report["physical_robot_motion"] = True
        if visual_output_dir is not None:
            report.setdefault("visual_check", {})["after"] = _capture_visual(
                camera_index=observer_camera_index,
                output_dir=visual_output_dir,
                label="after",
                before_path=before_image_path,
            )
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


def _blockers(
    *,
    payload: dict[str, Any],
    chunks: list[dict[str, Any]],
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
    observer_camera_index: int,
    visual_output_dir: Path | None,
    record_video: bool,
) -> list[str]:
    blockers = []
    if payload.get("status") != "ready_for_observer_backed_execution" or not payload.get("execution_ready"):
        blockers.append("Transition execution packet is not ready.")
    if payload.get("send_action_called") or payload.get("physical_robot_motion"):
        blockers.append("Transition execution packet must be no-actuation input evidence.")
    if observer_camera_index != 3:
        blockers.append("Observer camera index 3 is required for this real SO-100 loop.")
    if not chunks:
        blockers.append("Transition execution packet has no chunks.")
    for chunk in chunks:
        steps = chunk.get("steps") or []
        if len(steps) != 10:
            blockers.append(f"Execution chunk {chunk.get('chunk_index')} has {len(steps)} steps, expected 10.")
        for step in steps:
            target = step.get("target_command") or {}
            missing = [joint for joint in SO100_JOINT_ORDER if joint not in target]
            if missing:
                blockers.append(f"Step {step.get('step_index')} missing target joints {missing}.")
    if execute:
        if not human_confirmed:
            blockers.append("Human confirmation flag is required.")
        if not workspace_clear_confirmed:
            blockers.append("Workspace-clear confirmation flag is required.")
        if not record_video:
            blockers.append("Observer-backed execution must pass --record-video.")
        if visual_output_dir is None:
            blockers.append("Observer-backed execution must pass --visual-output-dir.")
    return blockers


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute or dry-run a gated SO-100 transition execution packet.")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="/dev/cu.usbmodem5AE60824791")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.15)
    args = parser.parse_args()
    print(
        json.dumps(
            execute_transition_packet(
                packet=args.packet,
                output=args.output,
                port=args.port,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                observer_camera_index=args.observer_camera_index,
                visual_output_dir=args.visual_output_dir,
                record_video=args.record_video,
                video_fps=args.video_fps,
                step_settle_seconds=args.step_settle_seconds,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
