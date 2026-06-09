#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.real_so100_execute_legalaspro_chunk import execute_legalaspro_chunk
from scripts.real_so100_lerobot_processor_dry import (
    DEFAULT_LOCAL_MODEL,
    _build_raw_observation,
    _load_episode_record,
    _load_runner,
    _to_numpy,
)
from scripts.real_so100_motor_state_snapshot import read_motor_state_snapshot
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose
from scripts.real_so100_observe import record_observation


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")


def run_legalaspro_iterations(
    *,
    output_dir: Path,
    instruction: str,
    iterations: int,
    port: str,
    calibration: Path,
    model_id: str,
    policy_type: str,
    allow_download: bool,
    device: str,
    state_units: str,
    action_steps: int,
    max_abs_delta_raw: float,
    step_settle_seconds: float,
    visual_camera_index: int,
    observe_duration_seconds: float,
    observe_fps: float,
    video_fps: float,
    home_pose: Path,
    execute: bool,
    human_confirmed: bool,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    iteration_reports: list[dict[str, Any]] = []
    physical_motion_sent = False
    task_failed = False
    task_error = None
    home_return: dict[str, Any] | None = None
    final_snapshot: dict[str, Any] | None = None
    runner_audit: dict[str, Any] | None = None
    runner_load_duration_s: float | None = None

    try:
        load_started = time.perf_counter()
        runner, runner_audit = _load_runner(
            model_id=model_id,
            policy_type=policy_type,
            local_files_only=not allow_download,
            device=device,
        )
        runner_load_duration_s = round(time.perf_counter() - load_started, 4)
        for index in range(iterations):
            iter_dir = output_dir / f"iter_{index + 1:03d}"
            pre_observe = record_observation(
                port=port,
                camera_indexes=[0, 1],
                output_dir=iter_dir / "observe",
                duration_seconds=observe_duration_seconds,
                fps=observe_fps,
                task=instruction,
                calibration_file=calibration,
                camera_roles={"0": "wrist_cam", "1": "wide_context_cam"},
                policy_camera_indexes=[0, 1],
                observer_camera_indexes=[],
            )
            iter_report: dict[str, Any] = {
                "iteration": index + 1,
                "pre_observe": pre_observe,
                "physical_robot_motion": False,
            }
            iteration_reports.append(iter_report)
            if not pre_observe.get("ok"):
                iter_report["status"] = "failed"
                iter_report["error"] = "pre_observe_failed"
                task_failed = True
                break

            smolvla = _run_lerobot_processor_dry_with_loaded_runner(
                runner=runner,
                runner_audit=runner_audit,
                episode=Path(str(pre_observe["episode_jsonl"])),
                frame_index=0,
                output_dir=iter_dir / "smolvla",
                instruction=instruction,
                model_id=model_id,
                policy_type=policy_type,
                local_files_only=not allow_download,
                device=device,
                calibration=calibration,
                state_units=state_units,
                top_camera_index="1",
                wrist_camera_index="0",
                camera_top_name="camera1",
                camera_wrist_name="camera2",
                action_steps=action_steps,
            )
            iter_report["smolvla"] = {
                "status": smolvla.get("status"),
                "report_path": smolvla.get("report_path"),
                "action_path": smolvla.get("action_path"),
                "raw_action_chunk_steps": smolvla.get("raw_action_chunk_steps"),
                "device_selected": (smolvla.get("runner_audit") or {}).get("device_selected"),
            }
            if smolvla.get("status") != "passed":
                iter_report["status"] = "failed"
                iter_report["error"] = "smolvla_failed"
                task_failed = True
                break

            execute_report = execute_legalaspro_chunk(
                action=Path(str(smolvla["action_path"])),
                output=iter_dir / "legalaspro_execute.json",
                port=port,
                calibration=calibration,
                execute=execute,
                human_confirmed=human_confirmed,
                action_steps=action_steps,
                max_abs_delta_raw=max_abs_delta_raw,
                step_settle_seconds=step_settle_seconds,
                camera_index=visual_camera_index,
                visual_output_dir=iter_dir / "visual",
                record_video=True,
                video_fps=video_fps,
                keep_torque_on_after_run=True,
            )
            iter_report["legalaspro_execute"] = {
                "path": str(iter_dir / "legalaspro_execute.json"),
                "status": execute_report.get("status"),
                "send_action_called": execute_report.get("send_action_called"),
                "policy_actions_executed": execute_report.get("policy_actions_executed"),
                "executed_action_steps": execute_report.get("executed_action_steps"),
                "observed_delta_raw": execute_report.get("observed_delta_raw"),
                "post_task_torque_disabled": execute_report.get("post_task_torque_disabled"),
                "torque_kept_on_for_home_return": execute_report.get("torque_kept_on_for_home_return"),
                "motion_video": (execute_report.get("motion_video") or {}).get("path"),
            }
            iter_report["physical_robot_motion"] = bool(execute_report.get("send_action_called"))
            physical_motion_sent = physical_motion_sent or iter_report["physical_robot_motion"]
            iter_report["status"] = execute_report.get("status")
            if execute_report.get("status") != "passed":
                task_failed = True
                break
    except Exception as exc:  # noqa: BLE001 - preserve recovery-triggering failure.
        task_failed = True
        task_error = repr(exc)
    finally:
        if execute and physical_motion_sent:
            home_return = move_to_home_pose(
                port=port,
                calibration=calibration,
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
        final_snapshot = read_motor_state_snapshot(
            port=port,
            calibration=calibration,
            output=output_dir / "post_task_motor_state_snapshot.json",
        )

    summary = {
        "status": "failed" if task_failed else "passed",
        "operation": "real_so100_legalaspro_iterative_task",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "instruction": instruction,
        "iterations_requested": iterations,
        "iterations_completed": len(iteration_reports),
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "policy_camera_indexes": [0, 1],
        "observer_camera_indexes": [],
        "visual_camera_index": visual_camera_index,
        "action_steps_per_iteration": action_steps,
        "max_abs_delta_raw": max_abs_delta_raw,
        "per_iteration_torque_policy": "keep torque on until final home return",
        "home_pose": str(home_pose),
        "task_error": task_error,
        "runner_loaded_once": runner_audit is not None,
        "runner_load_duration_s": runner_load_duration_s,
        "runner_audit": runner_audit,
        "task_home_return": {
            "path": str(output_dir / "task_home_return" / "report.json") if home_return else None,
            "status": home_return.get("status") if home_return else None,
            "post_task_torque_disabled": home_return.get("post_task_torque_disabled") if home_return else None,
            "executed_action_steps": home_return.get("executed_action_steps") if home_return else None,
        },
        "final_motor_snapshot": str(output_dir / "post_task_motor_state_snapshot.json"),
        "final_torque": {
            name: state.get("Torque_Enable")
            for name, state in (final_snapshot or {}).get("motors", {}).items()
        },
        "iteration_reports": iteration_reports,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _run_lerobot_processor_dry_with_loaded_runner(
    *,
    runner: Any,
    runner_audit: dict[str, Any],
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
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "blocked",
        "operation": "real_so100_lerobot_processor_dry",
        "method": "legalaspro_style_preprocessor_select_action_postprocessor_loaded_once_runner",
        "episode": str(episode),
        "frame_index": frame_index,
        "instruction": instruction,
        "model_id": model_id,
        "policy_type": policy_type,
        "local_files_only": local_files_only,
        "device_requested": device,
        "calibration": str(calibration),
        "state_units": state_units,
        "runner_reused": True,
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
                "preprocessor and postprocessor applied; policy runner loaded once and reused."
            ),
            "safe_to_execute": False,
            "note": "Dry-run proposal only. The separate execution step sends the chunk to SO-100.",
        }
        action_path.write_text(json.dumps(action_payload, indent=2, sort_keys=True), encoding="utf-8")
        blocker_path.write_text("", encoding="utf-8")
        report.update(
            {
                "status": "passed",
                "runner_audit": dict(runner_audit, loaded_once=True),
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
        blocker_path.write_text(
            "\n".join(
                [
                    "# Real SO-100 LeRobot Processor Dry-Run Blocker",
                    "",
                    f"- Model id: `{model_id}`",
                    f"- Policy type: `{policy_type}`",
                    f"- Local files only: `{local_files_only}`",
                    f"- Runner reused: `true`",
                    f"- Blocker: {blocker}",
                    "",
                    "No robot action was sent.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        report["blocker"] = blocker

    report["duration_s"] = round(time.perf_counter() - started, 4)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legalaspro-style SmolVLA iterations on real SO-100.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick the green figure.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--model-id", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--policy-type", default="smolvla")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--state-units",
        default="raw_ticks",
        choices=["raw_ticks", "lerobot_so100_position", "raw_ticks_unconverted"],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--action-steps", type=int, default=15)
    parser.add_argument("--max-abs-delta-raw", type=float, default=99999.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--visual-camera-index", type=int, default=1)
    parser.add_argument("--observe-duration-seconds", type=float, default=0.45)
    parser.add_argument("--observe-fps", type=float, default=2.0)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--home-pose", type=Path, default=DEFAULT_HOME_POSE)
    args = parser.parse_args()
    print(
        json.dumps(
            run_legalaspro_iterations(
                output_dir=args.output_dir,
                instruction=args.instruction,
                iterations=args.iterations,
                port=args.port,
                calibration=args.calibration,
                model_id=args.model_id,
                policy_type=args.policy_type,
                allow_download=args.allow_download,
                device=args.device,
                state_units=args.state_units,
                action_steps=args.action_steps,
                max_abs_delta_raw=args.max_abs_delta_raw,
                step_settle_seconds=args.step_settle_seconds,
                visual_camera_index=args.visual_camera_index,
                observe_duration_seconds=args.observe_duration_seconds,
                observe_fps=args.observe_fps,
                video_fps=args.video_fps,
                home_pose=args.home_pose,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
