#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.real_so100_execute_chunk import execute_action_chunk
from scripts.real_so100_execute_clipped_chunk import execute_clipped_chunk
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose
from scripts.real_so100_observe import record_observation
from scripts.run_external_mps_smolvla_dry import run_external_mps_smolvla_dry


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")
DEFAULT_CONFIG = Path(
    "/Users/minhaeng/.cache/huggingface/hub/models--lerobot--smolvla_base/"
    "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/config.json"
)
DEFAULT_ACTION_STATS = Path(
    "_workspace/real_so100/smolvla_proposal_move_right_u20cam_observer_off_001/"
    "action_metadata/policy_postprocessor_action_stats_so100_buffer.json"
)


def run_baseline_iterations(
    *,
    output_dir: Path,
    instruction: str,
    iterations: int,
    port: str,
    calibration: Path,
    metadata_config: Path,
    action_stats: Path,
    execute: bool,
    human_confirmed: bool,
    action_steps: int,
    max_abs_delta_raw: float,
    step_settle_seconds: float,
    visual_camera_index: int,
    observe_duration_seconds: float,
    observe_fps: float,
    video_fps: float,
    home_pose: Path,
    return_home_after_task: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    iteration_reports = []
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
            camera_roles={"0": "wrist_cam", "1": "egocentric_cam"},
            policy_camera_indexes=[0, 1],
            observer_camera_indexes=[],
        )
        iter_report: dict[str, Any] = {
            "iteration": index + 1,
            "pre_observe": pre_observe,
            "physical_robot_motion": False,
        }
        if not pre_observe.get("ok"):
            iter_report["status"] = "failed"
            iter_report["error"] = "pre_observe_failed"
            iteration_reports.append(iter_report)
            break

        episode = Path(str(pre_observe["episode_jsonl"]))
        smolvla = run_external_mps_smolvla_dry(
            episode=episode,
            frame_index=0,
            output_dir=iter_dir / "smolvla",
            instruction=instruction,
            calibration=calibration,
            action_steps=action_steps,
            device="auto",
            state_units="raw_ticks",
        )
        iter_report["smolvla"] = {
            "launcher_report": smolvla.get("status"),
            "report_path": smolvla.get("report_path"),
            "action_path": (smolvla.get("smolvla_report") or {}).get("action_path"),
            "device_selected": (smolvla.get("smolvla_report") or {}).get("device_selected"),
            "raw_action_chunk_steps": (smolvla.get("smolvla_report") or {}).get("raw_action_chunk_steps"),
        }
        if smolvla.get("status") != "passed":
            iter_report["status"] = "failed"
            iter_report["error"] = "smolvla_failed"
            iteration_reports.append(iter_report)
            break

        action_path = Path(str((smolvla["smolvla_report"])["action_path"]))
        dry_gate = execute_action_chunk(
            port=port,
            action=action_path,
            output=iter_dir / "execute_gate_dry.json",
            calibration=calibration,
            execute=False,
            human_confirmed=human_confirmed,
            experimental_adapter_confirmed=False,
            action_steps=action_steps,
            delta_scale_raw_ticks=2.0,
            max_abs_delta_raw=max_abs_delta_raw,
            step_settle_seconds=step_settle_seconds,
            camera_index=None,
            visual_output_dir=None,
            record_video=False,
            video_fps=video_fps,
            metadata_config=metadata_config,
            action_stats=action_stats,
            action_semantics="absolute_joint_position",
            gripper_semantics="higher_raw_opens",
            command_units="lerobot_so100_position",
            confirm_so100_joint_order=True,
        )
        iter_report["execute_gate_dry"] = {
            "path": str(iter_dir / "execute_gate_dry.json"),
            "status": dry_gate.get("status"),
            "dry_plan_status": (dry_gate.get("dry_plan") or {}).get("status"),
            "ready_for_execution": (dry_gate.get("dry_plan") or {}).get("ready_for_execution"),
            "blocker_count": len(dry_gate.get("blockers") or []),
        }

        clipped = execute_clipped_chunk(
            dry_run_result=iter_dir / "execute_gate_dry.json",
            output=iter_dir / "clipped_execute.json",
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
        )
        iter_report["clipped_execute"] = {
            "path": str(iter_dir / "clipped_execute.json"),
            "status": clipped.get("status"),
            "send_action_called": clipped.get("send_action_called"),
            "policy_actions_executed": clipped.get("policy_actions_executed"),
            "executed_action_steps": clipped.get("executed_action_steps"),
            "observed_delta_raw": clipped.get("observed_delta_raw"),
            "post_task_torque_disabled": clipped.get("post_task_torque_disabled"),
            "motion_video": (clipped.get("motion_video") or {}).get("path"),
        }
        iter_report["physical_robot_motion"] = bool(clipped.get("send_action_called"))
        iter_report["status"] = clipped.get("status")
        iteration_reports.append(iter_report)
        if clipped.get("status") != "passed":
            break

    physical_motion_sent = any(bool(item.get("physical_robot_motion")) for item in iteration_reports)
    home_return: dict[str, Any] | None = None
    if execute and return_home_after_task and physical_motion_sent:
        home_return = move_to_home_pose(
            port=port,
            calibration=calibration,
            home_pose=home_pose,
            output=output_dir / "task_home_return" / "report.json",
            execute=True,
            human_confirmed=human_confirmed,
            workspace_clear_confirmed=human_confirmed,
            max_abs_delta_raw=max_abs_delta_raw,
            step_settle_seconds=step_settle_seconds,
            camera_index=visual_camera_index,
            visual_output_dir=output_dir / "task_home_return" / "visual",
            record_video=True,
            video_fps=video_fps,
        )

    summary = {
        "status": "passed" if iteration_reports and all(item.get("status") == "passed" for item in iteration_reports) else "failed",
        "operation": "real_so100_fixed_prompt_clipped_baseline_iterations",
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
        "return_home_after_task": return_home_after_task,
        "home_pose": str(home_pose),
        "task_home_return": {
            "path": str(output_dir / "task_home_return" / "report.json") if home_return else None,
            "status": home_return.get("status") if home_return else None,
            "post_task_torque_disabled": home_return.get("post_task_torque_disabled") if home_return else None,
            "executed_action_steps": home_return.get("executed_action_steps") if home_return else None,
        },
        "iteration_reports": iteration_reports,
    }
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-prompt clipped SmolVLA baseline iterations on real SO-100.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default="Pick up the green Android figure.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--metadata-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--action-stats", type=Path, default=DEFAULT_ACTION_STATS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--max-abs-delta-raw", type=float, default=100.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--visual-camera-index", type=int, default=1)
    parser.add_argument("--observe-duration-seconds", type=float, default=0.45)
    parser.add_argument("--observe-fps", type=float, default=2.0)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--home-pose", type=Path, default=DEFAULT_HOME_POSE)
    parser.add_argument(
        "--no-return-home-after-task",
        action="store_true",
        help="Skip automatic home-pose recovery after physical execution. Use only for explicit recovery/debug sessions.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_baseline_iterations(
                output_dir=args.output_dir,
                instruction=args.instruction,
                iterations=args.iterations,
                port=args.port,
                calibration=args.calibration,
                metadata_config=args.metadata_config,
                action_stats=args.action_stats,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                action_steps=args.action_steps,
                max_abs_delta_raw=args.max_abs_delta_raw,
                step_settle_seconds=args.step_settle_seconds,
                visual_camera_index=args.visual_camera_index,
                observe_duration_seconds=args.observe_duration_seconds,
                observe_fps=args.observe_fps,
                video_fps=args.video_fps,
                home_pose=args.home_pose,
                return_home_after_task=not args.no_return_home_after_task,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
