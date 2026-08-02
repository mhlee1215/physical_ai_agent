"""Replay one LeRobot SO101 dataset episode on the physical follower arm.

The episode contract is intentionally strict:

* ``observation.state`` at frame 0 defines the absolute start pose.
* every row's ``action`` is the absolute SO101 joint-position command.
* missing frames, non-finite values, implicit trajectory clipping, and
  oversized trajectory steps are rejected before torque is enabled.
* Ctrl-C, tracking failure, or any exception disables torque in ``finally``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

import pyarrow.dataset as pds
from pydantic import BaseModel, ConfigDict, Field, model_validator

from physical_ai_agent.real_so100.sim_policy_bridge import sim_qpos_to_hardware_positions
from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER, load_calibration
from scripts.real_so100_micro_step import (
    _make_so100_bus,
    _probe_motion_video,
    _record_motion_video,
    _start_motion_video,
)


POSITION_RESOLUTION = 4095.0
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class EpisodeTrajectory:
    dataset_root: Path
    episode: int
    frame_indices: tuple[int, ...]
    start_state: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]
    dataset_fps: float

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_StrictModel):
    root: Path
    episode: int = Field(ge=0)


class HardwareConfig(_StrictModel):
    port: str = Field(min_length=1)
    calibration: Path
    home_pose: Path
    serial_num_retry: int = Field(ge=0)


class TrajectoryConfig(_StrictModel):
    fps: float | None
    alignment: Literal["absolute", "calibrated-start-relative"]
    max_trajectory_step_raw: float = Field(gt=0)
    max_bridge_step_raw: float = Field(gt=0)
    max_tracking_error_raw: float = Field(gt=0)
    max_start_error_raw: float = Field(gt=0)
    start_range_tolerance_raw: float = Field(ge=0)
    bridge_step_seconds: float = Field(ge=0)
    hold_final_seconds: float = Field(ge=0)


class ExecutionConfig(_StrictModel):
    enabled: bool
    operator_confirmed: bool
    workspace_clear_confirmed: bool
    direct_observer_confirmed: bool
    require_typed_confirmation: bool
    return_mode: Literal["home", "preflight", "none"]
    disable_torque_after_run: Literal[True]


class RecordingConfig(_StrictModel):
    enabled: bool
    camera_index: int = Field(ge=0)
    video_fps: float = Field(gt=0)


class ReplayConfig(_StrictModel):
    schema_version: Literal[1]
    dataset: DatasetConfig
    hardware: HardwareConfig
    trajectory: TrajectoryConfig
    execution: ExecutionConfig
    recording: RecordingConfig
    output_dir: Path

    @model_validator(mode="after")
    def validate_execution_gate(self) -> "ReplayConfig":
        if self.execution.enabled:
            confirmations = (
                self.execution.operator_confirmed,
                self.execution.workspace_clear_confirmed,
                self.execution.direct_observer_confirmed,
            )
            if not all(confirmations):
                raise ValueError(
                    "enabled hardware replay requires operator, workspace-clear, and direct-observer confirmations"
                )
            if not self.recording.enabled:
                raise ValueError("enabled hardware replay requires recording.enabled=true")
        return self


def load_episode_trajectory(dataset_root: Path, episode: int) -> EpisodeTrajectory:
    info_path = dataset_root / "meta" / "info.json"
    data_root = dataset_root / "data"
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset metadata not found: {info_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset parquet directory not found: {data_root}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"dataset fps must be positive, got {info.get('fps')!r}")

    table = pds.dataset(data_root, format="parquet").to_table(
        columns=["episode_index", "frame_index", "observation.state", "action"],
        filter=pds.field("episode_index") == int(episode),
    )
    rows = sorted(
        zip(
            table["frame_index"].to_pylist(),
            table["observation.state"].to_pylist(),
            table["action"].to_pylist(),
            strict=True,
        ),
        key=lambda row: int(row[0]),
    )
    if not rows:
        raise ValueError(f"episode {episode} does not exist in {dataset_root}")

    frames = tuple(int(row[0]) for row in rows)
    expected = tuple(range(len(rows)))
    if frames != expected:
        raise ValueError(f"episode {episode} must contain contiguous frames 0..{len(rows) - 1}; got {frames}")

    states = tuple(_finite_joint_vector(row[1], label=f"frame {row[0]} observation.state") for row in rows)
    actions = tuple(_finite_joint_vector(row[2], label=f"frame {row[0]} action") for row in rows)
    return EpisodeTrajectory(
        dataset_root=dataset_root,
        episode=int(episode),
        frame_indices=frames,
        start_state=states[0],
        actions=actions,
        dataset_fps=fps,
    )


def sim_qpos_to_raw_targets(
    qpos: Sequence[float], calibration: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    raw = sim_qpos_to_unbounded_raw(qpos, calibration)
    _assert_target_in_range(raw, calibration)
    return raw


def sim_qpos_to_unbounded_raw(
    qpos: Sequence[float], calibration: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    values = _finite_joint_vector(qpos, label="SO101 qpos")
    hardware = sim_qpos_to_hardware_positions(list(values))
    raw: dict[str, float] = {}
    for joint in SO100_JOINT_ORDER:
        item = calibration[joint]
        lower = float(item["range_min"])
        upper = float(item["range_max"])
        if joint == "gripper":
            value = lower + hardware[joint] * (upper - lower) / 100.0
        else:
            value = (lower + upper) / 2.0 + hardware[joint] * POSITION_RESOLUTION / 360.0
        raw[joint] = value
    return raw


def build_absolute_replay_plan(
    trajectory: EpisodeTrajectory,
    calibration: Mapping[str, Mapping[str, Any]],
    *,
    max_trajectory_step_raw: float,
    alignment: str = "absolute",
) -> dict[str, Any]:
    if max_trajectory_step_raw <= 0:
        raise ValueError("max_trajectory_step_raw must be positive")
    unaligned_start = sim_qpos_to_unbounded_raw(trajectory.start_state, calibration)
    if alignment == "absolute":
        _assert_target_in_range(unaligned_start, calibration)
        start_raw = unaligned_start
    elif alignment == "calibrated-start-relative":
        start_raw = _clip_target_to_calibration(unaligned_start, calibration)
    else:
        raise ValueError(f"unsupported alignment mode: {alignment}")
    alignment_offset = {
        joint: start_raw[joint] - unaligned_start[joint] for joint in SO100_JOINT_ORDER
    }
    action_targets: list[dict[str, Any]] = []
    previous = start_raw
    maximum = {joint: 0.0 for joint in SO100_JOINT_ORDER}
    for frame, action in zip(trajectory.frame_indices, trajectory.actions, strict=True):
        unaligned_target = sim_qpos_to_unbounded_raw(action, calibration)
        target = {
            joint: unaligned_target[joint] + alignment_offset[joint] for joint in SO100_JOINT_ORDER
        }
        _assert_target_in_range(target, calibration, label=f"frame {frame} action")
        delta = {joint: target[joint] - previous[joint] for joint in SO100_JOINT_ORDER}
        for joint, value in delta.items():
            maximum[joint] = max(maximum[joint], abs(value))
            if abs(value) > max_trajectory_step_raw:
                raise ValueError(
                    f"frame {frame} {joint} action delta {value:.3f} raw exceeds "
                    f"max_trajectory_step_raw={max_trajectory_step_raw}"
                )
        action_targets.append({"frame_index": frame, "target_raw": target, "delta_raw": delta})
        previous = target
    return {
        "dataset_root": str(trajectory.dataset_root),
        "episode": trajectory.episode,
        "frame_count": trajectory.frame_count,
        "dataset_fps": trajectory.dataset_fps,
        "alignment": alignment,
        "alignment_offset_raw": alignment_offset,
        "start_source": "observation.state at frame 0",
        "command_source": "action at each frame",
        "start_raw": start_raw,
        "actions": action_targets,
        "max_abs_trajectory_step_raw": maximum,
    }


def interpolate_raw_pose(
    start: Mapping[str, float], target: Mapping[str, float], *, max_step_raw: float
) -> list[dict[str, float]]:
    if max_step_raw <= 0:
        raise ValueError("max_step_raw must be positive")
    largest = max(abs(float(target[joint]) - float(start[joint])) for joint in SO100_JOINT_ORDER)
    count = max(1, int(math.ceil(largest / max_step_raw)))
    return [
        {
            joint: float(start[joint])
            + (float(target[joint]) - float(start[joint])) * float(index) / float(count)
            for joint in SO100_JOINT_ORDER
        }
        for index in range(1, count + 1)
    ]


def run_replay(args: SimpleNamespace) -> dict[str, Any]:
    trajectory = load_episode_trajectory(args.dataset_root.resolve(), args.episode)
    calibration = load_calibration(args.calibration)
    plan = build_absolute_replay_plan(
        trajectory,
        calibration,
        max_trajectory_step_raw=args.max_trajectory_step_raw,
        alignment=args.alignment,
    )
    fps = float(args.fps if args.fps is not None else trajectory.dataset_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("replay fps must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so101_absolute_dataset_episode_replay",
        "dataset_root": str(trajectory.dataset_root),
        "episode": trajectory.episode,
        "frame_count": trajectory.frame_count,
        "fps": fps,
        "duration_seconds": trajectory.frame_count / fps,
        "port": args.port,
        "calibration": str(args.calibration),
        "config_path": str(args.config_path),
        "config": args.config_payload,
        "plan": plan,
        "execute_requested": bool(args.execute),
        "send_action_called": False,
        "teacher_actions_executed": False,
        "policy_actions_executed": False,
        "post_task_torque_disabled": False,
        "status": "validated",
    }
    if args.return_mode == "home":
        report["home_return_plan"] = build_home_return_plan(args.home_pose, calibration)
    _write_report(output_dir, report)
    _print_plan_summary(report, output_dir)
    if not args.execute:
        print("Dry run only. Add --execute to move the robot.")
        return report
    if args.require_typed_confirmation:
        phrase = f"EXECUTE {trajectory.episode}"
        entered = input(f"Workspace clear and watching the robot? Type {phrase!r}: ").strip()
        if entered != phrase:
            report["status"] = "cancelled"
            report["blocker"] = "operator confirmation phrase did not match"
            _write_report(output_dir, report)
            return report

    bus, _motors = _make_so100_bus(args.port)
    capture = writer = video_result = None
    executed: list[dict[str, Any]] = []
    bridge_log: list[dict[str, Any]] = []
    return_log: list[dict[str, Any]] = []
    preflight_raw: dict[str, float] | None = None
    try:
        bus.connect(handshake=True)
        _verify_live_calibration(bus, calibration)
        preflight_raw = _read_positions(bus)
        report["readback_before_raw"] = preflight_raw
        _assert_readback_in_range(preflight_raw, calibration, tolerance=args.start_range_tolerance_raw)
        hold = _round_targets(preflight_raw)
        bus.sync_write("Goal_Position", hold, normalize=False, num_retry=args.serial_num_retry)
        bus.enable_torque(num_retry=args.serial_num_retry)
        report["hold_current_goal_preloaded_raw"] = hold
        report["torque_enabled"] = True

        if args.record_video:
            capture, writer, video_result = _start_motion_video(
                camera_index=args.camera_index,
                output_dir=output_dir / "visual",
                fps=args.video_fps,
            )
            report["motion_video"] = video_result

        bridge = interpolate_raw_pose(
            preflight_raw,
            plan["start_raw"],
            max_step_raw=args.max_bridge_step_raw,
        )
        report["bridge_step_count"] = len(bridge)
        for index, target in enumerate(bridge):
            item = _write_and_verify(
                bus,
                target,
                phase="bridge",
                index=index,
                settle_seconds=args.bridge_step_seconds,
                max_tracking_error_raw=args.max_tracking_error_raw,
                serial_num_retry=args.serial_num_retry,
            )
            bridge_log.append(item)
            report["send_action_called"] = True
            if capture is not None:
                _record_motion_video(
                    capture=capture,
                    writer=writer,
                    result=video_result,
                    duration_seconds=args.bridge_step_seconds,
                    fps=args.video_fps,
                )

        start_readback = _read_positions(bus)
        start_errors = {
            joint: float(plan["start_raw"][joint]) - start_readback[joint] for joint in SO100_JOINT_ORDER
        }
        report["start_pose_readback_raw"] = start_readback
        report["start_pose_error_raw"] = start_errors
        _raise_for_tracking_error(start_errors, args.max_start_error_raw, phase="start pose")

        period = 1.0 / fps
        started = time.monotonic()
        next_command_at = started
        for item in plan["actions"]:
            frame = int(item["frame_index"])
            _sleep_until(next_command_at)
            command = _round_targets(item["target_raw"])
            bus.sync_write(
                "Goal_Position", command, normalize=False, num_retry=args.serial_num_retry
            )
            command_sent_at = time.monotonic()
            next_command_at = command_sent_at + period
            report["send_action_called"] = True
            if capture is not None:
                _record_motion_video(
                    capture=capture,
                    writer=writer,
                    result=video_result,
                    duration_seconds=max(0.0, next_command_at - time.monotonic()),
                    fps=args.video_fps,
                )
            else:
                _sleep_until(next_command_at)
            readback = _read_positions(bus)
            errors = {joint: command[joint] - readback[joint] for joint in SO100_JOINT_ORDER}
            _raise_for_tracking_error(errors, args.max_tracking_error_raw, phase=f"episode frame {frame}")
            executed.append(
                {
                    "frame_index": frame,
                    "scheduled_seconds": frame * period,
                    "actual_seconds": command_sent_at - started,
                    "command_raw": command,
                    "readback_raw": readback,
                    "error_raw": errors,
                }
            )

        report["teacher_actions_executed"] = len(executed) == trajectory.frame_count
        report["executed_action_steps"] = len(executed)
        report["readback_after_trajectory_raw"] = _read_positions(bus)
        report["status"] = "passed"
        time.sleep(max(0.0, args.hold_final_seconds))
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "KeyboardInterrupt"
    except Exception as exc:  # noqa: BLE001 - retain exact hardware failure evidence.
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        report["bridge_steps"] = bridge_log
        report["executed_steps"] = executed
        if bus.is_connected and preflight_raw is not None and args.return_mode != "none":
            try:
                return_target = _load_return_target(args, preflight_raw, calibration)
                return_steps = interpolate_raw_pose(
                    _read_positions(bus), return_target, max_step_raw=args.max_bridge_step_raw
                )
                for index, target in enumerate(return_steps):
                    return_log.append(
                        _write_and_verify(
                            bus,
                            target,
                            phase="return",
                            index=index,
                            settle_seconds=args.bridge_step_seconds,
                            max_tracking_error_raw=args.max_tracking_error_raw,
                            serial_num_retry=args.serial_num_retry,
                        )
                    )
                report["return"] = {
                    "mode": args.return_mode,
                    "target_raw": return_target,
                    "steps": return_log,
                    "final_readback_raw": _read_positions(bus),
                }
            except Exception as exc:  # noqa: BLE001
                report["return"] = {"mode": args.return_mode, "status": "failed", "error": repr(exc)}
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if isinstance(video_result, dict):
            report["motion_video"] = {**video_result, **_probe_motion_video(Path(video_result["path"]))}
        if bus.is_connected:
            try:
                bus.disconnect(disable_torque=True)
                report["post_task_torque_disabled"] = True
            except Exception as exc:  # noqa: BLE001
                report["disconnect_error"] = repr(exc)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_report(output_dir, report)
    return report


def _finite_joint_vector(value: Sequence[Any], *, label: str) -> tuple[float, ...]:
    if len(value) != len(SO100_JOINT_ORDER):
        raise ValueError(f"{label} must have {len(SO100_JOINT_ORDER)} values, got {len(value)}")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _assert_target_in_range(
    target: Mapping[str, float],
    calibration: Mapping[str, Mapping[str, Any]],
    *,
    label: str = "qpos",
) -> None:
    for joint in SO100_JOINT_ORDER:
        lower = float(calibration[joint]["range_min"])
        upper = float(calibration[joint]["range_max"])
        value = float(target[joint])
        if not lower <= value <= upper:
            raise ValueError(
                f"{joint} {label} maps to raw target {value:.3f}, "
                f"outside calibration {lower:.0f}..{upper:.0f}"
            )


def _clip_target_to_calibration(
    target: Mapping[str, float], calibration: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    return {
        joint: min(
            float(calibration[joint]["range_max"]),
            max(float(calibration[joint]["range_min"]), float(target[joint])),
        )
        for joint in SO100_JOINT_ORDER
    }


def _verify_live_calibration(bus: Any, expected: Mapping[str, Mapping[str, Any]]) -> None:
    actual = bus.read_calibration()
    mismatches = []
    for joint in SO100_JOINT_ORDER:
        for field in ("homing_offset", "range_min", "range_max"):
            if int(getattr(actual[joint], field)) != int(expected[joint][field]):
                mismatches.append(f"{joint}.{field}")
    if mismatches:
        raise RuntimeError(f"live calibration differs from configured calibration: {mismatches}")


def _assert_readback_in_range(
    readback: Mapping[str, float],
    calibration: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float,
) -> None:
    violations = {}
    for joint in SO100_JOINT_ORDER:
        lower = float(calibration[joint]["range_min"])
        upper = float(calibration[joint]["range_max"])
        value = float(readback[joint])
        if value < lower - tolerance or value > upper + tolerance:
            violations[joint] = {"value": value, "range": [lower, upper]}
    if violations:
        raise RuntimeError(f"live readback is outside calibration: {violations}")


def _read_positions(bus: Any) -> dict[str, float]:
    return {
        joint: float(value)
        for joint, value in bus.sync_read("Present_Position", normalize=False).items()
        if joint in SO100_JOINT_ORDER
    }


def _round_targets(target: Mapping[str, float]) -> dict[str, int]:
    return {joint: int(round(float(target[joint]))) for joint in SO100_JOINT_ORDER}


def _write_and_verify(
    bus: Any,
    target: Mapping[str, float],
    *,
    phase: str,
    index: int,
    settle_seconds: float,
    max_tracking_error_raw: float,
    serial_num_retry: int,
) -> dict[str, Any]:
    command = _round_targets(target)
    bus.sync_write("Goal_Position", command, normalize=False, num_retry=serial_num_retry)
    time.sleep(max(0.0, settle_seconds))
    readback = _read_positions(bus)
    errors = {joint: command[joint] - readback[joint] for joint in SO100_JOINT_ORDER}
    _raise_for_tracking_error(errors, max_tracking_error_raw, phase=f"{phase} step {index}")
    return {"index": index, "command_raw": command, "readback_raw": readback, "error_raw": errors}


def _raise_for_tracking_error(errors: Mapping[str, float], limit: float, *, phase: str) -> None:
    violations = {joint: value for joint, value in errors.items() if abs(float(value)) > limit}
    if violations:
        raise RuntimeError(f"{phase} tracking error exceeds {limit} raw: {violations}")


def _load_return_target(
    args: SimpleNamespace,
    preflight_raw: Mapping[str, float],
    calibration: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    if args.return_mode == "preflight":
        return {joint: float(preflight_raw[joint]) for joint in SO100_JOINT_ORDER}
    return load_home_pose_raw(args.home_pose, calibration)


def load_home_pose_raw(
    path: Path, calibration: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    return build_home_return_plan(path, calibration)["target_raw"]


def build_home_return_plan(
    path: Path, calibration: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("target_raw", payload.get("positions_raw", payload))
    requested = {joint: float(source[joint]) for joint in SO100_JOINT_ORDER}
    target = _clip_target_to_calibration(requested, calibration)
    adjustments = {
        joint: {"requested": requested[joint], "calibrated": target[joint]}
        for joint in SO100_JOINT_ORDER
        if requested[joint] != target[joint]
    }
    return {
        "path": str(path),
        "name": payload.get("name"),
        "requested_raw": requested,
        "target_raw": target,
        "calibration_adjustments": adjustments,
    }


def _sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def _write_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def _print_plan_summary(report: Mapping[str, Any], output_dir: Path) -> None:
    plan = report["plan"]
    print(
        f"dataset={report['dataset_root']} episode={report['episode']} "
        f"frames={report['frame_count']} fps={report['fps']} duration={report['duration_seconds']:.2f}s"
    )
    print(f"start=observation.state[0] commands=action[0..{report['frame_count'] - 1}]")
    print(f"alignment={plan['alignment']}")
    print(f"max per-frame raw delta={max(plan['max_abs_trajectory_step_raw'].values()):.2f}")
    print(f"report={output_dir / 'report.json'}")


def load_replay_config(path: Path) -> ReplayConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReplayConfig.model_validate(payload)


def runtime_from_config(config: ReplayConfig, *, config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=config_path.resolve(),
        config_payload=config.model_dump(mode="json"),
        dataset_root=_repo_path(config.dataset.root),
        episode=config.dataset.episode,
        port=config.hardware.port,
        calibration=_repo_path(config.hardware.calibration),
        home_pose=_repo_path(config.hardware.home_pose),
        serial_num_retry=config.hardware.serial_num_retry,
        output_dir=_repo_path(config.output_dir),
        fps=config.trajectory.fps,
        alignment=config.trajectory.alignment,
        max_trajectory_step_raw=config.trajectory.max_trajectory_step_raw,
        max_bridge_step_raw=config.trajectory.max_bridge_step_raw,
        max_tracking_error_raw=config.trajectory.max_tracking_error_raw,
        max_start_error_raw=config.trajectory.max_start_error_raw,
        start_range_tolerance_raw=config.trajectory.start_range_tolerance_raw,
        bridge_step_seconds=config.trajectory.bridge_step_seconds,
        hold_final_seconds=config.trajectory.hold_final_seconds,
        execute=config.execution.enabled,
        require_typed_confirmation=config.execution.require_typed_confirmation,
        return_mode=config.execution.return_mode,
        record_video=config.recording.enabled,
        camera_index=config.recording.camera_index,
        video_fps=config.recording.video_fps,
    )


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _config_path_from_argv(argv: Sequence[str]) -> Path:
    if len(argv) != 2:
        raise SystemExit("usage: scripts/replay.sh <config.json>")
    return Path(argv[1])


def main() -> None:
    config_path = _config_path_from_argv(sys.argv)
    config = load_replay_config(config_path)
    args = runtime_from_config(config, config_path=config_path)
    report = run_replay(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "executed_action_steps": report.get("executed_action_steps", 0),
                "teacher_actions_executed": report.get("teacher_actions_executed", False),
                "post_task_torque_disabled": report.get("post_task_torque_disabled", False),
                "error": report.get("error"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report["status"] in {"validated", "cancelled", "passed"} else 1)


if __name__ == "__main__":
    main()
