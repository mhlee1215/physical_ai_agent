#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402


@dataclass(frozen=True)
class FinalDiagnosticCase:
    name: str
    model_profile: str
    official_gripper_root: str
    placement_gripper_command: float
    close_gripper_command: float
    close_steps: int
    lift_steps: int
    hold_after_close_steps: int
    micro_lift_fraction: float
    micro_lift_steps: int
    gravity: tuple[float, float, float]
    cube_mass: float
    pad_size: tuple[float, float, float] | None = None
    pad_offset_y: float = 0.0
    pad_offset_z: float = 0.0
    cube_offset: tuple[float, float, float] | None = None
    lift_delta: tuple[float, float, float] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run final raw Gate 8 diagnostics for 320-vs-280 adaptive myCobot "
            "grasp/lift: contact frames, lift outcomes, reduced-gravity split, "
            "and offline visual artifacts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/raw280_final_diagnostics/gate8_final_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--ros-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--ros2-root", type=Path, default=Path("_vendor/mycobot_ros2"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--render-every", type=int, default=8)
    return parser


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _cases(args)
    reports = [run_case(args, case) for case in cases]
    summary = {
        "status": "passed" if all(report["status"] == "passed" for report in reports) else "failed",
        "output_dir": str(args.output_dir),
        "case_count": len(reports),
        "cases": reports,
        "interpretation": _interpret(reports),
    }
    summary_path = args.output_dir / "final_diagnostic_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.output_dir / "final_diagnostic_report.md"
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "passed" else 1)


def _cases(args: argparse.Namespace) -> list[FinalDiagnosticCase]:
    return [
        FinalDiagnosticCase(
            name="320_raw_reference",
            model_profile=nexus.MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
            official_gripper_root=str(args.ros2_root),
            placement_gripper_command=0.25,
            close_gripper_command=-0.7,
            close_steps=80,
            lift_steps=60,
            hold_after_close_steps=0,
            micro_lift_fraction=0.0,
            micro_lift_steps=0,
            gravity=(0.0, 0.0, -9.81),
            cube_mass=0.005,
        ),
        FinalDiagnosticCase(
            name="280_best_stable_raw",
            model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            official_gripper_root=str(args.ros_root),
            placement_gripper_command=1.0,
            close_gripper_command=-0.1,
            close_steps=180,
            lift_steps=240,
            hold_after_close_steps=0,
            micro_lift_fraction=0.0,
            micro_lift_steps=0,
            gravity=(0.0, 0.0, -9.81),
            cube_mass=0.05,
            pad_size=(0.014, 0.009, 0.006),
            pad_offset_z=-0.016,
            cube_offset=(-0.006, -0.012, 0.0),
            lift_delta=(-0.08, 0.22, 0.0),
        ),
        FinalDiagnosticCase(
            name="280_two_stage_raw",
            model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            official_gripper_root=str(args.ros_root),
            placement_gripper_command=1.0,
            close_gripper_command=-0.1,
            close_steps=180,
            lift_steps=240,
            hold_after_close_steps=160,
            micro_lift_fraction=0.12,
            micro_lift_steps=80,
            gravity=(0.0, 0.0, -9.81),
            cube_mass=0.05,
            pad_size=(0.014, 0.009, 0.006),
            pad_offset_z=-0.016,
            cube_offset=(-0.006, -0.012, 0.0),
            lift_delta=(-0.08, 0.22, 0.0),
        ),
        FinalDiagnosticCase(
            name="280_low_gravity_raw",
            model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            official_gripper_root=str(args.ros_root),
            placement_gripper_command=1.0,
            close_gripper_command=-0.1,
            close_steps=180,
            lift_steps=240,
            hold_after_close_steps=0,
            micro_lift_fraction=0.0,
            micro_lift_steps=0,
            gravity=(0.0, 0.0, -1.0),
            cube_mass=0.05,
            pad_size=(0.014, 0.009, 0.006),
            pad_offset_z=-0.016,
            cube_offset=(-0.006, -0.012, 0.0),
            lift_delta=(-0.08, 0.22, 0.0),
        ),
        FinalDiagnosticCase(
            name="280_zero_gravity_raw",
            model_profile=nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
            official_gripper_root=str(args.ros_root),
            placement_gripper_command=1.0,
            close_gripper_command=-0.1,
            close_steps=180,
            lift_steps=240,
            hold_after_close_steps=0,
            micro_lift_fraction=0.0,
            micro_lift_steps=0,
            gravity=(0.0, 0.0, 0.0),
            cube_mass=0.05,
            pad_size=(0.014, 0.009, 0.006),
            pad_offset_z=-0.016,
            cube_offset=(-0.006, -0.012, 0.0),
            lift_delta=(-0.08, 0.22, 0.0),
        ),
    ]


def run_case(args: argparse.Namespace, case: FinalDiagnosticCase) -> dict[str, Any]:
    case_dir = args.output_dir / case.name
    frames_dir = case_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _install_case_overrides(case)
    env = nexus.MyCobotNexusEnv(
        nexus.MyCobotNexusConfig(
            asset_root=args.asset_root,
            work_dir=case_dir,
            official_gripper_root=Path(case.official_gripper_root),
            model_profile=case.model_profile,
            width=args.width,
            height=args.height,
            teacher_grasp_attachment_enabled=False,
        )
    )
    records: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    snapshots: dict[str, dict[str, Any]] = {}
    try:
        env.reset(seed=1)
        gate7 = nexus._adaptive_gate7_arm_qpos(case.model_profile)
        gate8 = nexus._adaptive_gate8_lift_arm_qpos(case.model_profile)
        nexus._set_adaptive_gate_arm_pose(env, gate7)
        env._set_gripper(command=case.placement_gripper_command)
        env._mujoco.mj_forward(env.model, env.data)
        pad_midpoint = env._finger_pad_midpoint()
        cube_offset = case.cube_offset or nexus._adaptive_gate7_cube_offset(case.model_profile)
        initial_cube_position = [
            float(pad_midpoint[0] + cube_offset[0]),
            float(pad_midpoint[1] + cube_offset[1]),
            float(nexus.TASK_CUBE_POS[2] + cube_offset[2]),
        ]
        for axis, value in enumerate(initial_cube_position):
            env.data.qpos[env._cube_freejoint_qpos_index + axis] = float(value)
        qvel_start = env._cube_freejoint_qvel_index
        env.data.qvel[qvel_start:qvel_start + 6] = 0.0
        env._cube_initial_pos = list(initial_cube_position)
        env._mujoco.mj_forward(env.model, env.data)

        step_index = 0
        step_index = _run_phase(
            env,
            records,
            frame_paths,
            frames_dir,
            "close",
            step_index,
            args.render_every,
            case.close_steps,
            gate7,
            gate7,
            case.placement_gripper_command,
            case.close_gripper_command,
        )
        snapshots["end_close"] = _snapshot(env, records[-1], initial_cube_position)

        if case.hold_after_close_steps:
            step_index = _run_phase(
                env,
                records,
                frame_paths,
                frames_dir,
                "hold_after_close",
                step_index,
                args.render_every,
                case.hold_after_close_steps,
                gate7,
                gate7,
                case.close_gripper_command,
                case.close_gripper_command,
            )
            snapshots["end_hold_after_close"] = _snapshot(env, records[-1], initial_cube_position)

        if case.micro_lift_steps:
            micro_target = tuple(
                float(start + (end - start) * case.micro_lift_fraction)
                for start, end in zip(gate7, gate8, strict=True)
            )
            step_index = _run_phase(
                env,
                records,
                frame_paths,
                frames_dir,
                "micro_lift",
                step_index,
                args.render_every,
                case.micro_lift_steps,
                gate7,
                micro_target,
                case.close_gripper_command,
                case.close_gripper_command,
            )
            snapshots["end_micro_lift"] = _snapshot(env, records[-1], initial_cube_position)

        lift_start_index = len(records)
        step_index = _run_phase(
            env,
            records,
            frame_paths,
            frames_dir,
            "lift",
            step_index,
            args.render_every,
            case.lift_steps,
            gate7,
            gate8,
            case.close_gripper_command,
            case.close_gripper_command,
        )
        snapshots["first_lift"] = _snapshot(env, records[lift_start_index], initial_cube_position)
        snapshots["end_lift"] = _snapshot(env, records[-1], initial_cube_position)
        _capture_frame(env, frames_dir / f"frame_{step_index:05d}_final.png", frame_paths)
    finally:
        scene_path = str(env.scene_path)
        env.close()

    trace_path = case_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    video_path, video_error = _write_mp4(frame_paths, case_dir / f"{case.name}.mp4", fps=args.fps)
    close_infos = [record["info"] for record in records if record["phase"] == "close"]
    lift_infos = [record["info"] for record in records if record["phase"] == "lift"]
    final_info = records[-1]["info"]
    final_cube_position = final_info["cube_position"]
    initial_cube_position = snapshots["end_close"]["initial_cube_position"]
    final_lift = float(final_cube_position[2]) - float(initial_cube_position[2])
    contact_stats = _contact_stats(snapshots)
    status = (
        "passed"
        if nexus._best_sustained_two_pad_contact(close_infos) >= 15
        and nexus._best_sustained_two_pad_contact(lift_infos) >= 30
        and int(final_info.get("gripper_cube_contact_pads", 0)) >= 2
        and final_lift >= 0.025
        else "failed"
    )
    report = {
        "name": case.name,
        "status": status,
        "case": asdict(case),
        "scene_path": scene_path,
        "trace_path": str(trace_path),
        "video_path": str(video_path) if video_path else "",
        "video_error": video_error,
        "frame_count": len(frame_paths),
        "close_best_sustained_contact_steps": nexus._best_sustained_two_pad_contact(close_infos),
        "lift_best_sustained_contact_steps": nexus._best_sustained_two_pad_contact(lift_infos),
        "lift_two_pad_contact_steps": sum(
            1 for info in lift_infos if int(info.get("gripper_cube_contact_pads", 0)) >= 2
        ),
        "final_gripper_cube_contact_pads": int(final_info.get("gripper_cube_contact_pads", 0)),
        "final_gripper_cube_contacts": int(final_info.get("gripper_cube_contacts", 0)),
        "final_cube_lift": final_lift,
        "final_cube_position": final_cube_position,
        "snapshots": snapshots,
        "contact_stats": contact_stats,
    }
    case_report_path = case_dir / "report.json"
    case_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_phase(
    env: nexus.MyCobotNexusEnv,
    records: list[dict[str, Any]],
    frame_paths: list[Path],
    frames_dir: Path,
    phase: str,
    step_index: int,
    render_every: int,
    steps: int,
    start_arm: tuple[float, ...],
    end_arm: tuple[float, ...],
    start_gripper: float,
    end_gripper: float,
) -> int:
    denominator = max(steps - 1, 1)
    hold_arm_pose_each_step = env.config.model_profile == nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER
    for phase_step in range(steps):
        alpha = nexus._smoothstep(phase_step / denominator) if phase == "lift" else phase_step / denominator
        arm = tuple(nexus._lerp_vector(list(start_arm), list(end_arm), alpha))
        gripper = start_gripper + (end_gripper - start_gripper) * alpha
        if hold_arm_pose_each_step:
            nexus._set_adaptive_gate_arm_pose(env, arm)
        obs, reward, terminated, truncated, info = env.step([*arm, gripper])
        if hold_arm_pose_each_step:
            nexus._set_adaptive_gate_arm_pose(env, arm)
        record = nexus._phase_record(phase, step_index, obs, reward, terminated, truncated, info)
        records.append(record)
        if step_index % max(render_every, 1) == 0:
            _capture_frame(env, frames_dir / f"frame_{step_index:05d}_{phase}.png", frame_paths)
        step_index += 1
    return step_index


def _capture_frame(env: nexus.MyCobotNexusEnv, path: Path, frame_paths: list[Path]) -> None:
    rgb = env.render().astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    frame_paths.append(path)


def _write_mp4(frame_paths: list[Path], path: Path, *, fps: int) -> tuple[Path | None, str]:
    if not frame_paths:
        return None, "no frames"
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return None, f"could not read first frame: {frame_paths[0]}"
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return None, "cv2.VideoWriter did not open"
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            writer.write(frame)
    writer.release()
    return path, ""


def _install_case_overrides(case: FinalDiagnosticCase) -> None:
    nexus.ADAPTIVE_FINGER_PAD_EULER = (0.0, 0.0, 0.0)
    if case.model_profile == nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
        left = nexus.ADAPTIVE_LEFT_FINGER_PAD_POS
        right = nexus.ADAPTIVE_RIGHT_FINGER_PAD_POS
        nexus.ADAPTIVE_280_LEFT_FINGER_PAD_POS = (
            left[0],
            left[1] + case.pad_offset_y,
            left[2] + case.pad_offset_z,
        )
        nexus.ADAPTIVE_280_RIGHT_FINGER_PAD_POS = (
            right[0],
            right[1] + case.pad_offset_y,
            right[2] + case.pad_offset_z,
        )
        if case.pad_size is not None:
            nexus.ADAPTIVE_280_FINGER_PAD_SIZE = case.pad_size
        if case.cube_offset is not None:
            nexus.ADAPTIVE_280_PI_GATE7_CUBE_OFFSET = case.cube_offset
        if case.lift_delta is not None:
            gate7 = nexus.ADAPTIVE_280_PI_GATE7_TABLE_ARM_QPOS
            nexus.ADAPTIVE_280_PI_GATE8_LIFT_ARM_QPOS = (
                gate7[0],
                gate7[1] + case.lift_delta[0],
                gate7[2] + case.lift_delta[1],
                gate7[3] + case.lift_delta[2],
                gate7[4],
                gate7[5],
            )
    _install_scene_override(case)


def _install_scene_override(case: FinalDiagnosticCase) -> None:
    original_build = nexus.build_mycobot_nexus_scene_model
    if getattr(original_build, "_final_diag_wrapper", False):
        original_build = original_build._final_diag_original_build  # type: ignore[attr-defined]

    def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> None:
        original_build(*wrapper_args, **wrapper_kwargs)
        scene_path = wrapper_kwargs.get("scene_path") if "scene_path" in wrapper_kwargs else wrapper_args[1]
        tree = ET.parse(scene_path)
        root = tree.getroot()
        option = root.find("option")
        if option is not None:
            option.set("gravity", " ".join(str(value) for value in case.gravity))
        for geom in root.findall(".//geom"):
            if geom.attrib.get("name") == nexus.TASK_CUBE_GEOM:
                geom.set("mass", str(case.cube_mass))
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    wrapper._final_diag_wrapper = True  # type: ignore[attr-defined]
    wrapper._final_diag_original_build = original_build  # type: ignore[attr-defined]
    nexus.build_mycobot_nexus_scene_model = wrapper


def _snapshot(
    env: nexus.MyCobotNexusEnv,
    record: dict[str, Any],
    initial_cube_position: list[float],
) -> dict[str, Any]:
    return {
        "step": int(record["step"]),
        "phase": record["phase"],
        "info": record["info"],
        "initial_cube_position": initial_cube_position,
        "cube_position": env._cube_position(),
        "pad_midpoint": [float(value) for value in env._finger_pad_midpoint()],
        "pad_frames": _pad_frames(env),
        "contacts": _pad_cube_contacts(env),
    }


def _pad_frames(env: nexus.MyCobotNexusEnv) -> dict[str, Any]:
    frames = {}
    for name in ("left_finger_pad", "right_finger_pad"):
        geom_id = env._mujoco.mj_name2id(env.model, env._mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            continue
        frames[name] = {
            "position": [float(value) for value in env.data.geom_xpos[geom_id]],
            "axes": np.asarray(env.data.geom_xmat[geom_id], dtype=float).reshape(3, 3).tolist(),
        }
    return frames


def _pad_cube_contacts(env: nexus.MyCobotNexusEnv) -> list[dict[str, Any]]:
    mujoco = env._mujoco
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, nexus.TASK_CUBE_GEOM)
    pad_ids = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"): "left_finger_pad",
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"): "right_finger_pad",
    }
    contacts = []
    for index in range(int(env.data.ncon)):
        contact = env.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id not in pair:
            continue
        pad_geom_ids = [geom_id for geom_id in pair if geom_id in pad_ids]
        if not pad_geom_ids:
            continue
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(env.model, env.data, index, force)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        normal = frame[0]
        world_force = frame.T @ force[:3]
        contacts.append(
            {
                "pad": pad_ids[pad_geom_ids[0]],
                "geom1": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                "geom2": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                "distance": float(contact.dist),
                "position": [float(value) for value in contact.pos],
                "normal": [float(value) for value in normal],
                "abs_normal_z": abs(float(normal[2])),
                "force_components": [float(value) for value in force],
                "world_force": [float(value) for value in world_force],
            }
        )
    return contacts


def _contact_stats(snapshots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    contacts = [
        contact
        for snapshot in snapshots.values()
        for contact in snapshot.get("contacts", [])
    ]
    abs_z = [float(contact["abs_normal_z"]) for contact in contacts]
    return {
        "sampled_contact_count": len(contacts),
        "avg_abs_normal_z": float(np.mean(abs_z)) if abs_z else None,
        "min_abs_normal_z": float(np.min(abs_z)) if abs_z else None,
        "side_like_contacts_abs_z_lt_0_5": sum(1 for value in abs_z if value < 0.5),
    }


def _interpret(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {report["name"]: report for report in reports}
    low = by_name.get("280_low_gravity_raw", {})
    zero = by_name.get("280_zero_gravity_raw", {})
    best = by_name.get("280_best_stable_raw", {})

    def retained_lift(report: dict[str, Any]) -> bool:
        return (
            float(report.get("final_cube_lift", 0.0)) >= 0.025
            and int(report.get("final_gripper_cube_contact_pads", 0)) >= 2
            and int(report.get("lift_best_sustained_contact_steps", 0)) >= 30
        )

    def launched_without_grasp(report: dict[str, Any]) -> bool:
        return (
            float(report.get("final_cube_lift", 0.0)) >= 0.025
            and int(report.get("final_gripper_cube_contact_pads", 0)) == 0
            and int(report.get("lift_best_sustained_contact_steps", 0)) == 0
        )

    return {
        "raw_280_passed_any_case": any(
            report["status"] == "passed" and report["name"].startswith("280_")
            for report in reports
        ),
        "low_or_zero_gravity_helped": retained_lift(low) or retained_lift(zero),
        "low_or_zero_gravity_launched_without_grasp": (
            launched_without_grasp(low) or launched_without_grasp(zero)
        ),
        "best_stable_summary": {
            "lift_best_sustained_contact_steps": best.get("lift_best_sustained_contact_steps"),
            "final_cube_lift": best.get("final_cube_lift"),
            "final_gripper_cube_contact_pads": best.get("final_gripper_cube_contact_pads"),
            "avg_abs_normal_z": best.get("contact_stats", {}).get("avg_abs_normal_z"),
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# myCobot Adaptive Raw Gate 8 Final Diagnostics",
        "",
        f"Overall status: `{summary['status']}`",
        "",
        "| Case | Status | Close best | Lift best | Lift two-pad | Final pads | Final lift | MP4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in summary["cases"]:
        lines.append(
            "| {name} | {status} | {close} | {lift} | {two_pad} | {pads} | {final_lift:.6f} | {video} |".format(
                name=report["name"],
                status=report["status"],
                close=report["close_best_sustained_contact_steps"],
                lift=report["lift_best_sustained_contact_steps"],
                two_pad=report["lift_two_pad_contact_steps"],
                pads=report["final_gripper_cube_contact_pads"],
                final_lift=float(report["final_cube_lift"]),
                video=report["video_path"],
            )
        )
    lines.extend(["", "```json", json.dumps(summary["interpretation"], indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
