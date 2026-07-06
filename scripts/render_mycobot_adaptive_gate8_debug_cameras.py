#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import diagnose_mycobot_adaptive_raw_gate8_final as final_diag  # noqa: E402
import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render myCobot Gate 8 with debug cameras: visible contact pads, "
            "simplified scene styling, auto-scored camera candidates, and event sheets."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/raw280_final_diagnostics/gate8_debug_cameras_001"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--ros-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--ros2-root", type=Path, default=Path("_vendor/mycobot_ros2"))
    parser.add_argument("--cases", default="320_raw_reference,280_best_stable_raw")
    parser.add_argument("--render-every", type=int, default=6)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--view-width", type=int, default=960)
    parser.add_argument("--view-height", type=int, default=720)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {value.strip() for value in args.cases.split(",") if value.strip()}
    reports = []
    for case in final_diag._cases(args):
        if case.name in wanted:
            reports.append(render_case(args, case))
    manifest = {
        "status": "passed",
        "output_dir": str(args.output_dir),
        "reports": reports,
    }
    manifest_path = args.output_dir / "debug_camera_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def render_case(args: argparse.Namespace, case: final_diag.FinalDiagnosticCase) -> dict[str, Any]:
    case_dir = args.output_dir / case.name
    frames_dir = case_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    final_diag._install_case_overrides(case)
    _install_debug_visual_override()
    env = nexus.MyCobotNexusEnv(
        nexus.MyCobotNexusConfig(
            asset_root=args.asset_root,
            work_dir=case_dir,
            official_gripper_root=Path(case.official_gripper_root),
            model_profile=case.model_profile,
            width=args.view_width,
            height=args.view_height,
            teacher_grasp_attachment_enabled=False,
        )
    )
    renderers: list[Any] = []
    records: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    frame_scores: list[dict[str, Any]] = []
    try:
        renderers = [
            env._mujoco.Renderer(env.model, height=args.view_height, width=args.view_width)
            for _ in range(3)
        ]
        _initialize_case(env, case)
        gate7 = nexus._adaptive_gate7_arm_qpos(case.model_profile)
        gate8 = nexus._adaptive_gate8_lift_arm_qpos(case.model_profile)
        step_index = 0
        step_index = _run_phase(
            env, renderers, records, frame_paths, frame_scores, frames_dir, case, "close",
            step_index, args.render_every, case.close_steps, gate7, gate7,
            case.placement_gripper_command, case.close_gripper_command,
        )
        if case.hold_after_close_steps:
            step_index = _run_phase(
                env, renderers, records, frame_paths, frame_scores, frames_dir, case, "hold_after_close",
                step_index, args.render_every, case.hold_after_close_steps, gate7, gate7,
                case.close_gripper_command, case.close_gripper_command,
            )
        if case.micro_lift_steps:
            micro_target = tuple(
                float(start + (end - start) * case.micro_lift_fraction)
                for start, end in zip(gate7, gate8, strict=True)
            )
            step_index = _run_phase(
                env, renderers, records, frame_paths, frame_scores, frames_dir, case, "micro_lift",
                step_index, args.render_every, case.micro_lift_steps, gate7, micro_target,
                case.close_gripper_command, case.close_gripper_command,
            )
        step_index = _run_phase(
            env, renderers, records, frame_paths, frame_scores, frames_dir, case, "lift",
            step_index, args.render_every, case.lift_steps, gate7, gate8,
            case.close_gripper_command, case.close_gripper_command,
        )
        _capture_debug_frame(
            env, renderers, records[-1], case, frames_dir / f"frame_{step_index:05d}_final.png",
            frame_paths, frame_scores,
        )
    finally:
        for renderer in renderers:
            renderer.close()
        env.close()

    video_path = case_dir / f"{case.name}_debug_cameras.mp4"
    _write_mp4(frame_paths, video_path, fps=args.fps)
    trace_path = case_dir / "debug_camera_trace.jsonl"
    trace_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    scores_path = case_dir / "debug_camera_frame_scores.json"
    scores_path.write_text(json.dumps(frame_scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sheet_path = _write_event_sheet(case_dir, records, frame_paths)
    report = {
        "name": case.name,
        "video_path": str(video_path),
        "trace_path": str(trace_path),
        "frame_scores_path": str(scores_path),
        "event_sheet_path": str(sheet_path),
        "frame_count": len(frame_paths),
        **_contact_summary(records),
        **_visibility_summary(frame_scores),
    }
    (case_dir / "debug_camera_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _install_debug_visual_override() -> None:
    original_build = nexus.build_mycobot_nexus_scene_model
    if getattr(original_build, "_debug_camera_wrapper", False):
        return

    def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> None:
        original_build(*wrapper_args, **wrapper_kwargs)
        scene_path = wrapper_kwargs.get("scene_path") if "scene_path" in wrapper_kwargs else wrapper_args[1]
        tree = ET.parse(scene_path)
        root = tree.getroot()
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        global_node = visual.find("global")
        if global_node is None:
            global_node = ET.SubElement(visual, "global")
        global_node.set("offwidth", "1920")
        global_node.set("offheight", "1440")
        for geom in root.findall(".//geom"):
            name = geom.attrib.get("name", "")
            mesh = geom.attrib.get("mesh", "")
            if name == nexus.TASK_CUBE_GEOM:
                geom.attrib.pop("material", None)
                geom.set("rgba", "1 0.03 0.0 1")
            elif name == "left_finger_pad":
                geom.set("rgba", "0.0 1.0 0.0 0.88")
            elif name == "right_finger_pad":
                geom.set("rgba", "0.0 0.25 1.0 0.88")
            elif "gripper" in name or "gripper" in mesh:
                geom.attrib.pop("material", None)
                geom.set("rgba", "0.08 0.08 0.08 0.72")
            elif name == "nexus_floor":
                geom.attrib.pop("material", None)
                geom.set("rgba", "0.86 0.86 0.82 1")
            elif geom.attrib.get("type") == "plane":
                geom.attrib.pop("material", None)
                geom.set("rgba", "0.88 0.88 0.84 0.35")
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    wrapper._debug_camera_wrapper = True  # type: ignore[attr-defined]
    nexus.build_mycobot_nexus_scene_model = wrapper


def _initialize_case(env: nexus.MyCobotNexusEnv, case: final_diag.FinalDiagnosticCase) -> None:
    env.reset(seed=1)
    gate7 = nexus._adaptive_gate7_arm_qpos(case.model_profile)
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


def _run_phase(
    env: nexus.MyCobotNexusEnv,
    renderers: list[Any],
    records: list[dict[str, Any]],
    frame_paths: list[Path],
    frame_scores: list[dict[str, Any]],
    frames_dir: Path,
    case: final_diag.FinalDiagnosticCase,
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
        record["diagnostics"] = _diagnostics(env)
        records.append(record)
        if step_index % max(render_every, 1) == 0:
            _capture_debug_frame(
                env, renderers, record, case, frames_dir / f"frame_{step_index:05d}_{phase}.png",
                frame_paths, frame_scores,
            )
        step_index += 1
    return step_index


def _diagnostics(env: nexus.MyCobotNexusEnv) -> dict[str, Any]:
    cube = np.asarray(env._cube_position(), dtype=float)
    tcp = np.asarray(env._tcp_position(), dtype=float)
    pad_mid = np.asarray(env._finger_pad_midpoint(), dtype=float)
    return {
        "cube_position": cube.tolist(),
        "tcp_position": tcp.tolist(),
        "pad_midpoint": pad_mid.tolist(),
        "tcp_to_cube": float(np.linalg.norm(tcp - cube)),
        "pad_midpoint_to_cube": float(np.linalg.norm(pad_mid - cube)),
    }


def _capture_debug_frame(
    env: nexus.MyCobotNexusEnv,
    renderers: list[Any],
    record: dict[str, Any],
    case: final_diag.FinalDiagnosticCase,
    path: Path,
    frame_paths: list[Path],
    frame_scores: list[dict[str, Any]],
) -> None:
    best_rgb, best_meta = _render_best_candidate(env, renderers[0])
    side_rgb, side_meta = _render_named_view(env, renderers[1], "side")
    top_rgb, top_meta = _render_named_view(env, renderers[2], "top")
    best = cv2.cvtColor(best_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    side = cv2.cvtColor(side_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    top = cv2.cvtColor(top_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    canvas = np.full((1440, 1920, 3), 246, np.uint8)
    canvas[0:720, 0:960] = best
    canvas[0:720, 960:1920] = side
    canvas[720:1200, 0:640] = cv2.resize(top, (640, 480), interpolation=cv2.INTER_AREA)
    _draw_overlay(canvas, record, case, best_meta, side_meta, top_meta)
    cv2.imwrite(str(path), canvas)
    frame_paths.append(path)
    frame_scores.append({
        "step": int(record["step"]),
        "phase": record["phase"],
        "best_camera": best_meta,
        "side_camera": side_meta,
        "top_camera": top_meta,
        "contact_pads": int(record["info"].get("gripper_cube_contact_pads", 0)),
        "contact_points": int(record["info"].get("gripper_cube_contacts", 0)),
    })


def _render_best_candidate(env: nexus.MyCobotNexusEnv, renderer: Any) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = []
    for azimuth in (30.0, 60.0, 90.0, 120.0, 150.0):
        for elevation in (-20.0, 0.0, 20.0):
            for distance in (1.1, 1.6):
                candidates.append(("auto", azimuth, elevation, distance))
    best_rgb = None
    best_meta: dict[str, Any] | None = None
    for name, azimuth, elevation, distance in candidates:
        rgb, meta = _render_camera(env, renderer, name, azimuth, elevation, distance)
        if best_meta is None or meta["visibility_score"] > best_meta["visibility_score"]:
            best_rgb = rgb
            best_meta = meta
    assert best_rgb is not None and best_meta is not None
    best_meta["name"] = "auto_best"
    return best_rgb, best_meta


def _render_named_view(env: nexus.MyCobotNexusEnv, renderer: Any, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "top":
        if env.config.model_profile == nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
            return _render_camera(env, renderer, "top", 90.0, -80.0, 1.6)
        return _render_camera(env, renderer, "top", 90.0, -84.0, 0.34)
    if env.config.model_profile == nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER:
        return _render_camera(env, renderer, "side", 60.0, -20.0, 1.6)
    return _render_camera(env, renderer, "side", 45.0, -18.0, 0.28)


def _render_camera(
    env: nexus.MyCobotNexusEnv,
    renderer: Any,
    name: str,
    azimuth: float,
    elevation: float,
    distance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    camera = env._mujoco.MjvCamera()
    camera.type = env._mujoco.mjtCamera.mjCAMERA_FREE
    cube = np.asarray(env._cube_position(), dtype=float)
    pad = np.asarray(env._finger_pad_midpoint(), dtype=float)
    target = 0.50 * cube + 0.50 * pad
    target[2] += 0.01
    camera.lookat[:] = target.tolist()
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    renderer.update_scene(env.data, camera=camera)
    rgb = renderer.render()
    meta = {
        "name": name,
        "azimuth": azimuth,
        "elevation": elevation,
        "distance": distance,
        **_visibility_score(rgb),
    }
    return rgb, meta


def _visibility_score(rgb: np.ndarray) -> dict[str, Any]:
    red = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)
    green = (rgb[:, :, 1] > 150) & (rgb[:, :, 0] < 120) & (rgb[:, :, 2] < 130)
    blue = (rgb[:, :, 2] > 150) & (rgb[:, :, 0] < 120) & (rgb[:, :, 1] < 150)
    red_count = int(red.sum())
    green_count = int(green.sum())
    blue_count = int(blue.sum())
    present = sum(1 for value in (red_count, green_count, blue_count) if value > 30)
    score = min(red_count, 4000) + min(green_count, 4000) + min(blue_count, 4000) + present * 3000
    return {
        "red_cube_pixels": red_count,
        "green_left_pad_pixels": green_count,
        "blue_right_pad_pixels": blue_count,
        "visible_debug_object_count": present,
        "visibility_score": int(score),
    }


def _draw_overlay(
    canvas: np.ndarray,
    record: dict[str, Any],
    case: final_diag.FinalDiagnosticCase,
    best_meta: dict[str, Any],
    side_meta: dict[str, Any],
    top_meta: dict[str, Any],
) -> None:
    cv2.rectangle(canvas, (0, 1200), (1920, 1440), (255, 255, 255), -1)
    _label_panel(canvas, "auto-selected visible camera", (24, 44), best_meta)
    _label_panel(canvas, "fixed side camera", (984, 44), side_meta)
    _label_panel(canvas, "fixed top camera", (24, 764), top_meta)
    cv2.line(canvas, (960, 0), (960, 720), (40, 40, 40), 2)
    cv2.line(canvas, (0, 720), (1920, 720), (40, 40, 40), 2)
    info = record["info"]
    diag = record["diagnostics"]
    pads = int(info.get("gripper_cube_contact_pads", 0))
    contacts = int(info.get("gripper_cube_contacts", 0))
    lift = float(info.get("cube_lift", 0.0))
    pad_dist = float(diag["pad_midpoint_to_cube"])
    tcp_dist = float(diag["tcp_to_cube"])
    color = (35, 150, 60) if pads >= 2 else (35, 165, 220) if pads == 1 else (45, 45, 220)
    cv2.putText(canvas, f"{case.name} | phase={record['phase']} | step={record['step']}",
                (36, 1250), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"pads touching cube: {pads}    contact points: {contacts}    cube lift: {lift:+.4f} m",
                (36, 1306), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
    cv2.putText(canvas, f"tcp->cube: {tcp_dist:.4f} m    pad-midpoint->cube: {pad_dist:.4f} m    gripper command: {float(info.get('gripper_command', 0.0)):+.3f}",
                (36, 1360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (45, 45, 45), 2, cv2.LINE_AA)
    cv2.putText(canvas, "debug colors: cube=red, left contact pad=green, right contact pad=blue",
                (1030, 1250), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, _verdict(pads, contacts, lift, pad_dist),
                (1030, 1320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA)


def _label_panel(canvas: np.ndarray, text: str, pos: tuple[int, int], meta: dict[str, Any]) -> None:
    x, y = pos
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    detail = (
        f"score={meta['visibility_score']} red={meta['red_cube_pixels']} "
        f"green={meta['green_left_pad_pixels']} blue={meta['blue_right_pad_pixels']} "
        f"az={meta['azimuth']:.0f} el={meta['elevation']:.0f} d={meta['distance']:.2f}"
    )
    cv2.putText(canvas, detail, (x, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA)


def _verdict(pads: int, contacts: int, lift: float, pad_dist: float) -> str:
    if pads >= 2:
        return "TOUCHING: both collision pads are in contact"
    if contacts > 0:
        return "PARTIAL TOUCH: contact exists, but not both pads"
    if pad_dist < 0.035:
        return "NEAR: pads are close, but contact is gone"
    if lift > 0.025:
        return "NOT A GRASP: cube moved without retained pad contact"
    return "NO TOUCH: no pad contact"


def _contact_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    touch_steps = [r for r in records if int(r["info"].get("gripper_cube_contacts", 0)) > 0]
    two_pad_steps = [r for r in records if int(r["info"].get("gripper_cube_contact_pads", 0)) >= 2]
    final = records[-1]
    return {
        "ever_touched": bool(touch_steps),
        "ever_two_pad_touch": bool(two_pad_steps),
        "first_touch_step": touch_steps[0]["step"] if touch_steps else None,
        "first_two_pad_touch_step": two_pad_steps[0]["step"] if two_pad_steps else None,
        "last_two_pad_touch_step": two_pad_steps[-1]["step"] if two_pad_steps else None,
        "final_pads": int(final["info"].get("gripper_cube_contact_pads", 0)),
        "final_contacts": int(final["info"].get("gripper_cube_contacts", 0)),
        "final_cube_lift": float(final["info"].get("cube_lift", 0.0)),
        "min_pad_midpoint_to_cube": min(float(r["diagnostics"]["pad_midpoint_to_cube"]) for r in records),
    }


def _visibility_summary(frame_scores: list[dict[str, Any]]) -> dict[str, Any]:
    best = [score["best_camera"] for score in frame_scores]
    return {
        "min_best_visibility_score": min(int(item["visibility_score"]) for item in best),
        "median_best_visibility_score": float(np.median([int(item["visibility_score"]) for item in best])),
        "min_best_visible_debug_object_count": min(int(item["visible_debug_object_count"]) for item in best),
        "frames_with_all_debug_objects_visible": sum(
            1 for item in best if int(item["visible_debug_object_count"]) >= 3
        ),
    }


def _write_event_sheet(case_dir: Path, records: list[dict[str, Any]], frame_paths: list[Path]) -> Path:
    touch = [r for r in records if int(r["info"].get("gripper_cube_contacts", 0)) > 0]
    two = [r for r in records if int(r["info"].get("gripper_cube_contact_pads", 0)) >= 2]
    first_lost = None
    if two:
        last_two_step = int(two[-1]["step"])
        for record in records:
            if int(record["step"]) > last_two_step and int(record["info"].get("gripper_cube_contact_pads", 0)) == 0:
                first_lost = record
                break
    picks = [
        ("start", records[0]),
        ("first touch", touch[0] if touch else records[0]),
        ("first two-pad", two[0] if two else records[0]),
        ("last two-pad", two[-1] if two else records[-1]),
        ("first lost", first_lost or records[-1]),
        ("final", records[-1]),
    ]
    sheet = np.full((2 * 360 + 90, 3 * 640, 3), 246, np.uint8)
    cv2.putText(sheet, case_dir.name, (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (20, 20, 20), 3, cv2.LINE_AA)
    for index, (title, record) in enumerate(picks):
        frame = cv2.imread(str(_nearest_frame(frame_paths, int(record["step"]))))
        tile = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        pads = int(record["info"].get("gripper_cube_contact_pads", 0))
        contacts = int(record["info"].get("gripper_cube_contacts", 0))
        lift = float(record["info"].get("cube_lift", 0.0))
        color = (35, 150, 60) if pads >= 2 else (35, 165, 220) if pads == 1 else (45, 45, 220)
        cv2.rectangle(tile, (0, 0), (640, 64), (255, 255, 255), -1)
        cv2.putText(tile, f"{title} | step {record['step']} | {record['phase']}",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(tile, f"pads {pads} contacts {contacts} lift {lift:+.4f}m",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
        y = 90 + (index // 3) * 360
        x = (index % 3) * 640
        sheet[y:y + 360, x:x + 640] = tile
    path = case_dir.parent / f"{case_dir.name}_debug_event_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def _nearest_frame(frame_paths: list[Path], step: int) -> Path:
    def frame_step(path: Path) -> int:
        return int(path.name.split("_")[1])
    return min(frame_paths, key=lambda path: abs(frame_step(path) - step))


def _write_mp4(frame_paths: list[Path], path: Path, *, fps: int) -> None:
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            writer.write(frame)
    writer.release()


if __name__ == "__main__":
    main()
