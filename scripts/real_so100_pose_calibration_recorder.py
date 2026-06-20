#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.real_so100_micro_step import _make_so100_bus
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")
DEFAULT_OUTPUT_ROOT = Path("_workspace/real_so100/pose_calibration")
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass(frozen=True)
class RecorderConfig:
    port: str
    output_dir: Path
    calibration: Path
    camera_indexes: list[int]
    discover_cameras: bool
    max_camera_index: int
    fps: float
    motor_hz: float
    duration_seconds: float | None
    primary_dedupe_camera: int | None
    hash_distance_threshold: int
    motor_distance_threshold: float
    max_selected_samples: int
    tts: bool
    start_phrase: str
    stop_phrase: str
    stop_file: Path | None
    interactive: bool
    synthetic: bool
    synthetic_cameras: int
    execute_random_motion: bool
    motion_strategy: str
    human_confirmed: bool
    workspace_clear_confirmed: bool
    random_motion_period_seconds: float
    frame_after_motion_delay_seconds: float
    random_step_fraction: float
    sweep_max_delta_raw: float
    random_seed: int


def run_pose_calibration_recorder(config: RecorderConfig) -> dict[str, Any]:
    session = _new_session_dir(config.output_dir)
    raw_dir = session / "raw"
    frames_dir = raw_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    camera_indexes = config.camera_indexes
    if config.synthetic and not camera_indexes:
        camera_indexes = list(range(config.synthetic_cameras))
    elif config.discover_cameras and not camera_indexes:
        camera_indexes = discover_cameras(max_index=config.max_camera_index)
    if not camera_indexes:
        raise ValueError("No camera indexes configured or discovered.")

    manifest: dict[str, Any] = {
        "operation": "real_so100_pose_calibration_recording",
        "session_dir": str(session),
        "started_at": _wall_time(),
        "port": config.port,
        "calibration": str(config.calibration),
        "camera_indexes": camera_indexes,
        "fps": config.fps,
        "motor_hz": config.motor_hz,
        "duration_seconds": config.duration_seconds,
        "frame_after_motion_delay_seconds": config.frame_after_motion_delay_seconds,
        "synthetic": config.synthetic,
        "execute_random_motion": config.execute_random_motion,
        "motion_strategy": config.motion_strategy,
        "human_confirmed": config.human_confirmed,
        "workspace_clear_confirmed": config.workspace_clear_confirmed,
        "writes_intended": bool(config.execute_random_motion),
        "send_action_called": False,
        "policy_actions_executed": False,
        "disconnect_disable_torque": False,
        "status": "started",
    }
    _write_json(session / "manifest_started.json", manifest)

    _speak(config, "포즈 캘리브레이션 시작합니다")

    if config.synthetic:
        raw_manifest = _record_synthetic(config=config, session=session, camera_indexes=camera_indexes)
    else:
        raw_manifest = _record_live(config=config, session=session, camera_indexes=camera_indexes)

    aligned = align_recording(session)
    selected = select_diverse_samples(
        session=session,
        primary_camera=config.primary_dedupe_camera,
        hash_distance_threshold=config.hash_distance_threshold,
        motor_distance_threshold=config.motor_distance_threshold,
        max_selected_samples=config.max_selected_samples,
    )
    html_path = build_html_report(session=session, manifest={**manifest, **raw_manifest}, aligned=aligned, selected=selected)

    manifest.update(raw_manifest)
    manifest.update(
        {
            "finished_at": _wall_time(),
            "aligned_samples": aligned["aligned_sample_count"],
            "selected_samples": selected["selected_sample_count"],
            "html": str(html_path),
            "status": "passed" if raw_manifest.get("ok") else "failed",
        }
    )
    _write_json(session / "manifest.json", manifest)
    return manifest


def _record_live(*, config: RecorderConfig, session: Path, camera_indexes: list[int]) -> dict[str, Any]:
    import cv2

    if config.execute_random_motion and not (config.human_confirmed and config.workspace_clear_confirmed):
        raise ValueError("--execute-random-motion requires --human-confirmed and --workspace-clear-confirmed.")

    raw_dir = session / "raw"
    frames_dir = raw_dir / "frames"
    frame_index_path = raw_dir / "frame_index.jsonl"
    motor_path = raw_dir / "motor_states.jsonl"
    events_path = raw_dir / "events.jsonl"
    frame_index_path.write_text("", encoding="utf-8")
    motor_path.write_text("", encoding="utf-8")
    events_path.write_text("", encoding="utf-8")

    captures = _open_cameras(camera_indexes)
    bus, motors = _make_so100_bus(config.port)
    calibration = _load_calibration(config.calibration)
    rng = random.Random(config.random_seed)
    sweep_targets: list[dict[str, int]] = []
    sweep_target_index = 0
    frame_period = 1.0 / config.fps
    motor_period = 1.0 / config.motor_hz
    deadline = None if config.duration_seconds is None else time.monotonic() + config.duration_seconds
    stop_requested = False
    target_by_joint: dict[str, int] | None = None
    next_frame_at = time.monotonic()
    next_motor_at = time.monotonic()
    next_random_motion_at = time.monotonic() + config.random_motion_period_seconds
    suppress_frames_until = 0.0
    frame_count = 0
    motor_count = 0
    skipped_frame_sets_after_motion = 0
    random_motion_count = 0
    send_action_called = False
    home_return_report: dict[str, Any] | None = None
    ok = False
    error = None

    try:
        bus.connect(handshake=True)
        if config.execute_random_motion and config.motion_strategy == "calibration_sweep":
            initial = _read_positions(bus)
            sweep_targets = _build_calibration_sweep_targets(
                current=initial,
                calibration=calibration,
                max_delta_raw=config.sweep_max_delta_raw,
            )
            _append_jsonl(
                events_path,
                {
                    "t": time.monotonic(),
                    "wall_time": _wall_time(),
                    "event": "calibration_sweep_targets_built",
                    "target_count": len(sweep_targets),
                    "sweep_max_delta_raw": config.sweep_max_delta_raw,
                },
            )
        while not stop_requested:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            if config.interactive and _stdin_has_line():
                line = sys.stdin.readline().strip()
                if config.stop_phrase in line:
                    stop_requested = True
                    _append_jsonl(events_path, {"t": now, "wall_time": _wall_time(), "event": "stop_phrase_received", "line": line})
                    continue
            if config.stop_file is not None and config.stop_file.exists():
                stop_requested = True
                _append_jsonl(
                    events_path,
                    {
                        "t": now,
                        "wall_time": _wall_time(),
                        "event": "stop_file_detected",
                        "stop_file": str(config.stop_file),
                    },
                )
                continue

            if now >= next_motor_at:
                state = _read_positions(bus)
                _append_jsonl(
                    motor_path,
                    {
                        "sample_index": motor_count,
                        "t": now,
                        "wall_time": _wall_time(),
                        "positions_raw": state,
                        "torque_enable": _read_torque(bus, motors),
                    },
                )
                motor_count += 1
                next_motor_at += motor_period

            if config.execute_random_motion and now >= next_random_motion_at:
                current = _read_positions(bus)
                if config.motion_strategy == "calibration_sweep":
                    if not sweep_targets:
                        stop_requested = True
                        _append_jsonl(
                            events_path,
                            {
                                "t": now,
                                "wall_time": _wall_time(),
                                "event": "calibration_sweep_completed_empty",
                            },
                        )
                        continue
                    if sweep_target_index >= len(sweep_targets):
                        stop_requested = True
                        _append_jsonl(
                            events_path,
                            {
                                "t": now,
                                "wall_time": _wall_time(),
                                "event": "calibration_sweep_completed",
                                "target_count": len(sweep_targets),
                            },
                        )
                        continue
                    target_by_joint = sweep_targets[sweep_target_index]
                    sweep_target_index += 1
                else:
                    target_by_joint = _random_target_near_current(
                        current=current,
                        calibration=calibration,
                        rng=rng,
                        step_fraction=config.random_step_fraction,
                    )
                bus.sync_write("Goal_Position", target_by_joint, normalize=False, num_retry=3)
                send_action_called = True
                random_motion_count += 1
                _append_jsonl(
                    events_path,
                    {
                        "t": now,
                        "wall_time": _wall_time(),
                        "event": "random_motion_target_sent",
                        "motion_strategy": config.motion_strategy,
                        "sweep_target_index": sweep_target_index - 1 if config.motion_strategy == "calibration_sweep" else None,
                        "target_raw": target_by_joint,
                        "frame_after_motion_delay_seconds": config.frame_after_motion_delay_seconds,
                    },
                )
                if config.frame_after_motion_delay_seconds > 0:
                    suppress_frames_until = max(
                        suppress_frames_until,
                        time.monotonic() + config.frame_after_motion_delay_seconds,
                    )
                next_random_motion_at += config.random_motion_period_seconds

            if now >= next_frame_at:
                if now < suppress_frames_until:
                    skipped_frame_sets_after_motion += 1
                    _append_jsonl(
                        events_path,
                        {
                            "t": now,
                            "wall_time": _wall_time(),
                            "event": "frame_set_skipped_after_motion",
                            "resume_at": suppress_frames_until,
                        },
                    )
                    next_frame_at += frame_period
                    time.sleep(0.002)
                    continue
                for camera_index, cap in captures.items():
                    grabbed_at = time.monotonic()
                    ok_read, frame = cap.read()
                    if not ok_read or frame is None:
                        _append_jsonl(
                            frame_index_path,
                            {
                                "camera": camera_index,
                                "frame": frame_count,
                                "t": grabbed_at,
                                "wall_time": _wall_time(),
                                "image": None,
                                "ok": False,
                            },
                        )
                        continue
                    camera_dir = frames_dir / f"camera_{camera_index}"
                    camera_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = camera_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    _append_jsonl(
                        frame_index_path,
                        {
                            "camera": camera_index,
                            "frame": frame_count,
                            "t": grabbed_at,
                            "wall_time": _wall_time(),
                            "image": str(frame_path),
                            "shape": list(frame.shape),
                            "ok": True,
                        },
                    )
                frame_count += 1
                next_frame_at += frame_period
            time.sleep(0.002)
        ok = True
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        for cap in captures.values():
            cap.release()
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            error = f"{error}; disconnect_error={exc!r}" if error else f"disconnect_error={exc!r}"
        if config.execute_random_motion and send_action_called:
            home_return_report = move_to_home_pose(
                port=config.port,
                calibration=config.calibration,
                home_pose=DEFAULT_HOME_POSE,
                output=session / "home_return_after_pose_calibration.json",
                execute=True,
                human_confirmed=config.human_confirmed,
                workspace_clear_confirmed=config.workspace_clear_confirmed,
                max_abs_delta_raw=120.0,
                step_settle_seconds=0.10,
                camera_index=camera_indexes[0] if camera_indexes else None,
                visual_output_dir=session / "home_return_visual",
                record_video=True,
                video_fps=max(4.0, min(config.fps, 12.0)),
            )

    return {
        "ok": ok,
        "error": error,
        "frame_sets_recorded": frame_count,
        "frame_sets_skipped_after_motion": skipped_frame_sets_after_motion,
        "motor_samples_recorded": motor_count,
        "random_motion_targets_sent": random_motion_count,
        "motion_strategy": config.motion_strategy,
        "sweep_targets_total": len(sweep_targets),
        "send_action_called": send_action_called,
        "disconnect_disable_torque": False,
        "home_return_report": str(session / "home_return_after_pose_calibration.json") if home_return_report else None,
        "home_return_status": home_return_report.get("status") if home_return_report else None,
        "post_task_torque_disabled": home_return_report.get("post_task_torque_disabled") if home_return_report else None,
        "raw_frame_index": str(frame_index_path),
        "raw_motor_states": str(motor_path),
    }


def _record_synthetic(*, config: RecorderConfig, session: Path, camera_indexes: list[int]) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    raw_dir = session / "raw"
    frames_dir = raw_dir / "frames"
    frame_index_path = raw_dir / "frame_index.jsonl"
    motor_path = raw_dir / "motor_states.jsonl"
    events_path = raw_dir / "events.jsonl"
    frame_index_path.write_text("", encoding="utf-8")
    motor_path.write_text("", encoding="utf-8")
    events_path.write_text("", encoding="utf-8")
    rng = random.Random(config.random_seed)
    calibration = _load_calibration(config.calibration) if config.calibration.exists() else _default_synthetic_calibration()
    duration = config.duration_seconds or 3.0
    frame_count = max(1, int(round(duration * config.fps)))
    motor_count = max(1, int(round(duration * config.motor_hz)))
    start = time.monotonic()

    synthetic_states = []
    for sample_index in range(motor_count):
        alpha = sample_index / max(motor_count - 1, 1)
        state = {}
        for joint in JOINT_ORDER:
            item = calibration[joint]
            center = (item["range_min"] + item["range_max"]) / 2
            amplitude = (item["range_max"] - item["range_min"]) * 0.28
            value = center + math.sin(alpha * math.tau * (1.0 + JOINT_ORDER.index(joint) * 0.13)) * amplitude
            state[joint] = int(round(max(item["range_min"], min(item["range_max"], value))))
        t = start + sample_index / config.motor_hz
        synthetic_states.append((t, state))
        _append_jsonl(
            motor_path,
            {
                "sample_index": sample_index,
                "t": t,
                "wall_time": _wall_time(),
                "positions_raw": state,
                "torque_enable": {joint: 0 for joint in JOINT_ORDER},
            },
        )

    for frame_index in range(frame_count):
        alpha = frame_index / max(frame_count - 1, 1)
        for camera_index in camera_indexes:
            camera_dir = frames_dir / f"camera_{camera_index}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            frame_path = camera_dir / f"frame_{frame_index:06d}.jpg"
            image = Image.new("RGB", (320, 240), (22 + camera_index * 16, 25, 31))
            draw = ImageDraw.Draw(image)
            x = int(40 + alpha * 220 + camera_index * 9 + rng.randint(-2, 2))
            y = int(70 + math.sin(alpha * math.tau + camera_index) * 38)
            draw.rectangle((x, y, x + 44, y + 32), fill=(40, 180, 78), outline=(230, 230, 230))
            draw.line((160, 230, x + 22, y + 16), fill=(180, 180, 190), width=4)
            draw.text((12, 12), f"cam {camera_index} frame {frame_index}", fill=(230, 230, 230))
            image.save(frame_path)
            _append_jsonl(
                frame_index_path,
                {
                    "camera": camera_index,
                    "frame": frame_index,
                    "t": start + frame_index / config.fps,
                    "wall_time": _wall_time(),
                    "image": str(frame_path),
                    "shape": [240, 320, 3],
                    "ok": True,
                },
            )
    return {
        "ok": True,
        "frame_sets_recorded": frame_count,
        "motor_samples_recorded": motor_count,
        "random_motion_targets_sent": 0,
        "send_action_called": False,
        "raw_frame_index": str(frame_index_path),
        "raw_motor_states": str(motor_path),
        "synthetic_note": "Synthetic images and motor states for recorder pipeline validation.",
    }


def align_recording(session: Path) -> dict[str, Any]:
    raw_dir = session / "raw"
    frame_rows = _read_jsonl(raw_dir / "frame_index.jsonl")
    motor_rows = _read_jsonl(raw_dir / "motor_states.jsonl")
    motor_rows = [row for row in motor_rows if isinstance(row.get("positions_raw"), dict)]
    frames_by_index: dict[int, list[dict[str, Any]]] = {}
    for row in frame_rows:
        if not row.get("ok") or row.get("image") is None:
            continue
        frames_by_index.setdefault(int(row["frame"]), []).append(row)

    aligned_path = session / "aligned_samples.jsonl"
    aligned_path.write_text("", encoding="utf-8")
    max_sync_error_ms = 0.0
    count = 0
    for frame_index in sorted(frames_by_index):
        rows = frames_by_index[frame_index]
        sample_t = sum(float(row["t"]) for row in rows) / len(rows)
        nearest = _nearest_motor_row(motor_rows, sample_t)
        if nearest is None:
            continue
        sync_error_ms = abs(float(nearest["t"]) - sample_t) * 1000.0
        max_sync_error_ms = max(max_sync_error_ms, sync_error_ms)
        sample = {
            "sample_index": count,
            "source_frame_index": frame_index,
            "t": sample_t,
            "cameras": {str(row["camera"]): row["image"] for row in rows},
            "camera_timestamps": {str(row["camera"]): row["t"] for row in rows},
            "motor_state_nearest": nearest["positions_raw"],
            "motor_t": nearest["t"],
            "sync_error_ms": round(sync_error_ms, 3),
        }
        _append_jsonl(aligned_path, sample)
        count += 1
    report = {
        "aligned_samples": str(aligned_path),
        "aligned_sample_count": count,
        "motor_sample_count": len(motor_rows),
        "max_sync_error_ms": round(max_sync_error_ms, 3),
    }
    _write_json(session / "alignment_report.json", report)
    return report


def select_diverse_samples(
    *,
    session: Path,
    primary_camera: int | None,
    hash_distance_threshold: int,
    motor_distance_threshold: float,
    max_selected_samples: int,
) -> dict[str, Any]:
    selected_dir = session / "selected"
    selected_frames_dir = selected_dir / "frames"
    selected_frames_dir.mkdir(parents=True, exist_ok=True)
    selected_path = selected_dir / "selected_samples.jsonl"
    rejected_path = selected_dir / "rejected_similar_samples.jsonl"
    selected_path.write_text("", encoding="utf-8")
    rejected_path.write_text("", encoding="utf-8")
    samples = _read_jsonl(session / "aligned_samples.jsonl")
    selected: list[dict[str, Any]] = []
    selected_hashes: list[int] = []
    selected_motors: list[dict[str, float]] = []

    for sample in samples:
        if len(selected) >= max_selected_samples:
            _append_jsonl(rejected_path, {**sample, "reject_reason": "max_selected_samples_reached"})
            continue
        image_path = _choose_primary_image(sample, primary_camera)
        if image_path is None:
            _append_jsonl(rejected_path, {**sample, "reject_reason": "no_primary_image"})
            continue
        image_hash = _average_hash(Path(image_path))
        motor_state = {name: float(value) for name, value in sample.get("motor_state_nearest", {}).items() if name in JOINT_ORDER}
        min_hash_distance = min((_hamming(image_hash, other) for other in selected_hashes), default=64)
        min_motor_distance = min((_motor_distance(motor_state, other) for other in selected_motors), default=1.0)
        keep = not selected or min_hash_distance >= hash_distance_threshold or min_motor_distance >= motor_distance_threshold
        enriched = {
            **sample,
            "primary_image": image_path,
            "image_hash": f"{image_hash:016x}",
            "min_hash_distance_to_selected": min_hash_distance,
            "min_motor_distance_to_selected": round(min_motor_distance, 6),
        }
        if keep:
            copied_cameras = {}
            for camera, path in sample.get("cameras", {}).items():
                src = Path(path)
                dst = selected_frames_dir / f"sample_{len(selected):04d}_camera_{camera}{src.suffix}"
                dst.write_bytes(src.read_bytes())
                copied_cameras[camera] = str(dst)
            enriched["selected_index"] = len(selected)
            enriched["selected_cameras"] = copied_cameras
            _append_jsonl(selected_path, enriched)
            selected.append(enriched)
            selected_hashes.append(image_hash)
            selected_motors.append(motor_state)
        else:
            enriched["reject_reason"] = "similar_visual_and_motor_state"
            _append_jsonl(rejected_path, enriched)

    report = {
        "selected_samples": str(selected_path),
        "rejected_samples": str(rejected_path),
        "selected_sample_count": len(selected),
        "input_aligned_sample_count": len(samples),
        "hash_distance_threshold": hash_distance_threshold,
        "motor_distance_threshold": motor_distance_threshold,
        "max_selected_samples": max_selected_samples,
    }
    _write_json(selected_dir / "selection_report.json", report)
    return report


def build_html_report(*, session: Path, manifest: dict[str, Any], aligned: dict[str, Any], selected: dict[str, Any]) -> Path:
    selected_rows = _read_jsonl(Path(selected["selected_samples"]))
    cards = []
    for row in selected_rows[:80]:
        images = []
        for camera, image_path in sorted(row.get("selected_cameras", {}).items(), key=lambda item: int(item[0])):
            images.append(
                f"<figure><img src='{html.escape(_rel(session, Path(image_path)))}' alt='camera {html.escape(camera)}'>"
                f"<figcaption>camera {html.escape(camera)}</figcaption></figure>"
            )
        motor = row.get("motor_state_nearest", {})
        motor_text = html.escape(json.dumps(motor, sort_keys=True))
        cards.append(
            "<article class='sample'>"
            f"<h3>sample {row.get('selected_index')}</h3>"
            f"<p>sync error: <b>{row.get('sync_error_ms')} ms</b> | "
            f"hash distance: {row.get('min_hash_distance_to_selected')} | "
            f"motor distance: {row.get('min_motor_distance_to_selected')}</p>"
            f"<div class='images'>{''.join(images)}</div>"
            f"<pre>{motor_text}</pre>"
            "</article>"
        )
    html_path = session / "pose_calibration_report.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SO-100 Pose Calibration Dataset</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f2; color: #202124; }}
    header {{ padding: 24px 32px; background: #25312c; color: #f7f7f2; }}
    main {{ padding: 24px 32px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .metric {{ background: #fff; border: 1px solid #d8d8d2; border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; font-size: 22px; }}
    .sample {{ background: #fff; border: 1px solid #d8d8d2; border-radius: 8px; padding: 14px; margin: 0 0 16px; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; border: 1px solid #ccc; border-radius: 6px; }}
    figcaption {{ font-size: 12px; color: #595959; margin-top: 4px; }}
    pre {{ overflow-x: auto; background: #f1f1ec; padding: 8px; border-radius: 6px; font-size: 12px; }}
    code {{ background: #ecece6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>SO-100 Pose Calibration Dataset</h1>
  <p>Images are aligned to nearest motor readback and filtered for visual or motor-state diversity.</p>
</header>
<main>
  <section class="metrics">
    <div class="metric"><b>{html.escape(str(manifest.get('status', 'unknown')))}</b><span>status</span></div>
    <div class="metric"><b>{html.escape(str(manifest.get('frame_sets_recorded', 0)))}</b><span>raw frame sets</span></div>
    <div class="metric"><b>{html.escape(str(aligned.get('aligned_sample_count', 0)))}</b><span>aligned samples</span></div>
    <div class="metric"><b>{html.escape(str(selected.get('selected_sample_count', 0)))}</b><span>selected diverse samples</span></div>
    <div class="metric"><b>{html.escape(str(aligned.get('max_sync_error_ms', 'n/a')))}</b><span>max sync error ms</span></div>
  </section>
  <section>
    <h2>Artifacts</h2>
    <p><code>{html.escape(str(session / 'aligned_samples.jsonl'))}</code></p>
    <p><code>{html.escape(str(selected.get('selected_samples')))}</code></p>
  </section>
  <section>{''.join(cards)}</section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def discover_cameras(*, max_index: int) -> list[int]:
    import cv2

    indexes = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    indexes.append(index)
        finally:
            cap.release()
    return indexes


def _open_cameras(indexes: list[int]) -> dict[int, Any]:
    import cv2

    captures = {}
    for index in indexes:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            for opened in captures.values():
                opened.release()
            raise RuntimeError(f"camera index {index} did not open")
        captures[index] = cap
    return captures


def _read_positions(bus: Any) -> dict[str, int]:
    values = bus.sync_read("Present_Position", normalize=False)
    return {name: int(value) for name, value in values.items()}


def _read_torque(bus: Any, motors: dict[str, Any]) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    for name in motors:
        try:
            result[name] = int(bus.read("Torque_Enable", name, normalize=False))
        except Exception as exc:  # noqa: BLE001
            result[name] = f"error: {exc!r}"
    return result


def _random_target_near_current(
    *,
    current: dict[str, int],
    calibration: dict[str, dict[str, float]],
    rng: random.Random,
    step_fraction: float,
) -> dict[str, int]:
    targets = {}
    for joint in JOINT_ORDER:
        item = calibration[joint]
        span = item["range_max"] - item["range_min"]
        delta = rng.uniform(-span * step_fraction, span * step_fraction)
        value = float(current[joint]) + delta
        targets[joint] = int(round(max(item["range_min"], min(item["range_max"], value))))
    return targets


def _build_calibration_sweep_targets(
    *,
    current: dict[str, int],
    calibration: dict[str, dict[str, float]],
    max_delta_raw: float,
) -> list[dict[str, int]]:
    if max_delta_raw <= 0:
        raise ValueError("sweep max delta must be positive")
    targets: list[dict[str, int]] = []
    simulated = {joint: int(current[joint]) for joint in JOINT_ORDER}
    center = {
        joint: int(round((calibration[joint]["range_min"] + calibration[joint]["range_max"]) / 2.0))
        for joint in JOINT_ORDER
    }
    for joint in JOINT_ORDER:
        for value in (
            calibration[joint]["range_min"],
            calibration[joint]["range_max"],
            center[joint],
        ):
            target = dict(simulated)
            target[joint] = int(round(value))
            for interpolated in _interpolate_joint_targets(
                current=simulated,
                target=target,
                max_delta_raw=max_delta_raw,
            ):
                targets.append(interpolated)
            simulated = dict(targets[-1]) if targets else target
    for interpolated in _interpolate_joint_targets(
        current=simulated,
        target=center,
        max_delta_raw=max_delta_raw,
    ):
        targets.append(interpolated)
    return targets


def _interpolate_joint_targets(
    *,
    current: dict[str, int],
    target: dict[str, int],
    max_delta_raw: float,
) -> list[dict[str, int]]:
    largest = max(abs(float(target[joint]) - float(current[joint])) for joint in JOINT_ORDER)
    steps = max(1, int(math.ceil(largest / max_delta_raw)))
    result = []
    for step in range(1, steps + 1):
        fraction = step / steps
        result.append(
            {
                joint: int(round(float(current[joint]) + ((float(target[joint]) - float(current[joint])) * fraction)))
                for joint in JOINT_ORDER
            }
        )
    return result


def _load_calibration(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for joint in JOINT_ORDER:
        item = payload.get(joint)
        if not isinstance(item, dict) or "range_min" not in item or "range_max" not in item:
            raise ValueError(f"calibration missing range for {joint}: {path}")
        result[joint] = {"range_min": float(item["range_min"]), "range_max": float(item["range_max"])}
    return result


def _default_synthetic_calibration() -> dict[str, dict[str, float]]:
    return {joint: {"range_min": 1000.0, "range_max": 3000.0} for joint in JOINT_ORDER}


def _nearest_motor_row(rows: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["t"]) - t))


def _choose_primary_image(sample: dict[str, Any], primary_camera: int | None) -> str | None:
    cameras = sample.get("cameras", {})
    if primary_camera is not None and str(primary_camera) in cameras:
        return cameras[str(primary_camera)]
    if not cameras:
        return None
    first_key = sorted(cameras, key=int)[0]
    return cameras[first_key]


def _average_hash(path: Path) -> int:
    from PIL import Image

    image = Image.open(path).convert("L").resize((8, 8))
    pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for index, pixel in enumerate(pixels):
        if pixel >= mean:
            value |= 1 << index
    return value


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _motor_distance(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 1.0
    diffs = []
    for joint in JOINT_ORDER:
        if joint in left and joint in right:
            diffs.append(abs(left[joint] - right[joint]) / 4095.0)
    return sum(diffs) / len(diffs) if diffs else 1.0


def _new_session_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = output_root / f"session_{stamp}"
    suffix = 1
    while session.exists():
        session = output_root / f"session_{stamp}_{suffix:02d}"
        suffix += 1
    session.mkdir(parents=True)
    return session


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _wall_time() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _rel(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _speak(config: RecorderConfig, text: str) -> None:
    if not config.tts:
        return
    try:
        subprocess.run(["say", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return


def _stdin_has_line() -> bool:
    import select

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(ready)


def _parse_args() -> RecorderConfig:
    parser = argparse.ArgumentParser(description="Record SO-100 pose calibration image+motor datasets.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--camera-index", type=int, action="append", default=[])
    parser.add_argument("--discover-cameras", action="store_true")
    parser.add_argument("--max-camera-index", type=int, default=6)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--motor-hz", type=float, default=30.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--primary-dedupe-camera", type=int)
    parser.add_argument("--hash-distance-threshold", type=int, default=6)
    parser.add_argument("--motor-distance-threshold", type=float, default=0.018)
    parser.add_argument("--max-selected-samples", type=int, default=200)
    parser.add_argument("--tts", action="store_true", help="Use macOS say for start notification.")
    parser.add_argument("--start-phrase", default="로봇 포즈 캘리브레이션 시작해줘")
    parser.add_argument("--stop-phrase", default="레코딩을 끝내줘")
    parser.add_argument("--stop-file", type=Path, help="Stop recording when this file appears.")
    parser.add_argument("--interactive", action="store_true", help="Wait for start phrase on stdin and stop on stop phrase.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic frames and motor states for tests.")
    parser.add_argument("--synthetic-cameras", type=int, default=2)
    parser.add_argument("--execute-random-motion", action="store_true")
    parser.add_argument(
        "--motion-strategy",
        choices=["random_near_current", "calibration_sweep"],
        default="random_near_current",
    )
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    parser.add_argument("--random-motion-period-seconds", type=float, default=1.0)
    parser.add_argument(
        "--frame-after-motion-delay-seconds",
        type=float,
        default=0.0,
        help="Skip saving camera frames for this long after each random motion target. Motor readback still records.",
    )
    parser.add_argument("--random-step-fraction", type=float, default=0.035)
    parser.add_argument("--sweep-max-delta-raw", type=float, default=120.0)
    parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()

    if args.interactive:
        for line in sys.stdin:
            if args.start_phrase in line.strip():
                break
        else:
            raise SystemExit("start phrase not received")

    return RecorderConfig(
        port=args.port,
        output_dir=args.output_dir,
        calibration=args.calibration,
        camera_indexes=args.camera_index,
        discover_cameras=args.discover_cameras,
        max_camera_index=args.max_camera_index,
        fps=args.fps,
        motor_hz=args.motor_hz,
        duration_seconds=args.duration_seconds,
        primary_dedupe_camera=args.primary_dedupe_camera,
        hash_distance_threshold=args.hash_distance_threshold,
        motor_distance_threshold=args.motor_distance_threshold,
        max_selected_samples=args.max_selected_samples,
        tts=args.tts,
        start_phrase=args.start_phrase,
        stop_phrase=args.stop_phrase,
        stop_file=args.stop_file,
        interactive=args.interactive,
        synthetic=args.synthetic,
        synthetic_cameras=args.synthetic_cameras,
        execute_random_motion=args.execute_random_motion,
        motion_strategy=args.motion_strategy,
        human_confirmed=args.human_confirmed,
        workspace_clear_confirmed=args.workspace_clear_confirmed,
        random_motion_period_seconds=args.random_motion_period_seconds,
        frame_after_motion_delay_seconds=args.frame_after_motion_delay_seconds,
        random_step_fraction=args.random_step_fraction,
        sweep_max_delta_raw=args.sweep_max_delta_raw,
        random_seed=args.random_seed,
    )


def main() -> None:
    config = _parse_args()
    result = run_pose_calibration_recorder(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
