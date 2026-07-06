#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FRAME_RE = re.compile(r"frame_(?P<step>\d+)_(?P<phase>[a-z_]+)\.png$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build annotated videos and storyboard/timeline images for myCobot raw Gate 8 diagnostics."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "visual_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((input_dir / "final_diagnostic_report.json").read_text(encoding="utf-8"))
    cases = summary["cases"]

    timeline_path = output_dir / "gate8_contact_timeline.png"
    _write_timeline(cases, timeline_path)

    storyboard_path = output_dir / "gate8_storyboard_320_vs_280.png"
    _write_storyboard(input_dir, cases, storyboard_path)

    annotated_paths = []
    for report in cases:
        path = _write_annotated_video(input_dir, output_dir, report, fps=args.fps)
        if path is not None:
            annotated_paths.append(str(path))

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "timeline": str(timeline_path),
        "storyboard": str(storyboard_path),
        "annotated_videos": annotated_paths,
    }
    manifest_path = output_dir / "visual_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _records(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(report["trace_path"]).read_text(encoding="utf-8").splitlines()]


def _write_timeline(cases: list[dict[str, Any]], path: Path) -> None:
    width = 1800
    row_h = 210
    left = 330
    right = 60
    top = 60
    bottom = 80
    height = top + bottom + row_h * len(cases)
    img = np.full((height, width, 3), 245, np.uint8)
    cv2.putText(img, "Raw Gate 8 contact timeline: green=two-pad contact, yellow=one-pad, red=no pad", (40, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    plot_w = width - left - right
    for idx, report in enumerate(cases):
        y0 = top + idx * row_h
        records = _records(report)
        if not records:
            continue
        n = max(len(records) - 1, 1)
        lifts = [float(r["info"].get("cube_lift", 0.0)) for r in records]
        max_lift = max(0.03, max(abs(v) for v in lifts))
        label = (
            f"{report['name']} | {report['status']} | close {report['close_best_sustained_contact_steps']} | "
            f"lift {report['lift_best_sustained_contact_steps']} | final pads {report['final_gripper_cube_contact_pads']} | "
            f"final lift {float(report['final_cube_lift']):+.4f}m"
        )
        cv2.putText(img, label, (35, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)

        bar_y = y0 + 54
        bar_h = 34
        for i, rec in enumerate(records):
            pads = int(rec["info"].get("gripper_cube_contact_pads", 0))
            color = (40, 170, 70) if pads >= 2 else (40, 190, 230) if pads == 1 else (55, 55, 220)
            x1 = left + int(i * plot_w / len(records))
            x2 = left + int((i + 1) * plot_w / len(records)) + 1
            cv2.rectangle(img, (x1, bar_y), (x2, bar_y + bar_h), color, -1)
        cv2.rectangle(img, (left, bar_y), (left + plot_w, bar_y + bar_h), (70, 70, 70), 1)
        cv2.putText(img, "pad contact", (35, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

        plot_y = y0 + 118
        plot_h = 60
        cv2.rectangle(img, (left, plot_y), (left + plot_w, plot_y + plot_h), (210, 210, 210), 1)
        zero_y = plot_y + plot_h // 2
        cv2.line(img, (left, zero_y), (left + plot_w, zero_y), (170, 170, 170), 1)
        pts = []
        for i, lift in enumerate(lifts):
            x = left + int(i * plot_w / n)
            y = int(zero_y - (lift / max_lift) * (plot_h * 0.45))
            y = max(plot_y + 2, min(plot_y + plot_h - 2, y))
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (200, 80, 30), 2, cv2.LINE_AA)
        cv2.putText(img, "cube lift", (35, plot_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

        phase_changes = []
        last = records[0]["phase"]
        for i, rec in enumerate(records):
            phase = rec["phase"]
            if phase != last:
                phase_changes.append((i, phase))
                last = phase
        for step_i, phase in phase_changes:
            x = left + int(step_i * plot_w / n)
            cv2.line(img, (x, bar_y - 8), (x, plot_y + plot_h + 8), (80, 80, 80), 1)
            cv2.putText(img, phase, (x + 4, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def _write_storyboard(input_dir: Path, cases: list[dict[str, Any]], path: Path) -> None:
    wanted = ["320_raw_reference", "280_best_stable_raw"]
    selected = [r for r in cases if r["name"] in wanted]
    cell_w, cell_h = 390, 280
    header_h = 70
    cols = 4
    rows = len(selected)
    img = np.full((header_h + rows * cell_h, cols * cell_w, 3), 245, np.uint8)
    cv2.putText(img, "Raw Gate 8 storyboard: 320 retains contact through lift; 280 loses contact before final lift",
                (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    for row, report in enumerate(selected):
        records = _records(report)
        frames = _frame_index(input_dir / report["name"] / "frames")
        chosen = _story_steps(records)
        for col, (title, step) in enumerate(chosen):
            frame_path = _nearest_frame(frames, step)
            frame = cv2.imread(str(frame_path)) if frame_path else None
            tile = _make_tile(frame, report, records[min(step, len(records) - 1)], title, cell_w, cell_h)
            y = header_h + row * cell_h
            x = col * cell_w
            img[y:y + cell_h, x:x + cell_w] = tile
    cv2.imwrite(str(path), img)


def _story_steps(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    close_steps = [i for i, r in enumerate(records) if r["phase"] == "close"]
    lift_steps = [i for i, r in enumerate(records) if r["phase"] == "lift"]
    two_pad_lift = [i for i in lift_steps if int(records[i]["info"].get("gripper_cube_contact_pads", 0)) >= 2]
    lost_after_lift = [
        i for i in lift_steps
        if int(records[i]["info"].get("gripper_cube_contact_pads", 0)) == 0
        and any(int(records[j]["info"].get("gripper_cube_contact_pads", 0)) >= 2 for j in lift_steps if j < i)
    ]
    return [
        ("start", 0),
        ("end close", close_steps[-1] if close_steps else 0),
        ("lift contact", two_pad_lift[min(len(two_pad_lift) - 1, 10)] if two_pad_lift else (lift_steps[0] if lift_steps else len(records) - 1)),
        ("final / lost", lost_after_lift[0] if lost_after_lift else len(records) - 1),
    ]


def _frame_index(frames_dir: Path) -> dict[int, Path]:
    out = {}
    for path in frames_dir.glob("frame_*.png"):
        match = FRAME_RE.match(path.name)
        if match:
            out[int(match.group("step"))] = path
    return out


def _nearest_frame(frames: dict[int, Path], step: int) -> Path | None:
    if not frames:
        return None
    key = min(frames, key=lambda candidate: abs(candidate - step))
    return frames[key]


def _make_tile(frame: np.ndarray | None, report: dict[str, Any], rec: dict[str, Any], title: str, width: int, height: int) -> np.ndarray:
    tile = np.full((height, width, 3), 238, np.uint8)
    if frame is not None:
        crop = _zoom_crop(frame)
        crop = cv2.resize(crop, (width, height - 76), interpolation=cv2.INTER_CUBIC)
        tile[56:height - 20, :] = crop
    pads = int(rec["info"].get("gripper_cube_contact_pads", 0))
    lift = float(rec["info"].get("cube_lift", 0.0))
    color = (45, 150, 65) if pads >= 2 else (40, 160, 220) if pads == 1 else (45, 45, 220)
    cv2.rectangle(tile, (0, 0), (width, 56), (255, 255, 255), -1)
    cv2.putText(tile, report["name"], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(tile, f"{title} | step {rec['step']} | pads {pads} | lift {lift:+.4f}m",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    return tile


def _write_annotated_video(input_dir: Path, output_dir: Path, report: dict[str, Any], fps: int) -> Path | None:
    records = _records(report)
    frames = sorted((input_dir / report["name"] / "frames").glob("frame_*.png"))
    if not frames:
        return None
    out_path = output_dir / f"{report['name']}_annotated_zoom.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (960, 620))
    if not writer.isOpened():
        return None
    for frame_path in frames:
        match = FRAME_RE.match(frame_path.name)
        step = int(match.group("step")) if match else 0
        rec = records[min(step, len(records) - 1)]
        frame = cv2.imread(str(frame_path))
        canvas = np.full((620, 960, 3), 245, np.uint8)
        zoom = cv2.resize(_zoom_crop(frame), (960, 500), interpolation=cv2.INTER_CUBIC)
        canvas[0:500, :] = zoom
        _draw_video_overlay(canvas, report, rec)
        writer.write(canvas)
    writer.release()
    return out_path


def _draw_video_overlay(canvas: np.ndarray, report: dict[str, Any], rec: dict[str, Any]) -> None:
    pads = int(rec["info"].get("gripper_cube_contact_pads", 0))
    contacts = int(rec["info"].get("gripper_cube_contacts", 0))
    lift = float(rec["info"].get("cube_lift", 0.0))
    phase = rec["phase"]
    color = (45, 150, 65) if pads >= 2 else (40, 160, 220) if pads == 1 else (45, 45, 220)
    cv2.rectangle(canvas, (0, 500), (960, 620), (255, 255, 255), -1)
    cv2.putText(canvas, f"{report['name']} | {report['status']} | phase {phase} | step {rec['step']}",
                (24, 532), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"pads touching cube: {pads} | contact points: {contacts} | cube lift: {lift:+.4f} m",
                (24, 568), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"final pads {report['final_gripper_cube_contact_pads']} | final lift {float(report['final_cube_lift']):+.4f} m",
                (24, 602), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (55, 55, 55), 1, cv2.LINE_AA)


def _zoom_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = int(w * 0.18)
    x1 = int(w * 0.98)
    y0 = int(h * 0.18)
    y1 = int(h * 0.92)
    return frame[y0:y1, x0:x1]


if __name__ == "__main__":
    main()
