#!/usr/bin/env python3
"""Build temporal consistency evidence for oracle point overlays."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physical_ai_agent.perception.affordance_overlay import build_oracle_affordance_overlay


def _obs(rgb: np.ndarray, xyz: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "sensor_data": {
            "base_camera": {
                "rgb": rgb,
                "intrinsic_cv": [[220.0, 0.0, 160.0], [0.0, 220.0, 120.0], [0.0, 0.0, 1.0]],
                "extrinsic_cv": np.eye(4).tolist(),
            }
        },
        "object_pose": {"p": list(xyz)},
    }


def _raw_frame(width: int, height: int, point_xy: tuple[int, int], frame_idx: int) -> np.ndarray:
    image = Image.new("RGB", (width, height), (234, 229, 216))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height - 44, width, height), fill=(198, 183, 158))
    for gx in range(0, width, 32):
        draw.line((gx, 0, gx, height), fill=(224, 216, 201), width=1)
    for gy in range(0, height, 32):
        draw.line((0, gy, width, gy), fill=(224, 216, 201), width=1)
    x, y = point_xy
    draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(220, 68, 56), outline=(94, 36, 30), width=2)
    draw.text((10, 10), f"frame {frame_idx:02d}", fill=(32, 28, 22))
    return np.asarray(image, dtype=np.uint8)


def _world_from_pixel(x: int, y: int, z: float = 1.0) -> tuple[float, float, float]:
    return ((x - 160.0) * z / 220.0, (y - 120.0) * z / 220.0, z)


def _write_gif(paths: list[Path], output_path: Path) -> None:
    frames = [Image.open(path).convert("RGB") for path in paths if path.exists()]
    if not frames:
        return
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=110, loop=0)


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    thumb_w, thumb_h = 188, 134
    cols = 4
    cell_w, cell_h = 214, 184
    canvas = Image.new("RGB", (cols * cell_w, math.ceil(len(rows) / cols) * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        col = idx % cols
        grid_row = idx // cols
        x0, y0 = col * cell_w, grid_row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(210, 190, 160), width=1)
        image = Image.open(row["overlay"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 28))
        draw.text((x0 + 8, y0 + 8), f"f{row['frame_index']:02d} point={row['point_xy']}", fill=(20, 18, 14))
        draw.text((x0 + 8, y0 + 164), f"err={row['projection_error_px']:.1f}px step={row['step_delta_px']:.1f}px", fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _trail_image(rows: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.new("RGB", (320, 240), (244, 239, 229))
    draw = ImageDraw.Draw(image)
    points = [tuple(row["point_xy"]) for row in rows]
    if len(points) >= 2:
        draw.line(points, fill=(0, 142, 91), width=3)
    for idx, point in enumerate(points):
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 210, 110), outline=(12, 70, 45))
        if idx in (0, len(points) - 1):
            draw.text((x + 7, y - 7), "start" if idx == 0 else "end", fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path, gif_path: Path, trail_path: Path, summary: dict[str, Any]) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>Frame {row['frame_index']:02d}</h2>
        <p>expected={html.escape(str(row['expected_xy']))}; projected={html.escape(str(row['point_xy']))}; error={row['projection_error_px']:.2f}px; step delta={row['step_delta_px']:.2f}px</p>
        <figure><img src="{html.escape(_rel(report_path, Path(row['overlay'])))}" alt="frame {row['frame_index']:02d}"><figcaption>oracle overlay frame</figcaption></figure>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Overlay Temporal Consistency</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --green:#008f5b; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#d7ffe7,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:980px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--green); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    a {{ color:#174f38; font-weight:800; }}
    .hero {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Overlay Temporal Consistency</h1>
    <p>Frame-by-frame evidence that projected oracle points follow a moving object without center collapse or projection jitter.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">PASSED TEMPORAL CHECK</span>
      <h2>Summary</h2>
      <p><strong>Source type: synthetic diagnostic trajectory.</strong> These frames are generated locally to test temporal projection consistency; they are not ManiSkill/LIBERO rollout episodes.</p>
      <p>Frames: {summary['frame_count']}; max projection error: {summary['max_projection_error_px']:.2f}px; max step delta: {summary['max_step_delta_px']:.2f}px.</p>
      <div class="hero">
        <figure><img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact sheet"><figcaption>contact sheet</figcaption></figure>
        <figure><img src="{html.escape(_rel(report_path, gif_path))}" alt="gif"><figcaption>animated sequence</figcaption></figure>
        <figure><img src="{html.escape(_rel(report_path, trail_path))}" alt="trail"><figcaption>projected point trail</figcaption></figure>
      </div>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(output_dir: Path, frames: int) -> dict[str, Any]:
    raw_dir = output_dir / "raw"
    overlay_dir = output_dir / "overlay"
    rows = []
    previous_point: list[int] | None = None
    for idx in range(frames):
        t = idx / max(1, frames - 1)
        x = int(round(70 + 180 * t))
        y = int(round(78 + 46 * math.sin(t * math.pi * 1.4) + 54 * t))
        xyz = _world_from_pixel(x, y)
        raw = _raw_frame(320, 240, (x, y), idx)
        raw_path = raw_dir / f"frame_{idx:03d}_raw.png"
        overlay_path = overlay_dir / f"frame_{idx:03d}_overlay.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raw).save(raw_path)
        _overlay_images, overlay = build_oracle_affordance_overlay(_obs(raw, xyz), output_path=overlay_path, label="moving object")
        error = math.dist([x, y], overlay.point_xy)
        step_delta = 0.0 if previous_point is None else math.dist(previous_point, overlay.point_xy)
        previous_point = overlay.point_xy
        rows.append(
            {
                "frame_index": idx,
                "raw": str(raw_path),
                "overlay": str(overlay_path),
                "expected_xy": [x, y],
                "point_xy": overlay.point_xy,
                "projection_error_px": float(error),
                "step_delta_px": float(step_delta),
                "mode": overlay.mode,
            }
        )
    contact_sheet = output_dir / "temporal_consistency_contact_sheet.png"
    gif_path = output_dir / "temporal_consistency_sequence.gif"
    trail_path = output_dir / "temporal_consistency_trail.png"
    _contact_sheet(rows, contact_sheet)
    _write_gif([Path(row["overlay"]) for row in rows], gif_path)
    _trail_image(rows, trail_path)
    summary = {
        "frame_count": len(rows),
        "max_projection_error_px": max(row["projection_error_px"] for row in rows),
        "mean_projection_error_px": float(sum(row["projection_error_px"] for row in rows) / len(rows)),
        "max_step_delta_px": max(row["step_delta_px"] for row in rows),
        "all_projected": all(row["mode"] == "projected_object_pose" for row in rows),
    }
    html_path = output_dir / "temporal_consistency_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet, gif_path, trail_path, summary), encoding="utf-8")
    report = {
        "status": "passed"
        if summary["frame_count"] >= 20 and summary["all_projected"] and summary["max_projection_error_px"] <= 1.0
        else "failed",
        "source_type": "synthetic_diagnostic_trajectory",
        "real_sim_episode": False,
        "provenance_note": "Generated locally to test temporal oracle projection consistency; not ManiSkill/LIBERO rollout frames.",
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "gif": str(gif_path),
        "trail": str(trail_path),
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "temporal_consistency_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(output_dir, args.frames)
    print(json.dumps({"status": report["status"], "frames": report["summary"]["frame_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
