#!/usr/bin/env python3
"""Build temporal evidence for visual-heuristic points on actual sim frames."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _copy_sequence(rows: list[dict[str, Any]], output_dir: Path, limit: int) -> list[dict[str, Any]]:
    sequence = []
    previous_point: list[int] | None = None
    for idx, row in enumerate(rows[:limit]):
        point = [int(row.get("heuristic", {}).get("point_xy", [0, 0])[0]), int(row.get("heuristic", {}).get("point_xy", [0, 0])[1])]
        step_delta = 0.0 if previous_point is None else math.dist(previous_point, point)
        previous_point = point
        overlay_src = Path(str(row["overlay"]))
        overlay_dst = output_dir / "sequence_overlay" / f"frame_{idx:03d}_{overlay_src.name}"
        overlay_dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(overlay_src).convert("RGB").save(overlay_dst)
        sequence.append(
            {
                "frame_index": idx,
                "source_frame": row.get("source_frame", ""),
                "overlay": str(overlay_dst),
                "point_xy": point,
                "step_delta_px": float(step_delta),
                "mode": row.get("heuristic", {}).get("mode", "unknown"),
            }
        )
    return sequence


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    cols = 4
    cell_w, cell_h = 252, 190
    canvas = Image.new("RGB", (cols * cell_w, math.ceil(len(rows) / cols) * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        col = idx % cols
        grid_row = idx // cols
        x0, y0 = col * cell_w, grid_row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(210, 190, 160), width=1)
        image = Image.open(row["overlay"]).convert("RGB")
        image.thumbnail((224, 128))
        canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 32))
        draw.text((x0 + 8, y0 + 8), f"f{row['frame_index']:02d} point={row['point_xy']}", fill=(20, 18, 14))
        draw.text((x0 + 8, y0 + 166), f"delta={row['step_delta_px']:.1f}px", fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _trail(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    first = Image.open(rows[0]["overlay"]).convert("RGB")
    canvas = first.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    points = [tuple(row["point_xy"]) for row in rows]
    if len(points) > 1:
        draw.line(points, fill=(255, 190, 0, 210), width=4)
    for idx, point in enumerate(points):
        x, y = point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 220, 80, 230), outline=(60, 34, 0, 230))
        if idx in (0, len(points) - 1):
            draw.text((x + 7, y - 8), "start" if idx == 0 else "end", fill=(0, 0, 0, 230))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _gif(rows: list[dict[str, Any]], output_path: Path) -> None:
    frames = [Image.open(row["overlay"]).convert("RGB") for row in rows if Path(row["overlay"]).exists()]
    if not frames:
        return
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path, trail: Path, gif: Path, summary: dict[str, Any]) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>Frame {row['frame_index']:02d}</h2>
        <p>point={html.escape(str(row['point_xy']))}; step delta={row['step_delta_px']:.2f}px; mode={html.escape(row['mode'])}</p>
        <p>Source frame: <code>{html.escape(str(row['source_frame']))}</code></p>
        <figure><img src="{html.escape(_rel(report_path, Path(row['overlay'])))}" alt="frame"><figcaption>actual sim heuristic overlay</figcaption></figure>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual Sim Heuristic Temporal Consistency</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --amber:#a66000; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#ffe1a3,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1080px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--amber); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    code {{ overflow-wrap:anywhere; }}
    .hero {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim Heuristic Temporal Consistency</h1>
    <p><strong>Source type: actual_sim_rgb_visual_heuristic_temporal.</strong> This uses actual saved simulation RGB frames and image-only heuristic points. It is not true oracle projection.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">ACTUAL SIM TEMPORAL HEURISTIC</span>
      <h2>Summary</h2>
      <p>Frames: {summary['frame_count']}; mean step delta: {summary['mean_step_delta_px']:.2f}px; max step delta: {summary['max_step_delta_px']:.2f}px.</p>
      <div class="hero">
        <figure><img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact"><figcaption>contact sheet</figcaption></figure>
        <figure><img src="{html.escape(_rel(report_path, trail))}" alt="trail"><figcaption>point trail</figcaption></figure>
        <figure><img src="{html.escape(_rel(report_path, gif))}" alt="gif"><figcaption>sequence GIF</figcaption></figure>
      </div>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_report(heuristic_manifest_path: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    manifest = _load_json(heuristic_manifest_path)
    source_rows = [row for row in manifest.get("rows", []) if isinstance(row, dict)]
    rows = _copy_sequence(source_rows, output_dir, limit)
    contact_sheet = output_dir / "actual_sim_heuristic_temporal_contact_sheet.png"
    trail_path = output_dir / "actual_sim_heuristic_temporal_trail.png"
    gif_path = output_dir / "actual_sim_heuristic_temporal_sequence.gif"
    _contact_sheet(rows, contact_sheet)
    _trail(rows, trail_path)
    _gif(rows, gif_path)
    deltas = [float(row["step_delta_px"]) for row in rows[1:]]
    summary = {
        "frame_count": len(rows),
        "mean_step_delta_px": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "max_step_delta_px": max(deltas) if deltas else 0.0,
    }
    html_path = output_dir / "actual_sim_heuristic_temporal_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet, trail_path, gif_path, summary), encoding="utf-8")
    report = {
        "status": "passed" if len(rows) >= 10 else "failed",
        "source_type": "actual_sim_rgb_visual_heuristic_temporal",
        "real_sim_episode": True,
        "true_oracle_projection": False,
        "sample_count": len(rows),
        "summary": summary,
        "source_manifest": str(heuristic_manifest_path),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "trail": str(trail_path),
        "gif": str(gif_path),
        "rows": rows,
    }
    (output_dir / "actual_sim_heuristic_temporal_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(Path(args.heuristic_manifest), output_dir, args.limit)
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
