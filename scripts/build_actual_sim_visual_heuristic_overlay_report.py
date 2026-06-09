#!/usr/bin/env python3
"""Build actual-sim RGB visual-heuristic affordance overlay evidence.

This uses saved simulator RGB frames only. It is not true oracle projection
because it does not use simulator object pose or camera matrices.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_SUFFIXES]


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _heuristic_point(rgb: np.ndarray) -> tuple[list[int], dict[str, Any]]:
    height, width = rgb.shape[:2]
    arr = rgb.astype(np.float32)
    # ManiSkill manipulation objects are often saturated and brighter than the
    # table/background. This intentionally avoids semantic claims; it is only a
    # visual prompt heuristic over real sim RGB.
    channel_span = arr.max(axis=2) - arr.min(axis=2)
    brightness = arr.mean(axis=2)
    y_grid = np.arange(height, dtype=np.float32)[:, None]
    table_prior = y_grid > height * 0.22
    mask = (channel_span > 28.0) & (brightness > 45.0) & table_prior
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return [width // 2, height // 2], {
            "mode": "image_center_fallback",
            "mask_pixels": int(len(xs)),
            "confidence": 0.25,
        }
    # Use a robust top cluster: largest connected component would be nicer, but
    # centroid of strongest saturated pixels is deterministic and dependency-free.
    scores = channel_span[ys, xs] + 0.2 * brightness[ys, xs]
    keep_count = max(12, min(len(xs), int(len(xs) * 0.35)))
    keep = np.argpartition(scores, -keep_count)[-keep_count:]
    x = int(round(float(xs[keep].mean())))
    y = int(round(float(ys[keep].mean())))
    return [max(0, min(width - 1, x)), max(0, min(height - 1, y))], {
        "mode": "visual_heuristic_saturation_centroid",
        "mask_pixels": int(len(xs)),
        "confidence": round(min(1.0, len(xs) / max(1.0, width * height * 0.08)), 4),
    }


def _draw_overlay(input_path: Path, output_path: Path) -> dict[str, Any]:
    image = Image.open(input_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    point_xy, metadata = _heuristic_point(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    x, y = point_xy
    radius = max(6, min(image.size) // 32)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 190, 0, 255), width=4)
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 190, 0, 230), width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 190, 0, 230), width=2)
    draw.rectangle((4, 4, 176, 24), fill=(0, 0, 0, 150))
    draw.text((8, 8), metadata["mode"], fill=(255, 220, 90, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "point_xy": point_xy,
        "image_shape": [int(rgb.shape[0]), int(rgb.shape[1]), 3],
        **metadata,
    }


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    cell_w, cell_h = 720, 238
    canvas = Image.new("RGB", (cell_w, cell_h * len(rows)), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        y = idx * cell_h
        draw.rectangle((0, y, cell_w - 1, y + cell_h - 1), outline=(210, 190, 160), width=2)
        draw.text(
            (14, y + 10),
            f"{row['case']} | {row['heuristic']['mode']} | point={row['heuristic']['point_xy']}",
            fill=(20, 18, 14),
        )
        for col, key in enumerate(("raw", "overlay")):
            image = Image.open(row[key]).convert("RGB")
            image.thumbnail((330, 174))
            x = 14 + col * 360
            canvas.paste(image, (x, y + 42))
            draw.text((x, y + 218), key, fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path, source_root: Path) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>{html.escape(row['case'])}</h2>
        <p>mode={html.escape(row['heuristic']['mode'])}; point={html.escape(str(row['heuristic']['point_xy']))}; mask_pixels={row['heuristic']['mask_pixels']}; confidence={row['heuristic']['confidence']}</p>
        <p>Source frame: <code>{html.escape(row['source_frame'])}</code></p>
        <div class="pair">
          <figure><img src="{html.escape(_rel(report_path, Path(row['raw'])))}" alt="raw"><figcaption>actual sim RGB</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['overlay'])))}" alt="overlay"><figcaption>visual heuristic overlay</figcaption></figure>
        </div>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual Sim Visual Heuristic Overlay</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --amber:#a66000; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#ffe2a6,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1060px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--amber); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    code {{ overflow-wrap:anywhere; }}
    .pair,.hero {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim Visual Heuristic Overlay</h1>
    <p><strong>Source type: actual_sim_rgb_visual_heuristic.</strong> These are actual saved simulation RGB frames from <code>{html.escape(str(source_root))}</code>. The point is selected by a simple visual saturation/brightness heuristic, not simulator oracle pose. This is stronger than center fallback but still not true oracle projection.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">ACTUAL SIM RGB HEURISTIC</span>
      <h2>Summary</h2>
      <p>Samples: {len(rows)}. This milestone tests whether a lightweight non-oracle visual layer can produce object-like point prompts on actual sim frames.</p>
      <div class="hero"><figure><img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact sheet"><figcaption>visual heuristic contact sheet</figcaption></figure></div>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_report(frame_root: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    frames = _images(frame_root)[:limit]
    raw_dir = output_dir / "raw"
    overlay_dir = output_dir / "overlay"
    rows = []
    for idx, frame in enumerate(frames):
        raw_path = raw_dir / f"{idx:03d}_{frame.name}"
        overlay_path = overlay_dir / f"{idx:03d}_{frame.stem}_visual_heuristic.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(frame).convert("RGB").save(raw_path)
        heuristic = _draw_overlay(frame, overlay_path)
        rows.append(
            {
                "case": f"sim_frame_{idx:03d}",
                "source_frame": str(frame),
                "raw": str(raw_path),
                "overlay": str(overlay_path),
                "heuristic": heuristic,
            }
        )
    contact_sheet = output_dir / "actual_sim_visual_heuristic_contact_sheet.png"
    _contact_sheet(rows, contact_sheet)
    html_path = output_dir / "actual_sim_visual_heuristic_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet, frame_root), encoding="utf-8")
    non_center = sum(
        1
        for row in rows
        if row["heuristic"]["mode"] != "image_center_fallback"
    )
    report = {
        "status": "passed" if len(rows) >= 10 and non_center >= 5 else "failed",
        "source_type": "actual_sim_rgb_visual_heuristic",
        "real_sim_episode": True,
        "true_oracle_projection": False,
        "provenance_note": "Uses actual saved simulator RGB frames and image-only visual heuristics; no simulator pose/camera oracle.",
        "sample_count": len(rows),
        "non_center_heuristic_count": non_center,
        "source_frame_root": str(frame_root),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "rows": rows,
    }
    (output_dir / "actual_sim_visual_heuristic_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(Path(args.frame_root), output_dir, args.limit)
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
