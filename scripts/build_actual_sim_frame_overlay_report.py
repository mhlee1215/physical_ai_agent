#!/usr/bin/env python3
"""Build an actual-simulation-frame overlay fallback report.

This report uses real saved simulator RGB frames. It intentionally does not
claim true oracle projection unless matching pose/camera metadata is present.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from physical_ai_agent.perception.affordance_overlay import build_center_overlay_from_image_path


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _images(path: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in suffixes]


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    cell_w, cell_h = 680, 238
    canvas = Image.new("RGB", (cell_w, cell_h * len(rows)), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        y = idx * cell_h
        draw.rectangle((0, y, cell_w - 1, y + cell_h - 1), outline=(210, 190, 160), width=2)
        draw.text((14, y + 10), f"{row['case']} | {row['mode']} | actual sim RGB", fill=(20, 18, 14))
        for col, key in enumerate(("raw", "overlay")):
            image = Image.open(row[key]).convert("RGB")
            image.thumbnail((310, 170))
            x = 14 + col * 330
            canvas.paste(image, (x, y + 42))
            draw.text((x, y + 218), key, fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_gif(paths: list[Path], output_path: Path) -> None:
    frames = [Image.open(path).convert("RGB") for path in paths if path.exists()]
    if not frames:
        return
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path, gif_path: Path, source_root: Path) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>{html.escape(row['case'])}</h2>
        <p>Source frame: <code>{html.escape(row['source_frame'])}</code></p>
        <p>Mode: <strong>{html.escape(row['mode'])}</strong>. This is an overlay-rendering fallback on actual simulator RGB, not pose/camera true oracle projection.</p>
        <div class="pair">
          <figure><img src="{html.escape(_rel(report_path, Path(row['raw'])))}" alt="raw"><figcaption>actual sim RGB</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['overlay'])))}" alt="overlay"><figcaption>fallback overlay</figcaption></figure>
        </div>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual Sim Frame Overlay Fallback</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --warn:#a66000; --green:#008f5b; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#fff2b8,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1020px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--warn); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    a {{ color:#174f38; font-weight:800; }}
    code {{ overflow-wrap:anywhere; }}
    .hero,.pair {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim Frame Overlay Fallback</h1>
    <p><strong>Source type: actual_sim_rgb_fallback.</strong> These images come from saved simulator rollout frames under <code>{html.escape(str(source_root))}</code>. Because matching object pose and camera matrices are not available in the saved frame artifact, this report proves overlay rendering on actual simulation RGB but does not prove true oracle point projection.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">ACTUAL SIM RGB, NOT TRUE ORACLE</span>
      <h2>Summary</h2>
      <p>Samples: {len(rows)}. <a href="{html.escape(_rel(report_path, contact_sheet))}">Open contact sheet</a>. <a href="{html.escape(_rel(report_path, gif_path))}">Open GIF</a>.</p>
      <div class="hero">
        <figure><img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact sheet"><figcaption>actual sim frame contact sheet</figcaption></figure>
        <figure><img src="{html.escape(_rel(report_path, gif_path))}" alt="gif"><figcaption>overlay sequence GIF</figcaption></figure>
      </div>
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
    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        raw_path = raw_dir / f"{idx:03d}_{frame.name}"
        overlay_path = overlay_dir / f"{idx:03d}_{frame.stem}_fallback_overlay.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(frame).convert("RGB").save(raw_path)
        overlay = build_center_overlay_from_image_path(frame, overlay_path, label="sim RGB fallback")
        rows.append(
            {
                "case": f"sim_frame_{idx:03d}",
                "source_frame": str(frame),
                "raw": str(raw_path),
                "overlay": str(overlay_path),
                "mode": overlay.mode,
                "point_xy": overlay.point_xy,
            }
        )
    contact_sheet = output_dir / "actual_sim_frame_overlay_contact_sheet.png"
    gif_path = output_dir / "actual_sim_frame_overlay_sequence.gif"
    _contact_sheet(rows, contact_sheet)
    _write_gif([Path(row["overlay"]) for row in rows], gif_path)
    html_path = output_dir / "actual_sim_frame_overlay_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet, gif_path, frame_root), encoding="utf-8")
    report = {
        "status": "passed_fallback_only" if len(rows) >= 10 else "failed",
        "source_type": "actual_sim_rgb_fallback",
        "real_sim_episode": True,
        "true_oracle_projection": False,
        "provenance_note": "Uses saved simulator RGB frames. Matching pose/camera metadata is missing, so overlays are center fallback only.",
        "sample_count": len(rows),
        "source_frame_root": str(frame_root),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "gif": str(gif_path),
        "rows": rows,
    }
    (output_dir / "actual_sim_frame_overlay_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
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
    return 0 if report["status"] == "passed_fallback_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
