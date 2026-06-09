#!/usr/bin/env python3
"""Build a report for actual-sim true oracle projection steps."""

from __future__ import annotations

import argparse
import html
import json
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


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        image = Image.new("RGB", (920, 180), (246, 241, 231))
        draw = ImageDraw.Draw(image)
        draw.text((20, 72), "No true-oracle actual-sim samples yet. Run CP24 with smolvla_affordance_oracle --real-images.", fill=(20, 18, 14))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return
    cell_w, cell_h = 760, 260
    canvas = Image.new("RGB", (cell_w, cell_h * len(rows)), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        y = idx * cell_h
        draw.rectangle((0, y, cell_w - 1, y + cell_h - 1), outline=(210, 190, 160), width=2)
        draw.text((14, y + 10), f"episode={row.get('episode')} step={row.get('step')} point={row.get('oracle_affordance', {}).get('point_xy')}", fill=(20, 18, 14))
        for col, key in enumerate(("raw_frame_path", "overlay_frame_path")):
            path = Path(str(row.get(key, "")))
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((330, 185))
            x = 14 + col * 370
            canvas.paste(image, (x, y + 44))
            draw.text((x, y + 236), "raw sim RGB" if col == 0 else "true oracle overlay", fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _html(report_path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]], contact_sheet: Path) -> str:
    status = str(manifest.get("status", "missing_manifest"))
    cards = []
    for row in rows:
        oracle = row.get("oracle_affordance", {})
        cards.append(
            f"""
      <article>
        <h2>Episode {html.escape(str(row.get('episode')))}, step {html.escape(str(row.get('step')))}</h2>
        <p>mode={html.escape(str(oracle.get('mode')))}; point={html.escape(str(oracle.get('point_xy')))}; object_pose_xyz={html.escape(str(oracle.get('object_pose_xyz')))}; camera_keys={html.escape(str(oracle.get('camera_metadata_keys')))}</p>
        <div class="pair">
          <figure><img src="{html.escape(_rel(report_path, Path(str(row.get('raw_frame_path')))))}" alt="raw"><figcaption>actual sim RGB at action input</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(str(row.get('overlay_frame_path')))))}" alt="overlay"><figcaption>true oracle overlay from same observation</figcaption></figure>
        </div>
      </article>
            """
        )
    if not cards:
        cards.append(
            """
      <article>
        <h2>No passed true-oracle samples yet</h2>
        <p>The repository code now writes the required manifest, but an actual renderer-capable CP24 run is still needed.</p>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual Sim True Oracle Projection</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --green:#008f5b; --block:#b5382d; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#cdf7ff,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1020px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:{'var(--green)' if status == 'passed' else 'var(--block)'}; border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    .pair {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
    code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim True Oracle Projection</h1>
    <p><strong>Source type: actual_sim_true_oracle_projection.</strong> This is the required evidence tier for paper claims. It requires actual sim RGB, object pose, camera metadata, and oracle overlay from the same action-input observation.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">{html.escape(status.upper())}</span>
      <h2>Summary</h2>
      <p>Strict true-oracle steps: {html.escape(str(manifest.get('strict_true_oracle_step_count', 0)))} / required {html.escape(str(manifest.get('min_steps', 10)))}.</p>
      <p><img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact sheet"></p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_report(manifest_path: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    strict_rows = [
        row for row in manifest.get("steps", [])
        if isinstance(row, dict) and row.get("strict_true_oracle_ready")
    ][:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = output_dir / "actual_sim_true_oracle_contact_sheet.png"
    _contact_sheet(strict_rows, contact_sheet)
    html_path = output_dir / "actual_sim_true_oracle_report.html"
    if not manifest:
        manifest = {
            "status": "missing_manifest",
            "source_type": "actual_sim_true_oracle_projection",
            "real_sim_episode": True,
            "true_oracle_projection": False,
            "min_steps": 10,
            "strict_true_oracle_step_count": 0,
            "provenance_note": "No true-oracle step manifest exists yet.",
        }
    html_path.write_text(_html(html_path, manifest, strict_rows, contact_sheet), encoding="utf-8")
    report = {
        "status": "passed" if manifest.get("status") == "passed" and len(strict_rows) >= min(10, limit) else "blocked",
        "source_type": "actual_sim_true_oracle_projection",
        "real_sim_episode": True,
        "true_oracle_projection": manifest.get("status") == "passed",
        "sample_count": len(strict_rows),
        "source_manifest": str(manifest_path),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
    }
    if manifest_path.exists():
        imported_manifest_path = output_dir / "smolvla_affordance_true_oracle_steps.json"
        imported_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report["imported_step_manifest"] = str(imported_manifest_path)
    (output_dir / "actual_sim_true_oracle_report_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    report = build_report(Path(args.manifest), Path(args.output_dir), args.limit)
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
