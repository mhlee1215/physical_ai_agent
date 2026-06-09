#!/usr/bin/env python3
"""Build policy-input readiness evidence from actual-sim heuristic overlays."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


FEATURE_SHAPE = (3, 224, 224)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _policy_preview(image_path: Path, output_path: Path) -> dict[str, Any]:
    channels, height, width = FEATURE_SHAPE
    image = Image.open(image_path).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    preview = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    draw.rectangle((0, 0, 166, 23), fill=(0, 0, 0))
    draw.text((6, 6), "actual sim policy tensor", fill=(255, 210, 60))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)
    return {
        "shape": list(tensor.shape),
        "dtype": "float32",
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    cell_w, cell_h = 780, 268
    canvas = Image.new("RGB", (cell_w, cell_h * len(rows)), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        y = idx * cell_h
        draw.rectangle((0, y, cell_w - 1, y + cell_h - 1), outline=(210, 190, 160), width=2)
        draw.text((14, y + 10), f"{row['case']} | tensor={row['tensor']['shape']} | {row['heuristic_mode']}", fill=(20, 18, 14))
        for col, key in enumerate(("raw", "overlay", "policy_preview")):
            image = Image.open(row[key]).convert("RGB")
            image.thumbnail((235, 178))
            x = 14 + col * 252
            canvas.paste(image, (x, y + 44))
            draw.text((x, y + 242), key, fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>{html.escape(row['case'])}</h2>
        <p>mode={html.escape(row['heuristic_mode'])}; point={html.escape(str(row['point_xy']))}; tensor={html.escape(str(row['tensor']['shape']))}; range={row['tensor']['min']:.3f}-{row['tensor']['max']:.3f}</p>
        <p>Source frame: <code>{html.escape(row['source_frame'])}</code></p>
        <div class="triplet">
          <figure><img src="{html.escape(_rel(report_path, Path(row['raw'])))}" alt="raw"><figcaption>actual sim RGB</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['overlay'])))}" alt="overlay"><figcaption>visual heuristic overlay</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['policy_preview'])))}" alt="policy preview"><figcaption>policy-ready tensor preview</figcaption></figure>
        </div>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual Sim Heuristic Policy-Input Readiness</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --amber:#a66000; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#ffe7b8,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1080px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--amber); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    code {{ overflow-wrap:anywhere; }}
    .triplet {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim Heuristic Policy-Input Readiness</h1>
    <p><strong>Source type: actual_sim_rgb_visual_heuristic_policy_input.</strong> These samples start from actual saved simulation RGB frames, apply an image-only heuristic affordance overlay, then convert the result into a SmolVLA-style image tensor preview. This is not true oracle projection.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">ACTUAL SIM RGB POLICY INPUT</span>
      <h2>Summary</h2>
      <p>Samples: {len(rows)}. Feature shape: [1, 3, 224, 224]. <img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact sheet"></p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_report(heuristic_manifest_path: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    heuristic_manifest = _load_json(heuristic_manifest_path)
    candidates = [row for row in heuristic_manifest.get("rows", []) if isinstance(row, dict)]
    rows = []
    for row in candidates[:limit]:
        overlay = Path(str(row["overlay"]))
        raw = Path(str(row["raw"]))
        preview = output_dir / "policy_preview" / f"{row['case']}_actual_sim_tensor_preview.png"
        tensor = _policy_preview(overlay, preview)
        rows.append(
            {
                "case": row["case"],
                "source_frame": row.get("source_frame", ""),
                "raw": str(raw),
                "overlay": str(overlay),
                "policy_preview": str(preview),
                "heuristic_mode": row.get("heuristic", {}).get("mode", "unknown"),
                "point_xy": row.get("heuristic", {}).get("point_xy", []),
                "tensor": tensor,
            }
        )
    contact_sheet = output_dir / "actual_sim_heuristic_policy_input_contact_sheet.png"
    _contact_sheet(rows, contact_sheet)
    html_path = output_dir / "actual_sim_heuristic_policy_input_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet), encoding="utf-8")
    report = {
        "status": "passed" if len(rows) >= 10 and all(row["tensor"]["shape"] == [1, 3, 224, 224] for row in rows) else "failed",
        "source_type": "actual_sim_rgb_visual_heuristic_policy_input",
        "real_sim_episode": True,
        "true_oracle_projection": False,
        "sample_count": len(rows),
        "feature_shape": [1, 3, 224, 224],
        "source_manifest": str(heuristic_manifest_path),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "rows": rows,
    }
    (output_dir / "actual_sim_heuristic_policy_input_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(Path(args.heuristic_manifest), output_dir, args.limit)
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
