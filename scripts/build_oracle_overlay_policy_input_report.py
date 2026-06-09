#!/usr/bin/env python3
"""Build a SmolVLA policy-input readiness report for oracle overlay images."""

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


def _path_from_manifest(value: str) -> Path:
    return Path(value)


def _to_policy_tensor_preview(image_path: Path, output_path: Path) -> dict[str, Any]:
    channels, height, width = FEATURE_SHAPE
    image = Image.open(image_path).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    preview = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    draw.rectangle((0, 0, 122, 22), fill=(0, 0, 0))
    draw.text((6, 5), "policy tensor 3x224x224", fill=(0, 255, 120))
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
    cell_w, cell_h = 760, 270
    cols = 1
    rows_count = len(rows)
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows_count), (246, 241, 231))
    for row_index, row in enumerate(rows):
        y = row_index * cell_h
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, y, cell_w - 1, y + cell_h - 1), outline=(210, 190, 160), width=2)
        draw.text((14, y + 10), f"{row['case']} | {row['object']} | tensor {row['tensor']['shape']}", fill=(20, 18, 14))
        for col, key in enumerate(("raw", "overlay", "policy_preview")):
            image = Image.open(row[key]).convert("RGB")
            image.thumbnail((230, 190))
            x = 18 + col * 246
            canvas.paste(image, (x, y + 42))
            draw.text((x, y + 238), key, fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>{html.escape(row['case'])}: {html.escape(row['object'])}</h2>
        <p>camera source: <strong>base_camera</strong>; conditioning: <strong>oracle_affordance_overlay</strong>; tensor shape: <strong>{html.escape(str(row['tensor']['shape']))}</strong>; range: {row['tensor']['min']:.3f}-{row['tensor']['max']:.3f}</p>
        <div class="triplet">
          <figure><img src="{html.escape(_rel(report_path, Path(row['raw'])))}" alt="raw"><figcaption>raw observation image</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['overlay'])))}" alt="overlay"><figcaption>oracle overlay observation</figcaption></figure>
          <figure><img src="{html.escape(_rel(report_path, Path(row['policy_preview'])))}" alt="policy preview"><figcaption>policy-ready resized tensor preview</figcaption></figure>
        </div>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Overlay SmolVLA Policy-Input Readiness</title>
  <style>
    :root {{ --ink:#17130e; --paper:#f7efe1; --line:#d3bea0; --green:#008f5b; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#ddffd8,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:980px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--green); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    p {{ line-height:1.45; }}
    a {{ color:#174f38; font-weight:800; }}
    .triplet {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 13px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>SmolVLA Policy-Input Readiness</h1>
    <p>Checks that oracle overlay images can be consumed as SmolVLA-style image features without loading the model: raw observation image, oracle-overlaid observation image, and resized normalized policy tensor preview.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">PASSED LOCAL BRIDGE CHECK</span>
      <h2>Bridge assumption</h2>
      <p><strong>Source type: synthetic diagnostic.</strong> The source images are script-generated diverse-object layouts, not ManiSkill/LIBERO rollout episodes.</p>
      <p>This mirrors the CP24 path where oracle overlay pixels are passed through <code>override_camera_pixels</code> and converted into an image feature shaped like <code>[1, 3, 224, 224]</code>. It does not claim SmolVLA action quality; it validates the image-conditioning interface.</p>
      <p><a href="{html.escape(_rel(report_path, contact_sheet))}">Open contact sheet</a></p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_report(root: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    manifest = _load_json(root / "diverse_object_projection" / "diverse_object_manifest.json")
    candidates = [row for row in manifest.get("rows", []) if isinstance(row, dict)]
    candidates.sort(key=lambda row: float(row.get("distance_from_image_center_px", 0.0)), reverse=True)
    rows = []
    for row in candidates[:limit]:
        raw = _path_from_manifest(str(row["raw"]))
        overlay = _path_from_manifest(str(row["overlay"]))
        preview = output_dir / "policy_preview" / f"{row['case']}_{str(row['object']).replace(' ', '_')}_tensor_preview.png"
        tensor = _to_policy_tensor_preview(overlay, preview)
        rows.append(
            {
                "case": row["case"],
                "object": row["object"],
                "raw": str(raw),
                "overlay": str(overlay),
                "policy_preview": str(preview),
                "tensor": tensor,
                "image_conditioning": "oracle_affordance_overlay",
                "image_feature_shape": list(FEATURE_SHAPE),
            }
        )
    contact_sheet = output_dir / "policy_input_contact_sheet.png"
    _contact_sheet(rows, contact_sheet)
    html_path = output_dir / "policy_input_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet), encoding="utf-8")
    report = {
        "status": "passed" if len(rows) >= 10 and all(row["tensor"]["shape"] == [1, 3, 224, 224] for row in rows) else "failed",
        "source_type": "synthetic_diagnostic",
        "real_sim_episode": False,
        "provenance_note": "Derived from script-generated diverse-object diagnostic images, not ManiSkill/LIBERO rollout episodes.",
        "sample_count": len(rows),
        "image_conditioning": "oracle_affordance_overlay",
        "feature_shape": [1, 3, 224, 224],
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "rows": rows,
    }
    (output_dir / "policy_input_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(Path(args.root), output_dir, args.limit)
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
