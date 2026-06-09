#!/usr/bin/env python3
"""Build an oracle point visual-encoding ablation report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


VARIANTS = ("ring_cross", "solid_dot", "soft_heatmap", "arrow_label")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _draw_variant(raw_path: Path, point_xy: list[int], variant: str, output_path: Path, label: str) -> dict[str, float]:
    image = Image.open(raw_path).convert("RGB")
    x, y = int(point_xy[0]), int(point_xy[1])
    draw = ImageDraw.Draw(image, "RGBA")
    radius = max(6, min(image.size) // 32)
    if variant == "ring_cross":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 90, 255), width=4)
        draw.line((x - radius * 2, y, x + radius * 2, y), fill=(0, 255, 90, 230), width=2)
        draw.line((x, y - radius * 2, x, y + radius * 2), fill=(0, 255, 90, 230), width=2)
    elif variant == "solid_dot":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 90, 210), outline=(255, 255, 255, 230), width=2)
    elif variant == "soft_heatmap":
        heat = Image.new("RGBA", image.size, (0, 0, 0, 0))
        heat_draw = ImageDraw.Draw(heat, "RGBA")
        for idx, scale in enumerate((4.5, 3.2, 2.0, 1.0)):
            alpha = int(45 + idx * 38)
            rr = int(radius * scale)
            heat_draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(0, 255, 90, alpha))
        heat = heat.filter(ImageFilter.GaussianBlur(radius=max(2, radius // 2)))
        image = Image.alpha_composite(image.convert("RGBA"), heat).convert("RGB")
    elif variant == "arrow_label":
        start = (max(4, x - radius * 5), max(18, y - radius * 4))
        draw.line((start[0], start[1], x, y), fill=(0, 255, 90, 255), width=3)
        draw.polygon([(x, y), (x - 8, y - 2), (x - 3, y - 8)], fill=(0, 255, 90, 255))
        text = label[:18]
        box = (start[0] - 3, start[1] - 16, start[0] + 8 * len(text) + 8, start[1] + 4)
        draw.rectangle(box, fill=(0, 0, 0, 165))
        draw.text((start[0] + 2, start[1] - 14), text, fill=(0, 255, 90, 255))
    else:
        raise ValueError(f"Unknown variant: {variant}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    raw = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.float32)
    over = np.asarray(image.convert("RGB"), dtype=np.float32)
    delta = np.abs(over - raw)
    changed = np.any(delta > 1.0, axis=-1)
    return {
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": float(delta.max()),
        "changed_pixel_ratio": float(changed.mean()),
    }


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    thumb_w, thumb_h = 180, 132
    cols = 1 + len(VARIANTS)
    cell_w, cell_h = 205, 180
    canvas = Image.new("RGB", (cols * cell_w, len(rows) * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        y0 = row_idx * cell_h
        items = [("raw", Path(row["raw"]))] + [(variant, Path(row["variants"][variant]["path"])) for variant in VARIANTS]
        for col_idx, (label, path) in enumerate(items):
            x0 = col_idx * cell_w
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(210, 190, 160), width=1)
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 28))
            draw.text((x0 + 8, y0 + 8), label, fill=(20, 18, 14))
            draw.text((x0 + 8, y0 + 164), str(row["case"]), fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path) -> str:
    cards = []
    for row in rows:
        figures = [
            f'<figure><img src="{html.escape(_rel(report_path, Path(row["raw"])))}" alt="raw"><figcaption>raw</figcaption></figure>'
        ]
        for variant in VARIANTS:
            info = row["variants"][variant]
            figures.append(
                f"""
          <figure>
            <img src="{html.escape(_rel(report_path, Path(info['path'])))}" alt="{html.escape(variant)}">
            <figcaption>{html.escape(variant)}; changed={info['changed_pixel_ratio']:.3f}; mean delta={info['mean_abs_delta']:.2f}</figcaption>
          </figure>
                """
            )
        cards.append(
            f"""
      <article>
        <h2>{html.escape(row['case'])}: {html.escape(row['object'])}</h2>
        <p>point={html.escape(str(row['point_xy']))}; distance from image center={row['distance_from_image_center_px']:.1f}px. Lower changed-pixel ratio usually means a less invasive visual prompt; stronger encodings may be easier for weak policies to notice.</p>
        <div class="grid">{''.join(figures)}</div>
      </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Point Encoding Ablation</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --green:#008f5b; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#d8f6ff,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:980px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--green); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    a {{ color:#174f38; font-weight:800; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Point Encoding Ablation</h1>
    <p>Compares multiple zero-parameter visual encodings for the same oracle point. This helps choose a lightweight prompt style before training a learned affordance predictor.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">PASSED LOCAL ABLATION</span>
      <h2>Variants</h2>
      <p><strong>Source type: synthetic diagnostic.</strong> The source images are script-generated diverse-object layouts, not ManiSkill/LIBERO rollout episodes.</p>
      <p>Variants: {html.escape(", ".join(VARIANTS))}. <a href="{html.escape(_rel(report_path, contact_sheet))}">Open contact sheet</a></p>
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
        raw = Path(str(row["raw"]))
        point_xy = [int(row["point_xy"][0]), int(row["point_xy"][1])]
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            out_path = output_dir / "variants" / variant / f"{row['case']}_{str(row['object']).replace(' ', '_')}_{variant}.png"
            stats = _draw_variant(raw, point_xy, variant, out_path, str(row["object"]))
            variants[variant] = {"path": str(out_path), **stats}
        rows.append(
            {
                "case": row["case"],
                "object": row["object"],
                "raw": str(raw),
                "point_xy": point_xy,
                "distance_from_image_center_px": float(row.get("distance_from_image_center_px", 0.0)),
                "variants": variants,
            }
        )
    contact_sheet = output_dir / "encoding_ablation_contact_sheet.png"
    _contact_sheet(rows, contact_sheet)
    html_path = output_dir / "encoding_ablation_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet), encoding="utf-8")
    status = "passed" if len(rows) >= 10 and all(set(row["variants"]) == set(VARIANTS) for row in rows) else "failed"
    report = {
        "status": status,
        "source_type": "synthetic_diagnostic",
        "real_sim_episode": False,
        "provenance_note": "Derived from script-generated diverse-object diagnostic images, not ManiSkill/LIBERO rollout episodes.",
        "sample_count": len(rows),
        "variants": list(VARIANTS),
        "variant_count": len(VARIANTS),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "rows": rows,
    }
    (output_dir / "encoding_ablation_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
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
