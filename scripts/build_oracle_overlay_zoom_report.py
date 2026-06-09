#!/usr/bin/env python3
"""Build zoomed raw-vs-oracle-overlay panels centered on the overlay marker."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _green_center(path: Path) -> tuple[int, int] | None:
    arr = np.asarray(Image.open(path).convert("RGB"))
    mask = (arr[:, :, 1] > 170) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 140)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def _crop_around(image: Image.Image, center: tuple[int, int] | None, size: int) -> Image.Image:
    width, height = image.size
    if center is None:
        cx, cy = width // 2, height // 2
    else:
        cx, cy = center
    half = size // 2
    left = max(0, min(width - size, cx - half))
    top = max(0, min(height - size, cy - half))
    right = min(width, left + size)
    bottom = min(height, top + size)
    crop = image.crop((left, top, right, bottom))
    if crop.size != (size, size):
        padded = Image.new("RGB", (size, size), (245, 242, 235))
        padded.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
        crop = padded
    return crop


def _panel(raw_path: Path, overlay_path: Path, output_path: Path, label: str, crop_size: int, scale: int) -> None:
    raw = Image.open(raw_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")
    center = _green_center(overlay_path)
    raw_crop = _crop_around(raw, center, crop_size).resize((crop_size * scale, crop_size * scale))
    overlay_crop = _crop_around(overlay, center, crop_size).resize((crop_size * scale, crop_size * scale))
    margin = 18
    header = 46
    width = raw_crop.width + overlay_crop.width + margin * 3
    height = raw_crop.height + header + margin
    canvas = Image.new("RGB", (width, height), (246, 244, 237))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), f"{label}: raw crop", fill=(18, 18, 18))
    draw.text((raw_crop.width + margin * 2, 12), "oracle-overlay crop", fill=(18, 18, 18))
    canvas.paste(raw_crop, (margin, header))
    canvas.paste(overlay_crop, (raw_crop.width + margin * 2, header))
    draw.rectangle((margin, header, margin + raw_crop.width - 1, header + raw_crop.height - 1), outline=(80, 70, 60), width=2)
    draw.rectangle(
        (
            raw_crop.width + margin * 2,
            header,
            raw_crop.width + margin * 2 + overlay_crop.width - 1,
            header + overlay_crop.height - 1,
        ),
        outline=(0, 190, 95),
        width=3,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((420, 230))
        canvas = Image.new("RGB", (440, 270), (246, 244, 237))
        canvas.paste(image, ((440 - image.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 244), path.stem[:52], fill=(18, 18, 18))
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 440, rows * 270), (230, 228, 220))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 440, (idx // cols) * 270))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_html(output_dir: Path, panels: list[Path], contact_sheet: Path) -> Path:
    html_path = output_dir / "zoom_report.html"
    cards = []
    for path in panels:
        cards.append(
            f"""
            <figure>
              <img src="{html.escape(path.relative_to(output_dir).as_posix())}" alt="{html.escape(path.name)}">
              <figcaption>{html.escape(path.stem)}</figcaption>
            </figure>
            """
        )
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zoomed Oracle Overlay Figures</title>
  <style>
    :root {{ --ink:#151511; --line:#d4c4aa; --green:#00c86d; }}
    body {{ margin:0; color:var(--ink); background:linear-gradient(135deg,#efe0c7,#f9f5ed 48%,#e4eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:44px 52px 20px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(34px,5vw,68px); letter-spacing:-.045em; }}
    main {{ padding:30px 52px 64px; display:grid; gap:24px; }}
    section {{ background:rgba(255,250,240,.86); border:1px solid var(--line); border-radius:26px; padding:24px; }}
    .sheet {{ width:min(100%, 980px); border-radius:16px; border:1px solid var(--line); background:white; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }}
    figure {{ margin:0; padding:12px; background:white; border:2px solid var(--green); border-radius:18px; }}
    img {{ width:100%; border-radius:12px; }}
    figcaption {{ margin-top:8px; font:13px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Zoomed Oracle Overlay Figures</h1>
    <p>Marker-centered crops make the raw-vs-overlay difference visible without squinting at a full contact sheet.</p>
  </header>
  <main>
    <section>
      <h2>Zoom contact sheet</h2>
      <img class="sheet" src="{html.escape(contact_sheet.relative_to(output_dir).as_posix())}" alt="zoom contact sheet">
    </section>
    <section>
      <h2>Panels</h2>
      <div class="grid">{''.join(cards)}</div>
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    comparison_root = Path(args.comparison_root)
    output_dir = Path(args.output_dir)
    raw_dir = comparison_root / "raw"
    overlay_dir = comparison_root / "overlay"
    raw_paths = sorted(raw_dir.glob("*_raw.png"))[: args.limit]
    panels = []
    for raw_path in raw_paths:
        overlay_name = raw_path.name.replace("_raw.png", "_overlay.png")
        overlay_path = overlay_dir / overlay_name
        if not overlay_path.exists():
            continue
        panel_path = output_dir / "panels" / raw_path.name.replace("_raw.png", "_zoom_pair.png")
        _panel(raw_path, overlay_path, panel_path, raw_path.stem.replace("_raw", ""), args.crop_size, args.scale)
        panels.append(panel_path)
    if len(panels) < 10:
        raise RuntimeError(f"expected at least 10 zoom panels, found {len(panels)}")
    contact_sheet = output_dir / "zoom_contact_sheet.png"
    _write_contact_sheet(panels, contact_sheet)
    html_path = _write_html(output_dir, panels, contact_sheet)
    manifest = {
        "status": "passed",
        "panel_count": len(panels),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "panels": [str(path) for path in panels],
    }
    (output_dir / "zoom_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"panel_count": len(panels), "html": str(html_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
