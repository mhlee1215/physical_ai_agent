#!/usr/bin/env python3
"""Build raw-vs-oracle-overlay visual comparison artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physical_ai_agent.perception.affordance_overlay import (
    build_center_overlay_from_image_path,
    build_oracle_affordance_overlay,
)


def _save_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def _make_scene(width: int, height: int, xyz: tuple[float, float, float]) -> np.ndarray:
    x_grad = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    blend = np.clip((x_grad + y_grad) / 2.0, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = (32 * (1.0 - blend) + 86 * blend).astype(np.uint8)
    rgb[:, :, 1] = (42 * (1.0 - blend) + 78 * blend).astype(np.uint8)
    rgb[:, :, 2] = (54 * (1.0 - blend) + 58 * blend).astype(np.uint8)
    px = int(round(120.0 * xyz[0] / xyz[2] + width / 2.0))
    py = int(round(120.0 * xyz[1] / xyz[2] + height / 2.0))
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    radius = max(10, min(width, height) // 14)
    draw.ellipse(
        (px - radius, py - radius, px + radius, py + radius),
        fill=(220, 90, 45),
        outline=(80, 30, 20),
        width=3,
    )
    draw.ellipse(
        (px - radius // 3, py - radius // 3, px + radius // 3, py + radius // 3),
        fill=(255, 220, 60),
    )
    return np.asarray(image, dtype=np.uint8)


def _obs(rgb: np.ndarray, xyz: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "sensor_data": {
            "base_camera": {
                "rgb": rgb,
            }
        },
        "sensor_param": {
            "base_camera": {
                "intrinsic_cv": np.asarray(
                    [
                        [120.0, 0.0, rgb.shape[1] / 2.0],
                        [0.0, 120.0, rgb.shape[0] / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                "extrinsic_cv": np.eye(4, dtype=np.float32),
            }
        },
        "obj_pose": {"p": np.asarray(xyz, dtype=np.float32)},
    }


def _side_by_side(raw_path: Path, overlay_path: Path, output_path: Path, title: str) -> None:
    raw = Image.open(raw_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")
    height = max(raw.height, overlay.height)
    width = raw.width + overlay.width + 18
    canvas = Image.new("RGB", (width, height + 48), (246, 244, 237))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{title} | baseline raw", fill=(18, 18, 18))
    draw.text((raw.width + 26, 8), "agentic oracle overlay", fill=(18, 18, 18))
    canvas.paste(raw, (0, 42))
    canvas.paste(overlay, (raw.width + 18, 42))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((300, 170))
        canvas = Image.new("RGB", (320, 210), (246, 244, 237))
        canvas.paste(image, ((320 - image.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 186), path.stem[:42], fill=(18, 18, 18))
        thumbs.append(canvas)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 210), (230, 228, 220))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 320, (idx // cols) * 210))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_html(output_dir: Path, rows: list[dict[str, Any]], contact_sheet: Path) -> Path:
    html_path = output_dir / "comparison_report.html"
    cards = []
    for row in rows:
        pair_path = Path(row["pair"])
        cards.append(
            f"""
            <figure class="card">
              <img src="{html.escape(pair_path.relative_to(output_dir).as_posix())}" alt="{html.escape(row['case'])}">
              <figcaption>
                <strong>{html.escape(row['case'])}</strong>
                <span>{html.escape(row['mode'])}</span>
              </figcaption>
            </figure>
            """
        )
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raw vs Oracle Overlay Comparison</title>
  <style>
    :root {{ --ink:#171711; --paper:#f6f0e4; --panel:#fffaf0; --line:#d4c4aa; --green:#00d968; }}
    body {{
      margin:0;
      background:radial-gradient(circle at 12% 6%, #fff1bd, transparent 28%),
        linear-gradient(135deg,#efe0c7,#f9f5ed 46%,#e4eee7);
      color:var(--ink);
      font-family:Charter, Georgia, serif;
    }}
    header {{ padding:44px 52px 18px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(34px,5vw,68px); letter-spacing:-.045em; }}
    main {{ padding:30px 52px 64px; }}
    section {{ margin-bottom:28px; padding:24px; background:rgba(255,250,240,.86); border:1px solid var(--line); border-radius:26px; }}
    .sheet {{ width:min(100%, 1120px); border-radius:16px; border:1px solid var(--line); background:white; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
    .card {{ margin:0; padding:12px; background:white; border:2px solid var(--green); border-radius:18px; }}
    .card img {{ width:100%; border-radius:12px; background:#2b2b2b; }}
    figcaption {{ display:grid; gap:4px; margin-top:8px; font-family:Avenir Next, Helvetica, sans-serif; font-size:13px; }}
  </style>
</head>
<body>
  <header>
    <h1>Raw vs Oracle Overlay Comparison</h1>
    <p>Baseline SmolVLA image input compared with the agentic oracle-overlay conditioned input.</p>
  </header>
  <main>
    <section>
      <h2>Contact sheet</h2>
      <img class="sheet" src="{html.escape(contact_sheet.relative_to(output_dir).as_posix())}" alt="comparison contact sheet">
    </section>
    <section>
      <h2>Pairs</h2>
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sim-frame-root")
    parser.add_argument("--synthetic-count", type=int, default=20)
    parser.add_argument("--sim-count", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    overlay_dir = output_dir / "overlay"
    pair_dir = output_dir / "pairs"
    rows: list[dict[str, Any]] = []

    for idx in range(args.synthetic_count):
        t = idx / max(1, args.synthetic_count - 1)
        xyz = (-0.36 + 0.72 * t, 0.24 * np.sin(t * np.pi * 2.0), 1.12)
        rgb = _make_scene(320, 200, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
        raw_path = raw_dir / f"synthetic_{idx:03d}_raw.png"
        overlay_path = overlay_dir / f"synthetic_{idx:03d}_overlay.png"
        pair_path = pair_dir / f"synthetic_{idx:03d}_pair.png"
        _save_image(raw_path, rgb)
        _overlays, metadata = build_oracle_affordance_overlay(
            _obs(rgb, (float(xyz[0]), float(xyz[1]), float(xyz[2]))),
            output_path=overlay_path,
            label=f"cmp {idx:02d}",
        )
        _side_by_side(raw_path, overlay_path, pair_path, f"synthetic {idx:03d}")
        rows.append({"case": f"synthetic_{idx:03d}", "mode": metadata.mode, "pair": str(pair_path)})

    if args.sim_frame_root:
        sim_paths = sorted(Path(args.sim_frame_root).rglob("*.png"))[: args.sim_count]
        for idx, raw_source in enumerate(sim_paths):
            raw_path = raw_dir / f"sim_{idx:03d}_raw.png"
            overlay_path = overlay_dir / f"sim_{idx:03d}_overlay.png"
            pair_path = pair_dir / f"sim_{idx:03d}_pair.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(raw_source).convert("RGB").save(raw_path)
            metadata = build_center_overlay_from_image_path(raw_path, overlay_path, label=f"sim cmp {idx:02d}")
            _side_by_side(raw_path, overlay_path, pair_path, f"sim {idx:03d}")
            rows.append({"case": f"sim_{idx:03d}", "mode": metadata.mode, "pair": str(pair_path)})

    contact_sheet = output_dir / "comparison_contact_sheet.png"
    _write_contact_sheet([Path(row["pair"]) for row in rows], contact_sheet)
    html_path = _write_html(output_dir, rows, contact_sheet)
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "passed" if len(rows) >= 10 else "insufficient_samples",
                "sample_count": len(rows),
                "html": str(html_path),
                "contact_sheet": str(contact_sheet),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"sample_count": len(rows), "html": str(html_path)}, indent=2))
    return 0 if len(rows) >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
