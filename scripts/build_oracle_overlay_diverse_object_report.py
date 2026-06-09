#!/usr/bin/env python3
"""Build diverse-object oracle projection validation figures.

This report is intentionally full-frame first. It verifies that oracle points
are not merely center overlays by scattering objects away from the image center
and checking projected point error against each object's known center.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from physical_ai_agent.perception.affordance_overlay import build_oracle_affordance_overlay


OBJECTS = [
    ("red cube", "square", (225, 74, 50), (95, 25, 20)),
    ("yellow bowl", "circle", (235, 185, 58), (120, 90, 18)),
    ("blue mug", "mug", (60, 125, 210), (20, 62, 115)),
    ("green bottle", "bottle", (72, 170, 95), (24, 90, 48)),
    ("brown book", "book", (145, 92, 55), (78, 45, 25)),
    ("cyan can", "can", (70, 190, 195), (20, 95, 100)),
]


def _project_xy(width: int, height: int, xyz: tuple[float, float, float]) -> tuple[int, int]:
    return int(round(120.0 * xyz[0] / xyz[2] + width / 2.0)), int(round(120.0 * xyz[1] / xyz[2] + height / 2.0))


def _draw_object(draw: ImageDraw.ImageDraw, xy: tuple[int, int], shape: str, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    x, y = xy
    if shape == "square":
        draw.rounded_rectangle((x - 18, y - 18, x + 18, y + 18), radius=5, fill=fill, outline=outline, width=3)
    elif shape == "circle":
        draw.ellipse((x - 21, y - 15, x + 21, y + 15), fill=fill, outline=outline, width=3)
        draw.arc((x - 24, y - 22, x + 24, y + 20), 10, 170, fill=(255, 235, 140), width=3)
    elif shape == "mug":
        draw.rounded_rectangle((x - 18, y - 16, x + 15, y + 18), radius=6, fill=fill, outline=outline, width=3)
        draw.arc((x + 8, y - 8, x + 31, y + 13), -80, 90, fill=outline, width=4)
    elif shape == "bottle":
        draw.rounded_rectangle((x - 10, y - 24, x + 10, y + 22), radius=6, fill=fill, outline=outline, width=3)
        draw.rectangle((x - 6, y - 34, x + 6, y - 20), fill=fill, outline=outline, width=2)
    elif shape == "book":
        draw.polygon([(x - 26, y - 16), (x + 18, y - 24), (x + 27, y + 14), (x - 18, y + 23)], fill=fill, outline=outline)
        draw.line((x - 14, y - 13, x + 20, y - 19), fill=(235, 205, 150), width=2)
    else:
        draw.rounded_rectangle((x - 15, y - 22, x + 15, y + 22), radius=9, fill=fill, outline=outline, width=3)
        draw.ellipse((x - 15, y - 25, x + 15, y - 14), fill=(120, 230, 230), outline=outline, width=2)


def _scene(width: int, height: int, target_xy: tuple[int, int], object_spec: tuple[str, str, tuple[int, int, int], tuple[int, int, int]], episode: int) -> np.ndarray:
    name, shape, fill, outline = object_spec
    x_grad = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    blend = np.clip((x_grad + y_grad) / 2.0, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = (36 * (1.0 - blend) + 96 * blend).astype(np.uint8)
    rgb[:, :, 1] = (44 * (1.0 - blend) + 84 * blend).astype(np.uint8)
    rgb[:, :, 2] = (52 * (1.0 - blend) + 62 * blend).astype(np.uint8)
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)

    distractors = [
        ((44 + episode * 37) % (width - 80) + 40, (70 + episode * 23) % (height - 90) + 45),
        ((120 + episode * 29) % (width - 80) + 40, (30 + episode * 41) % (height - 90) + 45),
    ]
    for idx, xy in enumerate(distractors):
        spec = OBJECTS[(episode + idx + 2) % len(OBJECTS)]
        _draw_object(draw, xy, spec[1], tuple(max(20, c - 70) for c in spec[2]), tuple(max(10, c - 45) for c in spec[3]))

    _draw_object(draw, target_xy, shape, fill, outline)
    draw.text((8, 8), f"episode {episode:02d}: target {name}", fill=(245, 245, 235))
    return np.asarray(image, dtype=np.uint8)


def _obs(rgb: np.ndarray, xyz: tuple[float, float, float], object_name: str) -> dict[str, Any]:
    return {
        "episode_metadata": {"target_object": object_name},
        "sensor_data": {"base_camera": {"rgb": rgb}},
        "sensor_param": {
            "base_camera": {
                "intrinsic_cv": np.asarray(
                    [[120.0, 0.0, rgb.shape[1] / 2.0], [0.0, 120.0, rgb.shape[0] / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                "extrinsic_cv": np.eye(4, dtype=np.float32),
            }
        },
        "obj_pose": {"p": np.asarray(xyz, dtype=np.float32)},
    }


def _contact_sheet(paths: list[Path], output_path: Path, thumb_size: tuple[int, int] = (260, 180), cols: int = 3) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size)
        canvas = Image.new("RGB", (thumb_size[0] + 20, thumb_size[1] + 42), (246, 244, 237))
        canvas.paste(image, ((canvas.width - image.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_size[1] + 18), path.stem[:42], fill=(18, 18, 18))
        thumbs.append(canvas)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb_size[0] + 20), rows * (thumb_size[1] + 42)), (230, 228, 220))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * thumb.width, (idx // cols) * thumb.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_html(output_dir: Path, rows: list[dict[str, Any]], full_sheet: Path, zoom_sheet: Path) -> Path:
    html_path = output_dir / "diverse_object_report.html"
    cards = []
    for row in rows:
        image = Path(row["overlay"])
        zoom = Path(row["zoom"])
        cards.append(
            f"""
            <figure>
              <img src="{html.escape(image.relative_to(output_dir).as_posix())}" alt="{html.escape(row['case'])}">
              <img src="{html.escape(zoom.relative_to(output_dir).as_posix())}" alt="{html.escape(row['case'])} zoom">
              <figcaption>
                <strong>{html.escape(row['case'])}</strong>
                <span>{html.escape(row['object'])}; point={row['point_xy']}; center distance={row['distance_from_image_center_px']:.1f}px; error={row['projection_error_px']:.1f}px</span>
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
  <title>Diverse Object Oracle Projection Report</title>
  <style>
    :root {{ --ink:#151511; --line:#d4c4aa; --green:#00c86d; }}
    body {{ margin:0; color:var(--ink); background:linear-gradient(135deg,#efe0c7,#f9f5ed 48%,#e4eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:44px 52px 20px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(34px,5vw,68px); letter-spacing:-.045em; }}
    main {{ padding:30px 52px 64px; display:grid; gap:24px; }}
    section {{ background:rgba(255,250,240,.86); border:1px solid var(--line); border-radius:26px; padding:24px; }}
    .sheet {{ width:min(100%, 980px); border-radius:16px; border:1px solid var(--line); background:white; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
    figure {{ margin:0; padding:12px; background:white; border:2px solid var(--green); border-radius:18px; }}
    figure img {{ width:100%; border-radius:12px; margin-bottom:8px; }}
    figcaption {{ display:grid; gap:4px; font:13px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Diverse Object Oracle Projection Report</h1>
    <p><strong>Source type: synthetic diagnostic.</strong> These images are script-generated object layouts, not ManiSkill/LIBERO rollout episodes.</p>
    <p>Full-frame validation across different target objects, distractors, and off-center positions. This report explicitly checks that overlays are not merely image-center markers.</p>
  </header>
  <main>
    <section><h2>Full-frame contact sheet</h2><img class="sheet" src="{html.escape(full_sheet.relative_to(output_dir).as_posix())}" alt="full-frame sheet"></section>
    <section><h2>Zoom contact sheet</h2><img class="sheet" src="{html.escape(zoom_sheet.relative_to(output_dir).as_posix())}" alt="zoom sheet"></section>
    <section><h2>Synthetic sample details</h2><div class="grid">{''.join(cards)}</div></section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def _zoom_panel(raw: Path, overlay: Path, output: Path, center: tuple[int, int], crop_size: int = 96, scale: int = 3) -> None:
    raw_img = Image.open(raw).convert("RGB")
    overlay_img = Image.open(overlay).convert("RGB")
    def crop(img: Image.Image) -> Image.Image:
        cx, cy = center
        left = max(0, min(img.width - crop_size, cx - crop_size // 2))
        top = max(0, min(img.height - crop_size, cy - crop_size // 2))
        return img.crop((left, top, left + crop_size, top + crop_size)).resize((crop_size * scale, crop_size * scale))
    raw_crop = crop(raw_img)
    overlay_crop = crop(overlay_img)
    canvas = Image.new("RGB", (raw_crop.width * 2 + 42, raw_crop.height + 44), (246, 244, 237))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), "raw", fill=(18, 18, 18))
    draw.text((raw_crop.width + 28, 10), "oracle overlay", fill=(18, 18, 18))
    canvas.paste(raw_crop, (14, 38))
    canvas.paste(overlay_crop, (raw_crop.width + 28, 38))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    overlay_dir = output_dir / "overlay"
    zoom_dir = output_dir / "zoom"
    rows: list[dict[str, Any]] = []
    width, height = 360, 240
    positions = [
        (-0.95, -0.55, 1.35), (-0.55, -0.42, 1.10), (-0.20, -0.60, 1.20), (0.32, -0.48, 1.18), (0.82, -0.36, 1.32),
        (-0.82, -0.08, 1.22), (-0.44, 0.02, 1.08), (0.24, 0.08, 1.14), (0.68, 0.10, 1.28), (0.98, 0.02, 1.42),
        (-0.76, 0.48, 1.32), (-0.34, 0.38, 1.16), (0.10, 0.52, 1.20), (0.52, 0.42, 1.24), (0.86, 0.34, 1.36),
    ]
    for episode in range(args.episodes):
        xyz = positions[episode % len(positions)]
        object_spec = OBJECTS[episode % len(OBJECTS)]
        expected_xy = _project_xy(width, height, xyz)
        rgb = _scene(width, height, expected_xy, object_spec, episode)
        raw_path = raw_dir / f"episode_{episode:03d}_{object_spec[0].replace(' ', '_')}_raw.png"
        overlay_path = overlay_dir / f"episode_{episode:03d}_{object_spec[0].replace(' ', '_')}_overlay.png"
        zoom_path = zoom_dir / f"episode_{episode:03d}_{object_spec[0].replace(' ', '_')}_zoom.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(raw_path)
        _overlay_images, metadata = build_oracle_affordance_overlay(_obs(rgb, xyz, object_spec[0]), output_path=overlay_path, label=object_spec[0])
        _zoom_panel(raw_path, overlay_path, zoom_path, expected_xy)
        actual_xy = tuple(metadata.point_xy)
        projection_error = float(np.linalg.norm(np.asarray(actual_xy) - np.asarray(expected_xy)))
        center_distance = float(np.linalg.norm(np.asarray(expected_xy) - np.asarray([width // 2, height // 2])))
        rows.append(
            {
                "case": f"episode_{episode:03d}",
                "object": object_spec[0],
                "expected_xy": list(expected_xy),
                "point_xy": list(actual_xy),
                "projection_error_px": projection_error,
                "distance_from_image_center_px": center_distance,
                "mode": metadata.mode,
                "raw": str(raw_path),
                "overlay": str(overlay_path),
                "zoom": str(zoom_path),
            }
        )
    full_sheet = output_dir / "diverse_object_full_contact_sheet.png"
    zoom_sheet = output_dir / "diverse_object_zoom_contact_sheet.png"
    _contact_sheet([Path(row["overlay"]) for row in rows], full_sheet)
    _contact_sheet([Path(row["zoom"]) for row in rows], zoom_sheet, thumb_size=(360, 190), cols=2)
    html_path = _write_html(output_dir, rows, full_sheet, zoom_sheet)
    non_center = sum(1 for row in rows if row["distance_from_image_center_px"] >= 45.0)
    passed = all(row["mode"] == "projected_object_pose" and row["projection_error_px"] <= 1.0 for row in rows) and non_center >= 20
    manifest = {
        "status": "passed" if passed else "failed",
        "source_type": "synthetic_diagnostic",
        "real_sim_episode": False,
        "provenance_note": "Script-generated object layouts; not ManiSkill/LIBERO rollout frames.",
        "episode_count": len(rows),
        "sample_count": len(rows),
        "non_center_episode_count": non_center,
        "html": str(html_path),
        "full_contact_sheet": str(full_sheet),
        "zoom_contact_sheet": str(zoom_sheet),
        "rows": rows,
    }
    (output_dir / "diverse_object_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "episodes": len(rows), "non_center": non_center, "html": str(html_path)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
