from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def build_overlay_gallery(
    image_root: Path,
    output_dir: Path,
    title: str,
    limit: int = 40,
    min_frames: int = 10,
) -> dict[str, Any]:
    """Build a lightweight HTML/contact-sheet/GIF gallery for overlay frames."""

    from PIL import Image, ImageDraw
    import numpy as np

    def image_paths(root: Path) -> list[Path]:
        suffixes = {".png", ".jpg", ".jpeg", ".webp"}
        paths = [path for path in sorted(root.rglob("*")) if path.suffix.lower() in suffixes]
        return paths[:limit]

    def rel(report_path: Path, image_path: Path) -> str:
        try:
            return image_path.relative_to(report_path.parent).as_posix()
        except ValueError:
            return image_path.as_posix()

    def marker_pixels(path: Path) -> int:
        arr = np.asarray(Image.open(path).convert("RGB"))
        green = (arr[:, :, 1] > 180) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 130)
        return int(green.sum())

    paths = image_paths(image_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "overlay_gallery_manifest.json"
    if len(paths) < min_frames:
        manifest = {
            "status": "skipped",
            "reason": f"expected at least {min_frames} images, found {len(paths)}",
            "title": title,
            "image_root": str(image_root),
            "frame_count": len(paths),
            "frames": [str(path) for path in paths],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    contact_sheet = output_dir / "contact_sheet.png"
    gif_path = output_dir / "overlay_sequence.gif"
    html_path = output_dir / "overlay_gallery.html"

    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((220, 160))
        canvas = Image.new("RGB", (240, 200), (246, 244, 237))
        canvas.paste(image, ((240 - image.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 176), path.stem[:34], fill=(18, 18, 18))
        thumbs.append(canvas)
    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows * 200), (230, 228, 220))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 240, (idx // cols) * 200))
    sheet.save(contact_sheet)

    frames = [Image.open(path).convert("RGB") for path in paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=160,
        loop=0,
    )

    cards = []
    for idx, path in enumerate(paths):
        cards.append(
            f"""
            <figure class="card">
              <img src="{html.escape(rel(html_path, path))}" alt="{html.escape(path.name)}">
              <figcaption>
                <strong>{idx:03d}: {html.escape(path.name)}</strong>
                <span>green marker pixels: {marker_pixels(path)}</span>
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
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #161611;
      --paper: #f6f0e3;
      --panel: #fffaf0;
      --line: #d6c6ad;
      --green: #00d968;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff0bd, transparent 30%),
        linear-gradient(135deg, #efe1c8, #f8f4ec 48%, #e8efe8);
      color: var(--ink);
      font-family: Charter, Georgia, serif;
    }}
    header {{ padding: 44px 52px 22px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 10px; font-size: clamp(34px, 5vw, 68px); letter-spacing: -0.045em; }}
    main {{ padding: 30px 52px 64px; }}
    section {{
      margin-bottom: 28px;
      padding: 24px;
      background: rgba(255, 250, 240, 0.84);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: 0 18px 48px rgba(54, 38, 20, 0.08);
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      font-family: Avenir Next, Helvetica, sans-serif;
    }}
    .pill {{ padding: 8px 12px; border: 1px solid var(--green); border-radius: 999px; background: rgba(0,217,104,.12); }}
    .hero {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .hero img, .card img {{
      width: 100%;
      border-radius: 14px;
      background: #2b2b2b;
      border: 1px solid var(--line);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 18px;
    }}
    .card {{
      margin: 0;
      padding: 12px;
      border-radius: 18px;
      background: white;
      border: 2px solid var(--green);
    }}
    figcaption {{
      display: grid;
      gap: 4px;
      margin-top: 9px;
      font-family: Avenir Next, Helvetica, sans-serif;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="summary">
      <span class="pill">{len(paths)} frames</span>
      <span class="pill">source: {html.escape(str(image_root))}</span>
    </div>
  </header>
  <main>
    <section class="hero">
      <figure>
        <img src="{html.escape(rel(html_path, contact_sheet))}" alt="contact sheet">
        <figcaption>Contact sheet for quick visual inspection.</figcaption>
      </figure>
      <figure>
        <img src="{html.escape(rel(html_path, gif_path))}" alt="animated gif">
        <figcaption>Animated overlay sequence.</figcaption>
      </figure>
    </section>
    <section>
      <h2>Frame gallery</h2>
      <div class="grid">{''.join(cards)}</div>
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )

    manifest = {
        "status": "passed",
        "title": title,
        "image_root": str(image_root),
        "frames": [str(path) for path in paths],
        "frame_count": len(paths),
        "contact_sheet": str(contact_sheet),
        "gif": str(gif_path),
        "html": str(html_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
