#!/usr/bin/env python3
"""Build a paper-ready SO101 training and closed-loop pipeline figure."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "docs" / "research" / "2026_06_21",
    ROOT / "papers" / "rss2026_semrob" / "figures",
]
BASE_NAME = "so101_training_loop_pipeline"
WIDTH = 1800
HEIGHT = 1050
ASSET_DIR = ROOT / "docs" / "research" / "2026_06_21" / "so101_pipeline_assets"


COLORS = {
    "ink": "#172026",
    "muted": "#5b6673",
    "paper": "#fbfcfd",
    "panel": "#ffffff",
    "line": "#d7dde5",
    "train": "#2563eb",
    "train_soft": "#eaf1ff",
    "eval": "#059669",
    "eval_soft": "#e8f7f0",
    "aug": "#d97706",
    "aug_soft": "#fff4df",
    "val": "#7c3aed",
    "val_soft": "#f2ecff",
    "cache": "#0891b2",
    "cache_soft": "#e6f7fb",
    "gray_soft": "#f2f5f8",
}

SAMPLE_DATASET = (
    ROOT
    / "_workspace"
    / "hf_training_integration_test"
    / "mhlee1215__so101-nexus-sim-dataset"
    / "datasets"
    / "pick_cube_grip_focus"
    / "validation"
    / "data"
    / "chunk-000"
    / "file-000.parquet"
)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def png_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    import base64

    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def extract_image_bytes(row: dict, column: str) -> bytes:
    value = row[column]
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is not None:
            return bytes(data)
    if hasattr(value, "as_py"):
        return extract_image_bytes({column: value.as_py()}, column)
    raise TypeError(f"Unsupported image cell for {column}: {type(value)!r}")


def draw_label(draw, xy: tuple[int, int], text: str, *, fill=(16, 24, 32)) -> None:
    try:
        from PIL import ImageFont

        font = ImageFont.truetype("Arial.ttf", 22)
    except Exception:
        font = None
    draw.text(xy, text, fill=fill, font=font)


def build_sample_assets() -> dict[str, Path]:
    """Create small representative image assets from a local SO101 dataset."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "camera_pair": ASSET_DIR / "camera_pair.png",
        "trajectory": ASSET_DIR / "trajectory_strip.png",
    }
    try:
        from PIL import Image, ImageDraw
        import pyarrow.parquet as pq
    except Exception:
        return paths

    if not SAMPLE_DATASET.exists():
        return paths

    table = pq.read_table(SAMPLE_DATASET)
    frame_indices = [0, table.num_rows // 2, table.num_rows - 1]
    rows = [table.slice(i, 1).to_pylist()[0] for i in frame_indices]
    first = rows[0]

    cam1 = Image.open(BytesIO(extract_image_bytes(first, "observation.images.camera1"))).convert("RGB")
    cam2 = Image.open(BytesIO(extract_image_bytes(first, "observation.images.camera2"))).convert("RGB")
    thumb_size = (360, 260)
    cam1 = cam1.resize((220, 220))
    cam2 = cam2.resize((220, 220))
    pair = Image.new("RGB", (760, 300), "white")
    draw = ImageDraw.Draw(pair)
    draw.rounded_rectangle((0, 0, 759, 299), radius=26, fill=(255, 255, 255), outline=(213, 221, 229), width=3)
    draw_label(draw, (28, 20), "Policy inputs")
    draw_label(draw, (92, 58), "camera1: egocentric")
    draw_label(draw, (468, 58), "camera2: wrist")
    pair.paste(cam1, (70, 88))
    pair.paste(cam2, (448, 88))
    pair.save(paths["camera_pair"])

    strip = Image.new("RGB", (760, 300), "white")
    draw = ImageDraw.Draw(strip)
    draw.rounded_rectangle((0, 0, 759, 299), radius=26, fill=(255, 255, 255), outline=(213, 221, 229), width=3)
    draw_label(draw, (28, 20), "Trajectory / loop evidence")
    labels = ["start", "approach", "grasp/lift"]
    for idx, row in enumerate(rows):
        img = Image.open(BytesIO(extract_image_bytes(row, "observation.images.camera1"))).convert("RGB")
        img = img.resize((180, 180))
        x = 55 + idx * 235
        strip.paste(img, (x, 66))
        draw_label(draw, (x + 38, 260), labels[idx])
    strip.save(paths["trajectory"])
    return paths


def text_block(x: int, y: int, lines: list[str], *, size: int = 25, color: str = "ink",
               weight: int = 500, leading: int | None = None) -> str:
    leading = leading or int(size * 1.35)
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else leading
        tspans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text font-size="{size}" font-weight="{weight}" fill="{COLORS[color]}" '
        f'font-family="Arial">{"".join(tspans)}</text>'
        .replace(">", f' x="{x}" y="{y}">', 1)
    )


def title_text(x: int, y: int, title: str, subtitle: str) -> str:
    return f"""
    <text x="{x}" y="{y}" font-size="48" font-weight="760" fill="{COLORS['ink']}"
      font-family="Arial">{esc(title)}</text>
    <text x="{x}" y="{y + 48}" font-size="24" font-weight="460" fill="{COLORS['muted']}"
      font-family="Arial">{esc(subtitle)}</text>
    """


def lane(x: int, y: int, w: int, h: int, label: str, color: str) -> str:
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{COLORS['panel']}"
      stroke="{COLORS['line']}" stroke-width="2" filter="url(#shadow)"/>
    <rect x="{x}" y="{y}" width="14" height="{h}" rx="7" fill="{COLORS[color]}"/>
    <text x="{x + 28}" y="{y + 42}" font-size="25" font-weight="760" fill="{COLORS[color]}"
      font-family="Arial">{esc(label)}</text>
    """


def box(x: int, y: int, w: int, h: int, title: str, lines: list[str], *,
        color: str, soft: str, icon: str) -> str:
    body_lines = []
    compact = h <= 145
    line_y = y + (70 if compact else 76)
    body_font = 19 if compact else 20
    line_step = 25 if compact else 28
    for i, line in enumerate(lines):
        body_lines.append(
            f'<text x="{x + 30}" y="{line_y + i * line_step}" font-size="{body_font}" font-weight="450" '
            f'fill="{COLORS["muted"]}" font-family="Arial">'
            f'{esc(line)}</text>'
        )
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{COLORS[soft]}"
      stroke="{COLORS[color]}" stroke-width="2"/>
    <circle cx="{x + 34}" cy="{y + 34}" r="18" fill="{COLORS[color]}"/>
    <text x="{x + 34}" y="{y + 42}" text-anchor="middle" font-size="22" font-weight="760"
      fill="white" font-family="Arial">{esc(icon)}</text>
    <text x="{x + 66}" y="{y + 42}" font-size="25" font-weight="760" fill="{COLORS['ink']}"
      font-family="Arial">{esc(title)}</text>
    {''.join(body_lines)}
    """


def pill(x: int, y: int, label: str, *, color: str, w: int | None = None) -> str:
    w = w or max(150, len(label) * 12 + 34)
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="38" rx="19" fill="white"
      stroke="{COLORS[color]}" stroke-width="2"/>
    <text x="{x + w / 2:.1f}" y="{y + 25}" text-anchor="middle" font-size="18"
      font-weight="650" fill="{COLORS[color]}" font-family="Arial">
      {esc(label)}</text>
    """


def arrow(x1: int, y1: int, x2: int, y2: int, *, color: str = "muted",
          label: str | None = None, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="9 9"' if dashed else ""
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    label_svg = ""
    if label:
        label_svg = f"""
        <rect x="{mid_x - 74}" y="{mid_y - 31}" width="148" height="29" rx="14"
          fill="{COLORS['paper']}" opacity="0.96"/>
        <text x="{mid_x}" y="{mid_y - 10}" text-anchor="middle" font-size="17"
          font-weight="650" fill="{COLORS[color]}" font-family="Arial">
          {esc(label)}</text>
        """
    return f"""
    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS[color]}"
      stroke-width="4" stroke-linecap="round" marker-end="url(#{color}-arrow)"{dash}/>
    {label_svg}
    """


def curve(path: str, *, color: str, label: str | None = None, lx: int = 0, ly: int = 0) -> str:
    label_svg = ""
    if label:
        label_svg = f"""
        <rect x="{lx - 92}" y="{ly - 24}" width="184" height="34" rx="17"
          fill="{COLORS['paper']}" opacity="0.96"/>
        <text x="{lx}" y="{ly}" text-anchor="middle" font-size="18" font-weight="700"
          fill="{COLORS[color]}" font-family="Arial">{esc(label)}</text>
        """
    return f"""
    <path d="{path}" fill="none" stroke="{COLORS[color]}" stroke-width="4"
      stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{color}-arrow)"/>
    {label_svg}
    """


def image_panel(x: int, y: int, w: int, h: int, title: str, image_uri: str | None) -> str:
    if image_uri:
        image = (
            f'<image x="{x + 18}" y="{y + 58}" width="{w - 36}" height="{h - 78}" '
            f'preserveAspectRatio="xMidYMid meet" href="{image_uri}"/>'
        )
    else:
        image = f"""
        <rect x="{x + 18}" y="{y + 58}" width="{w - 36}" height="{h - 78}" rx="14"
          fill="{COLORS['gray_soft']}" stroke="{COLORS['line']}" stroke-width="2"/>
        <text x="{x + w / 2:.1f}" y="{y + h / 2:.1f}" text-anchor="middle" font-size="22"
          font-weight="650" fill="{COLORS['muted']}" font-family="Arial">sample image unavailable</text>
        """
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{COLORS['panel']}"
      stroke="{COLORS['line']}" stroke-width="2" filter="url(#shadow)"/>
    <text x="{x + 24}" y="{y + 38}" font-size="25" font-weight="760" fill="{COLORS['ink']}"
      font-family="Arial">{esc(title)}</text>
    {image}
    """


def prompt_card(x: int, y: int, w: int, title: str, prompt: str, *, color: str = "aug") -> str:
    return f"""
    <rect x="{x}" y="{y}" width="{w}" height="56" rx="18" fill="white"
      stroke="{COLORS[color]}" stroke-width="2"/>
    <text x="{x + 24}" y="{y + 24}" font-size="17" font-weight="760" fill="{COLORS[color]}"
      font-family="Arial">{esc(title)}</text>
    <text x="{x + 24}" y="{y + 45}" font-size="19" font-weight="520" fill="{COLORS['ink']}"
      font-family="Arial">{esc(prompt)}</text>
    """


def build_svg() -> str:
    sample_paths = build_sample_assets()
    camera_pair_uri = png_data_uri(sample_paths["camera_pair"])
    trajectory_uri = png_data_uri(sample_paths["trajectory"])
    defs = f"""
    <defs>
      <filter id="shadow" x="-5%" y="-5%" width="110%" height="120%">
        <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#172026" flood-opacity="0.10"/>
      </filter>
      {''.join(
        f'<marker id="{name}-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" '
        f'orient="auto" markerUnits="strokeWidth">'
        f'<path d="M2,2 L10,6 L2,10 Z" fill="{value}"/></marker>'
        for name, value in COLORS.items()
        if name in {"train", "eval", "val", "aug", "cache", "muted"}
      )}
      <linearGradient id="topGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#eaf1ff"/>
        <stop offset="0.48" stop-color="#e8f7f0"/>
        <stop offset="1" stop-color="#fff4df"/>
      </linearGradient>
    </defs>
    """

    contract = f"""
    <rect x="92" y="140" width="1616" height="82" rx="20" fill="url(#topGrad)"
      stroke="{COLORS['line']}" stroke-width="2"/>
    <text x="124" y="174" font-size="23" font-weight="760" fill="{COLORS['ink']}"
      font-family="Arial">Fixed policy interface</text>
    <text x="124" y="203" font-size="20" font-weight="450" fill="{COLORS['muted']}"
      font-family="Arial">Same inputs and outputs in both lanes.</text>
    {pill(650, 162, "camera1: egocentric", color="train", w=235)}
    {pill(905, 162, "camera2: wrist", color="eval", w=175)}
    {pill(1100, 162, "SmolVLA 256px inputs", color="cache", w=245)}
    {pill(1365, 162, "6D state/action", color="val", w=190)}
    {pill(1575, 162, "15-step rollout", color="aug", w=145)}
    """

    training = f"""
    {lane(70, 260, 1025, 255, "1. Training pipeline", "train")}
    {box(105, 320, 255, 150, "Demonstrations", ["teacher rollouts", "task datasets", "camera-locked samples"], color="train", soft="train_soft", icon="D")}
    {box(420, 320, 250, 150, "SmolVLA tune", ["visual observations", "language instruction", "action chunks"], color="cache", soft="cache_soft", icon="T")}
    {box(730, 320, 325, 150, "Primitive policy", ["fine-tuned SmolVLA", "move / align / pick skills", "checkpoint exported"], color="val", soft="val_soft", icon="P")}
    {arrow(360, 395, 420, 395, color="train")}
    {arrow(670, 395, 730, 395, color="cache")}
    """

    loop = f"""
    {lane(70, 575, 1025, 275, "2. Closed-loop test pipeline", "eval")}
    {prompt_card(100, 630, 960, "Example planner prompt", "Task: pick and lift the green cube", color="aug")}
    {box(100, 705, 190, 140, "Checkpoint", ["SmolVLA policy", "same camera input", "no train augment"], color="val", soft="val_soft", icon="C")}
    {box(330, 705, 210, 140, "Qwen planner", ["tool schema", "ordered primitives", "no hidden reasoning"], color="aug", soft="aug_soft", icon="Q")}
    <rect x="580" y="705" width="235" height="140" rx="16" fill="{COLORS['gray_soft']}"
      stroke="{COLORS['line']}" stroke-width="2"/>
    <text x="610" y="742" font-size="25" font-weight="760" fill="{COLORS['ink']}" font-family="Arial">Tool calls</text>
    <rect x="610" y="760" width="155" height="24" rx="12" fill="white" stroke="{COLORS['train']}" stroke-width="2"/>
    <text x="687" y="778" text-anchor="middle" font-size="15" font-weight="700" fill="{COLORS['train']}" font-family="Arial">move(edge)</text>
    <rect x="610" y="790" width="155" height="24" rx="12" fill="white" stroke="{COLORS['cache']}" stroke-width="2"/>
    <text x="687" y="808" text-anchor="middle" font-size="15" font-weight="700" fill="{COLORS['cache']}" font-family="Arial">align(jaws)</text>
    <text x="610" y="833" font-size="16" font-weight="520" fill="{COLORS['muted']}" font-family="Arial">then pick_up(edge)</text>
    {box(855, 705, 205, 140, "Execute", ["run primitive policy", "valid-mask stop", "observe next state"], color="eval", soft="eval_soft", icon="X")}
    {arrow(290, 775, 330, 775, color="val")}
    {arrow(540, 775, 580, 775, color="aug")}
    {arrow(815, 775, 855, 775, color="eval")}
    {curve("M965 705 C965 610, 965 540, 930 470", color="val", label="use checkpoint", lx=1010, ly=565)}
    {curve("M950 845 C950 930, 230 930, 230 845", color="eval", label="metrics + failures", lx=565, ly=922)}
    """

    samples = f"""
    {image_panel(1165, 260, 545, 275, "Sample policy inputs", camera_pair_uri)}
    {image_panel(1165, 575, 545, 275, "Sample rollout evidence", trajectory_uri)}
    """

    footer = f"""
    <rect x="92" y="884" width="1616" height="76" rx="18" fill="{COLORS['gray_soft']}"
      stroke="{COLORS['line']}" stroke-width="2"/>
    <text x="124" y="917" font-size="22" font-weight="760" fill="{COLORS['ink']}"
      font-family="Arial">What a coworker should take away</text>
    <text x="124" y="946" font-size="20" font-weight="450" fill="{COLORS['muted']}"
      font-family="Arial">Core idea: train visual primitive policies from SO101 demonstrations, then let Qwen call those primitives inside a closed-loop robot test and feed failures back into the next data pass.</text>
    <text x="1695" y="1000" text-anchor="end" font-size="17" font-weight="450" fill="{COLORS['muted']}"
      font-family="Arial">Generated from repository pipeline notes, 2026-06-21</text>
    """

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  {defs}
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS['paper']}"/>
  {title_text(95, 76, "SO101 Training and Closed-Loop Evaluation Framework", "A high-level view of how visual demonstrations become deployable robot policies.")}
  {contract}
  {training}
  {loop}
  {samples}
  {footer}
</svg>
"""


def find_chrome() -> str | None:
    candidates = [
        os.environ.get("CHROME"),
        shutil.which("chrome-headless-shell"),
        shutil.which("chromium"),
        shutil.which("google-chrome"),
        shutil.which("Google Chrome"),
        "/Users/minhaeng/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def render_with_chrome(chrome: str, svg_path: Path, png_path: Path, pdf_path: Path) -> None:
    base = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
    ]
    subprocess.run(
        [
            *base,
            f"--window-size={WIDTH},{HEIGHT}",
            f"--screenshot={png_path}",
            svg_path.as_uri(),
        ],
        check=True,
    )
    subprocess.run(
        [
            *base,
            f"--print-to-pdf={pdf_path}",
            svg_path.as_uri(),
        ],
        check=True,
    )


def write_outputs() -> None:
    svg_text = build_svg()
    chrome = find_chrome()
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        svg_path = out_dir / f"{BASE_NAME}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")

        if chrome:
            render_with_chrome(
                chrome,
                svg_path,
                out_dir / f"{BASE_NAME}.png",
                out_dir / f"{BASE_NAME}.pdf",
            )
        else:
            print(f"warning: no Chrome renderer found; wrote SVG only: {svg_path}")


if __name__ == "__main__":
    write_outputs()
