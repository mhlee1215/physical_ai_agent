#!/usr/bin/env python3
"""Build a cross-checkpoint/episode actual-sim RGB diversity report."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
EPISODE_RE = re.compile(r"episode_(\d+)")


def _images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_SUFFIXES]


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _episode_key(path: Path) -> str:
    match = EPISODE_RE.search(path.name)
    return f"episode_{match.group(1)}" if match else "episode_unknown"


def _collect(checkpoints_root: Path, limit: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for checkpoint in sorted(checkpoints_root.glob("checkpoint_24*")):
        frames_dir = checkpoint / "maniskill_rollout" / "rollout_frames"
        for frame in _images(frames_dir):
            groups[(checkpoint.name, _episode_key(frame))].append(frame)
    rows = []
    for (checkpoint, episode), frames in sorted(groups.items()):
        if not frames:
            continue
        # Pick early, middle, late frames when possible.
        indices = sorted({0, len(frames) // 2, len(frames) - 1})
        for index in indices:
            frame = frames[index]
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "episode": episode,
                    "frame": str(frame),
                    "frame_name": frame.name,
                    "within_episode_index": index,
                    "episode_frame_count": len(frames),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    cols = 4
    cell_w, cell_h = 280, 206
    canvas = Image.new("RGB", (cols * cell_w, ((len(rows) + cols - 1) // cols) * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        col = idx % cols
        grid_row = idx // cols
        x0, y0 = col * cell_w, grid_row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(210, 190, 160), width=1)
        image = Image.open(row["frame"]).convert("RGB")
        image.thumbnail((252, 142))
        canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 38))
        draw.text((x0 + 8, y0 + 8), row["checkpoint"][:34], fill=(20, 18, 14))
        draw.text((x0 + 8, y0 + 24), f"{row['episode']} idx={row['within_episode_index']}", fill=(20, 18, 14))
        draw.text((x0 + 8, y0 + 184), row["frame_name"][:36], fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
      <article>
        <h2>{html.escape(row['checkpoint'])} / {html.escape(row['episode'])}</h2>
        <p>Frame: <code>{html.escape(row['frame_name'])}</code>; within-episode index: {row['within_episode_index']} / {row['episode_frame_count']}</p>
        <figure><img src="{html.escape(_rel(report_path, Path(row['frame'])))}" alt="sim frame"><figcaption>actual CP24 sim RGB</figcaption></figure>
      </article>
            """
        )
    checkpoint_count = len({row["checkpoint"] for row in rows})
    episode_count = len({(row["checkpoint"], row["episode"]) for row in rows})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual Sim Cross-Episode Diversity</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --green:#008f5b; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#e0f7ff,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1080px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    .summary, article {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:var(--green); border:1px solid currentColor; background:#fff; }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    code {{ overflow-wrap:anywhere; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim Cross-Episode Diversity</h1>
    <p><strong>Source type: actual_sim_rgb_cross_episode_diversity.</strong> This page uses only existing CP24 simulation RGB rollout frames and samples across checkpoints/episodes to avoid relying on one early rollout segment.</p>
  </header>
  <main>
    <section class="summary">
      <span class="pill">ACTUAL SIM DIVERSITY</span>
      <h2>Summary</h2>
      <p>Samples: {len(rows)}. Checkpoints: {checkpoint_count}. Checkpoint/episode groups: {episode_count}.</p>
      <figure><img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact sheet"><figcaption>cross-episode contact sheet</figcaption></figure>
    </section>
    <section><h2>Sample details</h2><div class="grid">{''.join(cards)}</div></section>
  </main>
</body>
</html>
"""


def build_report(checkpoints_root: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    rows = _collect(checkpoints_root, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = output_dir / "actual_sim_cross_episode_contact_sheet.png"
    _contact_sheet(rows, contact_sheet)
    html_path = output_dir / "actual_sim_cross_episode_report.html"
    html_path.write_text(_html(html_path, rows, contact_sheet), encoding="utf-8")
    checkpoint_count = len({row["checkpoint"] for row in rows})
    episode_group_count = len({(row["checkpoint"], row["episode"]) for row in rows})
    report = {
        "status": "passed" if len(rows) >= 10 and episode_group_count >= 3 else "failed",
        "source_type": "actual_sim_rgb_cross_episode_diversity",
        "real_sim_episode": True,
        "true_oracle_projection": False,
        "sample_count": len(rows),
        "checkpoint_count": checkpoint_count,
        "episode_group_count": episode_group_count,
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "rows": rows,
    }
    (output_dir / "actual_sim_cross_episode_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", default="_workspace/checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    report = build_report(Path(args.checkpoints_root), Path(args.output_dir), args.limit)
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
