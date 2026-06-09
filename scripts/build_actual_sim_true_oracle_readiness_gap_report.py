#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_OUTPUT_ROOT = Path("_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z")
DEFAULT_REPORT_DIR = DEFAULT_OUTPUT_ROOT / "actual_sim_true_oracle_readiness_gap"
FRAME_RE = re.compile(
    r"(?P<policy>[a-zA-Z0-9_]+)_episode_(?P<episode>\d+)_(?P<kind>reset|step)_(?P<step>\d+)\.png$"
)


@dataclass(frozen=True)
class ReadinessSample:
    sample_id: str
    source_frame: str
    display_frame: str
    checkpoint: str
    policy: str
    episode: int
    step: int
    has_actual_sim_rgb: bool
    has_same_step_object_pose: bool
    has_same_step_camera_metadata: bool
    has_true_oracle_overlay: bool
    tier: str
    status: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root)
    report_dir = Path(args.output_dir) if args.output_dir else root / "actual_sim_true_oracle_readiness_gap"
    images_dir = report_dir / "images"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    frames = _discover_actual_sim_frames()
    samples = _select_diverse_samples(frames, limit=args.limit)
    rendered = [_render_sample(sample, images_dir) for sample in samples]
    contact_sheet = report_dir / "actual_sim_true_oracle_readiness_gap_contact_sheet.jpg"
    _write_contact_sheet([Path(sample.display_frame) for sample in rendered], contact_sheet)

    manifest = {
        "status": "blocked_true_oracle_metadata_missing",
        "source_type": "actual_sim_true_oracle_readiness_gap",
        "real_sim_episode": True,
        "true_oracle_projection": False,
        "sample_count": len(rendered),
        "strict_true_oracle_step_count": 0,
        "minimum_required_samples": 10,
        "requirements": {
            "same_step_rgb": True,
            "same_step_object_pose": False,
            "same_step_camera_metadata": False,
            "same_step_true_oracle_overlay": False,
        },
        "samples": [asdict(sample) for sample in rendered],
        "contact_sheet": str(contact_sheet),
    }
    (report_dir / "actual_sim_true_oracle_readiness_gap_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / "actual_sim_true_oracle_readiness_gap.html").write_text(
        _html(rendered, manifest),
        encoding="utf-8",
    )
    print(report_dir / "actual_sim_true_oracle_readiness_gap.html")
    return 0


def _discover_actual_sim_frames() -> list[Path]:
    roots = [
        Path("_workspace/checkpoints"),
    ]
    frames: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        frames.extend(sorted(root.glob("**/maniskill_rollout/rollout_frames/*.png")))
    return [path for path in frames if FRAME_RE.search(path.name)]


def _select_diverse_samples(frames: list[Path], limit: int) -> list[Path]:
    scored: list[tuple[tuple[str, str, str], Path]] = []
    for path in frames:
        match = FRAME_RE.search(path.name)
        if not match:
            continue
        checkpoint = _checkpoint_name(path)
        key = (checkpoint, match.group("policy"), match.group("episode"))
        scored.append((key, path))

    chosen: list[Path] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for key, path in sorted(scored, key=lambda item: (item[0], item[1].name)):
        if key in seen_keys:
            continue
        chosen.append(path)
        seen_keys.add(key)
        if len(chosen) >= limit:
            return chosen

    for _, path in scored:
        if path not in chosen:
            chosen.append(path)
        if len(chosen) >= limit:
            break
    return chosen


def _render_sample(path: Path, images_dir: Path) -> ReadinessSample:
    match = FRAME_RE.search(path.name)
    if not match:
        raise RuntimeError(f"Unexpected frame name: {path}")
    checkpoint = _checkpoint_name(path)
    policy = match.group("policy")
    episode = int(match.group("episode"))
    step = int(match.group("step"))
    sample_id = f"{checkpoint}_{policy}_ep{episode:03d}_step{step:04d}"
    display_path = images_dir / f"{sample_id}.jpg"

    image = Image.open(path).convert("RGB")
    image.thumbnail((720, 420))
    canvas = Image.new("RGB", (760, 500), (246, 238, 218))
    x = (760 - image.width) // 2
    canvas.paste(image, (x, 24))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((14, 14, 746, 486), outline=(178, 63, 48), width=4)
    draw.rectangle((18, 430, 742, 482), fill=(255, 250, 239))
    draw.text((30, 438), "ACTUAL SIM RGB: present", fill=(28, 96, 61))
    draw.text((250, 438), "POSE: missing", fill=(178, 63, 48))
    draw.text((398, 438), "CAMERA: missing", fill=(178, 63, 48))
    draw.text((576, 438), "TIER O: blocked", fill=(178, 63, 48))
    draw.text((30, 462), f"{checkpoint} | {policy} | ep {episode} | step {step}", fill=(42, 48, 43))
    canvas.save(display_path, quality=92)

    return ReadinessSample(
        sample_id=sample_id,
        source_frame=str(path),
        display_frame=str(display_path),
        checkpoint=checkpoint,
        policy=policy,
        episode=episode,
        step=step,
        has_actual_sim_rgb=True,
        has_same_step_object_pose=False,
        has_same_step_camera_metadata=False,
        has_true_oracle_overlay=False,
        tier="A_not_O",
        status="blocked_metadata_missing",
    )


def _checkpoint_name(path: Path) -> str:
    parts = path.parts
    if "checkpoints" not in parts:
        return "unknown_checkpoint"
    index = parts.index("checkpoints")
    if index + 1 >= len(parts):
        return "unknown_checkpoint"
    return parts[index + 1]


def _write_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((380, 250))
        tile = Image.new("RGB", (380, 250), (255, 250, 239))
        tile.paste(image, ((380 - image.width) // 2, (250 - image.height) // 2))
        thumbs.append(tile)
    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 380, rows * 250), (238, 222, 188))
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 380
        y = (index // columns) * 250
        sheet.paste(thumb, (x, y))
    sheet.save(output_path, quality=92)


def _html(samples: list[ReadinessSample], manifest: dict[str, object]) -> str:
    cards = "\n".join(_card(sample) for sample in samples)
    sample_count = manifest["sample_count"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Actual Sim True-Oracle Readiness Gap</title>
  <style>
    :root {{
      --ink: #20271f;
      --muted: #66705f;
      --paper: #f7eed9;
      --panel: #fffaf0;
      --red: #b23f30;
      --green: #2a744d;
      --gold: #b38323;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 4%, rgba(178,63,48,.17), transparent 28%),
        radial-gradient(circle at 86% 12%, rgba(179,131,35,.18), transparent 30%),
        linear-gradient(135deg, #fbf4e4, #eadbbb);
    }}
    header {{ padding: 42px clamp(18px, 5vw, 72px) 22px; }}
    h1 {{
      margin: 0;
      max-width: 1100px;
      font-size: clamp(36px, 6vw, 78px);
      letter-spacing: -.06em;
      line-height: .94;
    }}
    .lead {{
      max-width: 980px;
      color: var(--muted);
      font-size: clamp(17px, 2vw, 22px);
      line-height: 1.45;
      margin-top: 18px;
    }}
    main {{ padding: 0 clamp(18px, 5vw, 72px) 70px; }}
    .verdict {{
      max-width: 1180px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .tile {{
      background: rgba(255,250,240,.82);
      border: 1px solid rgba(93,78,42,.25);
      border-radius: 22px;
      padding: 18px;
      min-height: 130px;
    }}
    .tile b {{ display: block; font-size: 28px; margin-bottom: 8px; }}
    .tile span {{ color: var(--muted); line-height: 1.35; }}
    .grid {{
      max-width: 1180px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(93,78,42,.25);
      border-radius: 24px;
      overflow: hidden;
      box-shadow: 0 18px 52px rgba(46,35,12,.13);
    }}
    .card img {{ width: 100%; display: block; }}
    .body {{ padding: 14px 16px 18px; }}
    .body h2 {{ font-size: 19px; margin: 0 0 8px; letter-spacing: -.02em; }}
    .body p {{ margin: 5px 0; color: var(--muted); line-height: 1.38; font-size: 14px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .chip {{
      padding: 5px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(32,39,31,.08);
    }}
    .bad {{ color: var(--red); }}
    .ok {{ color: var(--green); }}
    footer {{
      max-width: 1180px;
      margin-top: 30px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 980px) {{
      .verdict, .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim True-Oracle Readiness Gap</h1>
    <p class="lead">This report uses real simulator RGB frames only. It deliberately does not claim true-oracle projection: the missing pieces are same-step object pose and camera metadata.</p>
  </header>
  <main>
    <section class="verdict">
      <div class="tile"><b class="ok">{sample_count}</b><span>actual simulation RGB samples shown</span></div>
      <div class="tile"><b class="bad">0</b><span>strict true-oracle samples available</span></div>
      <div class="tile"><b class="bad">missing</b><span>same-step object pose in current saved artifacts</span></div>
      <div class="tile"><b class="bad">missing</b><span>same-step camera metadata in current saved artifacts</span></div>
    </section>
    <section class="card" style="max-width:1180px;margin-bottom:18px;">
      <img src="actual_sim_true_oracle_readiness_gap_contact_sheet.jpg" alt="readiness gap contact sheet" />
      <div class="body">
        <h2>Representative actual-sim RGB gap contact sheet</h2>
        <p>All panels are real simulator RGB frames with explicit missing pose/camera/Tier-O labels.</p>
      </div>
    </section>
    <section class="grid">
      {cards}
    </section>
    <footer>
      Manifest: actual_sim_true_oracle_readiness_gap_manifest.json. This is a Tier-A/Tier-O gap report, not a Tier-O success report.
    </footer>
  </main>
</body>
</html>
"""


def _card(sample: ReadinessSample) -> str:
    image_src = Path(sample.display_frame).name
    return f"""
      <article class="card">
        <img src="images/{image_src}" alt="{sample.sample_id}" />
        <div class="body">
          <h2>{sample.policy} episode {sample.episode}, step {sample.step}</h2>
          <p>{sample.checkpoint}</p>
          <div class="chips">
            <span class="chip ok">actual RGB present</span>
            <span class="chip bad">pose missing</span>
            <span class="chip bad">camera missing</span>
            <span class="chip bad">Tier O blocked</span>
          </div>
        </div>
      </article>
    """


if __name__ == "__main__":
    raise SystemExit(main())
