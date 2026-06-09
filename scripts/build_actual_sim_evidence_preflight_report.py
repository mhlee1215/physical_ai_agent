#!/usr/bin/env python3
"""Build a preflight report for actual-simulation oracle evidence artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_SUFFIXES]


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _checkpoint_rows(checkpoints_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_dir in sorted(checkpoints_root.glob("checkpoint_24*")):
        rollout_dir = checkpoint_dir / "maniskill_rollout"
        frames_dir = rollout_dir / "rollout_frames"
        true_oracle_manifest = rollout_dir / "smolvla_affordance_true_oracle_steps.json"
        fallback_manifest = rollout_dir / "smolvla_real_manifest.json"
        blocker = checkpoint_dir / "maniskill_blocker.md"
        checkpoint_report = _load_json(checkpoint_dir / "checkpoint_report.json")
        true_oracle = _load_json(true_oracle_manifest)
        frames = _images(frames_dir)
        rows.append(
            {
                "checkpoint": checkpoint_dir.name,
                "checkpoint_dir": str(checkpoint_dir),
                "rollout_dir": str(rollout_dir),
                "frame_count": len(frames),
                "sample_frames": [str(path) for path in frames[:6]],
                "has_blocker": blocker.exists(),
                "blocker": str(blocker) if blocker.exists() else None,
                "checkpoint_status": checkpoint_report.get("status", "missing"),
                "requested_env_id": checkpoint_report.get("metrics", {}).get("requested_env_id", checkpoint_report.get("requested_env_id", "")),
                "env_id": checkpoint_report.get("metrics", {}).get("env_id", checkpoint_report.get("env_id", "")),
                "true_oracle_manifest": str(true_oracle_manifest) if true_oracle_manifest.exists() else None,
                "true_oracle_status": true_oracle.get("status", "missing"),
                "strict_true_oracle_step_count": int(true_oracle.get("strict_true_oracle_step_count", 0) or 0),
                "smolvla_real_manifest": str(fallback_manifest) if fallback_manifest.exists() else None,
            }
        )
    return rows


def _representative_frames(rows: list[dict[str, Any]], limit: int) -> list[tuple[str, Path]]:
    picked: list[tuple[str, Path]] = []
    for row in rows:
        for frame in row["sample_frames"]:
            picked.append((row["checkpoint"], Path(frame)))
            if len(picked) >= limit:
                return picked
    return picked


def _contact_sheet(samples: list[tuple[str, Path]], output_path: Path) -> None:
    if not samples:
        image = Image.new("RGB", (920, 180), (246, 241, 231))
        ImageDraw.Draw(image).text((24, 76), "No actual sim RGB frames found.", fill=(20, 18, 14))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return
    cols = 4
    cell_w, cell_h = 260, 190
    rows = (len(samples) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (246, 241, 231))
    draw = ImageDraw.Draw(canvas)
    for idx, (checkpoint, frame) in enumerate(samples):
        col = idx % cols
        row = idx // cols
        x0, y0 = col * cell_w, row * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(210, 190, 160), width=1)
        image = Image.open(frame).convert("RGB")
        image.thumbnail((232, 132))
        canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 28))
        draw.text((x0 + 8, y0 + 8), checkpoint[:30], fill=(20, 18, 14))
        draw.text((x0 + 8, y0 + 166), frame.name[:34], fill=(20, 18, 14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _html(report_path: Path, rows: list[dict[str, Any]], contact_sheet: Path, samples: list[tuple[str, Path]]) -> str:
    table_rows = []
    for row in rows:
        true_status = row["true_oracle_status"]
        table_rows.append(
            f"""
        <tr>
          <td><code>{html.escape(row['checkpoint'])}</code></td>
          <td>{html.escape(str(row['checkpoint_status']))}</td>
          <td>{html.escape(str(row['frame_count']))}</td>
          <td>{html.escape(str(row['has_blocker']))}</td>
          <td>{html.escape(str(true_status))}</td>
          <td>{html.escape(str(row['strict_true_oracle_step_count']))}</td>
        </tr>
            """
        )
    cards = []
    for checkpoint, frame in samples:
        cards.append(
            f"""
        <figure>
          <img src="{html.escape(_rel(report_path, frame))}" alt="{html.escape(frame.name)}">
          <figcaption>{html.escape(checkpoint)} / {html.escape(frame.name)}</figcaption>
        </figure>
            """
        )
    ready = [row for row in rows if row["strict_true_oracle_step_count"] >= 10 and row["true_oracle_status"] == "passed"]
    frame_rich = [row for row in rows if row["frame_count"] >= 10]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual Sim Evidence Preflight</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --green:#008f5b; --block:#b5382d; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#f8ffd0,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1060px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    section {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:{'var(--green)' if ready else 'var(--block)'}; border:1px solid currentColor; background:#fff; }}
    table {{ width:100%; border-collapse:collapse; font-family:Avenir Next, Helvetica, sans-serif; font-size:14px; }}
    th,td {{ padding:10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    code {{ overflow-wrap:anywhere; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual Sim Evidence Preflight</h1>
    <p><strong>Source type: actual_sim_artifact_preflight.</strong> This report scans existing CP24 rollout artifacts and separates actual sim RGB availability from true-oracle projection readiness.</p>
  </header>
  <main>
    <section>
      <span class="pill">{'TRUE ORACLE READY' if ready else 'TRUE ORACLE NOT READY'}</span>
      <h2>Summary</h2>
      <p>Checkpoint dirs scanned: {len(rows)}. Dirs with at least 10 actual sim RGB frames: {len(frame_rich)}. Dirs with true-oracle >=10: {len(ready)}.</p>
      <p><img src="{html.escape(_rel(report_path, contact_sheet))}" alt="contact sheet"></p>
    </section>
    <section>
      <h2>Artifact table</h2>
      <table>
        <thead><tr><th>Checkpoint</th><th>Status</th><th>RGB frames</th><th>Blocker</th><th>True oracle status</th><th>Strict true oracle steps</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>Representative actual sim RGB frames</h2>
      <div class="grid">{''.join(cards)}</div>
    </section>
  </main>
</body>
</html>
"""


def build_report(checkpoints_root: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    rows = _checkpoint_rows(checkpoints_root)
    samples = _representative_frames(rows, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = output_dir / "actual_sim_evidence_preflight_contact_sheet.png"
    _contact_sheet(samples, contact_sheet)
    html_path = output_dir / "actual_sim_evidence_preflight.html"
    html_path.write_text(_html(html_path, rows, contact_sheet, samples), encoding="utf-8")
    ready_count = sum(1 for row in rows if row["strict_true_oracle_step_count"] >= 10 and row["true_oracle_status"] == "passed")
    frame_rich_count = sum(1 for row in rows if row["frame_count"] >= 10)
    report = {
        "status": "passed_true_oracle_ready" if ready_count else "passed_rgb_only_true_oracle_blocked",
        "source_type": "actual_sim_artifact_preflight",
        "real_sim_episode": True,
        "true_oracle_projection_ready": bool(ready_count),
        "checkpoint_count": len(rows),
        "frame_rich_checkpoint_count": frame_rich_count,
        "true_oracle_ready_checkpoint_count": ready_count,
        "sample_count": len(samples),
        "html": str(html_path),
        "contact_sheet": str(contact_sheet),
        "rows": rows,
    }
    (output_dir / "actual_sim_evidence_preflight_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", default="_workspace/checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    report = build_report(Path(args.checkpoints_root), Path(args.output_dir), args.limit)
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
