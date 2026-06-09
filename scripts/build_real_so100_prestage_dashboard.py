#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from scripts.video_preview import write_preview_gif


def build_prestage_dashboard(
    *,
    audit_manifest: Path,
    output: Path,
    title: str = "Real SO-100 Agentic Pre-stage Dashboard",
) -> dict[str, Any]:
    audit = _load_json(audit_manifest)
    pack = _load_json(Path(audit["pre_stage_pack"]))
    runbook = _load_json(Path(audit["runbook_manifest"])) if audit.get("runbook_manifest") else {}
    movement = _load_json(Path(pack["movement_report_manifest"]))
    gate = _load_json(Path(pack["gate_report_manifest"])) if pack.get("gate_report_manifest") else {}
    videos = _motion_videos_from_movement(movement)
    videos = _with_preview_gifs(videos=videos, output_dir=output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_html(
            title=title,
            output=output,
            audit=audit,
            pack=pack,
            runbook=runbook,
            movement=movement,
            gate=gate,
            videos=videos,
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "passed" if audit.get("status") == "passed" else "failed",
        "operation": "real_so100_prestage_dashboard",
        "audit_manifest": str(audit_manifest),
        "output_html": str(output),
        "current_gate_status": pack.get("current_gate_status"),
        "recommended_action": pack.get("recommended_action"),
        "allowed_physical_action": pack.get("allowed_physical_action"),
        "audit_status": audit.get("status"),
        "audit_failed_check_count": audit.get("failed_check_count"),
        "movement_report_html": pack.get("movement_report_html"),
        "gate_report_html": pack.get("gate_report_html"),
        "runbook_markdown": runbook.get("output_markdown"),
        "motion_videos": [item["path"] for item in videos],
        "motion_video_previews": [item["preview_gif"] for item in videos if item.get("preview_gif")],
        "purpose": "single-entry human dashboard for SO-100 agentic-layer pre-stage evidence",
    }
    manifest_path = output.with_suffix(".json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _motion_videos_from_movement(movement: dict[str, Any]) -> list[dict[str, Any]]:
    videos = []
    for report_path in movement.get("reports", []):
        report = _load_json(Path(report_path))
        video = report.get("motion_video")
        if video and video.get("path"):
            videos.append(
                {
                    "path": video["path"],
                    "report": report_path,
                    "frames": video.get("frames_recorded"),
                    "actual_codec": video.get("actual_codec"),
                    "actual_frame_count": video.get("actual_frame_count"),
                    "actual_fps": video.get("actual_fps"),
                    "first_frame_readable": video.get("first_frame_readable"),
                    "browser_preview_recommended": video.get("browser_preview_recommended"),
                    "joint": report.get("joint"),
                    "observed_delta_raw": report.get("observed_delta_raw"),
                }
            )
    return videos


def _with_preview_gifs(*, videos: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(videos):
        source = Path(item["path"])
        preview = output_dir / f"motion_preview_{index}.gif"
        try:
            write_preview_gif(source=source, output=preview)
            item["preview_gif"] = str(preview)
        except Exception as exc:  # noqa: BLE001 - keep dashboard usable even if preview generation fails.
            item["preview_error"] = repr(exc)
    return videos


def _render_html(
    *,
    title: str,
    output: Path,
    audit: dict[str, Any],
    pack: dict[str, Any],
    runbook: dict[str, Any],
    movement: dict[str, Any],
    gate: dict[str, Any],
    videos: list[dict[str, Any]],
) -> str:
    links = [
        ("Movement Report", pack.get("movement_report_html")),
        ("Gate Report", pack.get("gate_report_html")),
        ("Runbook", runbook.get("output_markdown")),
        ("Pre-stage Pack", audit.get("pre_stage_pack")),
        ("Audit Manifest", audit.get("manifest_path") or audit.get("pre_stage_pack")),
    ]
    link_cards = "\n".join(_link_card(label, path, output.parent) for label, path in links if path)
    video_cards = "\n".join(_video_card(item, output.parent) for item in videos)
    lesson_items = "\n".join(
        f"<li><strong>{html.escape(str(item.get('observation')))}</strong><br>{html.escape(str(item.get('agentic_update')))}</li>"
        for item in pack.get("agentic_lessons", [])
    )
    failed_checks = [item for item in audit.get("checks", []) if item.get("status") != "passed"]
    failed_html = "<li>None</li>" if not failed_checks else "\n".join(f"<li>{html.escape(str(item.get('name')))}</li>" for item in failed_checks)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ font-size: 24px; margin: 0 0 4px; }}
    h2 {{ font-size: 18px; margin-top: 24px; }}
    .note {{ color: #4b5563; margin-bottom: 18px; }}
    .summary, .links, .videos {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: #fff; }}
    .status {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
    .passed {{ background: #ecfdf5; color: #047857; }}
    .blocked {{ background: #fff7ed; color: #9a3412; }}
    .failed {{ background: #fef2f2; color: #b91c1c; }}
    video, img.preview {{ width: 100%; max-height: 300px; object-fit: contain; background: #f3f4f6; border: 1px solid #e5e7eb; }}
    img.preview {{ display: block; margin-bottom: 8px; }}
    code {{ font-size: 12px; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class=\"note\">Single-entry dashboard for the SO-100 agentic-layer pre-stage. It preserves video-backed movement evidence and gate blockers; it is not a benchmark success claim.</p>
  <section class=\"summary\">
    <div class=\"card\"><strong>Audit</strong><br><span class=\"status {html.escape(str(audit.get('status')))}\">{html.escape(str(audit.get('status')))}</span><br>{html.escape(str(audit.get('failed_check_count')))} failed checks</div>
    <div class=\"card\"><strong>Gate</strong><br><span class=\"status {html.escape(str(pack.get('current_gate_status')))}\">{html.escape(str(pack.get('current_gate_status')))}</span><br>{html.escape(str(pack.get('recommended_action')))}</div>
    <div class=\"card\"><strong>Physical Action</strong><br><code>{html.escape(str(pack.get('allowed_physical_action')))}</code></div>
    <div class=\"card\"><strong>Videos</strong><br>{html.escape(str(movement.get('video_count')))} current, {html.escape(str(movement.get('legacy_without_video_count')))} legacy without video</div>
    <div class=\"card\"><strong>Gate Report</strong><br>{html.escape(str(gate.get('current_gate_status')))} / {html.escape(str(gate.get('recommended_action')))}</div>
  </section>
  <h2>Open Artifacts</h2>
  <section class=\"links\">{link_cards}</section>
  <h2>Motion Videos</h2>
  <section class=\"videos\">{video_cards or '<p>No motion videos are linked.</p>'}</section>
  <h2>Agentic Lessons</h2>
  <ul>{lesson_items}</ul>
  <h2>Failed Audit Checks</h2>
  <ul>{failed_html}</ul>
</body>
</html>
"""


def _link_card(label: str, path: str, output_dir: Path) -> str:
    href = html.escape(os.path.relpath(path, start=output_dir))
    return f'<div class="card"><strong>{html.escape(label)}</strong><br><a href="{href}"><code>{html.escape(path)}</code></a></div>'


def _video_card(item: dict[str, Any], output_dir: Path) -> str:
    src = html.escape(os.path.relpath(item["path"], start=output_dir))
    preview = item.get("preview_gif")
    preview_html = ""
    if preview:
        preview_src = html.escape(os.path.relpath(preview, start=output_dir))
        preview_html = f'<img src="{preview_src}" alt="motion preview" class="preview">'
    preview_error = item.get("preview_error")
    if preview_error:
        preview_html = f'<p class="failed">Preview unavailable: {html.escape(str(preview_error))}</p>'
    return f"""
    <div class=\"card\">
      {preview_html}
      <video src=\"{src}\" controls preload=\"metadata\"></video>
      <p><strong>{html.escape(str(item.get('joint')))}</strong> delta={html.escape(str(item.get('observed_delta_raw')))}, frames={html.escape(str(item.get('frames')))}</p>
      <p>codec={html.escape(str(item.get('actual_codec')))}, actual_frames={html.escape(str(item.get('actual_frame_count')))}, fps={html.escape(str(item.get('actual_fps')))}, first_frame_readable={html.escape(str(item.get('first_frame_readable')))}</p>
      <p>{html.escape(_browser_preview_summary(item))}</p>
      <p><a href=\"{src}\">Open MP4</a> <code>{html.escape(str(item.get('path')))}</code></p>
    </div>
"""


def _browser_preview_summary(item: dict[str, Any]) -> str:
    recommended = item.get("browser_preview_recommended")
    if recommended is True:
        return "GIF preview is recommended because the MP4 codec may not play directly in the browser."
    if recommended is False:
        return "MP4 is likely browser-compatible; GIF preview is still kept for quick review."
    return "Browser playback compatibility is unknown; use the GIF preview for quick review."


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-entry dashboard for SO-100 pre-stage evidence.")
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Real SO-100 Agentic Pre-stage Dashboard")
    args = parser.parse_args()
    print(
        json.dumps(
            build_prestage_dashboard(audit_manifest=args.audit_manifest, output=args.output, title=args.title),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
