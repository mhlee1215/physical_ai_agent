#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from scripts.video_preview import write_preview_gif


def build_movement_report(
    *,
    reports: list[Path],
    output: Path,
    title: str = "Real SO-100 Movement Evidence Report",
) -> dict[str, Any]:
    rows = [_load_movement_report(path) for path in reports]
    output.parent.mkdir(parents=True, exist_ok=True)
    _attach_preview_gifs(rows=rows, output_dir=output.parent)
    output.write_text(_render_html(rows=rows, output=output, title=title), encoding="utf-8")
    manifest = {
        "status": "passed",
        "operation": "real_so100_movement_report",
        "output_html": str(output),
        "report_count": len(rows),
        "reports": [row["report_path"] for row in rows],
        "video_count": sum(1 for row in rows if row.get("motion_video_exists")),
        "video_preview_count": sum(1 for row in rows if row.get("motion_video_preview_exists")),
        "video_previews": [row["motion_video_preview"] for row in rows if row.get("motion_video_preview_exists")],
        "legacy_without_video_count": sum(1 for row in rows if not row.get("motion_video_exists")),
        "purpose": "qualitative evidence for agentic-layer improvement, not benchmark success",
    }
    manifest_path = output.with_suffix(".json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_movement_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    visual = payload.get("visual_check", {})
    before = visual.get("before", {})
    after = visual.get("after", {})
    video = payload.get("motion_video", {})
    video_path = Path(video["path"]) if video.get("path") else None
    return {
        "report_path": str(path),
        "status": payload.get("status"),
        "timestamp": payload.get("timestamp"),
        "joint": payload.get("joint"),
        "manual_delta_raw": payload.get("manual_delta_raw"),
        "observed_delta_raw": payload.get("observed_delta_raw"),
        "target_error_raw": payload.get("target_error_raw"),
        "send_action_called": payload.get("send_action_called"),
        "contact_probe_allowed": payload.get("contact_probe_allowed"),
        "mean_absdiff": after.get("mean_absdiff"),
        "visual_motion_detected": after.get("visual_motion_detected"),
        "before_image": before.get("image_path"),
        "after_image": after.get("image_path"),
        "diff_heatmap": _sibling_if_exists(after.get("image_path"), "diff_heatmap.jpg"),
        "motion_video": str(video_path) if video_path else None,
        "motion_video_exists": bool(video_path and video_path.exists()),
        "motion_video_frames": video.get("frames_recorded"),
        "motion_video_actual_codec": video.get("actual_codec"),
        "motion_video_actual_frame_count": video.get("actual_frame_count"),
        "motion_video_actual_fps": video.get("actual_fps"),
        "motion_video_first_frame_readable": video.get("first_frame_readable"),
        "motion_video_browser_preview_recommended": video.get("browser_preview_recommended"),
    }


def _sibling_if_exists(image_path: str | None, name: str) -> str | None:
    if not image_path:
        return None
    candidate = Path(image_path).parent / name
    return str(candidate) if candidate.exists() else None


def _attach_preview_gifs(*, rows: list[dict[str, Any]], output_dir: Path) -> None:
    for index, row in enumerate(rows):
        video = row.get("motion_video")
        if not video or not row.get("motion_video_exists"):
            continue
        preview = output_dir / f"motion_report_preview_{index}.gif"
        try:
            write_preview_gif(source=Path(video), output=preview)
            row["motion_video_preview"] = str(preview)
            row["motion_video_preview_exists"] = True
        except Exception as exc:  # noqa: BLE001 - report should still link the original MP4.
            row["motion_video_preview_exists"] = False
            row["motion_video_preview_error"] = repr(exc)


def _render_html(*, rows: list[dict[str, Any]], output: Path, title: str) -> str:
    body = "\n".join(_render_row(row, output.parent) for row in rows)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    .note {{ color: #4b5563; margin-bottom: 24px; }}
    .item {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; font-size: 13px; }}
    .media {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 12px; }}
    figure {{ margin: 0; }}
    img, video {{ width: 100%; max-height: 360px; object-fit: contain; background: #f3f4f6; border: 1px solid #e5e7eb; }}
    figcaption {{ font-size: 12px; color: #4b5563; margin-top: 4px; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; background: #eef2ff; color: #3730a3; font-size: 12px; }}
    .missing {{ color: #9a3412; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class=\"note\">Qualitative movement evidence for the real SO-100 agentic-layer pre-stage. These clips and frames support debugging and retry design; they are not benchmark success claims.</p>
  {body}
</body>
</html>
"""


def _render_row(row: dict[str, Any], output_dir: Path) -> str:
    media = []
    for label, key in [("Before", "before_image"), ("After", "after_image"), ("Diff heatmap", "diff_heatmap")]:
        if row.get(key):
            media.append(_figure(label, row[key], output_dir, video=False))
    if row.get("motion_video_exists"):
        if row.get("motion_video_preview_exists"):
            media.append(_figure("Motion preview", row["motion_video_preview"], output_dir, video=False))
        elif row.get("motion_video_preview_error"):
            media.append(f'<p class="missing">Motion preview unavailable: {html.escape(str(row["motion_video_preview_error"]))}</p>')
        media.append(_figure("Motion video", row["motion_video"], output_dir, video=True))
    else:
        media.append('<p class="missing">Motion video missing: legacy evidence recorded before --record-video was required.</p>')
    media_html = "\n".join(media)
    return f"""
  <section class=\"item\">
    <h2>{html.escape(str(row.get("joint")))} <span class=\"badge\">{html.escape(str(row.get("status")))}</span></h2>
    <div class=\"meta\">
      <div><strong>Report</strong><br><code>{html.escape(row["report_path"])}</code></div>
      <div><strong>Timestamp</strong><br>{html.escape(str(row.get("timestamp")))}</div>
      <div><strong>Manual delta raw</strong><br>{html.escape(str(row.get("manual_delta_raw")))}</div>
      <div><strong>Observed delta raw</strong><br>{html.escape(str(row.get("observed_delta_raw")))}</div>
      <div><strong>Mean absdiff</strong><br>{html.escape(str(row.get("mean_absdiff")))}</div>
      <div><strong>Visual motion</strong><br>{html.escape(str(row.get("visual_motion_detected")))}</div>
      <div><strong>Contact probe</strong><br>{html.escape(str(row.get("contact_probe_allowed")))}</div>
      <div><strong>Video codec</strong><br>{html.escape(str(row.get("motion_video_actual_codec")))}</div>
      <div><strong>Video frames</strong><br>{html.escape(_video_frame_summary(row))}</div>
      <div><strong>Browser preview</strong><br>{html.escape(_browser_preview_summary(row))}</div>
    </div>
    <div class=\"media\">{media_html}</div>
  </section>
"""


def _video_frame_summary(row: dict[str, Any]) -> str:
    recorded = row.get("motion_video_frames")
    actual = row.get("motion_video_actual_frame_count")
    fps = row.get("motion_video_actual_fps")
    readable = row.get("motion_video_first_frame_readable")
    return f"recorded={recorded}, actual={actual}, fps={fps}, first_frame_readable={readable}"


def _browser_preview_summary(row: dict[str, Any]) -> str:
    if not row.get("motion_video_exists"):
        return "pre-video evidence; no motion.mp4 was recorded for this legacy row"
    recommended = row.get("motion_video_browser_preview_recommended")
    if recommended is True:
        return "GIF preview recommended; MP4 may not play directly in browser"
    if recommended is False:
        return "MP4 likely browser-compatible"
    return "unknown"


def _figure(label: str, path: str, output_dir: Path, *, video: bool) -> str:
    src = html.escape(os.path.relpath(path, start=output_dir))
    safe_label = html.escape(label)
    if video:
        media = f'<video src="{src}" controls preload="metadata"></video>'
    else:
        media = f'<img src="{src}" alt="{safe_label}">'
    return f"<figure>{media}<figcaption>{safe_label}: <code>{html.escape(path)}</code></figcaption></figure>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report for real SO-100 movement evidence.")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Real SO-100 Movement Evidence Report")
    args = parser.parse_args()
    print(
        json.dumps(
            build_movement_report(reports=args.report, output=args.output, title=args.title),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
