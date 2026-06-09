#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


def build_gate_report(
    *,
    gate_manifest: Path,
    output: Path,
    title: str = "Real SO-100 Gate Evidence Report",
) -> dict[str, Any]:
    manifest = _load_json(gate_manifest)
    pregrasp = _load_json(Path(manifest["pregrasp_probe"]))
    jaw = _load_json(Path(manifest["jaw_readiness"]))
    next_action = _load_json(Path(manifest["next_action_gate"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    jaw_camera = str(manifest.get("wrist_camera_index") or next_action.get("evidence", {}).get("jaw_camera") or "0")
    overlays = _write_overlays(pregrasp=pregrasp, jaw=jaw, jaw_camera=jaw_camera, output_dir=output.parent)
    output.write_text(
        _render_html(
            title=title,
            output=output,
            manifest=manifest,
            pregrasp=pregrasp,
            jaw=jaw,
            jaw_camera=jaw_camera,
            next_action=next_action,
            overlays=overlays,
        ),
        encoding="utf-8",
    )
    report_manifest = {
        "status": "passed",
        "operation": "real_so100_gate_report",
        "gate_manifest": str(gate_manifest),
        "output_html": str(output),
        "current_gate_status": manifest.get("status"),
        "recommended_action": manifest.get("recommended_action"),
        "allowed_physical_action": manifest.get("allowed_physical_action"),
        "blockers": manifest.get("blockers", []),
        "pregrasp_status": manifest.get("pregrasp_status"),
        "jaw_status": manifest.get("jaw_status"),
        "overlays": overlays,
        "purpose": "visual gate evidence for the SO-100 agentic-layer pre-stage",
    }
    manifest_path = output.with_suffix(".json")
    report_manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(report_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return report_manifest


def _write_overlays(*, pregrasp: dict[str, Any], jaw: dict[str, Any], jaw_camera: str, output_dir: Path) -> dict[str, str]:
    overlays: dict[str, str] = {}
    for assessment in pregrasp.get("assessments", []):
        camera = str(assessment.get("camera"))
        image_path = assessment.get("image_path")
        if not image_path:
            continue
        output = output_dir / f"camera_{camera}_pregrasp_overlay.jpg"
        labels = []
        bbox = assessment.get("bbox_xyxy")
        if bbox:
            labels.append(
                {
                    "bbox_xyxy": bbox,
                    "label": f"camera {camera}: {'usable' if assessment.get('usable_for_pregrasp') else 'blocked'}",
                    "color_bgr": [0, 190, 55] if assessment.get("usable_for_pregrasp") else [0, 128, 255],
                }
            )
        if camera == jaw_camera:
            object_candidate = jaw.get("object_candidate")
            jaw_marker = jaw.get("jaw_marker_candidate")
            if object_candidate:
                labels.append(
                    {
                        "bbox_xyxy": object_candidate["bbox_xyxy"],
                        "label": f"camera {jaw_camera} object",
                        "color_bgr": [0, 255, 0],
                    }
                )
            if jaw_marker:
                labels.append(
                    {
                        "bbox_xyxy": jaw_marker["bbox_xyxy"],
                        "label": f"camera {jaw_camera} jaw marker",
                        "color_bgr": [255, 80, 0],
                    }
                )
        _draw_overlay(image_path=Path(image_path), output=output, labels=labels)
        overlays[camera] = str(output)
    return overlays


def _draw_overlay(*, image_path: Path, output: Path, labels: list[dict[str, Any]]) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    for item in labels:
        x1, y1, x2, y2 = [int(value) for value in item["bbox_xyxy"]]
        color = tuple(int(value) for value in item["color_bgr"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=3)
        cv2.putText(
            image,
            str(item["label"]),
            (max(0, x1), max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def _render_html(
    *,
    title: str,
    output: Path,
    manifest: dict[str, Any],
    pregrasp: dict[str, Any],
    jaw: dict[str, Any],
    jaw_camera: str,
    next_action: dict[str, Any],
    overlays: dict[str, str],
) -> str:
    camera_sections = "\n".join(_render_camera_section(item, overlays, output.parent) for item in pregrasp.get("assessments", []))
    blocker_text = ", ".join(str(item) for item in manifest.get("blockers", [])) or "none"
    jaw_blockers = ", ".join(str(item) for item in jaw.get("blockers", [])) or "none"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin: 18px 0 8px; }}
    .note {{ color: #4b5563; margin-bottom: 18px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px; }}
    .cell {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; }}
    .camera-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 16px; }}
    .camera {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
    img {{ width: 100%; max-height: 460px; object-fit: contain; background: #f3f4f6; border: 1px solid #e5e7eb; }}
    code {{ font-size: 12px; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; background: #eef2ff; color: #3730a3; font-size: 12px; }}
    .blocked {{ background: #fff7ed; color: #9a3412; }}
    .ready {{ background: #ecfdf5; color: #047857; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class=\"note\">Visual evidence for the real SO-100 agentic-layer pre-stage. This report explains why the next physical action is allowed or blocked; it is not a benchmark success claim.</p>
  <section class=\"summary\">
    <div class=\"cell\"><strong>Gate</strong><br><span class=\"badge {html.escape(str(manifest.get('status')))}\">{html.escape(str(manifest.get('status')))}</span></div>
    <div class=\"cell\"><strong>Recommended action</strong><br>{html.escape(str(manifest.get('recommended_action')))}</div>
    <div class=\"cell\"><strong>Allowed physical action</strong><br>{html.escape(str(manifest.get('allowed_physical_action')))}</div>
    <div class=\"cell\"><strong>Blockers</strong><br>{html.escape(blocker_text)}</div>
    <div class=\"cell\"><strong>Pregrasp</strong><br>{html.escape(str(pregrasp.get('status')))}</div>
    <div class=\"cell\"><strong>Camera {html.escape(jaw_camera)} jaw</strong><br>{html.escape(str(jaw.get('status')))}: {html.escape(jaw_blockers)}</div>
    <div class=\"cell\"><strong>Next action gate</strong><br><code>{html.escape(str(manifest.get('next_action_gate')))}</code></div>
  </section>
  <div class=\"camera-grid\">
    {camera_sections}
  </div>
  <h2>Agentic Notes</h2>
  <ul>
    {"".join(f"<li>{html.escape(str(note))}</li>" for note in next_action.get("notes", []))}
  </ul>
</body>
</html>
"""


def _render_camera_section(item: dict[str, Any], overlays: dict[str, str], output_dir: Path) -> str:
    camera = str(item.get("camera"))
    overlay = overlays.get(camera) or item.get("image_path")
    src = html.escape(os.path.relpath(str(overlay), start=output_dir))
    notes = ", ".join(str(note) for note in item.get("notes", [])) or "none"
    return f"""
    <section class=\"camera\">
      <h2>Camera {html.escape(camera)} <span class=\"badge {'ready' if item.get('usable_for_pregrasp') else 'blocked'}\">{'usable' if item.get('usable_for_pregrasp') else 'blocked'}</span></h2>
      <img src=\"{src}\" alt=\"camera {html.escape(camera)} gate overlay\">
      <p><strong>Object visible:</strong> {html.escape(str(item.get('object_visible')))}</p>
      <p><strong>Edge clipped:</strong> {html.escape(str(item.get('edge_clipped')))}</p>
      <p><strong>BBox:</strong> <code>{html.escape(str(item.get('bbox_xyxy')))}</code></p>
      <p><strong>Notes:</strong> {html.escape(notes)}</p>
    </section>
"""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual gate report for real SO-100 CP26 evidence.")
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Real SO-100 Gate Evidence Report")
    args = parser.parse_args()
    print(
        json.dumps(
            build_gate_report(gate_manifest=args.gate_manifest, output=args.output, title=args.title),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
