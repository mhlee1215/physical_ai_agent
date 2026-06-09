#!/usr/bin/env python3
"""Audit a live oracle overlay checkpoint output directory."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in suffixes]


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _find_manifest(root: Path) -> Path | None:
    candidates = [
        root / "affordance_oracle_probe_manifest.json",
        root / "smolvla_affordance_oracle_manifest.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted(root.glob("*affordance*manifest.json"))
    return found[0] if found else None


def _find_gallery_manifest(root: Path) -> Path | None:
    candidates = sorted(root.glob("*gallery/overlay_gallery_manifest.json"))
    return candidates[0] if candidates else None


def _oracle_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    first = manifest.get("oracle_affordance")
    if isinstance(first, dict) and first:
        records.append(first)
    for key in ("records", "steps", "policy_records"):
        value = manifest.get(key)
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    affordance = row.get("oracle_affordance")
                    if isinstance(affordance, dict):
                        merged = dict(affordance)
                        for source_key, target_key in (
                            ("episode", "source.episode"),
                            ("step", "source.step"),
                            ("raw_frame_path", "source.frame_path"),
                            ("overlay_frame_path", "source.overlay_path"),
                        ):
                            if source_key in row and target_key not in merged:
                                parent, child = target_key.split(".", 1)
                                source = merged.setdefault(parent, {})
                                if isinstance(source, dict):
                                    source[child] = row[source_key]
                        records.append(merged)
    return records


def _has_any(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value: Any = record
        found = True
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                found = False
                break
        if found and value not in (None, "", [], {}):
            return True
    return False


def _has_xy(record: dict[str, Any]) -> bool:
    for key in ("point_xy", "projected_xy", "pixel_xy", "uv", "overlay.point_xy"):
        value: Any = record
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                float(value[0])
                float(value[1])
                return True
            except (TypeError, ValueError):
                pass
    return False


def _valid_live_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for record in records:
        if record.get("mode") != "projected_object_pose":
            continue
        has_pose = _has_any(
            record,
            (
                "object_pose",
                "object_pose_xyz",
                "object_xyz",
                "world_xyz",
                "target_xyz",
                "raw_pose",
                "pose",
                "source.object_pose",
                "source.pose",
            ),
        )
        has_camera = _has_any(
            record,
            (
                "camera",
                "camera_metadata_keys",
                "camera_key",
                "camera_name",
                "camera_id",
                "camera_params",
                "intrinsic",
                "extrinsic",
                "sensor_data.camera",
                "source.camera",
            ),
        )
        has_frame = _has_any(
            record,
            (
                "image_path",
                "frame_path",
                "rgb_path",
                "overlay_path",
                "source.overlay_path",
                "input_image",
                "source.image_path",
                "source.frame_path",
            ),
        )
        has_timestep = _has_any(
            record,
            (
                "step",
                "timestep",
                "frame_index",
                "frame_id",
                "episode_step",
                "source.step",
                "source.timestep",
            ),
        )
        if _has_xy(record) and has_pose and has_camera and has_frame and has_timestep:
            valid.append(record)
    return valid


def audit_live_output(root: Path, output_dir: Path, min_frames: int) -> dict[str, Any]:
    blocker = root / "maniskill_blocker.md"
    checkpoint_report = _load_json(root / "checkpoint_report.json")
    manifest_path = _find_manifest(root)
    manifest = _load_json(manifest_path) if manifest_path else {}
    gallery_manifest_path = _find_gallery_manifest(root)
    gallery_manifest = _load_json(gallery_manifest_path) if gallery_manifest_path else {}

    frame_dirs = [
        root / "affordance_oracle_probe_frames",
        root / "smolvla_affordance_oracle_frames",
    ]
    frames = []
    for directory in frame_dirs:
        frames.extend(_images(directory))
    projected_records = [
        record for record in _oracle_records(manifest) if record.get("mode") == "projected_object_pose"
    ]
    valid_live_records = _valid_live_records(projected_records)
    checks = {
        "no_blocker": not blocker.exists(),
        "checkpoint_not_failed": checkpoint_report.get("status") != "failed",
        "manifest_exists": manifest_path is not None,
        "overlay_frames_minimum": len(frames) >= min_frames,
        "gallery_passed": gallery_manifest.get("status") == "passed",
        "gallery_frames_minimum": int(gallery_manifest.get("frame_count", 0)) >= min_frames,
        "projected_oracle_metadata_present": bool(projected_records),
        "projected_oracle_records_minimum": len(projected_records) >= min_frames,
        "strict_live_records_minimum": len(valid_live_records) >= min_frames,
    }
    status = "passed" if all(checks.values()) else "blocked" if blocker.exists() else "failed"
    audit = {
        "status": status,
        "root": str(root),
        "checks": checks,
        "frame_count": len(frames),
        "projected_oracle_metadata_count": len(projected_records),
        "strict_live_record_count": len(valid_live_records),
        "strict_live_record_requirements": [
            "mode == projected_object_pose",
            "projected pixel point present",
            "object pose/world position present",
            "camera identifier or camera parameters present",
            "RGB/frame path reference present",
            "step/timestep/frame index present",
        ],
        "checkpoint_report": str(root / "checkpoint_report.json") if (root / "checkpoint_report.json").exists() else None,
        "blocker": str(blocker) if blocker.exists() else None,
        "manifest": str(manifest_path) if manifest_path else None,
        "gallery_manifest": str(gallery_manifest_path) if gallery_manifest_path else None,
    }
    _write_html(output_dir / "live_oracle_audit.html", audit, frames[: min(20, len(frames))])
    (output_dir / "live_oracle_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return audit


def _write_html(report_path: Path, audit: dict[str, Any], frames: list[Path]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    check_items = "".join(
        f"<li class=\"{'ok' if passed else 'bad'}\">{html.escape(name)}: {passed}</li>"
        for name, passed in audit["checks"].items()
    )
    frame_cards = "".join(
        f"""
        <figure>
          <img src="{html.escape(_rel(report_path, path))}" alt="{html.escape(path.name)}">
          <figcaption>{html.escape(path.name)}</figcaption>
        </figure>
        """
        for path in frames
    )
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Oracle Overlay Audit</title>
  <style>
    :root {{ --ink:#151511; --line:#d4c4aa; --pass:#00c86d; --bad:#c64a32; }}
    body {{ margin:0; color:var(--ink); background:linear-gradient(135deg,#efe0c7,#f9f5ed 48%,#e4eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:44px 52px 20px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(34px,5vw,68px); letter-spacing:-.045em; }}
    main {{ padding:30px 52px 64px; display:grid; gap:24px; }}
    section {{ background:rgba(255,250,240,.86); border:1px solid var(--line); border-radius:26px; padding:24px; }}
    .status {{ display:inline-flex; padding:8px 12px; border-radius:999px; border:1px solid currentColor; font:800 13px Avenir Next, Helvetica, sans-serif; }}
    .passed {{ color:var(--pass); }} .blocked,.failed {{ color:var(--bad); }}
    li {{ font-family:Avenir Next, Helvetica, sans-serif; line-height:1.55; }}
    li.ok {{ color:#147b4d; }} li.bad {{ color:#a73524; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; }}
    figure {{ margin:0; padding:10px; border-radius:16px; background:white; border:1px solid var(--line); }}
    img {{ width:100%; border-radius:10px; background:#2b2b2b; }}
    figcaption {{ margin-top:8px; font:13px Avenir Next, Helvetica, sans-serif; }}
    code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>Live Oracle Overlay Audit</h1>
    <span class="status {html.escape(str(audit['status']))}">{html.escape(str(audit['status']).upper())}</span>
  </header>
  <main>
    <section>
      <h2>Checks</h2>
      <ul>{check_items}</ul>
      <p>Frame count: <strong>{audit['frame_count']}</strong></p>
      <p>Projected metadata count: <strong>{audit['projected_oracle_metadata_count']}</strong></p>
      <p>Strict live record count: <strong>{audit['strict_live_record_count']}</strong></p>
      <p>Blocker: <code>{html.escape(str(audit.get('blocker')))}</code></p>
    </section>
    <section>
      <h2>Sample frames</h2>
      <div class="grid">{frame_cards}</div>
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-frames", type=int, default=10)
    args = parser.parse_args()
    audit = audit_live_output(Path(args.root), Path(args.output_dir), args.min_frames)
    print(json.dumps({"status": audit["status"], "frame_count": audit["frame_count"]}, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
