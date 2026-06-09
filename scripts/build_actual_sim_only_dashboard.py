#!/usr/bin/env python3
"""Build an actual-simulation-only evidence dashboard."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


SECTIONS = [
    {
        "key": "preflight",
        "title": "Actual sim artifact preflight",
        "manifest": "actual_sim_evidence_preflight/actual_sim_evidence_preflight_manifest.json",
        "html": "actual_sim_evidence_preflight/actual_sim_evidence_preflight.html",
        "image": "actual_sim_evidence_preflight/actual_sim_evidence_preflight_contact_sheet.png",
        "claim": "Existing CP24 artifacts contain actual simulator RGB frames; true-oracle readiness is scanned separately.",
    },
    {
        "key": "fallback",
        "title": "Actual sim RGB center fallback overlay",
        "manifest": "actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_manifest.json",
        "html": "actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_report.html",
        "image": "actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_contact_sheet.png",
        "claim": "Overlay rendering works on actual simulator RGB, but this is center fallback, not oracle projection.",
    },
    {
        "key": "visual_heuristic",
        "title": "Actual sim visual heuristic overlay",
        "manifest": "actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_manifest.json",
        "html": "actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_report.html",
        "image": "actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_contact_sheet.png",
        "claim": "Image-only visual heuristics can place non-oracle point prompts on actual simulator RGB.",
    },
    {
        "key": "policy_input",
        "title": "Actual sim heuristic policy-input readiness",
        "manifest": "actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_manifest.json",
        "html": "actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_report.html",
        "image": "actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_contact_sheet.png",
        "claim": "Actual sim heuristic overlays can be converted to SmolVLA-style image tensor previews.",
    },
    {
        "key": "temporal",
        "title": "Actual sim heuristic temporal consistency",
        "manifest": "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_manifest.json",
        "html": "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_report.html",
        "image": "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_contact_sheet.png",
        "secondary_image": "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_trail.png",
        "claim": "Actual sim heuristic points can be inspected over a real rollout frame sequence.",
    },
    {
        "key": "true_oracle",
        "title": "Actual sim true oracle projection",
        "manifest": "live_true_oracle_projection/actual_sim_true_oracle_report_manifest.json",
        "html": "live_true_oracle_projection/actual_sim_true_oracle_report.html",
        "image": "live_true_oracle_projection/actual_sim_true_oracle_contact_sheet.png",
        "claim": "Required final evidence tier: actual sim RGB, object pose, camera metadata, and overlay from the same action-input observation.",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _row(root: Path, section: dict[str, str]) -> dict[str, Any]:
    manifest_path = root / section["manifest"]
    manifest = _load_json(manifest_path)
    return {
        "key": section["key"],
        "title": section["title"],
        "claim": section["claim"],
        "manifest": str(manifest_path),
        "html": str(root / section["html"]),
        "image": str(root / section["image"]),
        "secondary_image": str(root / section["secondary_image"]) if section.get("secondary_image") else None,
        "status": manifest.get("status", "missing"),
        "source_type": manifest.get("source_type", "missing"),
        "real_sim_episode": manifest.get("real_sim_episode", "missing"),
        "true_oracle_projection": manifest.get("true_oracle_projection", manifest.get("true_oracle_projection_ready", "missing")),
        "sample_count": manifest.get("sample_count", manifest.get("strict_true_oracle_step_count", 0)),
        "manifest_exists": manifest_path.exists(),
        "html_exists": (root / section["html"]).exists(),
        "image_exists": (root / section["image"]).exists(),
    }


def _html(report_path: Path, rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        image_html = ""
        for key in ("image", "secondary_image"):
            image = row.get(key)
            if image and Path(image).exists():
                image_html += (
                    f'<figure><img src="{html.escape(_rel(report_path, Path(image)))}" '
                    f'alt="{html.escape(row["title"])}"><figcaption>{html.escape(Path(image).name)}</figcaption></figure>'
                )
        status_class = "pass" if row["status"] in {"passed", "passed_fallback_only", "passed_rgb_only_true_oracle_blocked"} else "block"
        cards.append(
            f"""
      <section class="{status_class}">
        <span class="status">{html.escape(str(row['status']).upper())}</span>
        <h2>{html.escape(row['title'])}</h2>
        <p>{html.escape(row['claim'])}</p>
        <ul>
          <li>source_type: <code>{html.escape(str(row['source_type']))}</code></li>
          <li>real_sim_episode: <code>{html.escape(str(row['real_sim_episode']))}</code></li>
          <li>true_oracle_projection: <code>{html.escape(str(row['true_oracle_projection']))}</code></li>
          <li>sample_count: <code>{html.escape(str(row['sample_count']))}</code></li>
        </ul>
        <p><a href="{html.escape(_rel(report_path, Path(row['html'])))}">Open section report</a></p>
        <div class="images">{image_html}</div>
      </section>
            """
        )
    true_ready = any(row["key"] == "true_oracle" and row["status"] == "passed" for row in rows)
    sim_sections = sum(1 for row in rows if row["real_sim_episode"] is True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual-Sim-Only Agentic Overlay Dashboard</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --green:#008f5b; --block:#b5382d; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#e0f7ff,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1080px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    section {{ background:rgba(255,250,241,.9); border:2px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    section.pass {{ border-color:var(--green); }} section.block {{ border-color:var(--block); }}
    .status {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; border:1px solid currentColor; background:#fff; }}
    .pass .status {{ color:var(--green); }} .block .status {{ color:var(--block); }}
    h2 {{ margin:12px 0 8px; font-size:28px; letter-spacing:-.03em; }}
    ul {{ font-family:Avenir Next, Helvetica, sans-serif; line-height:1.55; }}
    a {{ color:#174f38; font-weight:800; }}
    code {{ overflow-wrap:anywhere; }}
    .images {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 12px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual-Sim-Only Agentic Overlay Dashboard</h1>
    <p>This page excludes synthetic diagnostics. It only summarizes actual CP24 simulation RGB evidence and separates image-only heuristic progress from the still-required true-oracle projection gate. Actual-sim sections: {sim_sections}. True-oracle ready: {true_ready}.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""


def build_dashboard(root: Path, output_dir: Path) -> dict[str, Any]:
    rows = [_row(root, section) for section in SECTIONS]
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "actual_sim_only_dashboard.html"
    html_path.write_text(_html(html_path, rows), encoding="utf-8")
    ready = any(row["key"] == "true_oracle" and row["status"] == "passed" for row in rows)
    report = {
        "status": "passed_true_oracle_ready" if ready else "passed_actual_sim_true_oracle_blocked",
        "source_type": "actual_sim_only_dashboard",
        "synthetic_included": False,
        "true_oracle_ready": ready,
        "sample_count": sum(int(row["sample_count"] or 0) for row in rows if row["key"] != "true_oracle"),
        "html": str(html_path),
        "rows": rows,
    }
    (output_dir / "actual_sim_only_dashboard_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_dashboard(Path(args.root), Path(args.output_dir))
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
