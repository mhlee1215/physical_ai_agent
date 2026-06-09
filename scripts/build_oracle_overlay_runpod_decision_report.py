#!/usr/bin/env python3
"""Build a RunPod lifecycle decision report for oracle overlay validation."""

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


def _rel(report_path: Path, path: str | Path) -> str:
    raw = Path(path)
    if raw.is_absolute():
        path_obj = raw
    else:
        path_obj = Path.cwd() / raw
    try:
        return path_obj.relative_to(report_path.parent).as_posix()
    except ValueError:
        try:
            return raw.relative_to(report_path.parent).as_posix()
        except ValueError:
            return raw.as_posix()


def _sample_rows(manifest: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [row for row in manifest.get("rows", []) if isinstance(row, dict)]
    rows.sort(key=lambda row: float(row.get("distance_from_image_center_px", 0.0)), reverse=True)
    return rows[:limit]


def _html(report_path: Path, root: Path, manifest: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    cards = []
    for sample in samples:
        overlay = sample.get("overlay", "")
        zoom = sample.get("zoom", "")
        cards.append(
            f"""
        <article class="sample">
          <div class="meta">
            <strong>{html.escape(str(sample.get("case", "sample")))}</strong>
            <span>{html.escape(str(sample.get("object", "object")))}</span>
            <span>distance from center: {float(sample.get("distance_from_image_center_px", 0.0)):.1f}px</span>
            <span>projection error: {float(sample.get("projection_error_px", 999.0)):.1f}px</span>
          </div>
          <div class="pair">
            <figure><img src="{html.escape(_rel(report_path, overlay))}" alt="overlay"><figcaption>overlay</figcaption></figure>
            <figure><img src="{html.escape(_rel(report_path, zoom))}" alt="zoom"><figcaption>zoom</figcaption></figure>
          </div>
        </article>
            """
        )

    links = [
        ("Milestone dashboard", root / "milestone_dashboard.html"),
        ("Diverse object report", root / "diverse_object_projection" / "diverse_object_report.html"),
        ("Center-bias audit", root / "center_bias_audit" / "center_bias_audit.html"),
        ("Live blocker audit", root / "live_audit_blocked_probe" / "live_oracle_audit.html"),
        ("Paper figure pack", root / "paper_figure_pack" / "figure_pack.html"),
    ]
    link_html = "".join(
        f'<li><a href="{html.escape(_rel(report_path, path))}">{html.escape(label)}</a></li>'
        for label, path in links
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RunPod Lifecycle Decision for Oracle Overlay</title>
  <style>
    :root {{ --ink:#17130e; --paper:#f7efe1; --panel:#fffaf1; --line:#d3bea0; --ok:#008f5b; --warn:#b96916; --block:#b5382d; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#ffe59a,transparent 24%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(38px,5vw,74px); letter-spacing:-.055em; }}
    header p {{ max-width:980px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:26px; }}
    section, .sample {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; border:1px solid currentColor; background:#fff; }}
    .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }} .block {{ color:var(--block); }}
    h2 {{ margin:14px 0 8px; font-size:30px; letter-spacing:-.03em; }}
    ul {{ font-family:Avenir Next, Helvetica, sans-serif; line-height:1.55; }}
    a {{ color:#174f38; font-weight:800; }}
    .samples {{ display:grid; gap:20px; }}
    .meta {{ display:flex; gap:12px; flex-wrap:wrap; font-family:Avenir Next, Helvetica, sans-serif; margin-bottom:12px; }}
    .meta span, .meta strong {{ background:#f1e6d7; border-radius:999px; padding:6px 10px; }}
    .pair {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }}
    figure {{ margin:0; padding:10px; background:white; border:1px solid var(--line); border-radius:18px; }}
    img {{ width:100%; border-radius:12px; display:block; background:#222; }}
    figcaption {{ margin-top:7px; font:700 13px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>RunPod Lifecycle Decision</h1>
    <p>Decision artifact for the Oracle Point Overlay milestone. It records why overlay/probe RunPod usage can stop now, what remains blocked, and when cloud GPU time becomes useful again.</p>
  </header>
  <main>
    <section>
      <span class="pill ok">STOP OVERLAY POD</span>
      <h2>Decision</h2>
      <ul>
        <li>Overlay/probe-only RunPod time is not needed for the current local evidence milestone.</li>
        <li>Static projection, rendering, diverse object overlay, center-bias audit, and report generation are reproducible locally.</li>
        <li>Baseline/parity experiment Pods should be left untouched unless the user explicitly asks to stop them.</li>
      </ul>
    </section>
    <section>
      <span class="pill block">LIVE GATE STILL BLOCKED</span>
      <h2>When RunPod becomes useful again</h2>
      <ul>
        <li>Use RunPod only for a reused, approved Pod that can produce live ManiSkill/SAPIEN RGB frames with matching object pose and camera parameters.</li>
        <li>Final live acceptance requires at least 10 live overlay frames, projected oracle metadata, gallery HTML/contact sheet/GIF, and a passing live audit.</li>
        <li>Known blocker remains Vulkan/SAPIEN GPU-driver support in tested environments; synthetic/static evidence is not a substitute for the live gate.</li>
      </ul>
    </section>
    <section>
      <span class="pill warn">REPRESENTATIVE LOCAL EVIDENCE</span>
      <h2>Current local milestone summary</h2>
      <ul>
        <li>Diverse object episodes: {html.escape(str(manifest.get("episode_count", "n/a")))}</li>
        <li>Non-center object episodes: {html.escape(str(manifest.get("non_center_episode_count", "n/a")))}</li>
        <li>Sample policy: show the farthest-from-center projected targets first to avoid center-marker ambiguity.</li>
      </ul>
      <ul>{link_html}</ul>
    </section>
    <section>
      <span class="pill ok">SAMPLES</span>
      <h2>Representative overlay samples</h2>
      <div class="samples">{''.join(cards)}</div>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "diverse_object_projection" / "diverse_object_manifest.json"
    manifest = _load_json(manifest_path)
    samples = _sample_rows(manifest, args.limit)
    report_path = output_dir / "runpod_lifecycle_decision.html"
    report_path.write_text(_html(report_path, root, manifest, samples), encoding="utf-8")
    out = {
        "status": "passed" if len(samples) >= 10 else "missing_samples",
        "decision": "stop_overlay_probe_pods_keep_baseline_parity",
        "sample_count": len(samples),
        "html": str(report_path),
        "source_manifest": str(manifest_path),
    }
    (output_dir / "runpod_lifecycle_decision.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
