#!/usr/bin/env python3
"""Audit whether oracle overlay evidence is center-biased."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def _bar_chart(counts: dict[str, int], output_path: Path, title: str) -> None:
    width, height = 900, 360
    margin_left, margin_top, margin_bottom = 160, 50, 70
    image = Image.new("RGB", (width, height), (246, 244, 237))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill=(18, 18, 18))
    if not counts:
        draw.text((24, 80), "No data", fill=(160, 30, 20))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return
    labels = list(counts)
    max_count = max(counts.values())
    chart_width = width - margin_left - 40
    bar_h = max(18, (height - margin_top - margin_bottom) // max(1, len(labels)) - 8)
    for idx, label in enumerate(labels):
        y = margin_top + idx * (bar_h + 8)
        value = counts[label]
        bar_w = int(chart_width * value / max_count)
        draw.text((24, y + 4), label[:24], fill=(18, 18, 18))
        draw.rounded_rectangle(
            (margin_left, y, margin_left + bar_w, y + bar_h),
            radius=5,
            fill=(0, 200, 105),
            outline=(0, 120, 70),
            width=2,
        )
        draw.text((margin_left + bar_w + 8, y + 4), str(value), fill=(18, 18, 18))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _bin_distances(values: list[float]) -> dict[str, int]:
    bins = {
        "0-44px center-like": 0,
        "45-89px off-center": 0,
        "90-134px far": 0,
        "135px+ very far": 0,
    }
    for value in values:
        if value < 45:
            bins["0-44px center-like"] += 1
        elif value < 90:
            bins["45-89px off-center"] += 1
        elif value < 135:
            bins["90-134px far"] += 1
        else:
            bins["135px+ very far"] += 1
    return bins


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--non-center-threshold", type=float, default=45.0)
    parser.add_argument("--min-non-center", type=int, default=20)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    distances = [float(row.get("distance_from_image_center_px", 0.0)) for row in rows]
    errors = [float(row.get("projection_error_px", 0.0)) for row in rows]
    object_counts = Counter(str(row.get("object", "unknown")) for row in rows)
    mode_counts = Counter(str(row.get("mode", "unknown")) for row in rows)
    non_center_count = sum(1 for value in distances if value >= args.non_center_threshold)
    passed = (
        len(rows) >= 10
        and non_center_count >= args.min_non_center
        and max(errors or [999.0]) <= 1.0
        and mode_counts.get("projected_object_pose", 0) == len(rows)
    )

    distance_bins = _bin_distances(distances)
    object_chart = output_dir / "object_distribution.png"
    distance_chart = output_dir / "center_distance_distribution.png"
    _bar_chart(dict(sorted(object_counts.items())), object_chart, "Target object distribution")
    _bar_chart(distance_bins, distance_chart, "Distance from image center distribution")

    audit = {
        "status": "passed" if passed else "failed",
        "episode_count": len(rows),
        "non_center_threshold_px": args.non_center_threshold,
        "non_center_count": non_center_count,
        "min_required_non_center": args.min_non_center,
        "max_projection_error_px": max(errors or [0.0]),
        "mean_distance_from_center_px": sum(distances) / len(distances) if distances else 0.0,
        "object_counts": dict(sorted(object_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "distance_bins": distance_bins,
        "object_chart": str(object_chart),
        "distance_chart": str(distance_chart),
    }
    audit_json = output_dir / "center_bias_audit.json"
    audit_html = output_dir / "center_bias_audit.html"
    audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    row_html = "".join(
        f"<tr><td>{html.escape(str(row.get('case')))}</td><td>{html.escape(str(row.get('object')))}</td>"
        f"<td>{float(row.get('distance_from_image_center_px', 0.0)):.1f}</td>"
        f"<td>{float(row.get('projection_error_px', 0.0)):.1f}</td>"
        f"<td>{html.escape(str(row.get('mode')))}</td></tr>"
        for row in rows
    )
    audit_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Overlay Center-Bias Audit</title>
  <style>
    :root {{ --ink:#151511; --line:#d4c4aa; --pass:#00c86d; --bad:#c64a32; }}
    body {{ margin:0; color:var(--ink); background:linear-gradient(135deg,#efe0c7,#f9f5ed 48%,#e4eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:44px 52px 20px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(34px,5vw,68px); letter-spacing:-.045em; }}
    main {{ padding:30px 52px 64px; display:grid; gap:24px; }}
    section {{ background:rgba(255,250,240,.86); border:1px solid var(--line); border-radius:26px; padding:24px; }}
    .status {{ display:inline-flex; padding:8px 12px; border-radius:999px; border:1px solid currentColor; font:800 13px Avenir Next, Helvetica, sans-serif; }}
    .passed {{ color:var(--pass); }} .failed {{ color:var(--bad); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
    img {{ width:100%; border-radius:14px; border:1px solid var(--line); background:white; }}
    table {{ width:100%; border-collapse:collapse; font:13px Avenir Next, Helvetica, sans-serif; }}
    td, th {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Overlay Center-Bias Audit</h1>
    <span class="status {html.escape(audit['status'])}">{html.escape(audit['status'].upper())}</span>
  </header>
  <main>
    <section>
      <h2>Summary</h2>
      <p>Episodes: <strong>{audit['episode_count']}</strong></p>
      <p>Non-center episodes >= {audit['non_center_threshold_px']:.0f}px: <strong>{audit['non_center_count']}</strong> / required {audit['min_required_non_center']}</p>
      <p>Max projection error: <strong>{audit['max_projection_error_px']:.1f}px</strong></p>
      <p>Mean distance from center: <strong>{audit['mean_distance_from_center_px']:.1f}px</strong></p>
    </section>
    <section class="grid">
      <figure><img src="{html.escape(object_chart.name)}" alt="object distribution"></figure>
      <figure><img src="{html.escape(distance_chart.name)}" alt="distance distribution"></figure>
    </section>
    <section>
      <h2>Episode table</h2>
      <table><thead><tr><th>Case</th><th>Object</th><th>Center distance px</th><th>Error px</th><th>Mode</th></tr></thead><tbody>{row_html}</tbody></table>
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(json.dumps({"status": audit["status"], "non_center_count": non_center_count, "html": str(audit_html)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
