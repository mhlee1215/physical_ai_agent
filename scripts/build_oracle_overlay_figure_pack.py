#!/usr/bin/env python3
"""Build the paper-facing oracle overlay figure pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIGURE_GROUPS = [
    {
        "claim": "Oracle projection targets diverse non-center objects rather than simply marking the image center.",
        "evidence": "diverse_object_projection/diverse_object_full_contact_sheet.png",
        "status": "pass",
        "title": "Primary evidence: diverse non-center object projection",
        "body": (
            "Oracle projection targets diverse non-center objects rather than simply marking "
            "the image center. This is the preferred representative figure for paper/slide review."
        ),
    },
    {
        "claim": "Zoomed crops confirm the marker lies on each intended target object across object categories and distractors.",
        "evidence": "diverse_object_projection/diverse_object_zoom_contact_sheet.png",
        "status": "pass",
        "title": "Primary evidence: zoomed target-object crops",
        "body": "Zoomed crops confirm the marker lies on each intended target object across object categories and distractors.",
    },
    {
        "claim": "Center-bias audit verifies that the diverse-object evidence is not just image-center marking.",
        "evidence": "center_bias_audit/center_distance_distribution.png",
        "status": "pass",
        "title": "Center-bias audit",
        "body": "Distance-from-center and object-distribution audits quantify non-center targets and balanced object categories.",
    },
    {
        "claim": "Oracle projection follows synthetic object locations across diverse static layouts.",
        "evidence": "local_html_validation/contact_sheet_projection.png",
        "status": "pass",
        "title": "Static oracle projection",
        "body": "Oracle projection follows synthetic object locations across diverse static layouts.",
    },
    {
        "claim": "Oracle projection follows a moving object trajectory with <=1 px synthetic projection error.",
        "evidence": "local_trajectory_validation/contact_sheet_projection_trajectory.png",
        "status": "pass",
        "title": "Moving-object trajectory",
        "body": "Oracle projection follows a moving object trajectory with <=1 px synthetic projection error.",
    },
    {
        "claim": "Common simulator pose dictionaries are parsed correctly.",
        "evidence": "local_pose_dict_validation/contact_sheet_dict_pose_trajectory.png",
        "status": "pass",
        "title": "Simulator pose dictionary robustness",
        "body": "Common simulator pose dictionaries such as {p: xyz} and {position: xyz} are parsed correctly.",
    },
    {
        "claim": "Camera parameters embedded under sensor_data[cam] are supported.",
        "evidence": "local_sensor_camera_validation/contact_sheet_sensor_data_camera_trajectory.png",
        "status": "pass",
        "title": "Camera parameters under sensor_data",
        "body": "Camera intrinsics/extrinsics embedded under sensor_data[cam] are supported.",
    },
    {
        "claim": "Preferred and fallback camera selection works in multi-camera observations.",
        "evidence": "local_multi_camera_validation/contact_sheet_multi_camera_trajectory.png",
        "status": "pass",
        "title": "Multi-camera preferred/fallback selection",
        "body": "Preferred base camera and fallback auxiliary camera selection both work.",
    },
    {
        "claim": "Projection failures fall back to a safe center overlay instead of crashing.",
        "evidence": "local_edge_case_validation/contact_sheet_projection_edge_cases.png",
        "status": "pass",
        "title": "Projection edge-case fallback safety",
        "body": "Invalid/out-of-frame projections fall back to center overlay rather than crashing.",
    },
    {
        "claim": "The overlay visibly changes the image input seen by SmolVLA-style policies.",
        "evidence": "raw_vs_overlay_comparison/comparison_contact_sheet.png",
        "status": "pass",
        "title": "Raw input vs oracle-overlay input",
        "body": "The overlay visibly changes the image input seen by SmolVLA-style policies.",
    },
    {
        "claim": "Saved sim frames can be rendered with overlays, but without true pose metadata this remains fallback evidence.",
        "evidence": "local_html_validation/contact_sheet_sim_frame_overlay.png",
        "status": "limited",
        "title": "Saved sim-frame overlay rendering",
        "body": (
            "Saved sim frames can be rendered with overlays. Limitation: saved PNGs do not contain live pose/camera "
            "metadata, so this is fallback rendering evidence rather than true live oracle projection."
        ),
    },
]


def _html(root: Path, groups: list[dict[str, str]], blocker_rel: str) -> str:
    sections = []
    for group in groups:
        evidence = group["evidence"]
        sections.append(
            f"""
    <section class="{group['status']}"><span class="status">{group['status'].upper()}</span><h2>{group['title']}</h2><p>{group['body']}</p><img src="../{evidence}" alt="{group['title']}"></section>"""
        )
    sections.append(
        f"""
    <section class="blocked"><span class="status">BLOCKED</span><h2>Live ManiSkill true oracle projection</h2><p>Claim not yet supported. The tested RunPod hosts fail before live frame capture due to Vulkan/SAPIEN renderer errors.</p><div class="note">Blocker artifact: <a href="{blocker_rel}">live_oracle_probe_20260606T2008Z/maniskill_blocker.md</a></div></section>"""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Point Overlay Figure Pack</title>
  <style>
    :root {{ --ink:#151511; --paper:#f7f0e4; --panel:#fffaf0; --line:#d4c4aa; --pass:#00c86d; --limited:#b1842f; --block:#c64a32; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 12% 8%,#fff1bd,transparent 28%),linear-gradient(135deg,#efe0c7,#f9f5ed 46%,#e4eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 22px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(38px,5vw,76px); letter-spacing:-.05em; }}
    header p {{ max-width:920px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:26px; }}
    section {{ background:rgba(255,250,240,.88); border:1px solid var(--line); border-radius:28px; padding:24px; box-shadow:0 18px 48px rgba(60,42,18,.08); }}
    .status {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; border:1px solid currentColor; background:white; }}
    .pass .status {{ color:var(--pass); }} .limited .status {{ color:var(--limited); }} .blocked .status {{ color:var(--block); }}
    h2 {{ margin:14px 0 8px; font-size:30px; letter-spacing:-.03em; }}
    p {{ font-size:16px; line-height:1.45; }}
    img {{ width:100%; border-radius:16px; border:1px solid var(--line); background:white; }}
    .note {{ padding:14px 16px; border-radius:16px; background:#f1eadf; font-family:Avenir Next, Helvetica, sans-serif; }}
    a {{ color:#174f38; font-weight:800; }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Point Overlay Figure Pack</h1>
    <p>Paper-facing visual evidence for the Agentic SmolVLA oracle overlay path. Primary figures emphasize diverse non-center target objects to avoid center-marker ambiguity. Live ManiSkill true projection remains explicitly blocked by renderer/Vulkan availability.</p>
  </header>
  <main>{''.join(sections)}
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--live-blocker-rel", default="../../live_oracle_probe_20260606T2008Z/maniskill_blocker.md")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    for group in FIGURE_GROUPS:
        item = {key: group[key] for key in ("claim", "evidence", "status")}
        item["exists"] = (root / group["evidence"]).exists()
        groups.append(item)
    manifest = {
        "status": "passed_static_blocked_live" if all(group["exists"] for group in groups) else "missing_evidence",
        "title": "Oracle Point Overlay Figure Pack",
        "sample_groups": groups,
    }
    (output_dir / "figure_pack_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "figure_pack.html").write_text(_html(root, FIGURE_GROUPS, args.live_blocker_rel), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figure_groups": len(groups), "html": str(output_dir / "figure_pack.html")}, indent=2))
    return 0 if manifest["status"] == "passed_static_blocked_live" else 1


if __name__ == "__main__":
    raise SystemExit(main())
