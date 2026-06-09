#!/usr/bin/env python3
"""Build a paper-claim matrix from actual-simulation evidence only."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


CLAIMS = [
    {
        "key": "actual_rgb_available",
        "claim": "Actual CP24 simulation RGB rollout frames are available for inspection.",
        "evidence": "actual_sim_evidence_preflight/actual_sim_evidence_preflight_manifest.json",
        "required_statuses": {"passed_rgb_only_true_oracle_blocked", "passed_true_oracle_ready"},
        "paper_status": "allowed",
    },
    {
        "key": "fallback_overlay_rendering",
        "claim": "The overlay renderer can draw point prompts on actual simulation RGB frames.",
        "evidence": "actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_manifest.json",
        "required_statuses": {"passed_fallback_only"},
        "paper_status": "allowed_with_limitation",
    },
    {
        "key": "visual_heuristic_prompt",
        "claim": "An image-only lightweight affordance heuristic can place non-oracle point prompts on actual simulation RGB.",
        "evidence": "actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_manifest.json",
        "required_statuses": {"passed"},
        "paper_status": "allowed_with_limitation",
    },
    {
        "key": "policy_input_bridge",
        "claim": "Actual-sim heuristic overlays can be converted into SmolVLA-style image input tensors.",
        "evidence": "actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_manifest.json",
        "required_statuses": {"passed"},
        "paper_status": "allowed_with_limitation",
    },
    {
        "key": "temporal_inspection",
        "claim": "Actual-sim heuristic point prompts can be inspected over a rollout frame sequence.",
        "evidence": "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_manifest.json",
        "required_statuses": {"passed"},
        "paper_status": "allowed_with_limitation",
    },
    {
        "key": "true_oracle_projection",
        "claim": "Oracle point projection works on actual simulation rollouts using same-step object pose and camera metadata.",
        "evidence": "live_true_oracle_projection/actual_sim_true_oracle_report_manifest.json",
        "required_statuses": {"passed"},
        "paper_status": "blocked_until_passed",
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


def _evaluate(root: Path) -> list[dict[str, Any]]:
    rows = []
    for claim in CLAIMS:
        manifest_path = root / claim["evidence"]
        manifest = _load_json(manifest_path)
        status = manifest.get("status", "missing")
        passed = status in claim["required_statuses"]
        rows.append(
            {
                "key": claim["key"],
                "claim": claim["claim"],
                "paper_status": claim["paper_status"] if passed else "not_supported",
                "supported": passed,
                "manifest_status": status,
                "source_type": manifest.get("source_type", "missing"),
                "real_sim_episode": manifest.get("real_sim_episode", "missing"),
                "true_oracle_projection": manifest.get(
                    "true_oracle_projection",
                    manifest.get("true_oracle_projection_ready", "missing"),
                ),
                "sample_count": manifest.get("sample_count", manifest.get("strict_true_oracle_step_count", 0)),
                "evidence": str(manifest_path),
            }
        )
    return rows


def _html(report_path: Path, rows: list[dict[str, Any]], actual_sim_dashboard: Path) -> str:
    table = []
    for row in rows:
        css = "ok" if row["supported"] else "bad"
        table.append(
            f"""
        <tr class="{css}">
          <td><code>{html.escape(row['key'])}</code></td>
          <td>{html.escape(row['claim'])}</td>
          <td>{html.escape(str(row['paper_status']))}</td>
          <td>{html.escape(str(row['manifest_status']))}</td>
          <td>{html.escape(str(row['source_type']))}</td>
          <td>{html.escape(str(row['sample_count']))}</td>
          <td><a href="{html.escape(_rel(report_path, Path(row['evidence'])))}">manifest</a></td>
        </tr>
            """
        )
    supported = sum(1 for row in rows if row["supported"])
    blocked = [row for row in rows if row["key"] == "true_oracle_projection" and not row["supported"]]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Actual-Sim Claim Matrix</title>
  <style>
    :root {{ --ink:#17130e; --line:#d3bea0; --paper:#f7efe1; --green:#008f5b; --bad:#b5382d; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 18% 0%,#f4ffd2,transparent 25%),linear-gradient(140deg,#efe0c7,#faf6ed 48%,#e6eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:48px 56px 24px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:1080px; font-size:19px; line-height:1.45; }}
    main {{ padding:32px 56px 70px; display:grid; gap:24px; }}
    section {{ background:rgba(255,250,241,.9); border:1px solid var(--line); border-radius:26px; padding:22px; box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .pill {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; color:{'var(--bad)' if blocked else 'var(--green)'}; border:1px solid currentColor; background:#fff; }}
    table {{ width:100%; border-collapse:collapse; font-family:Avenir Next, Helvetica, sans-serif; font-size:14px; }}
    th,td {{ padding:10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    tr.ok td:first-child {{ color:var(--green); font-weight:800; }}
    tr.bad td:first-child {{ color:var(--bad); font-weight:800; }}
    a {{ color:#174f38; font-weight:800; }}
    code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>Actual-Sim Claim Matrix</h1>
    <p>This page uses actual-simulation evidence only. It marks which statements are supported now and which must remain blocked until true-oracle projection evidence exists.</p>
  </header>
  <main>
    <section>
      <span class="pill">{'TRUE ORACLE CLAIM BLOCKED' if blocked else 'ALL CLAIMS SUPPORTED'}</span>
      <h2>Summary</h2>
      <p>Supported claims: {supported}/{len(rows)}. <a href="{html.escape(_rel(report_path, actual_sim_dashboard))}">Open actual-sim-only dashboard</a>.</p>
    </section>
    <section>
      <h2>Claim matrix</h2>
      <table>
        <thead><tr><th>Key</th><th>Claim</th><th>Paper status</th><th>Manifest status</th><th>Source type</th><th>Samples</th><th>Evidence</th></tr></thead>
        <tbody>{''.join(table)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def build_report(root: Path, output_dir: Path) -> dict[str, Any]:
    rows = _evaluate(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "actual_sim_claim_matrix.html"
    actual_sim_dashboard = root / "actual_sim_only_dashboard" / "actual_sim_only_dashboard.html"
    html_path.write_text(_html(html_path, rows, actual_sim_dashboard), encoding="utf-8")
    true_oracle_supported = any(row["key"] == "true_oracle_projection" and row["supported"] for row in rows)
    report = {
        "status": "passed_true_oracle_ready" if true_oracle_supported else "passed_claims_with_true_oracle_blocked",
        "source_type": "actual_sim_claim_matrix",
        "synthetic_included": False,
        "sample_count": sum(int(row["sample_count"] or 0) for row in rows if row["key"] != "true_oracle_projection"),
        "supported_claim_count": sum(1 for row in rows if row["supported"]),
        "total_claim_count": len(rows),
        "true_oracle_claim_supported": true_oracle_supported,
        "html": str(html_path),
        "rows": rows,
    }
    (output_dir / "actual_sim_claim_matrix_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_report(Path(args.root), Path(args.output_dir))
    print(json.dumps({"status": report["status"], "supported": report["supported_claim_count"], "html": report["html"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
