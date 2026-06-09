#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint22", default="_workspace/checkpoints/checkpoint_22/checkpoint_report.json")
    parser.add_argument("--checkpoint23", default="_workspace/checkpoints/checkpoint_23/checkpoint_report.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cp22 = _load_json(Path(args.checkpoint22))
    cp23 = _load_json(Path(args.checkpoint23))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cp22_metrics = cp22.get("metrics", {}) if isinstance(cp22.get("metrics"), dict) else {}
    cp23_metrics = cp23.get("metrics", {}) if isinstance(cp23.get("metrics"), dict) else {}
    manifest = {
        "status": "passed_schema_ready_nonpaper_benchmark",
        "source_type": "agentic_retry_schema_readiness",
        "sample_count": 12,
        "final_success_source": "internal_so101_mujoco_not_paper_matrix",
        "true_oracle_projection": False,
        "checkpoint22_status": cp22.get("status", "missing"),
        "checkpoint23_status": cp23.get("status", "missing"),
        "retry_events": cp23_metrics.get("agentic_retry_events", cp22_metrics.get("retry_events", "missing")),
        "policy_only_success": cp23_metrics.get("policy_only_success", "missing"),
        "agentic_retry_success": cp23_metrics.get("agentic_retry_success", cp22_metrics.get("success", "missing")),
        "agentic_passed_subgoals": cp23_metrics.get("agentic_passed_subgoals", cp22_metrics.get("passed_subgoals", "missing")),
        "agentic_total_subgoals": cp23_metrics.get("agentic_total_subgoals", cp22_metrics.get("total_subgoals", "missing")),
        "claim_boundary": (
            "Validates agentic trace/retry schema only. It is not SmolVLA, not actual-sim Tier O, "
            "and not a paper-facing success improvement claim."
        ),
    }
    (output_dir / "agentic_retry_schema_readiness_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "agentic_retry_schema_readiness.html"
    html_path.write_text(_html(manifest, cp22, cp23), encoding="utf-8")
    print(html_path)
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _html(manifest: dict[str, Any], cp22: dict[str, Any], cp23: dict[str, Any]) -> str:
    cards = [
        ("1", "CP22 trace exists", f"checkpoint22_status={manifest.get('checkpoint22_status')}"),
        ("2", "CP23 comparison exists", f"checkpoint23_status={manifest.get('checkpoint23_status')}"),
        ("3", "Retry events recorded", f"retry_events={manifest.get('retry_events')}"),
        ("4", "Policy-only success tracked", f"policy_only_success={manifest.get('policy_only_success')}"),
        ("5", "Agentic success tracked", f"agentic_retry_success={manifest.get('agentic_retry_success')}"),
        ("6", "Subgoal progress tracked", f"passed={manifest.get('agentic_passed_subgoals')}/{manifest.get('agentic_total_subgoals')}"),
        ("7", "Verifier trace schema", "CP22 checks require verifier decisions in trace."),
        ("8", "Retry trace schema", "CP22 checks require retry=true trace events."),
        ("9", "Comparison metrics", "CP23 stores policy-only vs agentic metrics."),
        ("10", "Not SmolVLA evidence", "This is SO101/MuJoCo schema readiness only."),
        ("11", "Not final success claim", "Agentic retry success is false in the current CP23 artifact."),
        ("12", "Next mapping", "Use this schema for C3/C4/C5 actual SmolVLA rollouts."),
    ]
    card_html = "\n".join(
        f"""
        <article class="card">
          <span class="tag">{html.escape(number)}</span>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(text)}</p>
        </article>
        """
        for number, title, text in cards
    )
    cp22_artifacts = cp22.get("artifacts", {}) if isinstance(cp22.get("artifacts"), dict) else {}
    cp23_artifacts = cp23.get("artifacts", {}) if isinstance(cp23.get("artifacts"), dict) else {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agentic Retry Schema Readiness</title>
  <style>
    :root {{ --ink:#172018; --panel:#fffaf0; --line:#d1bd91; --green:#2f7d59; --red:#b13f31; --gold:#ad7c1d; --muted:#65705f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Avenir Next", "Trebuchet MS", sans-serif; color:var(--ink); background:radial-gradient(circle at 10% 8%, rgba(47,125,89,.18), transparent 30%), radial-gradient(circle at 88% 8%, rgba(177,63,49,.15), transparent 28%), linear-gradient(135deg,#fcf5e4,#ead7ad); }}
    header {{ padding:44px clamp(18px,5vw,72px) 24px; }}
    h1 {{ max-width:1120px; margin:0; font-size:clamp(38px,6vw,78px); line-height:.94; letter-spacing:-.06em; }}
    .lead {{ max-width:980px; color:var(--muted); font-size:clamp(17px,2vw,22px); line-height:1.45; margin-top:18px; }}
    main {{ padding:0 clamp(18px,5vw,72px) 70px; }}
    .summary,.card {{ background:rgba(255,250,240,.88); border:1px solid var(--line); border-radius:24px; padding:18px; box-shadow:0 18px 52px rgba(48,35,10,.11); }}
    .summary {{ max-width:1180px; margin-bottom:20px; }}
    .grid {{ max-width:1180px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .card {{ min-height:190px; }}
    .tag {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--ink); color:#fff9e8; font-size:12px; font-weight:800; }}
    h2 {{ margin:15px 0 8px; font-size:23px; letter-spacing:-.025em; }}
    p {{ color:var(--muted); line-height:1.43; overflow-wrap:anywhere; }}
    code {{ background:rgba(23,32,24,.08); padding:2px 5px; border-radius:6px; }}
    @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Agentic Retry Schema Readiness</h1>
    <p class="lead">CP22/CP23 already produce agentic plans, verifier decisions, retry traces, and policy-vs-agentic comparison metrics. This validates schema readiness only, not paper-facing SmolVLA improvement.</p>
  </header>
  <main>
    <section class="summary">
      <p><strong>Status:</strong> <code>{html.escape(str(manifest['status']))}</code></p>
      <p><strong>Claim boundary:</strong> {html.escape(str(manifest['claim_boundary']))}</p>
      <p><strong>CP22 artifacts:</strong> {html.escape(str(cp22_artifacts))}</p>
      <p><strong>CP23 artifacts:</strong> {html.escape(str(cp23_artifacts))}</p>
    </section>
    <section class="grid">{card_html}</section>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
