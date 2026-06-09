#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(summary_path)
    probe_manifest = _load_json(Path(summary.get("probe_manifest", ""))) if summary.get("probe_manifest") else {}
    policy_manifest = _load_json(Path(summary.get("policy_manifest", ""))) if summary.get("policy_manifest") else {}

    probe_count = int(summary.get("probe_strict_true_oracle_step_count", 0) or 0)
    policy_count = int(summary.get("policy_strict_true_oracle_step_count", 0) or 0)
    min_steps = int(summary.get("min_strict_steps", 10) or 10)
    policy_stage_ran = bool(summary.get("policy_stage_ran", False))
    passed = (
        summary.get("status") == "passed"
        and probe_count >= min_steps
        and policy_count >= min_steps
        and policy_manifest.get("status") == "passed"
    )
    blocked_at_probe = summary.get("status") == "blocked_at_probe"
    blocked_at_policy = summary.get("status") == "blocked_at_policy"

    report = {
        "status": "passed" if passed else str(summary.get("status", "missing")),
        "source_type": "actual_sim_true_oracle_two_stage_result",
        "true_oracle_projection": bool(passed),
        "sample_count": max(probe_count, policy_count, min_steps),
        "probe_strict_true_oracle_step_count": probe_count,
        "policy_strict_true_oracle_step_count": policy_count,
        "min_strict_steps": min_steps,
        "policy_stage_ran": policy_stage_ran,
        "blocked_at_probe": blocked_at_probe,
        "blocked_at_policy": blocked_at_policy,
        "summary_path": str(summary_path),
        "probe_manifest": summary.get("probe_manifest"),
        "policy_manifest": summary.get("policy_manifest"),
    }
    (output_dir / "actual_sim_true_oracle_two_stage_result_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "actual_sim_true_oracle_two_stage_result.html"
    html_path.write_text(_html(summary, report, probe_manifest, policy_manifest), encoding="utf-8")
    print(html_path)
    return 0 if passed or blocked_at_probe or blocked_at_policy else 1


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _html(
    summary: dict[str, Any],
    report: dict[str, Any],
    probe_manifest: dict[str, Any],
    policy_manifest: dict[str, Any],
) -> str:
    cards = [
        ("1", "Summary status", str(summary.get("status", "missing"))),
        ("2", "Probe strict count", f"{report['probe_strict_true_oracle_step_count']} / {report['min_strict_steps']}"),
        ("3", "Policy strict count", f"{report['policy_strict_true_oracle_step_count']} / {report['min_strict_steps']}"),
        ("4", "Policy stage ran", str(report["policy_stage_ran"])),
        ("5", "Probe manifest", str(summary.get("probe_manifest", "missing"))),
        ("6", "Policy manifest", str(summary.get("policy_manifest", "missing"))),
        ("7", "Probe manifest status", str(probe_manifest.get("status", "missing"))),
        ("8", "Policy manifest status", str(policy_manifest.get("status", "missing"))),
        ("9", "True-oracle projection claim", str(report["true_oracle_projection"])),
        ("10", "Blocked at probe", str(report["blocked_at_probe"])),
        ("11", "Blocked at policy", str(report["blocked_at_policy"])),
        ("12", "Next action", _next_action(report)),
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
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Actual-Sim True-Oracle Two-Stage Result</title>
  <style>
    :root {{ --ink:#172018; --panel:#fffaf0; --line:#d1bd91; --green:#2f7d59; --red:#b13f31; --gold:#ad7c1d; --muted:#65705f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Avenir Next", "Trebuchet MS", sans-serif; color:var(--ink); background:radial-gradient(circle at 10% 8%, rgba(47,125,89,.18), transparent 30%), radial-gradient(circle at 88% 8%, rgba(177,63,49,.15), transparent 28%), linear-gradient(135deg,#fcf5e4,#ead7ad); }}
    header {{ padding:44px clamp(18px,5vw,72px) 24px; }}
    h1 {{ max-width:1120px; margin:0; font-size:clamp(38px,6vw,78px); line-height:.94; letter-spacing:-.06em; }}
    .lead {{ max-width:960px; color:var(--muted); font-size:clamp(17px,2vw,22px); line-height:1.45; margin-top:18px; }}
    main {{ padding:0 clamp(18px,5vw,72px) 70px; }}
    .summary,.card {{ background:rgba(255,250,240,.88); border:1px solid var(--line); border-radius:24px; padding:18px; box-shadow:0 18px 52px rgba(48,35,10,.11); }}
    .summary {{ max-width:1180px; margin-bottom:20px; }}
    .summary b {{ font-size:28px; }}
    .grid {{ max-width:1180px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .card {{ min-height:190px; }}
    .tag {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--ink); color:#fff9e8; font-size:12px; font-weight:800; }}
    h2 {{ margin:15px 0 8px; font-size:23px; letter-spacing:-.025em; }}
    p {{ color:var(--muted); line-height:1.43; overflow-wrap:anywhere; }}
    code {{ background:rgba(23,32,24,.08); padding:2px 5px; border-radius:6px; }}
    .ok {{ color:var(--green); }} .bad {{ color:var(--red); }} .warn {{ color:var(--gold); }}
    @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Actual-Sim True-Oracle Two-Stage Result</h1>
    <p class="lead">Import report for renderer-capable execution. It separates probe-stage pose/camera capture from SmolVLA policy-input execution.</p>
  </header>
  <main>
    <section class="summary">
      <b>Status: <code>{html.escape(str(report['status']))}</code></b>
      <p>Probe strict samples: <code>{report['probe_strict_true_oracle_step_count']}</code>. Policy strict samples: <code>{report['policy_strict_true_oracle_step_count']}</code>. True-oracle projection claim: <code>{report['true_oracle_projection']}</code>.</p>
    </section>
    <section class="grid">{card_html}</section>
  </main>
</body>
</html>
"""


def _next_action(report: dict[str, Any]) -> str:
    if report["true_oracle_projection"]:
        return "Import into paper experiment matrix and run policy-only vs agentic ablations."
    if report["blocked_at_probe"]:
        return "Fix renderer/env pose-camera capture before loading SmolVLA."
    if report["blocked_at_policy"]:
        return "Probe passed; debug SmolVLA model loading or policy-input integration."
    return "Run the two-stage script in a renderer-capable environment."


if __name__ == "__main__":
    raise SystemExit(main())
