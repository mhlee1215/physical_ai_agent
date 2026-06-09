#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = checkpoint_dir / "checkpoint_report.json"
    blocker_path = checkpoint_dir / "maniskill_blocker.md"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    blocker = blocker_path.read_text(encoding="utf-8") if blocker_path.exists() else "missing blocker report"
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    env_blockers = metrics.get("env_blockers", {}) if isinstance(metrics.get("env_blockers"), dict) else {}
    manifest = {
        "status": "blocked_renderer_incompatible_driver",
        "source_type": "actual_sim_true_oracle_probe_blocker",
        "real_sim_episode": False,
        "true_oracle_projection": False,
        "sample_count": 12,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_status": report.get("status", "missing"),
        "rollout_status": metrics.get("rollout_status", "missing"),
        "smolvla_ready": metrics.get("smolvla_ready", "missing"),
        "attempted_env_ids": metrics.get("attempted_env_ids", []),
        "env_blockers": env_blockers,
        "blocker_path": str(blocker_path),
    }
    (output_dir / "actual_sim_true_oracle_probe_blocker_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "actual_sim_true_oracle_probe_blocker.html"
    html_path.write_text(_html(manifest, blocker), encoding="utf-8")
    print(html_path)
    return 0


def _html(manifest: dict[str, object], blocker: str) -> str:
    cards = [
        ("1", "Probe path added", "Zero-action affordance_oracle_probe path exists and avoids SmolVLA model loading."),
        ("2", "Mac-local run attempted", "The probe was executed against CP24 with --real-images and require-maniskill."),
        ("3", "Renderer blocked", "Both attempted environments failed before rollout records were produced."),
        ("4", "Driver error", "The observed failure is vk::createInstanceUnique: ErrorIncompatibleDriver."),
        ("5", "SmolVLA not blocker", f"smolvla_ready={manifest.get('smolvla_ready')}; model loading was not used by the probe."),
        ("6", "No RGB samples", "rollout_episodes=0 and rollout_steps=0, so no actual RGB frames were captured."),
        ("7", "No Tier O manifest", "affordance_oracle_probe_true_oracle_steps.json was not produced."),
        ("8", "Cloud/local renderer needed", "Next execution needs a renderer-capable environment, likely Linux GPU with working Vulkan/SAPIEN."),
        ("9", "No new RunPod created", "This report documents the blocker; it does not create or stop Pods."),
        ("10", "Next after pass", "If the cheap probe passes, run SmolVLA affordance oracle path to test policy-input integration."),
        ("11", "Claim boundary", "This is blocker evidence, not policy success or true-oracle projection evidence."),
        ("12", "Actionable command", "The CP24 true-oracle probe command is retired; use maintained benchmark evaluation entrypoints for new runs."),
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
    blockers = "\n".join(
        f"<li><code>{html.escape(str(env))}</code>: {html.escape(str(error))}</li>"
        for env, error in (manifest.get("env_blockers") or {}).items()  # type: ignore[union-attr]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Actual-Sim True-Oracle Probe Blocker</title>
  <style>
    :root {{ --ink:#172018; --panel:#fffaf0; --line:#d1bd91; --red:#b13f31; --green:#2f7d59; --gold:#ad7c1d; --muted:#65705f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Avenir Next", "Trebuchet MS", sans-serif; color:var(--ink); background:radial-gradient(circle at 10% 8%, rgba(177,63,49,.18), transparent 30%), radial-gradient(circle at 88% 8%, rgba(47,125,89,.14), transparent 28%), linear-gradient(135deg,#fcf5e4,#ead7ad); }}
    header {{ padding:44px clamp(18px,5vw,72px) 24px; }}
    h1 {{ max-width:1120px; margin:0; font-size:clamp(38px,6vw,78px); line-height:.94; letter-spacing:-.06em; }}
    .lead {{ max-width:960px; color:var(--muted); font-size:clamp(17px,2vw,22px); line-height:1.45; margin-top:18px; }}
    main {{ padding:0 clamp(18px,5vw,72px) 70px; }}
    .summary {{ max-width:1180px; display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:20px; }}
    .box,.card,.blocker {{ background:rgba(255,250,240,.88); border:1px solid var(--line); border-radius:24px; padding:18px; box-shadow:0 18px 52px rgba(48,35,10,.11); }}
    .box b {{ display:block; font-size:30px; margin-bottom:8px; }}
    .box span,.card p,.blocker {{ color:var(--muted); line-height:1.43; }}
    .grid {{ max-width:1180px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .card {{ min-height:190px; }}
    .tag {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--ink); color:#fff9e8; font-size:12px; font-weight:800; }}
    h2 {{ margin:15px 0 8px; font-size:23px; letter-spacing:-.025em; }}
    code {{ background:rgba(23,32,24,.08); padding:2px 5px; border-radius:6px; }}
    .bad {{ color:var(--red); }} .ok {{ color:var(--green); }} .warn {{ color:var(--gold); }}
    .blocker {{ max-width:1180px; margin:20px 0; white-space:pre-wrap; }}
    @media (max-width:920px) {{ .summary,.grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Actual-Sim True-Oracle Probe Blocker</h1>
    <p class="lead">The cheap zero-action probe path was executed locally. It confirms the next blocker is renderer/driver compatibility before any true-oracle samples can be captured on this Mac.</p>
  </header>
  <main>
    <section class="summary">
      <div class="box"><b class="bad">{html.escape(str(manifest.get('checkpoint_status')))}</b><span>checkpoint status</span></div>
      <div class="box"><b class="bad">{html.escape(str(manifest.get('rollout_status')))}</b><span>rollout status</span></div>
      <div class="box"><b class="ok">{html.escape(str(manifest.get('smolvla_ready')))}</b><span>SmolVLA readiness reported by CP24</span></div>
      <div class="box"><b class="bad">0</b><span>captured true-oracle samples</span></div>
    </section>
    <section class="blocker">
      <h2>Environment blockers</h2>
      <ul>{blockers}</ul>
      <h2>Raw blocker note</h2>
      <pre>{html.escape(blocker)}</pre>
    </section>
    <section class="grid">{card_html}</section>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
