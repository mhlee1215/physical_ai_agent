#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONDITIONS = [
    {
        "key": "C0_policy_only",
        "title": "C0 Policy-only SmolVLA",
        "purpose": "Baseline success/failure behavior under fixed task IDs, seeds, and action budget.",
        "required_before_claim": "Environment success flags from policy-only rollouts.",
    },
    {
        "key": "C1_overlay_only_heuristic",
        "title": "C1 Overlay-only heuristic",
        "purpose": "Test whether cheap non-privileged image-space cueing changes behavior without retry/replan.",
        "required_before_claim": "Actual-sim heuristic overlay policy rollouts with environment success flags.",
    },
    {
        "key": "C2_overlay_only_true_oracle",
        "title": "C2 Overlay-only true oracle",
        "purpose": "Upper-bound value of perfect spatial cueing without agentic retry.",
        "required_before_claim": "Tier O true-oracle policy-input manifest plus environment success flags.",
    },
    {
        "key": "C3_agentic_only",
        "title": "C3 Agentic verifier/retry only",
        "purpose": "Isolate recovery benefit without visual input modification.",
        "required_before_claim": "Verifier/retry traces and final environment success flags.",
    },
    {
        "key": "C4_agentic_heuristic_overlay",
        "title": "C4 Agentic + heuristic overlay",
        "purpose": "Most deployable variant: non-privileged cueing plus verifier-driven recovery.",
        "required_before_claim": "Agentic traces, heuristic overlay inputs, latency, and environment success flags.",
    },
    {
        "key": "C5_agentic_true_oracle_overlay",
        "title": "C5 Agentic + true-oracle overlay",
        "purpose": "Upper-bound combined effect of perfect spatial cueing and recovery.",
        "required_before_claim": "Tier O manifest, agentic traces, and final environment success flags.",
    },
]


@dataclass(frozen=True)
class MatrixRow:
    key: str
    title: str
    status: str
    purpose: str
    required_before_claim: str
    environment_success_rate: str
    episode_count: str
    mean_retries: str
    latency_overhead_ms: str
    manifest_path: str | None
    claim_allowed: bool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_row(root, condition) for condition in CONDITIONS]
    claim_ready_count = sum(1 for row in rows if row.claim_allowed)
    report = {
        "status": "passed_schema_pending_results",
        "source_type": "agentic_smolvla_experiment_matrix_result",
        "sample_count": 12,
        "condition_count": len(rows),
        "claim_ready_count": claim_ready_count,
        "true_oracle_projection_required": True,
        "final_success_source": "environment_success_flags_only",
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "agentic_smolvla_experiment_matrix_result_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "agentic_smolvla_experiment_matrix_result.html"
    html_path.write_text(_html(rows, report), encoding="utf-8")
    print(html_path)
    return 0


def _row(root: Path, condition: dict[str, str]) -> MatrixRow:
    manifest_candidates = {
        "C0_policy_only": root / "actual_sim_evidence_preflight" / "actual_sim_evidence_preflight_manifest.json",
        "C1_overlay_only_heuristic": root / "actual_sim_visual_heuristic_overlay" / "actual_sim_visual_heuristic_manifest.json",
        "C2_overlay_only_true_oracle": root / "actual_sim_true_oracle_two_stage_result" / "actual_sim_true_oracle_two_stage_result_manifest.json",
        "C3_agentic_only": root / "agentic_retry_schema_readiness" / "agentic_retry_schema_readiness_manifest.json",
        "C4_agentic_heuristic_overlay": root / "agentic_retry_schema_readiness" / "agentic_retry_schema_readiness_manifest.json",
        "C5_agentic_true_oracle_overlay": root / "agentic_retry_schema_readiness" / "agentic_retry_schema_readiness_manifest.json",
    }
    manifest_path = manifest_candidates[condition["key"]]
    manifest = _load_json(manifest_path)
    status = _condition_status(condition["key"], manifest)
    claim_allowed = status == "claim_ready"
    return MatrixRow(
        key=condition["key"],
        title=condition["title"],
        status=status,
        purpose=condition["purpose"],
        required_before_claim=condition["required_before_claim"],
        environment_success_rate=_metric(manifest, "success_rate"),
        episode_count=_metric(manifest, "episode_count", fallback_keys=("episodes", "sample_count")),
        mean_retries=_metric(manifest, "mean_retries"),
        latency_overhead_ms=_metric(manifest, "latency_overhead_ms"),
        manifest_path=str(manifest_path) if manifest_path.exists() else None,
        claim_allowed=claim_allowed,
    )


def _condition_status(key: str, manifest: dict[str, Any]) -> str:
    if not manifest:
        return "pending_no_manifest"
    if key == "C0_policy_only":
        return "evidence_partial_needs_success_flags"
    if key == "C1_overlay_only_heuristic":
        return "evidence_partial_needs_policy_rollout"
    if key == "C2_overlay_only_true_oracle":
        return "claim_ready" if manifest.get("true_oracle_projection") is True else str(manifest.get("status", "blocked"))
    if key in {"C3_agentic_only", "C4_agentic_heuristic_overlay", "C5_agentic_true_oracle_overlay"}:
        if manifest.get("final_success_source") == "environment_success_flags":
            return "claim_ready"
        if manifest.get("status") == "passed_schema_ready_nonpaper_benchmark":
            return "schema_ready_needs_actual_smolvla_rollout"
        return "pending_agentic_rollout"
    return "pending"


def _metric(manifest: dict[str, Any], key: str, fallback_keys: tuple[str, ...] = ()) -> str:
    for candidate in (key, *fallback_keys):
        if candidate in manifest:
            return str(manifest[candidate])
    return "pending"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _html(rows: list[MatrixRow], report: dict[str, Any]) -> str:
    cards = "\n".join(_card(row, index + 1) for index, row in enumerate(rows))
    guardrails = [
        ("7", "Same seeds", "Every condition must share task IDs, seeds, checkpoint, and action budget."),
        ("8", "Env success only", "Final success comes from simulator/environment success flags only."),
        ("9", "Verifier separate", "Internal verifier success controls retry; it is not task success."),
        ("10", "Tier separation", "Synthetic, actual heuristic, and true-oracle evidence must not be mixed."),
        ("11", "Overhead required", "Latency and memory overhead must be reported because this is a lightweight VLA paper."),
        ("12", "No overlay novelty", "Visual cueing is an ablation/intervention, not the main novelty claim."),
    ]
    guardrail_html = "\n".join(
        f"""
        <article class="card guardrail">
          <span class="tag">{html.escape(number)}</span>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(text)}</p>
        </article>
        """
        for number, title, text in guardrails
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agentic SmolVLA Experiment Matrix Result</title>
  <style>
    :root {{ --ink:#172018; --panel:#fffaf0; --line:#d1bd91; --green:#2f7d59; --red:#b13f31; --gold:#ad7c1d; --blue:#326b8a; --muted:#65705f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Avenir Next", "Trebuchet MS", sans-serif; color:var(--ink); background:radial-gradient(circle at 10% 8%, rgba(50,107,138,.16), transparent 30%), radial-gradient(circle at 88% 8%, rgba(47,125,89,.16), transparent 28%), linear-gradient(135deg,#fcf5e4,#ead7ad); }}
    header {{ padding:44px clamp(18px,5vw,72px) 24px; }}
    h1 {{ max-width:1120px; margin:0; font-size:clamp(38px,6vw,78px); line-height:.94; letter-spacing:-.06em; }}
    .lead {{ max-width:980px; color:var(--muted); font-size:clamp(17px,2vw,22px); line-height:1.45; margin-top:18px; }}
    main {{ padding:0 clamp(18px,5vw,72px) 70px; }}
    .summary,.card {{ background:rgba(255,250,240,.88); border:1px solid var(--line); border-radius:24px; padding:18px; box-shadow:0 18px 52px rgba(48,35,10,.11); }}
    .summary {{ max-width:1180px; margin-bottom:20px; }}
    .summary b {{ font-size:28px; }}
    .grid {{ max-width:1180px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .card {{ min-height:245px; }}
    .guardrail {{ min-height:190px; }}
    .tag {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--ink); color:#fff9e8; font-size:12px; font-weight:800; }}
    h2 {{ margin:15px 0 8px; font-size:22px; letter-spacing:-.025em; }}
    p {{ color:var(--muted); line-height:1.43; overflow-wrap:anywhere; }}
    code {{ background:rgba(23,32,24,.08); padding:2px 5px; border-radius:6px; }}
    .claim_ready {{ color:var(--green); }} .pending,.blocked,.evidence_partial {{ color:var(--gold); }} .pending_no_manifest {{ color:var(--red); }}
    @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Agentic SmolVLA Experiment Matrix Result</h1>
    <p class="lead">A result collection scaffold for the paper's controlled matrix. Current status is schema-ready, with most behavioral claims pending actual rollout evidence.</p>
  </header>
  <main>
    <section class="summary">
      <b>Status: <code>{html.escape(str(report['status']))}</code></b>
      <p>Conditions: <code>{report['condition_count']}</code>. Claim-ready conditions: <code>{report['claim_ready_count']}</code>. Final success source: <code>{html.escape(str(report['final_success_source']))}</code>.</p>
    </section>
    <section class="grid">{cards}{guardrail_html}</section>
  </main>
</body>
</html>
"""


def _card(row: MatrixRow, number: int) -> str:
    status_class = "claim_ready" if row.claim_allowed else row.status.split("_")[0]
    return f"""
      <article class="card">
        <span class="tag">{number}</span>
        <h2>{html.escape(row.title)}</h2>
        <p><strong class="{html.escape(status_class)}">status: {html.escape(row.status)}</strong></p>
        <p>{html.escape(row.purpose)}</p>
        <p>success_rate=<code>{html.escape(row.environment_success_rate)}</code>; episodes=<code>{html.escape(row.episode_count)}</code>; retries=<code>{html.escape(row.mean_retries)}</code>; latency_ms=<code>{html.escape(row.latency_overhead_ms)}</code></p>
        <p>Needed: {html.escape(row.required_before_claim)}</p>
      </article>
    """


if __name__ == "__main__":
    raise SystemExit(main())
