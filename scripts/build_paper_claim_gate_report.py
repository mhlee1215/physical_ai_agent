#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimGate:
    key: str
    title: str
    status: str
    allowed_wording: str
    forbidden_wording: str
    evidence: str
    next_evidence: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gates = _gates(root)
    passed_count = sum(1 for gate in gates if gate.status == "allowed")
    blocked_count = sum(1 for gate in gates if gate.status == "blocked")
    caution_count = len(gates) - passed_count - blocked_count
    manifest = {
        "status": "passed_with_blocked_claims",
        "source_type": "paper_claim_gate",
        "sample_count": len(gates),
        "allowed_count": passed_count,
        "caution_count": caution_count,
        "blocked_count": blocked_count,
        "true_oracle_projection_claim_allowed": _gate_status(gates, "tier_o_true_oracle") == "allowed",
        "agentic_success_claim_allowed": _gate_status(gates, "agentic_success_improvement") == "allowed",
        "gates": [asdict(gate) for gate in gates],
    }
    (output_dir / "paper_claim_gate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_path = output_dir / "paper_claim_gate_report.html"
    html_path.write_text(_html(gates, manifest), encoding="utf-8")
    print(html_path)
    return 0


def _gates(root: Path) -> list[ClaimGate]:
    related = _load_json(root / "paper_progress_ledger" / "paper_progress_ledger_manifest.json")
    rgb = _load_json(root / "actual_sim_evidence_preflight" / "actual_sim_evidence_preflight_manifest.json")
    heuristic = _load_json(root / "actual_sim_visual_heuristic_overlay" / "actual_sim_visual_heuristic_manifest.json")
    codepath = _load_json(
        root
        / "actual_rgb_synthetic_metadata_true_oracle_codepath"
        / "actual_rgb_synthetic_metadata_codepath_manifest.json"
    )
    tier_o = _load_json(root / "actual_sim_true_oracle_two_stage_result" / "actual_sim_true_oracle_two_stage_result_manifest.json")
    matrix = _load_json(root / "agentic_smolvla_experiment_matrix_result" / "agentic_smolvla_experiment_matrix_result_manifest.json")
    schema = _load_json(root / "agentic_retry_schema_readiness" / "agentic_retry_schema_readiness_manifest.json")
    blocker = _load_json(root / "actual_sim_true_oracle_probe_blocker" / "actual_sim_true_oracle_probe_blocker_manifest.json")

    actual_rgb_ok = int(rgb.get("sample_count", 0) or 0) >= 10
    heuristic_ok = heuristic.get("source_type") == "actual_sim_rgb_visual_heuristic"
    codepath_ok = codepath.get("codepath_projection_ready") is True
    tier_o_ok = tier_o.get("true_oracle_projection") is True
    matrix_ready = matrix.get("source_type") == "agentic_smolvla_experiment_matrix_result"
    schema_ready = schema.get("status") == "passed_schema_ready_nonpaper_benchmark"
    renderer_blocked = blocker.get("status") == "blocked_renderer_incompatible_driver"
    related_ok = related.get("status") == "passed"

    return [
        ClaimGate(
            "paper_direction",
            "Paper direction",
            "allowed" if related_ok else "caution",
            "Frame the paper as agentic lightweight VLA control with spatial-cue ablations.",
            "Do not frame Oracle Point Overlay as the standalone main method.",
            "Paper progress ledger and related-work scan are present.",
            "Carry this framing into Intro and Related Work.",
        ),
        ClaimGate(
            "visual_overlay_novelty",
            "Visual overlay novelty",
            "blocked",
            "Use visual cueing as an ablation/intervention.",
            "Do not claim visual prompting or affordance overlay itself is novel.",
            "VP-VLA, TraceVLA, AVP, RoboPoint, AffordVLA/AffordanceVLA, CLIPort, and VoxPoser overlap.",
            "Cite close work prominently and position our contribution as controlled wrapper evaluation.",
        ),
        ClaimGate(
            "actual_sim_rgb",
            "Actual simulation RGB evidence",
            "allowed" if actual_rgb_ok else "blocked",
            "State that actual simulator RGB frames are available and used in reports.",
            "Do not say these frames prove oracle projection.",
            f"actual_sim_evidence_preflight sample_count={rgb.get('sample_count', 'missing')}.",
            "Keep RGB evidence separate from pose/camera oracle evidence.",
        ),
        ClaimGate(
            "heuristic_overlay",
            "Actual-sim heuristic overlay",
            "allowed" if heuristic_ok else "blocked",
            "State that image-only heuristic overlays can be rendered on actual sim RGB.",
            "Do not describe heuristic overlays as true oracle or privileged pose projection.",
            f"heuristic source_type={heuristic.get('source_type', 'missing')}.",
            "Run policy rollouts before claiming behavior improvement.",
        ),
        ClaimGate(
            "synthetic_metadata_codepath",
            "Actual RGB + synthetic metadata codepath",
            "caution" if codepath_ok else "blocked",
            "Use as diagnostic proof that projection codepath works when metadata is supplied.",
            "Do not use this as Tier O evidence.",
            f"codepath_projection_ready={codepath.get('codepath_projection_ready', 'missing')}.",
            "Replace synthetic metadata with real env pose/camera in renderer-capable environment.",
        ),
        ClaimGate(
            "tier_o_true_oracle",
            "Tier O actual-sim true oracle",
            "allowed" if tier_o_ok else "blocked",
            "If passed, state that same-step actual RGB, real pose, camera metadata, and projected overlay exist.",
            "Do not claim Tier O until strict_true_oracle_step_count >= 10 and true_oracle_projection == true.",
            f"two_stage_status={tier_o.get('status', 'missing')}; true_oracle_projection={tier_o.get('true_oracle_projection', 'missing')}.",
            "Run remote evidence pack in renderer-capable Linux/GPU environment.",
        ),
        ClaimGate(
            "renderer_blocker",
            "Mac-local renderer blocker",
            "allowed" if renderer_blocked else "caution",
            "State that Mac-local actual RGB probe is blocked by SAPIEN/Vulkan driver compatibility.",
            "Do not imply SmolVLA is the blocker for the zero-action probe.",
            f"probe blocker status={blocker.get('status', 'missing')}; smolvla_ready={blocker.get('smolvla_ready', 'missing')}.",
            "Use remote renderer-capable environment for Tier O capture.",
        ),
        ClaimGate(
            "agentic_schema",
            "Agentic retry schema",
            "caution" if schema_ready else "blocked",
            "State that retry/verifier trace schema exists from CP22/CP23.",
            "Do not claim SmolVLA success improvement from SO101/MuJoCo schema evidence.",
            f"schema status={schema.get('status', 'missing')}; retry_events={schema.get('retry_events', 'missing')}.",
            "Run C3-C5 actual SmolVLA matrix conditions.",
        ),
        ClaimGate(
            "experiment_matrix",
            "C0-C5 experiment matrix",
            "caution" if matrix_ready else "blocked",
            "State that the result collection matrix is defined and wired.",
            "Do not claim matrix results until environment success flags exist for each condition.",
            f"matrix status={matrix.get('status', 'missing')}; claim_ready_count={matrix.get('claim_ready_count', 'missing')}.",
            "Populate condition manifests from actual rollouts.",
        ),
        ClaimGate(
            "agentic_success_improvement",
            "Agentic success improvement",
            "blocked",
            "At most say this is the main hypothesis to test.",
            "Do not claim agentic wrapping improves final benchmark success yet.",
            "No C3-C5 SmolVLA actual rollout success evidence exists.",
            "Run policy-only vs agentic variants under identical seeds/budgets.",
        ),
        ClaimGate(
            "success_semantics",
            "Success semantics",
            "allowed",
            "Final success must be benchmark/environment success flags only.",
            "Do not use verifier success, overlay visibility, or diagnostic projection as final task success.",
            "Survey, matrix, and claim-gate docs encode this rule.",
            "Preserve this in paper tables and captions.",
        ),
        ClaimGate(
            "remote_evidence_pack",
            "Remote evidence pack",
            "allowed",
            "State that a non-destructive remote evidence runner exists.",
            "Do not say the runner itself proves Tier O.",
            "run_remote_true_oracle_evidence_pack.sh exists and dashboard links the handoff.",
            "Execute it only in an approved renderer-capable environment.",
        ),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _gate_status(gates: list[ClaimGate], key: str) -> str:
    for gate in gates:
        if gate.key == key:
            return gate.status
    return "missing"


def _html(gates: list[ClaimGate], manifest: dict[str, Any]) -> str:
    cards = "\n".join(_card(gate, index + 1) for index, gate in enumerate(gates))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Paper Claim Gate</title>
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
    .card {{ min-height:280px; border-width:2px; }}
    .allowed {{ border-color:var(--green); }} .caution {{ border-color:var(--gold); }} .blocked {{ border-color:var(--red); }}
    .tag {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--ink); color:#fff9e8; font-size:12px; font-weight:800; }}
    h2 {{ margin:15px 0 8px; font-size:22px; letter-spacing:-.025em; }}
    p {{ color:var(--muted); line-height:1.43; overflow-wrap:anywhere; }}
    code {{ background:rgba(23,32,24,.08); padding:2px 5px; border-radius:6px; }}
    @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Paper Claim Gate</h1>
    <p class="lead">A claim-level audit that prevents the manuscript from getting ahead of evidence. It separates allowed wording, caution-only diagnostic statements, and blocked claims.</p>
  </header>
  <main>
    <section class="summary">
      <p><strong>Status:</strong> <code>{html.escape(str(manifest['status']))}</code></p>
      <p><strong>Allowed:</strong> <code>{manifest['allowed_count']}</code>; <strong>Caution:</strong> <code>{manifest['caution_count']}</code>; <strong>Blocked:</strong> <code>{manifest['blocked_count']}</code>.</p>
      <p><strong>Tier O claim allowed:</strong> <code>{manifest['true_oracle_projection_claim_allowed']}</code>. <strong>Agentic success claim allowed:</strong> <code>{manifest['agentic_success_claim_allowed']}</code>.</p>
    </section>
    <section class="grid">{cards}</section>
  </main>
</body>
</html>
"""


def _card(gate: ClaimGate, number: int) -> str:
    return f"""
      <article class="card {html.escape(gate.status)}">
        <span class="tag">{number}</span>
        <h2>{html.escape(gate.title)}</h2>
        <p><strong>Status:</strong> <code>{html.escape(gate.status)}</code></p>
        <p><strong>Allowed:</strong> {html.escape(gate.allowed_wording)}</p>
        <p><strong>Forbidden:</strong> {html.escape(gate.forbidden_wording)}</p>
        <p><strong>Evidence:</strong> {html.escape(gate.evidence)}</p>
        <p><strong>Next:</strong> {html.escape(gate.next_evidence)}</p>
      </article>
    """


if __name__ == "__main__":
    raise SystemExit(main())
