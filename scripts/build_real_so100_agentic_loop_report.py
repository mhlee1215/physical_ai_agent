#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


def build_agentic_loop_report(
    *,
    contract: Path,
    next_plan: Path,
    reframe_advice: Path,
    output: Path,
    title: str = "Real SO-100 Agentic Loop Report",
    prompt_iteration: Path | None = None,
    policy_patch: Path | None = None,
    ready_path_fixture: Path | None = None,
) -> dict[str, Any]:
    contract_payload = _load_json(contract)
    plan = _load_json(next_plan)
    advice = _load_json(reframe_advice)
    prompt_iteration_payload = _load_json(prompt_iteration) if prompt_iteration else {}
    policy_patch_payload = _load_json(policy_patch) if policy_patch else {}
    ready_path_fixture_payload = _load_json(ready_path_fixture) if ready_path_fixture else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_html(
            title=title,
            output=output,
            contract=contract_payload,
            plan=plan,
            advice=advice,
            prompt_iteration=prompt_iteration_payload,
            policy_patch=policy_patch_payload,
            ready_path_fixture=ready_path_fixture_payload,
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "passed",
        "operation": "real_so100_agentic_loop_report",
        "output_html": str(output),
        "contract": str(contract),
        "next_plan": str(next_plan),
        "reframe_advice": str(reframe_advice),
        "prompt_iteration": str(prompt_iteration) if prompt_iteration else None,
        "agentic_policy_patch": str(policy_patch) if policy_patch else None,
        "ready_path_fixture": str(ready_path_fixture) if ready_path_fixture else None,
        "stage": plan.get("stage"),
        "physical_robot_motion": plan.get("physical_robot_motion"),
        "success_evidence_required": plan.get("required_evidence_before_success_claim", []),
        "policy_patch_rules": [item.get("id") for item in policy_patch_payload.get("rules", [])],
        "ready_path_step_types": ready_path_fixture_payload.get("next_step_types"),
        "task_success_claim_allowed": (
            prompt_iteration_payload.get("success_accounting", {}).get("task_success_claim_allowed")
            if prompt_iteration_payload
            else None
        ),
        "purpose": "human-readable state of the verifier-gated SO-100 SmolVLA agentic loop",
    }
    manifest_path = output.with_suffix(".json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _render_html(
    *,
    title: str,
    output: Path,
    contract: dict[str, Any],
    plan: dict[str, Any],
    advice: dict[str, Any],
    prompt_iteration: dict[str, Any],
    policy_patch: dict[str, Any],
    ready_path_fixture: dict[str, Any],
) -> str:
    policy = contract.get("policy", {})
    task_goal = contract.get("task_goal", {})
    agentic = contract.get("agentic_layer", {})
    verifier = agentic.get("verifier_contract", {})
    image_sections = _render_image_sections(advice=advice, output_dir=output.parent)
    reframe_items = "".join(
        f"<li>{html.escape(_diagnostic_text(item))}</li>"
        for item in advice.get("actions", [])
    )
    next_steps = "".join(_render_step(step) for step in plan.get("next_steps", []))
    external_blocker = _render_external_blocker(plan.get("external_setup_blocker", {}))
    post_external = "".join(_render_step(step) for step in plan.get("post_external_setup_verification", []))
    success_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in plan.get("required_evidence_before_success_claim", []))
    blockers = "".join(f"<li>{html.escape(str(item))}</li>" for item in agentic.get("blockers", []))
    guardrails = "".join(f"<li>{html.escape(str(item))}</li>" for item in plan.get("guardrails", []))
    policy_patch_section = _render_policy_patch(policy_patch)
    ready_path_section = _render_ready_path_fixture(ready_path_fixture)
    prompt_iteration_section = _render_prompt_iteration(prompt_iteration)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172026; background: #fbfbf8; }}
    h1 {{ font-size: 24px; margin: 0 0 4px; }}
    h2 {{ font-size: 17px; margin: 20px 0 8px; }}
    h3 {{ font-size: 14px; margin: 12px 0 6px; }}
    .note {{ color: #52616b; margin: 0 0 18px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }}
    .cell, .panel {{ border: 1px solid #d5d8d2; border-radius: 8px; background: white; padding: 12px; }}
    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; background: #e8f2ff; color: #17456d; font-size: 12px; }}
    .blocked {{ background: #fff1df; color: #8a4b00; }}
    .ready {{ background: #eaf7ee; color: #126536; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }}
    img {{ width: 100%; max-height: 420px; object-fit: contain; background: #f2f4f3; border: 1px solid #e0e3df; }}
    code {{ font-size: 12px; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f6f4; padding: 10px; border-radius: 6px; }}
    ul {{ padding-left: 20px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class=\"note\">SmolVLA proposes; the agentic layer decides through verifier gates. This report is not a benchmark success claim.</p>
  <section class=\"summary\">
    <div class=\"cell\"><strong>Task</strong><br>{html.escape(str(policy.get('instruction')))}</div>
    <div class=\"cell\"><strong>Stage</strong><br><span class=\"badge blocked\">{html.escape(str(plan.get('stage')))}</span></div>
    <div class=\"cell\"><strong>Physical motion now</strong><br>{html.escape(str(plan.get('physical_robot_motion')))}</div>
    <div class=\"cell\"><strong>Transport direction</strong><br>{html.escape(str(task_goal.get('transport_direction')))}</div>
    <div class=\"cell\"><strong>Policy cameras</strong><br>{html.escape(str(policy.get('policy_camera_indexes')))}</div>
    <div class=\"cell\"><strong>Observer cameras</strong><br>{html.escape(str(policy.get('observer_camera_indexes')))}</div>
  </section>
  <h2>Verifier State</h2>
  <section class=\"summary\">
    <div class=\"cell\"><strong>Pregrasp</strong><br>{html.escape(str(verifier.get('pregrasp_gate_status')))}</div>
    <div class=\"cell\"><strong>Jaw</strong><br>{html.escape(str(verifier.get('jaw_gate_status')))}</div>
    <div class=\"cell\"><strong>Last grasp</strong><br>{html.escape(str(verifier.get('last_grasp_outcome')))}</div>
    <div class=\"cell\"><strong>Relocation</strong><br>{html.escape(str(verifier.get('relocation_verifier_status')))}</div>
  </section>
  <h2>Current Camera Evidence</h2>
  <div class=\"grid\">{image_sections}</div>
  <h2>Policy Input Diagnostics</h2>
  <div class=\"panel\"><ul>{reframe_items}</ul></div>
  <h2>Autonomous Next Steps</h2>
  <div class=\"grid\">{next_steps or external_blocker}</div>
  <h2>Post External Setup Verification</h2>
  <div class=\"grid\">{post_external}</div>
  <h2>Required Before Any Success Claim</h2>
  <div class=\"panel\"><ul>{success_items}</ul></div>
  <h2>Agentic Policy Patch</h2>
  {policy_patch_section}
  <h2>Ready-Path Fixture</h2>
  {ready_path_section}
  <h2>Prompt Iteration Record</h2>
  {prompt_iteration_section}
  <h2>Current Blockers</h2>
  <div class=\"panel\"><ul>{blockers}</ul></div>
  <h2>Guardrails</h2>
  <div class=\"panel\"><ul>{guardrails}</ul></div>
</body>
</html>
"""


def _render_image_sections(*, advice: dict[str, Any], output_dir: Path) -> str:
    pregrasp_path = advice.get("pregrasp_probe")
    if not pregrasp_path:
        return ""
    pregrasp = _load_json(Path(pregrasp_path))
    sections = []
    for item in pregrasp.get("assessments", []):
        image_path = item.get("image_path")
        if not image_path:
            continue
        src = html.escape(os.path.relpath(str(image_path), start=output_dir))
        status = "ready" if item.get("usable_for_pregrasp") else "blocked"
        sections.append(
            f"""
            <section class=\"panel\">
              <h3>Camera {html.escape(str(item.get('camera')))} <span class=\"badge {status}\">{html.escape(status)}</span></h3>
              <img src=\"{src}\" alt=\"camera {html.escape(str(item.get('camera')))} frame\">
              <p><strong>BBox:</strong> <code>{html.escape(str(item.get('bbox_xyxy')))}</code></p>
              <p><strong>Edge clipped:</strong> {html.escape(str(item.get('edge_clipped')))}</p>
            </section>
            """
        )
    return "\n".join(sections)


def _render_step(step: dict[str, Any]) -> str:
    command = step.get("command")
    command_text = " ".join(str(item) for item in command) if isinstance(command, list) else ""
    advice = step.get("reframe_advice", []) or step.get("diagnostics", [])
    advice_text = "".join(
        f"<li>{html.escape(_diagnostic_text(item))}</li>"
        for item in advice
    )
    return f"""
    <section class=\"panel\">
      <h3>{html.escape(str(step.get('type')))}</h3>
      <p><strong>Physical motion:</strong> {html.escape(str(step.get('physical_robot_motion')))}</p>
      <p>{html.escape(str(step.get('operator_goal') or step.get('reason') or ''))}</p>
      {"<ul>" + advice_text + "</ul>" if advice_text else ""}
      {"<pre><code>" + html.escape(command_text) + "</code></pre>" if command_text else ""}
    </section>
    """


def _render_external_blocker(blocker: dict[str, Any]) -> str:
    if not blocker:
        return ""
    diagnostics = blocker.get("diagnostics", [])
    diagnostic_items = "".join(
        f"<li>{html.escape(_diagnostic_text(item))}</li>"
        for item in diagnostics
    )
    return f"""
    <section class=\"panel\">
      <h3>{html.escape(str(blocker.get('type')))}</h3>
      <p><strong>Agent actionable:</strong> {html.escape(str(blocker.get('agent_actionable')))}</p>
      <p><strong>VLA prompt allowed:</strong> {html.escape(str(blocker.get('vla_prompt_allowed')))}</p>
      <p>{html.escape(str(blocker.get('why_not_agent_action') or blocker.get('reason') or ''))}</p>
      {"<ul>" + diagnostic_items + "</ul>" if diagnostic_items else ""}
    </section>
    """


def _diagnostic_text(item: dict[str, Any]) -> str:
    nudge = item.get("image_space_nudge")
    raw_text = str(item.get("diagnostic_summary") or item.get("image_space_goal") or item.get("reason") or "")
    if isinstance(nudge, dict):
        text = _nudge_diagnostic_text(nudge)
        if text:
            if "external setup blocker" in raw_text:
                return f"external setup blocker; {text}"
            return text
    return _legacy_non_command_text(raw_text)


def _nudge_diagnostic_text(nudge: dict[str, Any]) -> str | None:
    shift = nudge.get("recommended_shift_px")
    margin = nudge.get("target_margin_px")
    bbox = nudge.get("current_bbox_xyxy")
    if not isinstance(shift, list) or len(shift) < 2:
        return None
    parts = []
    dx = float(shift[0])
    dy = float(shift[1])
    if abs(dx) >= 1.0:
        side = "left" if dx > 0 else "right"
        parts.append(f"about {abs(dx):.0f}px too far {side}")
    if abs(dy) >= 1.0:
        side = "high" if dy > 0 else "low"
        parts.append(f"about {abs(dy):.0f}px too {side}")
    if not parts:
        return "image-space diagnostic: detected target bbox already satisfies the policy-input margin"
    bbox_text = f" bbox={bbox}" if bbox else ""
    margin_text = f" for the {margin}px policy-input margin" if margin is not None else " for the policy-input margin"
    return "image-space diagnostic only: detected target bbox is " + " and ".join(parts) + margin_text + bbox_text


def _legacy_non_command_text(text: str) -> str:
    return (
        text.replace(
            "Image-space diagnostic: shift target appearance about 32px right.",
            "Image-space diagnostic only: detected target bbox is about 32px too far left for the policy-input margin.",
        )
        .replace(
            "Image-space diagnostic: shift target appearance about 32px right; shift target appearance about 32px down.",
            "Image-space diagnostic only: detected target bbox is about 32px too far left and about 32px too high for the policy-input margin.",
        )
        .replace(
            "shift target appearance about 32px right",
            "detected target bbox is about 32px too far left for the policy-input margin",
        )
        .replace(
            "shift target appearance about 32px down",
            "detected target bbox is about 32px too high for the policy-input margin",
        )
    )


def _render_policy_patch(policy_patch: dict[str, Any]) -> str:
    if not policy_patch:
        return '<div class="panel"><p>No policy patch linked.</p></div>'
    rules = "".join(
        f"<li><strong>{html.escape(str(item.get('id')))}</strong>: {html.escape(str(item.get('action')))}</li>"
        for item in policy_patch.get("rules", [])
    )
    normalizations = "".join(
        f"<li>{html.escape(str(item.get('from')))} -> {html.escape(str(item.get('to')))}</li>"
        for item in policy_patch.get("legacy_normalizations", [])
    )
    prompt = policy_patch.get("prompt_contract", {})
    return f"""
    <div class=\"panel\">
      <p><strong>VLA prompt target:</strong> {html.escape(str(prompt.get('vla_prompt_target')))}</p>
      <p><strong>Does not prompt operator:</strong> {html.escape(str(prompt.get('does_not_prompt_operator')))}</p>
      <p><strong>Task success claim allowed:</strong> {html.escape(str(policy_patch.get('success_accounting', {}).get('task_success_claim_allowed')))}</p>
      <h3>Rules</h3>
      <ul>{rules}</ul>
      <h3>Legacy Normalizations</h3>
      <ul>{normalizations}</ul>
    </div>
    """


def _render_ready_path_fixture(fixture: dict[str, Any]) -> str:
    if not fixture:
        return '<div class="panel"><p>No ready-path fixture linked.</p></div>'
    checks = "".join(
        f"<li>{html.escape(str(item.get('name')))}: {html.escape(str(item.get('passed')))}</li>"
        for item in fixture.get("checks", [])
    )
    steps = "".join(f"<li>{html.escape(str(item))}</li>" for item in fixture.get("next_step_types", []))
    return f"""
    <div class=\"panel\">
      <p><strong>Status:</strong> {html.escape(str(fixture.get('status')))}</p>
      <p><strong>Physical robot motion:</strong> {html.escape(str(fixture.get('physical_robot_motion')))}</p>
      <h3>Expected Ready Step Order</h3>
      <ul>{steps}</ul>
      <h3>Checks</h3>
      <ul>{checks}</ul>
    </div>
    """


def _render_prompt_iteration(prompt_iteration: dict[str, Any]) -> str:
    if not prompt_iteration:
        return '<div class="panel"><p>No prompt iteration linked.</p></div>'
    return f"""
    <div class=\"panel\">
      <p><strong>Policy patch:</strong> <code>{html.escape(str(prompt_iteration.get('agentic_policy_patch')))}</code></p>
      <p><strong>Task success claim allowed:</strong> {html.escape(str(prompt_iteration.get('success_accounting', {}).get('task_success_claim_allowed')))}</p>
      <p><strong>Next stage:</strong> {html.escape(str(prompt_iteration.get('next_iteration', {}).get('stage')))}</p>
      <p><strong>Next step type:</strong> {html.escape(str(prompt_iteration.get('next_iteration', {}).get('next_step_type')))}</p>
    </div>
    """


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-readable real SO-100 agentic loop report.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--next-plan", type=Path, required=True)
    parser.add_argument("--reframe-advice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Real SO-100 Agentic Loop Report")
    parser.add_argument("--prompt-iteration", type=Path)
    parser.add_argument("--policy-patch", type=Path)
    parser.add_argument("--ready-path-fixture", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_agentic_loop_report(
                contract=args.contract,
                next_plan=args.next_plan,
                reframe_advice=args.reframe_advice,
                output=args.output,
                title=args.title,
                prompt_iteration=args.prompt_iteration,
                policy_patch=args.policy_patch,
                ready_path_fixture=args.ready_path_fixture,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
