#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_real_so100_prestage_evidence import audit_prestage_evidence
from scripts.build_real_so100_agentic_prestage_pack import build_agentic_prestage_pack
from scripts.build_real_so100_gate_report import build_gate_report
from scripts.build_real_so100_movement_report import build_movement_report
from scripts.build_real_so100_prestage_dashboard import build_prestage_dashboard
from scripts.real_so100_next_runbook import build_next_runbook


def build_prestage_bundle(
    *,
    reports: list[Path],
    gate_manifest: Path,
    grasp_outcome: Path,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_payload = _load_json(gate_manifest)
    next_action_gate = Path(gate_payload["next_action_gate"])

    movement_html = output_dir / f"{label}_movement_report.html"
    gate_html = output_dir / f"{label}_gate_report.html"
    pack_json = output_dir / f"{label}_agentic_prestage_pack.json"
    runbook_md = output_dir / f"{label}_next_runbook.md"
    audit_json = output_dir / f"{label}_prestage_audit.json"
    dashboard_html = output_dir / f"{label}_prestage_dashboard.html"

    movement = build_movement_report(
        reports=reports,
        output=movement_html,
        title=f"{label} Movement Evidence Report",
    )
    gate = build_gate_report(
        gate_manifest=gate_manifest,
        output=gate_html,
        title=f"{label} Gate Evidence Report",
    )
    pack = build_agentic_prestage_pack(
        output=pack_json,
        movement_report_manifest=Path(movement["manifest_path"]),
        gate_report_manifest=Path(gate["manifest_path"]),
        next_action_gate=next_action_gate,
        grasp_outcome=grasp_outcome,
        title=f"{label} Agentic Pre-stage Evidence Pack",
    )
    runbook = build_next_runbook(pre_stage_pack=pack_json, output=runbook_md)
    audit = audit_prestage_evidence(
        pre_stage_pack=pack_json,
        runbook_manifest=Path(runbook["manifest_path"]),
        output=audit_json,
    )
    dashboard = build_prestage_dashboard(
        audit_manifest=audit_json,
        output=dashboard_html,
        title=f"{label} Agentic Pre-stage Dashboard",
    )

    manifest = {
        "status": "passed" if audit["status"] == "passed" and dashboard["status"] == "passed" else "failed",
        "operation": "real_so100_prestage_bundle",
        "label": label,
        "output_dir": str(output_dir),
        "movement_report_manifest": movement["manifest_path"],
        "movement_report_html": movement["output_html"],
        "gate_report_manifest": gate["manifest_path"],
        "gate_report_html": gate["output_html"],
        "pre_stage_pack": str(pack_json),
        "runbook_manifest": runbook["manifest_path"],
        "runbook_markdown": runbook["output_markdown"],
        "audit_manifest": str(audit_json),
        "dashboard_manifest": dashboard["manifest_path"],
        "dashboard_html": dashboard["output_html"],
        "current_gate_status": pack.get("current_gate_status"),
        "allowed_physical_action": pack.get("allowed_physical_action"),
        "video_count": pack.get("video_count"),
        "audit_failed_check_count": audit.get("failed_check_count"),
        "purpose": "one-command refresh for video-backed SO-100 agentic-layer pre-stage evidence",
    }
    manifest_path = output_dir / f"{label}_bundle.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the complete real SO-100 pre-stage evidence bundle.")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--grasp-outcome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/real_so100/reports"))
    parser.add_argument("--label", default="real_so100_prestage_bundle_next")
    args = parser.parse_args()
    print(
        json.dumps(
            build_prestage_bundle(
                reports=args.report,
                gate_manifest=args.gate_manifest,
                grasp_outcome=args.grasp_outcome,
                output_dir=args.output_dir,
                label=args.label,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
