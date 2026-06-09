#!/usr/bin/env python3
"""Audit oracle overlay milestone artifacts for report/sample completeness."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@dataclass(frozen=True)
class MilestoneSpec:
    key: str
    title: str
    html_path: str
    image_dir: str | None
    manifest_path: str | None
    sample_key: str | None
    min_samples: int
    representative_images: tuple[str, ...]
    note: str
    allow_blocked: bool = False


SPECS = [
    MilestoneSpec(
        key="local_static_validation",
        title="Local static projection/rendering validation",
        html_path="local_html_validation/validation_report.html",
        image_dir="local_html_validation",
        manifest_path="local_html_validation/validation_report.json",
        sample_key="total",
        min_samples=37,
        representative_images=(
            "local_html_validation/contact_sheet_projection.png",
            "local_html_validation/contact_sheet_rendering.png",
            "local_html_validation/contact_sheet_sim_frame_overlay.png",
        ),
        note="Projection, rendering, fallback, and saved-frame overlay cases.",
    ),
    MilestoneSpec(
        key="trajectory_validation",
        title="Moving-object trajectory validation",
        html_path="local_trajectory_validation/validation_report.html",
        image_dir="local_trajectory_validation/projection_trajectory",
        manifest_path="local_trajectory_validation/validation_report.json",
        sample_key=None,
        min_samples=20,
        representative_images=("local_trajectory_validation/contact_sheet_projection_trajectory.png",),
        note="Frame-by-frame moving object projection evidence.",
    ),
    MilestoneSpec(
        key="pose_dict_validation",
        title="Simulator pose dictionary validation",
        html_path="local_pose_dict_validation/validation_report.html",
        image_dir="local_pose_dict_validation/dict_pose_trajectory",
        manifest_path="local_pose_dict_validation/validation_report.json",
        sample_key=None,
        min_samples=20,
        representative_images=("local_pose_dict_validation/contact_sheet_dict_pose_trajectory.png",),
        note="Dict pose styles such as p and position.",
    ),
    MilestoneSpec(
        key="sensor_camera_validation",
        title="sensor_data camera parameter validation",
        html_path="local_sensor_camera_validation/validation_report.html",
        image_dir="local_sensor_camera_validation/sensor_data_camera_trajectory",
        manifest_path="local_sensor_camera_validation/validation_report.json",
        sample_key=None,
        min_samples=20,
        representative_images=("local_sensor_camera_validation/contact_sheet_sensor_data_camera_trajectory.png",),
        note="Camera parameters nested under sensor_data[cam].",
    ),
    MilestoneSpec(
        key="multi_camera_validation",
        title="Multi-camera preferred/fallback validation",
        html_path="local_multi_camera_validation/validation_report.html",
        image_dir="local_multi_camera_validation/multi_camera_trajectory",
        manifest_path="local_multi_camera_validation/validation_report.json",
        sample_key=None,
        min_samples=20,
        representative_images=("local_multi_camera_validation/contact_sheet_multi_camera_trajectory.png",),
        note="Preferred base camera and fallback camera selection.",
    ),
    MilestoneSpec(
        key="edge_case_validation",
        title="Projection edge-case fallback validation",
        html_path="local_edge_case_validation/validation_report.html",
        image_dir="local_edge_case_validation/projection_edge_cases",
        manifest_path="local_edge_case_validation/validation_report.json",
        sample_key=None,
        min_samples=20,
        representative_images=("local_edge_case_validation/contact_sheet_projection_edge_cases.png",),
        note="Behind-camera, out-of-frame, malformed, and missing metadata fallbacks.",
    ),
    MilestoneSpec(
        key="raw_vs_overlay_comparison",
        title="Raw-vs-oracle-overlay comparison",
        html_path="raw_vs_overlay_comparison/comparison_report.html",
        image_dir="raw_vs_overlay_comparison/overlay",
        manifest_path="raw_vs_overlay_comparison/comparison_manifest.json",
        sample_key="sample_count",
        min_samples=20,
        representative_images=("raw_vs_overlay_comparison/comparison_contact_sheet.png",),
        note="Visual comparison between baseline image and oracle-overlaid image.",
    ),
    MilestoneSpec(
        key="zoomed_representatives",
        title="Zoomed representative samples",
        html_path="zoomed_raw_vs_overlay/zoom_report.html",
        image_dir="zoomed_raw_vs_overlay/panels",
        manifest_path="zoomed_raw_vs_overlay/zoom_manifest.json",
        sample_key="panel_count",
        min_samples=10,
        representative_images=("zoomed_raw_vs_overlay/zoom_contact_sheet.png",),
        note="Large marker-centered panels for human review.",
    ),
    MilestoneSpec(
        key="diverse_object_projection",
        title="Diverse non-center object projection",
        html_path="diverse_object_projection/diverse_object_report.html",
        image_dir="diverse_object_projection/overlay",
        manifest_path="diverse_object_projection/diverse_object_manifest.json",
        sample_key="episode_count",
        min_samples=30,
        representative_images=(
            "diverse_object_projection/diverse_object_full_contact_sheet.png",
            "diverse_object_projection/diverse_object_zoom_contact_sheet.png",
        ),
        note="Primary evidence against center-marker ambiguity.",
    ),
    MilestoneSpec(
        key="center_bias_audit",
        title="Center-bias audit",
        html_path="center_bias_audit/center_bias_audit.html",
        image_dir="center_bias_audit",
        manifest_path="center_bias_audit/center_bias_audit.json",
        sample_key="non_center_count",
        min_samples=20,
        representative_images=(
            "center_bias_audit/center_distance_distribution.png",
            "center_bias_audit/object_distribution.png",
        ),
        note="Quantitative non-center and object-balance evidence.",
    ),
    MilestoneSpec(
        key="paper_figure_pack",
        title="Paper-facing figure pack",
        html_path="paper_figure_pack/figure_pack.html",
        image_dir=None,
        manifest_path="paper_figure_pack/figure_pack_manifest.json",
        sample_key="sample_groups",
        min_samples=10,
        representative_images=(
            "diverse_object_projection/diverse_object_full_contact_sheet.png",
            "diverse_object_projection/diverse_object_zoom_contact_sheet.png",
        ),
        note="Claim-labeled paper figures, with live blocker separated.",
    ),
    MilestoneSpec(
        key="runpod_lifecycle_decision",
        title="RunPod lifecycle decision",
        html_path="runpod_lifecycle_decision/runpod_lifecycle_decision.html",
        image_dir=None,
        manifest_path="runpod_lifecycle_decision/runpod_lifecycle_decision.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(
            "diverse_object_projection/diverse_object_full_contact_sheet.png",
            "diverse_object_projection/diverse_object_zoom_contact_sheet.png",
        ),
        note="Documents stop overlay/probe Pods, keep baseline/parity Pods, and when cloud is useful again.",
    ),
    MilestoneSpec(
        key="policy_input_readiness",
        title="SmolVLA policy-input readiness",
        html_path="policy_input_readiness/policy_input_report.html",
        image_dir="policy_input_readiness/policy_preview",
        manifest_path="policy_input_readiness/policy_input_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=("policy_input_readiness/policy_input_contact_sheet.png",),
        note="Raw observation, oracle overlay, and resized normalized policy tensor preview.",
    ),
    MilestoneSpec(
        key="encoding_ablation",
        title="Oracle point encoding ablation",
        html_path="encoding_ablation/encoding_ablation_report.html",
        image_dir="encoding_ablation/variants",
        manifest_path="encoding_ablation/encoding_ablation_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=("encoding_ablation/encoding_ablation_contact_sheet.png",),
        note="Compares ring/cross, solid dot, soft heatmap, and arrow-label visual prompts for the same oracle point.",
    ),
    MilestoneSpec(
        key="temporal_consistency",
        title="Oracle point temporal consistency",
        html_path="temporal_consistency/temporal_consistency_report.html",
        image_dir="temporal_consistency/overlay",
        manifest_path="temporal_consistency/temporal_consistency_manifest.json",
        sample_key=None,
        min_samples=20,
        representative_images=(
            "temporal_consistency/temporal_consistency_contact_sheet.png",
            "temporal_consistency/temporal_consistency_sequence.gif",
            "temporal_consistency/temporal_consistency_trail.png",
        ),
        note="Synthetic diagnostic trajectory; verifies stable projected points over time, not real sim rollout success.",
    ),
    MilestoneSpec(
        key="actual_sim_frame_overlay_fallback",
        title="Actual sim RGB overlay fallback",
        html_path="actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_report.html",
        image_dir="actual_sim_frame_overlay_fallback/overlay",
        manifest_path="actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(
            "actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_contact_sheet.png",
            "actual_sim_frame_overlay_fallback/actual_sim_frame_overlay_sequence.gif",
        ),
        note="Uses real saved simulator RGB frames; fallback overlay only because matching pose/camera metadata is absent.",
    ),
    MilestoneSpec(
        key="actual_sim_visual_heuristic_overlay",
        title="Actual sim visual heuristic overlay",
        html_path="actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_report.html",
        image_dir="actual_sim_visual_heuristic_overlay/overlay",
        manifest_path="actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=("actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_contact_sheet.png",),
        note="Uses actual saved simulator RGB frames and image-only visual heuristics; not true oracle projection.",
    ),
    MilestoneSpec(
        key="actual_sim_heuristic_policy_input",
        title="Actual sim heuristic policy-input readiness",
        html_path="actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_report.html",
        image_dir="actual_sim_heuristic_policy_input/policy_preview",
        manifest_path="actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=("actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_contact_sheet.png",),
        note="Actual sim RGB -> image-only heuristic overlay -> SmolVLA-style image tensor preview; not true oracle projection.",
    ),
    MilestoneSpec(
        key="actual_sim_heuristic_temporal",
        title="Actual sim heuristic temporal consistency",
        html_path="actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_report.html",
        image_dir="actual_sim_heuristic_temporal/sequence_overlay",
        manifest_path="actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(
            "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_contact_sheet.png",
            "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_trail.png",
            "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_sequence.gif",
        ),
        note="Actual sim RGB sequence with image-only heuristic point trail; not true oracle projection.",
    ),
    MilestoneSpec(
        key="actual_sim_evidence_preflight",
        title="Actual sim evidence preflight",
        html_path="actual_sim_evidence_preflight/actual_sim_evidence_preflight.html",
        image_dir="actual_sim_evidence_preflight",
        manifest_path="actual_sim_evidence_preflight/actual_sim_evidence_preflight_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=("actual_sim_evidence_preflight/actual_sim_evidence_preflight_contact_sheet.png",),
        note="Scans existing CP24 actual sim artifacts and separates RGB evidence from true-oracle readiness.",
    ),
    MilestoneSpec(
        key="actual_sim_only_dashboard",
        title="Actual-sim-only dashboard",
        html_path="actual_sim_only_dashboard/actual_sim_only_dashboard.html",
        image_dir="actual_sim_only_dashboard",
        manifest_path="actual_sim_only_dashboard/actual_sim_only_dashboard_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(
            "actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_contact_sheet.png",
            "actual_sim_heuristic_policy_input/actual_sim_heuristic_policy_input_contact_sheet.png",
            "actual_sim_heuristic_temporal/actual_sim_heuristic_temporal_contact_sheet.png",
        ),
        note="Synthetic-free dashboard summarizing actual CP24 simulation RGB evidence only.",
    ),
    MilestoneSpec(
        key="actual_sim_claim_matrix",
        title="Actual-sim claim matrix",
        html_path="actual_sim_claim_matrix/actual_sim_claim_matrix.html",
        image_dir=None,
        manifest_path="actual_sim_claim_matrix/actual_sim_claim_matrix_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(
            "actual_sim_only_dashboard/actual_sim_only_dashboard.html",
            "actual_sim_visual_heuristic_overlay/actual_sim_visual_heuristic_contact_sheet.png",
        ),
        note="Synthetic-free claim matrix showing which paper claims are supported and which true-oracle claim remains blocked.",
    ),
    MilestoneSpec(
        key="actual_sim_true_oracle_projection",
        title="Actual sim true oracle projection",
        html_path="live_true_oracle_projection/actual_sim_true_oracle_report.html",
        image_dir="live_true_oracle_projection",
        manifest_path="live_true_oracle_projection/actual_sim_true_oracle_report_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=("live_true_oracle_projection/actual_sim_true_oracle_contact_sheet.png",),
        note="Required final evidence tier; blocked until actual sim RGB, object pose, camera metadata, and overlay are saved from the same timestep.",
        allow_blocked=True,
    ),
    MilestoneSpec(
        key="actual_sim_true_oracle_readiness_gap",
        title="Actual sim true-oracle readiness gap",
        html_path="actual_sim_true_oracle_readiness_gap/actual_sim_true_oracle_readiness_gap.html",
        image_dir="actual_sim_true_oracle_readiness_gap/images",
        manifest_path="actual_sim_true_oracle_readiness_gap/actual_sim_true_oracle_readiness_gap_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(
            "actual_sim_true_oracle_readiness_gap/actual_sim_true_oracle_readiness_gap_contact_sheet.jpg",
        ),
        note="Actual simulator RGB samples showing why Tier O remains blocked: same-step pose/camera metadata is missing.",
    ),
    MilestoneSpec(
        key="actual_rgb_synthetic_metadata_true_oracle_codepath",
        title="Actual RGB + synthetic metadata true-oracle codepath diagnostic",
        html_path="actual_rgb_synthetic_metadata_true_oracle_codepath/actual_rgb_synthetic_metadata_codepath_report.html",
        image_dir="actual_rgb_synthetic_metadata_true_oracle_codepath/panels",
        manifest_path="actual_rgb_synthetic_metadata_true_oracle_codepath/actual_rgb_synthetic_metadata_codepath_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(
            "actual_rgb_synthetic_metadata_true_oracle_codepath/actual_rgb_synthetic_metadata_codepath_contact_sheet.jpg",
        ),
        note="Uses actual simulator RGB but synthetic pose/camera metadata; validates augmentation/projection codepath only, not Tier O.",
    ),
    MilestoneSpec(
        key="actual_sim_true_oracle_probe_blocker",
        title="Actual sim true-oracle probe blocker",
        html_path="actual_sim_true_oracle_probe_blocker/actual_sim_true_oracle_probe_blocker.html",
        image_dir=None,
        manifest_path="actual_sim_true_oracle_probe_blocker/actual_sim_true_oracle_probe_blocker_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Mac-local zero-action probe blocker showing SAPIEN/Vulkan renderer failure before true-oracle capture.",
    ),
    MilestoneSpec(
        key="actual_sim_true_oracle_remote_handoff",
        title="Actual sim true-oracle remote handoff",
        html_path="actual_sim_true_oracle_remote_handoff/actual_sim_true_oracle_remote_handoff_report.html",
        image_dir=None,
        manifest_path="actual_sim_true_oracle_remote_handoff/actual_sim_true_oracle_remote_handoff_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Renderer-capable environment handoff plan: probe first, then SmolVLA policy only after 10 strict probe samples.",
    ),
    MilestoneSpec(
        key="actual_sim_true_oracle_two_stage_result",
        title="Actual sim true-oracle two-stage result",
        html_path="actual_sim_true_oracle_two_stage_result/actual_sim_true_oracle_two_stage_result.html",
        image_dir=None,
        manifest_path="actual_sim_true_oracle_two_stage_result/actual_sim_true_oracle_two_stage_result_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Imported two-stage probe-then-policy result; currently blocked locally until renderer-capable execution succeeds.",
        allow_blocked=True,
    ),
    MilestoneSpec(
        key="paper_progress_ledger",
        title="Paper progress ledger",
        html_path="paper_progress_ledger/paper_progress_ledger_report.html",
        image_dir=None,
        manifest_path="paper_progress_ledger/paper_progress_ledger_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Claim-by-claim paper progress ledger with supported, diagnostic-only, and blocked evidence separated.",
    ),
    MilestoneSpec(
        key="agentic_smolvla_experiment_matrix_result",
        title="Agentic SmolVLA experiment matrix result",
        html_path="agentic_smolvla_experiment_matrix_result/agentic_smolvla_experiment_matrix_result.html",
        image_dir=None,
        manifest_path="agentic_smolvla_experiment_matrix_result/agentic_smolvla_experiment_matrix_result_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="C0-C5 result collection scaffold with current partial/pending/blocked condition states.",
    ),
    MilestoneSpec(
        key="agentic_retry_schema_readiness",
        title="Agentic retry schema readiness",
        html_path="agentic_retry_schema_readiness/agentic_retry_schema_readiness.html",
        image_dir=None,
        manifest_path="agentic_retry_schema_readiness/agentic_retry_schema_readiness_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="CP22/CP23 agentic trace/retry schema readiness; not paper-facing SmolVLA success evidence.",
    ),
    MilestoneSpec(
        key="renderer_env_preflight",
        title="Renderer environment preflight",
        html_path="renderer_env_preflight/renderer_env_preflight_report.html",
        image_dir=None,
        manifest_path="renderer_env_preflight/renderer_env_preflight_report_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Non-mutating import/GPU/Vulkan preflight before running the two-stage true-oracle gate.",
    ),
    MilestoneSpec(
        key="remote_true_oracle_evidence_pack",
        title="Remote true-oracle evidence pack",
        html_path="remote_true_oracle_evidence_pack/remote_true_oracle_evidence_pack_report.html",
        image_dir=None,
        manifest_path="remote_true_oracle_evidence_pack/remote_true_oracle_evidence_pack_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Single-command non-destructive handoff for preflight, two-stage gate, result import, and dashboard rebuild.",
    ),
    MilestoneSpec(
        key="paper_claim_gate",
        title="Paper claim gate",
        html_path="paper_claim_gate/paper_claim_gate_report.html",
        image_dir=None,
        manifest_path="paper_claim_gate/paper_claim_gate_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Claim-level audit separating allowed, caution-only, and blocked manuscript claims.",
    ),
    MilestoneSpec(
        key="agentic_smolvla_paper_outline",
        title="Agentic SmolVLA paper outline",
        html_path="agentic_smolvla_paper_outline/agentic_smolvla_paper_outline_report.html",
        image_dir=None,
        manifest_path="agentic_smolvla_paper_outline/agentic_smolvla_paper_outline_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Claim-safe paper outline scaffold for Intro, Related Work, Method, Experiments, Results, and Limitations.",
    ),
    MilestoneSpec(
        key="next_action_command_center",
        title="Next action command center",
        html_path="next_action_command_center/next_action_command_center_report.html",
        image_dir=None,
        manifest_path="next_action_command_center/next_action_command_center_manifest.json",
        sample_key="sample_count",
        min_samples=10,
        representative_images=(),
        note="Operational page for the next renderer-capable command, critical reports, and blocked claims.",
    ),
    MilestoneSpec(
        key="live_maniskill_gate",
        title="Live ManiSkill true oracle projection",
        html_path="live_audit_blocked_probe/live_oracle_audit.html",
        image_dir="live_audit_blocked_probe",
        manifest_path="live_audit_blocked_probe/live_oracle_audit.json",
        sample_key="frame_count",
        min_samples=10,
        representative_images=(),
        note="Expected to remain BLOCKED until live SAPIEN/Vulkan rendering produces real frames and metadata.",
        allow_blocked=True,
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_images(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return 1 if path.suffix.lower() in IMAGE_SUFFIXES else 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)


def _sample_count(root: Path, spec: MilestoneSpec, manifest: dict[str, Any]) -> int:
    if spec.sample_key:
        value = manifest.get(spec.sample_key)
        if isinstance(value, list):
            return len(value)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return _count_images(root / spec.image_dir) if spec.image_dir else 0


def _rel(report_path: Path, target: Path) -> str:
    try:
        return target.relative_to(report_path.parent).as_posix()
    except ValueError:
        return target.as_posix()


def _audit(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        html_path = root / spec.html_path
        manifest_path = root / spec.manifest_path if spec.manifest_path else None
        manifest = _load_json(manifest_path) if manifest_path else {}
        sample_count = _sample_count(root, spec, manifest)
        image_count = _count_images(root / spec.image_dir) if spec.image_dir else sample_count
        representative_images = [root / image for image in spec.representative_images]
        reps_exist = all(path.exists() for path in representative_images)
        html_exists = html_path.exists()
        manifest_exists = manifest_path.exists() if manifest_path else True
        sample_ok = sample_count >= spec.min_samples
        if spec.allow_blocked:
            manifest_status = manifest.get("status", "missing")
            passed_ok = html_exists and manifest_exists and sample_ok and reps_exist and manifest_status == "passed"
            blocked_ok = html_exists and manifest_exists and manifest_status in {
                "blocked",
                "blocked_at_probe",
                "blocked_at_policy",
                "missing_manifest",
                "missing_blocker_evidence",
            }
            if passed_ok:
                status = "passed"
            elif blocked_ok:
                status = "blocked_expected"
            else:
                status = "missing_blocker_evidence"
        else:
            ok = html_exists and manifest_exists and sample_ok and reps_exist
            status = "passed" if ok else "failed"
        rows.append(
            {
                "key": spec.key,
                "title": spec.title,
                "status": status,
                "html": str(html_path),
                "html_exists": html_exists,
                "manifest": str(manifest_path) if manifest_path else "",
                "manifest_exists": manifest_exists,
                "sample_count": sample_count,
                "min_samples": spec.min_samples,
                "image_count": image_count,
                "representative_images": [str(path) for path in representative_images],
                "representative_images_exist": reps_exist,
                "note": spec.note,
            }
        )
    return rows


def _html(report_path: Path, rows: list[dict[str, Any]]) -> str:
    sections = []
    for row in rows:
        status = row["status"]
        image_html = "".join(
            f"""
          <figure>
            <img src="{html.escape(_rel(report_path, Path(image)))}" alt="{html.escape(row['title'])}">
            <figcaption>{html.escape(Path(image).name)}</figcaption>
          </figure>
            """
            for image in row["representative_images"]
            if Path(image).exists()
        )
        sections.append(
            f"""
      <section class="{html.escape(status)}">
        <span class="status">{html.escape(status.upper())}</span>
        <h2>{html.escape(row['title'])}</h2>
        <ul>
          <li>HTML exists: {row['html_exists']}</li>
          <li>Manifest exists: {row['manifest_exists']}</li>
          <li>Samples: {row['sample_count']} / required {row['min_samples']}</li>
          <li>Images found: {row['image_count']}</li>
          <li>{html.escape(row['note'])}</li>
        </ul>
        <p><a href="{html.escape(_rel(report_path, Path(row['html'])))}">Open milestone HTML</a></p>
        <div class="images">{image_html}</div>
      </section>
            """
        )
    passed = sum(1 for row in rows if row["status"] == "passed")
    blocked = sum(1 for row in rows if row["status"] == "blocked_expected")
    failed = len(rows) - passed - blocked
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Overlay Artifact Completeness Audit</title>
  <style>
    :root {{ --ink:#17130e; --paper:#f7efe1; --line:#d5c1a5; --pass:#008f5b; --block:#bd5b14; --fail:#b5382d; }}
    body {{ margin:0; color:var(--ink); background:linear-gradient(135deg,#f0dfc6,#faf7ef 48%,#e5eee7); font-family:Charter, Georgia, serif; }}
    header {{ padding:46px 54px 22px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:clamp(36px,5vw,72px); letter-spacing:-.055em; }}
    header p {{ max-width:980px; font-size:19px; line-height:1.45; }}
    main {{ padding:30px 54px 70px; display:grid; gap:22px; }}
    section {{ padding:22px; border-radius:26px; background:rgba(255,250,241,.9); border:2px solid var(--line); box-shadow:0 16px 44px rgba(58,39,13,.08); }}
    .passed {{ border-color:var(--pass); }} .blocked_expected {{ border-color:var(--block); }} .failed,.missing_blocker_evidence {{ border-color:var(--fail); }}
    .status {{ display:inline-flex; padding:8px 12px; border-radius:999px; font:800 13px Avenir Next, Helvetica, sans-serif; border:1px solid currentColor; background:white; }}
    .passed .status {{ color:var(--pass); }} .blocked_expected .status {{ color:var(--block); }} .failed .status,.missing_blocker_evidence .status {{ color:var(--fail); }}
    h2 {{ margin:14px 0 8px; font-size:30px; letter-spacing:-.03em; }}
    ul {{ font-family:Avenir Next, Helvetica, sans-serif; line-height:1.55; }}
    a {{ color:#174f38; font-weight:800; }}
    .images {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin-top:14px; }}
    figure {{ margin:0; padding:10px; border:1px solid var(--line); border-radius:18px; background:white; }}
    img {{ width:100%; display:block; border-radius:12px; background:#222; }}
    figcaption {{ margin-top:7px; font:700 13px Avenir Next, Helvetica, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Overlay Artifact Completeness Audit</h1>
    <p>Checks that each completed milestone has an HTML report plus enough representative samples for review. Summary: {passed} passed, {blocked} expected blocked, {failed} failed.</p>
  </header>
  <main>{''.join(sections)}</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _audit(root)
    status = "passed_static_blocked_live" if all(row["status"] in {"passed", "blocked_expected"} for row in rows) else "failed"
    report = {
        "status": status,
        "passed_count": sum(1 for row in rows if row["status"] == "passed"),
        "blocked_expected_count": sum(1 for row in rows if row["status"] == "blocked_expected"),
        "failed_count": sum(1 for row in rows if row["status"] not in {"passed", "blocked_expected"}),
        "rows": rows,
    }
    json_path = output_dir / "artifact_completeness_audit.json"
    html_path = output_dir / "artifact_completeness_audit.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_html(html_path, rows), encoding="utf-8")
    print(json.dumps({"status": status, "html": str(html_path), "json": str(json_path)}, indent=2, sort_keys=True))
    return 0 if status == "passed_static_blocked_live" else 1


if __name__ == "__main__":
    raise SystemExit(main())
