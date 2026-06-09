#!/usr/bin/env python3
"""Build an audited milestone dashboard for oracle overlay validation."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in suffixes)


def _rel(report_path: Path, path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _status(ok: bool, blocked: bool = False) -> str:
    if ok:
        return "PASS"
    if blocked:
        return "BLOCKED"
    return "MISSING"


def _class(ok: bool, blocked: bool = False) -> str:
    if ok:
        return "pass"
    if blocked:
        return "block"
    return "missing"


def _milestone_card(
    report_path: Path,
    title: str,
    ok: bool,
    blocked: bool,
    details: list[str],
    links: list[tuple[str, Path]],
    images: list[tuple[str, Path]],
) -> str:
    link_html = "".join(
        f'<li><a href="{html.escape(_rel(report_path, path))}">{html.escape(label)}</a></li>'
        for label, path in links
        if path.exists()
    )
    detail_html = "".join(f"<li>{html.escape(item)}</li>" for item in details)
    image_html = "".join(
        f"""
        <figure>
          <img src="{html.escape(_rel(report_path, path))}" alt="{html.escape(label)}">
          <figcaption>{html.escape(label)}</figcaption>
        </figure>
        """
        for label, path in images
        if path.exists()
    )
    state = _status(ok, blocked)
    css = _class(ok, blocked)
    return f"""
    <section class="card {css}">
      <div class="state">{state}</div>
      <h2>{html.escape(title)}</h2>
      <ul>{detail_html}</ul>
      <ul class="links">{link_html}</ul>
      <div class="images">{image_html}</div>
    </section>
    """


def build_dashboard(root: Path, live_root: Path, output_path: Path) -> None:
    static_root = root / "local_html_validation"
    trajectory_root = root / "local_trajectory_validation"
    gallery_root = root / "gallery_builder_probe"
    pose_dict_root = root / "local_pose_dict_validation"
    sensor_camera_root = root / "local_sensor_camera_validation"
    multi_camera_root = root / "local_multi_camera_validation"
    edge_case_root = root / "local_edge_case_validation"
    comparison_root = root / "raw_vs_overlay_comparison"
    figure_pack_root = root / "paper_figure_pack"
    zoom_root = root / "zoomed_raw_vs_overlay"
    diverse_root = root / "diverse_object_projection"
    center_bias_root = root / "center_bias_audit"
    runpod_lifecycle_root = root / "runpod_lifecycle_decision"
    artifact_audit_root = root / "artifact_completeness_audit"
    policy_input_root = root / "policy_input_readiness"
    encoding_ablation_root = root / "encoding_ablation"
    temporal_consistency_root = root / "temporal_consistency"
    actual_sim_frame_root = root / "actual_sim_frame_overlay_fallback"
    actual_sim_heuristic_root = root / "actual_sim_visual_heuristic_overlay"
    actual_sim_policy_input_root = root / "actual_sim_heuristic_policy_input"
    actual_sim_temporal_root = root / "actual_sim_heuristic_temporal"
    actual_sim_preflight_root = root / "actual_sim_evidence_preflight"
    actual_sim_only_root = root / "actual_sim_only_dashboard"
    actual_sim_claim_matrix_root = root / "actual_sim_claim_matrix"
    actual_sim_true_oracle_root = root / "live_true_oracle_projection"
    actual_sim_true_oracle_gap_root = root / "actual_sim_true_oracle_readiness_gap"
    actual_rgb_synthetic_metadata_codepath_root = root / "actual_rgb_synthetic_metadata_true_oracle_codepath"
    actual_sim_true_oracle_probe_blocker_root = root / "actual_sim_true_oracle_probe_blocker"
    actual_sim_true_oracle_remote_handoff_root = root / "actual_sim_true_oracle_remote_handoff"
    actual_sim_true_oracle_two_stage_root = root / "actual_sim_true_oracle_two_stage_result"
    paper_progress_ledger_root = root / "paper_progress_ledger"
    experiment_matrix_result_root = root / "agentic_smolvla_experiment_matrix_result"
    agentic_retry_schema_root = root / "agentic_retry_schema_readiness"
    renderer_env_preflight_root = root / "renderer_env_preflight"
    remote_true_oracle_evidence_pack_root = root / "remote_true_oracle_evidence_pack"
    paper_claim_gate_root = root / "paper_claim_gate"
    paper_outline_root = root / "agentic_smolvla_paper_outline"
    next_action_command_center_root = root / "next_action_command_center"

    static_json = _load_json(static_root / "validation_report.json")
    trajectory_json = _load_json(trajectory_root / "validation_report.json")
    gallery_manifest = _load_json(gallery_root / "overlay_gallery_manifest.json")
    pose_dict_json = _load_json(pose_dict_root / "validation_report.json")
    sensor_camera_json = _load_json(sensor_camera_root / "validation_report.json")
    multi_camera_json = _load_json(multi_camera_root / "validation_report.json")
    edge_case_json = _load_json(edge_case_root / "validation_report.json")
    comparison_manifest = _load_json(comparison_root / "comparison_manifest.json")
    figure_pack_manifest = _load_json(figure_pack_root / "figure_pack_manifest.json")
    zoom_manifest = _load_json(zoom_root / "zoom_manifest.json")
    diverse_manifest = _load_json(diverse_root / "diverse_object_manifest.json")
    center_bias_audit = _load_json(center_bias_root / "center_bias_audit.json")
    runpod_lifecycle_manifest = _load_json(runpod_lifecycle_root / "runpod_lifecycle_decision.json")
    artifact_audit_manifest = _load_json(artifact_audit_root / "artifact_completeness_audit.json")
    policy_input_manifest = _load_json(policy_input_root / "policy_input_manifest.json")
    encoding_ablation_manifest = _load_json(encoding_ablation_root / "encoding_ablation_manifest.json")
    temporal_consistency_manifest = _load_json(temporal_consistency_root / "temporal_consistency_manifest.json")
    actual_sim_frame_manifest = _load_json(actual_sim_frame_root / "actual_sim_frame_overlay_manifest.json")
    actual_sim_heuristic_manifest = _load_json(actual_sim_heuristic_root / "actual_sim_visual_heuristic_manifest.json")
    actual_sim_policy_input_manifest = _load_json(
        actual_sim_policy_input_root / "actual_sim_heuristic_policy_input_manifest.json"
    )
    actual_sim_temporal_manifest = _load_json(actual_sim_temporal_root / "actual_sim_heuristic_temporal_manifest.json")
    actual_sim_preflight_manifest = _load_json(actual_sim_preflight_root / "actual_sim_evidence_preflight_manifest.json")
    actual_sim_only_manifest = _load_json(actual_sim_only_root / "actual_sim_only_dashboard_manifest.json")
    actual_sim_claim_matrix_manifest = _load_json(actual_sim_claim_matrix_root / "actual_sim_claim_matrix_manifest.json")
    actual_sim_true_oracle_manifest = _load_json(actual_sim_true_oracle_root / "actual_sim_true_oracle_report_manifest.json")
    actual_sim_true_oracle_gap_manifest = _load_json(
        actual_sim_true_oracle_gap_root / "actual_sim_true_oracle_readiness_gap_manifest.json"
    )
    actual_rgb_synthetic_metadata_codepath_manifest = _load_json(
        actual_rgb_synthetic_metadata_codepath_root / "actual_rgb_synthetic_metadata_codepath_manifest.json"
    )
    actual_sim_true_oracle_probe_blocker_manifest = _load_json(
        actual_sim_true_oracle_probe_blocker_root / "actual_sim_true_oracle_probe_blocker_manifest.json"
    )
    actual_sim_true_oracle_remote_handoff_manifest = _load_json(
        actual_sim_true_oracle_remote_handoff_root / "actual_sim_true_oracle_remote_handoff_manifest.json"
    )
    actual_sim_true_oracle_two_stage_manifest = _load_json(
        actual_sim_true_oracle_two_stage_root / "actual_sim_true_oracle_two_stage_result_manifest.json"
    )
    paper_progress_ledger_manifest = _load_json(paper_progress_ledger_root / "paper_progress_ledger_manifest.json")
    experiment_matrix_result_manifest = _load_json(
        experiment_matrix_result_root / "agentic_smolvla_experiment_matrix_result_manifest.json"
    )
    agentic_retry_schema_manifest = _load_json(agentic_retry_schema_root / "agentic_retry_schema_readiness_manifest.json")
    renderer_env_preflight_manifest = _load_json(
        renderer_env_preflight_root / "renderer_env_preflight_report_manifest.json"
    )
    remote_true_oracle_evidence_pack_manifest = _load_json(
        remote_true_oracle_evidence_pack_root / "remote_true_oracle_evidence_pack_manifest.json"
    )
    paper_claim_gate_manifest = _load_json(paper_claim_gate_root / "paper_claim_gate_manifest.json")
    paper_outline_manifest = _load_json(paper_outline_root / "agentic_smolvla_paper_outline_manifest.json")
    next_action_command_center_manifest = _load_json(
        next_action_command_center_root / "next_action_command_center_manifest.json"
    )
    live_report = _load_json(live_root / "checkpoint_report.json")
    blocker_path = live_root / "maniskill_blocker.md"

    static_ok = static_json.get("passed") == static_json.get("total") and static_json.get("total", 0) >= 37
    trajectory_ok = (
        trajectory_json.get("passed") == trajectory_json.get("total")
        and trajectory_json.get("total", 0) >= 57
        and _count_images(trajectory_root / "projection_trajectory") >= 20
    )
    gallery_ok = (
        int(gallery_manifest.get("frame_count", 0)) >= 10
        and (gallery_root / "overlay_gallery.html").exists()
        and (gallery_root / "contact_sheet.png").exists()
        and (gallery_root / "overlay_sequence.gif").exists()
    )
    pose_dict_ok = (
        pose_dict_json.get("passed") == pose_dict_json.get("total")
        and pose_dict_json.get("total", 0) >= 77
        and _count_images(pose_dict_root / "dict_pose_trajectory") >= 20
    )
    sensor_camera_ok = (
        sensor_camera_json.get("passed") == sensor_camera_json.get("total")
        and sensor_camera_json.get("total", 0) >= 97
        and _count_images(sensor_camera_root / "sensor_data_camera_trajectory") >= 20
    )
    multi_camera_ok = (
        multi_camera_json.get("passed") == multi_camera_json.get("total")
        and multi_camera_json.get("total", 0) >= 117
        and _count_images(multi_camera_root / "multi_camera_trajectory") >= 20
    )
    edge_case_ok = (
        edge_case_json.get("passed") == edge_case_json.get("total")
        and edge_case_json.get("total", 0) >= 137
        and _count_images(edge_case_root / "projection_edge_cases") >= 20
    )
    comparison_ok = (
        comparison_manifest.get("status") == "passed"
        and int(comparison_manifest.get("sample_count", 0)) >= 20
        and (comparison_root / "comparison_report.html").exists()
        and (comparison_root / "comparison_contact_sheet.png").exists()
    )
    figure_pack_groups = figure_pack_manifest.get("sample_groups", [])
    figure_pack_ok = (
        figure_pack_manifest.get("status") == "passed_static_blocked_live"
        and len(figure_pack_groups) >= 9
        and (figure_pack_root / "figure_pack.html").exists()
    )
    zoom_ok = (
        zoom_manifest.get("status") == "passed"
        and int(zoom_manifest.get("panel_count", 0)) >= 10
        and (zoom_root / "zoom_report.html").exists()
        and (zoom_root / "zoom_contact_sheet.png").exists()
    )
    diverse_ok = (
        diverse_manifest.get("status") == "passed"
        and int(diverse_manifest.get("episode_count", 0)) >= 30
        and int(diverse_manifest.get("non_center_episode_count", 0)) >= 20
        and (diverse_root / "diverse_object_report.html").exists()
        and (diverse_root / "diverse_object_full_contact_sheet.png").exists()
        and (diverse_root / "diverse_object_zoom_contact_sheet.png").exists()
    )
    center_bias_ok = (
        center_bias_audit.get("status") == "passed"
        and int(center_bias_audit.get("non_center_count", 0)) >= 20
        and float(center_bias_audit.get("max_projection_error_px", 999.0)) <= 1.0
        and (center_bias_root / "center_bias_audit.html").exists()
        and (center_bias_root / "center_distance_distribution.png").exists()
    )
    runpod_lifecycle_ok = (
        runpod_lifecycle_manifest.get("status") == "passed"
        and runpod_lifecycle_manifest.get("decision") == "stop_overlay_probe_pods_keep_baseline_parity"
        and int(runpod_lifecycle_manifest.get("sample_count", 0)) >= 10
        and (runpod_lifecycle_root / "runpod_lifecycle_decision.html").exists()
    )
    artifact_audit_ok = (
        artifact_audit_manifest.get("status") == "passed_static_blocked_live"
        and int(artifact_audit_manifest.get("failed_count", 1)) == 0
        and int(artifact_audit_manifest.get("passed_count", 0)) >= 12
        and (artifact_audit_root / "artifact_completeness_audit.html").exists()
    )
    policy_input_ok = (
        policy_input_manifest.get("status") == "passed"
        and int(policy_input_manifest.get("sample_count", 0)) >= 10
        and policy_input_manifest.get("feature_shape") == [1, 3, 224, 224]
        and (policy_input_root / "policy_input_report.html").exists()
        and (policy_input_root / "policy_input_contact_sheet.png").exists()
    )
    encoding_ablation_ok = (
        encoding_ablation_manifest.get("status") == "passed"
        and int(encoding_ablation_manifest.get("sample_count", 0)) >= 10
        and int(encoding_ablation_manifest.get("variant_count", 0)) >= 4
        and (encoding_ablation_root / "encoding_ablation_report.html").exists()
        and (encoding_ablation_root / "encoding_ablation_contact_sheet.png").exists()
    )
    temporal_summary = temporal_consistency_manifest.get("summary", {})
    temporal_consistency_ok = (
        temporal_consistency_manifest.get("status") == "passed"
        and int(temporal_summary.get("frame_count", 0)) >= 20
        and bool(temporal_summary.get("all_projected", False))
        and float(temporal_summary.get("max_projection_error_px", 999.0)) <= 1.0
        and (temporal_consistency_root / "temporal_consistency_report.html").exists()
        and (temporal_consistency_root / "temporal_consistency_contact_sheet.png").exists()
        and (temporal_consistency_root / "temporal_consistency_sequence.gif").exists()
    )
    actual_sim_frame_ok = (
        actual_sim_frame_manifest.get("status") == "passed_fallback_only"
        and actual_sim_frame_manifest.get("source_type") == "actual_sim_rgb_fallback"
        and actual_sim_frame_manifest.get("real_sim_episode") is True
        and actual_sim_frame_manifest.get("true_oracle_projection") is False
        and int(actual_sim_frame_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_frame_root / "actual_sim_frame_overlay_report.html").exists()
        and (actual_sim_frame_root / "actual_sim_frame_overlay_contact_sheet.png").exists()
    )
    actual_sim_heuristic_ok = (
        actual_sim_heuristic_manifest.get("status") == "passed"
        and actual_sim_heuristic_manifest.get("source_type") == "actual_sim_rgb_visual_heuristic"
        and actual_sim_heuristic_manifest.get("real_sim_episode") is True
        and actual_sim_heuristic_manifest.get("true_oracle_projection") is False
        and int(actual_sim_heuristic_manifest.get("sample_count", 0)) >= 10
        and int(actual_sim_heuristic_manifest.get("non_center_heuristic_count", 0)) >= 5
        and (actual_sim_heuristic_root / "actual_sim_visual_heuristic_report.html").exists()
        and (actual_sim_heuristic_root / "actual_sim_visual_heuristic_contact_sheet.png").exists()
    )
    actual_sim_policy_input_ok = (
        actual_sim_policy_input_manifest.get("status") == "passed"
        and actual_sim_policy_input_manifest.get("source_type") == "actual_sim_rgb_visual_heuristic_policy_input"
        and actual_sim_policy_input_manifest.get("real_sim_episode") is True
        and actual_sim_policy_input_manifest.get("true_oracle_projection") is False
        and actual_sim_policy_input_manifest.get("feature_shape") == [1, 3, 224, 224]
        and int(actual_sim_policy_input_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_policy_input_root / "actual_sim_heuristic_policy_input_report.html").exists()
        and (actual_sim_policy_input_root / "actual_sim_heuristic_policy_input_contact_sheet.png").exists()
    )
    actual_sim_temporal_summary = actual_sim_temporal_manifest.get("summary", {})
    actual_sim_temporal_ok = (
        actual_sim_temporal_manifest.get("status") == "passed"
        and actual_sim_temporal_manifest.get("source_type") == "actual_sim_rgb_visual_heuristic_temporal"
        and actual_sim_temporal_manifest.get("real_sim_episode") is True
        and actual_sim_temporal_manifest.get("true_oracle_projection") is False
        and int(actual_sim_temporal_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_temporal_root / "actual_sim_heuristic_temporal_report.html").exists()
        and (actual_sim_temporal_root / "actual_sim_heuristic_temporal_contact_sheet.png").exists()
        and (actual_sim_temporal_root / "actual_sim_heuristic_temporal_sequence.gif").exists()
    )
    actual_sim_preflight_ok = (
        actual_sim_preflight_manifest.get("status")
        in {"passed_rgb_only_true_oracle_blocked", "passed_true_oracle_ready"}
        and actual_sim_preflight_manifest.get("source_type") == "actual_sim_artifact_preflight"
        and int(actual_sim_preflight_manifest.get("sample_count", 0)) >= 10
        and int(actual_sim_preflight_manifest.get("frame_rich_checkpoint_count", 0)) >= 1
        and (actual_sim_preflight_root / "actual_sim_evidence_preflight.html").exists()
        and (actual_sim_preflight_root / "actual_sim_evidence_preflight_contact_sheet.png").exists()
    )
    actual_sim_only_ok = (
        actual_sim_only_manifest.get("status")
        in {"passed_actual_sim_true_oracle_blocked", "passed_true_oracle_ready"}
        and actual_sim_only_manifest.get("source_type") == "actual_sim_only_dashboard"
        and actual_sim_only_manifest.get("synthetic_included") is False
        and int(actual_sim_only_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_only_root / "actual_sim_only_dashboard.html").exists()
    )
    actual_sim_claim_matrix_ok = (
        actual_sim_claim_matrix_manifest.get("status")
        in {"passed_claims_with_true_oracle_blocked", "passed_true_oracle_ready"}
        and actual_sim_claim_matrix_manifest.get("source_type") == "actual_sim_claim_matrix"
        and actual_sim_claim_matrix_manifest.get("synthetic_included") is False
        and int(actual_sim_claim_matrix_manifest.get("sample_count", 0)) >= 10
        and int(actual_sim_claim_matrix_manifest.get("supported_claim_count", 0)) >= 5
        and (actual_sim_claim_matrix_root / "actual_sim_claim_matrix.html").exists()
    )
    actual_sim_true_oracle_ok = (
        actual_sim_true_oracle_manifest.get("status") == "passed"
        and actual_sim_true_oracle_manifest.get("source_type") == "actual_sim_true_oracle_projection"
        and actual_sim_true_oracle_manifest.get("true_oracle_projection") is True
        and int(actual_sim_true_oracle_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_true_oracle_root / "actual_sim_true_oracle_report.html").exists()
    )
    actual_sim_true_oracle_blocked = (
        actual_sim_true_oracle_manifest.get("status") == "blocked"
        and (actual_sim_true_oracle_root / "actual_sim_true_oracle_report.html").exists()
    )
    actual_sim_true_oracle_gap_ok = (
        actual_sim_true_oracle_gap_manifest.get("status") == "blocked_true_oracle_metadata_missing"
        and actual_sim_true_oracle_gap_manifest.get("source_type") == "actual_sim_true_oracle_readiness_gap"
        and actual_sim_true_oracle_gap_manifest.get("real_sim_episode") is True
        and actual_sim_true_oracle_gap_manifest.get("true_oracle_projection") is False
        and int(actual_sim_true_oracle_gap_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_true_oracle_gap_root / "actual_sim_true_oracle_readiness_gap.html").exists()
        and (actual_sim_true_oracle_gap_root / "actual_sim_true_oracle_readiness_gap_contact_sheet.jpg").exists()
    )
    actual_rgb_synthetic_metadata_codepath_ok = (
        actual_rgb_synthetic_metadata_codepath_manifest.get("status") == "passed"
        and actual_rgb_synthetic_metadata_codepath_manifest.get("source_type")
        == "actual_sim_rgb_synthetic_metadata_codepath_diagnostic"
        and actual_rgb_synthetic_metadata_codepath_manifest.get("real_sim_episode") is True
        and actual_rgb_synthetic_metadata_codepath_manifest.get("synthetic_metadata") is True
        and actual_rgb_synthetic_metadata_codepath_manifest.get("true_oracle_projection") is False
        and actual_rgb_synthetic_metadata_codepath_manifest.get("codepath_projection_ready") is True
        and int(actual_rgb_synthetic_metadata_codepath_manifest.get("sample_count", 0)) >= 10
        and (
            actual_rgb_synthetic_metadata_codepath_root / "actual_rgb_synthetic_metadata_codepath_report.html"
        ).exists()
        and (
            actual_rgb_synthetic_metadata_codepath_root
            / "actual_rgb_synthetic_metadata_codepath_contact_sheet.jpg"
        ).exists()
    )
    actual_sim_true_oracle_probe_blocker_ok = (
        actual_sim_true_oracle_probe_blocker_manifest.get("status") == "blocked_renderer_incompatible_driver"
        and actual_sim_true_oracle_probe_blocker_manifest.get("source_type") == "actual_sim_true_oracle_probe_blocker"
        and actual_sim_true_oracle_probe_blocker_manifest.get("true_oracle_projection") is False
        and int(actual_sim_true_oracle_probe_blocker_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_true_oracle_probe_blocker_root / "actual_sim_true_oracle_probe_blocker.html").exists()
    )
    actual_sim_true_oracle_remote_handoff_ok = (
        actual_sim_true_oracle_remote_handoff_manifest.get("status") == "passed"
        and actual_sim_true_oracle_remote_handoff_manifest.get("source_type")
        == "actual_sim_true_oracle_remote_handoff"
        and actual_sim_true_oracle_remote_handoff_manifest.get("true_oracle_projection") is False
        and int(actual_sim_true_oracle_remote_handoff_manifest.get("sample_count", 0)) >= 10
        and (
            actual_sim_true_oracle_remote_handoff_root / "actual_sim_true_oracle_remote_handoff_report.html"
        ).exists()
    )
    actual_sim_true_oracle_two_stage_ok = (
        actual_sim_true_oracle_two_stage_manifest.get("status") == "passed"
        and actual_sim_true_oracle_two_stage_manifest.get("source_type")
        == "actual_sim_true_oracle_two_stage_result"
        and actual_sim_true_oracle_two_stage_manifest.get("true_oracle_projection") is True
        and int(actual_sim_true_oracle_two_stage_manifest.get("sample_count", 0)) >= 10
        and (actual_sim_true_oracle_two_stage_root / "actual_sim_true_oracle_two_stage_result.html").exists()
    )
    actual_sim_true_oracle_two_stage_blocked = (
        actual_sim_true_oracle_two_stage_manifest.get("status") in {"blocked_at_probe", "blocked_at_policy"}
        and (actual_sim_true_oracle_two_stage_root / "actual_sim_true_oracle_two_stage_result.html").exists()
    )
    paper_progress_ledger_ok = (
        paper_progress_ledger_manifest.get("status") == "passed"
        and paper_progress_ledger_manifest.get("source_type") == "paper_progress_ledger"
        and int(paper_progress_ledger_manifest.get("sample_count", 0)) >= 10
        and (paper_progress_ledger_root / "paper_progress_ledger_report.html").exists()
    )
    experiment_matrix_result_ok = (
        experiment_matrix_result_manifest.get("status") == "passed_schema_pending_results"
        and experiment_matrix_result_manifest.get("source_type") == "agentic_smolvla_experiment_matrix_result"
        and int(experiment_matrix_result_manifest.get("sample_count", 0)) >= 10
        and int(experiment_matrix_result_manifest.get("condition_count", 0)) >= 6
        and (experiment_matrix_result_root / "agentic_smolvla_experiment_matrix_result.html").exists()
    )
    agentic_retry_schema_ok = (
        agentic_retry_schema_manifest.get("status") == "passed_schema_ready_nonpaper_benchmark"
        and agentic_retry_schema_manifest.get("source_type") == "agentic_retry_schema_readiness"
        and int(agentic_retry_schema_manifest.get("sample_count", 0)) >= 10
        and (agentic_retry_schema_root / "agentic_retry_schema_readiness.html").exists()
    )
    renderer_env_preflight_ok = (
        renderer_env_preflight_manifest.get("source_type") == "renderer_env_preflight_report"
        and int(renderer_env_preflight_manifest.get("sample_count", 0)) >= 10
        and (renderer_env_preflight_root / "renderer_env_preflight_report.html").exists()
    )
    remote_true_oracle_evidence_pack_ok = (
        remote_true_oracle_evidence_pack_manifest.get("status") == "passed_handoff_ready"
        and remote_true_oracle_evidence_pack_manifest.get("source_type") == "remote_true_oracle_evidence_pack"
        and int(remote_true_oracle_evidence_pack_manifest.get("sample_count", 0)) >= 10
        and (remote_true_oracle_evidence_pack_root / "remote_true_oracle_evidence_pack_report.html").exists()
    )
    paper_claim_gate_ok = (
        paper_claim_gate_manifest.get("status") == "passed_with_blocked_claims"
        and paper_claim_gate_manifest.get("source_type") == "paper_claim_gate"
        and int(paper_claim_gate_manifest.get("sample_count", 0)) >= 10
        and paper_claim_gate_manifest.get("true_oracle_projection_claim_allowed") is False
        and paper_claim_gate_manifest.get("agentic_success_claim_allowed") is False
        and (paper_claim_gate_root / "paper_claim_gate_report.html").exists()
    )
    paper_outline_ok = (
        paper_outline_manifest.get("status") == "passed_claim_safe_outline"
        and paper_outline_manifest.get("source_type") == "agentic_smolvla_paper_outline"
        and int(paper_outline_manifest.get("sample_count", 0)) >= 10
        and paper_outline_manifest.get("true_oracle_projection") is False
        and paper_outline_manifest.get("agentic_success_claim") is False
        and (paper_outline_root / "agentic_smolvla_paper_outline_report.html").exists()
    )
    next_action_command_center_ok = (
        next_action_command_center_manifest.get("status") == "passed_next_action_ready"
        and next_action_command_center_manifest.get("source_type") == "next_action_command_center"
        and int(next_action_command_center_manifest.get("sample_count", 0)) >= 10
        and next_action_command_center_manifest.get("true_oracle_projection") is False
        and next_action_command_center_manifest.get("agentic_success_claim") is False
        and (next_action_command_center_root / "next_action_command_center_report.html").exists()
    )
    live_ok = (
        live_report.get("status") == "passed"
        and not blocker_path.exists()
        and _count_images(live_root) >= 10
    )
    live_blocked = blocker_path.exists() or live_report.get("status") == "failed"

    cards = [
        _milestone_card(
            output_path,
            "Static projection/rendering/sim-frame validation",
            static_ok,
            False,
            [
                f"Passed: {static_json.get('passed', 'n/a')}/{static_json.get('total', 'n/a')}",
                "Acceptance: all cases pass and at least 37 total cases.",
            ],
            [
                ("Static HTML report", static_root / "validation_report.html"),
                ("Static JSON report", static_root / "validation_report.json"),
            ],
            [
                ("Projection contact sheet", static_root / "contact_sheet_projection.png"),
                ("Rendering contact sheet", static_root / "contact_sheet_rendering.png"),
                ("Sim-frame overlay contact sheet", static_root / "contact_sheet_sim_frame_overlay.png"),
            ],
        ),
        _milestone_card(
            output_path,
            "Moving-object oracle projection trajectory",
            trajectory_ok,
            False,
            [
                f"Passed: {trajectory_json.get('passed', 'n/a')}/{trajectory_json.get('total', 'n/a')}",
                f"Trajectory images: {_count_images(trajectory_root / 'projection_trajectory')}",
                "Acceptance: 20 trajectory frames and projected oracle point error <= 1 px per frame.",
            ],
            [
                ("Trajectory HTML report", trajectory_root / "validation_report.html"),
                ("Trajectory JSON report", trajectory_root / "validation_report.json"),
                ("Trajectory GIF", trajectory_root / "projection_trajectory.gif"),
            ],
            [
                ("Trajectory contact sheet", trajectory_root / "contact_sheet_projection_trajectory.png"),
            ],
        ),
        _milestone_card(
            output_path,
            "Reusable overlay gallery postprocessor",
            gallery_ok,
            False,
            [
                f"Gallery frames: {gallery_manifest.get('frame_count', 'n/a')}",
                "Acceptance: >=10 frames, HTML, contact sheet, GIF, manifest.",
            ],
            [
                ("Gallery HTML", gallery_root / "overlay_gallery.html"),
                ("Gallery manifest", gallery_root / "overlay_gallery_manifest.json"),
                ("Gallery GIF", gallery_root / "overlay_sequence.gif"),
            ],
            [
                ("Gallery contact sheet", gallery_root / "contact_sheet.png"),
            ],
        ),
        _milestone_card(
            output_path,
            "Simulator-style pose dict projection",
            pose_dict_ok,
            False,
            [
                f"Passed: {pose_dict_json.get('passed', 'n/a')}/{pose_dict_json.get('total', 'n/a')}",
                f"Dict-pose trajectory images: {_count_images(pose_dict_root / 'dict_pose_trajectory')}",
                "Acceptance: {p: xyz} and {position: xyz} pose styles project in 20 moving-object frames.",
            ],
            [
                ("Pose-dict HTML report", pose_dict_root / "validation_report.html"),
                ("Pose-dict JSON report", pose_dict_root / "validation_report.json"),
                ("Pose-dict GIF", pose_dict_root / "dict_pose_trajectory.gif"),
            ],
            [
                ("Pose-dict contact sheet", pose_dict_root / "contact_sheet_dict_pose_trajectory.png"),
            ],
        ),
        _milestone_card(
            output_path,
            "Sensor-data camera parameter projection",
            sensor_camera_ok,
            False,
            [
                f"Passed: {sensor_camera_json.get('passed', 'n/a')}/{sensor_camera_json.get('total', 'n/a')}",
                f"Sensor-data camera trajectory images: {_count_images(sensor_camera_root / 'sensor_data_camera_trajectory')}",
                "Acceptance: camera intrinsics/extrinsics embedded under sensor_data[cam] project in 20 moving-object frames.",
            ],
            [
                ("Sensor-data camera HTML report", sensor_camera_root / "validation_report.html"),
                ("Sensor-data camera JSON report", sensor_camera_root / "validation_report.json"),
                ("Sensor-data camera GIF", sensor_camera_root / "sensor_data_camera_trajectory.gif"),
            ],
            [
                (
                    "Sensor-data camera contact sheet",
                    sensor_camera_root / "contact_sheet_sensor_data_camera_trajectory.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Multi-camera preferred/fallback projection",
            multi_camera_ok,
            False,
            [
                f"Passed: {multi_camera_json.get('passed', 'n/a')}/{multi_camera_json.get('total', 'n/a')}",
                f"Multi-camera trajectory images: {_count_images(multi_camera_root / 'multi_camera_trajectory')}",
                "Acceptance: preferred base_camera and fallback aux_camera both project in 20 moving-object frames.",
            ],
            [
                ("Multi-camera HTML report", multi_camera_root / "validation_report.html"),
                ("Multi-camera JSON report", multi_camera_root / "validation_report.json"),
                ("Multi-camera GIF", multi_camera_root / "multi_camera_trajectory.gif"),
            ],
            [
                (
                    "Multi-camera contact sheet",
                    multi_camera_root / "contact_sheet_multi_camera_trajectory.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Projection edge-case fallback safety",
            edge_case_ok,
            False,
            [
                f"Passed: {edge_case_json.get('passed', 'n/a')}/{edge_case_json.get('total', 'n/a')}",
                f"Edge-case images: {_count_images(edge_case_root / 'projection_edge_cases')}",
                "Acceptance: behind-camera, out-of-frame, missing camera params, malformed matrices, and invalid pose all render center fallback.",
            ],
            [
                ("Edge-case HTML report", edge_case_root / "validation_report.html"),
                ("Edge-case JSON report", edge_case_root / "validation_report.json"),
                ("Edge-case GIF", edge_case_root / "projection_edge_cases.gif"),
            ],
            [
                (
                    "Edge-case fallback contact sheet",
                    edge_case_root / "contact_sheet_projection_edge_cases.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Raw-vs-oracle-overlay input comparison",
            comparison_ok,
            False,
            [
                f"Comparison samples: {comparison_manifest.get('sample_count', 'n/a')}",
                "Acceptance: >=20 baseline-vs-overlay pairs, HTML report, and contact sheet.",
            ],
            [
                ("Comparison HTML report", comparison_root / "comparison_report.html"),
                ("Comparison manifest", comparison_root / "comparison_manifest.json"),
            ],
            [
                (
                    "Raw-vs-overlay contact sheet",
                    comparison_root / "comparison_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Paper-facing figure pack",
            figure_pack_ok,
            False,
            [
                f"Figure groups: {len(figure_pack_groups)}",
                f"Manifest status: {figure_pack_manifest.get('status', 'missing')}",
                "Acceptance: figure pack HTML plus >=9 claim-labeled evidence sections.",
            ],
            [
                ("Figure pack HTML", figure_pack_root / "figure_pack.html"),
                ("Figure pack manifest", figure_pack_root / "figure_pack_manifest.json"),
            ],
            [
                (
                    "Raw-vs-overlay paper figure",
                    comparison_root / "comparison_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Zoomed raw-vs-overlay representative figures",
            zoom_ok,
            False,
            [
                f"Zoom panels: {zoom_manifest.get('panel_count', 'n/a')}",
                "Acceptance: >=10 marker-centered enlarged raw-vs-overlay panels, HTML report, and contact sheet.",
            ],
            [
                ("Zoomed HTML report", zoom_root / "zoom_report.html"),
                ("Zoom manifest", zoom_root / "zoom_manifest.json"),
            ],
            [
                (
                    "Zoomed raw-vs-overlay contact sheet",
                    zoom_root / "zoom_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Diverse non-center object projection",
            diverse_ok,
            False,
            [
                f"Episodes: {diverse_manifest.get('episode_count', 'n/a')}",
                f"Non-center episodes: {diverse_manifest.get('non_center_episode_count', 'n/a')}",
                "Acceptance: >=30 episodes, >=20 non-center targets, projected_object_pose for all, <=1 px projection error.",
            ],
            [
                ("Diverse object HTML report", diverse_root / "diverse_object_report.html"),
                ("Diverse object manifest", diverse_root / "diverse_object_manifest.json"),
            ],
            [
                (
                    "Diverse full-frame contact sheet",
                    diverse_root / "diverse_object_full_contact_sheet.png",
                ),
                (
                    "Diverse zoom contact sheet",
                    diverse_root / "diverse_object_zoom_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Center-bias audit for diverse objects",
            center_bias_ok,
            False,
            [
                f"Non-center episodes: {center_bias_audit.get('non_center_count', 'n/a')}",
                f"Max projection error px: {center_bias_audit.get('max_projection_error_px', 'n/a')}",
                "Acceptance: >=20 non-center targets and <=1 px max projection error.",
            ],
            [
                ("Center-bias audit HTML", center_bias_root / "center_bias_audit.html"),
                ("Center-bias audit JSON", center_bias_root / "center_bias_audit.json"),
            ],
            [
                (
                    "Distance-from-center distribution",
                    center_bias_root / "center_distance_distribution.png",
                ),
                (
                    "Target object distribution",
                    center_bias_root / "object_distribution.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "RunPod lifecycle decision",
            runpod_lifecycle_ok,
            False,
            [
                f"Decision: {runpod_lifecycle_manifest.get('decision', 'missing')}",
                f"Representative samples: {runpod_lifecycle_manifest.get('sample_count', 'n/a')}",
                "Acceptance: overlay/probe stop decision, baseline/parity keep-safe decision, and >=10 visual samples.",
            ],
            [
                ("RunPod lifecycle decision HTML", runpod_lifecycle_root / "runpod_lifecycle_decision.html"),
                ("RunPod lifecycle decision JSON", runpod_lifecycle_root / "runpod_lifecycle_decision.json"),
            ],
            [
                (
                    "Diverse full-frame evidence",
                    diverse_root / "diverse_object_full_contact_sheet.png",
                ),
                (
                    "Diverse zoom evidence",
                    diverse_root / "diverse_object_zoom_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Artifact completeness audit",
            artifact_audit_ok,
            False,
            [
                f"Passed milestones: {artifact_audit_manifest.get('passed_count', 'n/a')}",
                f"Expected blocked milestones: {artifact_audit_manifest.get('blocked_expected_count', 'n/a')}",
                f"Failed milestones: {artifact_audit_manifest.get('failed_count', 'n/a')}",
                "Acceptance: every completed milestone has HTML plus required sample count; live gate is explicitly blocked.",
            ],
            [
                ("Artifact completeness audit HTML", artifact_audit_root / "artifact_completeness_audit.html"),
                ("Artifact completeness audit JSON", artifact_audit_root / "artifact_completeness_audit.json"),
            ],
            [
                (
                    "Diverse full-frame evidence",
                    diverse_root / "diverse_object_full_contact_sheet.png",
                ),
                (
                    "Diverse zoom evidence",
                    diverse_root / "diverse_object_zoom_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "SmolVLA policy-input readiness",
            policy_input_ok,
            False,
            [
                f"Samples: {policy_input_manifest.get('sample_count', 'n/a')}",
                f"Feature shape: {policy_input_manifest.get('feature_shape', 'missing')}",
                f"Image conditioning: {policy_input_manifest.get('image_conditioning', 'missing')}",
                "Acceptance: >=10 raw/overlay/policy-preview samples and [1, 3, 224, 224] normalized image feature shape.",
            ],
            [
                ("Policy-input readiness HTML", policy_input_root / "policy_input_report.html"),
                ("Policy-input readiness manifest", policy_input_root / "policy_input_manifest.json"),
            ],
            [
                (
                    "Policy-input readiness contact sheet",
                    policy_input_root / "policy_input_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Oracle point encoding ablation",
            encoding_ablation_ok,
            False,
            [
                f"Samples: {encoding_ablation_manifest.get('sample_count', 'n/a')}",
                f"Variants: {encoding_ablation_manifest.get('variants', 'missing')}",
                "Acceptance: >=10 samples and >=4 visual prompt variants for the same oracle point.",
            ],
            [
                ("Encoding ablation HTML", encoding_ablation_root / "encoding_ablation_report.html"),
                ("Encoding ablation manifest", encoding_ablation_root / "encoding_ablation_manifest.json"),
            ],
            [
                (
                    "Encoding ablation contact sheet",
                    encoding_ablation_root / "encoding_ablation_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Oracle point temporal consistency",
            temporal_consistency_ok,
            False,
            [
                f"Source type: {temporal_consistency_manifest.get('source_type', 'missing')}",
                f"Real sim episode: {temporal_consistency_manifest.get('real_sim_episode', 'missing')}",
                f"Frames: {temporal_summary.get('frame_count', 'n/a')}",
                f"Max projection error px: {temporal_summary.get('max_projection_error_px', 'n/a')}",
                "Acceptance: >=20 synthetic trajectory frames, all projected_object_pose, <=1 px max projection error.",
            ],
            [
                ("Temporal consistency HTML", temporal_consistency_root / "temporal_consistency_report.html"),
                ("Temporal consistency manifest", temporal_consistency_root / "temporal_consistency_manifest.json"),
                ("Temporal consistency GIF", temporal_consistency_root / "temporal_consistency_sequence.gif"),
            ],
            [
                (
                    "Temporal consistency contact sheet",
                    temporal_consistency_root / "temporal_consistency_contact_sheet.png",
                ),
                (
                    "Temporal projected point trail",
                    temporal_consistency_root / "temporal_consistency_trail.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual sim RGB overlay fallback",
            actual_sim_frame_ok,
            False,
            [
                f"Source type: {actual_sim_frame_manifest.get('source_type', 'missing')}",
                f"Real sim episode: {actual_sim_frame_manifest.get('real_sim_episode', 'missing')}",
                f"True oracle projection: {actual_sim_frame_manifest.get('true_oracle_projection', 'missing')}",
                f"Samples: {actual_sim_frame_manifest.get('sample_count', 'n/a')}",
                "Acceptance: >=10 actual simulator RGB frames, clearly labeled fallback overlay, no true-oracle claim.",
            ],
            [
                ("Actual sim frame overlay HTML", actual_sim_frame_root / "actual_sim_frame_overlay_report.html"),
                ("Actual sim frame overlay manifest", actual_sim_frame_root / "actual_sim_frame_overlay_manifest.json"),
                ("Actual sim frame overlay GIF", actual_sim_frame_root / "actual_sim_frame_overlay_sequence.gif"),
            ],
            [
                (
                    "Actual sim frame overlay contact sheet",
                    actual_sim_frame_root / "actual_sim_frame_overlay_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual sim visual heuristic overlay",
            actual_sim_heuristic_ok,
            False,
            [
                f"Source type: {actual_sim_heuristic_manifest.get('source_type', 'missing')}",
                f"Real sim episode: {actual_sim_heuristic_manifest.get('real_sim_episode', 'missing')}",
                f"True oracle projection: {actual_sim_heuristic_manifest.get('true_oracle_projection', 'missing')}",
                f"Samples: {actual_sim_heuristic_manifest.get('sample_count', 'n/a')}",
                f"Non-center heuristic samples: {actual_sim_heuristic_manifest.get('non_center_heuristic_count', 'n/a')}",
                "Acceptance: >=10 actual simulator RGB frames with image-only non-oracle visual point prompts.",
            ],
            [
                ("Actual sim visual heuristic HTML", actual_sim_heuristic_root / "actual_sim_visual_heuristic_report.html"),
                ("Actual sim visual heuristic manifest", actual_sim_heuristic_root / "actual_sim_visual_heuristic_manifest.json"),
            ],
            [
                (
                    "Actual sim visual heuristic contact sheet",
                    actual_sim_heuristic_root / "actual_sim_visual_heuristic_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual sim heuristic policy-input readiness",
            actual_sim_policy_input_ok,
            False,
            [
                f"Source type: {actual_sim_policy_input_manifest.get('source_type', 'missing')}",
                f"Real sim episode: {actual_sim_policy_input_manifest.get('real_sim_episode', 'missing')}",
                f"True oracle projection: {actual_sim_policy_input_manifest.get('true_oracle_projection', 'missing')}",
                f"Samples: {actual_sim_policy_input_manifest.get('sample_count', 'n/a')}",
                f"Feature shape: {actual_sim_policy_input_manifest.get('feature_shape', 'missing')}",
                "Acceptance: >=10 actual sim RGB heuristic overlays converted to SmolVLA-style [1, 3, 224, 224] image tensors.",
            ],
            [
                (
                    "Actual sim heuristic policy-input HTML",
                    actual_sim_policy_input_root / "actual_sim_heuristic_policy_input_report.html",
                ),
                (
                    "Actual sim heuristic policy-input manifest",
                    actual_sim_policy_input_root / "actual_sim_heuristic_policy_input_manifest.json",
                ),
            ],
            [
                (
                    "Actual sim heuristic policy-input contact sheet",
                    actual_sim_policy_input_root / "actual_sim_heuristic_policy_input_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual sim heuristic temporal consistency",
            actual_sim_temporal_ok,
            False,
            [
                f"Source type: {actual_sim_temporal_manifest.get('source_type', 'missing')}",
                f"Real sim episode: {actual_sim_temporal_manifest.get('real_sim_episode', 'missing')}",
                f"True oracle projection: {actual_sim_temporal_manifest.get('true_oracle_projection', 'missing')}",
                f"Frames: {actual_sim_temporal_summary.get('frame_count', 'n/a')}",
                f"Mean step delta px: {actual_sim_temporal_summary.get('mean_step_delta_px', 'n/a')}",
                "Acceptance: >=10 actual sim RGB sequence frames with image-only heuristic point trail/GIF.",
            ],
            [
                (
                    "Actual sim heuristic temporal HTML",
                    actual_sim_temporal_root / "actual_sim_heuristic_temporal_report.html",
                ),
                (
                    "Actual sim heuristic temporal manifest",
                    actual_sim_temporal_root / "actual_sim_heuristic_temporal_manifest.json",
                ),
                (
                    "Actual sim heuristic temporal GIF",
                    actual_sim_temporal_root / "actual_sim_heuristic_temporal_sequence.gif",
                ),
            ],
            [
                (
                    "Actual sim heuristic temporal contact sheet",
                    actual_sim_temporal_root / "actual_sim_heuristic_temporal_contact_sheet.png",
                ),
                (
                    "Actual sim heuristic point trail",
                    actual_sim_temporal_root / "actual_sim_heuristic_temporal_trail.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual sim evidence preflight",
            actual_sim_preflight_ok,
            False,
            [
                f"Source type: {actual_sim_preflight_manifest.get('source_type', 'missing')}",
                f"Checkpoint dirs scanned: {actual_sim_preflight_manifest.get('checkpoint_count', 'n/a')}",
                f"RGB-rich checkpoints: {actual_sim_preflight_manifest.get('frame_rich_checkpoint_count', 'n/a')}",
                f"True-oracle-ready checkpoints: {actual_sim_preflight_manifest.get('true_oracle_ready_checkpoint_count', 'n/a')}",
                "Acceptance: scan actual CP24 artifacts and show >=10 actual sim RGB representative frames.",
            ],
            [
                ("Actual sim evidence preflight HTML", actual_sim_preflight_root / "actual_sim_evidence_preflight.html"),
                ("Actual sim evidence preflight manifest", actual_sim_preflight_root / "actual_sim_evidence_preflight_manifest.json"),
            ],
            [
                (
                    "Actual sim preflight contact sheet",
                    actual_sim_preflight_root / "actual_sim_evidence_preflight_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual-sim-only dashboard",
            actual_sim_only_ok,
            False,
            [
                f"Source type: {actual_sim_only_manifest.get('source_type', 'missing')}",
                f"Synthetic included: {actual_sim_only_manifest.get('synthetic_included', 'missing')}",
                f"True-oracle ready: {actual_sim_only_manifest.get('true_oracle_ready', 'missing')}",
                f"Actual-sim sample total: {actual_sim_only_manifest.get('sample_count', 'n/a')}",
                "Acceptance: one synthetic-free HTML page that summarizes only actual CP24 simulation RGB evidence.",
            ],
            [
                ("Actual-sim-only dashboard HTML", actual_sim_only_root / "actual_sim_only_dashboard.html"),
                ("Actual-sim-only dashboard manifest", actual_sim_only_root / "actual_sim_only_dashboard_manifest.json"),
            ],
            [
                (
                    "Actual sim visual heuristic contact sheet",
                    actual_sim_heuristic_root / "actual_sim_visual_heuristic_contact_sheet.png",
                ),
                (
                    "Actual sim policy-input contact sheet",
                    actual_sim_policy_input_root / "actual_sim_heuristic_policy_input_contact_sheet.png",
                ),
                (
                    "Actual sim temporal contact sheet",
                    actual_sim_temporal_root / "actual_sim_heuristic_temporal_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual-sim claim matrix",
            actual_sim_claim_matrix_ok,
            False,
            [
                f"Source type: {actual_sim_claim_matrix_manifest.get('source_type', 'missing')}",
                f"Synthetic included: {actual_sim_claim_matrix_manifest.get('synthetic_included', 'missing')}",
                f"Supported claims: {actual_sim_claim_matrix_manifest.get('supported_claim_count', 'n/a')}/{actual_sim_claim_matrix_manifest.get('total_claim_count', 'n/a')}",
                f"True-oracle claim supported: {actual_sim_claim_matrix_manifest.get('true_oracle_claim_supported', 'missing')}",
                "Acceptance: synthetic-free claim matrix with true-oracle claim explicitly blocked until evidence exists.",
            ],
            [
                ("Actual-sim claim matrix HTML", actual_sim_claim_matrix_root / "actual_sim_claim_matrix.html"),
                ("Actual-sim claim matrix manifest", actual_sim_claim_matrix_root / "actual_sim_claim_matrix_manifest.json"),
            ],
            [
                (
                    "Actual sim visual heuristic contact sheet",
                    actual_sim_heuristic_root / "actual_sim_visual_heuristic_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Next action command center",
            next_action_command_center_ok,
            False,
            [
                f"Source type: {next_action_command_center_manifest.get('source_type', 'missing')}",
                f"Status: {next_action_command_center_manifest.get('status', 'missing')}",
                f"Cards: {next_action_command_center_manifest.get('sample_count', 'n/a')}",
                f"Tier O claim: {next_action_command_center_manifest.get('true_oracle_projection', 'missing')}",
                f"Agentic success claim: {next_action_command_center_manifest.get('agentic_success_claim', 'missing')}",
                "Acceptance: one-page next command and critical report links are present.",
            ],
            [
                (
                    "Next action command center HTML",
                    next_action_command_center_root / "next_action_command_center_report.html",
                ),
                (
                    "Next action command center manifest",
                    next_action_command_center_root / "next_action_command_center_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Agentic SmolVLA paper outline",
            paper_outline_ok,
            False,
            [
                f"Source type: {paper_outline_manifest.get('source_type', 'missing')}",
                f"Status: {paper_outline_manifest.get('status', 'missing')}",
                f"Cards: {paper_outline_manifest.get('sample_count', 'n/a')}",
                f"Tier O claim: {paper_outline_manifest.get('true_oracle_projection', 'missing')}",
                f"Agentic success claim: {paper_outline_manifest.get('agentic_success_claim', 'missing')}",
                "Acceptance: outline exists and keeps unsupported result claims out of the manuscript scaffold.",
            ],
            [
                ("Agentic SmolVLA paper outline HTML", paper_outline_root / "agentic_smolvla_paper_outline_report.html"),
                (
                    "Agentic SmolVLA paper outline manifest",
                    paper_outline_root / "agentic_smolvla_paper_outline_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Paper claim gate",
            paper_claim_gate_ok,
            False,
            [
                f"Source type: {paper_claim_gate_manifest.get('source_type', 'missing')}",
                f"Allowed claims: {paper_claim_gate_manifest.get('allowed_count', 'n/a')}",
                f"Caution claims: {paper_claim_gate_manifest.get('caution_count', 'n/a')}",
                f"Blocked claims: {paper_claim_gate_manifest.get('blocked_count', 'n/a')}",
                f"Tier O claim allowed: {paper_claim_gate_manifest.get('true_oracle_projection_claim_allowed', 'missing')}",
                f"Agentic success claim allowed: {paper_claim_gate_manifest.get('agentic_success_claim_allowed', 'missing')}",
                "Acceptance: unsupported Tier O and agentic success claims remain explicitly blocked.",
            ],
            [
                ("Paper claim gate HTML", paper_claim_gate_root / "paper_claim_gate_report.html"),
                ("Paper claim gate manifest", paper_claim_gate_root / "paper_claim_gate_manifest.json"),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Remote true-oracle evidence pack",
            remote_true_oracle_evidence_pack_ok,
            False,
            [
                f"Source type: {remote_true_oracle_evidence_pack_manifest.get('source_type', 'missing')}",
                f"Status: {remote_true_oracle_evidence_pack_manifest.get('status', 'missing')}",
                f"Runner: {remote_true_oracle_evidence_pack_manifest.get('runner', 'missing')}",
                f"True-oracle projection: {remote_true_oracle_evidence_pack_manifest.get('true_oracle_projection', 'missing')}",
                "Acceptance: single-command handoff exists and explicitly does not mutate pod lifecycle.",
            ],
            [
                (
                    "Remote true-oracle evidence pack HTML",
                    remote_true_oracle_evidence_pack_root / "remote_true_oracle_evidence_pack_report.html",
                ),
                (
                    "Remote true-oracle evidence pack manifest",
                    remote_true_oracle_evidence_pack_root / "remote_true_oracle_evidence_pack_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Renderer environment preflight",
            renderer_env_preflight_ok,
            False,
            [
                f"Source type: {renderer_env_preflight_manifest.get('source_type', 'missing')}",
                f"Status: {renderer_env_preflight_manifest.get('status', 'missing')}",
                f"True-oracle projection: {renderer_env_preflight_manifest.get('true_oracle_projection', 'missing')}",
                "Acceptance: preflight report exists and remains clearly non-Tier-O.",
            ],
            [
                ("Renderer environment preflight HTML", renderer_env_preflight_root / "renderer_env_preflight_report.html"),
                (
                    "Renderer environment preflight manifest",
                    renderer_env_preflight_root / "renderer_env_preflight_report_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Agentic retry schema readiness",
            agentic_retry_schema_ok,
            False,
            [
                f"Source type: {agentic_retry_schema_manifest.get('source_type', 'missing')}",
                f"Status: {agentic_retry_schema_manifest.get('status', 'missing')}",
                f"Retry events: {agentic_retry_schema_manifest.get('retry_events', 'missing')}",
                f"Final success source: {agentic_retry_schema_manifest.get('final_success_source', 'missing')}",
                "Acceptance: CP22/CP23 schema evidence exists and is clearly marked non-paper-benchmark.",
            ],
            [
                ("Agentic retry schema readiness HTML", agentic_retry_schema_root / "agentic_retry_schema_readiness.html"),
                (
                    "Agentic retry schema readiness manifest",
                    agentic_retry_schema_root / "agentic_retry_schema_readiness_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Agentic SmolVLA experiment matrix result",
            experiment_matrix_result_ok,
            False,
            [
                f"Source type: {experiment_matrix_result_manifest.get('source_type', 'missing')}",
                f"Status: {experiment_matrix_result_manifest.get('status', 'missing')}",
                f"Conditions: {experiment_matrix_result_manifest.get('condition_count', 'n/a')}",
                f"Claim-ready conditions: {experiment_matrix_result_manifest.get('claim_ready_count', 'n/a')}",
                f"Final success source: {experiment_matrix_result_manifest.get('final_success_source', 'missing')}",
                "Acceptance: C0-C5 matrix exists and marks unsupported behavioral claims as pending.",
            ],
            [
                (
                    "Agentic SmolVLA experiment matrix result HTML",
                    experiment_matrix_result_root / "agentic_smolvla_experiment_matrix_result.html",
                ),
                (
                    "Agentic SmolVLA experiment matrix result manifest",
                    experiment_matrix_result_root / "agentic_smolvla_experiment_matrix_result_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Paper progress ledger",
            paper_progress_ledger_ok,
            False,
            [
                f"Source type: {paper_progress_ledger_manifest.get('source_type', 'missing')}",
                f"Cards: {paper_progress_ledger_manifest.get('sample_count', 'n/a')}",
                f"True-oracle projection: {paper_progress_ledger_manifest.get('true_oracle_projection', 'missing')}",
                "Acceptance: claim ledger exists and separates supported, diagnostic-only, and blocked evidence.",
            ],
            [
                ("Paper progress ledger HTML", paper_progress_ledger_root / "paper_progress_ledger_report.html"),
                ("Paper progress ledger manifest", paper_progress_ledger_root / "paper_progress_ledger_manifest.json"),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Actual sim true-oracle two-stage result",
            actual_sim_true_oracle_two_stage_ok,
            actual_sim_true_oracle_two_stage_blocked,
            [
                f"Source type: {actual_sim_true_oracle_two_stage_manifest.get('source_type', 'missing')}",
                f"Status: {actual_sim_true_oracle_two_stage_manifest.get('status', 'missing')}",
                f"Probe strict samples: {actual_sim_true_oracle_two_stage_manifest.get('probe_strict_true_oracle_step_count', 'n/a')}",
                f"Policy strict samples: {actual_sim_true_oracle_two_stage_manifest.get('policy_strict_true_oracle_step_count', 'n/a')}",
                f"True-oracle projection: {actual_sim_true_oracle_two_stage_manifest.get('true_oracle_projection', 'missing')}",
                "Acceptance: passed only after probe and policy stages both reach >=10 strict samples.",
            ],
            [
                (
                    "Actual sim true-oracle two-stage result HTML",
                    actual_sim_true_oracle_two_stage_root / "actual_sim_true_oracle_two_stage_result.html",
                ),
                (
                    "Actual sim true-oracle two-stage result manifest",
                    actual_sim_true_oracle_two_stage_root / "actual_sim_true_oracle_two_stage_result_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Actual sim true-oracle remote handoff",
            actual_sim_true_oracle_remote_handoff_ok,
            False,
            [
                f"Source type: {actual_sim_true_oracle_remote_handoff_manifest.get('source_type', 'missing')}",
                f"Samples/cards: {actual_sim_true_oracle_remote_handoff_manifest.get('sample_count', 'n/a')}",
                f"Two-stage script: {actual_sim_true_oracle_remote_handoff_manifest.get('two_stage_script', 'missing')}",
                "Acceptance: handoff HTML and manifest exist; no Pods created or stopped.",
            ],
            [
                (
                    "Actual sim true-oracle remote handoff HTML",
                    actual_sim_true_oracle_remote_handoff_root / "actual_sim_true_oracle_remote_handoff_report.html",
                ),
                (
                    "Actual sim true-oracle remote handoff manifest",
                    actual_sim_true_oracle_remote_handoff_root / "actual_sim_true_oracle_remote_handoff_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Actual sim true-oracle probe blocker",
            actual_sim_true_oracle_probe_blocker_ok,
            False,
            [
                f"Source type: {actual_sim_true_oracle_probe_blocker_manifest.get('source_type', 'missing')}",
                f"Checkpoint status: {actual_sim_true_oracle_probe_blocker_manifest.get('checkpoint_status', 'missing')}",
                f"Rollout status: {actual_sim_true_oracle_probe_blocker_manifest.get('rollout_status', 'missing')}",
                f"SmolVLA ready: {actual_sim_true_oracle_probe_blocker_manifest.get('smolvla_ready', 'missing')}",
                "Acceptance: Mac-local renderer blocker documented without creating/stopping Pods.",
            ],
            [
                (
                    "Actual sim true-oracle probe blocker HTML",
                    actual_sim_true_oracle_probe_blocker_root / "actual_sim_true_oracle_probe_blocker.html",
                ),
                (
                    "Actual sim true-oracle probe blocker manifest",
                    actual_sim_true_oracle_probe_blocker_root / "actual_sim_true_oracle_probe_blocker_manifest.json",
                ),
            ],
            [],
        ),
        _milestone_card(
            output_path,
            "Actual RGB + synthetic metadata true-oracle codepath diagnostic",
            actual_rgb_synthetic_metadata_codepath_ok,
            False,
            [
                f"Source type: {actual_rgb_synthetic_metadata_codepath_manifest.get('source_type', 'missing')}",
                f"Samples: {actual_rgb_synthetic_metadata_codepath_manifest.get('sample_count', 'n/a')}",
                f"Strict codepath ready: {actual_rgb_synthetic_metadata_codepath_manifest.get('strict_codepath_ready_count', 'n/a')}",
                f"True-oracle projection claim: {actual_rgb_synthetic_metadata_codepath_manifest.get('true_oracle_projection', 'missing')}",
                "Acceptance: >=10 actual RGB samples with projected overlays from synthetic metadata; diagnostic only.",
            ],
            [
                (
                    "Actual RGB synthetic metadata codepath HTML",
                    actual_rgb_synthetic_metadata_codepath_root
                    / "actual_rgb_synthetic_metadata_codepath_report.html",
                ),
                (
                    "Actual RGB synthetic metadata codepath manifest",
                    actual_rgb_synthetic_metadata_codepath_root
                    / "actual_rgb_synthetic_metadata_codepath_manifest.json",
                ),
            ],
            [
                (
                    "Actual RGB synthetic metadata codepath contact sheet",
                    actual_rgb_synthetic_metadata_codepath_root
                    / "actual_rgb_synthetic_metadata_codepath_contact_sheet.jpg",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual sim true-oracle readiness gap",
            actual_sim_true_oracle_gap_ok,
            False,
            [
                f"Source type: {actual_sim_true_oracle_gap_manifest.get('source_type', 'missing')}",
                f"Samples: {actual_sim_true_oracle_gap_manifest.get('sample_count', 'n/a')}",
                f"Strict true-oracle steps: {actual_sim_true_oracle_gap_manifest.get('strict_true_oracle_step_count', 'n/a')}",
                "Acceptance: >=10 actual simulator RGB samples with explicit missing pose/camera metadata labels.",
            ],
            [
                (
                    "Actual sim true-oracle readiness gap HTML",
                    actual_sim_true_oracle_gap_root / "actual_sim_true_oracle_readiness_gap.html",
                ),
                (
                    "Actual sim true-oracle readiness gap manifest",
                    actual_sim_true_oracle_gap_root / "actual_sim_true_oracle_readiness_gap_manifest.json",
                ),
            ],
            [
                (
                    "Actual sim true-oracle readiness gap contact sheet",
                    actual_sim_true_oracle_gap_root / "actual_sim_true_oracle_readiness_gap_contact_sheet.jpg",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Actual sim true oracle projection",
            actual_sim_true_oracle_ok,
            actual_sim_true_oracle_blocked,
            [
                f"Source type: {actual_sim_true_oracle_manifest.get('source_type', 'missing')}",
                f"True oracle projection: {actual_sim_true_oracle_manifest.get('true_oracle_projection', 'missing')}",
                f"Samples: {actual_sim_true_oracle_manifest.get('sample_count', 'n/a')}",
                "Acceptance: >=10 actual sim RGB + pose + camera + overlay samples from the same action-input observation.",
            ],
            [
                ("Actual sim true oracle HTML", actual_sim_true_oracle_root / "actual_sim_true_oracle_report.html"),
                ("Actual sim true oracle manifest", actual_sim_true_oracle_root / "actual_sim_true_oracle_report_manifest.json"),
            ],
            [
                (
                    "Actual sim true oracle contact sheet",
                    actual_sim_true_oracle_root / "actual_sim_true_oracle_contact_sheet.png",
                ),
            ],
        ),
        _milestone_card(
            output_path,
            "Live ManiSkill true oracle projection",
            live_ok,
            live_blocked,
            [
                f"Checkpoint status: {live_report.get('status', 'missing')}",
                f"Rollout status: {live_report.get('metrics', {}).get('rollout_status', 'missing')}",
                "Acceptance: live RGB frames with object pose/camera metadata, >=10 overlay frames, HTML gallery.",
                "Current blocker: RunPod Vulkan/SAPIEN driver support if BLOCKED.",
            ],
            [
                ("Live checkpoint report", live_root / "checkpoint_report.json"),
                ("Live blocker report", blocker_path),
            ],
            [],
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oracle Overlay Milestone Dashboard</title>
  <style>
    :root {{
      --ink: #151511;
      --paper: #f6f0e4;
      --panel: #fffaf0;
      --line: #d4c4aa;
      --pass: #00c86d;
      --block: #c64a32;
      --missing: #8c8171;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: radial-gradient(circle at 12% 6%, #fff1bd, transparent 28%),
        linear-gradient(135deg, #efe0c7, #f9f5ed 46%, #e4eee7);
      font-family: Charter, Georgia, serif;
    }}
    header {{ padding: 44px 52px 18px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 10px; font-size: clamp(36px, 5vw, 72px); letter-spacing: -0.05em; }}
    main {{ padding: 30px 52px 64px; display: grid; gap: 24px; }}
    .card {{
      padding: 24px;
      border-radius: 28px;
      background: rgba(255, 250, 240, .86);
      border: 2px solid var(--line);
      box-shadow: 0 18px 50px rgba(60, 42, 18, .08);
    }}
    .card.pass {{ border-color: var(--pass); }}
    .card.block {{ border-color: var(--block); }}
    .card.missing {{ border-color: var(--missing); }}
    .state {{
      display: inline-flex;
      padding: 8px 12px;
      border-radius: 999px;
      font-family: Avenir Next, Helvetica, sans-serif;
      font-weight: 800;
      background: #fff;
      border: 1px solid currentColor;
    }}
    .pass .state {{ color: var(--pass); }}
    .block .state {{ color: var(--block); }}
    .missing .state {{ color: var(--missing); }}
    h2 {{ margin: 14px 0 8px; font-size: 30px; letter-spacing: -.03em; }}
    ul {{ font-family: Avenir Next, Helvetica, sans-serif; line-height: 1.5; }}
    a {{ color: #174f38; font-weight: 800; }}
    .images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    figure {{ margin: 0; padding: 12px; background: white; border-radius: 18px; border: 1px solid var(--line); }}
    img {{ width: 100%; border-radius: 12px; background: #2a2a2a; }}
    figcaption {{ margin-top: 8px; font-family: Avenir Next, Helvetica, sans-serif; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>Oracle Overlay Milestone Dashboard</h1>
    <p>Audited status for static validation, trajectory validation, reusable gallery generation, and live ManiSkill evidence.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(json.dumps({"dashboard": str(output_path), "live_status": _status(live_ok, live_blocked)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--live-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_dashboard(Path(args.root), Path(args.live_root), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
