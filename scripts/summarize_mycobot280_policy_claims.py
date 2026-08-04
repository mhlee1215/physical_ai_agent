#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICY_SPECS = {
    "base": "No fine-tuning",
    "deterministic_finetuned": "Deterministic fine-tuning",
    "randomized_finetuned": "Randomized fine-tuning",
}
FRESH_DETERMINISTIC_ROW = "deterministic_fresh_randomized"
FRESH_RANDOMIZED_ROW = "randomized_fresh_randomized"


def build_claim_summary(
    reports: dict[str, dict[str, Any]], multiseed: dict[str, Any]
) -> dict[str, Any]:
    _validate_matched_reports(reports)
    _validate_multiseed(multiseed)

    matched_rows = {
        name: _summarize_report(report) for name, report in reports.items()
    }
    aggregate = multiseed["aggregate_rows"]
    det_aggregate = aggregate[FRESH_DETERMINISTIC_ROW]
    rand_aggregate = aggregate[FRESH_RANDOMIZED_ROW]
    comparison = multiseed["paired_comparisons"]
    comparison = comparison["fresh_randomized_training_data_effect"]
    source_caveat = multiseed["source_distribution_caveat"]

    return {
        "operation": "summarize_mycobot280_policy_claims",
        "status": "passed",
        "evaluation_contract": {
            "camera": "ground_pickup_closeup",
            "physics": "randomized_from_audited_source_manifest",
            "candidate_selection": "direct_unfiltered_draws_without_teacher_rejection",
            "steps_per_episode": int(reports["base"]["steps_per_episode"]),
            "matched_environment_seeds": [
                int(item["seed"]) for item in reports["base"]["episode_summaries"]
            ],
            "strict_definition": (
                "pickup/lift/contact/hold gates plus maximum pad-cube penetration "
                "at or below 3.0 mm"
            ),
            "functional_definition": (
                "pickup/lift/contact/hold gates with the penetration gate excluded"
            ),
        },
        "matched_schedule_rows": matched_rows,
        "fresh_randomized_multiseed": {
            "training_seeds": multiseed["training_seeds"],
            "deterministic": _aggregate_claim_row(det_aggregate),
            "randomized": _aggregate_claim_row(rand_aggregate),
            "strict_success_deltas_per_training_seed": comparison[
                "strict_success_count_deltas"
            ],
            "positive_training_seed_count": comparison[
                "positive_training_seed_count"
            ],
            "exact_two_sided_sign_test_p": comparison[
                "exact_two_sided_sign_test_p"
            ],
            "mean_paired_penetration_delta_mm": comparison[
                "mean_paired_penetration_delta_mm"
            ],
        },
        "teacher_filtering": {
            "applies_to": "randomized source demonstrations used for training and validation",
            "does_not_apply_to": "closed-loop policy evaluation rollouts",
            "accepted_attempts": int(source_caveat["accepted_attempts"]),
            "total_attempts": int(source_caveat["total_attempts"]),
            "highest_mass_quartile_accepted": int(
                source_caveat["highest_mass_quartile_accepted"]
            ),
            "highest_mass_quartile_attempted": int(
                source_caveat["highest_mass_quartile_attempted"]
            ),
            "highest_mass_quartile_validation_examples": int(
                source_caveat["highest_mass_quartile_validation_examples"]
            ),
            "validation_empty_friction_quartiles": source_caveat[
                "validation_empty_friction_quartiles"
            ],
        },
        "supported_claims": {
            "fine_tuning_changes_task_behavior": {
                "status": "supported_on_one_matched_11_episode_schedule",
                "evidence": (
                    "base pickup+hold 0/11; deterministic and randomized fine-tuned "
                    "policies each 11/11"
                ),
            },
            "randomized_improves_strict_success_direction": {
                "status": "replicated_engineering_evidence_not_statistical_significance",
                "evidence": (
                    "randomized exceeds deterministic strict success in 3/3 paired "
                    "training seeds on fresh randomized physics; exact sign-test p=0.25"
                ),
            },
            "randomized_improves_functional_success": {
                "status": "not_supported_as_a_material_advantage",
                "evidence": (
                    "penetration-excluded pickup+hold is 32/33 deterministic versus "
                    "33/33 randomized on fresh randomized physics"
                ),
            },
            "randomized_improves_contact_quality": {
                "status": "supported_directionally_under_the_fixed_3mm_gate",
                "evidence": (
                    "strict success is 4/33 versus 14/33 and mean paired maximum "
                    "penetration changes by -0.082 mm"
                ),
            },
        },
        "claim_boundary": (
            "The evidence supports scoped simulation claims only. It does not establish "
            "statistical significance, uniform physics coverage, real-robot transfer, "
            "or agentic retry benefit."
        ),
    }


def render_figure(summary: dict[str, Any], output_path: Path) -> None:
    import cv2
    import numpy as np

    canvas = np.full((980, 1600, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "myCobot 280 policy paradigm comparison - 2026-08-01",
        (320, 52),
        font,
        1.02,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Fresh randomized physics; close camera; every evaluation rollout retained",
        (420, 88),
        font,
        0.58,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )

    _draw_paradigm_scatter(canvas, summary)
    _draw_paired_replication(canvas, summary)

    notes = [
        "Fine-tuning signal (matched n=11): pickup+hold 0/11 base -> 11/11 for both fine-tuned policies.",
        "Without the penetration gate (pooled n=33): deterministic 32/33 vs randomized 33/33 pickup+hold.",
        "With the 3 mm gate (pooled n=33): deterministic 4/33 vs randomized 14/33 strict success.",
        "Training demonstrations are teacher-success-conditioned (60/73 accepted); policy evaluation is unfiltered.",
    ]
    colors = [(25, 25, 25), (25, 25, 25), (25, 25, 25), (45, 45, 150)]
    for index, (line, color) in enumerate(zip(notes, colors, strict=True)):
        cv2.putText(
            canvas,
            line,
            (145, 820 + index * 38),
            font,
            0.53 if index < 3 else 0.50,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write figure: {output_path}")


def _draw_paradigm_scatter(canvas: Any, summary: dict[str, Any]) -> None:
    import cv2

    left, right, top, bottom = 115, 755, 195, 725
    cv2.putText(
        canvas,
        "Task behavior vs physics-valid success",
        (195, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    for tick in (0, 25, 50, 75, 100):
        x = left + int((right - left) * tick / 100)
        y = bottom - int((bottom - top) * tick / 100)
        cv2.line(canvas, (x, top), (x, bottom), (232, 232, 232), 1)
        cv2.line(canvas, (left, y), (right, y), (232, 232, 232), 1)
        cv2.putText(canvas, str(tick), (x - 13, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(tick), (left - 42, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (45, 45, 45), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (45, 45, 45), 2)
    cv2.putText(canvas, "Pickup + hold, penetration excluded (%)", (260, bottom + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Strict success including 3 mm cap (%)", (150, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)

    rows = summary["matched_schedule_rows"]
    colors = {
        "base": (100, 100, 100),
        "deterministic_finetuned": (175, 95, 30),
        "randomized_finetuned": (50, 145, 55),
    }
    points: dict[str, tuple[int, int]] = {}
    for name, row in rows.items():
        x = left + int((right - left) * row["pickup_hold_success_rate"])
        y = bottom - int((bottom - top) * row["strict_success_rate"])
        points[name] = (x, y)

    cv2.arrowedLine(canvas, (points["base"][0] + 10, points["base"][1] - 8), (points["deterministic_finetuned"][0] - 12, points["deterministic_finetuned"][1] + 4), colors["deterministic_finetuned"], 2, cv2.LINE_AA, tipLength=0.03)
    cv2.arrowedLine(canvas, (points["deterministic_finetuned"][0], points["deterministic_finetuned"][1] - 12), (points["randomized_finetuned"][0], points["randomized_finetuned"][1] + 12), colors["randomized_finetuned"], 2, cv2.LINE_AA, tipLength=0.10)

    label_offsets = {
        "base": (15, -12),
        "deterministic_finetuned": (-245, 24),
        "randomized_finetuned": (-225, -14),
    }
    for name, point in points.items():
        row = rows[name]
        cv2.circle(canvas, point, 10, colors[name], -1, cv2.LINE_AA)
        dx, dy = label_offsets[name]
        label = f'{POLICY_SPECS[name]} ({row["pickup_hold_successes"]}/{row["episodes"]}, {row["strict_successes"]}/{row["episodes"]})'
        cv2.putText(canvas, label, (point[0] + dx, point[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.43, colors[name], 1, cv2.LINE_AA)


def _draw_paired_replication(canvas: Any, summary: dict[str, Any]) -> None:
    import cv2

    left, right, top, bottom = 900, 1490, 195, 725
    det_x, rand_x = 1050, 1340
    cv2.putText(canvas, "Three paired training seeds", (1025, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (25, 25, 25), 2, cv2.LINE_AA)
    for tick in (0, 3, 6, 9, 11):
        y = bottom - int((bottom - top) * tick / 11)
        cv2.line(canvas, (left, y), (right, y), (232, 232, 232), 1)
        cv2.putText(canvas, str(tick), (left - 30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (45, 45, 45), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (45, 45, 45), 2)
    cv2.putText(canvas, "Strict successes / 11", (left + 5, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Deterministic", (det_x - 65, bottom + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (175, 95, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Randomized", (rand_x - 57, bottom + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (50, 145, 55), 1, cv2.LINE_AA)

    multi = summary["fresh_randomized_multiseed"]
    det_values = multi["deterministic"]["strict_successes_per_training_seed"]
    rand_values = multi["randomized"]["strict_successes_per_training_seed"]
    seed_colors = [(65, 120, 205), (150, 85, 155), (160, 125, 35)]
    for seed, det, rand, color in zip(multi["training_seeds"], det_values, rand_values, seed_colors, strict=True):
        det_y = bottom - int((bottom - top) * det / 11)
        rand_y = bottom - int((bottom - top) * rand / 11)
        cv2.line(canvas, (det_x, det_y), (rand_x, rand_y), color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (det_x, det_y), 8, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (rand_x, rand_y), 8, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(seed), (rand_x + 18, rand_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Randomized higher in 3/3 pairs", (1030, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Exact sign test p=0.25 (low power)", (1010, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 45, 150), 1, cv2.LINE_AA)


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    episodes = report["episode_summaries"]
    strict = sum(bool(item["success"]) for item in episodes)
    pickup_hold = sum(_pickup_hold_success(item) for item in episodes)
    penetration_only = sum(
        item.get("failed_gates") == ["max_pad_cube_penetration_exceeded"]
        for item in episodes
    )
    return {
        "label": "",
        "episodes": len(episodes),
        "strict_successes": strict,
        "strict_success_rate": strict / len(episodes),
        "pickup_hold_successes": pickup_hold,
        "pickup_hold_success_rate": pickup_hold / len(episodes),
        "penetration_only_failures": penetration_only,
    }


def _aggregate_claim_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": int(row["episodes_total"]),
        "strict_successes": int(row["strict_successes_total"]),
        "strict_success_rate": float(row["pooled_strict_success_rate"]),
        "strict_successes_per_training_seed": row[
            "strict_successes_per_training_seed"
        ],
        "pickup_hold_successes": int(row["pickup_hold_successes_total"]),
        "pickup_hold_success_rate": float(row["pooled_pickup_hold_success_rate"]),
        "penetration_only_failures": int(row["penetration_only_failures_total"]),
    }


def _validate_matched_reports(reports: dict[str, dict[str, Any]]) -> None:
    if set(reports) != set(POLICY_SPECS):
        raise ValueError("reports must contain base, deterministic, and randomized policies")
    reference = reports["base"]
    for name, report in reports.items():
        if report.get("status") != "completed":
            raise ValueError(f"{name} report is not completed")
        environment = report.get("environment", {})
        if environment.get("render_camera_profile") != "ground_pickup_closeup":
            raise ValueError(f"{name} does not use the close camera")
        if environment.get("object_physics") != "randomized_from_audited_source_manifest":
            raise ValueError(f"{name} does not use fresh randomized physics")
        contract = report.get("schedule_contract", {})
        if contract.get("candidate_selection") != "direct_unfiltered_draws_without_teacher_rejection":
            raise ValueError(f"{name} evaluation is not an unfiltered direct draw")
        if int(contract.get("source_seed_overlap_count", -1)) != 0:
            raise ValueError(f"{name} overlaps source dataset attempts")
        feature = report.get("policy_runtime", {}).get("contract", {}).get("feature_contract", {})
        if not feature.get("exact_7d_state_action"):
            raise ValueError(f"{name} violates the exact-7D policy contract")
        if report.get("schedule") != reference.get("schedule"):
            raise ValueError(f"{name} schedule does not match the base report")
        if report.get("steps_per_episode") != reference.get("steps_per_episode"):
            raise ValueError(f"{name} horizon does not match the base report")
        candidates = [item.get("candidate") for item in report["episode_summaries"]]
        reference_candidates = [
            item.get("candidate") for item in reference["episode_summaries"]
        ]
        if candidates != reference_candidates:
            raise ValueError(f"{name} candidates do not match the base report")


def _validate_multiseed(multiseed: dict[str, Any]) -> None:
    if multiseed.get("status") != "passed":
        raise ValueError("multiseed summary is not passed")
    if int(multiseed.get("training_seed_count", 0)) < 2:
        raise ValueError("multiseed summary needs at least two training seeds")
    rows = multiseed.get("aggregate_rows", {})
    if FRESH_DETERMINISTIC_ROW not in rows or FRESH_RANDOMIZED_ROW not in rows:
        raise ValueError("multiseed summary lacks fresh-randomized rows")
    comparison = multiseed.get("paired_comparisons", {}).get(
        "fresh_randomized_training_data_effect", {}
    )
    if comparison.get("left") != FRESH_DETERMINISTIC_ROW:
        raise ValueError("multiseed deterministic comparison row is invalid")
    if comparison.get("right") != FRESH_RANDOMIZED_ROW:
        raise ValueError("multiseed randomized comparison row is invalid")


def _pickup_hold_success(item: dict[str, Any]) -> bool:
    return bool(
        float(item["final_cube_lift_m"]) >= 0.05
        and int(item["final_pad_cube_contacted_pads"]) >= 2
        and int(item["lift_best_sustained_two_pad_steps"]) >= 60
        and int(item["post_lift_hold_best_sustained_two_pad_steps"]) >= 300
        and float(item["post_lift_hold_min_cube_lift_m"]) >= 0.045
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize scoped myCobot policy claims and render their figure."
    )
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--deterministic-report", type=Path, required=True)
    parser.add_argument("--randomized-report", type=Path, required=True)
    parser.add_argument("--multiseed-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        "base": json.loads(args.base_report.read_text(encoding="utf-8")),
        "deterministic_finetuned": json.loads(
            args.deterministic_report.read_text(encoding="utf-8")
        ),
        "randomized_finetuned": json.loads(
            args.randomized_report.read_text(encoding="utf-8")
        ),
    }
    multiseed = json.loads(args.multiseed_summary.read_text(encoding="utf-8"))
    summary = build_claim_summary(reports, multiseed)
    for name, label in POLICY_SPECS.items():
        summary["matched_schedule_rows"][name]["label"] = label
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_figure(summary, args.output_figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
