#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROW_SPECS = {
    "deterministic_nominal": {
        "label": "Deterministic-trained / nominal",
        "physics": "fixed",
    },
    "randomized_nominal": {
        "label": "Randomized-trained / nominal",
        "physics": "fixed",
    },
    "deterministic_fresh_randomized": {
        "label": "Deterministic-trained / fresh randomized",
        "physics": "randomized_from_audited_source_manifest",
    },
    "randomized_fresh_randomized": {
        "label": "Randomized-trained / fresh randomized",
        "physics": "randomized_from_audited_source_manifest",
    },
}
PAIR_SPECS = {
    "nominal_training_data_effect": (
        "deterministic_nominal",
        "randomized_nominal",
    ),
    "fresh_randomized_training_data_effect": (
        "deterministic_fresh_randomized",
        "randomized_fresh_randomized",
    ),
}


def load_reports(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }


def summarize_pilot(
    reports: dict[str, dict[str, Any]],
    supervised: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _validate_reports(reports, supervised)
    rows: dict[str, Any] = {}
    for name, spec in ROW_SPECS.items():
        episodes = reports[name]["episode_summaries"]
        rows[name] = {
            "label": spec["label"],
            "episodes": len(episodes),
            "strict_successes": sum(bool(item["success"]) for item in episodes),
            "strict_success_rate": _mean(
                [float(bool(item["success"])) for item in episodes]
            ),
            "pickup_hold_successes": sum(_pickup_hold_success(item) for item in episodes),
            "pickup_hold_success_rate": _mean(
                [float(_pickup_hold_success(item)) for item in episodes]
            ),
            "mean_final_cube_lift_mm": 1000.0
            * _mean([float(item["final_cube_lift_m"]) for item in episodes]),
            "mean_max_pad_cube_penetration_mm": 1000.0
            * _mean([float(item["max_pad_cube_penetration_m"]) for item in episodes]),
            "max_pad_cube_penetration_mm": 1000.0
            * max(float(item["max_pad_cube_penetration_m"]) for item in episodes),
            "strict_success_seeds": [
                int(item["seed"]) for item in episodes if bool(item["success"])
            ],
            "penetration_only_failure_seeds": [
                int(item["seed"])
                for item in episodes
                if item.get("failed_gates") == ["max_pad_cube_penetration_exceeded"]
            ],
            "failure_reason_counts": reports[name]["aggregate"]["failure_reason_counts"],
        }

    comparisons: dict[str, Any] = {}
    for name, (left_name, right_name) in PAIR_SPECS.items():
        left = _episodes_by_seed(reports[left_name])
        right = _episodes_by_seed(reports[right_name])
        seeds = sorted(left)
        left_success = {seed for seed in seeds if bool(left[seed]["success"])}
        right_success = {seed for seed in seeds if bool(right[seed]["success"])}
        penetration_deltas = [
            1000.0
            * (
                float(right[seed]["max_pad_cube_penetration_m"])
                - float(left[seed]["max_pad_cube_penetration_m"])
            )
            for seed in seeds
        ]
        comparisons[name] = {
            "left": left_name,
            "right": right_name,
            "matched_seeds": seeds,
            "strict_success_count_delta": len(right_success) - len(left_success),
            "added_strict_success_seeds": sorted(right_success - left_success),
            "lost_strict_success_seeds": sorted(left_success - right_success),
            "mean_penetration_delta_mm": _mean(penetration_deltas),
            "lower_penetration_seed_count": sum(delta < 0.0 for delta in penetration_deltas),
        }

    base_loss = float(supervised["base"]["loss_mean"])
    tuned_loss = float(supervised["randomized_finetuned"]["loss_mean"])
    base_rmse = float(supervised["base"]["postprocessed_action_rmse_mean"])
    tuned_rmse = float(
        supervised["randomized_finetuned"]["postprocessed_action_rmse_mean"]
    )
    return {
        "operation": "summarize_mycobot280_randomized_training_pilot",
        "status": "passed",
        "rows": rows,
        "paired_comparisons": comparisons,
        "heldout_supervised": {
            "batches": int(supervised["base"]["batches_evaluated"]),
            "base_loss_mean": base_loss,
            "randomized_finetuned_loss_mean": tuned_loss,
            "loss_relative_change": (tuned_loss - base_loss) / base_loss,
            "base_postprocessed_action_rmse_mean": base_rmse,
            "randomized_finetuned_postprocessed_action_rmse_mean": tuned_rmse,
            "action_rmse_relative_change": (tuned_rmse - base_rmse) / base_rmse,
            "interpretation": (
                "Loss and postprocessed action RMSE disagree; neither is treated as a "
                "closed-loop success proxy."
            ),
        },
        "fresh_randomized_schedule_contract": reports[
            "randomized_fresh_randomized"
        ]["schedule_contract"],
        "pickup_hold_definition": {
            "final_cube_lift_m_min": 0.05,
            "final_pad_cube_contacted_pads_min": 2,
            "lift_best_sustained_two_pad_steps_min": 60,
            "post_lift_hold_best_sustained_two_pad_steps_min": 300,
            "post_lift_hold_min_cube_lift_m_min": 0.045,
            "penetration_gate_excluded": True,
        },
        "claim_boundary": (
            "Matched 11-seed pilot evidence from one 100-step randomized-policy training "
            "seed. Repeat policy-training seeds and use larger held-out randomized schedules "
            "before publication-level robustness claims."
        ),
    }


def render_figure(summary: dict[str, Any], output_path: Path) -> None:
    import cv2
    import numpy as np

    canvas = np.full((820, 1300, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "myCobot 280 randomized-training pilot - 2026-07-31",
        (205, 48),
        font,
        0.98,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Matched close camera; one 100-step training seed; 11 seeds per evaluation regime",
        (260, 80),
        font,
        0.56,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )

    rows = summary["rows"]
    names = [
        "deterministic_nominal",
        "randomized_nominal",
        "deterministic_fresh_randomized",
        "randomized_fresh_randomized",
    ]
    labels = [("Det.", "nominal"), ("Rand.", "nominal"), ("Det.", "fresh"), ("Rand.", "fresh")]
    colors = [(122, 122, 122), (44, 136, 69), (122, 122, 122), (44, 136, 69)]
    _draw_bar_chart(
        canvas,
        values=[100.0 * rows[name]["strict_success_rate"] for name in names],
        labels=labels,
        colors=colors,
        origin=(75, 490),
        size=(530, 300),
        maximum=100.0,
        title="Strict success (%)",
        value_labels=[f'{rows[name]["strict_successes"]}/11' for name in names],
        threshold=None,
    )
    _draw_bar_chart(
        canvas,
        values=[rows[name]["mean_max_pad_cube_penetration_mm"] for name in names],
        labels=labels,
        colors=colors,
        origin=(695, 490),
        size=(530, 300),
        maximum=3.6,
        title="Mean maximum penetration (mm)",
        value_labels=[
            f'{rows[name]["mean_max_pad_cube_penetration_mm"]:.2f}' for name in names
        ],
        threshold=3.0,
    )

    supervised = summary["heldout_supervised"]
    lines = [
        "Pickup + hold (penetration excluded): 11/11 in all four cells.",
        (
            "Held-out supervised (20 batches): loss "
            f'{supervised["base_loss_mean"]:.3f} -> '
            f'{supervised["randomized_finetuned_loss_mean"]:.3f}; action RMSE '
            f'{supervised["base_postprocessed_action_rmse_mean"]:.4f} -> '
            f'{supervised["randomized_finetuned_postprocessed_action_rmse_mean"]:.4f}.'
        ),
        "Pilot only: repeat training seeds and enlarge unseen randomized schedules.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (110, 625 + index * 45),
            font,
            0.58 if index < 2 else 0.54,
            (25, 25, 25) if index < 2 else (40, 40, 145),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write figure: {output_path}")


def _draw_bar_chart(
    canvas: Any,
    *,
    values: list[float],
    labels: list[tuple[str, str]],
    colors: list[tuple[int, int, int]],
    origin: tuple[int, int],
    size: tuple[int, int],
    maximum: float,
    title: str,
    value_labels: list[str],
    threshold: float | None,
) -> None:
    import cv2

    left, bottom = origin
    width, height = size
    top = bottom - height
    cv2.line(canvas, (left, top), (left, bottom), (55, 55, 55), 1)
    cv2.line(canvas, (left, bottom), (left + width, bottom), (55, 55, 55), 1)
    cv2.putText(canvas, title, (left, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
    if threshold is not None:
        y = bottom - int(height * threshold / maximum)
        for x in range(left, left + width, 16):
            cv2.line(canvas, (x, y), (min(x + 8, left + width), y), (35, 35, 180), 2)
        cv2.putText(canvas, "3 mm gate", (left + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 180), 1, cv2.LINE_AA)
    slot = width / len(values)
    for index, (value, label, color, value_label) in enumerate(zip(values, labels, colors, value_labels, strict=True)):
        center = int(left + slot * (index + 0.5))
        bar_height = int(height * min(value, maximum) / maximum)
        cv2.rectangle(canvas, (center - 35, bottom - bar_height), (center + 35, bottom), color, -1)
        cv2.putText(canvas, value_label, (center - 24, bottom - bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, label[0], (center - 25, bottom + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, label[1], (center - 31, bottom + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (40, 40, 40), 1, cv2.LINE_AA)


def _validate_reports(
    reports: dict[str, dict[str, Any]],
    supervised: dict[str, dict[str, Any]],
) -> None:
    missing = sorted(set(ROW_SPECS) - set(reports))
    if missing:
        raise ValueError(f"missing randomized-training rows: {missing}")
    if set(supervised) != {"base", "randomized_finetuned"}:
        raise ValueError("supervised reports must contain base and randomized_finetuned")
    for name, spec in ROW_SPECS.items():
        report = reports[name]
        if report.get("status") != "completed":
            raise ValueError(f"{name} is not completed")
        if report.get("environment", {}).get("render_camera_profile") != "ground_pickup_closeup":
            raise ValueError(f"{name} does not use the matched close camera")
        if report.get("environment", {}).get("object_physics") != spec["physics"]:
            raise ValueError(f"{name} object physics do not match the row contract")
        feature = report.get("policy_runtime", {}).get("contract", {}).get("feature_contract", {})
        if not feature.get("exact_7d_state_action") or feature.get("state_shape") != [7] or feature.get("action_shape") != [7]:
            raise ValueError(f"{name} does not satisfy the exact-7D contract")

    _validate_matched_pair(reports, "deterministic_nominal", "randomized_nominal")
    _validate_matched_pair(
        reports,
        "deterministic_fresh_randomized",
        "randomized_fresh_randomized",
        require_candidates=True,
    )
    for name in ("deterministic_fresh_randomized", "randomized_fresh_randomized"):
        contract = reports[name].get("schedule_contract", {})
        if contract.get("candidate_selection") != "direct_unfiltered_draws_without_teacher_rejection":
            raise ValueError(f"{name} does not use direct unfiltered randomized draws")
        if int(contract.get("source_seed_overlap_count", -1)) != 0:
            raise ValueError(f"{name} overlaps source-dataset attempt seeds")
    for name, report in supervised.items():
        if report.get("operation") != "evaluate_smolvla_supervised_loss":
            raise ValueError(f"{name} is not a supervised evaluation report")
        feature = report.get("contract", {}).get("feature_contract", {})
        if not feature.get("exact_7d_state_action"):
            raise ValueError(f"{name} supervised report does not satisfy exact-7D")


def _validate_matched_pair(
    reports: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
    *,
    require_candidates: bool = False,
) -> None:
    left = reports[left_name]
    right = reports[right_name]
    if left.get("schedule") != right.get("schedule"):
        raise ValueError(f"{left_name} and {right_name} schedules do not match")
    if left.get("steps_per_episode") != right.get("steps_per_episode"):
        raise ValueError(f"{left_name} and {right_name} step horizons do not match")
    if require_candidates:
        left_candidates = [item.get("candidate") for item in left["episode_summaries"]]
        right_candidates = [item.get("candidate") for item in right["episode_summaries"]]
        if left_candidates != right_candidates:
            raise ValueError(f"{left_name} and {right_name} randomized candidates do not match")


def _pickup_hold_success(item: dict[str, Any]) -> bool:
    return bool(
        float(item["final_cube_lift_m"]) >= 0.05
        and int(item["final_pad_cube_contacted_pads"]) >= 2
        and int(item["lift_best_sustained_two_pad_steps"]) >= 60
        and int(item["post_lift_hold_best_sustained_two_pad_steps"]) >= 300
        and float(item["post_lift_hold_min_cube_lift_m"]) >= 0.045
    )


def _episodes_by_seed(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["seed"]): item for item in report["episode_summaries"]}


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the myCobot 280 randomized-training pilot.")
    parser.add_argument("--deterministic-nominal-report", type=Path, required=True)
    parser.add_argument("--randomized-nominal-report", type=Path, required=True)
    parser.add_argument("--deterministic-randomized-report", type=Path, required=True)
    parser.add_argument("--randomized-randomized-report", type=Path, required=True)
    parser.add_argument("--base-supervised-report", type=Path, required=True)
    parser.add_argument("--finetuned-supervised-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path)
    args = parser.parse_args()

    reports = load_reports(
        {
            "deterministic_nominal": args.deterministic_nominal_report,
            "randomized_nominal": args.randomized_nominal_report,
            "deterministic_fresh_randomized": args.deterministic_randomized_report,
            "randomized_fresh_randomized": args.randomized_randomized_report,
        }
    )
    supervised = load_reports(
        {
            "base": args.base_supervised_report,
            "randomized_finetuned": args.finetuned_supervised_report,
        }
    )
    summary = summarize_pilot(reports, supervised)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_figure is not None:
        render_figure(summary, args.output_figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
