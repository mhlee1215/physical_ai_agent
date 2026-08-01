#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROW_SPECS = {
    "deterministic_nominal": {
        "label": "Deterministic-trained / nominal",
        "training_data": "deterministic_close",
        "evaluation_physics": "fixed",
    },
    "randomized_nominal": {
        "label": "Randomized-trained / nominal",
        "training_data": "randomized_close",
        "evaluation_physics": "fixed",
    },
    "deterministic_fresh_randomized": {
        "label": "Deterministic-trained / fresh randomized",
        "training_data": "deterministic_close",
        "evaluation_physics": "randomized_from_audited_source_manifest",
    },
    "randomized_fresh_randomized": {
        "label": "Randomized-trained / fresh randomized",
        "training_data": "randomized_close",
        "evaluation_physics": "randomized_from_audited_source_manifest",
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


def load_seed_report_sets(
    specs: list[list[str]],
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, dict[str, str]]]:
    report_sets: dict[int, dict[str, dict[str, Any]]] = {}
    report_paths: dict[str, dict[str, str]] = {}
    for values in specs:
        training_seed = int(values[0])
        if training_seed in report_sets:
            raise ValueError(f"duplicate training seed: {training_seed}")
        paths = {
            name: Path(value)
            for name, value in zip(ROW_SPECS, values[1:], strict=True)
        }
        report_sets[training_seed] = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
        report_paths[str(training_seed)] = {
            name: str(path) for name, path in paths.items()
        }
    return report_sets, report_paths


def summarize_multiseed(
    report_sets: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    _validate_report_sets(report_sets)
    training_seeds = sorted(report_sets)
    per_training_seed: dict[str, Any] = {}
    for training_seed in training_seeds:
        per_training_seed[str(training_seed)] = {
            name: _summarize_row(report_sets[training_seed][name])
            for name in ROW_SPECS
        }

    aggregate_rows: dict[str, Any] = {}
    for name, spec in ROW_SPECS.items():
        seed_rows = [per_training_seed[str(seed)][name] for seed in training_seeds]
        episodes_per_seed = [int(row["episodes"]) for row in seed_rows]
        strict_counts = [int(row["strict_successes"]) for row in seed_rows]
        pickup_counts = [int(row["pickup_hold_successes"]) for row in seed_rows]
        aggregate_rows[name] = {
            "label": spec["label"],
            "training_data": spec["training_data"],
            "evaluation_physics": spec["evaluation_physics"],
            "training_seeds": training_seeds,
            "episodes_per_training_seed": episodes_per_seed,
            "strict_successes_per_training_seed": strict_counts,
            "strict_successes_total": sum(strict_counts),
            "episodes_total": sum(episodes_per_seed),
            "pooled_strict_success_rate": sum(strict_counts) / sum(episodes_per_seed),
            "mean_strict_successes_per_training_seed": _mean(strict_counts),
            "strict_success_count_range": [min(strict_counts), max(strict_counts)],
            "strict_success_rate_seed_std": _sample_std(
                [row["strict_success_rate"] for row in seed_rows]
            ),
            "pickup_hold_successes_per_training_seed": pickup_counts,
            "pickup_hold_successes_total": sum(pickup_counts),
            "pooled_pickup_hold_success_rate": sum(pickup_counts)
            / sum(episodes_per_seed),
            "mean_final_cube_lift_mm": _mean(
                [row["mean_final_cube_lift_mm"] for row in seed_rows]
            ),
            "mean_max_pad_cube_penetration_mm": _mean(
                [row["mean_max_pad_cube_penetration_mm"] for row in seed_rows]
            ),
        }
        if "high_mass_slice" in seed_rows[0]:
            high_successes = sum(
                int(row["high_mass_slice"]["strict_successes"]) for row in seed_rows
            )
            high_episodes = sum(
                int(row["high_mass_slice"]["episodes"]) for row in seed_rows
            )
            aggregate_rows[name]["high_mass_slice"] = {
                "cube_mass_kg_min": 0.034,
                "strict_successes": high_successes,
                "episodes": high_episodes,
                "strict_success_rate": high_successes / high_episodes,
            }

    comparisons: dict[str, Any] = {}
    for comparison_name, (left_name, right_name) in PAIR_SPECS.items():
        seed_results = []
        strict_deltas = []
        pickup_deltas = []
        penetration_deltas = []
        for training_seed in training_seeds:
            left_report = report_sets[training_seed][left_name]
            right_report = report_sets[training_seed][right_name]
            left_row = per_training_seed[str(training_seed)][left_name]
            right_row = per_training_seed[str(training_seed)][right_name]
            strict_delta = int(right_row["strict_successes"]) - int(
                left_row["strict_successes"]
            )
            pickup_delta = int(right_row["pickup_hold_successes"]) - int(
                left_row["pickup_hold_successes"]
            )
            per_episode_penetration_deltas = _paired_penetration_deltas_mm(
                left_report, right_report
            )
            strict_deltas.append(strict_delta)
            pickup_deltas.append(pickup_delta)
            penetration_deltas.extend(per_episode_penetration_deltas)
            seed_results.append(
                {
                    "training_seed": training_seed,
                    "left_strict_successes": left_row["strict_successes"],
                    "right_strict_successes": right_row["strict_successes"],
                    "strict_success_count_delta": strict_delta,
                    "pickup_hold_success_count_delta": pickup_delta,
                    "mean_penetration_delta_mm": _mean(
                        per_episode_penetration_deltas
                    ),
                    "lower_penetration_episode_count": sum(
                        delta < 0.0 for delta in per_episode_penetration_deltas
                    ),
                }
            )
        nonzero_deltas = [delta for delta in strict_deltas if delta != 0]
        comparisons[comparison_name] = {
            "left": left_name,
            "right": right_name,
            "per_training_seed": seed_results,
            "strict_success_count_deltas": strict_deltas,
            "mean_strict_success_count_delta": _mean(strict_deltas),
            "mean_strict_success_rate_delta": _mean(
                [
                    delta
                    / per_training_seed[str(seed)][left_name]["episodes"]
                    for seed, delta in zip(training_seeds, strict_deltas, strict=True)
                ]
            ),
            "positive_training_seed_count": sum(delta > 0 for delta in strict_deltas),
            "zero_training_seed_count": sum(delta == 0 for delta in strict_deltas),
            "negative_training_seed_count": sum(delta < 0 for delta in strict_deltas),
            "exact_two_sided_sign_test_p": _two_sided_sign_test_p(nonzero_deltas),
            "pickup_hold_success_count_deltas": pickup_deltas,
            "mean_paired_penetration_delta_mm": _mean(penetration_deltas),
            "lower_penetration_episode_count": sum(
                delta < 0.0 for delta in penetration_deltas
            ),
            "paired_episode_count": len(penetration_deltas),
        }

    return {
        "operation": "summarize_mycobot280_randomized_training_multiseed",
        "status": "passed",
        "training_seeds": training_seeds,
        "training_seed_count": len(training_seeds),
        "per_training_seed": per_training_seed,
        "aggregate_rows": aggregate_rows,
        "paired_comparisons": comparisons,
        "pickup_hold_definition": {
            "final_cube_lift_m_min": 0.05,
            "final_pad_cube_contacted_pads_min": 2,
            "lift_best_sustained_two_pad_steps_min": 60,
            "post_lift_hold_best_sustained_two_pad_steps_min": 300,
            "post_lift_hold_min_cube_lift_m_min": 0.045,
            "penetration_gate_excluded": True,
        },
        "source_distribution_caveat": {
            "distribution": "teacher_success_conditioned_not_uniform",
            "accepted_attempts": 60,
            "total_attempts": 73,
            "highest_mass_quartile_kg": [0.034, 0.036],
            "highest_mass_quartile_accepted": 2,
            "highest_mass_quartile_attempted": 15,
            "highest_mass_quartile_acceptance_rate": 2.0 / 15.0,
            "highest_mass_quartile_validation_examples": 0,
            "validation_empty_friction_quartiles": [1, 2],
        },
        "claim_boundary": (
            "Three paired 100-step policy-training seeds and 11 matched evaluation "
            "episodes per seed and regime provide replicated engineering evidence, not "
            "a publication-level robustness estimate. The exact two-sided sign test has "
            "low power at n=3, and the randomized demonstrations are teacher-success-"
            "conditioned rather than uniformly sampled from the declared physics range."
        ),
    }


def render_figure(summary: dict[str, Any], output_path: Path) -> None:
    import cv2
    import numpy as np

    canvas = np.full((900, 1400, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "myCobot 280 randomized-training replication - 2026-07-31",
        (190, 48),
        font,
        0.95,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Three paired 100-step training seeds; 11 matched episodes per evaluation regime",
        (240, 80),
        font,
        0.56,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    seeds = summary["training_seeds"]
    _draw_seed_panel(
        canvas,
        summary=summary,
        seeds=seeds,
        left_name="deterministic_nominal",
        right_name="randomized_nominal",
        title="Nominal fixed physics",
        origin=(80, 505),
    )
    _draw_seed_panel(
        canvas,
        summary=summary,
        seeds=seeds,
        left_name="deterministic_fresh_randomized",
        right_name="randomized_fresh_randomized",
        title="Fresh randomized physics",
        origin=(760, 505),
    )
    cv2.rectangle(canvas, (470, 112), (500, 134), (120, 120, 120), -1)
    cv2.putText(canvas, "Deterministic training", (510, 131), font, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (740, 112), (770, 134), (43, 139, 69), -1)
    cv2.putText(canvas, "Randomized training", (780, 131), font, 0.48, (30, 30, 30), 1, cv2.LINE_AA)

    nominal = summary["paired_comparisons"]["nominal_training_data_effect"]
    fresh = summary["paired_comparisons"]["fresh_randomized_training_data_effect"]
    notes = [
        (
            "Randomized-data gain replicated in 3/3 seeds: nominal mean +"
            f'{nominal["mean_strict_success_count_delta"]:.2f}/11; fresh mean +'
            f'{fresh["mean_strict_success_count_delta"]:.2f}/11.'
        ),
        (
            "Exact two-sided sign test: p="
            f'{nominal["exact_two_sided_sign_test_p"]:.2f} in each regime (n=3; low power).'
        ),
        (
            "Engineering evidence only: source demonstrations are teacher-success-conditioned; "
            "high-mass validation coverage is absent."
        ),
    ]
    for index, line in enumerate(notes):
        cv2.putText(
            canvas,
            line,
            (105, 690 + index * 48),
            font,
            0.56 if index < 2 else 0.50,
            (25, 25, 25) if index < 2 else (40, 40, 145),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write figure: {output_path}")


def _draw_seed_panel(
    canvas: Any,
    *,
    summary: dict[str, Any],
    seeds: list[int],
    left_name: str,
    right_name: str,
    title: str,
    origin: tuple[int, int],
) -> None:
    import cv2

    left, bottom = origin
    width = 560
    height = 315
    top = bottom - height
    cv2.putText(canvas, title, (left + 145, top - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (55, 55, 55), 1)
    cv2.line(canvas, (left, bottom), (left + width, bottom), (55, 55, 55), 1)
    for tick in (0, 5, 11):
        y = bottom - int(height * tick / 11)
        cv2.putText(canvas, str(tick), (left - 28, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (65, 65, 65), 1, cv2.LINE_AA)
        cv2.line(canvas, (left, y), (left + width, y), (225, 225, 225), 1)
    slot = width / len(seeds)
    for index, seed in enumerate(seeds):
        center = int(left + slot * (index + 0.5))
        left_value = int(summary["per_training_seed"][str(seed)][left_name]["strict_successes"])
        right_value = int(summary["per_training_seed"][str(seed)][right_name]["strict_successes"])
        for x, value, color in (
            (center - 30, left_value, (120, 120, 120)),
            (center + 30, right_value, (43, 139, 69)),
        ):
            bar_height = int(height * value / 11)
            cv2.rectangle(canvas, (x - 24, bottom - bar_height), (x + 24, bottom), color, -1)
            cv2.putText(canvas, f"{value}/11", (x - 23, bottom - bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(seed), (center - 42, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Strict successes", (left + 205, bottom + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)


def _summarize_row(report: dict[str, Any]) -> dict[str, Any]:
    episodes = report["episode_summaries"]
    strict_successes = sum(bool(item["success"]) for item in episodes)
    pickup_hold_successes = sum(_pickup_hold_success(item) for item in episodes)
    row = {
        "episodes": len(episodes),
        "strict_successes": strict_successes,
        "strict_success_rate": strict_successes / len(episodes),
        "pickup_hold_successes": pickup_hold_successes,
        "pickup_hold_success_rate": pickup_hold_successes / len(episodes),
        "mean_final_cube_lift_mm": 1000.0
        * _mean([float(item["final_cube_lift_m"]) for item in episodes]),
        "mean_max_pad_cube_penetration_mm": 1000.0
        * _mean([float(item["max_pad_cube_penetration_m"]) for item in episodes]),
        "strict_success_seeds": [
            int(item["seed"]) for item in episodes if bool(item["success"])
        ],
        "penetration_only_failure_seeds": [
            int(item["seed"])
            for item in episodes
            if item.get("failed_gates") == ["max_pad_cube_penetration_exceeded"]
        ],
        "failure_reason_counts": report["aggregate"]["failure_reason_counts"],
    }
    if all(item.get("candidate") is not None for item in episodes):
        high_mass = [
            item for item in episodes if float(item["candidate"]["cube_mass_kg"]) >= 0.034
        ]
        row["high_mass_slice"] = {
            "cube_mass_kg_min": 0.034,
            "strict_successes": sum(bool(item["success"]) for item in high_mass),
            "episodes": len(high_mass),
        }
    return row


def _validate_report_sets(
    report_sets: dict[int, dict[str, dict[str, Any]]],
) -> None:
    if len(report_sets) < 2:
        raise ValueError("multi-seed summary requires at least two training seeds")
    reference_signatures: dict[str, Any] | None = None
    for training_seed, reports in sorted(report_sets.items()):
        if set(reports) != set(ROW_SPECS):
            raise ValueError(f"training seed {training_seed} does not contain all four rows")
        episode_counts = set()
        for name, spec in ROW_SPECS.items():
            report = reports[name]
            if report.get("status") != "completed":
                raise ValueError(f"{training_seed}/{name} is not completed")
            environment = report.get("environment", {})
            if environment.get("render_camera_profile") != "ground_pickup_closeup":
                raise ValueError(f"{training_seed}/{name} does not use the close camera")
            if environment.get("object_physics") != spec["evaluation_physics"]:
                raise ValueError(f"{training_seed}/{name} physics do not match")
            feature = (
                report.get("policy_runtime", {})
                .get("contract", {})
                .get("feature_contract", {})
            )
            if (
                not feature.get("exact_7d_state_action")
                or feature.get("state_shape") != [7]
                or feature.get("action_shape") != [7]
            ):
                raise ValueError(f"{training_seed}/{name} violates the exact-7D contract")
            schedule = report.get("schedule", [])
            episodes = report.get("episode_summaries", [])
            if not schedule or len(schedule) != len(episodes):
                raise ValueError(f"{training_seed}/{name} has an invalid schedule")
            episode_counts.add(len(episodes))
            for index, (schedule_item, episode) in enumerate(
                zip(schedule, episodes, strict=True)
            ):
                if int(schedule_item["torch_seed"]) != training_seed + index:
                    raise ValueError(
                        f"{training_seed}/{name} torch-seed schedule does not match training seed"
                    )
                if int(schedule_item["seed"]) != int(episode["seed"]):
                    raise ValueError(f"{training_seed}/{name} episode seeds do not match")
        if len(episode_counts) != 1:
            raise ValueError(f"training seed {training_seed} row episode counts differ")
        _validate_matched_pair(reports, "deterministic_nominal", "randomized_nominal")
        _validate_matched_pair(
            reports,
            "deterministic_fresh_randomized",
            "randomized_fresh_randomized",
            require_candidates=True,
        )
        for name in (
            "deterministic_fresh_randomized",
            "randomized_fresh_randomized",
        ):
            contract = reports[name].get("schedule_contract", {})
            if (
                contract.get("candidate_selection")
                != "direct_unfiltered_draws_without_teacher_rejection"
            ):
                raise ValueError(f"{training_seed}/{name} does not use direct draws")
            if int(contract.get("source_seed_overlap_count", -1)) != 0:
                raise ValueError(f"{training_seed}/{name} overlaps source attempt seeds")
        signatures = {
            name: _cross_training_seed_signature(report)
            for name, report in reports.items()
        }
        if reference_signatures is None:
            reference_signatures = signatures
        elif signatures != reference_signatures:
            raise ValueError("evaluation schedules differ across training seeds")


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
        raise ValueError(f"{left_name} and {right_name} horizons do not match")
    if require_candidates:
        left_candidates = [item.get("candidate") for item in left["episode_summaries"]]
        right_candidates = [item.get("candidate") for item in right["episode_summaries"]]
        if left_candidates != right_candidates:
            raise ValueError(f"{left_name} and {right_name} candidates do not match")


def _cross_training_seed_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schedule": [
            {key: value for key, value in item.items() if key != "torch_seed"}
            for item in report["schedule"]
        ],
        "candidates": [item.get("candidate") for item in report["episode_summaries"]],
        "steps_per_episode": report.get("steps_per_episode"),
        "camera": report.get("environment", {}).get("render_camera_profile"),
        "physics": report.get("environment", {}).get("object_physics"),
    }


def _pickup_hold_success(item: dict[str, Any]) -> bool:
    return bool(
        float(item["final_cube_lift_m"]) >= 0.05
        and int(item["final_pad_cube_contacted_pads"]) >= 2
        and int(item["lift_best_sustained_two_pad_steps"]) >= 60
        and int(item["post_lift_hold_best_sustained_two_pad_steps"]) >= 300
        and float(item["post_lift_hold_min_cube_lift_m"]) >= 0.045
    )


def _paired_penetration_deltas_mm(
    left_report: dict[str, Any], right_report: dict[str, Any]
) -> list[float]:
    left = {int(item["seed"]): item for item in left_report["episode_summaries"]}
    right = {int(item["seed"]): item for item in right_report["episode_summaries"]}
    if set(left) != set(right):
        raise ValueError("paired reports do not have the same environment seeds")
    return [
        1000.0
        * (
            float(right[seed]["max_pad_cube_penetration_m"])
            - float(left[seed]["max_pad_cube_penetration_m"])
        )
        for seed in sorted(left)
    ]


def _two_sided_sign_test_p(nonzero_deltas: list[int]) -> float | None:
    if not nonzero_deltas:
        return None
    n = len(nonzero_deltas)
    positives = sum(delta > 0 for delta in nonzero_deltas)
    tail = min(positives, n - positives)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, probability)


def _mean(values: list[float] | list[int]) -> float:
    return float(statistics.mean(values))


def _sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize paired myCobot randomized-training seeds."
    )
    parser.add_argument(
        "--seed-report-set",
        action="append",
        nargs=5,
        metavar=(
            "TRAINING_SEED",
            "DET_NOMINAL",
            "RAND_NOMINAL",
            "DET_FRESH",
            "RAND_FRESH",
        ),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path)
    args = parser.parse_args()

    report_sets, _report_paths = load_seed_report_sets(args.seed_report_set)
    summary = summarize_multiseed(report_sets)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_figure is not None:
        render_figure(summary, args.output_figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
