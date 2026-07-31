#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROW_SPECS = {
    "base_wide": {
        "label": "Base + wide",
        "camera": "full_robot",
    },
    "finetuned_wide": {
        "label": "Fine-tuned + wide",
        "camera": "full_robot",
    },
    "base_close": {
        "label": "Base + close",
        "camera": "ground_pickup_closeup",
    },
    "finetuned_close": {
        "label": "Fine-tuned + close",
        "camera": "ground_pickup_closeup",
    },
}
PAIR_SPECS = {
    "finetuning_effect_wide": ("base_wide", "finetuned_wide"),
    "finetuning_effect_close": ("base_close", "finetuned_close"),
    "camera_effect_base": ("base_wide", "base_close"),
    "camera_effect_finetuned": ("finetuned_wide", "finetuned_close"),
}


def load_reports(eval_root: Path) -> dict[str, dict[str, Any]]:
    return {
        name: json.loads((eval_root / name / "eval_report.json").read_text(encoding="utf-8"))
        for name in ROW_SPECS
    }


def summarize_reports(
    reports: dict[str, dict[str, Any]],
    *,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    _validate_reports(reports)
    rows: dict[str, Any] = {}
    for name, spec in ROW_SPECS.items():
        report = reports[name]
        episodes = report["episode_summaries"]
        rows[name] = {
            "label": spec["label"],
            "camera_profile": spec["camera"],
            "episodes": len(episodes),
            "successful_episodes": sum(bool(item["success"]) for item in episodes),
            "success_rate": _mean([float(bool(item["success"])) for item in episodes]),
            "mean_final_cube_lift_mm": 1000.0
            * _mean([float(item["final_cube_lift_m"]) for item in episodes]),
            "mean_post_hold_min_cube_lift_mm": 1000.0
            * _mean([float(item["post_lift_hold_min_cube_lift_m"]) for item in episodes]),
            "mean_max_pad_cube_penetration_mm": 1000.0
            * _mean([float(item["max_pad_cube_penetration_m"]) for item in episodes]),
            "max_pad_cube_penetration_mm": 1000.0
            * max(float(item["max_pad_cube_penetration_m"]) for item in episodes),
            "mean_clipped_action_values": _mean(
                [float(item["clipped_action_values"]) for item in episodes]
            ),
            "failure_reason_counts": report["aggregate"]["failure_reason_counts"],
            "successful_seeds": [
                int(item["seed"]) for item in episodes if bool(item["success"])
            ],
        }

    pairs: dict[str, Any] = {}
    for name, (left_name, right_name) in PAIR_SPECS.items():
        left = _episodes_by_seed(reports[left_name])
        right = _episodes_by_seed(reports[right_name])
        seeds = sorted(left)
        lift_deltas = [
            1000.0
            * (
                float(right[seed]["final_cube_lift_m"])
                - float(left[seed]["final_cube_lift_m"])
            )
            for seed in seeds
        ]
        left_success = {seed for seed in seeds if bool(left[seed]["success"])}
        right_success = {seed for seed in seeds if bool(right[seed]["success"])}
        pairs[name] = {
            "left": left_name,
            "right": right_name,
            "matched_seeds": seeds,
            "mean_final_lift_delta_mm": _mean(lift_deltas),
            "positive_lift_delta_seeds": sum(delta > 0.0 for delta in lift_deltas),
            "success_count_delta": len(right_success) - len(left_success),
            "added_success_seeds": sorted(right_success - left_success),
            "lost_success_seeds": sorted(left_success - right_success),
        }

    return {
        "operation": "summarize_mycobot280_camera_ablation",
        "status": "passed",
        "eval_root": str(eval_root.resolve()) if eval_root is not None else None,
        "rows": rows,
        "paired_comparisons": pairs,
        "schedule": reports["base_wide"]["schedule"],
        "steps_per_episode": reports["base_wide"]["steps_per_episode"],
        "claim_boundary": (
            "Matched 11-environment-seed engineering evidence from one training seed; "
            "repeat training seeds before publication-level camera-effect claims."
        ),
    }


def render_figure(
    summary: dict[str, Any],
    *,
    wide_frame: Path,
    close_frame: Path,
    output_path: Path,
) -> None:
    import cv2
    import numpy as np

    names = list(ROW_SPECS)
    labels = [
        ("Base", "wide"),
        ("Fine-tuned", "wide"),
        ("Base", "close"),
        ("Fine-tuned", "close"),
    ]
    colors = [(120, 112, 104), (92, 122, 31), (53, 95, 180), (191, 111, 39)]
    rows = summary["rows"]
    canvas = np.full((1000, 1400, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "myCobot 280 matched camera ablation - 2026-07-31",
        (260, 38),
        font,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Fixed physics, identical 11 seeds/yaws, one 100-step training seed",
        (340, 70),
        font,
        0.62,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )

    for x, frame_path, title in (
        (105, wide_frame, "Wide camera: cube occupies 0.02-0.07%"),
        (805, close_frame, "Close camera: cube occupies 3.0-3.7%"),
    ):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read camera frame: {frame_path}")
        image = cv2.resize(image, (420, 420), interpolation=cv2.INTER_NEAREST)
        canvas[125:545, x : x + 420] = image
        cv2.rectangle(canvas, (x, 125), (x + 419, 544), (35, 35, 35), 2)
        cv2.putText(
            canvas,
            title,
            (x - 15, 110),
            font,
            0.56,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    success = [100.0 * float(rows[name]["success_rate"]) for name in names]
    lifts = [float(rows[name]["mean_final_cube_lift_mm"]) for name in names]
    _draw_bar_chart(
        canvas,
        values=success,
        labels=labels,
        colors=colors,
        origin=(75, 930),
        size=(580, 310),
        maximum=55.0,
        title="Strict success (%)",
        value_labels=[
            f'{int(rows[name]["successful_episodes"])}/{int(rows[name]["episodes"])}'
            for name in names
        ],
        threshold=None,
    )
    _draw_bar_chart(
        canvas,
        values=lifts,
        labels=labels,
        colors=colors,
        origin=(750, 930),
        size=(580, 310),
        maximum=70.0,
        title="Mean final cube lift (mm)",
        value_labels=[f"{value:.1f}" for value in lifts],
        threshold=50.0,
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
    cv2.putText(
        canvas,
        title,
        (left, top - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    if threshold is not None:
        y = bottom - int(height * threshold / maximum)
        for x in range(left, left + width, 16):
            cv2.line(
                canvas,
                (x, y),
                (min(x + 8, left + width), y),
                (34, 34, 178),
                2,
            )
        cv2.putText(
            canvas,
            "50 mm gate",
            (left + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (34, 34, 178),
            1,
            cv2.LINE_AA,
        )
    slot = width / len(values)
    bar_width = 76
    for index, (value, color, label, value_label) in enumerate(
        zip(values, colors, labels, value_labels, strict=True)
    ):
        center = int(left + slot * (index + 0.5))
        bar_height = int(height * value / maximum)
        cv2.rectangle(
            canvas,
            (center - bar_width // 2, bottom - bar_height),
            (center + bar_width // 2, bottom),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            value_label,
            (center - 22, bottom - bar_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label[0],
            (center - 42, bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label[1],
            (center - 25, bottom + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )


def _validate_reports(reports: dict[str, dict[str, Any]]) -> None:
    missing = sorted(set(ROW_SPECS) - set(reports))
    if missing:
        raise ValueError(f"missing camera-ablation rows: {missing}")
    reference_schedule = reports["base_wide"].get("schedule")
    reference_steps = reports["base_wide"].get("steps_per_episode")
    reference_seeds = [int(item["seed"]) for item in reference_schedule]
    for name, spec in ROW_SPECS.items():
        report = reports[name]
        if report.get("status") != "completed":
            raise ValueError(f"{name} is not completed")
        if report.get("schedule") != reference_schedule:
            raise ValueError(f"{name} schedule does not match base_wide")
        if report.get("steps_per_episode") != reference_steps:
            raise ValueError(f"{name} step horizon does not match base_wide")
        if report.get("environment", {}).get("render_camera_profile") != spec["camera"]:
            raise ValueError(f"{name} camera profile does not match its row contract")
        if report.get("environment", {}).get("object_physics") != "fixed":
            raise ValueError(f"{name} object physics are not fixed")
        feature_contract = (
            report.get("policy_runtime", {})
            .get("contract", {})
            .get("feature_contract", {})
        )
        if not feature_contract.get("exact_7d_state_action"):
            raise ValueError(f"{name} does not declare the exact-7D contract")
        if feature_contract.get("state_shape") != [7] or feature_contract.get("action_shape") != [7]:
            raise ValueError(f"{name} state/action shape is not 7D")
        seeds = sorted(int(item["seed"]) for item in report["episode_summaries"])
        if seeds != sorted(reference_seeds):
            raise ValueError(f"{name} episode seeds do not match the schedule")


def _episodes_by_seed(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["seed"]): item for item in report["episode_summaries"]}


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the four-cell myCobot 280 camera ablation."
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path)
    parser.add_argument("--wide-frame", type=Path)
    parser.add_argument("--close-frame", type=Path)
    args = parser.parse_args()

    reports = load_reports(args.eval_root)
    summary = summarize_reports(reports, eval_root=args.eval_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_figure is not None:
        if args.wide_frame is None or args.close_frame is None:
            raise ValueError("--wide-frame and --close-frame are required with --output-figure")
        render_figure(
            summary,
            wide_frame=args.wide_frame,
            close_frame=args.close_frame,
            output_path=args.output_figure,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
