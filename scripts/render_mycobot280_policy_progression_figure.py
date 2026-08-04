#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_progression_series(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("operation") != "summarize_mycobot280_policy_claims":
        raise ValueError("input is not a myCobot policy claim summary")
    if summary.get("status") != "passed":
        raise ValueError("policy claim summary is not passed")

    matched = summary["matched_schedule_rows"]
    base = matched["base"]
    deterministic = matched["deterministic_finetuned"]
    randomized = matched["randomized_finetuned"]
    multi = summary["fresh_randomized_multiseed"]
    training_seeds = [int(seed) for seed in multi["training_seeds"]]
    det_values = [
        int(value)
        for value in multi["deterministic"]["strict_successes_per_training_seed"]
    ]
    rand_values = [
        int(value)
        for value in multi["randomized"]["strict_successes_per_training_seed"]
    ]
    if not (len(training_seeds) == len(det_values) == len(rand_values) == 3):
        raise ValueError("main figure requires exactly three paired training seeds")

    episodes = int(base["episodes"])
    if episodes != 11:
        raise ValueError("main figure requires the matched 11-episode schedule")
    if int(multi["deterministic"]["episodes"]) != episodes * len(training_seeds):
        raise ValueError("deterministic pooled episode count does not match 3 x 11")
    if int(multi["randomized"]["episodes"]) != episodes * len(training_seeds):
        raise ValueError("randomized pooled episode count does not match 3 x 11")
    if det_values[0] != int(deterministic["strict_successes"]):
        raise ValueError("matched deterministic control does not anchor the first seed")
    if rand_values[0] != int(randomized["strict_successes"]):
        raise ValueError("matched randomized control does not anchor the first seed")

    return {
        "shared_base": {
            "strict_successes": int(base["strict_successes"]),
            "pickup_hold_successes": int(base["pickup_hold_successes"]),
            "episodes": episodes,
            "model_count": 1,
        },
        "training_pairs": [
            {
                "training_seed": seed,
                "deterministic_strict_successes": det,
                "randomized_strict_successes": rand,
                "episodes": episodes,
            }
            for seed, det, rand in zip(
                training_seeds, det_values, rand_values, strict=True
            )
        ],
        "positive_pair_count": int(multi["positive_training_seed_count"]),
        "sign_test_p": float(multi["exact_two_sided_sign_test_p"]),
        "evaluation_contract": summary["evaluation_contract"],
    }


def render_progression_figure(series: dict[str, Any], output_path: Path) -> None:
    import cv2
    import numpy as np

    canvas = np.full((850, 1400, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "myCobot 280 strict-success progression",
        (365, 55),
        font,
        1.08,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Fresh randomized physics | close camera | 11 matched unfiltered episodes per point",
        (325, 92),
        font,
        0.58,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )

    left, right, top, bottom = 120, 1260, 155, 650
    x_positions = [260, 700, 1140]
    for tick in (0, 3, 6, 9, 11):
        y = _value_to_y(tick, top, bottom)
        cv2.line(canvas, (left, y), (right, y), (230, 230, 230), 1)
        cv2.putText(
            canvas,
            str(tick),
            (left - 38, y + 5),
            font,
            0.43,
            (65, 65, 65),
            1,
            cv2.LINE_AA,
        )
    cv2.line(canvas, (left, top), (left, bottom), (45, 45, 45), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (45, 45, 45), 2)
    cv2.putText(
        canvas,
        "Strict successes / 11",
        (left, top - 20),
        font,
        0.54,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )

    labels = ["No fine-tuning", "Deterministic FT", "Randomized FT"]
    label_colors = [(90, 90, 90), (175, 95, 30), (50, 145, 55)]
    for x, label, color in zip(x_positions, labels, label_colors, strict=True):
        width = cv2.getTextSize(label, font, 0.52, 1)[0][0]
        cv2.putText(
            canvas,
            label,
            (x - width // 2, bottom + 42),
            font,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    base = series["shared_base"]
    base_point = (
        x_positions[0],
        _value_to_y(int(base["strict_successes"]), top, bottom),
    )
    colors = [(55, 125, 205), (150, 80, 150), (170, 115, 35)]
    jitters = [-13, 0, 13]
    label_offsets = [(-52, -18), (18, 21), (18, 4)]
    for pair, color, jitter, label_offset in zip(
        series["training_pairs"], colors, jitters, label_offsets, strict=True
    ):
        det_point = (
            x_positions[1] + jitter,
            _value_to_y(pair["deterministic_strict_successes"], top, bottom),
        )
        rand_point = (
            x_positions[2] + jitter,
            _value_to_y(pair["randomized_strict_successes"], top, bottom),
        )
        _draw_dotted_line(canvas, base_point, det_point, color)
        cv2.line(canvas, det_point, rand_point, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, det_point, 9, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, rand_point, 9, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f'{pair["deterministic_strict_successes"]}',
            (det_point[0] - 5, det_point[1] - 15),
            font,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        dx, dy = label_offset
        cv2.putText(
            canvas,
            f'{pair["training_seed"]}: {pair["randomized_strict_successes"]}/11',
            (rand_point[0] + dx, rand_point[1] + dy),
            font,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.drawMarker(
        canvas,
        base_point,
        (75, 75, 75),
        markerType=cv2.MARKER_DIAMOND,
        markerSize=20,
        thickness=3,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Shared base: 0/11 strict, 0/11 pickup+hold",
        (base_point[0] - 105, base_point[1] - 23),
        font,
        0.44,
        (75, 75, 75),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Dotted: shared base reference -> deterministic fine-tuning",
        (145, 735),
        font,
        0.51,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Solid: paired deterministic -> randomized fine-tuning",
        (145, 772),
        font,
        0.51,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Randomized higher in 3/3 training-seed pairs; exact sign test p=0.25 (low power)",
        (650, 735),
        font,
        0.50,
        (45, 45, 145),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Baseline is one shared model, not three independently trained baseline checkpoints.",
        (650, 772),
        font,
        0.47,
        (45, 45, 145),
        1,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write figure: {output_path}")


def _value_to_y(value: int, top: int, bottom: int) -> int:
    return bottom - int((bottom - top) * value / 11)


def _draw_dotted_line(
    canvas: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    import cv2

    distance = math.dist(start, end)
    if distance == 0:
        return
    dash = 12.0
    gap = 8.0
    position = 0.0
    while position < distance:
        segment_end = min(position + dash, distance)
        p1 = (
            int(start[0] + (end[0] - start[0]) * position / distance),
            int(start[1] + (end[1] - start[1]) * position / distance),
        )
        p2 = (
            int(start[0] + (end[0] - start[0]) * segment_end / distance),
            int(start[1] + (end[1] - start[1]) * segment_end / distance),
        )
        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
        position += dash + gap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the myCobot base-to-fine-tuning progression figure."
    )
    parser.add_argument("--claim-summary", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.claim_summary.read_text(encoding="utf-8"))
    series = build_progression_series(summary)
    render_progression_figure(series, args.output_figure)
    print(json.dumps(series, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
