#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_mycobot_280_ground_pickup_deterministic_dataset import run_check as run_deterministic_check  # noqa: E402
from scripts.export_mycobot_280_ground_pickup_randomized_dataset import (  # noqa: E402
    export_randomized_dataset,
    split_uniqueness_audit,
)

DETERMINISTIC_LADDER = "deterministic_predecessor_ladder"
RANDOMIZED_DATASET = "ground_pickup_seeded_randomized_dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Guard the merged myCobot 280 pre-aligned, raw ground-pickup, and deterministic "
            "dataset ladder before validating the randomized dataset exporter."
        )
    )
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/checks/mycobot_280_ground_pickup_randomized_dataset_001"),
    )
    parser.add_argument("--prealigned-episodes", type=int, default=2)
    parser.add_argument("--deterministic-train-episodes", type=int, default=4)
    parser.add_argument("--deterministic-val-episodes", type=int, default=2)
    parser.add_argument("--randomized-train-episodes", type=int, default=10)
    parser.add_argument("--randomized-val-episodes", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2800)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--render-every", type=int, default=999)
    parser.add_argument("--fps", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_check(
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        output_dir=args.output_dir,
        prealigned_episodes=args.prealigned_episodes,
        deterministic_train_episodes=args.deterministic_train_episodes,
        deterministic_val_episodes=args.deterministic_val_episodes,
        randomized_train_episodes=args.randomized_train_episodes,
        randomized_val_episodes=args.randomized_val_episodes,
        max_attempts=args.max_attempts,
        seed=args.seed,
        width=args.width,
        height=args.height,
        render_every=args.render_every,
        fps=args.fps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


def run_check(
    *,
    asset_root: Path,
    official_gripper_root: Path,
    output_dir: Path,
    prealigned_episodes: int,
    deterministic_train_episodes: int,
    deterministic_val_episodes: int,
    randomized_train_episodes: int,
    randomized_val_episodes: int,
    max_attempts: int,
    seed: int,
    width: int,
    height: int,
    render_every: int,
    fps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    deterministic_report = run_deterministic_check(
        asset_root=asset_root,
        official_gripper_root=official_gripper_root,
        output_dir=output_dir / DETERMINISTIC_LADDER,
        prealigned_episodes=prealigned_episodes,
        train_episodes=deterministic_train_episodes,
        val_episodes=deterministic_val_episodes,
        max_attempts=max_attempts,
        width=width,
        height=height,
        render_every=render_every,
        fps=fps,
    )
    stages = [
        {
            "name": DETERMINISTIC_LADDER,
            "status": deterministic_report.get("status"),
            "claim": "Merged pre-aligned, raw ground-pickup, and deterministic dataset capabilities still pass.",
            "report": deterministic_report,
        }
    ]
    if deterministic_report.get("status") != "passed":
        return _write_report(output_dir, _report(stages))

    randomized_root = output_dir / RANDOMIZED_DATASET
    randomized_manifest = export_randomized_dataset(
        output_dir=randomized_root,
        train_episodes=randomized_train_episodes,
        val_episodes=randomized_val_episodes,
        seed=seed,
        max_attempts=max_attempts,
        asset_root=asset_root,
        official_gripper_root=official_gripper_root,
        width=width,
        height=height,
        render_every=render_every,
        fps=fps,
        yaw_min=-0.20,
        yaw_max=0.20,
        axis_jitter_m=0.0,
        side_jitter_m=0.0,
        mass_min_kg=0.028,
        mass_max_kg=0.036,
        cube_friction_min=3.4,
        cube_friction_max=4.0,
    )
    validation = validate_randomized_manifest(
        randomized_manifest,
        dataset_root=randomized_root,
        train_episodes=randomized_train_episodes,
        val_episodes=randomized_val_episodes,
    )
    stages.append(
        {
            "name": RANDOMIZED_DATASET,
            "status": validation["status"],
            "claim": (
                "Seeded randomized pose/mass/friction candidates are rejection-sampled into "
                "non-overlapping train and validation splits using the existing raw-contact criteria."
            ),
            "output_dir": str(randomized_root),
            "validation": validation,
        }
    )
    return _write_report(output_dir, _report(stages))


def validate_randomized_manifest(
    manifest: dict[str, Any],
    *,
    dataset_root: Path,
    train_episodes: int,
    val_episodes: int,
) -> dict[str, Any]:
    errors: list[str] = []
    expected = int(train_episodes) + int(val_episodes)
    splits = manifest.get("splits", {})
    train = splits.get("train", {}) if isinstance(splits, dict) else {}
    validation = splits.get("validation", {}) if isinstance(splits, dict) else {}
    aggregate = manifest.get("aggregate_metrics", {})
    pose_coverage = aggregate.get("pose_coverage", {}) if isinstance(aggregate, dict) else {}
    factor_coverage = aggregate.get("factor_coverage", {}) if isinstance(aggregate, dict) else {}
    declared_split_audit = manifest.get("split_uniqueness_audit", {})

    _expect(errors, "format", manifest.get("format"), "mycobot_jsonl_v1")
    _expect(errors, "schema_version", manifest.get("schema_version"), 2)
    _expect(errors, "generation_mode", manifest.get("generation_mode"), "seeded_randomized_teacher_aligned_rejection_sampled")
    _expect(errors, "randomization_enabled", manifest.get("randomization_enabled"), True)
    _expect(errors, "teacher_attachment_enabled", manifest.get("teacher_attachment_enabled"), False)
    _expect(errors, "object_teleport_during_pickup_lift", manifest.get("object_teleport_during_pickup_lift"), False)
    calibration = manifest.get("randomized_contact_calibration", {})
    _expect(errors, "randomized_contact_calibration.lift_scale", calibration.get("lift_scale"), 1.05)
    _expect(errors, "randomized_contact_calibration.pad_cube_solref", calibration.get("pad_cube_solref"), [0.01, 1.0])
    _expect(errors, "requested_episodes", manifest.get("requested_episodes"), expected)
    _expect(errors, "accepted_episodes", manifest.get("accepted_episodes"), expected)
    _expect(errors, "failed_episodes", manifest.get("failed_episodes"), [])
    _expect(errors, "train.accepted_episodes", train.get("accepted_episodes"), train_episodes)
    _expect(errors, "validation.accepted_episodes", validation.get("accepted_episodes"), val_episodes)

    if int(pose_coverage.get("unique_pose_count", -1)) != expected:
        errors.append("every accepted episode must have a unique initial cube XY pose")
    if int(pose_coverage.get("unique_trajectory_hashes", -1)) != expected:
        errors.append("every accepted episode must have a unique trajectory hash")
    for factor in ("cube_mass_kg", "cube_friction"):
        coverage = factor_coverage.get(factor)
        if not isinstance(coverage, dict) or float(coverage.get("span", 0.0)) <= 0.0:
            errors.append(f"factor coverage {factor} must have a positive span")

    if float(aggregate.get("min_final_cube_lift_m", 0.0)) < 0.05:
        errors.append("minimum final lift is below 50 mm")
    if int(aggregate.get("min_post_lift_hold_sustained_two_pad_steps", 0)) < 300:
        errors.append("post-lift two-pad hold is below 300 steps")
    if float(aggregate.get("min_post_lift_hold_cube_lift_m", 0.0)) < 0.045:
        errors.append("post-lift hold height is below 45 mm")
    if float(aggregate.get("max_pad_cube_penetration_m", 999.0)) > 0.003:
        errors.append("pad/cube penetration exceeds 3 mm")
    if not 0.0 < float(manifest.get("acceptance_rate", 0.0)) <= 1.0:
        errors.append("acceptance_rate must be in (0, 1]")

    summaries = [
        summary
        for split in (train, validation)
        for summary in split.get("episode_summaries", [])
        if isinstance(summary, dict)
    ]
    if len(summaries) != expected:
        errors.append(f"found {len(summaries)} accepted summaries, expected {expected}")
    train_summaries = [summary for summary in train.get("episode_summaries", []) if isinstance(summary, dict)]
    validation_summaries = [
        summary for summary in validation.get("episode_summaries", []) if isinstance(summary, dict)
    ]
    recomputed_split_audit = split_uniqueness_audit(train_summaries, validation_summaries)
    for field, recomputed in recomputed_split_audit.items():
        if int(declared_split_audit.get(field, -1)) != int(recomputed):
            errors.append(
                f"declared split audit {field}={declared_split_audit.get(field)!r} "
                f"does not match recomputed value {recomputed}"
            )
    for field in ("seed_overlap_count", "pose_overlap_count", "trajectory_hash_overlap_count", "factor_overlap_count"):
        if int(recomputed_split_audit.get(field, -1)) != 0:
            errors.append(f"recomputed split audit {field} must be zero")
    missing_paths = [
        str(dataset_root / str(summary.get("path", "")))
        for summary in summaries
        if not (dataset_root / str(summary.get("path", ""))).is_file()
    ]
    if missing_paths:
        errors.append(f"missing accepted episode files: {missing_paths[:3]}")
    randomization = manifest.get("randomization", {})
    if not isinstance(randomization, dict):
        errors.append("randomization declaration must be a mapping")
    for summary in summaries:
        candidate = summary.get("candidate")
        if not isinstance(candidate, dict):
            errors.append(f"episode {summary.get('episode_index')} has no candidate factors")
            continue
        _check_candidate_ranges(errors, candidate, randomization)
        _check_episode_rows(errors, dataset_root, manifest, summary)

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "accepted_episodes": manifest.get("accepted_episodes"),
        "rejected_attempt_count": len(manifest.get("rejected_attempts", [])),
        "acceptance_rate": manifest.get("acceptance_rate"),
        "split_uniqueness_audit": recomputed_split_audit,
        "aggregate_metrics": aggregate,
    }


def _check_candidate_ranges(
    errors: list[str],
    candidate: dict[str, Any],
    randomization: dict[str, Any],
) -> None:
    range_fields = {
        "cube_mass_kg": "cube_mass_kg",
        "cube_friction": "cube_friction",
        "yaw_delta_rad": "yaw_delta_rad",
    }
    for candidate_key, manifest_key in range_fields.items():
        limits = randomization.get(manifest_key, {})
        if not isinstance(limits, dict) or "min" not in limits or "max" not in limits:
            errors.append(f"randomization range {manifest_key} must declare min and max")
            continue
        value = float(candidate.get(candidate_key, float("nan")))
        if not float(limits["min"]) <= value <= float(limits["max"]):
            errors.append(f"candidate {candidate_key}={value} is outside declared range")
    for field in ("cube_axis_offset_m", "cube_side_offset_m"):
        limits = randomization.get(field, {})
        if not isinstance(limits, dict) or not {"center", "jitter_min", "jitter_max"} <= limits.keys():
            errors.append(f"randomization range {field} must declare center and jitter bounds")
            continue
        value = float(candidate.get(field, float("nan")))
        minimum = float(limits["center"]) + float(limits["jitter_min"])
        maximum = float(limits["center"]) + float(limits["jitter_max"])
        if not minimum <= value <= maximum:
            errors.append(f"candidate {field}={value} is outside declared range")
    for field in ("support_friction", "pad_friction"):
        declaration = randomization.get(field, {})
        if not isinstance(declaration, dict) or "fixed" not in declaration:
            errors.append(f"randomization {field} must declare a fixed value")
            continue
        value = float(candidate.get(field, float("nan")))
        if value != float(declaration["fixed"]):
            errors.append(f"candidate {field}={value} differs from declared fixed value")


def _check_episode_rows(
    errors: list[str],
    dataset_root: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    path = dataset_root / str(summary.get("path", ""))
    if not path.is_file():
        return
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"failed to parse episode {path}: {exc}")
        return
    expected_frames = int(summary.get("frames", -1))
    if len(rows) != expected_frames:
        errors.append(f"episode {path} has {len(rows)} rows, expected {expected_frames}")
    joint_count = len(manifest.get("joint_names", []))
    if joint_count <= 0 or manifest.get("action_names") != manifest.get("joint_names"):
        errors.append("joint_names must be non-empty and match action_names")
        return
    render_every = int(manifest.get("render_every", 0))
    if render_every <= 0:
        errors.append("render_every must be positive")
        return
    required = {
        "episode_index",
        "split",
        "split_episode_index",
        "frame_index",
        "timestamp",
        "phase",
        "task",
        "observation",
        "action",
        "reward",
        "done",
        "info",
    }
    rendered = 0
    summary_candidate = summary.get("candidate", {})
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"episode {path} row {row_index} must be a JSON object")
            continue
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"episode {path} row {row_index} is missing fields {missing}")
            continue
        identity = {
            "episode_index": summary.get("episode_index"),
            "split": summary.get("split"),
            "split_episode_index": summary.get("split_episode_index"),
        }
        for field, expected in identity.items():
            if row.get(field) != expected:
                errors.append(
                    f"episode {path} row {row_index} {field}={row.get(field)!r} "
                    f"does not match summary {expected!r}"
                )
        if int(row.get("frame_index", -1)) != row_index:
            errors.append(f"episode {path} row {row_index} has a non-sequential frame_index")
        action = row.get("action")
        observation = row.get("observation")
        state = observation.get("state") if isinstance(observation, dict) else None
        if not isinstance(action, list) or len(action) != joint_count:
            errors.append(f"episode {path} row {row_index} has an invalid action shape")
        if not isinstance(state, list) or len(state) != joint_count + 3:
            errors.append(f"episode {path} row {row_index} has an invalid state shape")
        info = row.get("info")
        row_candidate = info.get("candidate", {}) if isinstance(info, dict) else {}
        if any(row_candidate.get(key) != value for key, value in summary_candidate.items()):
            errors.append(f"episode {path} row {row_index} candidate metadata differs from its summary")
        images = observation.get("images", {}) if isinstance(observation, dict) else {}
        image_path = images.get("render") if isinstance(images, dict) else None
        if row_index % render_every == 0:
            rendered += 1
            if not isinstance(image_path, str) or not image_path:
                errors.append(f"episode {path} row {row_index} is missing its rendered image reference")
            elif not (dataset_root / image_path).is_file():
                errors.append(f"episode {path} row {row_index} references missing image {image_path}")
    if rendered != int(summary.get("rendered_frames", -1)):
        errors.append(
            f"episode {path} has {rendered} expected rendered rows, "
            f"summary reports {summary.get('rendered_frames')!r}"
        )


def _expect(errors: list[str], field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{field} {actual!r} != expected {expected!r}")


def _report(stages: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [stage["name"] for stage in stages if stage.get("status") != "passed"]
    return {
        "status": "passed" if not failed else "failed",
        "purpose": (
            "Preserve the merged myCobot 280 dataset ladder and validate bounded seeded "
            "randomization with rejection sampling and split-isolation audits."
        ),
        "protected_capabilities": [DETERMINISTIC_LADDER],
        "new_capability": RANDOMIZED_DATASET,
        "failed_stages": failed,
        "stages": stages,
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "check_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    main()
