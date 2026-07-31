#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
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
REQUIRED_PHASES = (
    "approach_down_to_cube_on_mat",
    "close_on_cube_on_mat",
    "hold_before_lift",
    "lift_from_mat",
    "post_lift_hold",
)
EXPECTED_CAMERA = {
    "profile": "ground_pickup_closeup",
    "target": "initial_cube_plus_35mm_z",
    "distance_m": 0.24,
    "azimuth_deg": 215.0,
    "elevation_deg": -10.0,
}
MIN_RED_CUBE_FRACTION = 0.02


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
        "--existing-dataset-root",
        type=Path,
        help="Validate an existing randomized dataset in place without generating simulation output.",
    )
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
    if args.existing_dataset_root is not None:
        report = validate_existing_dataset(args.existing_dataset_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["status"] == "passed" else 1)

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


def validate_existing_dataset(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "dataset_root": str(dataset_root),
            "errors": [f"cannot read randomized dataset manifest {manifest_path}: {exc}"],
        }
    if not isinstance(manifest, dict):
        return {
            "status": "failed",
            "dataset_root": str(dataset_root),
            "errors": [f"randomized dataset manifest {manifest_path} must be a JSON object"],
        }
    splits = manifest.get("splits", {})
    train = splits.get("train", {}) if isinstance(splits, dict) else {}
    validation = splits.get("validation", {}) if isinstance(splits, dict) else {}
    return validate_randomized_manifest(
        manifest,
        dataset_root=dataset_root,
        train_episodes=int(train.get("accepted_episodes", -1)),
        val_episodes=int(validation.get("accepted_episodes", -1)),
    )


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
    camera = manifest.get("observation_camera", {})

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
    _expect(errors, "image_mime_type", manifest.get("image_mime_type"), "image/bmp")
    _expect(errors, "object_suite.object_color", manifest.get("object_suite", {}).get("object_color"), "red")
    _check_camera_contract(errors, camera)

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
        randomization = {}
    episode_audits: list[dict[str, Any]] = []
    for summary in summaries:
        candidate = summary.get("candidate")
        if not isinstance(candidate, dict):
            errors.append(f"episode {summary.get('episode_index')} has no candidate factors")
            continue
        _check_candidate_ranges(errors, candidate, randomization)
        audit = _check_episode_rows(errors, dataset_root, manifest, summary)
        if audit:
            episode_audits.append(audit)

    rejection_audit = _check_attempt_provenance(
        errors,
        manifest=manifest,
        summaries=summaries,
        randomization=randomization,
    )
    source_audit = _aggregate_episode_audits(errors, episode_audits, expected=expected)

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "accepted_episodes": manifest.get("accepted_episodes"),
        "rejected_attempt_count": len(manifest.get("rejected_attempts", [])),
        "acceptance_rate": manifest.get("acceptance_rate"),
        "split_uniqueness_audit": recomputed_split_audit,
        "aggregate_metrics": aggregate,
        "camera_contract": camera,
        "source_episode_audit": source_audit,
        "attempt_provenance_audit": rejection_audit,
    }


def _check_camera_contract(errors: list[str], camera: Any) -> None:
    if not isinstance(camera, dict):
        errors.append("observation_camera must be a mapping")
        return
    for field, expected in EXPECTED_CAMERA.items():
        _expect(errors, f"observation_camera.{field}", camera.get(field), expected)
    width = camera.get("width_px")
    height = camera.get("height_px")
    if (width is None) != (height is None):
        errors.append("observation_camera width_px and height_px must be declared together")
    if width is not None:
        try:
            if int(width) <= 0 or int(height) <= 0:
                errors.append("observation_camera dimensions must be positive")
        except (TypeError, ValueError):
            errors.append("observation_camera dimensions must be integers")


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
) -> dict[str, Any]:
    path = dataset_root / str(summary.get("path", ""))
    if not path.is_file():
        return {}
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"failed to parse episode {path}: {exc}")
        return {}
    expected_frames = int(summary.get("frames", -1))
    if len(rows) != expected_frames:
        errors.append(f"episode {path} has {len(rows)} rows, expected {expected_frames}")
    joint_count = len(manifest.get("joint_names", []))
    if joint_count <= 0 or manifest.get("action_names") != manifest.get("joint_names"):
        errors.append("joint_names must be non-empty and match action_names")
        return {}
    render_every = int(manifest.get("render_every", 0))
    if render_every <= 0:
        errors.append("render_every must be positive")
        return {}
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
    phase_counts: dict[str, int] = {}
    phase_images: dict[str, list[Path]] = {}
    first_contact_step: int | None = None
    guard_failures = {"cube_mat": 0, "pad_mat": 0, "gripper_mat": 0}
    attached_rows = 0
    post_hold_failures = 0
    max_penetration = 0.0
    final_ground_pickup: dict[str, Any] = {}
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
        if isinstance(info, dict) and bool(info.get("grasp_attached")):
            attached_rows += 1
        ground_pickup = info.get("ground_pickup") if isinstance(info, dict) else None
        if not isinstance(ground_pickup, dict):
            errors.append(f"episode {path} row {row_index} is missing ground_pickup source metrics")
        else:
            phase = str(row.get("phase", ""))
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if ground_pickup.get("phase") != phase:
                errors.append(f"episode {path} row {row_index} ground_pickup phase differs from row phase")
            contacted_pads = int(ground_pickup.get("pad_cube_contacted_pads", -1))
            if contacted_pads > 0 and first_contact_step is None:
                first_contact_step = int(ground_pickup.get("step", row_index))
            mat_guard = ground_pickup.get("mat_guard", {})
            pad_guard = ground_pickup.get("pad_mat_guard", {})
            gripper_guard = ground_pickup.get("gripper_visual_mat_guard", {})
            if not isinstance(mat_guard, dict) or not bool(mat_guard.get("bottom_on_or_above_mat")):
                guard_failures["cube_mat"] += 1
            if row_index == 0 and (not isinstance(mat_guard, dict) or not bool(mat_guard.get("passed"))):
                guard_failures["cube_mat"] += 1
            if not isinstance(pad_guard, dict) or not bool(pad_guard.get("passed")):
                guard_failures["pad_mat"] += 1
            if not isinstance(gripper_guard, dict) or not bool(gripper_guard.get("passed")):
                guard_failures["gripper_mat"] += 1
            depth = ground_pickup.get("pad_cube_contact_depth", {})
            penetration = float(depth.get("max_penetration_m", 0.0)) if isinstance(depth, dict) else 0.0
            max_penetration = max(max_penetration, penetration)
            if phase == "post_lift_hold" and (
                contacted_pads < 2
                or float(ground_pickup.get("cube_lift_m", 0.0))
                < float(manifest.get("success_criteria", {}).get("post_lift_hold_min_cube_lift_m", 0.045))
            ):
                post_hold_failures += 1
            final_ground_pickup = ground_pickup
        images = observation.get("images", {}) if isinstance(observation, dict) else {}
        image_path = images.get("render") if isinstance(images, dict) else None
        if row_index % render_every == 0:
            rendered += 1
            if not isinstance(image_path, str) or not image_path:
                errors.append(f"episode {path} row {row_index} is missing its rendered image reference")
            elif not (dataset_root / image_path).is_file():
                errors.append(f"episode {path} row {row_index} references missing image {image_path}")
            else:
                phase = str(row.get("phase", ""))
                resolved = dataset_root / image_path
                samples = phase_images.setdefault(phase, [])
                if not samples:
                    samples.append(resolved)
                elif samples[-1] != resolved:
                    if len(samples) == 1:
                        samples.append(resolved)
                    else:
                        samples[-1] = resolved
    if rendered != int(summary.get("rendered_frames", -1)):
        errors.append(
            f"episode {path} has {rendered} expected rendered rows, "
            f"summary reports {summary.get('rendered_frames')!r}"
        )
    if attached_rows:
        errors.append(f"episode {path} enables teacher attachment in {attached_rows} rows")
    for guard, count in guard_failures.items():
        if count:
            errors.append(f"episode {path} has {count} failing {guard} source guards")
    missing_source_phases = sorted(set(REQUIRED_PHASES) - set(phase_counts))
    if missing_source_phases:
        errors.append(f"episode {path} has no source rows for phases {missing_source_phases}")
    if (
        rows
        and isinstance(rows[0], dict)
        and int(rows[0].get("info", {}).get("ground_pickup", {}).get("pad_cube_contacted_pads", -1)) != 0
    ):
        errors.append(f"episode {path} does not start clear of the cube")
    if post_hold_failures:
        errors.append(f"episode {path} has {post_hold_failures} invalid post-lift hold rows")
    criteria = manifest.get("success_criteria", {})
    if final_ground_pickup and (
        float(final_ground_pickup.get("cube_lift_m", 0.0)) < float(criteria.get("final_cube_lift_m", 0.05))
        or int(final_ground_pickup.get("pad_cube_contacted_pads", 0))
        < int(criteria.get("final_gripper_cube_contact_pads", 2))
    ):
        errors.append(f"episode {path} terminal row does not satisfy lift/contact criteria")
    if first_contact_step != summary.get("first_contact_step"):
        errors.append(
            f"episode {path} first contact step {first_contact_step!r} "
            f"does not match summary {summary.get('first_contact_step')!r}"
        )
    summary_checks = {
        "first_frame_pad_cube_contacted_pads": 0,
        "cube_bottom_on_or_above_mat_all_steps": True,
        "pad_mat_guard_passed_all_steps": True,
        "gripper_visual_mat_guard_passed_all_steps": True,
        "final_gripper_cube_contact_pads": 2,
    }
    for field, expected in summary_checks.items():
        if summary.get(field) != expected:
            errors.append(f"episode {path} summary {field}={summary.get(field)!r}, expected {expected!r}")
    maximum_allowed = float(criteria.get("max_pad_cube_penetration_m", 0.003))
    if max_penetration > maximum_allowed:
        errors.append(
            f"episode {path} row-level pad/cube penetration {max_penetration:.6f} exceeds {maximum_allowed:.6f}"
        )
    if abs(max_penetration - float(summary.get("max_pad_cube_penetration_m", -1.0))) > 1e-9:
        errors.append(f"episode {path} row-level penetration does not match its summary")

    if render_every == 1:
        missing_phases = sorted(set(REQUIRED_PHASES) - set(phase_images))
        if missing_phases:
            errors.append(f"episode {path} has no rendered visibility samples for phases {missing_phases}")
    visibility: list[dict[str, Any]] = []
    for phase in REQUIRED_PHASES:
        for image in phase_images.get(phase, []):
            try:
                metrics = _bmp_red_object_metrics(image)
            except (OSError, ValueError, struct.error) as exc:
                errors.append(f"episode {path} has unreadable visibility sample {image}: {exc}")
                continue
            metrics["phase"] = phase
            metrics["path"] = str(image.relative_to(dataset_root))
            visibility.append(metrics)
            if float(metrics["red_fraction"]) < MIN_RED_CUBE_FRACTION:
                errors.append(
                    f"episode {path} {phase} cube occupancy {metrics['red_fraction']:.4f} "
                    f"is below {MIN_RED_CUBE_FRACTION:.4f}"
                )
            if bool(metrics["touches_border"]):
                errors.append(f"episode {path} {phase} cube visibility touches the image border")
    declared_width = manifest.get("observation_camera", {}).get("width_px")
    declared_height = manifest.get("observation_camera", {}).get("height_px")
    for metrics in visibility:
        if declared_width is not None and (
            int(metrics["width_px"]) != int(declared_width)
            or int(metrics["height_px"]) != int(declared_height)
        ):
            errors.append(f"episode {path} visibility sample dimensions differ from camera contract")
    return {
        "episode_index": summary.get("episode_index"),
        "rows": len(rows),
        "phase_counts": phase_counts,
        "sampled_images": len(visibility),
        "visibility": visibility,
        "max_row_penetration_m": max_penetration,
    }


def _bmp_red_object_metrics(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 54 or payload[:2] != b"BM":
        raise ValueError("not a BMP file")
    pixel_offset = struct.unpack_from("<I", payload, 10)[0]
    width = struct.unpack_from("<i", payload, 18)[0]
    signed_height = struct.unpack_from("<i", payload, 22)[0]
    bits_per_pixel = struct.unpack_from("<H", payload, 28)[0]
    compression = struct.unpack_from("<I", payload, 30)[0]
    height = abs(signed_height)
    if width <= 0 or height <= 0 or bits_per_pixel not in (24, 32) or compression != 0:
        raise ValueError("unsupported BMP layout")
    bytes_per_pixel = bits_per_pixel // 8
    row_stride = ((width * bits_per_pixel + 31) // 32) * 4
    if pixel_offset + row_stride * height > len(payload):
        raise ValueError("truncated BMP pixels")
    red_count = 0
    min_x, min_y = width, height
    max_x = max_y = -1
    for display_y in range(height):
        source_y = display_y if signed_height < 0 else height - 1 - display_y
        row_start = pixel_offset + source_y * row_stride
        for x in range(width):
            pixel = row_start + x * bytes_per_pixel
            blue, green, red = payload[pixel : pixel + 3]
            if red >= 130 and red - green >= 45 and red - blue >= 35:
                red_count += 1
                min_x, min_y = min(min_x, x), min(min_y, display_y)
                max_x, max_y = max(max_x, x), max(max_y, display_y)
    touches_border = bool(
        red_count and (min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1)
    )
    return {
        "width_px": width,
        "height_px": height,
        "red_pixel_count": red_count,
        "red_fraction": red_count / float(width * height),
        "touches_border": touches_border,
    }


def _aggregate_episode_audits(
    errors: list[str],
    audits: list[dict[str, Any]],
    *,
    expected: int,
) -> dict[str, Any]:
    if len(audits) != expected:
        errors.append(f"source audit covered {len(audits)} episodes, expected {expected}")
    visibility = [sample for audit in audits for sample in audit.get("visibility", [])]
    resolutions = sorted({(int(item["width_px"]), int(item["height_px"])) for item in visibility})
    if len(resolutions) > 1:
        errors.append(f"randomized source images have inconsistent resolutions: {resolutions}")
    phase_sample_counts = {
        phase: sum(1 for item in visibility if item.get("phase") == phase)
        for phase in REQUIRED_PHASES
    }
    return {
        "episodes_checked": len(audits),
        "rows_checked": sum(int(audit.get("rows", 0)) for audit in audits),
        "visibility_samples_checked": len(visibility),
        "phase_visibility_sample_counts": phase_sample_counts,
        "detected_resolutions_px": [list(resolution) for resolution in resolutions],
        "min_red_cube_fraction": min((float(item["red_fraction"]) for item in visibility), default=0.0),
        "max_red_cube_fraction": max((float(item["red_fraction"]) for item in visibility), default=0.0),
        "border_touch_samples": sum(bool(item["touches_border"]) for item in visibility),
        "max_row_penetration_m": max(
            (float(audit.get("max_row_penetration_m", 0.0)) for audit in audits),
            default=0.0,
        ),
    }


def _check_attempt_provenance(
    errors: list[str],
    *,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    randomization: dict[str, Any],
) -> dict[str, Any]:
    rejected = manifest.get("rejected_attempts", [])
    if not isinstance(rejected, list):
        errors.append("rejected_attempts must be a list")
        rejected = []
    attempts = [*summaries, *[item for item in rejected if isinstance(item, dict)]]
    declared_attempts = int(manifest.get("attempt_count", -1))
    indices = [int(item.get("attempt_index", -1)) for item in attempts]
    seeds = []
    for item in attempts:
        candidate = item.get("candidate")
        seed = item.get("seed")
        if seed is None and isinstance(candidate, dict):
            seed = candidate.get("spawn_seed")
        try:
            seeds.append(int(seed))
        except (TypeError, ValueError):
            seeds.append(-1)
    if len(attempts) != declared_attempts:
        errors.append(f"attempt_count={declared_attempts} but accepted plus rejected attempts={len(attempts)}")
    if sorted(indices) != list(range(max(declared_attempts, 0))):
        errors.append("accepted/rejected attempt indexes must be unique and contiguous")
    if len(set(seeds)) != len(seeds):
        errors.append("accepted/rejected attempt seeds must be unique")
    for item in rejected:
        if not isinstance(item, dict):
            errors.append("every rejected attempt must be a mapping")
            continue
        if item.get("success") is not False or not item.get("reason"):
            errors.append(f"rejected attempt {item.get('attempt_index')} lacks failure provenance")
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            errors.append(f"rejected attempt {item.get('attempt_index')} lacks candidate factors")
            continue
        _check_candidate_ranges(errors, candidate, randomization)
        if int(candidate.get("spawn_seed", -1)) != int(item.get("seed", -2)):
            errors.append(f"rejected attempt {item.get('attempt_index')} seed differs from candidate")
    return {
        "declared_attempts": declared_attempts,
        "accepted_attempts": len(summaries),
        "rejected_attempts": len(rejected),
        "unique_attempt_indices": len(set(indices)),
        "unique_attempt_seeds": len(set(seeds)),
    }


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
