#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import physical_ai_agent.sim.mycobot_nexus_env as nexus  # noqa: E402
import scripts.export_mycobot_280_ground_pickup_teacher_dataset as teacher  # noqa: E402
from scripts.run_mycobot_280_ground_pickup_poc import (  # noqa: E402
    CUBE_AXIS_OFFSET,
    CUBE_HALF_SIZE,
    CUBE_MASS,
    CUBE_SIDE_OFFSET,
    MAT_FRICTION,
    PAD_FRICTION,
    PICKUP_QPOS,
    LIFT_QPOS,
    WORK_MAT_TOP_Z,
)

DEFAULT_TRAIN_EPISODES = 50
DEFAULT_VALIDATION_EPISODES = 10
DEFAULT_YAW_MIN = -0.20
DEFAULT_YAW_MAX = 0.20
DEFAULT_AXIS_JITTER_M = 0.0
DEFAULT_SIDE_JITTER_M = 0.0
DEFAULT_MASS_MIN_KG = 0.028
DEFAULT_MASS_MAX_KG = 0.036
DEFAULT_CUBE_FRICTION_MIN = 3.4
DEFAULT_CUBE_FRICTION_MAX = 4.0
OBJECT_ID = "cube_030"
OBJECT_COLOR = "red"
RENDER_CAMERA_PROFILE = "ground_pickup_closeup"
RANDOMIZED_LIFT_SCALE = 1.05
RANDOMIZED_CONTACT_SOLREF = (0.010, 1.0)


@dataclass(frozen=True)
class RandomizedCandidate:
    attempt_index: int
    spawn_seed: int
    yaw_delta_rad: float
    cube_axis_offset_m: float
    cube_side_offset_m: float
    cube_mass_kg: float
    cube_friction: float
    support_friction: float
    pad_friction: float
    object_id: str = OBJECT_ID
    object_color: str = OBJECT_COLOR

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.spawn_seed,
            round(self.yaw_delta_rad, 9),
            round(self.cube_axis_offset_m, 9),
            round(self.cube_side_offset_m, 9),
            round(self.cube_mass_kg, 9),
            round(self.cube_friction, 9),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a seeded, rejection-sampled myCobot 280 Pi cube-from-mat teacher "
            "dataset with bounded pose, mass, and object-friction randomization."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_randomized_v1"),
    )
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=DEFAULT_VALIDATION_EPISODES)
    parser.add_argument("--seed", type=int, default=2800)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--yaw-min", type=float, default=DEFAULT_YAW_MIN)
    parser.add_argument("--yaw-max", type=float, default=DEFAULT_YAW_MAX)
    parser.add_argument("--axis-jitter-m", type=float, default=DEFAULT_AXIS_JITTER_M)
    parser.add_argument("--side-jitter-m", type=float, default=DEFAULT_SIDE_JITTER_M)
    parser.add_argument("--mass-min-kg", type=float, default=DEFAULT_MASS_MIN_KG)
    parser.add_argument("--mass-max-kg", type=float, default=DEFAULT_MASS_MAX_KG)
    parser.add_argument("--cube-friction-min", type=float, default=DEFAULT_CUBE_FRICTION_MIN)
    parser.add_argument("--cube-friction-max", type=float, default=DEFAULT_CUBE_FRICTION_MAX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_randomized_dataset(
        output_dir=args.output_dir,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        seed=args.seed,
        max_attempts=args.max_attempts,
        asset_root=args.asset_root,
        official_gripper_root=args.official_gripper_root,
        width=args.width,
        height=args.height,
        render_every=args.render_every,
        fps=args.fps,
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        axis_jitter_m=args.axis_jitter_m,
        side_jitter_m=args.side_jitter_m,
        mass_min_kg=args.mass_min_kg,
        mass_max_kg=args.mass_max_kg,
        cube_friction_min=args.cube_friction_min,
        cube_friction_max=args.cube_friction_max,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["accepted_episodes"] != manifest["requested_episodes"] or manifest["failed_episodes"]:
        raise SystemExit(1)


def export_randomized_dataset(
    *,
    output_dir: Path,
    train_episodes: int,
    val_episodes: int,
    seed: int,
    max_attempts: int,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
    render_every: int,
    fps: int,
    yaw_min: float,
    yaw_max: float,
    axis_jitter_m: float,
    side_jitter_m: float,
    mass_min_kg: float,
    mass_max_kg: float,
    cube_friction_min: float,
    cube_friction_max: float,
) -> dict[str, Any]:
    requested = int(train_episodes) + int(val_episodes)
    _validate_request(
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        yaw_min=yaw_min,
        yaw_max=yaw_max,
        axis_jitter_m=axis_jitter_m,
        side_jitter_m=side_jitter_m,
        mass_min_kg=mass_min_kg,
        mass_max_kg=mass_max_kg,
        cube_friction_min=cube_friction_min,
        cube_friction_max=cube_friction_max,
    )
    if max_attempts <= 0:
        max_attempts = max(requested * 5, requested)

    teacher._patch_nexus_work_mat_scene_nodes()
    teacher._prepare_output_root(output_dir)
    split_targets = {"train": int(train_episodes), "validation": int(val_episodes)}
    split_counts = {"train": 0, "validation": 0}
    split_summaries: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    split_schedule = teacher._split_schedule(train_episodes, val_episodes)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_signatures: set[tuple[Any, ...]] = set()
    total_rows = 0

    for candidate in randomized_candidates(
        seed=seed,
        max_attempts=max_attempts,
        yaw_min=yaw_min,
        yaw_max=yaw_max,
        axis_jitter_m=axis_jitter_m,
        side_jitter_m=side_jitter_m,
        mass_min_kg=mass_min_kg,
        mass_max_kg=mass_max_kg,
        cube_friction_min=cube_friction_min,
        cube_friction_max=cube_friction_max,
    ):
        if len(accepted) >= requested:
            break
        if candidate.signature in accepted_signatures:
            rejected.append(
                {
                    "attempt_index": candidate.attempt_index,
                    "seed": candidate.spawn_seed,
                    "success": False,
                    "reason": "duplicate_randomization_signature",
                    "candidate": asdict(candidate),
                }
            )
            continue
        split = split_schedule[len(accepted)]
        split_episode_index = split_counts[split]
        try:
            summary = teacher._export_attempt(
                output_dir=output_dir,
                split=split,
                split_episode_index=split_episode_index,
                global_episode_index=len(accepted),
                attempt_index=candidate.attempt_index,
                seed=candidate.spawn_seed,
                yaw_delta=candidate.yaw_delta_rad,
                asset_root=asset_root,
                official_gripper_root=official_gripper_root,
                width=width,
                height=height,
                fps=fps,
                render_every=render_every,
                cube_axis_offset=candidate.cube_axis_offset_m,
                cube_side_offset=candidate.cube_side_offset_m,
                cube_mass=candidate.cube_mass_kg,
                cube_friction=candidate.cube_friction,
                support_friction=candidate.support_friction,
                pad_friction=candidate.pad_friction,
                candidate_metadata={
                    "spawn_seed": candidate.spawn_seed,
                    "object_id": candidate.object_id,
                    "object_color": candidate.object_color,
                },
                render_camera_profile=RENDER_CAMERA_PROFILE,
                refresh_model_constants=True,
                lift_scale=RANDOMIZED_LIFT_SCALE,
                contact_solref=RANDOMIZED_CONTACT_SOLREF,
            )
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {
                    "attempt_index": candidate.attempt_index,
                    "split_target": split,
                    "seed": candidate.spawn_seed,
                    "success": False,
                    "reason": f"exception: {exc}",
                    "candidate": asdict(candidate),
                }
            )
            continue
        if not summary["success"]:
            rejection = teacher._rejection_from_summary(summary)
            rejection["candidate"] = asdict(candidate)
            rejected.append(rejection)
            continue
        accepted_signatures.add(candidate.signature)
        accepted.append(summary)
        split_summaries[split].append(summary)
        split_counts[split] += 1
        total_rows += int(summary["frames"])

    failed_episodes = [
        {
            "split": split,
            "missing": target - split_counts[split],
            "target": target,
            "accepted": split_counts[split],
        }
        for split, target in split_targets.items()
        if split_counts[split] != target
    ]
    aggregate = teacher._aggregate_metrics(accepted, rejected)
    split_audit = split_uniqueness_audit(split_summaries["train"], split_summaries["validation"])
    attempt_count = len(accepted) + len(rejected)
    manifest = {
        "format": "mycobot_jsonl_v1",
        "schema_version": 2,
        "dataset_id": output_dir.name,
        "robot": "myCobot 280 Pi + adaptive gripper",
        "robot_id": "mycobot_280_pi",
        "model_profile": nexus.MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
        "task": teacher.TASK,
        "dataset_kind": "randomized_ground_pickup_jsonl",
        "generation_mode": "seeded_randomized_teacher_aligned_rejection_sampled",
        "randomization_enabled": True,
        "trajectory": "true_cube_from_work_mat_open_align_close_grasp_lift_post_lift_hold",
        "teacher_attachment_enabled": False,
        "object_teleport_during_pickup_lift": False,
        "zero_gravity_close": True,
        "post_step_snap_enabled": False,
        "cube_starts_on_work_mat": True,
        "object_suite": {
            "suite_id": "object_suite_v0",
            "object_id": OBJECT_ID,
            "object_color": OBJECT_COLOR,
            "dimensions_m": [2.0 * CUBE_HALF_SIZE] * 3,
            "note": "Object factors are recorded in every accepted episode; pad friction stays embodiment-calibrated.",
        },
        "simulator": "MuJoCo",
        "simulator_version": mujoco.__version__,
        "cube_half_size_m": CUBE_HALF_SIZE,
        "work_mat_top_z_m": WORK_MAT_TOP_Z,
        "randomized_contact_calibration": {
            "lift_scale": RANDOMIZED_LIFT_SCALE,
            "pad_cube_solref": list(RANDOMIZED_CONTACT_SOLREF),
            "scope": "randomized exporter only; merged deterministic defaults remain unchanged",
        },
        "success_criteria": {
            "final_cube_lift_m": 0.05,
            "final_gripper_cube_contact_pads": 2,
            "lift_best_sustained_two_pad_steps": 60,
            "post_lift_hold_best_sustained_two_pad_steps": 300,
            "post_lift_hold_min_cube_lift_m": 0.045,
            "max_pad_cube_penetration_m": 0.003,
        },
        "randomization": {
            "sampler": "numpy_pcg64_seeded_per_attempt",
            "root_seed": int(seed),
            "yaw_delta_rad": {"min": float(yaw_min), "max": float(yaw_max)},
            "cube_axis_offset_m": {
                "center": CUBE_AXIS_OFFSET,
                "jitter_min": -float(axis_jitter_m),
                "jitter_max": float(axis_jitter_m),
            },
            "cube_side_offset_m": {
                "center": CUBE_SIDE_OFFSET,
                "jitter_min": -float(side_jitter_m),
                "jitter_max": float(side_jitter_m),
            },
            "cube_mass_kg": {"min": float(mass_min_kg), "max": float(mass_max_kg)},
            "cube_friction": {"min": float(cube_friction_min), "max": float(cube_friction_max)},
            "support_friction": {"fixed": MAT_FRICTION},
            "pad_friction": {"fixed": PAD_FRICTION},
            "base_pickup_qpos": [float(value) for value in PICKUP_QPOS],
            "base_lift_qpos": [float(value) for value in LIFT_QPOS],
        },
        "requested_episodes": requested,
        "accepted_episodes": len(accepted),
        "episodes": len(accepted),
        "passed_episodes": len(accepted),
        "frames": total_rows,
        "attempt_count": attempt_count,
        "acceptance_rate": (len(accepted) / attempt_count) if attempt_count else 0.0,
        "aggregate_metrics": aggregate,
        "split_uniqueness_audit": split_audit,
        "splits": {
            split: {
                "requested_episodes": split_targets[split],
                "accepted_episodes": split_counts[split],
                "episode_summaries": split_summaries[split],
            }
            for split in ("train", "validation")
        },
        "rejected_attempts": rejected,
        "failed_episodes": failed_episodes,
        "fps": int(fps),
        "render_every": int(render_every),
        "image_mime_type": "image/bmp",
        "observation_camera": {
            "profile": RENDER_CAMERA_PROFILE,
            "target": "initial_cube_plus_35mm_z",
            "distance_m": 0.24,
            "azimuth_deg": 215.0,
            "elevation_deg": -10.0,
        },
        "joint_names": teacher.JOINT_NAMES,
        "action_names": teacher.JOINT_NAMES,
        "viewer": {
            "type": "mycobot_jsonl",
            "serve_script": "scripts/serve_so101_dataset_viewer.py",
            "env": f"MYCOBOT_TEMP_DATASETS={output_dir.name}={output_dir}",
        },
        "notes": (
            "Seeded randomized extension of the merged deterministic pose-diverse dataset. "
            "The default sampler varies reachable yaw-derived XY starts, cube mass, and "
            "cube friction while preserving calibrated terminal alignment. Optional terminal "
            "offset jitter is exposed for later robustness experiments but defaults to zero. "
            "Only episodes satisfying the existing raw-contact lift, hold, guard, and "
            "penetration criteria are accepted. Calibrated 280 pad friction and support "
            "friction stay fixed. Broader arm/table collision remains visual/guarded as in "
            "the predecessor POC."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def randomized_candidates(
    *,
    seed: int,
    max_attempts: int,
    yaw_min: float,
    yaw_max: float,
    axis_jitter_m: float,
    side_jitter_m: float,
    mass_min_kg: float,
    mass_max_kg: float,
    cube_friction_min: float,
    cube_friction_max: float,
) -> Iterator[RandomizedCandidate]:
    for attempt_index in range(max_attempts):
        spawn_seed = int(seed) + attempt_index
        rng = np.random.default_rng(spawn_seed)
        yield RandomizedCandidate(
            attempt_index=attempt_index,
            spawn_seed=spawn_seed,
            yaw_delta_rad=float(rng.uniform(yaw_min, yaw_max)),
            cube_axis_offset_m=float(CUBE_AXIS_OFFSET + rng.uniform(-axis_jitter_m, axis_jitter_m)),
            cube_side_offset_m=float(CUBE_SIDE_OFFSET + rng.uniform(-side_jitter_m, side_jitter_m)),
            cube_mass_kg=float(rng.uniform(mass_min_kg, mass_max_kg)),
            cube_friction=float(rng.uniform(cube_friction_min, cube_friction_max)),
            support_friction=float(MAT_FRICTION),
            pad_friction=float(PAD_FRICTION),
        )


def split_uniqueness_audit(
    train_summaries: list[dict[str, Any]],
    validation_summaries: list[dict[str, Any]],
) -> dict[str, int]:
    train_seeds = {int(summary["seed"]) for summary in train_summaries}
    validation_seeds = {int(summary["seed"]) for summary in validation_summaries}
    train_poses = {_pose_key(summary) for summary in train_summaries}
    validation_poses = {_pose_key(summary) for summary in validation_summaries}
    train_hashes = {str(summary["trajectory_hash"]) for summary in train_summaries}
    validation_hashes = {str(summary["trajectory_hash"]) for summary in validation_summaries}
    train_factors = {_factor_key(summary) for summary in train_summaries}
    validation_factors = {_factor_key(summary) for summary in validation_summaries}
    return {
        "train_unique_seed_count": len(train_seeds),
        "validation_unique_seed_count": len(validation_seeds),
        "seed_overlap_count": len(train_seeds & validation_seeds),
        "train_unique_pose_count": len(train_poses),
        "validation_unique_pose_count": len(validation_poses),
        "pose_overlap_count": len(train_poses & validation_poses),
        "train_unique_trajectory_hash_count": len(train_hashes),
        "validation_unique_trajectory_hash_count": len(validation_hashes),
        "trajectory_hash_overlap_count": len(train_hashes & validation_hashes),
        "train_unique_factor_count": len(train_factors),
        "validation_unique_factor_count": len(validation_factors),
        "factor_overlap_count": len(train_factors & validation_factors),
    }


def _pose_key(summary: dict[str, Any]) -> tuple[float, float]:
    xy = summary["initial_cube_xy"]
    return round(float(xy[0]), 6), round(float(xy[1]), 6)


def _factor_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    candidate = summary.get("candidate", {})
    return (
        round(float(candidate["cube_mass_kg"]), 8),
        round(float(candidate["cube_friction"]), 8),
        round(float(candidate["cube_axis_offset_m"]), 8),
        round(float(candidate["cube_side_offset_m"]), 8),
    )


def _validate_request(
    *,
    train_episodes: int,
    val_episodes: int,
    yaw_min: float,
    yaw_max: float,
    axis_jitter_m: float,
    side_jitter_m: float,
    mass_min_kg: float,
    mass_max_kg: float,
    cube_friction_min: float,
    cube_friction_max: float,
) -> None:
    if train_episodes < 0 or val_episodes < 0 or train_episodes + val_episodes <= 0:
        raise ValueError("at least one non-negative train or validation episode is required")
    if yaw_min >= yaw_max:
        raise ValueError("yaw-min must be less than yaw-max")
    if axis_jitter_m < 0.0 or side_jitter_m < 0.0:
        raise ValueError("pose jitter magnitudes must be non-negative")
    if mass_min_kg <= 0.0 or mass_min_kg >= mass_max_kg:
        raise ValueError("mass range must be positive and increasing")
    if cube_friction_min <= 0.0 or cube_friction_min >= cube_friction_max:
        raise ValueError("cube friction range must be positive and increasing")


if __name__ == "__main__":
    main()
