#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PREALIGNED_POC = "prealigned_gate8_teacher_poc"
GROUND_PICKUP_POC = "ground_pickup_raw_contact_poc"
DETERMINISTIC_DATASET = "ground_pickup_pose_diverse_deterministic_dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Guard the myCobot 280 Pi adaptive ladder before exporting the deterministic "
            "pose-diverse ground-pickup teacher dataset POC."
        )
    )
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_workspace/checks/mycobot_280_ground_pickup_pose_diverse_dataset_001"),
    )
    parser.add_argument("--prealigned-episodes", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=0)
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
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        max_attempts=args.max_attempts,
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
    train_episodes: int,
    val_episodes: int,
    max_attempts: int,
    width: int,
    height: int,
    render_every: int,
    fps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []

    stages.append(
        _run_prealigned_gate8_teacher_poc(
            output_dir=output_dir / PREALIGNED_POC,
            episodes=prealigned_episodes,
            asset_root=asset_root,
            official_gripper_root=official_gripper_root,
            width=width,
            height=height,
            render_every=render_every,
            fps=fps,
        )
    )
    if stages[-1]["status"] != "passed":
        return _write_report(output_dir, _report(stages))

    stages.append(
        _run_ground_pickup_raw_contact_poc(
            output_dir=output_dir / GROUND_PICKUP_POC,
            asset_root=asset_root,
            official_gripper_root=official_gripper_root,
            width=width,
            height=height,
        )
    )
    if stages[-1]["status"] != "passed":
        return _write_report(output_dir, _report(stages))

    stages.append(
        _run_ground_pickup_pose_diverse_dataset(
            output_dir=output_dir / DETERMINISTIC_DATASET,
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            max_attempts=max_attempts,
            asset_root=asset_root,
            official_gripper_root=official_gripper_root,
            width=width,
            height=height,
            render_every=render_every,
            fps=fps,
        )
    )
    return _write_report(output_dir, _report(stages))


def _run_prealigned_gate8_teacher_poc(
    *,
    output_dir: Path,
    episodes: int,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
    render_every: int,
    fps: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/export_mycobot_280_pi_adaptive_teacher_dataset.py",
        "--asset-root",
        str(asset_root),
        "--official-gripper-root",
        str(official_gripper_root),
        "--output-dir",
        str(output_dir),
        "--episodes",
        str(episodes),
        "--width",
        str(width),
        "--height",
        str(height),
        "--render-every",
        str(render_every),
        "--fps",
        str(fps),
    ]
    result = _run_command(command)
    manifest = _load_json(output_dir / "manifest.json")
    passed = (
        result["returncode"] == 0
        and manifest.get("episodes") == episodes
        and manifest.get("failed_episodes") == []
    )
    return {
        "name": PREALIGNED_POC,
        "status": "passed" if passed else "failed",
        "claim": "Already-merged pre-aligned 280 Gate 8 teacher POC remains runnable.",
        "output_dir": str(output_dir),
        "command": command,
        "result": result,
        "evidence": {
            "episodes": manifest.get("episodes"),
            "failed_episodes": manifest.get("failed_episodes"),
            "success_criteria": manifest.get("success_criteria"),
        },
    }


def _run_ground_pickup_raw_contact_poc(
    *,
    output_dir: Path,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/run_mycobot_280_ground_pickup_poc.py",
        "--asset-root",
        str(asset_root),
        "--official-gripper-root",
        str(official_gripper_root),
        "--output-dir",
        str(output_dir),
        "--width",
        str(width),
        "--height",
        str(height),
        "--video-every",
        "0",
    ]
    result = _run_command(command)
    report = _load_json(output_dir / "ground_pickup_report.json")
    completion = report.get("completion_standard", {})
    passed = (
        result["returncode"] == 0
        and report.get("status") == "passed"
        and completion.get("status") == "passed"
        and all(bool(value) for value in completion.get("checks", {}).values())
    )
    return {
        "name": GROUND_PICKUP_POC,
        "status": "passed" if passed else "failed",
        "claim": "Already-merged non-pre-aligned cube-from-mat raw-contact POC remains runnable.",
        "output_dir": str(output_dir),
        "command": command,
        "result": result,
        "evidence": {
            "status": report.get("status"),
            "completion_standard": completion,
        },
    }


def _run_ground_pickup_pose_diverse_dataset(
    *,
    output_dir: Path,
    train_episodes: int,
    val_episodes: int,
    max_attempts: int,
    asset_root: Path,
    official_gripper_root: Path,
    width: int,
    height: int,
    render_every: int,
    fps: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/export_mycobot_280_ground_pickup_teacher_dataset.py",
        "--asset-root",
        str(asset_root),
        "--official-gripper-root",
        str(official_gripper_root),
        "--output-dir",
        str(output_dir),
        "--train-episodes",
        str(train_episodes),
        "--val-episodes",
        str(val_episodes),
        "--max-attempts",
        str(max_attempts),
        "--width",
        str(width),
        "--height",
        str(height),
        "--render-every",
        str(render_every),
        "--fps",
        str(fps),
    ]
    result = _run_command(command)
    manifest = _load_json(output_dir / "manifest.json")
    aggregate = manifest.get("aggregate_metrics", {})
    coverage = aggregate.get("pose_coverage", {})
    splits = manifest.get("splits", {})
    train = splits.get("train", {})
    validation = splits.get("validation", {})
    expected = train_episodes + val_episodes
    split_uniqueness = _split_uniqueness(train, validation)
    passed = (
        result["returncode"] == 0
        and manifest.get("generation_mode") == "deterministic_pose_diverse_teacher_aligned"
        and manifest.get("randomization_enabled") is False
        and manifest.get("teacher_attachment_enabled") is False
        and manifest.get("object_teleport_during_pickup_lift") is False
        and manifest.get("requested_episodes") == expected
        and manifest.get("accepted_episodes") == expected
        and manifest.get("episodes") == expected
        and manifest.get("passed_episodes") == expected
        and manifest.get("failed_episodes") == []
        and train.get("accepted_episodes") == train_episodes
        and validation.get("accepted_episodes") == val_episodes
        and aggregate.get("passed_episodes") == expected
        and coverage.get("unique_pose_count") == expected
        and coverage.get("unique_trajectory_hashes") == expected
        and split_uniqueness["seed_overlap_count"] == 0
        and split_uniqueness["pose_overlap_count"] == 0
        and float(aggregate.get("min_final_cube_lift_m", 0.0)) >= 0.05
        and int(aggregate.get("min_post_lift_hold_sustained_two_pad_steps", 0)) >= 300
        and float(aggregate.get("min_post_lift_hold_cube_lift_m", 0.0)) >= 0.045
        and float(aggregate.get("max_pad_cube_penetration_m", 999.0)) <= 0.003
    )
    return {
        "name": DETERMINISTIC_DATASET,
        "status": "passed" if passed else "failed",
        "claim": "New deterministic pose-diverse 280 cube-from-mat dataset POC exports train/validation episodes.",
        "output_dir": str(output_dir),
        "command": command,
        "result": result,
        "evidence": {
            "generation_mode": manifest.get("generation_mode"),
            "randomization_enabled": manifest.get("randomization_enabled"),
            "requested_episodes": manifest.get("requested_episodes"),
            "accepted_episodes": manifest.get("accepted_episodes"),
            "failed_episodes": manifest.get("failed_episodes"),
            "rejected_attempt_count": len(manifest.get("rejected_attempts", [])),
            "splits": {
                "train": {
                    "requested_episodes": train.get("requested_episodes"),
                    "accepted_episodes": train.get("accepted_episodes"),
                },
                "validation": {
                    "requested_episodes": validation.get("requested_episodes"),
                    "accepted_episodes": validation.get("accepted_episodes"),
                },
            },
            "split_uniqueness": split_uniqueness,
            "aggregate_metrics": aggregate,
        },
    }


def _split_uniqueness(train: dict[str, Any], validation: dict[str, Any]) -> dict[str, int]:
    train_summaries = train.get("episode_summaries", [])
    validation_summaries = validation.get("episode_summaries", [])
    train_seeds = {int(summary["seed"]) for summary in train_summaries if "seed" in summary}
    validation_seeds = {int(summary["seed"]) for summary in validation_summaries if "seed" in summary}
    train_poses = {_pose_key(summary) for summary in train_summaries if "initial_cube_xy" in summary}
    validation_poses = {_pose_key(summary) for summary in validation_summaries if "initial_cube_xy" in summary}
    return {
        "train_unique_seed_count": len(train_seeds),
        "validation_unique_seed_count": len(validation_seeds),
        "seed_overlap_count": len(train_seeds & validation_seeds),
        "train_unique_pose_count": len(train_poses),
        "validation_unique_pose_count": len(validation_poses),
        "pose_overlap_count": len(train_poses & validation_poses),
    }


def _pose_key(summary: dict[str, Any]) -> tuple[float, float]:
    xy = summary["initial_cube_xy"]
    return round(float(xy[0]), 6), round(float(xy[1]), 6)


def _run_command(command: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    env.setdefault("MUJOCO_GL", "egl")
    process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True)
    return {
        "returncode": process.returncode,
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _report(stages: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [stage["name"] for stage in stages if stage["status"] != "passed"]
    return {
        "status": "passed" if not failed else "failed",
        "purpose": (
            "Regression guard for the 280 ladder: preserve the pre-aligned Gate 8 teacher POC, "
            "preserve the non-pre-aligned raw cube-from-mat POC, then validate deterministic "
            "pose-diverse ground-pickup dataset export."
        ),
        "protected_capabilities": [PREALIGNED_POC, GROUND_PICKUP_POC],
        "new_capability": DETERMINISTIC_DATASET,
        "failed_stages": failed,
        "stages": stages,
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "check_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
