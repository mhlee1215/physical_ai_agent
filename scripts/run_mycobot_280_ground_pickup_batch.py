#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic 280 Pi cube-from-mat pickup POC episodes.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/rollouts/mycobot_280_ground_pickup_batch_001"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--video-every", type=int, default=0)
    parser.add_argument("--lift-qpos", type=str, default=None)
    parser.add_argument("--required-final-lift", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    script = Path(__file__).with_name("run_mycobot_280_ground_pickup_poc.py")
    for index in range(args.episodes):
        episode_dir = args.output_dir / f"episode_{index:03d}"
        cmd = [
            sys.executable,
            str(script),
            "--output-dir",
            str(episode_dir),
            "--asset-root",
            str(args.asset_root),
            "--official-gripper-root",
            str(args.official_gripper_root),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--seed",
            str(args.seed_start + index),
            "--video-every",
            str(args.video_every),
        ]
        if args.lift_qpos:
            cmd.extend(["--lift-qpos", args.lift_qpos])
        result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report_path = episode_dir / "ground_pickup_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "status": "failed",
                "completion_standard_status": "failed",
                "error": result.stdout[-4000:],
            }
        summaries.append(_episode_summary(index, report, result.returncode))

    passed = [item for item in summaries if item["status"] == "passed" and item["completion_standard_status"] == "passed"]
    min_lift = min((float(item["final_cube_lift_m"]) for item in summaries), default=0.0)
    min_sustained = min((int(item["lift_best_sustained_two_pad_steps"]) for item in summaries), default=0)
    min_post_hold_sustained = min((int(item["post_lift_hold_best_sustained_two_pad_steps"]) for item in summaries), default=0)
    min_post_hold_lift = min((float(item["post_lift_hold_min_cube_lift_m"]) for item in summaries), default=0.0)
    max_penetration = max((float(item["max_pad_cube_penetration_m"]) for item in summaries), default=0.0)
    max_lift_penetration = max((float(item["max_lift_pad_cube_penetration_m"]) for item in summaries), default=0.0)
    batch_status = "passed" if len(passed) == args.episodes and min_lift >= args.required_final_lift else "failed"
    report = {
        "status": batch_status,
        "episodes": args.episodes,
        "passed_episodes": len(passed),
        "required_final_lift_m": args.required_final_lift,
        "min_final_cube_lift_m": min_lift,
        "min_lift_sustained_two_pad_steps": min_sustained,
        "min_post_lift_hold_sustained_two_pad_steps": min_post_hold_sustained,
        "min_post_lift_hold_cube_lift_m": min_post_hold_lift,
        "max_pad_cube_penetration_m": max_penetration,
        "max_lift_pad_cube_penetration_m": max_lift_penetration,
        "teacher_attachment_enabled": False,
        "object_teleport_during_pickup_lift": False,
        "episode_summaries": summaries,
    }
    report_path = args.output_dir / "ground_pickup_batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if batch_status == "passed" else 1)


def _episode_summary(index: int, report: dict[str, Any], returncode: int) -> dict[str, Any]:
    return {
        "episode": index,
        "returncode": returncode,
        "status": report.get("status", "failed"),
        "completion_standard_status": report.get("completion_standard_status", "failed"),
        "final_cube_lift_m": report.get("final_cube_lift_m", 0.0),
        "final_pad_cube_contacted_pads": report.get("final_pad_cube_contacted_pads", 0),
        "lift_best_sustained_two_pad_steps": report.get("lift_best_sustained_two_pad_steps", 0),
        "post_lift_hold_best_sustained_two_pad_steps": report.get("post_lift_hold_best_sustained_two_pad_steps", 0),
        "post_lift_hold_min_cube_lift_m": report.get("post_lift_hold_min_cube_lift_m", 0.0),
        "max_pad_cube_penetration_m": report.get("max_pad_cube_penetration_m", 0.0),
        "max_lift_pad_cube_penetration_m": report.get("max_lift_pad_cube_penetration_m", 0.0),
    }


if __name__ == "__main__":
    main()
