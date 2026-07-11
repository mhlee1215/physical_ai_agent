#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple 280 raw-contact lift physics trials and save representative videos.")
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/rollouts/mycobot_280_raw_lift_batch_001"))
    parser.add_argument("--official-gripper-root", type=Path, default=Path("_vendor/mycobot_ros"))
    parser.add_argument("--asset-root", type=Path, default=Path("_vendor/mycobot_mujoco"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--lift-steps", type=int, default=280)
    parser.add_argument("--video-every", type=int, default=4)
    parser.add_argument("--video-fps", type=float, default=24.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _default_cases()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_dir = args.output_dir / f"case_{index:02d}_{case['name']}"
        video_path = None
        if case.get("video"):
            video_path = case_dir / f"{case['name']}.mp4"
        command = [
            sys.executable,
            "scripts/run_mycobot_280_raw_lift_rollout.py",
            "--output-dir", str(case_dir),
            "--asset-root", str(args.asset_root),
            "--official-gripper-root", str(args.official_gripper_root),
            "--width", str(args.width),
            "--height", str(args.height),
            "--lift-steps", str(args.lift_steps),
            "--seed", str(case["seed"]),
            "--cube-offset-x", str(case["cube_offset"][0]),
            "--cube-offset-y", str(case["cube_offset"][1]),
            "--cube-offset-z", str(case["cube_offset"][2]),
        ]
        if case.get("actuated_lift"):
            command.append("--actuated-lift")
        if "contact_command" in case:
            command.extend(["--contact-command", str(case["contact_command"])])
        if video_path is not None:
            command.extend(["--video-path", str(video_path), "--video-every", str(args.video_every), "--video-fps", str(args.video_fps)])
        env = dict(os.environ)
        env.setdefault("MUJOCO_GL", "egl")
        env["PYTHONPATH"] = "src:."
        completed = subprocess.run(command, cwd=Path.cwd(), env=env, text=True, capture_output=True)
        report_path = case_dir / "mycobot_280_raw_lift_rollout_report.json"
        if not report_path.exists():
            raise RuntimeError(f"case {case['name']} did not write report\nstdout={completed.stdout}\nstderr={completed.stderr}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "index": index,
            "name": case["name"],
            "returncode": completed.returncode,
            "expected_kind": case["expected_kind"],
            "status": report["status"],
            "actuated_lift": report["actuated_lift"],
            "seed": report["seed"],
            "cube_offset_m": report["cube_offset_m"],
            "lift_best_sustained_two_pad_steps": report["lift_best_sustained_two_pad_steps"],
            "lift_two_pad_steps": report["lift_two_pad_steps"],
            "final_cube_lift_m": report["final_cube_lift_m"],
            "final_pad_cube_contacted_pads": report["final_pad_cube_contacted_pads"],
            "first_lift_contact_loss_step": report["first_lift_contact_loss_step"],
            "pre_lift_z_delta_m": report["pre_lift_z_alignment"]["pad_mid_minus_cube_center_m"],
            "final_z_delta_m": report["final_z_alignment"]["pad_mid_minus_cube_center_m"],
            "report_path": str(report_path),
            "sheet_path": report["sheet_path"],
            "video_path": report.get("video_path"),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    success_rows = [row for row in rows if row["status"] == "passed"]
    failure_rows = [row for row in rows if row["status"] != "passed"]
    summary = {
        "status": "passed" if success_rows and failure_rows else "failed",
        "case_count": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(failure_rows),
        "success_rate": len(success_rows) / max(len(rows), 1),
        "rows": rows,
        "representative_success_video": next((row["video_path"] for row in rows if row["status"] == "passed" and row.get("video_path")), None),
        "representative_failure_video": next((row["video_path"] for row in rows if row["status"] != "passed" and row.get("video_path")), None),
    }
    summary_path = args.output_dir / "raw_lift_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "passed" else 1)


def _default_cases() -> list[dict[str, Any]]:
    return [
        {"name": "success_center", "expected_kind": "success", "actuated_lift": True, "seed": 1, "cube_offset": (0.0, 0.0, 0.0), "video": True},
        {"name": "success_y_plus_1mm", "expected_kind": "success", "actuated_lift": True, "seed": 2, "cube_offset": (0.0, 0.001, 0.0)},
        {"name": "success_y_minus_1mm", "expected_kind": "success", "actuated_lift": True, "seed": 3, "cube_offset": (0.0, -0.001, 0.0)},
        {"name": "success_z_plus_1mm", "expected_kind": "success", "actuated_lift": True, "seed": 4, "cube_offset": (0.0, 0.0, 0.001)},
        {"name": "failure_direct_qpos_center", "expected_kind": "failure", "actuated_lift": False, "seed": 11, "cube_offset": (0.0, 0.0, 0.0), "video": True},
        {"name": "failure_direct_qpos_y_plus_1mm", "expected_kind": "failure", "actuated_lift": False, "seed": 12, "cube_offset": (0.0, 0.001, 0.0)},
    ]


if __name__ == "__main__":
    main()
