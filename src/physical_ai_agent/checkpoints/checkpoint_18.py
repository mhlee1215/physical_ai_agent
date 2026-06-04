from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from physical_ai_agent.sim.so101_camera_input import capture_so101_inputs
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID


@dataclass(frozen=True)
class Checkpoint18Report:
    checkpoint: str
    status: str
    python: str
    platform: str
    output_dir: str
    checks: dict[str, bool]
    metrics: dict[str, object]
    artifacts: dict[str, str]


def run_checkpoint(
    output_dir: Path,
    env_id: str = DEFAULT_SO101_ENV_ID,
    steps: int = 8,
) -> Checkpoint18Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = capture_so101_inputs(
        output_dir=output_dir / "so101_policy_inputs",
        env_id=env_id,
        steps=steps,
        camera_names=("wrist_cam", "egocentric_cam", "top_down"),
    )
    first_frame = capture.frames[0] if capture.frames else None
    first_camera_frames = first_frame.camera_frames if first_frame is not None else {}
    checks = {
        "cp18_wrist_policy_input_saved": "wrist_cam" in first_camera_frames
        and Path(first_camera_frames["wrist_cam"]).exists(),
        "cp18_egocentric_policy_input_saved": "egocentric_cam" in first_camera_frames
        and Path(first_camera_frames["egocentric_cam"]).exists(),
        "cp18_top_down_debug_input_saved": "top_down" in first_camera_frames
        and Path(first_camera_frames["top_down"]).exists(),
        "cp18_policy_debug_roles_recorded": capture.policy_input_names == ["wrist_cam", "egocentric_cam"]
        and capture.debug_input_names == ["top_down"],
        "cp18_preview_saved": Path(capture.preview_path).exists()
        and Path(capture.preview_gif_path).exists(),
    }
    artifacts = {
        "input_manifest": capture.manifest_path,
        "input_preview": capture.preview_path,
        "input_preview_gif": capture.preview_gif_path,
        "checkpoint_report": str(output_dir / "checkpoint_report.json"),
    }
    for camera_name, path in first_camera_frames.items():
        artifacts[f"sample_camera_{camera_name}"] = path
    metrics = {
        "env_id": env_id,
        "frames": len(capture.frames),
        "visual_input_names": sorted(first_camera_frames),
        "policy_input_names": capture.policy_input_names,
        "debug_input_names": capture.debug_input_names,
        "lerobot_policy_feature_keys": capture.lerobot_policy_feature_keys,
    }
    report = Checkpoint18Report(
        checkpoint="checkpoint_18_so101_egocentric_policy_inputs",
        status="passed" if all(checks.values()) else "failed",
        python=platform.python_version(),
        platform=platform.platform(),
        output_dir=str(output_dir),
        checks=checks,
        metrics=metrics,
        artifacts=artifacts,
    )
    Path(report.artifacts["checkpoint_report"]).write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 18 SO101 egocentric policy input capture.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_18")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(output_dir=Path(args.output_dir), env_id=args.env_id, steps=args.steps)
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(f"{report.checkpoint}: {report.status}")
        print(f"output_dir={report.output_dir}")
        for name, passed in report.checks.items():
            print(f"- {'PASS' if passed else 'FAIL'} {name}")
        print(
            "metrics="
            f"env:{report.metrics['env_id']} "
            f"policy_inputs:{','.join(report.metrics['policy_input_names'])} "
            f"debug_inputs:{','.join(report.metrics['debug_input_names'])}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
