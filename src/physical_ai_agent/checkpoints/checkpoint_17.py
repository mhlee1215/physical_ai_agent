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
class Checkpoint17Report:
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
) -> Checkpoint17Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = capture_so101_inputs(
        output_dir=output_dir / "so101_multi_inputs",
        env_id=env_id,
        steps=steps,
        include_virtual_cameras=True,
    )
    first_frame = capture.frames[0] if capture.frames else None
    first_camera_frames = first_frame.camera_frames if first_frame is not None else {}
    checks = {
        "cp17_wrist_cam_saved": "wrist_cam" in first_camera_frames
        and Path(first_camera_frames["wrist_cam"]).exists(),
        "cp17_top_down_saved": "top_down" in first_camera_frames
        and Path(first_camera_frames["top_down"]).exists(),
        "cp17_two_visual_inputs_per_step": bool(capture.frames)
        and all({"wrist_cam", "top_down"}.issubset(frame.camera_frames) for frame in capture.frames),
        "cp17_multi_input_manifest_saved": Path(capture.manifest_path).exists()
        and Path(capture.manifest_path).stat().st_size > 0,
        "cp17_multi_input_preview_saved": Path(capture.preview_path).exists()
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
        "camera_specs": [asdict(spec) for spec in capture.camera_specs],
        "lerobot_feature_keys": [
            "observation.images.wrist_cam",
            "observation.images.top_down",
        ],
    }
    report = Checkpoint17Report(
        checkpoint="checkpoint_17_so101_multi_camera_input",
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
    parser = argparse.ArgumentParser(description="Checkpoint 17 SO101 multi-camera input capture.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_17")
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
            f"frames:{report.metrics['frames']} "
            f"visual_inputs:{','.join(report.metrics['visual_input_names'])}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
