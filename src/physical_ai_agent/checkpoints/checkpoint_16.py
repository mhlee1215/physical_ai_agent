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
class Checkpoint16Report:
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
) -> Checkpoint16Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = capture_so101_inputs(output_dir=output_dir / "so101_inputs", env_id=env_id, steps=steps)
    camera_specs = {spec.env_id: spec for spec in capture.camera_specs}
    current_spec = camera_specs[env_id]
    first_frame = capture.frames[0] if capture.frames else None
    checks = {
        "cp16_camera_specs_listed": bool(capture.camera_specs)
        and all(spec.camera_names for spec in capture.camera_specs),
        "cp16_wrist_cam_available": "wrist_cam" in current_spec.camera_names,
        "cp16_camera_frames_saved": first_frame is not None
        and "wrist_cam" in first_frame.camera_frames
        and Path(first_frame.camera_frames["wrist_cam"]).exists(),
        "cp16_input_manifest_saved": Path(capture.manifest_path).exists()
        and Path(capture.manifest_path).stat().st_size > 0,
        "cp16_input_preview_saved": Path(capture.preview_path).exists()
        and Path(capture.preview_gif_path).exists(),
    }
    artifacts = {
        "input_manifest": capture.manifest_path,
        "input_preview": capture.preview_path,
        "input_preview_gif": capture.preview_gif_path,
        "checkpoint_report": str(output_dir / "checkpoint_report.json"),
    }
    if first_frame is not None:
        for camera_name, path in first_frame.camera_frames.items():
            artifacts[f"sample_camera_{camera_name}"] = path
    metrics = {
        "env_id": env_id,
        "frames": len(capture.frames),
        "camera_specs": [asdict(spec) for spec in capture.camera_specs],
        "current_observation_shape": current_spec.observation_shape,
        "current_camera_names": current_spec.camera_names,
    }
    report = Checkpoint16Report(
        checkpoint="checkpoint_16_so101_camera_input",
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
    parser = argparse.ArgumentParser(description="Checkpoint 16 SO101 camera input capture.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_16")
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
            f"cameras:{','.join(report.metrics['current_camera_names'])}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
