#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_PYTHON = ".venv-mps-nightly/bin/python"


def run_external_mps_smolvla_dry(
    *,
    episode: Path,
    output_dir: Path,
    instruction: str,
    frame_index: int = 0,
    calibration: Path | None = None,
    python: str = DEFAULT_PYTHON,
    action_steps: int = 10,
    device: str = "auto",
    state_units: str = "raw_ticks",
    wrist_camera_index: str = "0",
    egocentric_camera_index: str = "1",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "external_mps_stdout.log"
    stderr_path = output_dir / "external_mps_stderr.log"
    report_path = output_dir / "smolvla_dry_report.json"

    command = [
        python,
        "scripts/real_so100_smolvla_dry.py",
        "--episode",
        str(episode),
        "--frame-index",
        str(frame_index),
        "--output-dir",
        str(output_dir),
        "--instruction",
        instruction,
        "--state-units",
        state_units,
        "--device",
        device,
        "--wrist-camera-index",
        wrist_camera_index,
        "--egocentric-camera-index",
        egocentric_camera_index,
        "--action-steps",
        str(action_steps),
    ]
    if calibration is not None:
        command.extend(["--calibration", str(calibration)])

    shell_command = (
        "cd /Users/minhaeng/workspace/physical_ai_agent; "
        "PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src:. "
        + " ".join(shlex.quote(part) for part in command)
        + f" > {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}"
    )
    result = subprocess.run(
        [
            "osascript",
            "-e",
            "on run argv\n"
            "  do shell script (item 1 of argv)\n"
            "end run",
            shell_command,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    launcher_report = {
        "status": "passed" if result.returncode == 0 and report_path.exists() else "failed",
        "operation": "run_external_mps_smolvla_dry",
        "execution_context": "macos_osascript_do_shell_script_no_terminal_window",
        "python": python,
        "episode": str(episode),
        "output_dir": str(output_dir),
        "wrist_camera_index": wrist_camera_index,
        "egocentric_camera_index": egocentric_camera_index,
        "report_path": str(report_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": result.returncode,
        "osascript_stdout": result.stdout,
        "osascript_stderr": result.stderr,
    }
    if report_path.exists():
        launcher_report["smolvla_report"] = json.loads(report_path.read_text(encoding="utf-8"))
    (output_dir / "external_mps_launcher_report.json").write_text(
        json.dumps(launcher_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return launcher_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmolVLA dry inference through macOS shell without opening Terminal.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--state-units", default="raw_ticks", choices=["raw_ticks", "lerobot_so100_position"])
    parser.add_argument("--wrist-camera-index", default="0")
    parser.add_argument("--egocentric-camera-index", default="1")
    args = parser.parse_args()
    print(
        json.dumps(
            run_external_mps_smolvla_dry(
                episode=args.episode,
                frame_index=args.frame_index,
                output_dir=args.output_dir,
                instruction=args.instruction,
                calibration=args.calibration,
                python=args.python,
                action_steps=args.action_steps,
                device=args.device,
                state_units=args.state_units,
                wrist_camera_index=args.wrist_camera_index,
                egocentric_camera_index=args.egocentric_camera_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
