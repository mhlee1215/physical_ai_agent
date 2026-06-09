from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from physical_ai_agent.policies.smolvla_real import run_real_smolvla_inference_probe
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv


@dataclass(frozen=True)
class Checkpoint19Report:
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
    model_id: str = DEFAULT_SMOLVLA_MODEL_ID,
    steps: int = 4,
    allow_download: bool = False,
    require_real_smolvla: bool = False,
) -> Checkpoint19Report:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    try:
        obs, _info = env.reset(seed=0)
        action_dim = env.action_dim
    finally:
        env.close()

    smolvla = run_real_smolvla_inference_probe(
        output_dir=output_dir / "smolvla_camera_inputs",
        observation=obs,
        action_dim=action_dim,
        env_id=env_id,
        rollout_steps=steps,
        model_id=model_id,
        local_files_only=not allow_download,
        use_real_camera_inputs=True,
    )
    mapping_values = set(smolvla.image_feature_mapping.values())
    strict_ok = smolvla.status == "passed"
    checks = {
        "cp19_real_smolvla_attempted": Path(smolvla.report_path).exists(),
        "cp19_real_camera_inputs_enabled": smolvla.used_real_camera_inputs,
        "cp19_wrist_and_egocentric_mapped": {"wrist_cam", "egocentric_cam"}.issubset(mapping_values),
        "cp19_zero_image_tensor_removed": "zero" not in mapping_values,
        "cp19_input_manifest_saved": Path(smolvla.input_manifest_path).exists()
        and Path(smolvla.input_manifest_path).stat().st_size > 0,
        "cp19_input_preview_saved": Path(smolvla.input_preview_path).exists()
        and Path(smolvla.input_preview_gif_path).exists(),
        "cp19_smolvla_rollout_trace_saved": Path(smolvla.trace_path).exists()
        and Path(smolvla.trace_path).stat().st_size > 0,
        "cp19_smolvla_strict": strict_ok if require_real_smolvla else True,
    }
    artifacts = {
        "smolvla_report": smolvla.report_path,
        "smolvla_blocker": smolvla.blocker_path,
        "smolvla_trace": smolvla.trace_path,
        "smolvla_frame": smolvla.frame_path,
        "smolvla_gif": smolvla.gif_path,
        "input_manifest": smolvla.input_manifest_path,
        "input_preview": smolvla.input_preview_path,
        "input_preview_gif": smolvla.input_preview_gif_path,
        "checkpoint_report": str(output_dir / "checkpoint_report.json"),
    }
    metrics = {
        "env_id": env_id,
        "model_id": model_id,
        "allow_download": allow_download,
        "require_real_smolvla": require_real_smolvla,
        "smolvla_status": smolvla.status,
        "smolvla_rollout_steps": smolvla.rollout_steps,
        "smolvla_action_shape": smolvla.action_shape,
        "image_feature_mapping": smolvla.image_feature_mapping,
        "policy_input_names": ["wrist_cam", "egocentric_cam"],
        "debug_input_names": ["top_down"],
    }
    report = Checkpoint19Report(
        checkpoint="checkpoint_19_smolvla_real_camera_inputs",
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
    parser = argparse.ArgumentParser(description="Checkpoint 19 SmolVLA real camera input rollout.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_19")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--require-real-smolvla", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(
        output_dir=Path(args.output_dir),
        env_id=args.env_id,
        model_id=args.model_id,
        steps=args.steps,
        allow_download=args.allow_download,
        require_real_smolvla=args.require_real_smolvla,
    )
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
            f"smolvla:{report.metrics['smolvla_status']} "
            f"steps:{report.metrics['smolvla_rollout_steps']}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
