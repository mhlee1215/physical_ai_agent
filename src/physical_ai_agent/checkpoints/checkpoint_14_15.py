from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from physical_ai_agent.policies.smolvla_real import run_real_smolvla_inference_probe
from physical_ai_agent.sim.so101_3d_render import render_so101_3d_rollout
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv


@dataclass(frozen=True)
class Checkpoint1415Report:
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
    steps: int = 24,
    allow_download: bool = False,
    require_3d_render: bool = False,
    require_real_smolvla: bool = False,
) -> Checkpoint1415Report:
    started_at = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    obs, _info = env.reset(seed=0)
    action_dim = env.action_dim
    env.close()

    render = render_so101_3d_rollout(output_dir / "render_3d", env_id=env_id, steps=steps)
    smolvla = run_real_smolvla_inference_probe(
        output_dir=output_dir / "smolvla_real",
        observation=obs,
        action_dim=action_dim,
        env_id=env_id,
        rollout_steps=min(steps, 6),
        model_id=model_id,
        local_files_only=not allow_download,
    )

    checks = {
        "cp14_3d_render_attempted": bool(render.attempts),
        "cp14_3d_render_saved_or_blocker_documented": (
            render.status == "passed" and Path(render.frame_path).exists() and Path(render.gif_path).exists()
        )
        or (Path(render.blocker_path).exists() and Path(render.blocker_path).stat().st_size > 0),
        "cp14_3d_render_strict": render.status == "passed" if require_3d_render else True,
        "cp15_real_smolvla_inference_attempted": Path(smolvla.report_path).exists(),
        "cp15_real_smolvla_action_or_blocker_documented": (
            smolvla.status == "passed"
            and len(smolvla.action) == action_dim
            and Path(smolvla.trace_path).exists()
            and Path(smolvla.trace_path).stat().st_size > 0
        )
        or (Path(smolvla.blocker_path).exists() and Path(smolvla.blocker_path).stat().st_size > 0),
        "cp15_real_smolvla_strict": smolvla.status == "passed" if require_real_smolvla else True,
    }
    artifacts = {
        "render_frame": render.frame_path,
        "render_gif": render.gif_path,
        "render_report": render.report_path,
        "render_blocker": render.blocker_path,
        "smolvla_report": smolvla.report_path,
        "smolvla_blocker": smolvla.blocker_path,
        "smolvla_trace": smolvla.trace_path,
        "smolvla_frame": smolvla.frame_path,
        "smolvla_gif": smolvla.gif_path,
        "checkpoint_report": str(output_dir / "checkpoint_report.json"),
    }
    metrics = {
        "env_id": env_id,
        "model_id": model_id,
        "duration_s": round(perf_counter() - started_at, 4),
        "action_dim": action_dim,
        "render_status": render.status,
        "render_frames": render.frames,
        "smolvla_status": smolvla.status,
        "smolvla_action_shape": smolvla.action_shape,
        "smolvla_rollout_steps": smolvla.rollout_steps,
        "allow_download": allow_download,
        "require_3d_render": require_3d_render,
        "require_real_smolvla": require_real_smolvla,
    }
    report = Checkpoint1415Report(
        checkpoint="checkpoint_14_15_so101_3d_render_real_smolvla",
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
    parser = argparse.ArgumentParser(description="Checkpoint 14-15 SO101 3D render and real SmolVLA gate.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_14_15")
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--require-3d-render", action="store_true")
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
        require_3d_render=args.require_3d_render,
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
            f"render:{report.metrics['render_status']} "
            f"smolvla:{report.metrics['smolvla_status']}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
