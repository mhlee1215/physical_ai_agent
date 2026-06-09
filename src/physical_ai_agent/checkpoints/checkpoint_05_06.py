from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from physical_ai_agent.policies.chunked_random_policy import (
    ChunkedRandomPolicy,
    ChunkedRandomPolicyConfig,
)
from physical_ai_agent.policies.smolvla_adapter import (
    DEFAULT_SMOLVLA_MODEL_ID,
    SmolVLAPolicyAdapter,
    probe_smolvla,
)
from physical_ai_agent.sim.tiny_mujoco_env import TinyMujocoConfig, TinyMujocoEnv


@dataclass(frozen=True)
class Checkpoint0506Report:
    checkpoint: str
    status: str
    python: str
    platform: str
    output_dir: str
    duration_s: float
    checks: dict[str, bool]
    smolvla_ready: bool
    smolvla_blockers: list[str]
    artifacts: dict[str, str]


def run_checkpoint(
    output_dir: Path,
    model_id: str = DEFAULT_SMOLVLA_MODEL_ID,
    require_real_smolvla: bool = False,
) -> Checkpoint0506Report:
    started_at = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = TinyMujocoEnv(TinyMujocoConfig(episode_steps=8))
    observation = env.reset()
    policy = ChunkedRandomPolicy(
        ChunkedRandomPolicyConfig(action_dim=env.action_dim, chunk_size=4, seed=0)
    )
    chunk = policy.action_chunk(observation, "move the puck randomly")
    next_observation, _reward, _done, info = env.step(chunk.first_action())
    smolvla_adapter = SmolVLAPolicyAdapter(model_id)
    smolvla_probe = probe_smolvla(model_id)

    checks = {
        "cp05_policy_adapter_created": policy.name == "chunked_random",
        "cp05_action_chunk_created": chunk.chunk_size == 4,
        "cp05_action_dim_matches_env": all(len(action) == env.action_dim for action in chunk.actions),
        "cp05_action_chunk_executes_one_step": next_observation.step == 1 and info["finite_state"],
        "cp06_smolvla_adapter_created": smolvla_adapter.name == "smolvla",
        "cp06_smolvla_probe_ran": smolvla_probe.model_id == model_id,
        "cp06_smolvla_policy_class_importable": smolvla_probe.policy_class_importable
        or bool(smolvla_probe.blockers),
        "cp06_smolvla_ready_or_blocker_documented": smolvla_probe.ready
        or bool(smolvla_probe.blockers),
    }
    if require_real_smolvla:
        checks["cp06_smolvla_import_path_required"] = smolvla_probe.ready

    blocker_path = output_dir / "smolvla_blocker.md"
    blocker_path.write_text(_blocker_markdown(smolvla_probe), encoding="utf-8")
    report = Checkpoint0506Report(
        checkpoint="checkpoint_05_06_policy_adapter_smolvla_probe",
        status="passed" if all(checks.values()) else "failed",
        python=platform.python_version(),
        platform=platform.platform(),
        output_dir=str(output_dir),
        duration_s=round(perf_counter() - started_at, 4),
        checks=checks,
        smolvla_ready=smolvla_probe.ready,
        smolvla_blockers=smolvla_probe.blockers,
        artifacts={
            "smolvla_blocker": str(blocker_path),
            "checkpoint_report": str(output_dir / "checkpoint_report.json"),
        },
    )
    (output_dir / "checkpoint_report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _blocker_markdown(probe: object) -> str:
    probe_data = asdict(probe)
    lines = [
        "# SmolVLA CP06 Probe",
        "",
        f"- Model id: `{probe_data['model_id']}`",
        f"- Ready: `{probe_data['ready']}`",
        "",
        "## Imports",
        "",
    ]
    for name, ok in probe_data["imports"].items():
        lines.append(f"- [{'x' if ok else ' '}] `{name}`")
    lines.extend(
        [
            f"- [{'x' if probe_data['policy_class_importable'] else ' '}] "
            "`lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy`",
        ]
    )
    lines.extend(["", "## Blockers", ""])
    blockers = probe_data["blockers"]
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in probe_data["notes"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 05-06 policy adapter and SmolVLA probe.")
    parser.add_argument(
        "--output-dir",
        default="_workspace/checkpoints/checkpoint_05_06",
        help="Directory for checkpoint artifacts.",
    )
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument(
        "--require-real-smolvla",
        action="store_true",
        help="Fail if the LeRobot SmolVLA import path is not ready.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(
        output_dir=Path(args.output_dir),
        model_id=args.model_id,
        require_real_smolvla=args.require_real_smolvla,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(f"{report.checkpoint}: {report.status}")
        print(f"output_dir={report.output_dir}")
        for name, passed in report.checks.items():
            print(f"- {'PASS' if passed else 'FAIL'} {name}")
        print(f"smolvla_ready={report.smolvla_ready}")
        if report.smolvla_blockers:
            print("smolvla_blockers=" + "; ".join(report.smolvla_blockers))

    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
