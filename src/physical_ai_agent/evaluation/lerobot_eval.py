from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from physical_ai_agent.evaluation.agentic_layers import build_agentic_layer, write_debug_artifacts


LIBERO_CAMERA_NAME_MAPPING = '{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
METAWORLD_RENAME_MAP = '{"observation.image":"observation.images.camera1"}'


@dataclass(frozen=True)
class LeRobotEvalConfig:
    benchmark: str
    output_dir: str
    policy_path: str
    env_task: str
    n_episodes: int
    batch_size: int
    seed: int | None = None
    env_task_ids: str | None = None
    env_camera_name_mapping: str | None = None
    eval_use_async_envs: str | None = None
    env_max_parallel_tasks: int | None = None
    policy_empty_cameras: int | None = None
    policy_device: str | None = None
    policy_use_amp: str | None = None
    policy_n_action_steps: int | None = None
    rename_map: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    mujoco_gl: str = "egl"

    def build_argv(self) -> list[str]:
        argv = [
            "lerobot-eval",
            f"--output_dir={self.output_dir}",
            f"--policy.path={self.policy_path}",
            f"--env.type={self.benchmark}",
            f"--env.task={self.env_task}",
            f"--eval.batch_size={self.batch_size}",
            f"--eval.n_episodes={self.n_episodes}",
        ]
        if self.env_task_ids:
            argv.append(f"--env.task_ids={self.env_task_ids}")
        if self.env_camera_name_mapping:
            argv.append(f"--env.camera_name_mapping={self.env_camera_name_mapping}")
        if self.eval_use_async_envs is not None:
            argv.append(f"--eval.use_async_envs={self.eval_use_async_envs}")
        if self.env_max_parallel_tasks is not None:
            argv.append(f"--env.max_parallel_tasks={self.env_max_parallel_tasks}")
        if self.policy_empty_cameras is not None:
            argv.append(f"--policy.empty_cameras={self.policy_empty_cameras}")
        if self.policy_device:
            argv.append(f"--policy.device={self.policy_device}")
        if self.policy_use_amp is not None:
            argv.append(f"--policy.use_amp={self.policy_use_amp}")
        if self.policy_n_action_steps is not None:
            argv.append(f"--policy.n_action_steps={self.policy_n_action_steps}")
        if self.rename_map:
            argv.append(f"--rename_map={self.rename_map}")
        if self.seed is not None:
            argv.append(f"--seed={self.seed}")
        argv.extend(self.extra_args)
        return argv

    def build_shell_command(self) -> str:
        quoted = " \\\n  ".join(shlex.quote(arg) for arg in self.build_argv())
        return f'MUJOCO_GL="${{MUJOCO_GL:-{shlex.quote(self.mujoco_gl)}}}" exec {quoted}'

    def build_shell_script(self) -> str:
        return "#!/bin/sh\nset -eu\n\n" + self.build_shell_command() + "\n"


def build_libero_config(args: argparse.Namespace) -> LeRobotEvalConfig:
    camera_mapping = None if args.camera_name_mapping == "none" else args.camera_name_mapping
    return LeRobotEvalConfig(
        benchmark="libero",
        output_dir=args.output_dir,
        policy_path=args.policy_path or "lerobot/smolvla_libero",
        env_task=args.tasks or "libero_spatial,libero_object,libero_goal,libero_10",
        env_task_ids=args.task_ids,
        env_camera_name_mapping=camera_mapping,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        eval_use_async_envs=args.use_async_envs,
        env_max_parallel_tasks=args.max_parallel_tasks,
        policy_empty_cameras=args.policy_empty_cameras,
        seed=args.seed,
        extra_args=tuple(args.extra_args),
        mujoco_gl=args.mujoco_gl,
    )


def build_metaworld_config(args: argparse.Namespace) -> LeRobotEvalConfig:
    return LeRobotEvalConfig(
        benchmark="metaworld",
        output_dir=args.output_dir,
        policy_path=args.policy_path or "lerobot/smolvla_metaworld",
        env_task=args.tasks or "easy,medium,hard,very_hard",
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        policy_empty_cameras=args.policy_empty_cameras,
        policy_device=args.policy_device,
        policy_use_amp=args.policy_use_amp,
        policy_n_action_steps=args.n_action_steps,
        rename_map=args.rename_map,
        seed=0 if args.seed is None else args.seed,
        extra_args=tuple(args.extra_args),
        mujoco_gl=args.mujoco_gl,
    )


def build_config(args: argparse.Namespace) -> LeRobotEvalConfig:
    if args.extra_args_text:
        args.extra_args.extend(shlex.split(args.extra_args_text))
    if args.benchmark == "libero":
        return build_libero_config(args)
    if args.benchmark == "metaworld":
        return build_metaworld_config(args)
    raise ValueError(f"unsupported benchmark: {args.benchmark}")


def write_artifacts_from_args(args: argparse.Namespace, config: LeRobotEvalConfig) -> dict[str, str]:
    layer = build_agentic_layer(args.agentic_layer, retry_budget=args.retry_budget)
    layered_config = layer.apply(config)
    return write_debug_artifacts(
        output_root=Path(args.artifact_root),
        config=layered_config,
        layer=layer,
        command=layered_config.build_shell_command(),
    )


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m physical_ai_agent.evaluation.lerobot_eval",
        description="Build shared LeRobot evaluation commands for supported SmolVLA benchmarks.",
        add_help=add_help,
    )
    parser.add_argument("--benchmark", choices=("libero", "metaworld"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-path")
    parser.add_argument("--tasks")
    parser.add_argument("--task-ids")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--write-command")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--extra-arg", dest="extra_args", action="append", default=[])
    parser.add_argument("--extra-args", dest="extra_args_text", default="")
    parser.add_argument("--agentic-layer", default="baseline", choices=("baseline", "episode_retry"))
    parser.add_argument("--retry-budget", type=int, default=1)
    parser.add_argument("--artifact-root")

    libero = parser.add_argument_group("LIBERO options")
    libero.add_argument("--camera-name-mapping", default=LIBERO_CAMERA_NAME_MAPPING)
    libero.add_argument("--use-async-envs", default="false")
    libero.add_argument("--max-parallel-tasks", type=int, default=1)

    metaworld = parser.add_argument_group("Meta-World options")
    metaworld.add_argument("--rename-map", default=METAWORLD_RENAME_MAP)
    metaworld.add_argument("--policy-device", default="cuda")
    metaworld.add_argument("--policy-use-amp", default="false")
    metaworld.add_argument("--n-action-steps", type=int)

    parser.add_argument("--policy-empty-cameras", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    if args.artifact_root is None:
        args.artifact_root = str(Path(args.output_dir).parent)
    layer = build_agentic_layer(args.agentic_layer, retry_budget=args.retry_budget)
    config = layer.apply(config)
    script = config.build_shell_script()
    command = config.build_shell_command()

    if args.write_command:
        path = Path(args.write_command)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
    write_debug_artifacts(
        output_root=Path(args.artifact_root),
        config=config,
        layer=layer,
        command=command,
    )
    if args.print_command:
        print(command)
    if args.execute:
        import os

        os.execvpe(config.build_argv()[0], config.build_argv(), {**os.environ, "MUJOCO_GL": config.mujoco_gl})
    if not args.print_command and not args.write_command and not args.execute:
        parser.error("choose at least one of --print-command, --write-command, or --execute")


if __name__ == "__main__":
    main()
