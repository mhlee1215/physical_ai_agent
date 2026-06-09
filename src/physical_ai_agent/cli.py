from __future__ import annotations

import argparse

try:
    from rich.console import Console
except ModuleNotFoundError:  # pragma: no cover - exercised by plain system Python smoke.
    class Console:  # type: ignore[no-redef]
        def print(self, *objects: object, **_: object) -> None:
            print(*objects)

from physical_ai_agent import __version__
from physical_ai_agent.evaluation.lerobot_eval import (
    build_config as build_lerobot_eval_config,
    build_parser as build_lerobot_eval_parser,
    write_artifacts_from_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physical-ai-agent",
        description="Agentic physical AI evaluation stack.",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    subparsers = parser.add_subparsers(dest="command")
    lerobot_parser = subparsers.add_parser(
        "lerobot-eval",
        parents=[build_lerobot_eval_parser(add_help=False)],
        add_help=False,
        help="Build a shared LeRobot evaluation command.",
    )
    lerobot_parser.set_defaults(command="lerobot-eval")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    if args.version:
        console.print(__version__)
        return

    if args.command == "lerobot-eval":
        config = build_lerobot_eval_config(args)
        if args.artifact_root is None:
            from pathlib import Path

            args.artifact_root = str(Path(args.output_dir).parent)
        write_artifacts_from_args(args, config)
        if args.write_command:
            from pathlib import Path

            path = Path(args.write_command)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(config.build_shell_script(), encoding="utf-8")
            path.chmod(0o755)
        if args.print_command:
            console.print(config.build_shell_command())
        if args.execute:
            import os

            os.execvpe(
                config.build_argv()[0],
                config.build_argv(),
                {**os.environ, "MUJOCO_GL": config.mujoco_gl},
            )
        return

    console.print("[bold]physical-ai-agent[/bold] scaffold is ready.")
    console.print("Next step: add the LIBERO smoke test and baseline evaluator.")
