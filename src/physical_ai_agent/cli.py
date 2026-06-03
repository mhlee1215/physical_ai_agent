from __future__ import annotations

import argparse

from rich.console import Console

from physical_ai_agent import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physical-ai-agent",
        description="Agentic physical AI evaluation stack.",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    if args.version:
        console.print(__version__)
        return

    console.print("[bold]physical-ai-agent[/bold] scaffold is ready.")
    console.print("Next step: add the LIBERO smoke test and baseline evaluator.")

