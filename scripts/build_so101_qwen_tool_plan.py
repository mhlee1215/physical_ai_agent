#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physical_ai_agent.agent_core.qwen_so101_tool_planner import (
    DEFAULT_MODEL,
    DEFAULT_OBJECT,
    DEFAULT_TASK,
    OpenAICompatibleQwenClient,
    QwenSO101ToolPlanner,
    plan_to_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a SO101 primitive tool plan with Qwen3.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    planner = QwenSO101ToolPlanner(
        client=OpenAICompatibleQwenClient(base_url=args.base_url, api_key=args.api_key),
        model=args.model,
    )
    plan = planner.plan(task=args.task, target_object=args.object)
    payload = plan_to_dict(plan)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
