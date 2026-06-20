#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from physical_ai_agent.agent_core.qwen_so101_closed_loop import (
    parse_primitive_policy_routes,
    resolve_policy_routes,
    run_closed_loop_plan,
    write_plan_only_report,
)
from physical_ai_agent.agent_core.qwen_so101_tool_planner import (
    DEFAULT_MODEL,
    DEFAULT_OBJECT,
    DEFAULT_TASK,
    ChatToolClient,
    OpenAICompatibleQwenClient,
    QwenSO101ToolPlanner,
    SO101PrimitiveCall,
    SO101ToolPlan,
)
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Qwen3-planned SO101 primitive chain as a SmolVLA closed-loop evaluation."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--qwen-model", default=DEFAULT_MODEL)
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--qwen-response-json", type=Path)
    parser.add_argument("--qwen-plan-json", type=Path)
    parser.add_argument("--policy-path", default=None, help="Default SmolVLA checkpoint/path for every primitive.")
    parser.add_argument(
        "--primitive-policy",
        action="append",
        default=[],
        help="Per-primitive route as primitive_id=policy_path. Repeat for three separately trained checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/qwen_so101_closed_loop"))
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=98100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-steps-per-primitive", type=int, default=None)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate Qwen plan and policy routing without loading SO101/SmolVLA.",
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = _load_or_build_plan(args)
    primitive_policy_paths = parse_primitive_policy_routes(args.primitive_policy)
    if args.plan_only:
        routes = resolve_policy_routes(
            plan,
            default_policy_path=args.policy_path,
            primitive_policy_paths=primitive_policy_paths,
        )
        report = write_plan_only_report(
            plan=plan,
            output_dir=args.output_dir,
            policy_routes=routes,
        )
    else:
        report = run_closed_loop_plan(
            plan=plan,
            output_dir=args.output_dir,
            default_policy_path=args.policy_path,
            primitive_policy_paths=primitive_policy_paths,
            env_id=args.env_id,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            local_files_only=not args.allow_download,
            max_steps_per_primitive=args.max_steps_per_primitive,
            policy_n_action_steps=args.policy_n_action_steps,
            policy_num_steps=args.policy_num_steps,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_pass and report["status"] not in {"passed", "planned"}:
        sys.exit(1)


def _load_or_build_plan(args: argparse.Namespace) -> SO101ToolPlan:
    if args.qwen_plan_json:
        return _plan_from_dict(json.loads(args.qwen_plan_json.read_text(encoding="utf-8")))
    qwen_client, _source = _qwen_client(
        qwen_base_url=args.qwen_base_url,
        qwen_response_json=args.qwen_response_json,
        qwen_api_key=args.qwen_api_key,
    )
    planner = QwenSO101ToolPlanner(client=qwen_client, model=args.qwen_model)
    return planner.plan(task=args.task, target_object=args.object)


def _qwen_client(
    *,
    qwen_base_url: str | None,
    qwen_response_json: Path | None,
    qwen_api_key: str | None,
) -> tuple[ChatToolClient, str]:
    if qwen_response_json is not None:
        return SavedQwenResponseClient(qwen_response_json), "saved_response_json"
    if not qwen_base_url:
        raise ValueError("Provide --qwen-base-url, --qwen-response-json, or --qwen-plan-json.")
    return OpenAICompatibleQwenClient(base_url=qwen_base_url, api_key=qwen_api_key), "live_openai_compatible"


class SavedQwenResponseClient:
    def __init__(self, response_path: Path) -> None:
        self.response_path = response_path

    def create_tool_plan(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str,
        temperature: float,
    ) -> dict:
        del model, messages, tools, tool_choice, temperature
        return json.loads(self.response_path.read_text(encoding="utf-8"))


def _plan_from_dict(payload: dict) -> SO101ToolPlan:
    if "plan" in payload and isinstance(payload["plan"], dict):
        payload = payload["plan"]
    calls = [
        SO101PrimitiveCall(
            index=int(item["index"]),
            fn=str(item["fn"]),
            object=str(item["object"]),
            primitive_id=str(item["primitive_id"]),
            prompt=str(item["prompt"]),
            max_steps=int(item["max_steps"]),
        )
        for item in payload["calls"]
    ]
    return SO101ToolPlan(
        task=str(payload["task"]),
        model=str(payload["model"]),
        thinking_mode=str(payload.get("thinking_mode", "non-thinking")),
        calls=calls,
    )


if __name__ == "__main__":
    main()
