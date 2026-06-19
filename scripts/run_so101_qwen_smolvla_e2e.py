#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from physical_ai_agent.agent_core.qwen_so101_tool_planner import (
    DEFAULT_MODEL,
    DEFAULT_OBJECT,
    DEFAULT_TASK,
    ChatToolClient,
    OpenAICompatibleQwenClient,
    QwenSO101ToolPlanner,
    SO101ToolPlan,
    plan_to_dict,
)
from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from physical_ai_agent.policies.smolvla_real import run_real_smolvla_inference_probe
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real Qwen3 planner + SmolVLA SO101 inference smoke test."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--qwen-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--qwen-base-url",
        default=None,
        help="OpenAI-compatible Qwen endpoint, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--qwen-response-json",
        type=Path,
        default=None,
        help="Use a saved OpenAI-compatible Qwen response instead of calling a live endpoint.",
    )
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--smolvla-model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/qwen_smolvla_e2e"))
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--use-real-camera-inputs", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_e2e(
        task=args.task,
        target_object=args.object,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
        qwen_response_json=args.qwen_response_json,
        qwen_api_key=args.qwen_api_key,
        smolvla_model_id=args.smolvla_model_id,
        env_id=args.env_id,
        output_dir=args.output_dir,
        rollout_steps=args.rollout_steps,
        device=args.device,
        allow_download=args.allow_download,
        use_real_camera_inputs=args.use_real_camera_inputs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_pass and report["status"] != "passed":
        sys.exit(1)


def run_e2e(
    *,
    task: str,
    target_object: str,
    qwen_model: str,
    qwen_base_url: str | None,
    qwen_response_json: Path | None,
    qwen_api_key: str | None,
    smolvla_model_id: str,
    env_id: str,
    output_dir: Path,
    rollout_steps: int,
    device: str,
    allow_download: bool,
    use_real_camera_inputs: bool,
) -> dict:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen_client, qwen_source = _qwen_client(
        qwen_base_url=qwen_base_url,
        qwen_response_json=qwen_response_json,
        qwen_api_key=qwen_api_key,
    )
    planner = QwenSO101ToolPlanner(client=qwen_client, model=qwen_model)
    plan = planner.plan(task=task, target_object=target_object)
    prompt = _task_prompt_from_plan(plan)

    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    try:
        obs, _info = env.reset(seed=0)
        action_dim = env.action_dim
    finally:
        env.close()

    smolvla = run_real_smolvla_inference_probe(
        output_dir=output_dir / "smolvla_real",
        observation=obs,
        action_dim=action_dim,
        env_id=env_id,
        rollout_steps=rollout_steps,
        model_id=smolvla_model_id,
        local_files_only=not allow_download,
        use_real_camera_inputs=use_real_camera_inputs,
        device=device,
        task_prompt=prompt,
    )
    qwen_plan_path = output_dir / "qwen_tool_plan.json"
    qwen_plan_path.write_text(
        json.dumps(plan_to_dict(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "operation": "so101_qwen_smolvla_e2e",
        "status": "passed" if smolvla.status == "passed" else "blocked",
        "duration_s": round(perf_counter() - started, 4),
        "qwen": {
            "model": qwen_model,
            "base_url": qwen_base_url,
            "source": qwen_source,
            "response_json": str(qwen_response_json) if qwen_response_json else None,
            "plan_path": str(qwen_plan_path),
            "validated_order": [call.fn for call in plan.calls],
            "primitive_ids": [call.primitive_id for call in plan.calls],
        },
        "smolvla": {
            "model_id": smolvla_model_id,
            "status": smolvla.status,
            "report_path": smolvla.report_path,
            "trace_path": smolvla.trace_path,
            "blocker_path": smolvla.blocker_path,
            "action_shape": smolvla.action_shape,
            "rollout_steps": smolvla.rollout_steps,
            "task_prompt": prompt,
        },
        "env": {
            "env_id": env_id,
            "action_dim": action_dim,
            "use_real_camera_inputs": use_real_camera_inputs,
            "allow_download": allow_download,
            "device": device,
        },
        "report_path": str(output_dir / "qwen_smolvla_e2e_report.json"),
    }
    Path(report["report_path"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _task_prompt_from_plan(plan: SO101ToolPlan) -> str:
    primitive_lines = " ".join(
        f"{call.index + 1}. {call.prompt}" for call in plan.calls
    )
    return f"{plan.task}. Execute this validated SO101 primitive chain: {primitive_lines}"


def _qwen_client(
    *,
    qwen_base_url: str | None,
    qwen_response_json: Path | None,
    qwen_api_key: str | None,
) -> tuple[ChatToolClient, str]:
    if qwen_response_json is not None:
        return SavedQwenResponseClient(qwen_response_json), "saved_response_json"
    if not qwen_base_url:
        raise ValueError("Provide --qwen-base-url for live Qwen or --qwen-response-json for offline mock output.")
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


if __name__ == "__main__":
    main()
