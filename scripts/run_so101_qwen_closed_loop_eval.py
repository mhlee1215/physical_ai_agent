#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from physical_ai_agent.agent_core.qwen_so101_closed_loop import (
    LoopArtifactConfig,
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
    parser.add_argument("--precondition-plan-json", type=Path)
    parser.add_argument("--policy-path", default=None, help="Default SmolVLA checkpoint/path for every primitive.")
    parser.add_argument(
        "--primitive-policy",
        action="append",
        default=[],
        help="Per-primitive route as primitive_id=policy_path. Repeat for three separately trained checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_workspace/qwen_so101_closed_loop"))
    parser.add_argument("--env-id", default=DEFAULT_SO101_ENV_ID)
    parser.add_argument(
        "--env-object-color",
        choices=["red", "orange", "yellow", "green", "blue", "purple", "black", "white"],
        help="For MuJoCoPickLift-v1, force a single cube target color so the rendered object matches the prompt.",
    )
    parser.add_argument("--env-cube-half-size", type=float, default=0.0125)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=98100)
    parser.add_argument("--start-contract", default="default_reset")
    parser.add_argument(
        "--start-report-path",
        type=Path,
        help="SO101 export report whose episode order is replayed exactly for closed-loop starts.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-steps-per-primitive", type=int, default=None)
    parser.add_argument("--policy-n-action-steps", type=int, default=15)
    parser.add_argument("--policy-num-steps", type=int, default=10)
    parser.add_argument("--valid-mask-checkpoint", type=Path)
    parser.add_argument("--valid-mask-threshold", type=float, default=0.5)
    parser.add_argument("--valid-mask-consecutive", type=int, default=2)
    parser.add_argument("--record-loop-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--render-loop-media",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render PNG/MP4 media during the rollout. Default records replay metadata only.",
    )
    parser.add_argument("--artifact-width", type=int, default=128)
    parser.add_argument("--artifact-height", type=int, default=128)
    parser.add_argument("--artifact-fps", type=int, default=12)
    parser.add_argument("--artifact-every-n-steps", type=int, default=1)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate Qwen plan and policy routing without loading SO101/SmolVLA.",
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan, qwen_artifacts = _load_or_build_plan(args)
    precondition_plan = _load_precondition_plan(args.precondition_plan_json)
    _write_qwen_artifacts(args.output_dir, qwen_artifacts, plan)
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
        if args.valid_mask_checkpoint is None:
            raise SystemExit("--valid-mask-checkpoint is required for Qwen closed-loop tests")
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
            valid_mask_checkpoint=args.valid_mask_checkpoint,
            valid_mask_threshold=args.valid_mask_threshold,
            valid_mask_consecutive=args.valid_mask_consecutive,
            artifact_config=LoopArtifactConfig(
                enabled=bool(args.record_loop_artifacts),
                render_media=bool(args.render_loop_media),
                width=int(args.artifact_width),
                height=int(args.artifact_height),
                fps=int(args.artifact_fps),
                every_n_steps=int(args.artifact_every_n_steps),
            ),
            env_config=_env_config_metadata_for_args(args),
            start_contract=args.start_contract,
            start_report_path=args.start_report_path,
            precondition_plan=precondition_plan,
            env_factory=_env_factory_for_args(args),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_pass and report["status"] not in {"passed", "planned"}:
        sys.exit(1)


def _load_or_build_plan(args: argparse.Namespace) -> tuple[SO101ToolPlan, dict[str, object]]:
    _validate_prompt_env_alignment(args)
    if args.qwen_plan_json:
        payload = json.loads(args.qwen_plan_json.read_text(encoding="utf-8"))
        return _plan_from_dict(payload), {"source": "qwen_plan_json", "plan_json": payload}
    qwen_client, source = _qwen_client(
        qwen_base_url=args.qwen_base_url,
        qwen_response_json=args.qwen_response_json,
        qwen_api_key=args.qwen_api_key,
    )
    recording_client = RecordingQwenClient(qwen_client)
    planner = QwenSO101ToolPlanner(client=recording_client, model=args.qwen_model)
    plan = planner.plan(task=args.task, target_object=args.object)
    return plan, {
        "source": source,
        "request": recording_client.last_request,
        "response": recording_client.last_response,
    }


def _load_precondition_plan(path: Path | None) -> SO101ToolPlan | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _plan_from_dict(payload)


def _validate_prompt_env_alignment(args: argparse.Namespace) -> None:
    if args.env_id == "MuJoCoReach-v1":
        raise ValueError(
            "Qwen edge-grasp closed-loop tests must not use MuJoCoReach-v1; "
            "use MuJoCoPickLift-v1 so the visual target is a graspable cube."
        )
    if args.env_object_color:
        expected = f"{args.env_object_color} cube"
        if str(args.object).strip().lower() != expected:
            raise ValueError(
                f"Qwen target object {args.object!r} does not match closed-loop "
                f"env object {expected!r}."
            )


def _env_factory_for_args(args: argparse.Namespace):
    env_kwargs = _env_kwargs_for_args(args)

    def make_env(env_id: str, render_mode: str | None):
        if env_id == "MuJoCoPickLift-v1" and not args.env_object_color:
            from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

            if render_mode is not None:
                raise ValueError("high-contrast SO101 pick-lift closed-loop env only supports render_mode=None")
            return _GymEnvAdapter(make_high_contrast_picklift_env(), env_id=env_id)
        from physical_ai_agent.sim.so101_nexus_env import SO101NexusEnv

        return SO101NexusEnv(env_id, render_mode, env_kwargs=env_kwargs)

    return make_env


class _GymEnvAdapter:
    def __init__(self, env, *, env_id: str) -> None:
        self.env = env
        self.env_id = env_id

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_dim(self) -> int:
        return int(self.action_space.shape[0])

    def reset(self, seed: int = 0):
        obs, info = self.env.reset(seed=seed)
        return _as_float_list(obs), dict(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return _as_float_list(obs), float(reward), bool(terminated), bool(truncated), dict(info)

    def close(self) -> None:
        self.env.close()


def _as_float_list(values) -> list[float]:
    return [float(value) for value in values]


def _env_kwargs_for_args(args: argparse.Namespace) -> dict[str, object]:
    if args.env_id != "MuJoCoPickLift-v1" or not args.env_object_color:
        if args.env_id == "MuJoCoPickLift-v1":
            return {
                "object_shape": "cube",
                "object_color": "dataset_seeded_high_contrast",
                "object_colors": ["red", "blue", "green"],
                "cube_half_sizes": [0.0125, 0.015, 0.0175],
                "source": "train_so101_wrist_ego_visual_servo.make_high_contrast_picklift_env",
                "start_contract": args.start_contract,
            }
        return {}
    from so101_nexus_core.config import PickConfig
    from so101_nexus_core.objects import CubeObject

    return {
        "config": PickConfig(
            objects=CubeObject(
                color=args.env_object_color,
                half_size=float(args.env_cube_half_size),
            ),
            n_distractors=0,
        )
    }


def _env_config_metadata_for_args(args: argparse.Namespace) -> dict[str, object]:
    if args.env_id != "MuJoCoPickLift-v1" or not args.env_object_color:
        return {}
    return {
        "object_shape": "cube",
        "object_color": args.env_object_color,
        "cube_half_size": float(args.env_cube_half_size),
        "n_distractors": 0,
        "start_contract": args.start_contract,
    }


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


class RecordingQwenClient:
    def __init__(self, client: ChatToolClient) -> None:
        self.client = client
        self.last_request: dict | None = None
        self.last_response: dict | None = None

    def create_tool_plan(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str,
        temperature: float,
    ) -> dict:
        self.last_request = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
        }
        response = self.client.create_tool_plan(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        self.last_response = response
        return response


def _write_qwen_artifacts(output_dir: Path, artifacts: dict[str, object], plan: SO101ToolPlan) -> None:
    qwen_dir = output_dir / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    (qwen_dir / "qwen_artifacts_manifest.json").write_text(
        json.dumps(
            {
                "source": artifacts.get("source"),
                "request_path": str(qwen_dir / "qwen_raw_request.json")
                if artifacts.get("request") is not None
                else None,
                "response_path": str(qwen_dir / "qwen_raw_response.json")
                if artifacts.get("response") is not None
                else None,
                "plan_path": str(qwen_dir / "qwen_plan.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if artifacts.get("request") is not None:
        (qwen_dir / "qwen_raw_request.json").write_text(
            json.dumps(artifacts["request"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if artifacts.get("response") is not None:
        (qwen_dir / "qwen_raw_response.json").write_text(
            json.dumps(artifacts["response"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (qwen_dir / "qwen_plan.json").write_text(
        json.dumps({"plan": plan_to_serializable(plan)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def plan_to_serializable(plan: SO101ToolPlan) -> dict:
    return {
        "task": plan.task,
        "model": plan.model,
        "thinking_mode": plan.thinking_mode,
        "calls": [
            {
                "index": call.index,
                "fn": call.fn,
                "object": call.object,
                "primitive_id": call.primitive_id,
                "prompt": call.prompt,
                "max_steps": call.max_steps,
            }
            for call in plan.calls
        ],
    }


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
