from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from physical_ai_agent.agent_core.qwen_so101_tool_planner import SO101ToolPlan, plan_to_dict
from physical_ai_agent.policies.smolvla_real import (
    _build_batch_for_policy,
    _clip_action,
    _load_pretrained_policy,
    _policy_device_metadata,
    _tensor_to_float_list,
)
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv


PolicyLoader = Callable[[str, bool, str], Any]
EnvFactory = Callable[[str, str | None], Any]
BatchBuilder = Callable[..., tuple[dict[str, Any], dict[str, str]]]


@dataclass(frozen=True)
class PrimitivePolicyRoute:
    primitive_id: str
    policy_path: str


def parse_primitive_policy_routes(values: list[str]) -> dict[str, str]:
    routes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--primitive-policy values must use primitive_id=policy_path, "
                f"got {value!r}"
            )
        primitive_id, policy_path = value.split("=", 1)
        primitive_id = primitive_id.strip()
        policy_path = policy_path.strip()
        if not primitive_id or not policy_path:
            raise ValueError(f"invalid primitive policy route: {value!r}")
        routes[primitive_id] = policy_path
    return routes


def resolve_policy_routes(
    plan: SO101ToolPlan,
    *,
    default_policy_path: str | None,
    primitive_policy_paths: dict[str, str],
) -> list[PrimitivePolicyRoute]:
    routes = []
    missing = []
    for call in plan.calls:
        policy_path = primitive_policy_paths.get(call.primitive_id) or default_policy_path
        if not policy_path:
            missing.append(call.primitive_id)
            continue
        routes.append(PrimitivePolicyRoute(primitive_id=call.primitive_id, policy_path=policy_path))
    if missing:
        raise ValueError(
            "missing policy path for primitive(s): "
            f"{', '.join(missing)}. Provide --policy-path or --primitive-policy."
        )
    return routes


def write_plan_only_report(
    *,
    plan: SO101ToolPlan,
    output_dir: Path,
    policy_routes: list[PrimitivePolicyRoute],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "operation": "so101_qwen_closed_loop_eval",
        "status": "planned",
        "plan": plan_to_dict(plan),
        "policy_routes": [asdict(route) for route in policy_routes],
        "report_path": str(output_dir / "qwen_closed_loop_eval_report.json"),
    }
    _write_report(report, Path(report["report_path"]))
    return report


def run_closed_loop_plan(
    *,
    plan: SO101ToolPlan,
    output_dir: Path,
    default_policy_path: str | None,
    primitive_policy_paths: dict[str, str] | None = None,
    env_id: str = DEFAULT_SO101_ENV_ID,
    episodes: int = 1,
    seed: int = 0,
    device: str = "auto",
    local_files_only: bool = True,
    max_steps_per_primitive: int | None = None,
    env_factory: EnvFactory = SO101NexusEnv,
    policy_loader: PolicyLoader = _load_pretrained_policy,
    batch_builder: BatchBuilder = _build_batch_for_policy,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    primitive_policy_paths = primitive_policy_paths or {}
    policy_routes = resolve_policy_routes(
        plan,
        default_policy_path=default_policy_path,
        primitive_policy_paths=primitive_policy_paths,
    )
    route_by_primitive = {route.primitive_id: route.policy_path for route in policy_routes}
    policy_cache: dict[str, Any] = {}
    policy_metadata: dict[str, dict[str, Any]] = {}
    episodes_out = []
    blocker: str | None = None

    try:
        for episode in range(episodes):
            episodes_out.append(
                _run_episode(
                    plan=plan,
                    output_dir=output_dir,
                    route_by_primitive=route_by_primitive,
                    policy_cache=policy_cache,
                    policy_metadata=policy_metadata,
                    policy_loader=policy_loader,
                    env_factory=env_factory,
                    env_id=env_id,
                    seed=seed + episode,
                    episode=episode,
                    device=device,
                    local_files_only=local_files_only,
                    max_steps_per_primitive=max_steps_per_primitive,
                    batch_builder=batch_builder,
                )
            )
        status = "passed"
    except Exception as exc:  # noqa: BLE001
        status = "blocked"
        blocker = _short_error(exc)

    successful = [
        row["final_success"]
        for row in episodes_out
        if isinstance(row.get("final_success"), bool)
    ]
    report = {
        "operation": "so101_qwen_closed_loop_eval",
        "status": status,
        "blocker": blocker,
        "duration_s": round(perf_counter() - started, 4),
        "env_id": env_id,
        "seed": seed,
        "episodes_requested": int(episodes),
        "episodes_completed": len(episodes_out),
        "success_rate": (sum(1 for value in successful if value) / len(successful)) if successful else None,
        "plan": plan_to_dict(plan),
        "policy_routes": [asdict(route) for route in policy_routes],
        "policy_metadata": policy_metadata,
        "episodes": episodes_out,
        "report_path": str(output_dir / "qwen_closed_loop_eval_report.json"),
    }
    _write_report(report, Path(report["report_path"]))
    if blocker:
        (output_dir / "qwen_closed_loop_blocker.md").write_text(
            f"# Qwen SO101 Closed-Loop Blocker\n\n- Blocker: `{blocker}`\n",
            encoding="utf-8",
        )
    return report


def _run_episode(
    *,
    plan: SO101ToolPlan,
    output_dir: Path,
    route_by_primitive: dict[str, str],
    policy_cache: dict[str, Any],
    policy_metadata: dict[str, dict[str, Any]],
    policy_loader: PolicyLoader,
    env_factory: EnvFactory,
    env_id: str,
    seed: int,
    episode: int,
    device: str,
    local_files_only: bool,
    max_steps_per_primitive: int | None,
    batch_builder: BatchBuilder,
) -> dict[str, Any]:
    env = env_factory(env_id, None)
    trace_path = output_dir / f"qwen_closed_loop_episode_{episode:03d}.jsonl"
    records = []
    primitive_summaries = []
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    try:
        obs, reset_info = env.reset(seed=seed)
        action_dim = int(env.action_dim)
        global_step = 0
        for call in plan.calls:
            policy_path = route_by_primitive[call.primitive_id]
            policy = _policy_for_route(
                policy_cache=policy_cache,
                policy_metadata=policy_metadata,
                policy_loader=policy_loader,
                policy_path=policy_path,
                local_files_only=local_files_only,
                device=device,
            )
            if hasattr(policy, "reset"):
                policy.reset()
            primitive_records = 0
            primitive_reward = 0.0
            step_budget = (
                min(int(call.max_steps), int(max_steps_per_primitive))
                if max_steps_per_primitive is not None
                else int(call.max_steps)
            )
            for primitive_step in range(step_budget):
                batch, image_feature_mapping = batch_builder(
                    policy,
                    obs,
                    camera_pixels={},
                    instruction=call.prompt,
                    local_files_only=local_files_only,
                )
                raw_action = policy.select_action(batch)
                action = _clip_action(_action_to_float_list(raw_action), action_dim)
                obs, reward, terminated, truncated, info = env.step(action)
                final_info = dict(info)
                total_reward += float(reward)
                primitive_reward += float(reward)
                row = {
                    "episode": episode,
                    "global_step": global_step,
                    "primitive_step": primitive_step,
                    "fn": call.fn,
                    "primitive_id": call.primitive_id,
                    "prompt": call.prompt,
                    "policy_path": policy_path,
                    "observation": obs,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _jsonable_info(info),
                    "image_feature_mapping": image_feature_mapping,
                }
                records.append(row)
                primitive_records += 1
                global_step += 1
                if terminated or truncated:
                    break
            primitive_summaries.append(
                {
                    "fn": call.fn,
                    "primitive_id": call.primitive_id,
                    "prompt": call.prompt,
                    "policy_path": policy_path,
                    "steps": primitive_records,
                    "reward": primitive_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "episode": episode,
        "seed": seed,
        "reset_info": _jsonable_info(reset_info),
        "steps": len(records),
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_info": _jsonable_info(final_info),
        "final_success": _success_from_info(final_info),
        "primitive_summaries": primitive_summaries,
        "trace_path": str(trace_path),
    }


def _policy_for_route(
    *,
    policy_cache: dict[str, Any],
    policy_metadata: dict[str, dict[str, Any]],
    policy_loader: PolicyLoader,
    policy_path: str,
    local_files_only: bool,
    device: str,
) -> Any:
    if policy_path not in policy_cache:
        policy = policy_loader(policy_path, local_files_only, device)
        policy_cache[policy_path] = policy
        policy_metadata[policy_path] = _policy_device_metadata(policy)
    return policy_cache[policy_path]


def _success_from_info(info: dict[str, Any]) -> bool | None:
    for key in ("success", "is_success", "task_success", "is_obj_placed"):
        if key in info:
            return bool(info[key])
    return None


def _action_to_float_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return _tensor_to_float_list(value)


def _jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in dict(info).items():
        try:
            json.dumps(value)
            out[str(key)] = value
        except TypeError:
            out[str(key)] = str(value)
    return out


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
