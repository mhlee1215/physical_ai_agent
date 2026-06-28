from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from physical_ai_agent.agent_core.qwen_so101_tool_planner import (
    SO101PrimitiveCall,
    SO101ToolPlan,
    plan_to_dict,
)
from physical_ai_agent.policies.smolvla_real import (
    _build_batch_for_policy,
    _clip_action,
    _load_pretrained_policy,
    _policy_device_metadata,
    _tensor_to_float_list,
)
from physical_ai_agent.policies.lerobot_policy_runner import load_lerobot_policy_runner
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv


PolicyLoader = Callable[[str, bool, str], Any]
EnvFactory = Callable[[str, str | None], Any]
BatchBuilder = Callable[..., tuple[dict[str, Any], dict[str, str]]]


@dataclass(frozen=True)
class LoopArtifactConfig:
    enabled: bool = False
    render_media: bool = False
    width: int = 128
    height: int = 128
    fps: int = 12
    every_n_steps: int = 1


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


def _routing_plan_with_preconditions(
    plan: SO101ToolPlan,
    precondition_calls: list[SO101PrimitiveCall],
) -> SO101ToolPlan:
    if not precondition_calls:
        return plan
    return SO101ToolPlan(
        task=plan.task,
        model=plan.model,
        thinking_mode=plan.thinking_mode,
        calls=[*precondition_calls, *plan.calls],
    )


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
    policy_n_action_steps: int | None = 15,
    policy_num_steps: int | None = 10,
    valid_mask_checkpoint: Path | None = None,
    valid_mask_threshold: float = 0.5,
    valid_mask_consecutive: int = 2,
    valid_mask_head: Any | None = None,
    artifact_config: LoopArtifactConfig | None = None,
    env_config: dict[str, Any] | None = None,
    start_contract: str = "default_reset",
    start_report_path: Path | None = None,
    precondition_plan: SO101ToolPlan | None = None,
    action_contract_mode: str = "processor",
    env_factory: EnvFactory = SO101NexusEnv,
    policy_loader: PolicyLoader = _load_pretrained_policy,
    batch_builder: BatchBuilder = _build_batch_for_policy,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    primitive_policy_paths = primitive_policy_paths or {}
    if valid_mask_head is None:
        if valid_mask_checkpoint is not None:
            from physical_ai_agent.policies.so101_valid_mask import load_valid_mask_head

            valid_mask_head = load_valid_mask_head(valid_mask_checkpoint, device=None if device == "auto" else device)
    precondition_calls = list(precondition_plan.calls) if precondition_plan is not None else []
    routing_plan = _routing_plan_with_preconditions(plan, precondition_calls)
    policy_routes = resolve_policy_routes(
        routing_plan,
        default_policy_path=default_policy_path,
        primitive_policy_paths=primitive_policy_paths,
    )
    dataset_action_bounds = _dataset_action_bounds_for_policy_routes(policy_routes)
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
                    policy_n_action_steps=policy_n_action_steps,
                    policy_num_steps=policy_num_steps,
                    valid_mask_head=valid_mask_head,
                    valid_mask_threshold=valid_mask_threshold,
                    valid_mask_consecutive=valid_mask_consecutive,
                    artifact_config=artifact_config or LoopArtifactConfig(),
                    env_config=env_config or {},
                    start_contract=start_contract,
                    start_report_path=start_report_path,
                    precondition_calls=precondition_calls,
                    action_contract_mode=action_contract_mode,
                    dataset_action_bounds=dataset_action_bounds,
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
        "env_config": env_config or {},
        "start_contract": start_contract,
        "start_report_path": str(start_report_path) if start_report_path else None,
        "precondition_plan": plan_to_dict(precondition_plan) if precondition_plan else None,
        "camera_contract": {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
            "observation.images.camera3": "wrist_cam duplicate",
        },
        "seed": seed,
        "episodes_requested": int(episodes),
        "episodes_completed": len(episodes_out),
        "success_rate": (sum(1 for value in successful if value) / len(successful)) if successful else None,
        "plan": plan_to_dict(plan),
        "policy_routes": [asdict(route) for route in policy_routes],
        "valid_mask": {
            "checkpoint": str(valid_mask_checkpoint) if valid_mask_checkpoint else None,
            "threshold": float(valid_mask_threshold),
            "consecutive": int(valid_mask_consecutive),
            "required_for_loop_test": valid_mask_head is not None,
            "mode": "valid_mask" if valid_mask_head is not None else "fixed_horizon",
        },
        "policy_metadata": policy_metadata,
        "policy_rollout_config": _merged_policy_rollout_config(policy_metadata),
        "action_contract_mode": action_contract_mode,
        "loop_artifact_config": asdict(artifact_config or LoopArtifactConfig()),
        "episodes": episodes_out,
        "report_path": str(output_dir / "qwen_closed_loop_eval_report.json"),
    }
    report["action_contract"] = _action_contract_summary(
        episodes_out=episodes_out,
        policy_routes=policy_routes,
    )
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
    policy_n_action_steps: int | None,
    policy_num_steps: int | None,
    valid_mask_head: Any,
    valid_mask_threshold: float,
    valid_mask_consecutive: int,
    artifact_config: LoopArtifactConfig,
    env_config: dict[str, Any],
    start_contract: str,
    start_report_path: Path | None,
    precondition_calls: list[SO101PrimitiveCall],
    action_contract_mode: str,
    dataset_action_bounds: dict[str, Any],
    batch_builder: BatchBuilder,
) -> dict[str, Any]:
    env = env_factory(env_id, None)
    trace_path = output_dir / f"qwen_closed_loop_episode_{episode:03d}.jsonl"
    episode_media_dir = output_dir / "media" / f"episode_{episode:03d}"
    records = []
    primitive_summaries = []
    renderers = _make_renderers_or_none(env, artifact_config)
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    precondition_summaries = []
    terminated = False
    truncated = False
    try:
        obs, reset_info = env.reset(seed=seed)
        start_state = _apply_start_contract_to_env(
            env=env,
            start_contract=start_contract,
            seed=seed,
            episode_index=episode,
            object_color=_start_contract_object_color(env_config),
            start_report_path=start_report_path,
        )
        if start_state.get("observation") is not None:
            obs = start_state["observation"]
        action_dim = int(env.action_dim)
        for call in precondition_calls:
            pre_summary, obs, reward_delta, final_info, terminated, truncated = _run_precondition_call(
                call=call,
                obs=obs,
                env=env,
                renderers=renderers,
                route_by_primitive=route_by_primitive,
                policy_cache=policy_cache,
                policy_metadata=policy_metadata,
                policy_loader=policy_loader,
                action_dim=action_dim,
                device=device,
                local_files_only=local_files_only,
                max_steps_per_primitive=max_steps_per_primitive,
                policy_n_action_steps=policy_n_action_steps,
                policy_num_steps=policy_num_steps,
                valid_mask_head=valid_mask_head,
                valid_mask_threshold=valid_mask_threshold,
                valid_mask_consecutive=valid_mask_consecutive,
                action_contract_mode=action_contract_mode,
                dataset_action_bounds=dataset_action_bounds,
                batch_builder=batch_builder,
            )
            precondition_summaries.append(pre_summary)
            total_reward += reward_delta
            if terminated or truncated:
                break
        global_step = 0
        for call in plan.calls:
            if terminated or truncated:
                break
            policy_path = route_by_primitive[call.primitive_id]
            policy_executor = _policy_for_route(
                policy_cache=policy_cache,
                policy_metadata=policy_metadata,
                policy_loader=policy_loader,
                policy_path=policy_path,
                local_files_only=local_files_only,
                device=device,
            )
            policy = _policy_model(policy_executor)
            _override_policy_rollout_config(
                policy,
                n_action_steps=policy_n_action_steps,
                num_steps=policy_num_steps,
            )
            policy_rollout_config = _policy_rollout_config(policy)
            policy_metadata.setdefault(policy_path, {}).update(
                {"rollout_config": policy_rollout_config}
            )
            if hasattr(policy, "reset"):
                policy.reset()
            primitive_records = 0
            primitive_reward = 0.0
            primitive_frame_paths: list[str] = []
            primitive_row_indexes: list[int] = []
            camera_pixels = _render_policy_cameras(env, renderers)
            _require_policy_cameras(camera_pixels)
            step_budget = (
                min(int(call.max_steps), int(max_steps_per_primitive))
                if max_steps_per_primitive is not None
                else int(call.max_steps)
            )
            valid_mask_decisions: list[dict[str, Any]] = []
            valid_mask_stop_reason = "max_horizon"
            while primitive_records < step_budget:
                remaining_budget = step_budget - primitive_records
                chunk_horizon = _valid_mask_requery_horizon(
                    policy_rollout_config=policy_rollout_config,
                    policy_n_action_steps=policy_n_action_steps,
                    remaining_budget=remaining_budget,
                )
                if valid_mask_head is None:
                    valid_mask_budget = chunk_horizon
                    valid_mask_reason = "fixed_horizon"
                    valid_mask_probs = []
                else:
                    valid_mask_budget, valid_mask_reason, valid_mask_probs = _valid_mask_primitive_budget(
                        policy=policy,
                        obs=obs,
                        camera_pixels=camera_pixels,
                        instruction=call.prompt,
                        local_files_only=local_files_only,
                        batch_builder=batch_builder,
                        valid_mask_head=valid_mask_head,
                        max_horizon=chunk_horizon,
                        threshold=valid_mask_threshold,
                        consecutive=valid_mask_consecutive,
                    )
                execute_steps = min(remaining_budget, chunk_horizon, valid_mask_budget)
                valid_mask_decision = {
                    "chunk_start_primitive_step": int(primitive_records),
                    "chunk_start_global_step": int(global_step),
                    "budget": int(execute_steps),
                    "chunk_horizon": int(chunk_horizon),
                    "reason": valid_mask_reason,
                    "probs": valid_mask_probs,
                }
                valid_mask_decisions.append(valid_mask_decision)
                for _chunk_step in range(execute_steps):
                    primitive_step = primitive_records
                    record_media = _should_render_media(artifact_config, primitive_step)
                    camera_pixels = _render_policy_cameras(env, renderers)
                    _require_policy_cameras(camera_pixels)
                    policy_input_images = (
                        _write_policy_input_images(
                            camera_pixels=camera_pixels,
                            episode_media_dir=episode_media_dir,
                            global_step=global_step,
                        )
                        if record_media
                        else {}
                    )
                    batch, image_feature_mapping = batch_builder(
                        policy,
                        obs,
                        camera_pixels=camera_pixels,
                        instruction=call.prompt,
                        local_files_only=local_files_only,
                    )
                    processed_action = _select_env_action_with_trace(
                        policy_executor=policy_executor,
                        policy=policy,
                        obs=obs,
                        action_base_qpos=_current_action_qpos(env, obs, action_dim),
                        camera_pixels=camera_pixels,
                        instruction=call.prompt,
                        action_dim=action_dim,
                        action_contract_mode=action_contract_mode,
                        dataset_action_bounds=dataset_action_bounds,
                    )
                    action = processed_action["action"]
                    obs, reward, terminated, truncated, info = env.step(action)
                    robot_frame_path = (
                        _write_robot_frame(
                            env=env,
                            renderers=renderers,
                            episode_media_dir=episode_media_dir,
                            global_step=global_step,
                        )
                        if record_media
                        else None
                    )
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
                        "policy_rollout_config": policy_rollout_config,
                        "valid_mask": valid_mask_decision,
                        "observation": obs,
                        "sim_snapshot": _snapshot_sim_state_or_none(env),
                        "action": action,
                        "policy_output": {
                            **processed_action,
                        },
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": _jsonable_info(info),
                        "image_feature_mapping": image_feature_mapping,
                        "policy_input_camera_names": sorted(camera_pixels),
                        "media": {
                            "policy_input_images": policy_input_images,
                            "robot_frame": robot_frame_path,
                            "render_mode": "inline" if record_media else "deferred",
                        },
                        "render_replay": {
                            "env_id": env_id,
                            "env_config": env_config,
                            "start_contract": start_contract,
                            "seed": seed,
                            "artifact_width": artifact_config.width,
                            "artifact_height": artifact_config.height,
                            "artifact_fps": artifact_config.fps,
                            "artifact_every_n_steps": artifact_config.every_n_steps,
                        },
                    }
                    primitive_row_indexes.append(len(records))
                    if robot_frame_path:
                        primitive_frame_paths.append(robot_frame_path)
                    records.append(row)
                    primitive_records += 1
                    global_step += 1
                    if action_contract_mode == "visual_servo_delta_q" and _visual_servo_should_stop(processed_action):
                        valid_mask_stop_reason = "visual_servo_stop"
                        break
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
                if valid_mask_reason == "valid_mask_stop":
                    valid_mask_stop_reason = "valid_mask_stop"
                    break
            primitive_videos = _write_primitive_videos(
                frame_paths=primitive_frame_paths,
                episode_media_dir=episode_media_dir,
                primitive_id=call.primitive_id,
                iteration=len(primitive_summaries) + 1,
                fps=artifact_config.fps,
            )
            if primitive_videos:
                for row_index in primitive_row_indexes:
                    records[row_index].setdefault("media", {}).update(primitive_videos)
            primitive_summaries.append(
                {
                    "fn": call.fn,
                    "primitive_id": call.primitive_id,
                    "prompt": call.prompt,
                    "policy_path": policy_path,
                    "policy_rollout_config": policy_rollout_config,
                    "valid_mask": {
                        "budget": int(primitive_records),
                        "reason": valid_mask_stop_reason,
                        "decisions": valid_mask_decisions,
                    },
                    "steps": primitive_records,
                    "reward": primitive_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "media": primitive_videos,
                }
            )
            if terminated or truncated:
                break
    finally:
        _close_renderers(renderers)
        env.close()

    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "episode": episode,
        "seed": seed,
        "start_contract": start_contract,
        "start_contract_state": _jsonable_info(start_state),
        "precondition_summaries": precondition_summaries,
        "reset_info": _jsonable_info(reset_info),
        "steps": len(records),
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_info": _jsonable_info(final_info),
        "final_success": _success_from_info(final_info),
        "primitive_summaries": primitive_summaries,
        "trace_path": str(trace_path),
        "media_root": str(episode_media_dir) if artifact_config.render_media else None,
        "action_space": _action_space_summary(env),
    }


def _action_space_summary(env: Any) -> dict[str, Any]:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return {}
    return {
        "low": _action_to_float_list(getattr(action_space, "low", [])),
        "high": _action_to_float_list(getattr(action_space, "high", [])),
    }


def _action_contract_summary(
    *,
    episodes_out: list[dict[str, Any]],
    policy_routes: list[PrimitivePolicyRoute],
) -> dict[str, Any]:
    dataset_bounds = _dataset_action_bounds_for_policy_routes(policy_routes)
    env_bounds = next((episode.get("action_space") for episode in episodes_out if episode.get("action_space")), {})
    traces = [Path(str(episode.get("trace_path"))) for episode in episodes_out if episode.get("trace_path")]
    records = []
    for trace_path in traces:
        if trace_path.exists():
            records.extend(_read_jsonl(trace_path))
    fields = {
        "env_action": lambda row: row.get("action"),
        "processor_raw_action": lambda row: (row.get("policy_output") or {}).get("processor_raw_action"),
        "processor_postprocessed_action": lambda row: (row.get("policy_output") or {}).get("processor_postprocessed_action"),
    }
    summary = {
        "dataset_action_bounds": dataset_bounds,
        "env_action_bounds": env_bounds,
        "record_count": len(records),
        "fields": {},
    }
    for name, getter in fields.items():
        values = [getter(row) for row in records]
        values = [value for value in values if isinstance(value, list) and value]
        summary["fields"][name] = _action_values_summary(
            values,
            dataset_bounds=dataset_bounds,
            env_bounds=env_bounds,
        )
    return summary


def _dataset_action_bounds_for_policy_routes(policy_routes: list[PrimitivePolicyRoute]) -> dict[str, Any]:
    for route in policy_routes:
        bounds = _dataset_action_bounds_for_policy_path(Path(route.policy_path))
        if bounds:
            return bounds
    return {}


def _dataset_action_bounds_for_policy_path(policy_path: Path) -> dict[str, Any]:
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.exists():
        return {}
    try:
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        dataset_root = Path(str((train_config.get("dataset") or {}).get("root") or ""))
        if not dataset_root.is_absolute():
            dataset_root = Path.cwd() / dataset_root
        stats_path = dataset_root / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        action_stats = stats.get("action") or {}
        return {
            "dataset_root": str(dataset_root),
            "min": [float(value) for value in action_stats.get("min", [])],
            "max": [float(value) for value in action_stats.get("max", [])],
            "mean": [float(value) for value in action_stats.get("mean", [])],
            "std": [float(value) for value in action_stats.get("std", [])],
        }
    except Exception:
        return {}


def _action_values_summary(
    values: list[list[float]],
    *,
    dataset_bounds: dict[str, Any],
    env_bounds: dict[str, Any],
) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    try:
        import numpy as np

        array = np.asarray(values, dtype=float)
        summary: dict[str, Any] = {
            "count": int(array.shape[0]),
            "dim": int(array.shape[1]) if array.ndim == 2 else None,
            "min": np.round(array.min(axis=0), 6).tolist(),
            "max": np.round(array.max(axis=0), 6).tolist(),
            "mean": np.round(array.mean(axis=0), 6).tolist(),
            "std": np.round(array.std(axis=0), 6).tolist(),
        }
        for label, bounds in (("dataset", dataset_bounds), ("env", env_bounds)):
            low = bounds.get("min") or bounds.get("low")
            high = bounds.get("max") or bounds.get("high")
            if not low or not high:
                continue
            low_array = np.asarray(low, dtype=float)[: array.shape[1]]
            high_array = np.asarray(high, dtype=float)[: array.shape[1]]
            summary[f"below_{label}_min_ratio"] = np.round((array < low_array).mean(axis=0), 6).tolist()
            summary[f"above_{label}_max_ratio"] = np.round((array > high_array).mean(axis=0), 6).tolist()
            summary[f"any_outside_{label}_range_ratio"] = round(
                float(((array < low_array) | (array > high_array)).any(axis=1).mean()),
                6,
            )
        return summary
    except Exception as exc:
        return {"count": len(values), "error": _short_error(exc)}


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
        runner = _load_policy_runner_or_legacy_policy(policy_path, local_files_only, device, policy_loader)
        policy_cache[policy_path] = runner
        policy_metadata[policy_path] = _policy_execution_metadata(runner)
    return policy_cache[policy_path]


def _load_policy_runner_or_legacy_policy(
    policy_path: str,
    local_files_only: bool,
    device: str,
    policy_loader: PolicyLoader,
) -> Any:
    try:
        return load_lerobot_policy_runner(
            policy_path,
            device=_selected_policy_device(device),
            policy_type="smolvla",
            local_files_only=local_files_only,
        )
    except Exception:
        return policy_loader(policy_path, local_files_only, device)


def _selected_policy_device(device: str) -> str:
    if str(device).lower() != "auto":
        return str(device)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _policy_model(executor: Any) -> Any:
    return getattr(executor, "policy", executor)


def _policy_execution_metadata(executor: Any) -> dict[str, Any]:
    policy = _policy_model(executor)
    metadata = _policy_device_metadata(policy)
    metadata["processor_source"] = getattr(executor, "processor_source", "legacy_policy_no_processors")
    metadata["preprocessor_steps"] = [type(step).__name__ for step in getattr(getattr(executor, "preprocessor", None), "steps", [])]
    metadata["postprocessor_steps"] = [type(step).__name__ for step in getattr(getattr(executor, "postprocessor", None), "steps", [])]
    return metadata


def _select_env_action_with_trace(
    *,
    policy_executor: Any,
    policy: Any,
    obs: Any,
    action_base_qpos: Any | None = None,
    camera_pixels: dict[str, Any],
    instruction: str | None,
    action_dim: int,
    action_contract_mode: str,
    dataset_action_bounds: dict[str, Any],
) -> dict[str, Any]:
    if action_contract_mode == "legacy" or not hasattr(policy_executor, "select_action_with_trace"):
        raw_action = policy.select_action(
            _build_batch_for_policy(
                policy,
                obs,
                camera_pixels=camera_pixels,
                instruction=instruction,
                local_files_only=True,
            )[0]
        )
        action = _clip_action(_action_to_float_list(raw_action), action_dim)
        return {
            "action": action,
            "processor_raw_action": action,
            "processor_postprocessed_action": action,
            "processor_source": "legacy_policy_no_processors",
            "preprocessor_steps": [],
            "postprocessor_steps": [],
            "action_contract_mode": "legacy",
        }
    if action_contract_mode not in {"processor", "processor_dataset_clamp", "processor_delta_q", "visual_servo_delta_q"}:
        raise ValueError(f"unsupported action_contract_mode={action_contract_mode!r}")
    if hasattr(policy_executor, "select_action_with_trace"):
        raw_observation = _build_raw_lerobot_observation_for_runner(
            policy=policy,
            observation=obs,
            camera_pixels=camera_pixels,
            instruction=instruction,
        )
        if action_contract_mode == "visual_servo_delta_q":
            visual_trace = policy_executor.predict_visual_servo_with_trace(raw_observation)
            delta_q_action = _clip_action(_action_to_float_list(visual_trace["delta_q"]), action_dim)
            action = _action_from_delta_q(action_base_qpos or obs, delta_q_action, action_dim=action_dim)
            return {
                "action": action,
                "delta_q_action": delta_q_action,
                "visual_servo_prediction": visual_trace,
                "processor_source": visual_trace.get("processor_source"),
                "preprocessor_steps": [],
                "postprocessor_steps": [],
                "action_contract_mode": action_contract_mode,
            }
        trace = policy_executor.select_action_with_trace(raw_observation)
        raw_action = _clip_action(_action_to_float_list(trace["raw_action"]), action_dim)
        postprocessed_action = _clip_action(_action_to_float_list(trace["postprocessed_action"]), action_dim)
        action = _clip_action(_action_to_float_list(trace["action"]), action_dim)
        unclamped_action = list(action)
        delta_q_action = None
        if action_contract_mode == "processor_delta_q":
            delta_q_action = list(action)
            action = _action_from_delta_q(action_base_qpos or obs, delta_q_action, action_dim=action_dim)
        if action_contract_mode == "processor_dataset_clamp":
            action = _clamp_action_to_bounds(action, dataset_action_bounds)
        return {
            "action": action,
            "delta_q_action": delta_q_action,
            "processor_raw_action": raw_action,
            "processor_postprocessed_action": postprocessed_action,
            "unclamped_env_action": unclamped_action,
            "processor_source": trace.get("processor_source"),
            "preprocessor_steps": trace.get("preprocessor_steps"),
            "postprocessor_steps": trace.get("postprocessor_steps"),
            "action_contract_mode": action_contract_mode,
        }
    raise RuntimeError("unreachable action contract path")


def _visual_servo_should_stop(processed_action: dict[str, Any]) -> bool:
    prediction = processed_action.get("visual_servo_prediction")
    if not isinstance(prediction, dict):
        return False
    camera2 = prediction.get("camera2")
    if not isinstance(camera2, dict):
        return False
    try:
        return (
            float(prediction.get("stop_prob", 0.0)) >= 0.7
            and abs(float(camera2.get("dx_norm", 1.0))) <= 0.08
            and abs(float(camera2.get("dy_norm", 1.0))) <= 0.08
            and abs(float(camera2.get("edge_angle_error", 1.0))) <= 0.12
        )
    except (TypeError, ValueError):
        return False


def _action_from_delta_q(obs: Any, delta_q: list[float], *, action_dim: int) -> list[float]:
    state = _action_to_float_list(obs)
    if len(state) < action_dim:
        raise ValueError(f"observation state has {len(state)} dims, expected at least {action_dim}")
    return [float(state[index] + delta_q[index]) for index in range(action_dim)]


def _current_action_qpos(env: Any, fallback_obs: Any, action_dim: int) -> list[float]:
    gym_env = _gym_env_or_none(env)
    if gym_env is None:
        return _action_to_float_list(fallback_obs)[:action_dim]
    try:
        qpos = _current_sim_qpos(gym_env)
    except Exception:  # noqa: BLE001
        return _action_to_float_list(fallback_obs)[:action_dim]
    if len(qpos) < action_dim:
        return _action_to_float_list(fallback_obs)[:action_dim]
    return qpos[:action_dim]


def _clamp_action_to_bounds(action: list[float], bounds: dict[str, Any]) -> list[float]:
    low = bounds.get("min")
    high = bounds.get("max")
    if not low or not high:
        return action
    clamped = []
    for index, value in enumerate(action):
        if index >= len(low) or index >= len(high):
            clamped.append(float(value))
            continue
        clamped.append(min(max(float(value), float(low[index])), float(high[index])))
    return clamped


def _build_raw_lerobot_observation_for_runner(
    *,
    policy: Any,
    observation: Any,
    camera_pixels: dict[str, Any],
    instruction: str | None,
) -> dict[str, Any]:
    import torch

    config = policy.config
    state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else min(32, len(observation))
    state = torch.zeros(state_dim, dtype=torch.float32)
    source = torch.tensor(list(observation)[:state_dim], dtype=torch.float32)
    state[: len(source)] = source
    raw_observation: dict[str, Any] = {
        "observation.state": state,
        "task": instruction or "",
    }
    for index, (key, feature) in enumerate(config.image_features.items()):
        source_name = _source_camera_name_for_feature(index, camera_pixels)
        if source_name in camera_pixels:
            raw_observation[key] = _pixels_to_lerobot_image_tensor(camera_pixels[source_name], feature.shape)
    return raw_observation


def _source_camera_name_for_feature(index: int, camera_pixels: dict[str, Any]) -> str:
    preferred = ["egocentric_cam", "wrist_cam", "wrist_cam"]
    if index < len(preferred):
        return preferred[index]
    return next(iter(camera_pixels), "zero")


def _pixels_to_lerobot_image_tensor(pixels: Any, feature_shape: tuple[int, ...]) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    channels, height, width = feature_shape
    if channels != 3:
        raise ValueError(f"Expected RGB image feature with 3 channels, got {feature_shape}")
    image = Image.fromarray(pixels).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _run_precondition_call(
    *,
    call: SO101PrimitiveCall,
    obs: Any,
    env: Any,
    renderers: dict[str, Any],
    route_by_primitive: dict[str, str],
    policy_cache: dict[str, Any],
    policy_metadata: dict[str, dict[str, Any]],
    policy_loader: PolicyLoader,
    action_dim: int,
    device: str,
    local_files_only: bool,
    max_steps_per_primitive: int | None,
    policy_n_action_steps: int | None,
    policy_num_steps: int | None,
    valid_mask_head: Any,
    valid_mask_threshold: float,
    valid_mask_consecutive: int,
    action_contract_mode: str,
    dataset_action_bounds: dict[str, Any],
    batch_builder: BatchBuilder,
) -> tuple[dict[str, Any], Any, float, dict[str, Any], bool, bool]:
    policy_path = route_by_primitive[call.primitive_id]
    policy_executor = _policy_for_route(
        policy_cache=policy_cache,
        policy_metadata=policy_metadata,
        policy_loader=policy_loader,
        policy_path=policy_path,
        local_files_only=local_files_only,
        device=device,
    )
    policy = _policy_model(policy_executor)
    _override_policy_rollout_config(
        policy,
        n_action_steps=policy_n_action_steps,
        num_steps=policy_num_steps,
    )
    policy_rollout_config = _policy_rollout_config(policy)
    policy_metadata.setdefault(policy_path, {}).update({"rollout_config": policy_rollout_config})
    if hasattr(policy, "reset"):
        policy.reset()
    camera_pixels = _render_policy_cameras(env, renderers)
    _require_policy_cameras(camera_pixels)
    step_budget = (
        min(int(call.max_steps), int(max_steps_per_primitive))
        if max_steps_per_primitive is not None
        else int(call.max_steps)
    )
    valid_mask_decisions: list[dict[str, Any]] = []
    valid_mask_stop_reason = "max_horizon"
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    primitive_steps = 0
    while primitive_steps < step_budget:
        remaining_budget = step_budget - primitive_steps
        chunk_horizon = _valid_mask_requery_horizon(
            policy_rollout_config=policy_rollout_config,
            policy_n_action_steps=policy_n_action_steps,
            remaining_budget=remaining_budget,
        )
        if valid_mask_head is None:
            valid_mask_budget = chunk_horizon
            valid_mask_reason = "fixed_horizon"
            valid_mask_probs = []
        else:
            valid_mask_budget, valid_mask_reason, valid_mask_probs = _valid_mask_primitive_budget(
                policy=policy,
                obs=obs,
                camera_pixels=camera_pixels,
                instruction=call.prompt,
                local_files_only=local_files_only,
                batch_builder=batch_builder,
                valid_mask_head=valid_mask_head,
                max_horizon=chunk_horizon,
                threshold=valid_mask_threshold,
                consecutive=valid_mask_consecutive,
            )
        execute_steps = min(remaining_budget, chunk_horizon, valid_mask_budget)
        valid_mask_decision = {
            "chunk_start_primitive_step": int(primitive_steps),
            "budget": int(execute_steps),
            "chunk_horizon": int(chunk_horizon),
            "reason": valid_mask_reason,
            "probs": valid_mask_probs,
        }
        valid_mask_decisions.append(valid_mask_decision)
        for _chunk_step in range(execute_steps):
            camera_pixels = _render_policy_cameras(env, renderers)
            _require_policy_cameras(camera_pixels)
            action = _select_env_action_with_trace(
                policy_executor=policy_executor,
                policy=policy,
                obs=obs,
                action_base_qpos=_current_action_qpos(env, obs, action_dim),
                camera_pixels=camera_pixels,
                instruction=call.prompt,
                action_dim=action_dim,
                action_contract_mode=action_contract_mode,
                dataset_action_bounds=dataset_action_bounds,
            )["action"]
            obs, reward, terminated, truncated, info = env.step(action)
            final_info = dict(info)
            total_reward += float(reward)
            primitive_steps += 1
            if terminated or truncated:
                break
        if terminated or truncated:
            break
        if valid_mask_reason == "valid_mask_stop":
            valid_mask_stop_reason = "valid_mask_stop"
            break
    summary = {
        "fn": call.fn,
        "primitive_id": call.primitive_id,
        "prompt": call.prompt,
        "policy_path": policy_path,
        "policy_rollout_config": policy_rollout_config,
        "valid_mask": {
            "budget": int(primitive_steps),
            "reason": valid_mask_stop_reason,
            "decisions": valid_mask_decisions,
        },
        "steps": int(primitive_steps),
        "reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "recorded": False,
    }
    return summary, obs, total_reward, final_info, bool(terminated), bool(truncated)


def _apply_start_contract_to_env(
    *,
    env: Any,
    start_contract: str,
    seed: int,
    episode_index: int | None = None,
    object_color: str | None = None,
    start_report_path: Path | None = None,
) -> dict[str, Any]:
    contract = str(start_contract or "default_reset")
    if contract == "default_reset":
        return {"contract": contract, "applied": False, "mode": "env_reset"}
    gym_env = _gym_env_or_none(env)
    if gym_env is None:
        return {
            "contract": contract,
            "applied": False,
            "mode": "unsupported_env",
            "reason": "env has no gymnasium unwrapped simulator",
        }
    try:
        qpos, details = _dataset_backed_start_qpos(
            gym_env,
            contract=contract,
            seed=seed,
            episode_index=episode_index,
            object_color=object_color,
            start_report_path=start_report_path,
        )
        if qpos is None and contract == "full_chain_reset":
            return {"contract": contract, "applied": False, "mode": "env_reset", **details}
        if qpos is None:
            qpos, details = _fixed_jaw_edge_start_qpos(gym_env, contract=contract, seed=seed)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to apply SO101 start_contract={contract!r}: {_short_error(exc)}") from exc
    sim_snapshot = details.get("sim_snapshot") if isinstance(details, dict) else None
    if sim_snapshot:
        _restore_sim_state(gym_env, sim_snapshot)
    else:
        _set_sim_qpos(gym_env, qpos)
    observation = _current_sim_qpos(gym_env)
    details.update(
        {
            "contract": contract,
            "applied": True,
            "mode": str(details.get("mode") or "deterministic_teacher_ik_qpos"),
            "observation": observation,
        }
    )
    return details


def _gym_env_or_none(env: Any) -> Any | None:
    candidate = getattr(env, "env", env)
    if hasattr(candidate, "unwrapped") and hasattr(candidate, "action_space"):
        return candidate
    return None


def _dataset_backed_start_qpos(
    gym_env: Any,
    *,
    contract: str,
    seed: int,
    episode_index: int | None = None,
    object_color: str | None = None,
    start_report_path: Path | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    if not callable(getattr(gym_env, "reset", None)):
        return None, {}
    mapping = {
        "full_chain_reset": ("move_over_cube_edge", "validation", "move_over_cube_edge_q_start"),
        "move_over_cube_edge": ("move_over_cube_edge", "validation", "move_over_cube_edge_q_start"),
        "align_pick_reset": ("align_fixed_jaw_cube_edge", "validation", "align_fixed_jaw_q_start"),
        "align_fixed_jaw_cube_edge": ("align_fixed_jaw_cube_edge", "validation", "align_fixed_jaw_q_start"),
        "move_and_align_cube_edge": ("move_and_align_cube_edge", "validation", "move_and_align_q_start"),
        "pick_up_reset": ("grip_from_edge_cube", "validation", "grip_from_edge_q_start"),
        "grip_from_edge_cube": ("grip_from_edge_cube", "validation", "grip_from_edge_q_start"),
    }
    if contract not in mapping:
        return None, {}
    skill, split, phase = mapping[contract]
    explicit_report = start_report_path is not None
    report_path = Path(start_report_path) if start_report_path is not None else (
        Path.cwd()
        / "_workspace"
        / "hf_datasets"
        / "mhlee1215__so101-nexus-sim-dataset"
        / "datasets"
        / skill
        / split
        / "so101_lerobot_export_report.json"
    )
    dataset_split = "loop_validation" if explicit_report else split
    if not report_path.exists():
        return None, {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    indexed_episodes = [
        (index, episode)
        for index, episode in enumerate(report.get("episodes", []))
        if isinstance(episode, dict) and episode.get("q_start")
    ]
    color = str(object_color or "green").strip().lower()
    color_episodes = [
        (index, episode)
        for index, episode in indexed_episodes
        if str(episode.get("object_color") or "").strip().lower() == color
    ]
    candidates = color_episodes or indexed_episodes
    if not candidates:
        return None, {}
    if episode_index is not None:
        selection_index = int(episode_index)
        if explicit_report and selection_index >= len(candidates):
            raise ValueError(
                f"closed-loop start report {report_path} has {len(candidates)} matching episodes; "
                f"cannot replay episode_index={selection_index}"
            )
        selection_index = selection_index % len(candidates)
    else:
        selection_index = int(seed) % len(candidates)
    source_index, selected = candidates[selection_index]
    dataset_seed = selected.get("seed")
    if dataset_seed is not None:
        gym_env.reset(seed=int(dataset_seed))
    q_start = [float(value) for value in selected["q_start"]]
    return q_start, {
        "phase": phase,
        "mode": "exported_dataset_qpos",
        "source": "explicit_loop_validation_first_frame_q_start" if explicit_report else "exported_validation_dataset_q_start",
        "dataset_skill": skill,
        "dataset_split": dataset_split,
        "dataset_report": str(report_path),
        "dataset_report_explicit": explicit_report,
        "dataset_episode_seed": dataset_seed,
        "dataset_episode_index": selected.get("episode_index", source_index),
        "dataset_source_index": source_index,
        "dataset_candidate_index": selection_index,
        "dataset_candidate_count": len(candidates),
        "dataset_selection": "episode_index" if episode_index is not None else "seed_modulo",
        "dataset_object_color": selected.get("object_color"),
        "dataset_object_shape": selected.get("object_shape"),
        "dataset_task": selected.get("task"),
        "q_start": q_start,
        "sim_snapshot": selected.get("sim_snapshot"),
        "q_edge": selected.get("q_edge"),
        "q_above": selected.get("q_above"),
        "static_edge_error": _jsonable_info(selected.get("start_static_edge_error") or {}),
    }


def _start_contract_object_color(env_config: dict[str, Any]) -> str | None:
    color = env_config.get("object_color") if isinstance(env_config, dict) else None
    if color is None or str(color) == "dataset_seeded_high_contrast":
        return "green"
    return str(color)


def _fixed_jaw_edge_start_qpos(gym_env: Any, *, contract: str, seed: int) -> tuple[Any, dict[str, Any]]:
    del seed
    helpers = _fixed_jaw_export_helpers()
    candidates = helpers["_make_fast_fixed_jaw_teacher_targets"](gym_env)
    if not candidates:
        raise RuntimeError("no fixed-jaw edge IK candidates available")
    best = max(candidates, key=lambda item: float(item["meta"].get("score", -1e9)))
    meta = dict(best["meta"])
    q_edge = helpers["_make_fixed_jaw_edge_qpos"](gym_env, best["q_open"], meta)
    q_above = helpers["_make_fixed_jaw_above_qpos"](
        gym_env,
        q_edge,
        meta,
        move_target_z_offset=0.06,
    )
    low = gym_env.action_space.low
    if contract in {"align_pick_reset", "align_fixed_jaw_cube_edge"}:
        q_start = q_above.copy()
        q_start[-1] = float(low[-1])
        phase = "edge_above_closed_gripper"
    elif contract in {"pick_up_reset", "grip_from_edge_cube"}:
        q_start = q_edge.copy()
        q_start[-1] = helpers["_open_gripper_value"](gym_env)
        phase = "edge_contact_open_gripper"
    else:
        raise ValueError(f"unknown SO101 start_contract: {contract}")
    helpers["_set_qpos"](gym_env, q_start)
    edge_error = helpers["_static_finger_edge_error"](gym_env, meta)
    tcp_delta = helpers["_tcp_to_object_delta"](gym_env)
    return q_start, {
        "phase": phase,
        "source": "fixed_jaw_edge_teacher_ik",
        "q_start": [float(value) for value in q_start],
        "q_edge": [float(value) for value in q_edge],
        "q_above": [float(value) for value in q_above],
        "candidate_meta": _jsonable_info(meta),
        "static_edge_error": _jsonable_info(edge_error),
        "tcp_to_object_delta": [float(value) for value in tcp_delta],
    }


def _fixed_jaw_export_helpers() -> dict[str, Any]:
    try:
        from export_so101_teacher_rollouts_lerobot import (
            _current_qpos,
            _make_fast_fixed_jaw_teacher_targets,
            _make_fixed_jaw_above_qpos,
            _make_fixed_jaw_edge_qpos,
            _open_gripper_value,
            _set_qpos,
            _static_finger_edge_error,
            _tcp_to_object_delta,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "fixed-jaw edge start contracts require the local SO101 teacher export helpers"
        ) from exc
    return {
        "_current_qpos": _current_qpos,
        "_make_fast_fixed_jaw_teacher_targets": _make_fast_fixed_jaw_teacher_targets,
        "_make_fixed_jaw_above_qpos": _make_fixed_jaw_above_qpos,
        "_make_fixed_jaw_edge_qpos": _make_fixed_jaw_edge_qpos,
        "_open_gripper_value": _open_gripper_value,
        "_set_qpos": _set_qpos,
        "_static_finger_edge_error": _static_finger_edge_error,
        "_tcp_to_object_delta": _tcp_to_object_delta,
    }


def _set_sim_qpos(gym_env: Any, qpos: Any) -> None:
    helpers = _fixed_jaw_export_helpers()
    helpers["_set_qpos"](gym_env, qpos)


def _current_sim_qpos(gym_env: Any) -> list[float]:
    helpers = _fixed_jaw_export_helpers()
    return [float(value) for value in helpers["_current_qpos"](gym_env)]


def _snapshot_sim_state_or_none(env: Any) -> dict[str, list[float]] | None:
    gym_env = _gym_env_or_none(env)
    if gym_env is None:
        return None
    unwrapped = getattr(gym_env, "unwrapped", gym_env)
    data = getattr(unwrapped, "data", None)
    if data is None:
        return None
    snapshot: dict[str, list[float]] = {}
    for key in ("qpos", "qvel", "ctrl"):
        values = getattr(data, key, None)
        if values is not None:
            snapshot[key] = [float(value) for value in values]
    return snapshot or None


def _restore_sim_state(gym_env: Any, snapshot: dict[str, Any]) -> None:
    unwrapped = getattr(gym_env, "unwrapped", gym_env)
    data = getattr(unwrapped, "data", None)
    if data is None:
        raise RuntimeError("cannot restore simulator state: env has no data")
    for key in ("qpos", "qvel", "ctrl"):
        if key not in snapshot:
            continue
        target = getattr(data, key, None)
        if target is None:
            continue
        values = list(snapshot[key])
        if len(values) != len(target):
            raise ValueError(f"sim_snapshot.{key} length {len(values)} does not match env {key} length {len(target)}")
        target[:] = values
    try:
        import mujoco

        mujoco.mj_forward(unwrapped.model, data)
    except Exception:
        pass


def _override_policy_rollout_config(
    policy: Any,
    *,
    n_action_steps: int | None,
    num_steps: int | None,
) -> None:
    config = getattr(policy, "config", None)
    if config is None:
        return
    if n_action_steps is not None:
        if n_action_steps < 1:
            raise ValueError(f"policy_n_action_steps must be positive, got {n_action_steps}")
        chunk_size = getattr(config, "chunk_size", None)
        if chunk_size is not None and n_action_steps > int(chunk_size):
            raise ValueError(f"policy_n_action_steps={n_action_steps} exceeds chunk_size={chunk_size}")
        config.n_action_steps = int(n_action_steps)
    if num_steps is not None:
        if num_steps < 1:
            raise ValueError(f"policy_num_steps must be positive, got {num_steps}")
        config.num_steps = int(num_steps)
    if hasattr(policy, "reset"):
        policy.reset()


def _policy_rollout_config(policy: Any) -> dict[str, Any]:
    config = getattr(policy, "config", None)
    return {
        "chunk_size": getattr(config, "chunk_size", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "num_steps": getattr(config, "num_steps", None),
    }


def _merged_policy_rollout_config(policy_metadata: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    configs = [
        value.get("rollout_config")
        for value in policy_metadata.values()
        if isinstance(value.get("rollout_config"), dict)
    ]
    if not configs:
        return None
    first = dict(configs[0])
    if all(config == first for config in configs):
        return first
    return {"per_policy": configs}


def _valid_mask_primitive_budget(
    *,
    policy: Any,
    obs: Any,
    camera_pixels: dict[str, Any],
    instruction: str,
    local_files_only: bool,
    batch_builder: BatchBuilder,
    valid_mask_head: Any,
    max_horizon: int,
    threshold: float,
    consecutive: int,
) -> tuple[int, str, list[float]]:
    if not hasattr(policy, "predict_action_chunk"):
        raise ValueError("Qwen closed-loop valid-mask tests require policy.predict_action_chunk")
    batch, _mapping = batch_builder(
        policy,
        obs,
        camera_pixels=camera_pixels,
        instruction=instruction,
        local_files_only=local_files_only,
    )
    action_chunk = policy.predict_action_chunk(batch)
    state_for_head = batch.get("observation.state", obs) if isinstance(batch, dict) else obs
    valid_probs_tensor = valid_mask_head.predict_valid_probs(state_for_head, action_chunk)
    valid_probs = _valid_probs_to_float_list(
        valid_probs_tensor[0] if hasattr(valid_probs_tensor, "__getitem__") else valid_probs_tensor
    )
    horizon, reason = _execution_horizon_from_valid_probs(
        valid_probs,
        max_horizon=max_horizon,
        threshold=threshold,
        consecutive=consecutive,
    )
    if hasattr(policy, "reset"):
        policy.reset()
    return int(horizon), reason, [float(value) for value in valid_probs[: int(max_horizon)]]


def _valid_mask_requery_horizon(
    *,
    policy_rollout_config: dict[str, Any],
    policy_n_action_steps: int | None,
    remaining_budget: int,
) -> int:
    candidates = [max(1, int(remaining_budget))]
    if policy_n_action_steps is not None:
        candidates.append(max(1, int(policy_n_action_steps)))
    else:
        configured_steps = policy_rollout_config.get("n_action_steps")
        if configured_steps is not None:
            candidates.append(max(1, int(configured_steps)))
    return min(candidates)


def _execution_horizon_from_valid_probs(
    valid_probs: Any,
    *,
    max_horizon: int,
    threshold: float,
    consecutive: int,
) -> tuple[int, str]:
    probs = _valid_probs_to_float_list(valid_probs)
    horizon = max(1, int(max_horizon))
    if probs:
        horizon = min(horizon, len(probs))
    stop_index = _first_invalid_step(probs[:horizon], threshold=threshold, consecutive=consecutive)
    if stop_index is None:
        return horizon, "max_horizon"
    return max(1, min(horizon, int(stop_index))), "valid_mask_stop"


def _first_invalid_step(valid_probs: Any, *, threshold: float, consecutive: int) -> int | None:
    probs = _valid_probs_to_float_list(valid_probs)
    if not probs:
        return None
    consecutive = max(1, int(consecutive))
    invalid_run = 0
    for index, value in enumerate(probs):
        if value < float(threshold):
            invalid_run += 1
            if invalid_run >= consecutive:
                return index - consecutive + 1
        else:
            invalid_run = 0
    return None


def _valid_probs_to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_valid_probs_to_float_list(item))
            else:
                out.append(float(item))
        return out
    return [float(value)]


def _success_from_info(info: dict[str, Any]) -> bool | None:
    for key in ("success", "is_success", "task_success", "is_obj_placed"):
        if key in info:
            return bool(info[key])
    return None


def _should_render_media(config: LoopArtifactConfig, primitive_step: int) -> bool:
    return bool(config.enabled and config.render_media) and primitive_step % max(1, int(config.every_n_steps)) == 0


def _raw_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _make_renderers_or_none(env: Any, config: LoopArtifactConfig) -> dict[str, Any]:
    try:
        import mujoco

        raw_env = _raw_env(env)
        renderers = {
            "egocentric_cam": mujoco.Renderer(raw_env.unwrapped.model, height=config.height, width=config.width),
            "wrist_cam": mujoco.Renderer(raw_env.unwrapped.model, height=config.height, width=config.width),
        }
        if config.enabled and config.render_media:
            renderers["top_down"] = mujoco.Renderer(raw_env.unwrapped.model, height=config.height, width=config.width)
        return renderers
    except Exception:
        return {}


def _render_policy_cameras(env: Any, renderers: dict[str, Any]) -> dict[str, Any]:
    from physical_ai_agent.sim.so101_camera_input import _make_camera, postprocess_camera_frame

    raw_env = _raw_env(env)
    pixels = {}
    for camera_name in ("egocentric_cam", "wrist_cam"):
        renderer = renderers.get(camera_name)
        if renderer is None:
            continue
        renderer.update_scene(raw_env.unwrapped.data, camera=_make_camera(raw_env, camera_name))
        pixels[camera_name] = postprocess_camera_frame(camera_name, renderer.render())
    return pixels


def _require_policy_cameras(camera_pixels: dict[str, Any]) -> None:
    missing = [name for name in ("egocentric_cam", "wrist_cam") if name not in camera_pixels]
    if missing:
        raise RuntimeError(
            "closed-loop policy camera render failed; refusing to evaluate SmolVLA with blank "
            f"visual inputs. missing={missing}"
        )


def _render_robot_frame(env: Any, renderers: dict[str, Any]) -> Any | None:
    from physical_ai_agent.sim.so101_camera_input import _make_camera

    raw_env = _raw_env(env)
    camera_name = "top_down" if "top_down" in renderers else "egocentric_cam"
    renderer = renderers.get(camera_name)
    if renderer is None:
        return None
    renderer.update_scene(raw_env.unwrapped.data, camera=_make_camera(raw_env, camera_name))
    return renderer.render()


def _write_policy_input_images(
    *,
    camera_pixels: dict[str, Any],
    episode_media_dir: Path,
    global_step: int,
) -> dict[str, str]:
    out: dict[str, str] = {}
    input_dir = episode_media_dir / "policy_inputs"
    for camera_name, pixels in camera_pixels.items():
        path = input_dir / f"step_{global_step:04d}_{camera_name}.png"
        _write_image(path, pixels)
        out[camera_name] = str(path)
    return out


def _write_robot_frame(
    *,
    env: Any,
    renderers: dict[str, Any],
    episode_media_dir: Path,
    global_step: int,
) -> str | None:
    frame = _render_robot_frame(env, renderers)
    if frame is None:
        return None
    path = episode_media_dir / "robot_frames" / f"step_{global_step:04d}_top_down.png"
    _write_image(path, frame)
    return str(path)


def _write_primitive_videos(
    *,
    frame_paths: list[str],
    episode_media_dir: Path,
    primitive_id: str,
    iteration: int,
    fps: int,
) -> dict[str, str]:
    if not frame_paths:
        return {}
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(path) for path in frame_paths]
        video_dir = episode_media_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        stem = f"iteration_{iteration:02d}_{primitive_id}"
        gif_path = video_dir / f"{stem}.gif"
        mp4_path = video_dir / f"{stem}.mp4"
        imageio.mimsave(gif_path, frames, fps=max(1, int(fps)))
        imageio.mimsave(mp4_path, frames, fps=max(1, int(fps)))
        return {"iteration_video_gif": str(gif_path), "iteration_video_mp4": str(mp4_path)}
    except Exception:
        return {}


def _write_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(path)


def _close_renderers(renderers: dict[str, Any]) -> None:
    for renderer in renderers.values():
        try:
            renderer.close()
        except Exception:
            pass


def _action_to_float_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return _tensor_to_float_list(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
