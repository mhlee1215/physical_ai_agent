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
    env_factory: EnvFactory = SO101NexusEnv,
    policy_loader: PolicyLoader = _load_pretrained_policy,
    batch_builder: BatchBuilder = _build_batch_for_policy,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    primitive_policy_paths = primitive_policy_paths or {}
    if valid_mask_head is None:
        if valid_mask_checkpoint is None:
            raise ValueError("Qwen closed-loop tests require valid_mask_checkpoint")
        from physical_ai_agent.policies.so101_valid_mask import load_valid_mask_head

        valid_mask_head = load_valid_mask_head(valid_mask_checkpoint, device=None if device == "auto" else device)
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
                    policy_n_action_steps=policy_n_action_steps,
                    policy_num_steps=policy_num_steps,
                    valid_mask_head=valid_mask_head,
                    valid_mask_threshold=valid_mask_threshold,
                    valid_mask_consecutive=valid_mask_consecutive,
                    artifact_config=artifact_config or LoopArtifactConfig(),
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
        "valid_mask": {
            "checkpoint": str(valid_mask_checkpoint) if valid_mask_checkpoint else None,
            "threshold": float(valid_mask_threshold),
            "consecutive": int(valid_mask_consecutive),
            "required_for_loop_test": True,
        },
        "policy_metadata": policy_metadata,
        "policy_rollout_config": _merged_policy_rollout_config(policy_metadata),
        "loop_artifact_config": asdict(artifact_config or LoopArtifactConfig()),
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
    policy_n_action_steps: int | None,
    policy_num_steps: int | None,
    valid_mask_head: Any,
    valid_mask_threshold: float,
    valid_mask_consecutive: int,
    artifact_config: LoopArtifactConfig,
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
            step_budget = (
                min(int(call.max_steps), int(max_steps_per_primitive))
                if max_steps_per_primitive is not None
                else int(call.max_steps)
            )
            valid_mask_budget, valid_mask_reason, valid_mask_probs = _valid_mask_primitive_budget(
                policy=policy,
                obs=obs,
                camera_pixels={},
                instruction=call.prompt,
                local_files_only=local_files_only,
                batch_builder=batch_builder,
                valid_mask_head=valid_mask_head,
                max_horizon=step_budget,
                threshold=valid_mask_threshold,
                consecutive=valid_mask_consecutive,
            )
            step_budget = min(step_budget, valid_mask_budget)
            for primitive_step in range(step_budget):
                record_media = _should_render_media(artifact_config, primitive_step)
                camera_pixels = _render_policy_cameras(env, renderers) if renderers else {}
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
                raw_action = policy.select_action(batch)
                action = _clip_action(_action_to_float_list(raw_action), action_dim)
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
                    "valid_mask": {
                        "budget": int(step_budget),
                        "reason": valid_mask_reason,
                        "probs": valid_mask_probs,
                    },
                    "observation": obs,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _jsonable_info(info),
                    "image_feature_mapping": image_feature_mapping,
                    "media": {
                        "policy_input_images": policy_input_images,
                        "robot_frame": robot_frame_path,
                        "render_mode": "inline" if record_media else "deferred",
                    },
                    "render_replay": {
                        "env_id": env_id,
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
                if terminated or truncated:
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
                        "budget": int(step_budget),
                        "reason": valid_mask_reason,
                        "probs": valid_mask_probs,
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


def _execution_horizon_from_valid_probs(
    valid_probs: Any,
    *,
    max_horizon: int,
    threshold: float,
    consecutive: int,
) -> tuple[int, str]:
    horizon = max(1, int(max_horizon))
    stop_index = _first_invalid_step(valid_probs, threshold=threshold, consecutive=consecutive)
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
    if not (config.enabled and config.render_media):
        return {}
    try:
        import mujoco

        raw_env = _raw_env(env)
        return {
            "egocentric_cam": mujoco.Renderer(raw_env.unwrapped.model, height=config.height, width=config.width),
            "wrist_cam": mujoco.Renderer(raw_env.unwrapped.model, height=config.height, width=config.width),
            "top_down": mujoco.Renderer(raw_env.unwrapped.model, height=config.height, width=config.width),
        }
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
