from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID, probe_smolvla


DEFAULT_MANISKILL_ENV_ID = "PickCube-v1"
DEFAULT_MANISKILL_FALLBACK_ENV_IDS = ("Empty-v1",)
DEFAULT_MANISKILL_POLICIES = ("random",)
SUPPORTED_MANISKILL_POLICIES = ("random", "zero", "smolvla_dry", "smolvla_real")


@dataclass(frozen=True)
class ManiSkillRolloutResult:
    status: str
    requested_env_id: str
    env_id: str
    attempted_env_ids: list[str]
    env_blockers: dict[str, str]
    episodes: int
    steps: int
    success_count: int
    success: bool
    blocker: str | None
    trace_path: str
    metrics_path: str
    summary_path: str


@dataclass(frozen=True)
class Checkpoint24Report:
    checkpoint: str
    status: str
    python: str
    platform: str
    output_dir: str
    duration_s: float
    checks: dict[str, bool]
    metrics: dict[str, object]
    artifacts: dict[str, str]


def run_checkpoint(
    output_dir: Path,
    env_id: str = DEFAULT_MANISKILL_ENV_ID,
    fallback_env_ids: tuple[str, ...] = DEFAULT_MANISKILL_FALLBACK_ENV_IDS,
    episodes: int = 1,
    steps: int = 8,
    policies: tuple[str, ...] = DEFAULT_MANISKILL_POLICIES,
    model_id: str = DEFAULT_SMOLVLA_MODEL_ID,
    require_maniskill: bool = False,
    allow_download: bool = False,
    real_images: bool = False,
) -> Checkpoint24Report:
    started_at = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout = _run_maniskill_rollout(
        output_dir=output_dir / "maniskill_rollout",
        env_id=env_id,
        fallback_env_ids=fallback_env_ids,
        episodes=episodes,
        steps=steps,
        policies=policies,
        model_id=model_id,
        allow_download=allow_download,
        real_images=real_images,
    )
    smolvla_probe = probe_smolvla(model_id)
    plan_path = output_dir / "smolvla_maniskill_eval_plan.md"
    blocker_path = output_dir / "maniskill_blocker.md"
    robocasa_plan_path = output_dir / "checkpoint_25_robocasa_plan.md"
    _write_smolvla_maniskill_plan(
        path=plan_path,
        env_id=env_id,
        model_id=model_id,
        smolvla_ready=smolvla_probe.ready,
        smolvla_blockers=smolvla_probe.blockers,
    )
    _write_maniskill_blocker(blocker_path, rollout)
    _write_robocasa_checkpoint_plan(robocasa_plan_path)

    rollout_ok = rollout.status == "passed"
    checks = {
        "cp24_maniskill_research_checkpoint_registered": True,
        "cp24_maniskill_import_or_blocker_documented": rollout_ok or bool(rollout.blocker),
        "cp24_rollout_trace_or_blocker_saved": Path(rollout.trace_path).exists()
        or Path(blocker_path).stat().st_size > 0,
        "cp24_metrics_or_blocker_saved": Path(rollout.metrics_path).exists()
        or Path(blocker_path).stat().st_size > 0,
        "cp24_smolvla_eval_pipeline_plan_saved": plan_path.exists() and plan_path.stat().st_size > 0,
        "cp24_robocasa_checkpoint_registered": robocasa_plan_path.exists()
        and "checkpoint_25" in robocasa_plan_path.read_text(encoding="utf-8"),
    }
    if require_maniskill:
        checks["cp24_require_real_maniskill_rollout"] = rollout_ok

    report_path = output_dir / "checkpoint_report.json"
    artifacts = {
        "maniskill_trace": rollout.trace_path,
        "maniskill_metrics": rollout.metrics_path,
        "maniskill_summary": rollout.summary_path,
        "maniskill_blocker": str(blocker_path),
        "smolvla_maniskill_eval_plan": str(plan_path),
        "robocasa_checkpoint_plan": str(robocasa_plan_path),
        "checkpoint_report": str(report_path),
    }
    metrics = {
        "env_id": env_id,
        "executed_env_id": rollout.env_id,
        "attempted_env_ids": rollout.attempted_env_ids,
        "env_blockers": rollout.env_blockers,
        "episodes_requested": episodes,
        "rollout_episodes": rollout.episodes,
        "steps_requested": steps,
        "policies_requested": list(policies),
        "rollout_status": rollout.status,
        "rollout_steps": rollout.steps,
        "rollout_success_count": rollout.success_count,
        "rollout_success": rollout.success,
        "model_id": model_id,
        "smolvla_ready": smolvla_probe.ready,
        "smolvla_blockers": smolvla_probe.blockers,
        "require_maniskill": require_maniskill,
        "allow_download": allow_download,
        "real_images": real_images,
    }
    status = "passed" if all(checks.values()) else "failed"
    report = Checkpoint24Report(
        checkpoint="checkpoint_24_maniskill_hab_smolvla_eval_planning",
        status=status,
        python=platform.python_version(),
        platform=platform.platform(),
        output_dir=str(output_dir),
        duration_s=round(perf_counter() - started_at, 4),
        checks=checks,
        metrics=metrics,
        artifacts=artifacts,
    )
    report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_maniskill_rollout(
    output_dir: Path,
    env_id: str,
    fallback_env_ids: tuple[str, ...],
    episodes: int,
    steps: int,
    policies: tuple[str, ...],
    model_id: str,
    allow_download: bool,
    real_images: bool,
) -> ManiSkillRolloutResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "episodes.jsonl"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    smolvla_bridge_path = output_dir / "smolvla_dry_bridge_manifest.json"
    smolvla_real_path = output_dir / "smolvla_real_manifest.json"
    smolvla_real_image_path = output_dir / "smolvla_real_input.png"
    smolvla_real_frames_dir = output_dir / "smolvla_real_frames"
    smolvla_real_gif_path = output_dir / "smolvla_real_rollout.gif"
    rollout_frames_dir = output_dir / "rollout_frames"
    rollout_gifs_dir = output_dir / "rollout_gifs"
    for path in (
        trace_path,
        metrics_path,
        summary_path,
        smolvla_bridge_path,
        smolvla_real_path,
        smolvla_real_image_path,
        smolvla_real_gif_path,
    ):
        if path.exists():
            path.unlink()
    for directory in (smolvla_real_frames_dir, rollout_frames_dir, rollout_gifs_dir):
        if directory.exists():
            shutil.rmtree(directory)
    if real_images:
        smolvla_real_frames_dir.mkdir(parents=True, exist_ok=True)
        rollout_frames_dir.mkdir(parents=True, exist_ok=True)
        rollout_gifs_dir.mkdir(parents=True, exist_ok=True)

    attempted_env_ids = [env_id, *[candidate for candidate in fallback_env_ids if candidate != env_id]]
    env_blockers: dict[str, str] = {}
    env = None
    executed_env_id = env_id
    try:
        gymnasium = importlib.import_module("gymnasium")
        importlib.import_module("mani_skill")
        _patch_maniskill_headless_backend()
        for candidate_env_id in attempted_env_ids:
            try:
                env = _make_env(gymnasium, candidate_env_id, real_images=real_images)
                executed_env_id = candidate_env_id
                break
            except Exception as exc:  # noqa: BLE001
                env_blockers[candidate_env_id] = _short_error(exc)
        if env is None:
            raise RuntimeError("; ".join(env_blockers.values()))
    except Exception as exc:  # noqa: BLE001
        blocker = _short_error(exc)
        return ManiSkillRolloutResult(
            status="blocked",
            requested_env_id=env_id,
            env_id=executed_env_id,
            attempted_env_ids=attempted_env_ids,
            env_blockers=env_blockers,
            episodes=0,
            steps=0,
            success_count=0,
            success=False,
            blocker=blocker,
            trace_path=str(trace_path),
            metrics_path=str(metrics_path),
            summary_path=str(summary_path),
        )

    unsupported_policies = sorted(set(policies) - set(SUPPORTED_MANISKILL_POLICIES))
    if unsupported_policies:
        return ManiSkillRolloutResult(
            status="blocked",
            requested_env_id=env_id,
            env_id=executed_env_id,
            attempted_env_ids=attempted_env_ids,
            env_blockers=env_blockers,
            episodes=0,
            steps=0,
            success_count=0,
            success=False,
            blocker=f"Unsupported policies: {', '.join(unsupported_policies)}",
            trace_path=str(trace_path),
            metrics_path=str(metrics_path),
            summary_path=str(summary_path),
        )

    policy_runtime: dict[str, Any] = {}
    if "smolvla_real" in policies:
        try:
            policy_runtime["smolvla_real"] = _load_smolvla_real_policy(
                model_id=model_id,
                local_files_only=not allow_download,
            )
            policy_runtime["smolvla_real_use_real_images"] = real_images
        except Exception as exc:  # noqa: BLE001
            close = getattr(env, "close", None)
            if callable(close):
                close()
            blocker = _short_error(exc)
            smolvla_real_path.write_text(
                json.dumps(
                    {
                        "policy": "smolvla_real",
                        "model_id": model_id,
                        "local_files_only": not allow_download,
                        "status": "blocked",
                        "blocker": blocker,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return ManiSkillRolloutResult(
                status="blocked",
                requested_env_id=env_id,
                env_id=executed_env_id,
                attempted_env_ids=attempted_env_ids,
                env_blockers=env_blockers,
                episodes=0,
                steps=0,
                success_count=0,
                success=False,
                blocker=blocker,
                trace_path=str(trace_path),
                metrics_path=str(metrics_path),
                summary_path=str(summary_path),
            )

    records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    rollout_frame_paths: list[str] = []
    success_count = 0
    try:
        for policy in policies:
            for episode in range(episodes):
                reset_result = env.reset(seed=episode)
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                episode_frame_paths: list[str] = []
                episode_gif_path = rollout_gifs_dir / f"{policy}_episode_{episode:03d}.gif"
                if policy == "smolvla_real" and real_images and not smolvla_real_image_path.exists():
                    _write_observation_image(obs, smolvla_real_image_path)
                if real_images:
                    frame_path = _write_rollout_frame(
                        obs,
                        rollout_frames_dir,
                        policy=policy,
                        episode=episode,
                        step=0,
                        phase="reset",
                    )
                    episode_frame_paths.append(str(frame_path))
                    if policy == "smolvla_real":
                        legacy_frame_path = _write_rollout_frame(
                            obs,
                            smolvla_real_frames_dir,
                            policy=policy,
                            episode=episode,
                            step=0,
                            phase="reset",
                        )
                        rollout_frame_paths.append(str(legacy_frame_path))
                episode_success = False
                episode_reward = 0.0
                episode_steps = 0
                for step in range(steps):
                    action, policy_metadata = _policy_action(
                        env,
                        obs=obs,
                        policy=policy,
                        seed=episode * steps + step,
                        runtime=policy_runtime,
                    )
                    step_result = env.step(action)
                    obs, reward, terminated, truncated, info = _normalize_step(step_result)
                    if real_images:
                        frame_path = _write_rollout_frame(
                            obs,
                            rollout_frames_dir,
                            policy=policy,
                            episode=episode,
                            step=step + 1,
                            phase="step",
                        )
                        episode_frame_paths.append(str(frame_path))
                        if policy == "smolvla_real":
                            legacy_frame_path = _write_rollout_frame(
                                obs,
                                smolvla_real_frames_dir,
                                policy=policy,
                                episode=episode,
                                step=step + 1,
                                phase="step",
                            )
                            rollout_frame_paths.append(str(legacy_frame_path))
                    step_success = _info_success(info)
                    episode_success = episode_success or step_success
                    episode_reward += _float_value(reward)
                    episode_steps += 1
                    records.append(
                        {
                            "policy": policy,
                            "episode": episode,
                            "step": step,
                            "reward": _jsonable_scalar(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "success": step_success,
                            "observation_summary": _summarize_observation(obs),
                            "action_summary": _summarize_action(action),
                            "policy_metadata": policy_metadata,
                        }
                    )
                    if terminated or truncated:
                        break
                if episode_success:
                    success_count += 1
                if episode_frame_paths:
                    _write_rollout_gif([Path(path) for path in episode_frame_paths], episode_gif_path)
                episode_summaries.append(
                    {
                        "policy": policy,
                        "episode": episode,
                        "steps": episode_steps,
                        "success": episode_success,
                        "reward_sum": round(episode_reward, 6),
                        "rollout_gif": str(episode_gif_path) if episode_gif_path.exists() else None,
                        "rollout_frames": episode_frame_paths,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        return ManiSkillRolloutResult(
            status="blocked",
            requested_env_id=env_id,
            env_id=executed_env_id,
            attempted_env_ids=attempted_env_ids,
            env_blockers=env_blockers,
            episodes=len(episode_summaries),
            steps=len(records),
            success_count=success_count,
            success=success_count > 0,
            blocker=_short_error(exc),
            trace_path=str(trace_path),
            metrics_path=str(metrics_path),
            summary_path=str(summary_path),
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    if "smolvla_dry" in policies:
        _write_smolvla_dry_bridge_manifest(smolvla_bridge_path, env, records)
    if "smolvla_real" in policies:
        if rollout_frame_paths:
            _write_rollout_gif([Path(path) for path in rollout_frame_paths], smolvla_real_gif_path)
        _write_smolvla_real_manifest(
            smolvla_real_path,
            policy_runtime,
            env,
            records,
            model_id,
            not allow_download,
            str(smolvla_real_image_path) if smolvla_real_image_path.exists() else None,
            str(smolvla_real_gif_path) if smolvla_real_gif_path.exists() else None,
            rollout_frame_paths,
        )

    policy_metrics = _policy_metrics(episode_summaries)
    metrics = {
        "requested_env_id": env_id,
        "env_id": executed_env_id,
        "attempted_env_ids": attempted_env_ids,
        "env_blockers": env_blockers,
        "policies": list(policies),
        "episodes_per_policy": episodes,
        "episodes": len(episode_summaries),
        "episode_summaries": episode_summaries,
        "steps": len(records),
        "success_count": success_count,
        "success": success_count > 0,
        "success_rate": success_count / len(episode_summaries) if episode_summaries else 0.0,
        "mean_reward_sum": (
            round(
                sum(float(summary["reward_sum"]) for summary in episode_summaries)
                / len(episode_summaries),
                6,
            )
            if episode_summaries
            else 0.0
        ),
        "mean_episode_steps": (
            round(
                sum(int(summary["steps"]) for summary in episode_summaries)
                / len(episode_summaries),
                6,
            )
            if episode_summaries
            else 0.0
        ),
        "benchmark_family": "ManiSkill / ManiSkill-HAB",
        "policy_metrics": policy_metrics,
        "smolvla_dry_bridge_manifest": str(smolvla_bridge_path) if "smolvla_dry" in policies else None,
        "smolvla_real_manifest": str(smolvla_real_path) if "smolvla_real" in policies else None,
        "smolvla_real_input_image": (
            str(smolvla_real_image_path)
            if "smolvla_real" in policies and smolvla_real_image_path.exists()
            else None
        ),
        "smolvla_real_rollout_gif": (
            str(smolvla_real_gif_path)
            if "smolvla_real" in policies and smolvla_real_gif_path.exists()
            else None
        ),
        "smolvla_real_rollout_frames": rollout_frame_paths if "smolvla_real" in policies else [],
        "rollout_gifs_dir": str(rollout_gifs_dir) if real_images else None,
        "rollout_frames_dir": str(rollout_frames_dir) if real_images else None,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary_policy_lines = [
        "| Policy | Episodes | Success | Success rate | Mean reward sum | Mean steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in policies:
        values = policy_metrics[policy]
        summary_policy_lines.append(
            "| "
            f"{policy} | "
            f"{values['episodes']} | "
            f"{values['success_count']} | "
            f"{values['success_rate']} | "
            f"{values['mean_reward_sum']} | "
            f"{values['mean_episode_steps']} |"
        )
    summary_path.write_text(
        "\n".join(
            [
                "# Checkpoint 24 ManiSkill Evaluation",
                "",
                f"- Requested env id: `{env_id}`",
                f"- Executed env id: `{executed_env_id}`",
                f"- Policies: `{', '.join(policies)}`",
                f"- Episodes per policy: `{episodes}`",
                f"- Total episodes: `{len(episode_summaries)}`",
                f"- Steps: `{len(records)}`",
                f"- Success count: `{success_count}`",
                f"- Success rate: `{metrics['success_rate']}`",
                "- Mode: baseline policy evaluation for research-benchmark integration",
                "",
                "## Baseline Metrics",
                "",
                *summary_policy_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return ManiSkillRolloutResult(
        status="passed",
        requested_env_id=env_id,
        env_id=executed_env_id,
        attempted_env_ids=attempted_env_ids,
        env_blockers=env_blockers,
        episodes=len(episode_summaries),
        steps=len(records),
        success_count=success_count,
        success=success_count > 0,
        blocker=None,
        trace_path=str(trace_path),
        metrics_path=str(metrics_path),
        summary_path=str(summary_path),
    )


def _make_env(gymnasium: Any, env_id: str, real_images: bool = False) -> Any:
    if real_images:
        candidates = [
            {"obs_mode": "rgb", "control_mode": "pd_joint_delta_pos", "render_mode": "rgb_array"},
            {"obs_mode": "rgb", "control_mode": "pd_joint_delta_pos"},
        ]
    else:
        candidates = [
            {
                "obs_mode": "state",
                "control_mode": "pd_joint_delta_pos",
                "render_backend": "none",
            },
            {
                "obs_mode": "state_dict",
                "control_mode": "pd_joint_delta_pos",
                "render_backend": "none",
            },
            {"obs_mode": "state", "control_mode": "pd_joint_delta_pos"},
            {"obs_mode": "state_dict", "control_mode": "pd_joint_delta_pos"},
            {"obs_mode": "state", "control_mode": "pd_joint_delta_pos", "render_mode": "rgb_array"},
            {"obs_mode": "state_dict", "control_mode": "pd_joint_delta_pos", "render_mode": "rgb_array"},
            {"obs_mode": "state"},
            {},
        ]
    errors: list[str] = []
    for kwargs in candidates:
        try:
            return gymnasium.make(env_id, **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(_short_error(exc))
    raise RuntimeError("; ".join(errors[-2:]))


def _patch_maniskill_headless_backend() -> None:
    """Allow render_backend='none' to stay renderer-free on macOS.

    ManiSkill 3.0.1 documents render_backend='none', but its macOS branch currently
    forces a SAPIEN CPU render device before that option is checked. In headless
    Mac agent sessions this can fail at Vulkan instance creation even for pure
    state-observation rollouts. Keep the patch scoped to the no-render path.
    """

    try:
        backend = importlib.import_module("mani_skill.envs.utils.system.backend")
        sapien_env = importlib.import_module("mani_skill.envs.sapien_env")
        render_utils = importlib.import_module("mani_skill.render.utils")
    except Exception:  # noqa: BLE001
        return

    original = getattr(backend, "parse_sim_and_render_backend", None)
    if original is None or getattr(original, "_physical_ai_headless_patch", False):
        return

    def parse_without_macos_render_force(sim_backend: str, render_backend: str) -> Any:
        if render_backend not in ("none", None):
            return original(sim_backend, render_backend)
        original_system = backend.platform.system
        backend.platform.system = lambda: "Linux"
        try:
            return original(sim_backend, render_backend)
        finally:
            backend.platform.system = original_system

    parse_without_macos_render_force._physical_ai_headless_patch = True  # type: ignore[attr-defined]
    backend.parse_sim_and_render_backend = parse_without_macos_render_force
    sapien_env.parse_sim_and_render_backend = parse_without_macos_render_force

    original_can_render = getattr(render_utils, "can_render", None)
    if original_can_render is None or getattr(original_can_render, "_physical_ai_headless_patch", False):
        return

    def can_render_without_none_device(device: Any) -> bool:
        if device is None:
            return False
        return original_can_render(device)

    can_render_without_none_device._physical_ai_headless_patch = True  # type: ignore[attr-defined]
    render_utils.can_render = can_render_without_none_device


def _policy_action(
    env: Any,
    obs: Any,
    policy: str,
    seed: int,
    runtime: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if policy == "random":
        return _sample_action(env, seed=seed), {"bridge": "action_space.sample"}
    if policy == "zero":
        return _zero_action(env), {"bridge": "zero_action"}
    if policy == "smolvla_dry":
        return _smolvla_dry_action(env, obs)
    if policy == "smolvla_real":
        return _smolvla_real_action(
            env,
            obs,
            runtime["smolvla_real"],
            use_real_images=bool(runtime.get("smolvla_real_use_real_images")),
        )
    raise ValueError(f"Unsupported policy: {policy}")


def _sample_action(env: Any, seed: int) -> Any:
    action_space = getattr(env, "action_space")
    seed_fn = getattr(action_space, "seed", None)
    if callable(seed_fn):
        seed_fn(seed)
    return action_space.sample()


def _zero_action(env: Any) -> Any:
    action_space = getattr(env, "action_space")
    shape = getattr(action_space, "shape", None)
    if shape is None:
        return 0
    try:
        numpy = importlib.import_module("numpy")
        action = numpy.zeros(shape, dtype=getattr(action_space, "dtype", numpy.float32))
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is not None and high is not None:
            action = numpy.clip(action, low, high)
        return action
    except Exception:  # noqa: BLE001
        return _sample_action(env, seed=0)


def _smolvla_dry_action(env: Any, obs: Any) -> tuple[Any, dict[str, Any]]:
    """Map ManiSkill observations through a SmolVLA-shaped dry bridge.

    This does not run model weights. It verifies the local evaluation contract:
    observation -> bounded numeric policy features -> ManiSkill action-space action.
    """

    numeric_features = _flatten_numeric_observation(obs, limit=64)
    feature_mean = (
        sum(numeric_features) / len(numeric_features)
        if numeric_features
        else 0.0
    )
    action = _bounded_dry_action(env, feature_mean)
    return action, {
        "bridge": "smolvla_dry",
        "model_id": DEFAULT_SMOLVLA_MODEL_ID,
        "feature_keys": [
            "observation.state",
            "observation.images.camera1",
            "task",
        ],
        "state_dim": len(numeric_features),
        "image_shape": [3, 512, 512],
        "instruction": "Perform the ManiSkill task.",
        "note": "Dry bridge only; no SmolVLA model weights were executed.",
    }


def _bounded_dry_action(env: Any, feature_mean: float) -> Any:
    action_space = getattr(env, "action_space")
    shape = getattr(action_space, "shape", None)
    if shape is None:
        return 0
    try:
        numpy = importlib.import_module("numpy")
        size = int(numpy.prod(shape))
        base = numpy.linspace(-1.0, 1.0, num=size, dtype=numpy.float32).reshape(shape)
        action = 0.05 * numpy.tanh(base + float(feature_mean))
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is not None and high is not None:
            action = numpy.clip(action, low, high)
        return action.astype(getattr(action_space, "dtype", numpy.float32), copy=False)
    except Exception:  # noqa: BLE001
        return _zero_action(env)


def _flatten_numeric_observation(obs: Any, limit: int) -> list[float]:
    values: list[float] = []

    def visit(value: Any) -> None:
        if len(values) >= limit:
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key])
                if len(values) >= limit:
                    return
            return
        try:
            numpy = importlib.import_module("numpy")
            array = numpy.asarray(value, dtype=numpy.float32).reshape(-1)
            for item in array[: max(0, limit - len(values))]:
                values.append(float(item))
            return
        except Exception:  # noqa: BLE001
            pass
        if isinstance(value, (int, float)):
            values.append(float(value))

    visit(obs)
    return values


def _write_smolvla_dry_bridge_manifest(path: Path, env: Any, records: list[dict[str, Any]]) -> None:
    smolvla_records = [
        record for record in records if record.get("policy") == "smolvla_dry"
    ]
    first_metadata = (
        smolvla_records[0].get("policy_metadata", {})
        if smolvla_records
        else {}
    )
    action_space = getattr(env, "action_space", None)
    path.write_text(
        json.dumps(
            {
                "policy": "smolvla_dry",
                "model_id": DEFAULT_SMOLVLA_MODEL_ID,
                "feature_keys": first_metadata.get("feature_keys", []),
                "state_dim": first_metadata.get("state_dim", 0),
                "image_shape": first_metadata.get("image_shape", [3, 512, 512]),
                "instruction": first_metadata.get("instruction", "Perform the ManiSkill task."),
                "action_space": _shape_or_type(action_space),
                "steps": len(smolvla_records),
                "note": "Dry bridge validates observation/action mapping only; no model weights were executed.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_smolvla_real_policy(model_id: str, local_files_only: bool) -> Any:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return SmolVLAPolicy.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        map_location="cpu",
        device="cpu",
    )


def _smolvla_real_action(
    env: Any,
    obs: Any,
    policy: Any,
    use_real_images: bool,
) -> tuple[Any, dict[str, Any]]:
    batch, metadata = _build_maniskill_smolvla_batch(policy, obs, use_real_images=use_real_images)
    raw_action = policy.select_action(batch)
    flat_action = _tensor_to_float_list(raw_action)
    action = _clip_to_action_space(env, flat_action)
    metadata.update(
        {
            "bridge": "smolvla_real",
            "raw_action_dim": len(flat_action),
            "executed_action_summary": _summarize_action(action),
            "note": "Pretrained SmolVLA select_action executed on a ManiSkill-shaped batch.",
        }
    )
    return action, metadata


def _build_maniskill_smolvla_batch(
    policy: Any,
    obs: Any,
    use_real_images: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    config = policy.config
    numeric_features = _flatten_numeric_observation(obs, limit=256)
    state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else min(32, len(numeric_features))
    state = torch.zeros(1, state_dim, dtype=torch.float32)
    if numeric_features:
        source = torch.tensor(numeric_features[:state_dim], dtype=torch.float32)
        state[0, : len(source)] = source
    batch: dict[str, Any] = {
        OBS_STATE: state,
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
    }
    camera_pixels = _extract_rgb_images(obs) if use_real_images else {}
    if use_real_images and not camera_pixels:
        raise RuntimeError("No ManiSkill RGB camera observations found for --real-images")
    image_feature_mapping: dict[str, str] = {}
    for index, (key, feature) in enumerate(config.image_features.items()):
        source_name = _source_maniskill_camera_name(index, camera_pixels)
        image_feature_mapping[key] = source_name
        if source_name in camera_pixels:
            batch[key] = _pixels_to_tensor(camera_pixels[source_name], feature.shape)
        else:
            batch[key] = torch.zeros(1, *feature.shape, dtype=torch.float32)
    return (
        {key: value.to(config.device) if hasattr(value, "to") else value for key, value in batch.items()},
        {
            "feature_keys": sorted(str(key) for key in batch.keys()),
            "state_dim": state_dim,
            "image_feature_mapping": image_feature_mapping,
            "language_tokens": 4,
            "real_images": bool(camera_pixels),
            "camera_sources": sorted(camera_pixels),
        },
    )


def _source_maniskill_camera_name(index: int, camera_pixels: dict[str, Any]) -> str:
    if not camera_pixels:
        return "zero_camera" if index == 0 else f"zero_camera_{index + 1}"
    preferred = ["base_camera", "base_camera", "base_camera"]
    if index < len(preferred) and preferred[index] in camera_pixels:
        return preferred[index]
    return next(iter(camera_pixels))


def _extract_rgb_images(obs: Any) -> dict[str, Any]:
    images: dict[str, Any] = {}
    if not isinstance(obs, dict):
        return images
    sensor_data = obs.get("sensor_data", {})
    if not isinstance(sensor_data, dict):
        return images
    for camera_name, camera_data in sorted(sensor_data.items(), key=lambda item: str(item[0])):
        if not isinstance(camera_data, dict) or "rgb" not in camera_data:
            continue
        try:
            images[str(camera_name)] = _rgb_tensor_to_hwc_uint8(camera_data["rgb"])
        except Exception:  # noqa: BLE001
            continue
    return images


def _rgb_tensor_to_hwc_uint8(value: Any) -> Any:
    numpy = importlib.import_module("numpy")
    try:
        import torch

        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = numpy.asarray(value)
    except Exception:  # noqa: BLE001
        array = numpy.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = numpy.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with shape HxWxC, got {array.shape}")
    array = array[..., :3]
    if array.dtype != numpy.uint8:
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = numpy.clip(array, 0, 255).astype(numpy.uint8)
    return array


def _pixels_to_tensor(pixels: Any, feature_shape: tuple[int, ...]) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    channels, height, width = feature_shape
    if channels != 3:
        raise ValueError(f"Expected RGB image feature with 3 channels, got {feature_shape}")
    image = Image.fromarray(pixels).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _write_observation_image(obs: Any, path: Path) -> None:
    from PIL import Image

    images = _extract_rgb_images(obs)
    if not images:
        raise RuntimeError("No ManiSkill RGB camera observations found to save")
    first_pixels = images[next(iter(images))]
    Image.fromarray(first_pixels).save(path)


def _write_rollout_frame(
    obs: Any,
    frames_dir: Path,
    policy: str,
    episode: int,
    step: int,
    phase: str,
) -> Path:
    from PIL import Image

    images = _extract_rgb_images(obs)
    if not images:
        raise RuntimeError("No ManiSkill RGB camera observations found to save rollout frame")
    first_pixels = images[next(iter(images))]
    frame_path = frames_dir / f"{policy}_episode_{episode:03d}_{phase}_{step:04d}.png"
    Image.fromarray(first_pixels).save(frame_path)
    return frame_path


def _write_rollout_gif(frame_paths: list[Path], gif_path: Path) -> None:
    from PIL import Image

    if not frame_paths:
        return
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    try:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    finally:
        for frame in frames:
            frame.close()


def _tensor_to_float_list(value: Any) -> list[float]:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        pass
    try:
        numpy = importlib.import_module("numpy")
        return [float(item) for item in numpy.asarray(value, dtype=numpy.float32).reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        return [float(item) for item in value]


def _clip_to_action_space(env: Any, action: list[float]) -> Any:
    action_space = getattr(env, "action_space")
    shape = getattr(action_space, "shape", None)
    if shape is None:
        return action[0] if action else 0
    numpy = importlib.import_module("numpy")
    size = int(numpy.prod(shape))
    if len(action) >= size:
        values = action[:size]
    else:
        values = action + [0.0 for _ in range(size - len(action))]
    array = numpy.asarray(values, dtype=getattr(action_space, "dtype", numpy.float32)).reshape(shape)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is not None and high is not None:
        array = numpy.clip(array, low, high)
    return array


def _write_smolvla_real_manifest(
    path: Path,
    runtime: dict[str, Any],
    env: Any,
    records: list[dict[str, Any]],
    model_id: str,
    local_files_only: bool,
    input_image_path: str | None,
    rollout_gif_path: str | None,
    rollout_frame_paths: list[str],
) -> None:
    smolvla_records = [
        record for record in records if record.get("policy") == "smolvla_real"
    ]
    first_metadata = (
        smolvla_records[0].get("policy_metadata", {})
        if smolvla_records
        else {}
    )
    path.write_text(
        json.dumps(
            {
                "policy": "smolvla_real",
                "status": "passed" if smolvla_records else "no_steps",
                "model_id": model_id,
                "local_files_only": local_files_only,
                "feature_keys": first_metadata.get("feature_keys", []),
                "state_dim": first_metadata.get("state_dim", 0),
                "image_feature_mapping": first_metadata.get("image_feature_mapping", {}),
                "camera_sources": first_metadata.get("camera_sources", []),
                "real_images": first_metadata.get("real_images", False),
                "raw_action_dim": first_metadata.get("raw_action_dim", 0),
                "action_space": _shape_or_type(getattr(env, "action_space", None)),
                "steps": len(smolvla_records),
                "loaded": "smolvla_real" in runtime,
                "input_image": input_image_path,
                "rollout_gif": rollout_gif_path,
                "rollout_frames": rollout_frame_paths,
                "note": (
                    "Pretrained SmolVLA was loaded and select_action() executed with real ManiSkill RGB camera observations."
                    if first_metadata.get("real_images", False)
                    else "Pretrained SmolVLA was loaded and select_action() executed; zero image tensors are still used for this minimal ManiSkill probe."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _policy_metrics(episode_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in episode_summaries:
        grouped.setdefault(str(summary["policy"]), []).append(summary)
    metrics: dict[str, dict[str, Any]] = {}
    for policy, summaries in sorted(grouped.items()):
        success_count = sum(1 for summary in summaries if bool(summary["success"]))
        metrics[policy] = {
            "episodes": len(summaries),
            "success_count": success_count,
            "success_rate": round(success_count / len(summaries), 6) if summaries else 0.0,
            "mean_reward_sum": (
                round(sum(float(summary["reward_sum"]) for summary in summaries) / len(summaries), 6)
                if summaries
                else 0.0
            ),
            "mean_episode_steps": (
                round(sum(int(summary["steps"]) for summary in summaries) / len(summaries), 6)
                if summaries
                else 0.0
            ),
        }
    return metrics


def _normalize_step(step_result: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated), bool(truncated), dict(info)
    obs, reward, done, info = step_result
    return obs, reward, bool(done), False, dict(info)


def _info_success(info: dict[str, Any]) -> bool:
    for key in ("success", "is_success"):
        if key in info:
            return bool(_jsonable_scalar(info[key]))
    return False


def _summarize_observation(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return {
            str(key): _shape_or_type(value)
            for key, value in sorted(obs.items(), key=lambda item: str(item[0]))
        }
    return {"observation": _shape_or_type(obs)}


def _summarize_action(action: Any) -> dict[str, Any]:
    return {"action": _shape_or_type(action)}


def _shape_or_type(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": [int(dim) for dim in shape], "type": type(value).__name__}
    if isinstance(value, dict):
        return {str(key): _shape_or_type(nested) for key, nested in value.items()}
    return {"type": type(value).__name__}


def _jsonable_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _float_value(value: Any) -> float:
    scalar = _jsonable_scalar(value)
    if isinstance(scalar, (int, float)):
        return float(scalar)
    return 0.0


def _write_maniskill_blocker(path: Path, rollout: ManiSkillRolloutResult) -> None:
    if rollout.status == "passed":
        lines = ["# ManiSkill CP24 Blocker", ""]
        if rollout.env_blockers:
            lines.extend(
                [
                    "- Pipeline executable: `True`",
                    f"- Requested env id: `{rollout.requested_env_id}`",
                    f"- Executed fallback env id: `{rollout.env_id}`",
                    "",
                    "## Blocked Target Envs",
                    "",
                ]
            )
            lines.extend(f"- `{env_id}`: {blocker}" for env_id, blocker in rollout.env_blockers.items())
        else:
            lines.append("- None")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    path.write_text(
        "\n".join(
            [
                "# ManiSkill CP24 Blocker",
                "",
                f"- Env id: `{rollout.env_id}`",
                f"- Requested env id: `{rollout.requested_env_id}`",
                f"- Attempted env ids: `{', '.join(rollout.attempted_env_ids)}`",
                f"- Blocker: `{rollout.blocker}`",
                "",
                "## Install",
                "",
                "```bash",
                "sh scripts/bootstrap_checkpoint_24.sh",
                "sh scripts/checkpoint_24.sh --require-maniskill",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_smolvla_maniskill_plan(
    path: Path,
    env_id: str,
    model_id: str,
    smolvla_ready: bool,
    smolvla_blockers: list[str],
) -> None:
    blocker_lines = smolvla_blockers or ["None"]
    path.write_text(
        "\n".join(
            [
                "# SmolVLA to ManiSkill Evaluation Plan",
                "",
                f"- Research checkpoint: `checkpoint_24`",
                f"- First env target: `{env_id}`",
                "- HAB target: add ManiSkill-HAB task ids after the base ManiSkill smoke passes",
                f"- Model id: `{model_id}`",
                f"- Current SmolVLA import ready: `{smolvla_ready}`",
                "",
                "## Install Gate",
                "",
                "```bash",
                "sh scripts/bootstrap_checkpoint_24.sh",
                "sh scripts/checkpoint_24.sh --require-maniskill",
                "```",
                "",
                "## Pipeline Steps",
                "",
                "1. Run ManiSkill state-observation rollout and save `episodes.jsonl`, "
                "`metrics.json`, and `summary.md`.",
                "2. Add a ManiSkill observation bridge that maps RGB/state observations "
                "into LeRobot-style SmolVLA features.",
                "3. Add an action bridge that clips SmolVLA action chunks into the "
                "ManiSkill action space.",
                "4. Evaluate `policy_only` first, then reuse the CP20-23 "
                "planner/verifier/retry comparison contract.",
                "5. Report success rate, retry count, recovery success, verifier errors, "
                "latency, and trace/video artifacts.",
                "",
                "## Current SmolVLA Blockers",
                "",
                *[f"- {blocker}" for blocker in blocker_lines],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_robocasa_checkpoint_plan(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Checkpoint 25 RoboCasa Evaluation Plan",
                "",
                "- checkpoint: `checkpoint_25_robocasa_long_horizon_eval`",
                "- benchmark family: RoboCasa / RoboCasa365",
                "- purpose: long-horizon household manipulation evaluation for planner, "
                "verifier, retry, and replan logic",
                "- first gate: dependency probe plus one robosuite/RoboCasa reset-step rollout",
                "- strict gate: task success metrics, trace/video artifacts, and "
                "`policy_only` vs `agentic_retry` comparison",
                "- Mac note: asset download is large; keep CP25 separate from CP24 so "
                "ManiSkill remains the lightweight local gate",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 24 ManiSkill/HAB research eval planning.")
    parser.add_argument("--output-dir", default="_workspace/checkpoints/checkpoint_24")
    parser.add_argument("--env-id", default=DEFAULT_MANISKILL_ENV_ID)
    parser.add_argument(
        "--fallback-env-id",
        action="append",
        default=list(DEFAULT_MANISKILL_FALLBACK_ENV_IDS),
        help="Fallback ManiSkill env id used to verify the pipeline in headless sessions.",
    )
    parser.add_argument(
        "--no-fallback-env",
        action="store_true",
        help="Disable fallback envs so the requested env must execute.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--policy",
        action="append",
        choices=SUPPORTED_MANISKILL_POLICIES,
        default=list(DEFAULT_MANISKILL_POLICIES),
        help="Baseline policy to evaluate. Repeat for multiple baselines.",
    )
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow SmolVLA real policy loading to download model files if they are not cached.",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="Use ManiSkill RGB camera observations for smolvla_real instead of zero image tensors.",
    )
    parser.add_argument("--require-maniskill", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_checkpoint(
        output_dir=Path(args.output_dir),
        env_id=args.env_id,
        fallback_env_ids=() if args.no_fallback_env else tuple(args.fallback_env_id),
        episodes=args.episodes,
        steps=args.steps,
        policies=tuple(dict.fromkeys(args.policy)),
        model_id=args.model_id,
        require_maniskill=args.require_maniskill,
        allow_download=args.allow_download,
        real_images=args.real_images,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(f"{report.checkpoint}: {report.status}")
        print(f"output_dir={report.output_dir}")
        for name, passed in report.checks.items():
            print(f"- {'PASS' if passed else 'FAIL'} {name}")
        print(
            "metrics="
            f"env:{report.metrics['executed_env_id']} "
            f"episodes:{report.metrics['rollout_episodes']} "
            f"rollout:{report.metrics['rollout_status']} "
            f"success_rate:{report.metrics['rollout_success_count']}/"
            f"{report.metrics['rollout_episodes']} "
            f"smolvla_ready:{report.metrics['smolvla_ready']}"
        )
    if report.status != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
