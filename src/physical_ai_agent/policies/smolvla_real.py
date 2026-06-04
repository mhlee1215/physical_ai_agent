from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from physical_ai_agent.sim.so101_nexus_env import DEFAULT_SO101_ENV_ID, SO101NexusEnv, SO101Step


@dataclass(frozen=True)
class SmolVLAInferenceResult:
    model_id: str
    status: str
    action: list[float]
    action_shape: list[int]
    duration_s: float
    report_path: str
    blocker_path: str
    blocker: str | None
    pretrained: bool
    local_files_only: bool
    trace_path: str
    frame_path: str
    gif_path: str
    rollout_steps: int


def run_real_smolvla_inference_probe(
    output_dir: Path,
    observation: list[float],
    action_dim: int,
    env_id: str = DEFAULT_SO101_ENV_ID,
    rollout_steps: int = 6,
    model_id: str = DEFAULT_SMOLVLA_MODEL_ID,
    local_files_only: bool = True,
) -> SmolVLAInferenceResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "smolvla_real_inference_report.json"
    blocker_path = output_dir / "smolvla_real_inference_blocker.md"
    trace_path = output_dir / "smolvla_real_rollout.jsonl"
    frame_path = output_dir / "smolvla_real_rollout_3d.png"
    gif_path = output_dir / "smolvla_real_rollout_3d.gif"
    _unlink_if_exists(blocker_path, trace_path, frame_path, gif_path)
    started_at = perf_counter()
    action: list[float] = []
    action_shape: list[int] = []
    blocker = None
    records: list[SO101Step] = []

    try:
        policy = _load_pretrained_policy(model_id=model_id, local_files_only=local_files_only)
        records, frames = _run_policy_rollout(
            policy=policy,
            env_id=env_id,
            action_dim=action_dim,
            rollout_steps=rollout_steps,
        )
        if not records:
            raise RuntimeError("SmolVLA rollout produced no SO101 steps")
        if not frames:
            raise RuntimeError("SmolVLA rollout produced no 3D RGB frames")
        if records:
            action = records[0].action
            action_shape = [1, len(action)]
        _write_rgb_frames(frames, frame_path, gif_path)
        _write_trace(records, trace_path)
        status = "passed"
        blocker_path.write_text("", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        blocker = _short_error(exc)
        status = "blocked"
        _write_smolvla_blocker(blocker_path, model_id, local_files_only, blocker)

    result = SmolVLAInferenceResult(
        model_id=model_id,
        status=status,
        action=action,
        action_shape=action_shape,
        duration_s=round(perf_counter() - started_at, 4),
        report_path=str(report_path),
        blocker_path=str(blocker_path),
        blocker=blocker,
        pretrained=True,
        local_files_only=local_files_only,
        trace_path=str(trace_path),
        frame_path=str(frame_path),
        gif_path=str(gif_path),
        rollout_steps=len(records),
    )
    report_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _run_policy_rollout(
    policy: Any,
    env_id: str,
    action_dim: int,
    rollout_steps: int,
) -> tuple[list[SO101Step], list[Any]]:
    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    renderer = None
    frames: list[Any] = []
    records: list[SO101Step] = []
    try:
        obs, _info = env.reset(seed=3)
        renderer = _make_renderer_or_none(env)
        for step in range(rollout_steps):
            batch = _build_batch_for_policy(policy, obs)
            raw_action = policy.select_action(batch)
            action = _clip_action(_tensor_to_float_list(raw_action), action_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            records.append(
                SO101Step(
                    step=step,
                    observation=obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info={key: str(value) for key, value in info.items()},
                )
            )
            if renderer is not None:
                renderer.update_scene(env.env.unwrapped.data)
                frames.append(renderer.render())
            if terminated or truncated:
                break
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    return records, frames


def _make_renderer_or_none(env: SO101NexusEnv):
    try:
        import mujoco

        return mujoco.Renderer(env.env.unwrapped.model, height=480, width=640)
    except Exception:  # noqa: BLE001
        return None


def _load_pretrained_policy(model_id: str, local_files_only: bool):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return SmolVLAPolicy.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        map_location="cpu",
        device="cpu",
    )


def _build_batch_for_policy(policy: Any, observation: list[float]) -> dict[str, Any]:
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    config = policy.config
    state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else min(32, len(observation))
    state = torch.zeros(1, state_dim, dtype=torch.float32)
    source = torch.tensor(observation[:state_dim], dtype=torch.float32)
    state[0, : len(source)] = source
    batch: dict[str, Any] = {
        OBS_STATE: state,
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
    }
    for key, feature in config.image_features.items():
        batch[key] = torch.zeros(1, *feature.shape, dtype=torch.float32)
    return {key: value.to(config.device) if hasattr(value, "to") else value for key, value in batch.items()}


def _write_trace(records: list[SO101Step], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_rgb_frames(frames: list[Any], frame_path: Path, gif_path: Path) -> None:
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    images[-1].save(frame_path)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=120, loop=0)


def _tensor_to_float_list(value: Any) -> list[float]:
    import torch

    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [float(item) for item in value]


def _clip_action(action: list[float], action_dim: int) -> list[float]:
    if len(action) >= action_dim:
        return action[:action_dim]
    return action + [0.0 for _ in range(action_dim - len(action))]


def _write_smolvla_blocker(path: Path, model_id: str, local_files_only: bool, blocker: str) -> None:
    lines = [
        "# CP15 Real SmolVLA Inference Blocker",
        "",
        f"- Model id: `{model_id}`",
        f"- Local files only: `{local_files_only}`",
        f"- Blocker: `{blocker}`",
        "",
        "This gate requires LeRobot's `SmolVLAPolicy.from_pretrained()` to load pretrained weights and",
        "successfully execute `select_action()` on a SO101-shaped observation batch and step SO101-Nexus.",
        "",
        "To allow Hugging Face download from a normal networked terminal, run:",
        "",
        "```bash",
        "sh scripts/checkpoint_14_15.sh --allow-download --require-real-smolvla",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _short_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text.replace("\n", " ")[:800]


def _unlink_if_exists(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
