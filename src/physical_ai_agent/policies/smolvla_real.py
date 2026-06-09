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
    input_manifest_path: str
    input_preview_path: str
    input_preview_gif_path: str
    image_feature_mapping: dict[str, str]
    used_real_camera_inputs: bool
    device_requested: str
    device_selected: str
    device_probe: dict[str, Any]
    device_fallback_reason: str | None


def run_real_smolvla_inference_probe(
    output_dir: Path,
    observation: list[float],
    action_dim: int,
    env_id: str = DEFAULT_SO101_ENV_ID,
    rollout_steps: int = 6,
    model_id: str = DEFAULT_SMOLVLA_MODEL_ID,
    local_files_only: bool = True,
    use_real_camera_inputs: bool = False,
    device: str = "auto",
) -> SmolVLAInferenceResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "smolvla_real_inference_report.json"
    blocker_path = output_dir / "smolvla_real_inference_blocker.md"
    trace_path = output_dir / "smolvla_real_rollout.jsonl"
    frame_path = output_dir / "smolvla_real_rollout_3d.png"
    gif_path = output_dir / "smolvla_real_rollout_3d.gif"
    input_manifest_path = output_dir / "smolvla_real_input_manifest.json"
    input_preview_path = output_dir / "smolvla_real_input_preview.png"
    input_preview_gif_path = output_dir / "smolvla_real_input_preview.gif"
    _unlink_if_exists(
        blocker_path,
        trace_path,
        frame_path,
        gif_path,
        input_manifest_path,
        input_preview_path,
        input_preview_gif_path,
    )
    started_at = perf_counter()
    action: list[float] = []
    action_shape: list[int] = []
    blocker = None
    records: list[SO101Step] = []
    image_feature_mapping: dict[str, str] = {}
    device_metadata: dict[str, Any] = {}

    try:
        policy = _load_pretrained_policy(
            model_id=model_id,
            local_files_only=local_files_only,
            device=device,
        )
        device_metadata = _policy_device_metadata(policy)
        records, frames, image_feature_mapping = _run_policy_rollout(
            policy=policy,
            env_id=env_id,
            action_dim=action_dim,
            rollout_steps=rollout_steps,
            output_dir=output_dir,
            use_real_camera_inputs=use_real_camera_inputs,
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
        input_manifest_path=str(input_manifest_path),
        input_preview_path=str(input_preview_path),
        input_preview_gif_path=str(input_preview_gif_path),
        image_feature_mapping=image_feature_mapping,
        used_real_camera_inputs=use_real_camera_inputs,
        device_requested=str(device_metadata.get("device_requested", device)),
        device_selected=str(device_metadata.get("device_selected", "unknown")),
        device_probe=dict(device_metadata.get("device_probe", {})),
        device_fallback_reason=device_metadata.get("device_fallback_reason"),
    )
    report_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _run_policy_rollout(
    policy: Any,
    env_id: str,
    action_dim: int,
    rollout_steps: int,
    output_dir: Path,
    use_real_camera_inputs: bool,
) -> tuple[list[SO101Step], list[Any], dict[str, str]]:
    import json

    from physical_ai_agent.sim.so101_camera_input import SO101InputFrame, _make_camera, _write_input_preview

    env = SO101NexusEnv(env_id=env_id, render_mode=None)
    renderer = None
    camera_renderers: dict[str, Any] = {}
    frames: list[Any] = []
    records: list[SO101Step] = []
    input_records: list[SO101InputFrame] = []
    image_feature_mapping: dict[str, str] = {}
    input_frames_dir = output_dir / "input_frames"
    input_frames_dir.mkdir(parents=True, exist_ok=True)
    _clear_dir(input_frames_dir)
    try:
        obs, _info = env.reset(seed=3)
        renderer = _make_renderer_or_none(env)
        if use_real_camera_inputs:
            import mujoco

            camera_renderers = {
                name: mujoco.Renderer(env.env.unwrapped.model, height=480, width=640)
                for name in ("wrist_cam", "egocentric_cam", "top_down")
            }
        for step in range(rollout_steps):
            camera_pixels = (
                _render_policy_cameras(env, camera_renderers)
                if use_real_camera_inputs
                else {}
            )
            batch, image_feature_mapping = _build_batch_for_policy(policy, obs, camera_pixels)
            raw_action = policy.select_action(batch)
            action = _clip_action(_tensor_to_float_list(raw_action), action_dim)
            camera_paths: dict[str, str] = {}
            if camera_pixels:
                for camera_name, pixels in camera_pixels.items():
                    camera_path = input_frames_dir / f"step_{step:03d}_{camera_name}.png"
                    _write_image(pixels, camera_path)
                    camera_paths[camera_name] = str(camera_path)
            obs, reward, terminated, truncated, info = env.step(action)
            if camera_paths:
                input_records.append(
                    SO101InputFrame(
                        step=step,
                        observation=obs,
                        action=action,
                        reward=reward,
                        camera_frames=camera_paths,
                    )
                )
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
        for camera_renderer in camera_renderers.values():
            camera_renderer.close()
        env.close()
    if input_records:
        input_manifest_path = output_dir / "smolvla_real_input_manifest.json"
        input_manifest_path.write_text(
            json.dumps(
                {
                    "env_id": env_id,
                    "policy_input_names": ["wrist_cam", "egocentric_cam"],
                    "debug_input_names": ["top_down"],
                    "image_feature_mapping": image_feature_mapping,
                    "frames": [asdict(record) for record in input_records],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_input_preview(
            input_records,
            output_dir / "smolvla_real_input_preview.png",
            output_dir / "smolvla_real_input_preview.gif",
        )
    return records, frames, image_feature_mapping


def _make_renderer_or_none(env: SO101NexusEnv):
    try:
        import mujoco

        return mujoco.Renderer(env.env.unwrapped.model, height=480, width=640)
    except Exception:  # noqa: BLE001
        return None


def _load_pretrained_policy(model_id: str, local_files_only: bool, device: str = "auto"):
    from unittest.mock import patch

    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_with_expert

    device_plan = _select_policy_device(device)

    def _from_pretrained(selected_device: str):
        return SmolVLAPolicy.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            map_location=selected_device,
            device=selected_device,
        )

    if not local_files_only:
        return _load_with_cpu_fallback(
            loader=_from_pretrained,
            device_plan=device_plan,
        )

    vlm_snapshot = snapshot_download(
        repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        local_files_only=True,
    )

    def _local_from_pretrained(original):
        def wrapped(pretrained_model_name_or_path, *args, **kwargs):
            if pretrained_model_name_or_path == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
                pretrained_model_name_or_path = vlm_snapshot
            kwargs.setdefault("local_files_only", True)
            return original(pretrained_model_name_or_path, *args, **kwargs)

        return wrapped

    with (
        patch.object(
            smolvlm_with_expert.AutoModelForImageTextToText,
            "from_pretrained",
            _local_from_pretrained(smolvlm_with_expert.AutoModelForImageTextToText.from_pretrained),
        ),
        patch.object(
            smolvlm_with_expert.AutoProcessor,
            "from_pretrained",
            _local_from_pretrained(smolvlm_with_expert.AutoProcessor.from_pretrained),
        ),
        patch.object(
            smolvlm_with_expert.AutoConfig,
            "from_pretrained",
            _local_from_pretrained(smolvlm_with_expert.AutoConfig.from_pretrained),
        ),
    ):
        def _local_policy(selected_device: str):
            return SmolVLAPolicy.from_pretrained(
                model_id,
                local_files_only=True,
                map_location=selected_device,
                device=selected_device,
            )

        return _load_with_cpu_fallback(
            loader=_local_policy,
            device_plan=device_plan,
        )


def _select_policy_device(device: str, torch_module: Any | None = None) -> dict[str, Any]:
    normalized = device.lower()
    if normalized not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported SmolVLA device '{device}'. Use auto, cpu, mps, or cuda.")

    probe = _torch_device_probe(torch_module)
    fallback_reason = None
    selected = normalized
    if normalized == "auto":
        if probe["mps_available"]:
            selected = "mps"
        else:
            selected = "cpu"
            fallback_reason = _mps_unavailable_reason(probe)
    elif normalized == "mps" and not probe["mps_available"]:
        selected = "cpu"
        fallback_reason = _mps_unavailable_reason(probe)
    elif normalized == "cuda" and not probe["cuda_available"]:
        selected = "cpu"
        fallback_reason = "CUDA requested but torch.cuda.is_available() is false."

    return {
        "requested": normalized,
        "selected": selected,
        "probe": probe,
        "fallback_reason": fallback_reason,
    }


def _torch_device_probe(torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        import torch as torch_module

    mps_backend = getattr(torch_module.backends, "mps", None)
    return {
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "mps_built": bool(mps_backend and mps_backend.is_built()),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "cuda_available": bool(torch_module.cuda.is_available()),
    }


def _mps_unavailable_reason(probe: dict[str, Any]) -> str:
    if not probe.get("mps_built"):
        return "MPS requested by auto policy, but this PyTorch build has no MPS backend."
    return "MPS requested by auto policy, but torch.backends.mps.is_available() is false."


def _load_with_cpu_fallback(loader: Any, device_plan: dict[str, Any]):
    selected = str(device_plan["selected"])
    fallback_reason = device_plan.get("fallback_reason")
    try:
        policy = loader(selected)
    except Exception as exc:  # noqa: BLE001 - preserve local MPS fallback detail.
        if selected == "cpu":
            raise
        fallback_reason = f"{selected} policy load failed; fell back to CPU: {_short_error(exc)}"
        policy = loader("cpu")
        selected = "cpu"

    _attach_policy_device_metadata(
        policy,
        requested=str(device_plan["requested"]),
        selected=selected,
        probe=dict(device_plan["probe"]),
        fallback_reason=fallback_reason,
    )
    return policy


def _attach_policy_device_metadata(
    policy: Any,
    *,
    requested: str,
    selected: str,
    probe: dict[str, Any],
    fallback_reason: str | None,
) -> None:
    setattr(policy, "_physical_ai_agent_device_requested", requested)
    setattr(policy, "_physical_ai_agent_device_selected", selected)
    setattr(policy, "_physical_ai_agent_device_probe", probe)
    setattr(policy, "_physical_ai_agent_device_fallback_reason", fallback_reason)


def _policy_device_metadata(policy: Any) -> dict[str, Any]:
    return {
        "device_requested": getattr(policy, "_physical_ai_agent_device_requested", "unknown"),
        "device_selected": getattr(policy, "_physical_ai_agent_device_selected", "unknown"),
        "device_probe": getattr(policy, "_physical_ai_agent_device_probe", {}),
        "device_fallback_reason": getattr(policy, "_physical_ai_agent_device_fallback_reason", None),
    }


def _render_policy_cameras(env: SO101NexusEnv, renderers: dict[str, Any]) -> dict[str, Any]:
    from physical_ai_agent.sim.so101_camera_input import _make_camera

    camera_pixels = {}
    for camera_name, renderer in renderers.items():
        renderer.update_scene(env.env.unwrapped.data, camera=_make_camera(env.env, camera_name))
        camera_pixels[camera_name] = renderer.render()
    return camera_pixels


def _build_batch_for_policy(
    policy: Any,
    observation: list[float],
    camera_pixels: dict[str, Any] | None = None,
    instruction: str | None = None,
    local_files_only: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    config = policy.config
    state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else min(32, len(observation))
    state = torch.zeros(1, state_dim, dtype=torch.float32)
    source = torch.tensor(observation[:state_dim], dtype=torch.float32)
    state[0, : len(source)] = source
    language_tokens, language_attention_mask = _language_tokens_for_policy(
        policy,
        instruction=instruction,
        local_files_only=local_files_only,
    )
    batch: dict[str, Any] = {OBS_STATE: state}
    batch[OBS_LANGUAGE_TOKENS] = language_tokens
    batch[OBS_LANGUAGE_ATTENTION_MASK] = language_attention_mask
    image_feature_mapping = {}
    for index, (key, feature) in enumerate(config.image_features.items()):
        source_name = _source_camera_name(index, camera_pixels or {})
        image_feature_mapping[key] = source_name
        if camera_pixels and source_name in camera_pixels:
            batch[key] = _pixels_to_tensor(camera_pixels[source_name], feature.shape)
        else:
            batch[key] = torch.zeros(1, *feature.shape, dtype=torch.float32)
    return (
        {key: value.to(config.device) if hasattr(value, "to") else value for key, value in batch.items()},
        image_feature_mapping,
    )


def _language_tokens_for_policy(
    policy: Any,
    *,
    instruction: str | None,
    local_files_only: bool,
):
    import torch

    if not instruction:
        return torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool)

    from transformers import AutoTokenizer

    config = policy.config
    prompt = instruction if instruction.endswith("\n") else f"{instruction}\n"
    tokenizer_model_name = config.vlm_model_name
    if local_files_only and tokenizer_model_name == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
        from huggingface_hub import snapshot_download

        tokenizer_model_name = snapshot_download(
            repo_id=tokenizer_model_name,
            local_files_only=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name,
        local_files_only=local_files_only,
    )
    tokenized = tokenizer(
        [prompt],
        padding=config.pad_language_to,
        truncation=True,
        max_length=config.tokenizer_max_length,
        return_tensors="pt",
    )
    return tokenized["input_ids"].to(torch.long), tokenized["attention_mask"].to(torch.bool)


def _source_camera_name(index: int, camera_pixels: dict[str, Any]) -> str:
    preferred = ["wrist_cam", "egocentric_cam", "egocentric_cam"]
    if index < len(preferred):
        return preferred[index]
    return next(iter(camera_pixels), "zero")


def _pixels_to_tensor(pixels: Any, feature_shape: tuple[int, ...]):
    import numpy as np
    import torch
    from PIL import Image

    channels, height, width = feature_shape
    image = Image.fromarray(pixels).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if channels != 3:
        raise ValueError(f"Expected RGB image feature with 3 channels, got {feature_shape}")
    return tensor


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
        "# Real SmolVLA Inference Blocker",
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
        "sh scripts/view_so101_live.sh --browser-only --policy smolvla --allow-download --smolvla-action-steps 15 --show-inputs --fps 2 --max-steps 1",
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


def _clear_dir(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def _write_image(pixels: Any, path: Path) -> None:
    from PIL import Image

    Image.fromarray(pixels).save(path)
