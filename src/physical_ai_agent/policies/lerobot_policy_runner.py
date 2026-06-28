from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.so101_visual_servo import VisualServoError, visual_servo_delta_q


@dataclass
class LeRobotPolicyRunner:
    """Shared LeRobot inference path with policy processors applied.

    This mirrors LeRobot's eval rollout contract:
    env_preprocessor -> policy preprocessor -> policy.select_action ->
    policy postprocessor -> env_postprocessor.
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any | None = None
    env_postprocessor: Any | None = None
    processor_source: str = "unknown"
    visual_servo_head: Any | None = None

    def select_action(self, observation: dict[str, Any]) -> Any:
        return self.select_action_with_trace(observation)["action"]

    def select_action_with_trace(self, observation: dict[str, Any]) -> dict[str, Any]:
        action_key = _lerobot_action_key()
        inference_mode = _torch_inference_mode()

        processed_observation = dict(observation)
        if self.env_preprocessor is not None:
            processed_observation = self.env_preprocessor(processed_observation)

        processed_observation = self.preprocessor(processed_observation)
        processed_observation = _move_tensors_to_policy_device(processed_observation, self.policy)
        with inference_mode:
            raw_action = self.policy.select_action(processed_observation)
        postprocessed_action = self.postprocessor(raw_action)

        if self.env_postprocessor is not None:
            action_transition = self.env_postprocessor({action_key: postprocessed_action})
            action = action_transition[action_key]
        else:
            action = postprocessed_action
        return {
            "action": action,
            "raw_action": raw_action,
            "postprocessed_action": postprocessed_action,
            "processor_source": self.processor_source,
            "preprocessor_steps": [type(step).__name__ for step in getattr(self.preprocessor, "steps", [])],
            "postprocessor_steps": [type(step).__name__ for step in getattr(self.postprocessor, "steps", [])],
        }

    def predict_visual_servo_with_trace(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.visual_servo_head is None:
            raise RuntimeError("visual_servo_head.pt is required for visual_servo_delta_q rollout")
        inference_mode = _torch_inference_mode()
        processed_observation = dict(observation)
        if self.env_preprocessor is not None:
            processed_observation = self.env_preprocessor(processed_observation)
        processed_observation = self.preprocessor(processed_observation)
        processed_observation = _move_tensors_to_policy_device(processed_observation, self.policy)
        with inference_mode:
            pred = self.visual_servo_head(processed_observation)
        camera1 = pred["camera1"][0].detach().float().cpu().tolist()
        camera2 = pred["camera2"][0].detach().float().cpu().tolist()
        stop_prob = float(_torch_sigmoid(pred["stop_logit"][0]))
        delta_q = visual_servo_delta_q(
            VisualServoError(
                wrist_dx_norm=float(camera2[0]),
                wrist_dy_norm=float(camera2[1]),
                edge_angle_error=float(camera2[2]),
                stop_prob=stop_prob,
            )
        ).tolist()
        return {
            "camera1": {"dx_norm": float(camera1[0]), "dy_norm": float(camera1[1]), "edge_angle_error": float(camera1[2]), "visible": True},
            "camera2": {"dx_norm": float(camera2[0]), "dy_norm": float(camera2[1]), "edge_angle_error": float(camera2[2]), "visible": True},
            "stop_prob": stop_prob,
            "delta_q": [float(value) for value in delta_q],
            "processor_source": self.processor_source,
        }


def load_lerobot_policy_runner(
    policy_path: str,
    *,
    device: str = "cuda",
    policy_type: str = "smolvla",
    rename_map: dict[str, str] | None = None,
    local_files_only: bool = True,
) -> LeRobotPolicyRunner:
    """Load a LeRobot policy and its saved pre/postprocessors.

    Use this for custom benchmark environments that cannot call
    `lerobot.scripts.lerobot_eval` directly. It prevents the common bug where a
    checkpoint is loaded but its normalizer/unnormalizer processors are skipped.
    """

    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_cls = get_policy_class(policy_type)
    policy = _load_policy_from_pretrained(
        policy_cls=policy_cls,
        policy_path=policy_path,
        local_files_only=local_files_only,
        map_location=device,
        device=device,
    )
    policy.eval()

    preprocessor_overrides: dict[str, Any] = {
        "device_processor": {"device": str(policy.config.device)},
    }
    if rename_map is not None:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": rename_map}

    preprocessor, postprocessor, processor_source = _load_pre_post_processors(
        policy=policy,
        policy_path=policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    _align_processor_stats_to_declared_features(preprocessor)
    _align_processor_stats_to_declared_features(postprocessor)
    visual_servo_head = _load_visual_servo_head_if_present(Path(policy_path), device=device)
    return LeRobotPolicyRunner(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        processor_source=processor_source,
        visual_servo_head=visual_servo_head,
    )


def _load_visual_servo_head_if_present(policy_path: Path, *, device: str) -> Any | None:
    path = policy_path / "visual_servo_head.pt"
    if not path.exists():
        return None
    from physical_ai_agent.policies.so101_visual_servo_head import load_visual_servo_head

    return load_visual_servo_head(path, device=device)


def _torch_sigmoid(value: Any) -> Any:
    import torch

    return torch.sigmoid(value).detach().cpu()


def _load_pre_post_processors(*, policy: Any, policy_path: str, preprocessor_overrides: dict[str, Any]):
    from lerobot.policies.factory import make_pre_post_processors

    root = Path(policy_path)
    has_preprocessor = (root / "policy_preprocessor.json").exists()
    has_postprocessor = (root / "policy_postprocessor.json").exists()
    if has_preprocessor:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        return preprocessor, postprocessor, "saved_preprocessor_and_postprocessor"

    preprocessor, fallback_postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=None,
    )
    if not has_postprocessor:
        return preprocessor, fallback_postprocessor, "factory_preprocessor_and_factory_postprocessor"

    try:
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=policy_path,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    except Exception:  # noqa: BLE001 - fallback preserves loadability for checkpoints with incomplete processor artifacts.
        postprocessor = fallback_postprocessor
        return preprocessor, postprocessor, "factory_preprocessor_and_factory_postprocessor_after_saved_postprocessor_error"

    return preprocessor, postprocessor, "factory_preprocessor_and_saved_postprocessor"


def _align_processor_stats_to_declared_features(processor: Any) -> None:
    for step in getattr(processor, "steps", []):
        features = getattr(step, "features", None)
        stats = getattr(step, "stats", None)
        tensor_stats = getattr(step, "_tensor_stats", None)
        if not isinstance(features, dict) or not isinstance(stats, dict):
            continue
        for key, feature in features.items():
            expected_size = _feature_flat_size(feature)
            if expected_size <= 0:
                continue
            if key in stats:
                _trim_stat_entry(stats[key], expected_size)
            if isinstance(tensor_stats, dict) and key in tensor_stats:
                _trim_stat_entry(tensor_stats[key], expected_size)


def _load_policy_from_pretrained(
    *,
    policy_cls: Any,
    policy_path: str,
    local_files_only: bool,
    map_location: str,
    device: str,
) -> Any:
    if not local_files_only:
        return policy_cls.from_pretrained(
            policy_path,
            local_files_only=False,
            map_location=map_location,
            device=device,
        )

    try:
        from huggingface_hub import snapshot_download
        from unittest.mock import patch
        import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_with_expert
    except Exception:  # noqa: BLE001
        return policy_cls.from_pretrained(
            policy_path,
            local_files_only=True,
            map_location=map_location,
            device=device,
        )

    try:
        vlm_snapshot = snapshot_download(
            repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        vlm_snapshot = None

    if vlm_snapshot is None:
        return policy_cls.from_pretrained(
            policy_path,
            local_files_only=True,
            map_location=map_location,
            device=device,
        )

    def _local_from_pretrained(original: Any):
        def wrapped(pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> Any:
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
        return policy_cls.from_pretrained(
            policy_path,
            local_files_only=True,
            map_location=map_location,
            device=device,
        )


def _feature_flat_size(feature: Any) -> int:
    size = 1
    shape = getattr(feature, "shape", None)
    if shape is None:
        return 0
    for dim in shape:
        size *= int(dim)
    return size


def _trim_stat_entry(entry: Any, expected_size: int) -> None:
    if not isinstance(entry, dict):
        return
    for stat_key in ("mean", "std", "min", "max"):
        if stat_key not in entry:
            continue
        value = entry[stat_key]
        if isinstance(value, list) and len(value) > expected_size:
            entry[stat_key] = value[:expected_size]
            continue
        shape = getattr(value, "shape", None)
        if not shape or int(shape[-1]) <= expected_size:
            continue
        entry[stat_key] = value[..., :expected_size]


def _move_tensors_to_policy_device(value: Any, policy: Any) -> Any:
    device = getattr(getattr(policy, "config", None), "device", None)
    if device is None:
        return value
    try:
        import torch
    except Exception:  # noqa: BLE001
        return value

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors_to_policy_device(item, policy) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_policy_device(item, policy) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_policy_device(item, policy) for item in value)
    return value


def _lerobot_action_key() -> str:
    try:
        from lerobot.utils.constants import ACTION

        return str(ACTION)
    except Exception:  # noqa: BLE001
        return "action"


def _torch_inference_mode():
    try:
        import torch

        return torch.inference_mode()
    except Exception:  # noqa: BLE001
        from contextlib import nullcontext

        return nullcontext()
