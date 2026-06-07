from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def select_action(self, observation: dict[str, Any]) -> Any:
        action_key = _lerobot_action_key()
        inference_mode = _torch_inference_mode()

        processed_observation = dict(observation)
        if self.env_preprocessor is not None:
            processed_observation = self.env_preprocessor(processed_observation)

        processed_observation = self.preprocessor(processed_observation)
        processed_observation = _move_tensors_to_policy_device(processed_observation, self.policy)
        with inference_mode:
            action = self.policy.select_action(processed_observation)
        action = self.postprocessor(action)

        if self.env_postprocessor is not None:
            action_transition = self.env_postprocessor({action_key: action})
            action = action_transition[action_key]
        return action


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

    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(
        policy_path,
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

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    _align_processor_stats_to_declared_features(preprocessor)
    _align_processor_stats_to_declared_features(postprocessor)
    return LeRobotPolicyRunner(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


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
