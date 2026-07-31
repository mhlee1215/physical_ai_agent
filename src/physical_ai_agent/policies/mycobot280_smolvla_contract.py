from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


STATE_KEY = "observation.state"
ACTION_KEY = "action"
CAMERA_KEY = "observation.images.camera1"
EXPECTED_STATE_DIM = 7
EXPECTED_ACTION_DIM = 7
CONTRACT_FILENAME = "mycobot280_feature_contract.json"
ZERO_VARIANCE_THRESHOLD = 1e-7


def adapt_policy_to_mycobot280_dataset(policy: Any, dataset_meta: Any) -> dict[str, Any]:
    """Bind a loaded SmolVLA policy to the exact myCobot dataset features."""
    config = getattr(policy, "config", None)
    if config is None:
        raise ValueError("SmolVLA policy has no config to adapt")
    return adapt_policy_config_to_dataset(config, dataset_meta)


def adapt_policy_config_to_dataset(config: Any, dataset_meta: Any) -> dict[str, Any]:
    from lerobot.configs import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features

    features = dataset_to_policy_features(dataset_meta.features)
    state = _require_feature(features, STATE_KEY, FeatureType.STATE, EXPECTED_STATE_DIM)
    action = _require_feature(features, ACTION_KEY, FeatureType.ACTION, EXPECTED_ACTION_DIM)
    camera = _require_feature(features, CAMERA_KEY, FeatureType.VISUAL)

    config.input_features = {STATE_KEY: state, CAMERA_KEY: camera}
    config.output_features = {ACTION_KEY: action}
    if hasattr(config, "empty_cameras"):
        config.empty_cameras = 0

    action_names = dataset_meta.features.get(ACTION_KEY, {}).get("names")
    if action_names is not None and hasattr(config, "action_feature_names"):
        config.action_feature_names = list(action_names)
    set_dataset_feature_metadata = getattr(config, "set_dataset_feature_metadata", None)
    if callable(set_dataset_feature_metadata):
        set_dataset_feature_metadata(dataset_meta.features)
    config._runtime_dataset_meta = dataset_meta

    return policy_feature_contract(config)


def make_mycobot280_pre_post_processors(
    *,
    policy: Any,
    dataset_meta: Any,
    policy_path: str,
    selected_device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Create dataset-stat processors or load a validated local 7D checkpoint pair."""
    from lerobot.policies.factory import make_pre_post_processors

    feature_contract = adapt_policy_to_mycobot280_dataset(policy, dataset_meta)
    processor_overrides = {"device_processor": {"device": selected_device}}
    local_policy_path = Path(policy_path).expanduser()
    if local_policy_path.is_dir():
        normalization_adjustments: list[dict[str, Any]] = []
        processor_contract = validate_saved_processor_contract(local_policy_path)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=str(local_policy_path),
            preprocessor_overrides=processor_overrides,
        )
        source = "validated_saved_checkpoint"
    else:
        dataset_stats, normalization_adjustments = stabilize_zero_variance_statistics(dataset_meta.stats)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_stats,
            preprocessor_overrides=processor_overrides,
        )
        processor_contract = feature_contract
        source = "dataset_statistics"

    return preprocessor, postprocessor, {
        "feature_contract": feature_contract,
        "processor_contract": processor_contract,
        "processor_source": source,
        "normalization_adjustments": normalization_adjustments,
    }


def stabilize_zero_variance_statistics(
    dataset_stats: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Center constant dimensions on their observed value before mean/std scaling."""
    stabilized = deepcopy(dataset_stats)
    adjustments: list[dict[str, Any]] = []
    for key in (STATE_KEY, ACTION_KEY):
        feature_stats = stabilized.get(key)
        if feature_stats is None:
            raise ValueError(f"Dataset statistics are missing required feature {key!r}")
        means = [float(value) for value in feature_stats["mean"]]
        stds = [float(value) for value in feature_stats["std"]]
        minimums = [float(value) for value in feature_stats["min"]]
        maximums = [float(value) for value in feature_stats["max"]]
        lengths = {len(means), len(stds), len(minimums), len(maximums)}
        if len(lengths) != 1:
            raise ValueError(f"Dataset statistics for {key!r} have inconsistent dimensions")
        for index, (std, minimum, maximum) in enumerate(
            zip(stds, minimums, maximums, strict=True)
        ):
            observed_range = maximum - minimum
            if abs(std) <= ZERO_VARIANCE_THRESHOLD and abs(observed_range) <= ZERO_VARIANCE_THRESHOLD:
                replacement = (minimum + maximum) / 2.0
                adjustments.append(
                    {
                        "feature": key,
                        "dimension": index,
                        "original_mean": means[index],
                        "replacement_mean": replacement,
                        "std": std,
                        "observed_range": observed_range,
                        "reason": "constant_dimension_centering",
                    }
                )
                means[index] = replacement
        feature_stats["mean"] = means
    return stabilized, adjustments


def policy_feature_contract(config: Any) -> dict[str, Any]:
    input_features = getattr(config, "input_features", {})
    output_features = getattr(config, "output_features", {})
    return {
        "state_key": STATE_KEY,
        "state_shape": _shape_list(input_features.get(STATE_KEY)),
        "camera_keys": sorted(
            key for key in input_features if key.startswith("observation.images.")
        ),
        "action_key": ACTION_KEY,
        "action_shape": _shape_list(output_features.get(ACTION_KEY)),
        "exact_7d_state_action": (
            _shape_list(input_features.get(STATE_KEY)) == [EXPECTED_STATE_DIM]
            and _shape_list(output_features.get(ACTION_KEY)) == [EXPECTED_ACTION_DIM]
        ),
    }


def validate_saved_processor_contract(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    preprocessor_path = checkpoint_dir / "policy_preprocessor.json"
    postprocessor_path = checkpoint_dir / "policy_postprocessor.json"
    contract_path = checkpoint_dir / CONTRACT_FILENAME
    for path in (preprocessor_path, postprocessor_path, contract_path):
        if not path.is_file():
            raise ValueError(f"Saved processor config is missing: {path}")

    saved_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    postprocessor = json.loads(postprocessor_path.read_text(encoding="utf-8"))
    pre_features = _serialized_features(preprocessor)
    post_features = _serialized_features(postprocessor)
    state_shape = _serialized_shape(pre_features, STATE_KEY)
    pre_action_shape = _serialized_shape(pre_features, ACTION_KEY)
    post_action_shape = _serialized_shape(post_features, ACTION_KEY)
    camera_keys = sorted(
        key for key in pre_features if key.startswith("observation.images.")
    )

    problems: list[str] = []
    if state_shape != [EXPECTED_STATE_DIM]:
        problems.append(f"preprocessor state shape is {state_shape}, expected [7]")
    if pre_action_shape != [EXPECTED_ACTION_DIM]:
        problems.append(f"preprocessor action shape is {pre_action_shape}, expected [7]")
    if post_action_shape != [EXPECTED_ACTION_DIM]:
        problems.append(f"postprocessor action shape is {post_action_shape}, expected [7]")
    if camera_keys != [CAMERA_KEY]:
        problems.append(f"preprocessor cameras are {camera_keys}, expected [{CAMERA_KEY!r}]")
    if not saved_contract.get("feature_contract", {}).get("exact_7d_state_action"):
        problems.append("checkpoint feature marker does not declare exact 7D state/action")
    if saved_contract.get("processor_source") != "dataset_statistics":
        problems.append("checkpoint feature marker does not declare dataset-stat processors")
    if problems:
        raise ValueError("Saved myCobot processor contract is invalid: " + "; ".join(problems))

    return {
        "state_shape": state_shape,
        "action_shape": post_action_shape,
        "camera_keys": camera_keys,
        "exact_7d_state_action": True,
        "preprocessor_path": str(preprocessor_path),
        "postprocessor_path": str(postprocessor_path),
        "contract_path": str(contract_path),
        "normalization_adjustments": saved_contract.get("normalization_adjustments", []),
    }


def _require_feature(
    features: dict[str, Any],
    key: str,
    expected_type: Any,
    expected_dim: int | None = None,
) -> Any:
    feature = features.get(key)
    if feature is None:
        raise ValueError(f"myCobot dataset is missing required feature {key!r}")
    if getattr(feature, "type", None) is not expected_type:
        raise ValueError(f"myCobot feature {key!r} has unexpected type {feature.type!r}")
    shape = _shape_list(feature)
    if expected_dim is not None and shape != [expected_dim]:
        raise ValueError(
            f"myCobot feature {key!r} has shape {shape}, expected [{expected_dim}]"
        )
    return feature


def _shape_list(feature: Any) -> list[int] | None:
    if feature is None:
        return None
    shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
    return [int(value) for value in shape] if shape is not None else None


def _serialized_features(payload: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for step in payload.get("steps", []):
        config_features = step.get("config", {}).get("features", {})
        features.update(config_features)
    return features


def _serialized_shape(features: dict[str, Any], key: str) -> list[int] | None:
    return _shape_list(features.get(key))
