from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt, ValidationError, model_validator


SCHEMA_PATH = Path("configs/so101/schemas/training_config.schema.json")
TRAINING_CONFIG_DIR = Path("configs/so101/training")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExtensibleModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class GridBalanceConfig(ExtensibleModel):
    camera_key: Literal["observation.images.camera1"] | None = None
    grid_size: NonNegativeFloat | None = None
    frame_index: NonNegativeFloat | None = None
    min_area: NonNegativeFloat | None = None
    bins: list[int] | None = None
    visible_episodes: int | None = None
    episodes_per_bin: NonNegativeFloat | None = None
    bin_counts: dict[str, int] | None = None


class HfMergeSourceConfig(ExtensibleModel):
    name: str
    repo_id: str
    hf_repo_id: str
    hf_path_in_repo: str
    hf_repo_type: str | None = None
    expected_episodes: PositiveInt | None = None
    expected_frames: PositiveInt | None = None


class DatasetConfig(ExtensibleModel):
    name: str | None = None
    repo_id: str | None = None
    root: str | None = None
    split: str | None = None
    hf_repo_id: str | None = None
    hf_repo_type: str | None = None
    hf_path_in_repo: str | None = None
    grid_bin_sidecar: str | None = None
    expected_episodes: PositiveInt | None = None
    expected_frames: PositiveInt | None = None
    image_cache_dir: str | None = None
    hf_merge_sources: list[HfMergeSourceConfig] | None = None
    grid_balance: GridBalanceConfig | None = None

    @model_validator(mode="after")
    def require_dataset_source(self) -> DatasetConfig:
        has_direct = bool(self.repo_id and self.root)
        has_merge = bool(self.hf_merge_sources)
        if not has_direct and not has_merge:
            raise ValueError("dataset must define repo_id/root or hf_merge_sources")
        return self


class CameraContractConfig(StrictModel):
    camera1: Literal["egocentric_cam"] = Field(alias="observation.images.camera1")
    camera2: Literal["wrist_cam"] = Field(alias="observation.images.camera2")
    camera3: Literal["wrist_cam duplicate"] | None = Field(default=None, alias="observation.images.camera3")


class TensorBoardConfig(StrictModel):
    log_input_images_every_n_steps: int = Field(ge=0)
    log_input_metadata_every_n_steps: int = Field(ge=0)


class TrainingConfig(StrictModel):
    batch_size: PositiveInt | None = None
    num_workers: int | None = Field(default=None, ge=0)
    policy_repo_id: str | None = None
    policy_push_to_hub: bool | None = None
    lightning_precision: str | None = None
    steps_per_epoch: PositiveInt | None = None
    validation_max_batches: PositiveInt | None = None
    checkpoint_retention_policy: Literal["best_val_and_closed_loop", "keep_all", "none"] | None = None


class PredecodedImageCacheConfig(StrictModel):
    root_env: str | None = None
    default_root: str | None = None
    train: str | bool | dict[str, str] | None = None
    validation: str | bool | dict[str, str] | None = None


class AugmentationConfig(StrictModel):
    state_jitter_std: NonNegativeFloat | None = None
    state_jitter_arm_only: bool | None = None
    state_dropout_prob: float | None = Field(default=None, ge=0, lt=1)
    state_dropout_keep_gripper: bool | None = None
    image_camera_dropout_prob: float | None = Field(default=None, ge=0, lt=1)
    image_patch_dropout_prob: float | None = Field(default=None, ge=0, lt=1)
    image_patch_mask_ratio: float | None = Field(default=None, ge=0, lt=1)
    image_color_jitter: bool | None = None
    image_color_jitter_strength: NonNegativeFloat | None = None
    image_sharpness_jitter: bool | None = None
    image_affine_degrees: NonNegativeFloat | None = None
    image_affine_translate: NonNegativeFloat | None = None
    image_noise_std: NonNegativeFloat | None = None
    image_blur_prob: float | None = Field(default=None, ge=0, lt=1)
    image_blur_kernel_size: PositiveInt | None = None
    image_motion_blur_prob: float | None = Field(default=None, ge=0, lt=1)
    image_motion_blur_kernel_size: PositiveInt | None = None
    gpu_image_augmentation: bool | None = None


class WeightedStepsConfig(ExtensibleModel):
    steps: PositiveInt | None = None
    weight: NonNegativeFloat | None = None


class ActionSmoothnessConfig(ExtensibleModel):
    weight: NonNegativeFloat | None = None
    include_gripper: bool | None = None


class ActionRmseSweepConfig(ExtensibleModel):
    enabled: bool | None = None
    episodes: PositiveInt | None = None
    tensorboard_tag: str | None = None
    n_action_steps: list[PositiveInt] | None = None


class ClosedLoopTestCaseConfig(ExtensibleModel):
    id: str
    description: str | None = None
    episodes: PositiveInt | None = None
    steps: PositiveInt | None = None
    seed: int | None = None
    start_contract: str | None = None
    task_prompt: str | None = None
    qwen_object: str | None = None
    env_object_color: str | None = None
    success_metric: str | None = None
    start_report_path: str | None = None
    plan_json: str | None = None
    start_dataset: DatasetConfig | None = None


class ClosedLoopConfig(ExtensibleModel):
    runner: Literal["picklift", "qwen_chain"] | None = None
    eval_skill_mode: str | None = None
    execution_policy: str | None = None
    scenario: str | None = None
    env_id: str | None = None
    task_prompt: str | None = None
    qwen_object: str | None = None
    env_object_color: str | None = None
    action_contract_mode: str | None = None
    base_seed: int | None = None
    record_rollout_gif: bool | None = None
    success_metric: str | None = None
    success_threshold: float | None = None
    valid_mask_checkpoint: str | None = None
    action_rmse_sweep: ActionRmseSweepConfig | None = None
    test_cases: list[ClosedLoopTestCaseConfig] | None = None

    @model_validator(mode="after")
    def test_case_ids_are_unique(self) -> ClosedLoopConfig:
        if not self.test_cases:
            return self
        seen: set[str] = set()
        for case in self.test_cases:
            if case.id in seen:
                raise ValueError(f"duplicate closed_loop test case id {case.id!r}")
            seen.add(case.id)
        return self

    @model_validator(mode="after")
    def require_training_debug_evidence(self) -> ClosedLoopConfig:
        if self.action_rmse_sweep is None:
            raise ValueError("closed_loop.action_rmse_sweep is required for every closed-loop training config")
        if self.action_rmse_sweep.enabled is not True:
            raise ValueError("closed_loop.action_rmse_sweep.enabled must be true")
        if not self.action_rmse_sweep.n_action_steps:
            raise ValueError("closed_loop.action_rmse_sweep.n_action_steps must be non-empty")
        return self


class SO101TrainingConfig(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        title="SO101 training run config",
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://physical-ai-agent.local/schemas/so101/training_config.schema.json",
        },
    )

    name: str
    description: str
    scenario: str | None = None
    execution_policy: str | None = None
    task: str
    prompt: str | None = None
    action_mode: Literal["absolute_qpos", "delta_q"] | None = None
    delta_action_source: Any | None = None
    debug_notes: Any | None = None
    train_dataset: DatasetConfig | None = None
    train_datasets: list[DatasetConfig] | None = None
    validation_dataset: DatasetConfig
    camera_contract: CameraContractConfig
    prompt_contract: dict[str, Any] | None = None
    training: TrainingConfig | None = None
    predecoded_image_cache: PredecodedImageCacheConfig | None = None
    tensorboard: TensorBoardConfig
    closed_loop: ClosedLoopConfig | None = None
    augmentation: AugmentationConfig
    action_chunk_consistency: WeightedStepsConfig | None = None
    action_smoothness: ActionSmoothnessConfig | None = None
    action_teacher_importance: dict[str, Any] | None = None
    dataset_generation: dict[str, Any] | None = None
    dataset_generation_augmentation: dict[str, Any] | None = None
    reachable_bin_filter: dict[str, Any] | None = None
    visual_servo: dict[str, Any] | None = None

    @model_validator(mode="after")
    def exactly_one_train_source(self) -> SO101TrainingConfig:
        has_single = self.train_dataset is not None
        has_multi = self.train_datasets is not None
        if has_single == has_multi:
            raise ValueError("define exactly one of train_dataset or train_datasets")
        if self.train_datasets is not None:
            names = [dataset.name for dataset in self.train_datasets if dataset.name]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"duplicate train_datasets names: {duplicates}")
        return self


@dataclass(frozen=True)
class SO101TrainingConfigValidation:
    path: Path
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_so101_training_schema(repo_root: Path | None = None) -> dict[str, Any]:
    _ = repo_root
    return SO101TrainingConfig.model_json_schema(by_alias=True)


def validate_so101_training_config_file(
    path: Path,
    *,
    repo_root: Path | None = None,
    strict: bool = True,
) -> SO101TrainingConfigValidation:
    root = repo_root or Path.cwd()
    config_path = path if path.is_absolute() else root / path
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return SO101TrainingConfigValidation(config_path, [f"{_display_path(config_path, root)}: invalid JSON: {exc}"])
    except OSError as exc:
        return SO101TrainingConfigValidation(config_path, [f"{_display_path(config_path, root)}: cannot read file: {exc}"])
    return SO101TrainingConfigValidation(
        config_path,
        validate_so101_training_config(config, path=config_path, repo_root=root, strict=strict),
    )


def validate_so101_training_config(
    config: dict[str, Any],
    *,
    path: Path | None = None,
    repo_root: Path | None = None,
    strict: bool = True,
) -> list[str]:
    root = repo_root or Path.cwd()
    label = _display_path(path, root) if path is not None else "<config>"
    if not isinstance(config, dict):
        return [f"{label}: config must be a JSON object"]
    if strict:
        return _strict_errors(config, label)
    return _relaxed_errors(config, label)


def validate_so101_training_config_dir(
    config_dir: Path = TRAINING_CONFIG_DIR,
    *,
    repo_root: Path | None = None,
) -> list[SO101TrainingConfigValidation]:
    root = repo_root or Path.cwd()
    directory = config_dir if config_dir.is_absolute() else root / config_dir
    return [
        validate_so101_training_config_file(path, repo_root=root, strict=True)
        for path in sorted(directory.glob("*.json"))
    ]


def parse_so101_training_config(config: dict[str, Any]) -> SO101TrainingConfig:
    return SO101TrainingConfig.model_validate(config)


def _strict_errors(config: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if ("train_dataset" in config) == ("train_datasets" in config):
        errors.append(f"{label}: define exactly one of train_dataset or train_datasets")
    try:
        SO101TrainingConfig.model_validate(config)
    except ValidationError as exc:
        errors.extend(
            error
            for error in _pydantic_errors(exc, label)
            if "define exactly one of train_dataset or train_datasets" not in error
        )
        return errors
    errors.extend(_cross_field_errors(config, label, strict=True))
    return errors


def _relaxed_errors(config: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    allowed_top_keys = set(SO101TrainingConfig.model_fields)
    for key in sorted(set(config) - allowed_top_keys):
        errors.append(f"{label}: unknown top-level key {key!r}; add it to the Pydantic model before using it")

    has_single = "train_dataset" in config
    has_multi = "train_datasets" in config
    if has_single and has_multi:
        errors.append(f"{label}: define exactly one of train_dataset or train_datasets")
    if has_single:
        errors.extend(_validate_model(DatasetConfig, config.get("train_dataset"), f"{label}.train_dataset"))
    if has_multi:
        value = config.get("train_datasets")
        if not isinstance(value, list) or not value:
            errors.append(f"{label}.train_datasets: must be a non-empty list")
        else:
            seen_names: set[str] = set()
            for index, dataset in enumerate(value):
                errors.extend(_validate_model(DatasetConfig, dataset, f"{label}.train_datasets[{index}]"))
                if isinstance(dataset, dict) and dataset.get("name"):
                    name = str(dataset["name"])
                    if name in seen_names:
                        errors.append(f"{label}.train_datasets[{index}]: duplicate dataset name {name!r}")
                    seen_names.add(name)

    relaxed_models: tuple[tuple[str, type[BaseModel]], ...] = (
        ("validation_dataset", DatasetConfig),
        ("camera_contract", CameraContractConfig),
        ("tensorboard", TensorBoardConfig),
        ("training", TrainingConfig),
        ("predecoded_image_cache", PredecodedImageCacheConfig),
        ("augmentation", AugmentationConfig),
        ("closed_loop", ClosedLoopConfig),
        ("action_chunk_consistency", WeightedStepsConfig),
        ("action_smoothness", ActionSmoothnessConfig),
    )
    for key, model in relaxed_models:
        if key in config:
            errors.extend(_validate_model(model, config[key], f"{label}.{key}"))
    errors.extend(_cross_field_errors(config, label, strict=False))
    return errors


def _validate_model(model: type[BaseModel], value: Any, label: str) -> list[str]:
    try:
        model.model_validate(value)
    except ValidationError as exc:
        return _pydantic_errors(exc, label)
    return []


def _cross_field_errors(config: dict[str, Any], label: str, *, strict: bool) -> list[str]:
    errors: list[str] = []
    cache = config.get("predecoded_image_cache")
    train_datasets = config.get("train_datasets")
    if isinstance(cache, dict) and isinstance(cache.get("train"), dict) and isinstance(train_datasets, list):
        known = {str(item.get("name")) for item in train_datasets if isinstance(item, dict) and item.get("name")}
        unknown = sorted(set(cache["train"]) - known)
        for key in unknown:
            errors.append(f"{label}.predecoded_image_cache.train: cache mapping key {key!r} is not a train_datasets name")

    closed_loop = config.get("closed_loop")
    if isinstance(closed_loop, dict):
        has_prompt = bool(closed_loop.get("task_prompt") or config.get("prompt"))
        test_cases = closed_loop.get("test_cases") if isinstance(closed_loop.get("test_cases"), list) else []
        if not has_prompt and not any(isinstance(case, dict) and case.get("task_prompt") for case in test_cases):
            errors.append(
                f"{label}.closed_loop: task_prompt is required unless top-level prompt or per-test-case task_prompt is set"
            )
        for index, case in enumerate(test_cases):
            if not isinstance(case, dict):
                continue
            for key in ("episodes", "steps", "seed"):
                if strict and key not in case:
                    errors.append(f"{label}.closed_loop.test_cases[{index}].{key}: is required")
    return errors


def _pydantic_errors(exc: ValidationError, label: str) -> list[str]:
    errors: list[str] = []
    for error in exc.errors(include_url=False):
        loc = ".".join(str(part) for part in error["loc"])
        suffix = f".{loc}" if loc else ""
        errors.append(f"{label}{suffix}: {error['msg']}")
    return errors


def _display_path(path: Path | None, root: Path) -> str:
    if path is None:
        return "<config>"
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
