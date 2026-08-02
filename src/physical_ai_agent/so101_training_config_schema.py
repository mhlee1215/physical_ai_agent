from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)


SCHEMA_PATH = Path("configs/so101/schemas/training_config.schema.json")
TRAINING_CONFIG_DIR = Path("configs/so101/training")
TRAINING_DEFAULT_CONFIG_DIR = Path("configs/so101/training_defaults")


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
    state_jitter_prob: float | None = Field(default=None, ge=0, le=1)
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


class ActionOverlapConsistencyConfig(ExtensibleModel):
    offset: PositiveInt | None = None
    horizon: PositiveInt | None = None
    weight: NonNegativeFloat | None = None


class ActionRequeryConsistencyConfig(StrictModel):
    offset: PositiveInt
    horizon: PositiveInt
    weight: NonNegativeFloat


class ActionSmoothnessConfig(ExtensibleModel):
    weight: NonNegativeFloat | None = None
    include_gripper: bool | None = None


class ActionWristRollCircularConfig(StrictModel):
    weight: NonNegativeFloat
    joint_index: int = Field(ge=0)
    period_radians: float = Field(gt=0)


class ModelInputsConfig(StrictModel):
    active_image_features: list[
        Literal[
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ]
    ]

    @model_validator(mode="after")
    def active_cameras_are_unique(self) -> ModelInputsConfig:
        if not self.active_image_features:
            raise ValueError("active_image_features must select at least one camera")
        if len(set(self.active_image_features)) != len(self.active_image_features):
            raise ValueError("active_image_features must not contain duplicates")
        return self


class PeftTrainingConfig(StrictModel):
    enabled: bool
    method_type: Literal["LORA"]
    base_model_name_or_path: str
    target_modules: str | list[str]
    full_training_modules: list[str]
    r: PositiveInt


class TrainingDataLoadingConfig(StrictModel):
    predecoded_image_cache: PredecodedImageCacheConfig | None = None
    tensorboard: TensorBoardConfig | None = None


class TrainingLossesConfig(ExtensibleModel):
    action_prefix: WeightedStepsConfig | None = None
    action_chunk_consistency: WeightedStepsConfig | None = None
    action_overlap_consistency: ActionOverlapConsistencyConfig | None = None
    action_requery_consistency: ActionRequeryConsistencyConfig | None = None
    action_smoothness: ActionSmoothnessConfig | None = None
    action_wrist_roll_circular: ActionWristRollCircularConfig | None = None
    action_teacher_importance: dict[str, Any] | None = None


class SO101DatasetSectionConfig(StrictModel):
    train_dataset: DatasetConfig | None = None
    train_datasets: list[DatasetConfig] | None = None
    validation_dataset: DatasetConfig | None = None
    validation_datasets: list[DatasetConfig] | None = None
    camera_contract: CameraContractConfig
    prompt_contract: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None
    generation_augmentation: dict[str, Any] | None = None
    reachable_bin_filter: dict[str, Any] | None = None

    @model_validator(mode="after")
    def exactly_one_train_source(self) -> SO101DatasetSectionConfig:
        _validate_train_source_choice(self.train_dataset, self.train_datasets)
        _validate_validation_source_choice(self.validation_dataset, self.validation_datasets)
        return self


class SO101RuntimeTrainingConfig(StrictModel):
    training: TrainingConfig | None = None
    model_inputs: ModelInputsConfig | None = None
    peft: PeftTrainingConfig | None = None
    data_loading: TrainingDataLoadingConfig | None = None
    losses: TrainingLossesConfig | None = None
    augmentation: AugmentationConfig | None = None
    closed_loop: ClosedLoopConfig | None = None
    visual_servo: dict[str, Any] | None = None


class ActionRmseSweepConfig(ExtensibleModel):
    enabled: bool | None = None
    episodes: PositiveInt | None = None
    tensorboard_tag: str | None = None
    y_axis_max: PositiveFloat | None = None
    n_action_steps: list[PositiveInt] | None = None
    render_policy_inference_only: bool | None = None
    timeline_mode: Literal["start_dataset", "phase_chain"] | None = None
    test_cases: list[str] | None = None
    phase_contract_test_case_id: str | None = None


class TemporalEnsembleConfig(StrictModel):
    enabled: bool
    decay: NonNegativeFloat


class ClosedLoopEnvironmentConfig(StrictModel):
    camera_rig_config: str
    target_object_color: Literal["red", "blue", "green"]
    object_half_sizes: list[float] = Field(min_length=1)
    spawn_center: tuple[float, float]
    spawn_min_radius: NonNegativeFloat
    spawn_max_radius: NonNegativeFloat
    spawn_angle_half_range_deg: NonNegativeFloat

    @model_validator(mode="after")
    def validate_object_and_spawn_contract(self) -> ClosedLoopEnvironmentConfig:
        if any(float(value) <= 0.0 for value in self.object_half_sizes):
            raise ValueError("object_half_sizes values must be positive")
        if float(self.spawn_max_radius) < float(self.spawn_min_radius):
            raise ValueError("spawn_max_radius must be >= spawn_min_radius")
        return self


class ClosedLoopObservationRendererConfig(StrictModel):
    mode: Literal["blender_cycles_live"]
    render_policy_inference_only: bool
    camera_keys: list[Literal["observation.images.camera1", "observation.images.camera2"]]
    width: PositiveInt
    height: PositiveInt
    source_width: PositiveInt | None = None
    source_height: PositiveInt | None = None
    policy_resize: Literal["direct_square_render", "center_crop_square_then_resize"] | None = None
    camera_rig_config: str | None = None
    profile_from_camera_rig: bool | None = None
    samples: PositiveInt
    denoise: bool
    compute_device_type: str | None = None
    cycles_seed: int
    lighting_profile: Literal["studio_small_08", "flat", "directional_key_fill_rim_v4"]
    key_light_power: NonNegativeFloat
    fill_light_power: NonNegativeFloat
    world_strength: NonNegativeFloat
    hdri_rotation_deg: float
    exposure: float
    color_management: Literal["Filmic", "Standard", "AgX"]
    color_look: str
    gamma: float
    output_format: Literal["PNG", "JPEG"]
    sample_clamp_indirect: NonNegativeFloat
    background_wall: bool
    stable_tabletop: bool
    scene_profile: Literal["neutral", "black_table_clutter", "pbr_workbench_v4", "pbr_workshop_v4"]
    robot_material: Literal["plastic", "matte_pla", "metal"]
    material_profile: str
    camera_lens: float
    asset_root: str
    blender_bin: str
    max_mesh_geoms: PositiveInt
    preserve_pinhole_renders: bool | None = None
    bevel_width_range_m: tuple[NonNegativeFloat, NonNegativeFloat] | None = None
    bevel_segments: PositiveInt | None = None
    lens_distortion: dict[str, Any] | None = None
    visual_props: list[dict[str, Any]] | None = None
    lights: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def require_policy_camera_pair(self) -> ClosedLoopObservationRendererConfig:
        expected = ["observation.images.camera1", "observation.images.camera2"]
        if self.camera_keys != expected:
            raise ValueError(f"camera_keys must be exactly {expected}")
        source_fields = (self.source_width, self.source_height, self.policy_resize)
        if any(value is not None for value in source_fields) and not all(value is not None for value in source_fields):
            raise ValueError("source_width, source_height, and policy_resize must be declared together")
        if self.source_width is not None and self.source_height is not None:
            if int(self.source_width) < int(self.width) or int(self.source_height) < int(self.height):
                raise ValueError("source render resolution cannot be smaller than policy output resolution")
            if self.policy_resize == "direct_square_render" and int(self.source_width) != int(self.source_height):
                raise ValueError("direct_square_render requires a square source render")
        if self.profile_from_camera_rig and not self.camera_rig_config:
            raise ValueError("profile_from_camera_rig requires camera_rig_config")
        return self


class ClosedLoopTensorBoardMediaConfig(StrictModel):
    train_reference_frequency: Literal["once_per_run", "every_checkpoint", "disabled"]
    chain_rollout_layout: Literal["per_episode", "combined"] | None = None
    render_test_cases: list[str] | None = None

    @model_validator(mode="after")
    def render_test_case_ids_are_unique(self) -> ClosedLoopTensorBoardMediaConfig:
        if self.render_test_cases is not None and len(set(self.render_test_cases)) != len(
            self.render_test_cases
        ):
            raise ValueError("render_test_cases must not contain duplicates")
        return self


class ClosedLoopPhaseVerifierConfig(StrictModel):
    kind: Literal["approach", "alignment", "grip_lift"]
    tcp_position_tolerance_m: NonNegativeFloat | None = None
    gripper_open_tolerance_rad: NonNegativeFloat | None = None
    edge_xy_tolerance_m: NonNegativeFloat | None = None
    jaw_angle_tolerance_deg: NonNegativeFloat | None = None
    lift_height_m: NonNegativeFloat | None = None
    hold_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_kind_specific_thresholds(self) -> ClosedLoopPhaseVerifierConfig:
        required = {
            "approach": ("tcp_position_tolerance_m", "gripper_open_tolerance_rad"),
            "alignment": (
                "edge_xy_tolerance_m",
                "jaw_angle_tolerance_deg",
                "gripper_open_tolerance_rad",
            ),
            "grip_lift": ("lift_height_m",),
        }[self.kind]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.kind} verifier missing thresholds: {', '.join(missing)}")
        if self.kind == "grip_lift" and self.hold_steps <= 0:
            raise ValueError("grip_lift verifier hold_steps must be positive")
        return self


class ClosedLoopPhaseConfig(StrictModel):
    id: Literal["approach", "alignment", "grip_lift"]
    prompt: str
    max_steps: PositiveInt
    reference_length_multiplier: PositiveFloat | None = None
    reference_report_path: str
    verifier: ClosedLoopPhaseVerifierConfig

    @model_validator(mode="after")
    def verifier_matches_phase(self) -> ClosedLoopPhaseConfig:
        if self.verifier.kind != self.id:
            raise ValueError(f"phase {self.id!r} requires verifier kind {self.id!r}")
        return self


class ClosedLoopPhaseContractConfig(StrictModel):
    mode: Literal["primitive", "chain"]
    handoff_mode: Literal["continuous", "oracle_reset"]
    phases: list[ClosedLoopPhaseConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_phase_sequence(self) -> ClosedLoopPhaseContractConfig:
        ids = [phase.id for phase in self.phases]
        expected = ["approach", "alignment", "grip_lift"]
        if self.mode == "primitive" and len(ids) != 1:
            raise ValueError("primitive phase contract must contain exactly one phase")
        if self.mode == "chain" and ids != expected:
            raise ValueError(f"chain phase contract must use {expected}")
        if self.mode == "primitive" and self.handoff_mode != "continuous":
            raise ValueError("primitive phase contract handoff_mode must be continuous")
        return self


class ClosedLoopTestCaseConfig(ExtensibleModel):
    id: str
    description: str | None = None
    schedule: Literal["periodic", "final", "manual"] = "periodic"
    episodes: PositiveInt | None = None
    episode_indices: list[int] | None = None
    source_grid_bins: list[int] | None = None
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
    env_config: ClosedLoopEnvironmentConfig | None = None
    phase_contract: ClosedLoopPhaseContractConfig | None = None

    @model_validator(mode="after")
    def validate_phase_test_contract(self) -> ClosedLoopTestCaseConfig:
        if self.episode_indices is not None:
            if any(index < 0 for index in self.episode_indices):
                raise ValueError("episode_indices must be non-negative")
            if len(set(self.episode_indices)) != len(self.episode_indices):
                raise ValueError("episode_indices must not contain duplicates")
            if self.episodes is not None and len(self.episode_indices) != self.episodes:
                raise ValueError("episodes must equal len(episode_indices)")
        if (
            self.phase_contract is not None
            and self.phase_contract.handoff_mode == "oracle_reset"
            and self.schedule != "manual"
        ):
            raise ValueError("oracle_reset phase contracts are manual diagnostics only")
        return self


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
    temporal_ensemble: TemporalEnsembleConfig | None = None
    observation_renderer: ClosedLoopObservationRendererConfig | None = None
    tensorboard_media: ClosedLoopTensorBoardMediaConfig | None = None
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
        render_test_cases = (
            self.tensorboard_media.render_test_cases
            if self.tensorboard_media is not None
            else None
        )
        unknown_render_cases = sorted(set(render_test_cases or []) - seen)
        if unknown_render_cases:
            raise ValueError(
                "tensorboard_media.render_test_cases contains unknown test ids: "
                + ", ".join(unknown_render_cases)
            )
        if self.action_rmse_sweep is not None:
            unknown_rmse_cases = sorted(
                set(self.action_rmse_sweep.test_cases or []) - seen
            )
            if unknown_rmse_cases:
                raise ValueError(
                    "action_rmse_sweep.test_cases contains unknown test ids: "
                    + ", ".join(unknown_rmse_cases)
                )
            source_case_id = self.action_rmse_sweep.phase_contract_test_case_id
            if source_case_id is not None and source_case_id not in seen:
                raise ValueError(
                    "action_rmse_sweep.phase_contract_test_case_id contains unknown test id: "
                    + source_case_id
                )
            if self.action_rmse_sweep.timeline_mode == "phase_chain":
                if source_case_id is None:
                    raise ValueError(
                        "phase_chain action RMSE requires phase_contract_test_case_id"
                    )
                source_case = next(case for case in self.test_cases if case.id == source_case_id)
                if (
                    source_case.phase_contract is None
                    or source_case.phase_contract.mode != "chain"
                ):
                    raise ValueError(
                        "phase_chain action RMSE source test case must use a chain phase contract"
                    )
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
    default_config: str | None = None
    scenario: str | None = None
    execution_policy: str | None = None
    task: str
    prompt: str | None = None
    action_mode: Literal["absolute_qpos", "delta_q"] | None = None
    delta_action_source: Any | None = None
    debug_notes: Any | None = None
    dataset: SO101DatasetSectionConfig | None = None
    training_config: SO101RuntimeTrainingConfig | None = None
    train_dataset: DatasetConfig | None = None
    train_datasets: list[DatasetConfig] | None = None
    validation_dataset: DatasetConfig | None = None
    validation_datasets: list[DatasetConfig] | None = None
    camera_contract: CameraContractConfig
    prompt_contract: dict[str, Any] | None = None
    training: TrainingConfig | None = None
    predecoded_image_cache: PredecodedImageCacheConfig | None = None
    tensorboard: TensorBoardConfig
    closed_loop: ClosedLoopConfig | None = None
    augmentation: AugmentationConfig
    action_prefix: WeightedStepsConfig | None = None
    action_chunk_consistency: WeightedStepsConfig | None = None
    action_overlap_consistency: ActionOverlapConsistencyConfig | None = None
    action_requery_consistency: ActionRequeryConsistencyConfig | None = None
    action_smoothness: ActionSmoothnessConfig | None = None
    action_wrist_roll_circular: ActionWristRollCircularConfig | None = None
    action_teacher_importance: dict[str, Any] | None = None
    model_inputs: ModelInputsConfig | None = None
    peft: PeftTrainingConfig | None = None
    dataset_generation: dict[str, Any] | None = None
    dataset_generation_augmentation: dict[str, Any] | None = None
    reachable_bin_filter: dict[str, Any] | None = None
    visual_servo: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def expand_structured_sections(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return normalize_so101_training_config(value)
        return value

    @model_validator(mode="after")
    def exactly_one_train_source(self) -> SO101TrainingConfig:
        _validate_train_source_choice(self.train_dataset, self.train_datasets)
        _validate_validation_source_choice(self.validation_dataset, self.validation_datasets)
        renderer = self.closed_loop.observation_renderer if self.closed_loop is not None else None
        if renderer is not None:
            validation_specs = self.validation_datasets or (
                [self.validation_dataset] if self.validation_dataset is not None else []
            )
            validation_roots = {
                str(dataset.root or "")
                for dataset in validation_specs
                if dataset.root
            }
            if not validation_roots or any("photoreal" not in root for root in validation_roots):
                raise ValueError(
                    "closed_loop.observation_renderer=blender_cycles_live requires photoreal validation datasets"
                )
            for index, case in enumerate(self.closed_loop.test_cases or []):
                start_root = str(case.start_dataset.root or "") if case.start_dataset is not None else ""
                if start_root not in validation_roots:
                    raise ValueError(
                        f"closed_loop.test_cases[{index}].start_dataset.root must match a validation dataset root"
                    )
                report_path = str(case.start_report_path or "")
                if not any(report_path.startswith(root.rstrip("/") + "/") for root in validation_roots):
                    raise ValueError(
                        f"closed_loop.test_cases[{index}].start_report_path must be inside a validation dataset root"
                    )
                if case.phase_contract is not None:
                    for phase in case.phase_contract.phases:
                        if not any(
                            phase.reference_report_path.startswith(root.rstrip("/") + "/")
                            for root in validation_roots
                        ):
                            raise ValueError(
                                f"closed_loop.test_cases[{index}] phase {phase.id!r} reference report "
                                "must be inside a validation dataset root"
                            )
                if renderer.camera_rig_config is not None and case.env_config is not None:
                    if case.env_config.camera_rig_config != renderer.camera_rig_config:
                        raise ValueError(
                            f"closed_loop.test_cases[{index}].env_config.camera_rig_config must match "
                            "closed_loop.observation_renderer.camera_rig_config"
                        )
                if case.env_config is not None and case.env_object_color is not None:
                    if case.env_config.target_object_color != case.env_object_color:
                        raise ValueError(
                            f"closed_loop.test_cases[{index}].env_object_color must match "
                            "env_config.target_object_color"
                        )
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
    try:
        config = resolve_so101_training_config_defaults(config, path=path, repo_root=root)
    except ValueError as exc:
        return [f"{label}: {exc}"]
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


def parse_so101_training_config(
    config: dict[str, Any],
    *,
    path: Path | None = None,
    repo_root: Path | None = None,
) -> SO101TrainingConfig:
    resolved = resolve_so101_training_config_defaults(config, path=path, repo_root=repo_root)
    return SO101TrainingConfig.model_validate(normalize_so101_training_config(resolved))


def resolve_so101_training_config_defaults(
    config: dict[str, Any],
    *,
    path: Path | None = None,
    repo_root: Path | None = None,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Merge a dataset-specific training config with its default recipe.

    ``default_config`` is for non-dataset training defaults only. Dataset roots,
    train/validation splits, camera contract, and dataset generation metadata
    stay in the concrete config.
    """

    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    default_ref = config.get("default_config")
    if not default_ref:
        return json.loads(json.dumps(config))
    root = repo_root or Path.cwd()
    default_path = _resolve_default_config_path(str(default_ref), path=path, repo_root=root)
    seen = set(_seen or set())
    if default_path in seen:
        raise ValueError(f"default_config cycle detected at {_display_path(default_path, root)}")
    seen.add(default_path)
    try:
        default_payload = json.loads(default_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"default_config {_display_path(default_path, root)} invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"default_config {_display_path(default_path, root)} cannot be read: {exc}") from exc
    if not isinstance(default_payload, dict):
        raise ValueError(f"default_config {_display_path(default_path, root)} must be a JSON object")
    _reject_dataset_keys_in_default(default_payload, default_path, root)
    default_payload = resolve_so101_training_config_defaults(
        default_payload,
        path=default_path,
        repo_root=root,
        _seen=seen,
    )
    return _deep_merge(default_payload, config)


def _resolve_default_config_path(default_ref: str, *, path: Path | None, repo_root: Path) -> Path:
    ref_path = Path(default_ref)
    if ref_path.is_absolute():
        return ref_path
    repo_relative = repo_root / ref_path
    if repo_relative.exists():
        return repo_relative.resolve()
    if path is not None:
        config_dir_relative = ((path if path.is_absolute() else repo_root / path).parent / ref_path).resolve()
        if config_dir_relative.exists():
            return config_dir_relative
    defaults_relative = (repo_root / TRAINING_DEFAULT_CONFIG_DIR / ref_path).resolve()
    return defaults_relative


def _reject_dataset_keys_in_default(default_payload: dict[str, Any], default_path: Path, repo_root: Path) -> None:
    dataset_keys = {
        "dataset",
        "train_dataset",
        "train_datasets",
        "validation_dataset",
        "validation_datasets",
        "camera_contract",
        "prompt_contract",
        "dataset_generation",
        "dataset_generation_augmentation",
        "reachable_bin_filter",
    }
    present = sorted(key for key in dataset_keys if key in default_payload)
    if present:
        raise ValueError(
            f"default_config {_display_path(default_path, repo_root)} contains dataset fields: {', '.join(present)}"
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def normalize_so101_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the flat runtime shape used by existing SO101 launch code.

    New configs may keep dataset facts under ``dataset`` and runtime knobs under
    ``training_config``. The launcher still consumes the historical flat keys, so
    this function is the single compatibility bridge.
    """

    normalized = json.loads(json.dumps(config))
    dataset_section = normalized.get("dataset")
    if isinstance(dataset_section, dict):
        _copy_if_absent(normalized, "train_dataset", dataset_section.get("train_dataset"))
        _copy_if_absent(normalized, "train_datasets", dataset_section.get("train_datasets"))
        _copy_if_absent(normalized, "validation_dataset", dataset_section.get("validation_dataset"))
        _copy_if_absent(normalized, "validation_datasets", dataset_section.get("validation_datasets"))
        _copy_if_absent(normalized, "camera_contract", dataset_section.get("camera_contract"))
        _copy_if_absent(normalized, "prompt_contract", dataset_section.get("prompt_contract"))
        _copy_if_absent(normalized, "dataset_generation", dataset_section.get("generation"))
        _copy_if_absent(normalized, "dataset_generation_augmentation", dataset_section.get("generation_augmentation"))
        _copy_if_absent(normalized, "reachable_bin_filter", dataset_section.get("reachable_bin_filter"))

    training_section = normalized.get("training_config")
    if isinstance(training_section, dict):
        _copy_if_absent(normalized, "training", training_section.get("training"))
        _copy_if_absent(normalized, "model_inputs", training_section.get("model_inputs"))
        _copy_if_absent(normalized, "peft", training_section.get("peft"))
        _copy_if_absent(normalized, "augmentation", training_section.get("augmentation"))
        _copy_if_absent(normalized, "closed_loop", training_section.get("closed_loop"))
        _copy_if_absent(normalized, "visual_servo", training_section.get("visual_servo"))
        data_loading = training_section.get("data_loading")
        if isinstance(data_loading, dict):
            _copy_if_absent(normalized, "predecoded_image_cache", data_loading.get("predecoded_image_cache"))
            _copy_if_absent(normalized, "tensorboard", data_loading.get("tensorboard"))
        losses = training_section.get("losses")
        if isinstance(losses, dict):
            _copy_if_absent(normalized, "action_prefix", losses.get("action_prefix"))
            _copy_if_absent(normalized, "action_chunk_consistency", losses.get("action_chunk_consistency"))
            _copy_if_absent(normalized, "action_overlap_consistency", losses.get("action_overlap_consistency"))
            _copy_if_absent(normalized, "action_requery_consistency", losses.get("action_requery_consistency"))
            _copy_if_absent(normalized, "action_smoothness", losses.get("action_smoothness"))
            _copy_if_absent(normalized, "action_wrist_roll_circular", losses.get("action_wrist_roll_circular"))
            _copy_if_absent(normalized, "action_teacher_importance", losses.get("action_teacher_importance"))
    return normalized


def _copy_if_absent(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and key not in target:
        target[key] = value


def _validate_train_source_choice(
    train_dataset: DatasetConfig | None,
    train_datasets: list[DatasetConfig] | None,
) -> None:
    has_single = train_dataset is not None
    has_multi = train_datasets is not None
    if has_single == has_multi:
        raise ValueError("define exactly one of train_dataset or train_datasets")
    if train_datasets is not None:
        names = [dataset.name for dataset in train_datasets if dataset.name]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate train_datasets names: {duplicates}")


def _validate_validation_source_choice(
    validation_dataset: DatasetConfig | None,
    validation_datasets: list[DatasetConfig] | None,
) -> None:
    has_single = validation_dataset is not None
    has_multi = validation_datasets is not None
    if has_single == has_multi:
        raise ValueError("define exactly one of validation_dataset or validation_datasets")
    if validation_datasets is not None:
        if not validation_datasets:
            raise ValueError("validation_datasets must be non-empty")
        names = [dataset.name for dataset in validation_datasets if dataset.name]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate validation_datasets names: {duplicates}")


def _strict_errors(config: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    normalized = normalize_so101_training_config(config)
    if ("train_dataset" in normalized) == ("train_datasets" in normalized):
        errors.append(f"{label}: define exactly one of train_dataset or train_datasets")
    if ("validation_dataset" in normalized) == ("validation_datasets" in normalized):
        errors.append(f"{label}: define exactly one of validation_dataset or validation_datasets")
    try:
        SO101TrainingConfig.model_validate(normalized)
    except ValidationError as exc:
        errors.extend(
            error
            for error in _pydantic_errors(exc, label)
            if "define exactly one of train_dataset or train_datasets" not in error
            and "define exactly one of validation_dataset or validation_datasets" not in error
        )
        return errors
    errors.extend(_cross_field_errors(normalized, label, strict=True))
    return errors


def _relaxed_errors(config: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    normalized = normalize_so101_training_config(config)
    allowed_top_keys = set(SO101TrainingConfig.model_fields)
    for key in sorted(set(config) - allowed_top_keys):
        errors.append(f"{label}: unknown top-level key {key!r}; add it to the Pydantic model before using it")

    has_single = "train_dataset" in normalized
    has_multi = "train_datasets" in normalized
    if has_single and has_multi:
        errors.append(f"{label}: define exactly one of train_dataset or train_datasets")
    if has_single:
        errors.extend(_validate_model(DatasetConfig, normalized.get("train_dataset"), f"{label}.train_dataset"))
    if has_multi:
        value = normalized.get("train_datasets")
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

    has_single_validation = "validation_dataset" in normalized
    has_multi_validation = "validation_datasets" in normalized
    if has_single_validation and has_multi_validation:
        errors.append(f"{label}: define exactly one of validation_dataset or validation_datasets")
    if has_single_validation:
        errors.extend(
            _validate_model(
                DatasetConfig,
                normalized.get("validation_dataset"),
                f"{label}.validation_dataset",
            )
        )
    if has_multi_validation:
        value = normalized.get("validation_datasets")
        if not isinstance(value, list) or not value:
            errors.append(f"{label}.validation_datasets: must be a non-empty list")
        else:
            seen_names: set[str] = set()
            for index, dataset in enumerate(value):
                errors.extend(
                    _validate_model(
                        DatasetConfig,
                        dataset,
                        f"{label}.validation_datasets[{index}]",
                    )
                )
                if isinstance(dataset, dict) and dataset.get("name"):
                    name = str(dataset["name"])
                    if name in seen_names:
                        errors.append(f"{label}.validation_datasets[{index}]: duplicate dataset name {name!r}")
                    seen_names.add(name)

    relaxed_models: tuple[tuple[str, type[BaseModel]], ...] = (
        ("camera_contract", CameraContractConfig),
        ("tensorboard", TensorBoardConfig),
        ("training", TrainingConfig),
        ("predecoded_image_cache", PredecodedImageCacheConfig),
        ("augmentation", AugmentationConfig),
        ("closed_loop", ClosedLoopConfig),
        ("action_prefix", WeightedStepsConfig),
        ("action_chunk_consistency", WeightedStepsConfig),
        ("action_overlap_consistency", ActionOverlapConsistencyConfig),
        ("action_requery_consistency", ActionRequeryConsistencyConfig),
        ("action_smoothness", ActionSmoothnessConfig),
        ("action_wrist_roll_circular", ActionWristRollCircularConfig),
        ("model_inputs", ModelInputsConfig),
        ("peft", PeftTrainingConfig),
    )
    for key, model in relaxed_models:
        if key in normalized:
            errors.extend(_validate_model(model, normalized[key], f"{label}.{key}"))
    errors.extend(_cross_field_errors(normalized, label, strict=False))
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
    validation_datasets = config.get("validation_datasets")
    if (
        isinstance(cache, dict)
        and isinstance(cache.get("validation"), dict)
        and isinstance(validation_datasets, list)
    ):
        known = {
            str(item.get("name"))
            for item in validation_datasets
            if isinstance(item, dict) and item.get("name")
        }
        unknown = sorted(set(cache["validation"]) - known)
        for key in unknown:
            errors.append(
                f"{label}.predecoded_image_cache.validation: cache mapping key {key!r} "
                "is not a validation_datasets name"
            )

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
