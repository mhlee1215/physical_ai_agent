"""Pydantic contract for recipe-backed SO101 dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExporterCommonSpec(StrictModel):
    fps: int = Field(gt=0)
    width: int = Field(default=256, gt=0)
    height: int = Field(default=256, gt=0)
    capture_render_replay: bool = False
    teacher_style: str | None = None
    approach_steps: int | None = Field(default=None, ge=0)
    settle_steps: int | None = Field(default=None, ge=0)
    close_steps: int | None = Field(default=None, ge=0)
    lift_steps: int | None = Field(default=None, ge=0)
    lift_target_height: float | None = None
    lift_controller_z_error: float | None = Field(default=None, ge=0)
    trajectory_variant: str | None = None
    start_mode: str | None = None
    grip_the_cube_start_profile: str | None = None
    move_target_z_offset: float | None = None
    near_target_joint_std: float | None = Field(default=None, ge=0)
    near_target_xy_std: float | None = Field(default=None, ge=0)
    close_alignment_gate_mode: str | None = None
    skill_mode: str | None = None
    terminal_hold_steps: int | None = Field(default=None, ge=0)
    edge_contact_xy_success_threshold: float | None = Field(default=None, ge=0)
    edge_contact_parallel_success_threshold_deg: float | None = Field(default=None, ge=0)
    max_attempt_multiplier: int | None = Field(default=None, gt=0)
    grid_balance_size: int | None = Field(default=None, gt=0)
    grid_balance_spawn_lookup: bool = False
    grid_lookup_preserve_order: bool = False
    grid_lookup_x_min: float | None = None
    grid_lookup_x_max: float | None = None
    grid_lookup_y_min: float | None = None
    grid_lookup_y_max: float | None = None
    grid_lookup_resolution: int | None = Field(default=None, gt=0)
    deterministic_camera_bin_lookup: bool = False
    target_object_color: str | None = None
    spawn_center_x: float | None = None
    spawn_center_y: float | None = None
    spawn_min_radius: float | None = Field(default=None, ge=0)
    spawn_max_radius: float | None = Field(default=None, ge=0)
    spawn_angle_half_range_deg: float | None = Field(default=None, ge=0)
    object_half_sizes: str | None = None

    @model_validator(mode="after")
    def validate_common_contract(self) -> ExporterCommonSpec:
        if (self.width, self.height) != (256, 256):
            raise ValueError("SO101 dataset cameras must be 256x256")
        if (
            self.spawn_min_radius is not None
            and self.spawn_max_radius is not None
            and self.spawn_max_radius < self.spawn_min_radius
        ):
            raise ValueError("spawn radius range is invalid")
        return self


class BinSpec(StrictModel):
    id: int = Field(ge=0)
    episodes: int = Field(gt=0)
    seed: int = Field(ge=0)
    lookup_start_index: int = Field(ge=0)
    shard: str | None = None


class ClosedLoopSpec(StrictModel):
    episodes: int = Field(gt=0)
    bins: list[int] = Field(min_length=1)
    output: str
    success_metric: str | None = None
    lift_success_height: float | None = None
    exclude_source_reports: list[str] = Field(default_factory=list)


class ObjectPoolEntry(StrictModel):
    slot: int = Field(ge=0)
    color: str
    half_size: float = Field(gt=0)


class RenderEnvironmentSpec(StrictModel):
    factory: Literal["make_high_contrast_picklift_env"] = "make_high_contrast_picklift_env"
    target_object_color: str = "green"
    object_half_sizes: list[float] = Field(default_factory=lambda: [0.0125, 0.015, 0.0175])
    spawn_center: list[float] = Field(
        default_factory=lambda: [0.15, 0.0], min_length=2, max_length=2
    )
    spawn_min_radius: float = 0.10
    spawn_max_radius: float = 0.30
    spawn_angle_half_range_deg: float = 90.0
    n_distractors: int = Field(default=0, ge=0)
    action_repeat: int = Field(default=1, gt=0)
    object_pool_order: list[ObjectPoolEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges(self) -> RenderEnvironmentSpec:
        if not self.object_half_sizes or any(value <= 0 for value in self.object_half_sizes):
            raise ValueError("object_half_sizes must contain positive values")
        if self.spawn_min_radius < 0 or self.spawn_max_radius < self.spawn_min_radius:
            raise ValueError("spawn radius range is invalid")
        if self.object_pool_order:
            slots = [entry.slot for entry in self.object_pool_order]
            if len(slots) != len(set(slots)):
                raise ValueError("object_pool_order slots must be unique")
            if [entry.half_size for entry in self.object_pool_order] != self.object_half_sizes:
                raise ValueError("object_pool_order half sizes must match object_half_sizes")
        return self


class RenderReplaySpec(StrictModel):
    enabled: bool = True
    output_dir: str = "render_replay"
    state_spec: Literal["mjSTATE_INTEGRATION"] = "mjSTATE_INTEGRATION"
    store_geom_world_transforms: bool = True
    store_camera_world_transforms: bool = True
    environment: RenderEnvironmentSpec = Field(default_factory=RenderEnvironmentSpec)


class RenderProfileSpec(StrictModel):
    mode: Literal["blender_cycles"] = "blender_cycles"
    output_dir: str
    material_profile: str | None = None
    scene_profile: Literal["neutral", "black_table_clutter"] = "neutral"
    asset_root: str = "_workspace/photoreal_assets"
    blender_bin: str = "blender"
    width: int = Field(default=256, gt=0)
    height: int = Field(default=256, gt=0)
    samples: int = Field(default=32, gt=0)
    denoise: bool = True
    cycles_seed: int = 98200
    camera_keys: list[str] = Field(
        default_factory=lambda: ["observation.images.camera1", "observation.images.camera2"],
        min_length=1,
    )
    duplicate_camera3_from_camera2: bool = True
    blender_batch_size: int = Field(default=4, gt=0)
    robot_material: Literal["plastic", "matte_pla", "metal"] = "matte_pla"
    lighting_profile: Literal["studio_small_08", "flat"] = "studio_small_08"
    key_light_power: float = Field(default=42.0, ge=0)
    fill_light_power: float = Field(default=5.0, ge=0)
    world_strength: float = Field(default=0.28, ge=0)
    hdri_rotation_deg: float = 35.0
    exposure: float = -1.3
    color_management: Literal["Filmic", "Standard", "AgX"] = "Filmic"
    color_look: str = "Medium High Contrast"
    gamma: float = Field(default=1.0, gt=0)
    output_format: Literal["PNG", "JPEG"] = "PNG"

    @model_validator(mode="after")
    def enforce_policy_resolution(self) -> RenderProfileSpec:
        if (self.width, self.height) != (256, 256):
            raise ValueError("SO101 rendered training derivatives must be 256x256")
        return self


class SplitSpec(StrictModel):
    kind: Literal["generated", "render_derivative"] = "generated"
    output_root: str
    repo_id: str
    bins: list[BinSpec] = Field(default_factory=list)
    lookup_cache: str | None = None
    closed_loop: ClosedLoopSpec | None = None
    source_split: str | None = None
    render: RenderProfileSpec | None = None
    expected_episodes: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_kind(self) -> SplitSpec:
        if self.kind == "generated" and not self.bins:
            raise ValueError("generated split requires bins")
        if self.kind == "render_derivative":
            if not self.source_split or self.render is None:
                raise ValueError("render_derivative split requires source_split and render")
            if self.bins:
                raise ValueError("render_derivative split must not declare bins")
        return self


class LookupBuilderSpec(StrictModel):
    name: str
    output: str
    source_reports: list[str] = Field(min_length=1)
    grid_size: int = Field(gt=0)
    resolution: int = Field(gt=0)
    x_range: list[float] = Field(min_length=2, max_length=2)
    y_range: list[float] = Field(min_length=2, max_length=2)
    bins: list[int] = Field(min_length=1)
    candidate_start_index: int | None = Field(default=None, ge=0)


class SidecarSpec(StrictModel):
    camera_key: str
    grid_size: int = Field(gt=0)
    frame_index: int = Field(ge=0)
    min_area: int = Field(ge=0)
    bin_source: str | None = None


class AuditSpec(StrictModel):
    expected_prompt: str
    expected_resolution: list[int] = Field(min_length=2, max_length=2)
    expected_min_lift_height: float | None = None
    expected_min_lift_steps: int | None = None
    terminal_hold_action_tolerance: float | None = None


class OverlapAuditSpec(StrictModel):
    name: str
    reference_root: str
    reference_bins: dict[str, int]
    output: str


class DatasetGenerationRecipe(StrictModel):
    schema_version: Literal[1]
    name: str
    version: str | None = None
    description: str
    source_datasets: list[str] = Field(default_factory=list)
    exporter_revision: str
    exporter: str
    lookup_builder_script: str | None = None
    merge_script: str
    sidecar_script: str
    closed_loop_script: str | None = None
    audit_script: str
    render_replay_script: str = "scripts/build_so101_render_replay_sidecar.py"
    photoreal_builder_script: str = "scripts/build_so101_photoreal_lerobot_dataset.py"
    lookup_cache: str
    lookup_builders: list[LookupBuilderSpec] = Field(default_factory=list)
    common: ExporterCommonSpec
    sidecar: SidecarSpec
    render_replay: RenderReplaySpec | None = None
    splits: dict[str, SplitSpec]
    audit: AuditSpec
    overlap_audits: list[OverlapAuditSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> DatasetGenerationRecipe:
        if not self.splits:
            raise ValueError("recipe must define at least one split")
        for name, split in self.splits.items():
            if split.kind == "render_derivative":
                source = self.splits.get(split.source_split or "")
                if source is None or source.kind != "generated":
                    raise ValueError(f"{name}.source_split must reference a generated split")
                expected = sum(item.episodes for item in source.bins)
                if split.expected_episodes not in (None, expected):
                    raise ValueError(
                        f"{name}.expected_episodes must match source split ({expected})"
                    )
                if self.render_replay is None or not self.render_replay.enabled:
                    raise ValueError("render_derivative requires enabled render_replay")
        environment = self.render_replay.environment if self.render_replay else None
        if environment is not None:
            color = self.common.target_object_color
            if color is not None and str(color) != environment.target_object_color:
                raise ValueError("common.target_object_color must match render_replay.environment")
            if self.common.object_half_sizes is not None:
                common_sizes = [
                    float(value)
                    for value in self.common.object_half_sizes.split(",")
                    if value.strip()
                ]
                if common_sizes != environment.object_half_sizes:
                    raise ValueError(
                        "common.object_half_sizes must match render_replay.environment"
                    )
            for key, expected in (
                ("spawn_min_radius", environment.spawn_min_radius),
                ("spawn_max_radius", environment.spawn_max_radius),
                ("spawn_angle_half_range_deg", environment.spawn_angle_half_range_deg),
            ):
                value = getattr(self.common, key)
                if value is not None and float(value) != float(expected):
                    raise ValueError(f"common.{key} must match render_replay.environment")
            center = (self.common.spawn_center_x, self.common.spawn_center_y)
            if all(value is not None for value in center) and [
                float(value) for value in center
            ] != list(environment.spawn_center):
                raise ValueError("common spawn center must match render_replay.environment")
        if any(split.kind == "render_derivative" for split in self.splits.values()):
            if not self.common.capture_render_replay:
                raise ValueError("render_derivative requires common.capture_render_replay=true")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def load_dataset_generation_recipe(path: Path) -> DatasetGenerationRecipe:
    return DatasetGenerationRecipe.model_validate_json(path.read_text(encoding="utf-8"))


def write_dataset_generation_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(DatasetGenerationRecipe.model_json_schema(), indent=2), encoding="utf-8"
    )
    return path
