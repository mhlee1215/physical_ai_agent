"""Pydantic contract for recipe-backed SO101 dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FromScratchSourceSpec(StrictModel):
    mode: Literal["from_scratch"]


class FromExistingDatasetSourceSpec(StrictModel):
    mode: Literal["from_existing_dataset"]
    datasets: list[str] = Field(min_length=1)
    operation: Literal["regenerate_teacher", "render_derivative", "episode_subset"]


class FromSpawnCatalogSourceSpec(StrictModel):
    mode: Literal["from_spawn_catalog"]
    catalogs: list[str] = Field(min_length=1)


DatasetSourceSpec = Annotated[
    FromScratchSourceSpec | FromExistingDatasetSourceSpec | FromSpawnCatalogSourceSpec,
    Field(discriminator="mode"),
]


class HardwareStartPoseSpec(StrictModel):
    contract: Literal[
        "lerobot_calibrated_so101_position_to_mujoco_qpos",
        "camera_image_aligned_so101_mujoco_qpos",
    ]
    readback_artifact: str
    calibration_artifact: str
    joint_order: list[str] = Field(min_length=6, max_length=6)
    raw_positions: list[float] = Field(min_length=6, max_length=6)
    lerobot_positions: list[float] = Field(min_length=6, max_length=6)
    sim_qpos: list[float] = Field(min_length=6, max_length=6)
    camera_rig_config: str | None = None
    image_reference_artifacts: list[str] = Field(default_factory=list)
    alignment_method: Literal["manual_multiview_silhouette_alignment"] | None = None

    @model_validator(mode="after")
    def validate_joint_order(self) -> HardwareStartPoseSpec:
        expected = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        if self.joint_order != expected:
            raise ValueError(f"hardware start pose joint_order must be {expected}")
        if self.contract == "camera_image_aligned_so101_mujoco_qpos":
            if not self.camera_rig_config:
                raise ValueError("image-aligned start pose requires camera_rig_config")
            if len(self.image_reference_artifacts) < 2:
                raise ValueError(
                    "image-aligned start pose requires camera1 and camera2 reference artifacts"
                )
            if self.alignment_method is None:
                raise ValueError("image-aligned start pose requires alignment_method")
        return self


class Camera2CloseTraceSpec(StrictModel):
    mode: Literal["strict_image_trace", "preclose_and_early_trace"]
    pre_close_max_deg: float = Field(gt=0, le=90)
    close_25_max_deg: float = Field(gt=0, le=90)
    close_50_max_deg: float = Field(gt=0, le=90)
    close_75_max_deg: float | None = Field(default=None, gt=0, le=90)

    @model_validator(mode="after")
    def validate_trace_contract(self) -> Camera2CloseTraceSpec:
        if self.mode == "strict_image_trace" and self.close_75_max_deg is None:
            raise ValueError("strict_image_trace requires close_75_max_deg")
        if self.mode == "preclose_and_early_trace" and self.close_75_max_deg is not None:
            raise ValueError("preclose_and_early_trace must omit close_75_max_deg")
        return self


class ContactAlignmentSpec(StrictModel):
    contract: Literal["jaw_line_vs_contact_face_normal_through_cube_center"]
    max_pre_close_error_deg: float = Field(gt=0, le=90)
    camera2_trace: Camera2CloseTraceSpec | None = None


class GeometryContactAlignmentGateSpec(StrictModel):
    kind: Literal["geometry_contact_alignment"]
    contract: Literal["jaw_line_vs_contact_face_normal_through_cube_center"]
    max_pre_close_error_deg: float = Field(gt=0, le=90)


class Camera2VisualAlignmentGateSpec(StrictModel):
    kind: Literal["camera2_visual_alignment"]
    camera_key: Literal["observation.images.camera2"]
    edge_mode: Literal["top_contact"]
    strategy: Literal["constructive_refine_then_probe"]
    mode: Literal["strict_image_trace", "preclose_and_early_trace"]
    pre_close_max_deg: float = Field(gt=0, le=90)
    close_25_max_deg: float = Field(gt=0, le=90)
    close_50_max_deg: float = Field(gt=0, le=90)
    close_75_max_deg: float | None = Field(default=None, gt=0, le=90)

    @model_validator(mode="after")
    def validate_trace_contract(self) -> Camera2VisualAlignmentGateSpec:
        if self.mode == "strict_image_trace" and self.close_75_max_deg is None:
            raise ValueError("strict_image_trace requires close_75_max_deg")
        if self.mode == "preclose_and_early_trace" and self.close_75_max_deg is not None:
            raise ValueError("preclose_and_early_trace must omit close_75_max_deg")
        return self


class GripperFloorClearanceGateSpec(StrictModel):
    kind: Literal["gripper_floor_clearance"]
    min_clearance_m: float = Field(gt=0.0)
    geom_scope: Literal["all_gripper_collision_geoms"] = "all_gripper_collision_geoms"


InspectionGateSpec = Annotated[
    GeometryContactAlignmentGateSpec
    | Camera2VisualAlignmentGateSpec
    | GripperFloorClearanceGateSpec,
    Field(discriminator="kind"),
]


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
    contact_alignment: ContactAlignmentSpec | None = None
    inspection_gates: list[InspectionGateSpec] = Field(default_factory=list)
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
    target_object_yaw_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    object_half_sizes: str | None = None
    camera_rig_config: str | None = None

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
        if self.contact_alignment is not None and (
            self.close_alignment_gate_mode is not None
            or self.edge_contact_parallel_success_threshold_deg is not None
            or self.inspection_gates
        ):
            raise ValueError(
                "contact_alignment replaces close_alignment_gate_mode and "
                "edge_contact_parallel_success_threshold_deg, and cannot be combined "
                "with inspection_gates"
            )
        gate_kinds = [gate.kind for gate in self.inspection_gates]
        if len(gate_kinds) != len(set(gate_kinds)):
            raise ValueError("inspection_gates must not repeat a gate kind")
        if self.inspection_gates and (
            self.close_alignment_gate_mode is not None
            or self.edge_contact_parallel_success_threshold_deg is not None
        ):
            raise ValueError(
                "inspection_gates replace close_alignment_gate_mode and "
                "edge_contact_parallel_success_threshold_deg"
            )
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
    target_object_yaw_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    n_distractors: int = Field(default=0, ge=0)
    action_repeat: int = Field(default=1, gt=0)
    object_pool_order: list[ObjectPoolEntry] = Field(default_factory=list)
    camera_rig_config: str | None = None
    camera_rig_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

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
        if bool(self.camera_rig_config) != bool(self.camera_rig_sha256):
            raise ValueError(
                "camera_rig_config and camera_rig_sha256 must be declared together"
            )
        return self


class RenderReplaySpec(StrictModel):
    enabled: bool = True
    output_dir: str = "render_replay"
    capture_mode: Literal["teacher_time_exact", "verified_action_replay"] = "teacher_time_exact"
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
    skip_existing: bool = True
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
    determinism_probe: bool = True
    determinism_max_channel_diff: int = Field(default=1, ge=0, le=255)
    determinism_max_changed_pixels: int = Field(default=16, ge=0)

    @model_validator(mode="after")
    def enforce_policy_resolution(self) -> RenderProfileSpec:
        if (self.width, self.height) != (256, 256):
            raise ValueError("SO101 rendered training derivatives must be 256x256")
        return self


class EpisodeSubsetSpec(StrictModel):
    metric: Literal["camera2_top_contact_alignment_error_deg"]
    camera_key: Literal["observation.images.camera2"]
    edge_mode: Literal["top-contact"]
    max_angle_deg: float = Field(gt=0, le=90)
    selection_source_root: str | None = None


class SplitSpec(StrictModel):
    kind: Literal["generated", "render_derivative", "episode_subset"] = "generated"
    output_root: str
    repo_id: str
    bins: list[BinSpec] = Field(default_factory=list)
    lookup_cache: str | None = None
    closed_loop: ClosedLoopSpec | None = None
    source_split: str | None = None
    source_dataset_root: str | None = None
    render_replay_sidecar: str | None = None
    expected_bins: dict[int, int] = Field(default_factory=dict)
    render: RenderProfileSpec | None = None
    expected_episodes: int | None = Field(default=None, gt=0)
    subset: EpisodeSubsetSpec | None = None

    @model_validator(mode="after")
    def validate_kind(self) -> SplitSpec:
        if self.kind == "generated" and not self.bins:
            raise ValueError("generated split requires bins")
        if self.kind == "render_derivative":
            sources = [bool(self.source_split), bool(self.source_dataset_root)]
            if sum(sources) != 1 or self.render is None:
                raise ValueError(
                    "render_derivative split requires exactly one of source_split or "
                    "source_dataset_root, plus render"
                )
            if self.bins:
                raise ValueError("render_derivative split must not declare bins")
            if self.source_dataset_root:
                if self.expected_episodes is None:
                    raise ValueError("external render source requires expected_episodes")
                if not self.render_replay_sidecar:
                    raise ValueError("external render source requires render_replay_sidecar")
                if not self.expected_bins:
                    raise ValueError("external render source requires expected_bins")
        if self.kind == "episode_subset":
            if not self.source_dataset_root or self.expected_episodes is None or self.subset is None:
                raise ValueError(
                    "episode_subset split requires source_dataset_root, expected_episodes, and subset"
                )
            if self.bins or self.source_split or self.render is not None or self.render_replay_sidecar:
                raise ValueError(
                    "episode_subset split must not declare bins, source_split, render, or "
                    "render_replay_sidecar"
                )
            if not self.expected_bins:
                raise ValueError("episode_subset split requires expected_bins")
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
    schema_version: Literal[1, 2]
    name: str
    version: str | None = None
    description: str
    source: DatasetSourceSpec | None = None
    source_datasets: list[str] = Field(default_factory=list)
    start_pose: HardwareStartPoseSpec | None = None
    exporter_revision: str
    exporter: str
    subset_script: str = "scripts/filter_so101_lerobot_visual_alignment.py"
    lookup_builder_script: str | None = None
    merge_script: str
    sidecar_script: str
    closed_loop_script: str | None = None
    audit_script: str
    render_replay_script: str = "scripts/build_so101_render_replay_sidecar.py"
    photoreal_builder_script: str = "scripts/build_so101_photoreal_lerobot_dataset.py"
    render_determinism_script: str = "scripts/verify_so101_render_determinism.py"
    lookup_cache: str | None = None
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
        self._validate_source_contract()
        if self.schema_version == 2:
            if self.common.skill_mode == "grip_the_cube_v1":
                geometry_gates = [
                    gate
                    for gate in self.common.inspection_gates
                    if gate.kind == "geometry_contact_alignment"
                ]
                if self.common.contact_alignment is None and len(geometry_gates) != 1:
                    raise ValueError(
                        "schema_version 2 grip_the_cube_v1 recipes require "
                        "one common.inspection_gates geometry_contact_alignment gate"
                    )
            if self.common.close_alignment_gate_mode is not None:
                raise ValueError(
                    "schema_version 2 uses structured alignment inspection gates instead "
                    "of close_alignment_gate_mode"
                )
        for name, split in self.splits.items():
            if split.kind == "render_derivative":
                if split.source_split:
                    source = self.splits.get(split.source_split)
                    if source is None or source.kind != "generated":
                        raise ValueError(f"{name}.source_split must reference a generated split")
                    expected = sum(item.episodes for item in source.bins)
                    if split.expected_episodes not in (None, expected):
                        raise ValueError(
                            f"{name}.expected_episodes must match source split ({expected})"
                        )
                if self.render_replay is None or not self.render_replay.enabled:
                    raise ValueError("render_derivative requires enabled render_replay")
                if (
                    split.source_dataset_root
                    and self.render_replay.capture_mode != "verified_action_replay"
                ):
                    raise ValueError(
                        "external render sources require "
                        "render_replay.capture_mode=verified_action_replay"
                    )
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
            if self.common.target_object_yaw_deg != environment.target_object_yaw_deg:
                raise ValueError(
                    "common.target_object_yaw_deg must match render_replay.environment"
                )
            center = (self.common.spawn_center_x, self.common.spawn_center_y)
            if all(value is not None for value in center) and [
                float(value) for value in center
            ] != list(environment.spawn_center):
                raise ValueError("common spawn center must match render_replay.environment")
            if self.common.camera_rig_config != environment.camera_rig_config:
                raise ValueError(
                    "common.camera_rig_config must match render_replay.environment"
                )
        if any(
            split.kind == "render_derivative" and split.source_split
            for split in self.splits.values()
        ):
            if not self.common.capture_render_replay:
                raise ValueError(
                    "generated render derivatives require common.capture_render_replay=true"
                )
        return self

    def _validate_source_contract(self) -> None:
        if self.schema_version == 1:
            if self.source is not None:
                raise ValueError("schema_version 1 uses legacy source_datasets")
            return
        if self.source is None:
            raise ValueError("schema_version 2 requires source")
        if self.source_datasets:
            raise ValueError("schema_version 2 uses source.datasets, not source_datasets")

        generated = [split for split in self.splits.values() if split.kind == "generated"]
        external_render_roots = {
            split.source_dataset_root
            for split in self.splits.values()
            if split.kind == "render_derivative" and split.source_dataset_root
        }
        subset_roots = {
            split.source_dataset_root
            for split in self.splits.values()
            if split.kind == "episode_subset" and split.source_dataset_root
        }
        if self.source.mode == "from_scratch":
            if self.lookup_builders:
                raise ValueError("from_scratch must not read lookup source_reports")
            if external_render_roots or subset_roots:
                raise ValueError("from_scratch must not reference an external dataset root")
            return

        if self.source.mode == "from_spawn_catalog":
            if not generated:
                raise ValueError("from_spawn_catalog requires generated splits")
            if self.lookup_builders:
                raise ValueError("from_spawn_catalog must not run lookup_builders")
            if external_render_roots or subset_roots:
                raise ValueError("from_spawn_catalog cannot use external render sources")
            referenced = {
                str(Path(split.lookup_cache or self.lookup_cache)) for split in generated
            }
            declared = {str(Path(value)) for value in self.source.catalogs}
            if referenced != declared:
                raise ValueError(
                    "source.catalogs must exactly match generated split lookup_cache values"
                )
            return

        declared = {str(Path(value)) for value in self.source.datasets}
        if self.source.operation == "regenerate_teacher":
            if not generated or not self.lookup_builders:
                raise ValueError(
                    "regenerate_teacher requires generated splits and lookup_builders"
                )
            referenced = {
                str(Path(report).parent)
                for builder in self.lookup_builders
                for report in builder.source_reports
            }
            if referenced != declared:
                raise ValueError(
                    "source.datasets must exactly match lookup_builder source report roots"
                )
            if external_render_roots:
                raise ValueError("regenerate_teacher cannot use external render sources")
            if subset_roots:
                raise ValueError("regenerate_teacher cannot contain episode subsets")
            return

        if self.source.operation == "episode_subset":
            if generated or self.lookup_builders or external_render_roots:
                raise ValueError(
                    "episode_subset source mode may contain only episode_subset splits"
                )
            if not subset_roots or subset_roots != declared:
                raise ValueError(
                    "source.datasets must exactly match episode_subset source_dataset_root values"
                )
            return

        if generated or subset_roots:
            raise ValueError("render_derivative source mode cannot contain generated splits")
        if self.lookup_builders:
            raise ValueError("render_derivative source mode must not run lookup_builders")
        if external_render_roots != declared:
            raise ValueError(
                "source.datasets must exactly match render_derivative source_dataset_root values"
            )

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
