from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DistortionConfig(StrictModel):
    model: Literal["opencv_brown_conrady"]
    coefficients: tuple[float, float, float, float, float]
    calibration_status: str


class CameraSensorConfig(StrictModel):
    model_name: str
    source_resolution: tuple[int, int]
    horizontal_fov_degrees: float = Field(gt=0.0, lt=180.0)
    vertical_fov_degrees: float = Field(gt=0.0, lt=180.0)
    reported_diagonal_fov_degrees: tuple[float, float]
    distortion: DistortionConfig
    board_half_size_m: tuple[float, float, float]
    lens_size_m: tuple[float, float]
    board_distance_behind_pinhole_m: float = Field(ge=0.0)
    lens_distance_behind_pinhole_m: float = Field(ge=0.0)


class CameraRigAssetsConfig(StrictModel):
    wrist_stl_path: Path
    wrist_stl_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wrist_mesh_scale: tuple[float, float, float]
    overhead_stl_dir: Path
    overhead_stl_sha256: dict[str, str]
    overhead_mesh_scale: tuple[float, float, float]
    generated_asset_dir: Path

    @model_validator(mode="after")
    def validate_overhead_assets(self) -> CameraRigAssetsConfig:
        expected = {
            "arm_base.stl",
            "cam_mount_bottom.stl",
            "cam_mount_middle.stl",
            "cam_mount_top.stl",
        }
        if set(self.overhead_stl_sha256) != expected:
            raise ValueError(
                "overhead_stl_sha256 must contain exactly the four official STL files"
            )
        if any(len(value) != 64 for value in self.overhead_stl_sha256.values()):
            raise ValueError("every overhead STL SHA-256 must contain 64 hex characters")
        return self


class RobotPoseConfig(StrictModel):
    joint_names: tuple[str, ...]
    home_qpos: tuple[float, ...]

    @model_validator(mode="after")
    def validate_joint_contract(self) -> RobotPoseConfig:
        if len(self.joint_names) != len(self.home_qpos):
            raise ValueError("joint_names and home_qpos must have the same length")
        return self


class PreviewEnvironmentConfig(StrictModel):
    seed: int = Field(ge=0)
    target_object_color: str
    object_half_sizes_m: tuple[float, ...]
    spawn_center_xy_m: tuple[float, float]
    spawn_min_radius_m: float = Field(ge=0.0)
    spawn_max_radius_m: float = Field(gt=0.0)
    spawn_angle_half_range_degrees: float = Field(gt=0.0, le=180.0)

    @model_validator(mode="after")
    def validate_spawn_radius(self) -> PreviewEnvironmentConfig:
        if self.spawn_max_radius_m < self.spawn_min_radius_m:
            raise ValueError("spawn_max_radius_m must be >= spawn_min_radius_m")
        return self


class WristCameraMountConfig(StrictModel):
    preset: str
    camera_body: str
    collision_mesh: str
    mount_position_gripper_m: tuple[float, float, float]
    rear_up_direction_gripper: tuple[float, float, float]
    rear_up_offset_m: float = Field(ge=0.0)
    mount_downward_angle_degrees: float
    optical_downward_angle_degrees: float
    optical_target_distance_m: float = Field(gt=0.0)
    source_center_mm: tuple[float, float, float]
    source_forward: tuple[float, float, float]
    pixel_postprocess_rotation_degrees: int

    @property
    def camera_position_gripper(self) -> tuple[float, float, float]:
        return tuple(
            self.mount_position_gripper_m[index]
            - self.rear_up_offset_m * self.rear_up_direction_gripper[index]
            for index in range(3)
        )

    @property
    def camera_forward_gripper(self) -> tuple[float, float, float]:
        angle = math.radians(self.optical_downward_angle_degrees)
        return (0.0, math.cos(angle), -math.sin(angle))

    @property
    def camera_up_gripper(self) -> tuple[float, float, float]:
        angle = math.radians(self.optical_downward_angle_degrees)
        return (0.0, -math.sin(angle), -math.cos(angle))

    @property
    def camera_quaternion_wxyz(self) -> tuple[float, float, float, float]:
        half_pitch = math.radians((90.0 - self.optical_downward_angle_degrees) / 2.0)
        return (0.0, 0.0, -math.sin(half_pitch), math.cos(half_pitch))

    @property
    def optical_target_gripper(self) -> tuple[float, float, float]:
        return tuple(
            self.camera_position_gripper[index]
            + self.optical_target_distance_m * self.camera_forward_gripper[index]
            for index in range(3)
        )


class OverheadCameraMountConfig(StrictModel):
    preset: str
    rig_world_position_m: tuple[float, float, float]
    rig_quaternion_wxyz: tuple[float, float, float, float]
    robot_base_world_position_m: tuple[float, float, float]
    tower_position_cad_m: tuple[float, float, float]
    tower_quaternion_cad_wxyz: tuple[float, float, float, float]
    mesh_translation_cad_m: dict[str, tuple[float, float, float]]
    mesh_quaternion_cad_wxyz: dict[str, tuple[float, float, float, float]]
    camera_mount_face_center_top_part_cad_m: tuple[float, float, float]
    upper_mast_translation_cad_m: tuple[float, float, float]
    camera_mount_face_normal_cad: tuple[float, float, float]
    camera_mount_quaternion_cad_wxyz: tuple[float, float, float, float]
    camera_quaternion_cad_wxyz: tuple[float, float, float, float]
    camera_downward_angle_degrees: float
    camera_pinhole_protrusion_m: float = Field(ge=0.0)
    connector_insertion_depth_m: float = Field(ge=0.0)
    arm_base_to_lower_mast_insertion_depth_m: float = Field(ge=0.0)
    pixel_postprocess_rotation_degrees: int

    @model_validator(mode="after")
    def validate_part_transforms(self) -> OverheadCameraMountConfig:
        parts = {
            "arm_base.stl",
            "cam_mount_bottom.stl",
            "cam_mount_middle.stl",
            "cam_mount_top.stl",
        }
        if set(self.mesh_translation_cad_m) != parts:
            raise ValueError("mesh_translation_cad_m must contain every official STL part")
        if set(self.mesh_quaternion_cad_wxyz) != parts:
            raise ValueError("mesh_quaternion_cad_wxyz must contain every official STL part")
        return self

    @property
    def camera_mount_face_center_cad_m(self) -> tuple[float, float, float]:
        return tuple(
            self.camera_mount_face_center_top_part_cad_m[index]
            + self.upper_mast_translation_cad_m[index]
            for index in range(3)
        )

    @property
    def camera_pinhole_cad_m(self) -> tuple[float, float, float]:
        return tuple(
            self.camera_mount_face_center_cad_m[index]
            + self.camera_pinhole_protrusion_m * self.camera_mount_face_normal_cad[index]
            for index in range(3)
        )

    @property
    def camera_forward_world(self) -> tuple[float, float, float]:
        angle = math.radians(self.camera_downward_angle_degrees)
        return (math.cos(angle), 0.0, -math.sin(angle))

    @property
    def camera_up_world(self) -> tuple[float, float, float]:
        angle = math.radians(self.camera_downward_angle_degrees)
        return (math.sin(angle), 0.0, math.cos(angle))


class FreeEvidenceViewConfig(StrictModel):
    kind: Literal["free"]
    name: str
    filename: str
    lookat_m: tuple[float, float, float]
    distance_m: float = Field(gt=0.0)
    azimuth_degrees: float
    elevation_degrees: float
    fovy_degrees: float = Field(gt=0.0, lt=180.0)
    aperture_fstop: float = Field(gt=0.0)
    use_depth_of_field: bool
    clip_start_m: float = Field(gt=0.0)


class MountedEvidenceViewConfig(StrictModel):
    kind: Literal["mounted_front"]
    name: str
    filename: str
    camera_name: Literal["egocentric_cam", "wrist_cam"]
    distance_m: float = Field(gt=0.0)
    fovy_degrees: float = Field(gt=0.0, lt=180.0)
    aperture_fstop: float = Field(gt=0.0)
    use_depth_of_field: bool
    clip_start_m: float = Field(gt=0.0)


EvidenceViewConfig = Annotated[
    FreeEvidenceViewConfig | MountedEvidenceViewConfig,
    Field(discriminator="kind"),
]


class ContactSheetConfig(StrictModel):
    filename: str
    tile_size: int = Field(gt=0)
    header_height: int = Field(ge=0)
    font_size: int = Field(gt=0)
    background_rgb: tuple[int, int, int]
    foreground_rgb: tuple[int, int, int]


class RenderOutputConfig(StrictModel):
    photoreal_subdir: str
    camera1_policy_filename: str
    camera2_policy_filename: str
    report_filename: str
    contact_sheet: ContactSheetConfig

    @model_validator(mode="after")
    def validate_relative_outputs(self) -> RenderOutputConfig:
        for value in (
            self.photoreal_subdir,
            self.camera1_policy_filename,
            self.camera2_policy_filename,
            self.report_filename,
            self.contact_sheet.filename,
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("render output paths must stay inside render.output_dir")
        return self


class PhotorealRenderConfig(StrictModel):
    mode: Literal["blender_cycles_live"]
    output_dir: Path
    render_policy_inference_only: bool
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    policy_size: int = Field(gt=0)
    policy_resize: Literal["center_crop_square_then_resize"]
    samples: int = Field(gt=0)
    denoise: bool
    cycles_seed: int = Field(ge=0)
    lighting_profile: str
    key_light_power: float = Field(ge=0.0)
    fill_light_power: float = Field(ge=0.0)
    world_strength: float = Field(ge=0.0)
    hdri_rotation_degrees: float
    exposure: float
    color_management: str
    color_look: str
    gamma: float = Field(gt=0.0)
    output_format: Literal["PNG"]
    sample_clamp_indirect: float = Field(ge=0.0)
    background_wall: bool
    stable_tabletop: bool
    scene_profile: str
    robot_material: str
    material_profile: Path
    material_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    camera_lens_mm: float = Field(gt=0.0)
    photoreal_asset_root: Path
    blender_bin: str
    expected_blender_version: str
    compute_device_type: Literal["METAL"]
    max_mesh_geoms: int = Field(gt=0)
    preserve_pinhole_renders: bool
    evidence_views: tuple[EvidenceViewConfig, ...]
    outputs: RenderOutputConfig

    @model_validator(mode="after")
    def validate_evidence_views(self) -> PhotorealRenderConfig:
        names = [view.name for view in self.evidence_views]
        filenames = [view.filename for view in self.evidence_views]
        if len(names) != len(set(names)):
            raise ValueError("evidence view names must be unique")
        if len(filenames) != len(set(filenames)):
            raise ValueError("evidence view filenames must be unique")
        unsafe_filename = any(
            Path(filename).is_absolute() or ".." in Path(filename).parts
            for filename in filenames
        )
        if unsafe_filename:
            raise ValueError("evidence view filenames must stay inside render.output_dir")
        if "external_scene" not in names:
            raise ValueError("evidence_views must define external_scene")
        return self


class SO101CameraRigRenderConfig(StrictModel):
    schema_version: Literal[1]
    status: str
    preset: str
    camera_contract: dict[str, str]
    sensor: CameraSensorConfig
    assets: CameraRigAssetsConfig
    robot: RobotPoseConfig
    environment: PreviewEnvironmentConfig
    camera1: OverheadCameraMountConfig
    camera2: WristCameraMountConfig
    render: PhotorealRenderConfig

    @model_validator(mode="after")
    def validate_contract(self) -> SO101CameraRigRenderConfig:
        expected = {
            "observation.images.camera1": "egocentric_cam",
            "observation.images.camera2": "wrist_cam",
        }
        if self.camera_contract != expected:
            raise ValueError(f"camera_contract must be exactly {expected}")
        if self.preset != self.camera1.preset:
            raise ValueError("root preset and camera1 preset must match")
        return self


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repository_path(path: Path) -> Path:
    return path if path.is_absolute() else repository_root() / path


def load_so101_camera_rig_render_config(path: Path) -> SO101CameraRigRenderConfig:
    config_path = path.expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return SO101CameraRigRenderConfig.model_validate(payload)


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
