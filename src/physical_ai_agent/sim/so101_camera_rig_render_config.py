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


class FastenerAlignmentContract(StrictModel):
    reference: Literal["four_base_fastener_hole_centers"]
    fastener_count: Literal[4]
    printed_base_stl: Literal["arm_base.stl"]
    robot_base_mesh: Literal["base_so101_v2.stl"]
    robot_base_mesh_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_center_error_m: float = Field(gt=0.0)
    measured_max_center_error_m: float = Field(ge=0.0)
    measured_rotation_error_degrees: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_measured_alignment(self) -> FastenerAlignmentContract:
        if self.measured_max_center_error_m > self.max_center_error_m:
            raise ValueError(
                "measured fastener-hole error exceeds the locked assembly tolerance"
            )
        return self


class CameraRigAssemblyLockConfig(StrictModel):
    state: Literal["locked"]
    lock_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    fastener_alignment: FastenerAlignmentContract
    fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WristCameraMountConfig(StrictModel):
    preset: str
    assembly_mode: Literal["pcb_flush_lens_through_center_hole"]
    camera_body: str
    collision_mesh: str
    mount_face_center_gripper_m: tuple[float, float, float]
    mount_plate_thickness_m: float | None = Field(default=None, gt=0.0)
    lens_protrusion_m: float = Field(gt=0.0)
    mount_downward_angle_degrees: float
    optical_downward_angle_degrees: float
    effective_vertical_fov_degrees: float | None = Field(
        default=None,
        gt=0.0,
        lt=180.0,
    )
    optical_target_distance_m: float = Field(gt=0.0)
    source_center_mm: tuple[float, float, float]
    source_forward: tuple[float, float, float]
    pixel_postprocess_rotation_degrees: int

    @model_validator(mode="after")
    def validate_flush_optical_axis(self) -> WristCameraMountConfig:
        if not math.isclose(
            self.mount_downward_angle_degrees,
            self.optical_downward_angle_degrees,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "a flush PCB camera must look along the mount-face normal"
            )
        return self

    @property
    def board_contact_center_gripper_m(self) -> tuple[float, float, float]:
        thickness = self.mount_plate_thickness_m or 0.0
        return tuple(
            self.mount_face_center_gripper_m[index]
            - thickness * self.camera_forward_gripper[index]
            for index in range(3)
        )

    @property
    def camera_position_gripper(self) -> tuple[float, float, float]:
        return tuple(
            self.board_contact_center_gripper_m[index]
            + self.lens_protrusion_m * self.camera_forward_gripper[index]
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
    assembly_mode: Literal["pcb_flush_lens_through_center_hole"]
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
    virtual_optical_yaw_degrees: float = Field(default=0.0, ge=-15.0, le=15.0)
    effective_vertical_fov_degrees: float | None = Field(
        default=None,
        gt=0.0,
        lt=180.0,
    )
    camera_pinhole_protrusion_m: float = Field(ge=0.0)
    mount_plate_thickness_m: float | None = Field(default=None, gt=0.0)
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
        normal_length = math.sqrt(
            sum(value * value for value in self.camera_mount_face_normal_cad)
        )
        if not math.isclose(normal_length, 1.0, abs_tol=1e-9):
            raise ValueError("camera_mount_face_normal_cad must be a unit vector")
        for actual, expected in zip(
            self.camera_forward_cad,
            self.camera_mount_face_normal_cad,
            strict=True,
        ):
            if not math.isclose(actual, expected, abs_tol=1e-9):
                raise ValueError(
                    "a flush overhead PCB camera must look through the mount-face hole"
                )
        camera_forward = _rotate_vector_wxyz(
            self.camera_quaternion_cad_wxyz,
            (0.0, 0.0, -1.0),
        )
        for actual, expected in zip(
            camera_forward,
            self.camera_mount_face_normal_cad,
            strict=True,
        ):
            if not math.isclose(actual, expected, abs_tol=1e-9):
                raise ValueError(
                    "camera_quaternion_cad_wxyz must align the optical axis to the mount hole"
                )
        return self

    @property
    def camera_forward_cad(self) -> tuple[float, float, float]:
        angle = math.radians(self.camera_downward_angle_degrees)
        return (math.cos(angle), -math.sin(angle), 0.0)

    @property
    def camera_mount_face_center_cad_m(self) -> tuple[float, float, float]:
        return tuple(
            self.camera_mount_face_center_top_part_cad_m[index]
            + self.upper_mast_translation_cad_m[index]
            for index in range(3)
        )

    @property
    def camera_board_contact_center_cad_m(self) -> tuple[float, float, float]:
        thickness = self.mount_plate_thickness_m or 0.0
        return tuple(
            self.camera_mount_face_center_cad_m[index]
            - thickness * self.camera_mount_face_normal_cad[index]
            for index in range(3)
        )

    @property
    def camera_pinhole_cad_m(self) -> tuple[float, float, float]:
        return tuple(
            self.camera_board_contact_center_cad_m[index]
            + self.camera_pinhole_protrusion_m * self.camera_mount_face_normal_cad[index]
            for index in range(3)
        )

    @property
    def camera_forward_world(self) -> tuple[float, float, float]:
        tower_forward = _rotate_vector_wxyz(
            self.tower_quaternion_cad_wxyz,
            self.effective_camera_forward_cad,
        )
        return _rotate_vector_wxyz(self.rig_quaternion_wxyz, tower_forward)

    @property
    def camera_up_world(self) -> tuple[float, float, float]:
        tower_up = _rotate_vector_wxyz(
            self.tower_quaternion_cad_wxyz,
            self.effective_camera_up_cad,
        )
        return _rotate_vector_wxyz(self.rig_quaternion_wxyz, tower_up)

    @property
    def effective_camera_quaternion_cad_wxyz(
        self,
    ) -> tuple[float, float, float, float]:
        half_yaw = math.radians(self.virtual_optical_yaw_degrees) / 2.0
        local_yaw = (math.cos(half_yaw), 0.0, math.sin(half_yaw), 0.0)
        return _multiply_quaternions_wxyz(
            self.camera_quaternion_cad_wxyz,
            local_yaw,
        )

    @property
    def effective_camera_forward_cad(self) -> tuple[float, float, float]:
        return _rotate_vector_wxyz(
            self.effective_camera_quaternion_cad_wxyz,
            (0.0, 0.0, -1.0),
        )

    @property
    def effective_camera_up_cad(self) -> tuple[float, float, float]:
        return _rotate_vector_wxyz(
            self.effective_camera_quaternion_cad_wxyz,
            (0.0, 1.0, 0.0),
        )


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


class HashedRenderAssetConfig(StrictModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BlenderSceneAssetConfig(StrictModel):
    name: str
    blend: HashedRenderAssetConfig
    object_name: str
    dependencies: tuple[HashedRenderAssetConfig, ...]
    position_m: tuple[float, float, float]
    rotation_euler_degrees: tuple[float, float, float]
    scale_xyz: tuple[float, float, float]


class RenderLightConfig(StrictModel):
    name: str
    type: Literal["AREA", "SPOT"]
    position_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    power: float = Field(gt=0.0)
    color_rgb: tuple[float, float, float]
    size_m: float = Field(gt=0.0)
    spot_size_degrees: float = Field(gt=0.0, lt=180.0)
    spot_blend: float = Field(ge=0.0, le=1.0)


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
    policy_resize: Literal[
        "center_crop_square_then_resize",
        "direct_square_render",
    ]
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
    bevel_width_mm_range: tuple[float, float] | None = None
    bevel_segments: int = Field(default=3, ge=1, le=8)
    scene_assets: tuple[BlenderSceneAssetConfig, ...] = ()
    lights: tuple[RenderLightConfig, ...] = ()
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
        if self.bevel_width_mm_range is not None:
            minimum, maximum = self.bevel_width_mm_range
            if minimum <= 0.0 or maximum < minimum:
                raise ValueError("bevel_width_mm_range must be positive and ordered")
        asset_names = [asset.name for asset in self.scene_assets]
        if len(asset_names) != len(set(asset_names)):
            raise ValueError("scene asset names must be unique")
        light_names = [light.name for light in self.lights]
        if len(light_names) != len(set(light_names)):
            raise ValueError("render light names must be unique")
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
    assembly_lock: CameraRigAssemblyLockConfig | None = None
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
        if self.render.policy_resize == "center_crop_square_then_resize":
            sensor_width, sensor_height = self.sensor.source_resolution
            sensor_aspect = sensor_width / sensor_height
            render_aspect = self.render.source_width / self.render.source_height
            if not math.isclose(render_aspect, sensor_aspect, abs_tol=0.01):
                raise ValueError(
                    "render source aspect ratio must match the physical sensor before "
                    "the square policy crop"
                )
        elif self.render.source_width != self.render.source_height:
            raise ValueError(
                "direct_square_render requires a square source render"
            )
        if not math.isclose(
            self.camera1.camera_pinhole_protrusion_m,
            2.0 * self.sensor.lens_size_m[1],
            abs_tol=1e-9,
        ):
            raise ValueError(
                "camera1 pinhole protrusion must equal the modeled lens length"
            )
        if not math.isclose(
            self.camera2.lens_protrusion_m,
            2.0 * self.sensor.lens_size_m[1],
            abs_tol=1e-9,
        ):
            raise ValueError(
                "camera2 lens_protrusion_m must equal the modeled lens length"
            )
        if not math.isclose(
            self.sensor.lens_distance_behind_pinhole_m,
            self.sensor.lens_size_m[1],
            abs_tol=1e-9,
        ):
            raise ValueError(
                "lens center must be one lens half-length behind the pinhole"
            )
        for camera_name, protrusion in (
            ("camera1", self.camera1.camera_pinhole_protrusion_m),
            ("camera2", self.camera2.lens_protrusion_m),
        ):
            expected_board_distance = protrusion + self.sensor.board_half_size_m[2]
            if not math.isclose(
                self.sensor.board_distance_behind_pinhole_m,
                expected_board_distance,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"PCB front face must touch the rear {camera_name} mount face"
                )
        for camera_name, protrusion, plate_thickness in (
            (
                "camera1",
                self.camera1.camera_pinhole_protrusion_m,
                self.camera1.mount_plate_thickness_m,
            ),
            (
                "camera2",
                self.camera2.lens_protrusion_m,
                self.camera2.mount_plate_thickness_m,
            ),
        ):
            if plate_thickness is not None and protrusion <= plate_thickness:
                raise ValueError(
                    f"{camera_name} lens must pass through and protrude beyond its mount plate"
                )
        if self.assembly_lock is not None:
            printed_base = self.assembly_lock.fastener_alignment.printed_base_stl
            if printed_base not in self.assets.overhead_stl_sha256:
                raise ValueError(
                    "assembly lock printed base must be present in overhead assets"
                )
            actual_fingerprint = assembly_fingerprint_sha256(
                self.model_dump(mode="json")
            )
            if actual_fingerprint != self.assembly_lock.fingerprint_sha256:
                raise ValueError(
                    "camera-rig assembly lock mismatch: a physical asset, camera "
                    "mount, or base transform changed"
                )
        return self


def assembly_fingerprint_sha256(payload: dict[str, object]) -> str:
    assets = payload["assets"]
    sensor = payload["sensor"]
    assembly_lock = payload["assembly_lock"]
    if not isinstance(assets, dict):
        raise TypeError("assets must be a mapping")
    if not isinstance(sensor, dict):
        raise TypeError("sensor must be a mapping")
    if not isinstance(assembly_lock, dict):
        raise TypeError("assembly_lock must be a mapping")
    fingerprint_payload = {
        "lock_id": assembly_lock["lock_id"],
        "preset": payload["preset"],
        "source_assets": {
            "wrist_stl_path": assets["wrist_stl_path"],
            "wrist_stl_sha256": assets["wrist_stl_sha256"],
            "wrist_mesh_scale": assets["wrist_mesh_scale"],
            "overhead_stl_dir": assets["overhead_stl_dir"],
            "overhead_stl_sha256": assets["overhead_stl_sha256"],
            "overhead_mesh_scale": assets["overhead_mesh_scale"],
        },
        "camera_module_geometry": {
            "board_half_size_m": sensor["board_half_size_m"],
            "lens_size_m": sensor["lens_size_m"],
            "board_distance_behind_pinhole_m": sensor[
                "board_distance_behind_pinhole_m"
            ],
            "lens_distance_behind_pinhole_m": sensor[
                "lens_distance_behind_pinhole_m"
            ],
        },
        "camera1_assembly": _fingerprint_camera_assembly(payload["camera1"]),
        "camera2_assembly": _fingerprint_camera_assembly(payload["camera2"]),
        "fastener_alignment": assembly_lock["fastener_alignment"],
    }
    canonical = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _fingerprint_camera_assembly(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("camera assembly must be a mapping")
    normalized = dict(payload)
    if normalized.get("mount_plate_thickness_m") is None:
        normalized.pop("mount_plate_thickness_m", None)
    return normalized


def _rotate_vector_wxyz(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isclose(norm, 1.0, abs_tol=1e-9):
        raise ValueError("camera quaternion must be normalized")
    qx, qy, qz = x / norm, y / norm, z / norm
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + (w / norm) * tx + (qy * tz - qz * ty),
        vy + (w / norm) * ty + (qz * tx - qx * tz),
        vz + (w / norm) * tz + (qx * ty - qy * tx),
    )


def _multiply_quaternions_wxyz(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    norm = math.sqrt(sum(value * value for value in result))
    if math.isclose(norm, 0.0, abs_tol=1e-12):
        raise ValueError("camera quaternion composition cannot be zero")
    return tuple(value / norm for value in result)


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
