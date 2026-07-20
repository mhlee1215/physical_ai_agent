from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.sim.so101_camera_rig_render_config import (
    SO101CameraRigRenderConfig,
    resolve_repository_path,
)
from physical_ai_agent.sim.so101_wrist_camera_mount import (
    _ROBOT_XML_BUILD_LOCK,
    INNOMAKER_U20CAM_BOARD_DISTANCE_BEHIND_PINHOLE_M,
    INNOMAKER_U20CAM_BOARD_HALF_SIZE_M,
    INNOMAKER_U20CAM_CANDIDATE_DISTORTION_COEFFICIENTS,
    INNOMAKER_U20CAM_DISTORTION_CALIBRATION_STATUS,
    INNOMAKER_U20CAM_DISTORTION_MODEL,
    INNOMAKER_U20CAM_HORIZONTAL_FOV_DEGREES,
    INNOMAKER_U20CAM_LENS_DISTANCE_BEHIND_PINHOLE_M,
    INNOMAKER_U20CAM_LENS_SIZE_M,
    INNOMAKER_U20CAM_SOURCE_RESOLUTION,
    INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES,
    INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES,
    INTEGRATED_32X32_UVC_CAMERA_OPTICAL_AXIS_OFFSET_DEGREES,
    INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_GRIPPER,
    INTEGRATED_32X32_UVC_CAMERA_POSITION,
    INTEGRATED_32X32_UVC_CAMERA_REAR_UP_OFFSET_M,
    INTEGRATED_32X32_UVC_PRESET,
    prepare_integrated_32x32_uvc_robot_xml,
)

OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET = "official_overhead_and_integrated_32x32_uvc"
OFFICIAL_OVERHEAD_CAMERA_FOVY_DEGREES = INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES

OFFICIAL_OVERHEAD_SOURCE_SHA256 = {
    "arm_base.stl": "169adfd40bcca689334efd1188c9b42cc03c914dc0afeaa98cb5431013610833",
    "cam_mount_bottom.stl": "b3545b6cae437210e17b7dcfee2e12e00dc7a59ece9264f4b13ab9fd8ceb8088",
    "cam_mount_middle.stl": "f6f3980df74777f74cf53a4697cbb63e4e000ab53f418de209559214f733f84b",
    "cam_mount_top.stl": "206762be157918967741eb53665cac796485c4609ac64d4c9444d85c7fcb2c4c",
}

# The official STLs are authored in millimeters with CAD +Y as mast-up, but
# middle and top retain their own insertion-reference coordinates.  The
# assembly guide requires bottom -> middle -> top stacking, so the two upper
# pieces need connector-derived Y translations instead of being overlaid in
# their raw STL coordinates. A +90 degree rotation around X then maps CAD +Y
# to MuJoCo +Z.
OVERHEAD_MESH_SCALE = (0.001, 0.001, 0.001)
OVERHEAD_RIG_QUATERNION_WXYZ = (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)

# The black mast sections share a connector axis, but their STL origins do not
# share the yellow lower mast's axis. The lower mast's top connector is centred
# at CAD X=18.7 mm, Z=36.5125 mm. Each upper section slides 7.85 mm into the
# section below, matching the middle/top overlap preserved in the source STLs.
OVERHEAD_UPPER_MAST_TRANSLATION_CAD_M = (0.0187, 0.1881, 0.0365125)
OVERHEAD_CONNECTOR_INSERTION_DEPTH_M = 0.00785

# Keep the mast's source orientation so the camera head still faces the
# manipulation workspace, but attach the lower floor through the arm base's
# south-edge socket. After the rig-frame rotation, CAD south (-Z) is
# screen/world right (+Y), so the mast sits beside the base rather than behind
# it. The 37.4 mm-wide lower-floor tab is aligned to the matching 37.4 mm-wide
# arm-base socket, then fully inserted through the socket's 10 mm depth. The
# tower is also lowered 5 mm in CAD Y so both 7.2 mm printed floor plates are
# coplanar.
OVERHEAD_ARM_BASE_JOINT_MIN_CAD_X_M = -0.09392091755867004
OVERHEAD_ARM_BASE_JOINT_MAX_CAD_X_M = -0.05652091602478027
OVERHEAD_LOWER_MAST_JOINT_MIN_CAD_X_M = 0.0
OVERHEAD_LOWER_MAST_JOINT_MAX_CAD_X_M = 0.037400001525878906
OVERHEAD_ARM_BASE_SOCKET_OUTER_CAD_Z_M = -0.13295004272460938
OVERHEAD_ARM_BASE_SOCKET_INNER_CAD_Z_M = -0.12295004272460938
OVERHEAD_LOWER_MAST_TAB_OUTER_CAD_Z_M = 0.07302499389648438
OVERHEAD_LOWER_MAST_TAB_INNER_CAD_Z_M = 0.08302499389648438
OVERHEAD_ARM_BASE_TO_LOWER_MAST_INSERTION_DEPTH_M = 0.010
OVERHEAD_TOWER_POSITION_CAD_M = (
    OVERHEAD_ARM_BASE_JOINT_MIN_CAD_X_M
    - OVERHEAD_LOWER_MAST_JOINT_MIN_CAD_X_M,
    -0.005,
    OVERHEAD_ARM_BASE_SOCKET_OUTER_CAD_Z_M
    - OVERHEAD_LOWER_MAST_TAB_OUTER_CAD_Z_M,
)
OVERHEAD_TOWER_QUATERNION_CAD_WXYZ = (1.0, 0.0, 0.0, 0.0)
OVERHEAD_MESH_TRANSLATION_CAD_M = {
    "arm_base.stl": (0.0, 0.0, 0.0),
    "cam_mount_bottom.stl": (0.0, 0.0, 0.0),
    "cam_mount_middle.stl": OVERHEAD_UPPER_MAST_TRANSLATION_CAD_M,
    "cam_mount_top.stl": OVERHEAD_UPPER_MAST_TRANSLATION_CAD_M,
}
# The top STL's +X-facing angled board plane already points at the simulated
# home gripper and manipulation workspace, so every mast section retains its
# source orientation.
OVERHEAD_TOP_MOUNT_QUATERNION_CAD_WXYZ = (1.0, 0.0, 0.0, 0.0)
OVERHEAD_MESH_QUATERNION_CAD_WXYZ = {
    "arm_base.stl": (1.0, 0.0, 0.0, 0.0),
    "cam_mount_bottom.stl": (1.0, 0.0, 0.0, 0.0),
    "cam_mount_middle.stl": (1.0, 0.0, 0.0, 0.0),
    "cam_mount_top.stl": OVERHEAD_TOP_MOUNT_QUATERNION_CAD_WXYZ,
}

# The arm-base bounds center is the single-follower mounting center in the
# official assembly. Translate it to the simulated SO101 base origin and place
# the 7.2 mm printed plate on the table. The robot base is raised by the same
# amount so it sits on the plate instead of intersecting it.
OVERHEAD_ARM_BASE_THICKNESS_M = 0.0072
OVERHEAD_RIG_WORLD_POSITION = (0.04912092, -0.05343047, 0.0050)
OVERHEAD_ARM_BASE_FRONT_EDGE_CAD_X_M = -0.004320921421051026
SO101_BASE_SHELL_FRONT_EDGE_FROM_ROOT_X_M = 0.06463529232458115
SO101_BASE_WORLD_POSITION = (
    OVERHEAD_RIG_WORLD_POSITION[0]
    + OVERHEAD_ARM_BASE_FRONT_EDGE_CAD_X_M
    - SO101_BASE_SHELL_FRONT_EDGE_FROM_ROOT_X_M,
    0.0,
    OVERHEAD_ARM_BASE_THICKNESS_M,
)

# The printed top mount has a 65 degree downward face normal. A live frame from
# the installed camera showed that using that normal directly places the mast
# inside camera1. The installed module's effective optical axis is therefore
# calibrated independently from the mesh plane: 50 degrees downward preserves
# the measured wide FOV while keeping the physical mast behind the image plane.
OVERHEAD_CAMERA_MOUNT_FACE_NORMAL_CAD = (
    0.42261826174069944,
    -0.9063077870366499,
    0.0,
)
OVERHEAD_CAMERA_MOUNT_QUATERNION_CAD_WXYZ = (
    0.5963678105290182,
    -0.37992819659091526,
    -0.5963678105290181,
    -0.37992819659091526,
)
OFFICIAL_OVERHEAD_CAMERA_DOWNWARD_ANGLE_DEGREES = 50.0
_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS = math.radians(
    OFFICIAL_OVERHEAD_CAMERA_DOWNWARD_ANGLE_DEGREES
)
OVERHEAD_CAMERA_FORWARD_CAD = (
    math.cos(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
    -math.sin(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
    0.0,
)
OVERHEAD_CAMERA_UP_CAD = (
    math.sin(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
    math.cos(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
    0.0,
)
OVERHEAD_CAMERA_QUATERNION_CAD_WXYZ = (
    0.6408563820557885,
    -0.2988362387301198,
    -0.6408563820557885,
    -0.2988362387301199,
)
OVERHEAD_CAMERA_FORWARD_WORLD = (
    math.cos(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
    0.0,
    -math.sin(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
)
OVERHEAD_CAMERA_UP_WORLD = (
    math.sin(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
    0.0,
    math.cos(_OVERHEAD_CAMERA_DOWNWARD_ANGLE_RADIANS),
)
# The four M2-hole centres on the top STL define the camera-board centre. The
# installed lens protrudes beyond that mounting plane; 20 mm is the reviewed
# camera1 viewpoint offset that keeps the optical centre clear of the tower.
OVERHEAD_CAMERA_PINHOLE_PROTRUSION_M = 0.020
OVERHEAD_CAMERA_MOUNT_FACE_CENTER_TOP_PART_CAD_M = (
    0.029523,
    0.362249,
    0.0,
)
OVERHEAD_CAMERA_MOUNT_FACE_CENTER_CAD_M = (
    OVERHEAD_CAMERA_MOUNT_FACE_CENTER_TOP_PART_CAD_M[0]
    + OVERHEAD_UPPER_MAST_TRANSLATION_CAD_M[0],
    OVERHEAD_CAMERA_MOUNT_FACE_CENTER_TOP_PART_CAD_M[1]
    + OVERHEAD_UPPER_MAST_TRANSLATION_CAD_M[1],
    OVERHEAD_CAMERA_MOUNT_FACE_CENTER_TOP_PART_CAD_M[2]
    + OVERHEAD_UPPER_MAST_TRANSLATION_CAD_M[2],
)
OVERHEAD_CAMERA_PINHOLE_CAD = tuple(
    OVERHEAD_CAMERA_MOUNT_FACE_CENTER_CAD_M[index]
    + OVERHEAD_CAMERA_PINHOLE_PROTRUSION_M
    * OVERHEAD_CAMERA_MOUNT_FACE_NORMAL_CAD[index]
    for index in range(3)
)


@dataclass(frozen=True)
class SO101OfficialCameraRigAsset:
    preset: str
    source_dir: str
    source_sha256: dict[str, str]
    staged_assets: dict[str, str]
    robot_xml: str
    manifest_path: str
    camera1_position_world: tuple[float, float, float]
    camera1_forward_world: tuple[float, float, float]
    camera1_up_world: tuple[float, float, float]
    camera1_fovy_degrees: float


def default_official_overhead_source_dir() -> Path:
    configured = os.environ.get("SO101_OVERHEAD_32X32_UVC_STL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "_workspace"
        / "so101_camera_mount_assets"
        / "overhead_32x32_uvc"
    )


def prepare_official_32x32_uvc_camera_rig_xml(
    overhead_source_dir: Path | None = None,
    *,
    output_dir: Path | None = None,
    rig_config: SO101CameraRigRenderConfig | None = None,
) -> SO101OfficialCameraRigAsset:
    configured_source_dir = (
        resolve_repository_path(rig_config.assets.overhead_stl_dir)
        if rig_config is not None
        else default_official_overhead_source_dir()
    )
    source_dir = (overhead_source_dir or configured_source_dir).resolve()
    expected_hashes = (
        rig_config.assets.overhead_stl_sha256
        if rig_config is not None
        else OFFICIAL_OVERHEAD_SOURCE_SHA256
    )
    source_hashes = _verify_official_assets(source_dir, expected_hashes=expected_hashes)
    repo_root = Path(__file__).resolve().parents[3]
    configured_build_dir = (
        resolve_repository_path(rig_config.assets.generated_asset_dir)
        if rig_config is not None
        else repo_root / "_workspace" / "so101_camera_mount_assets" / "generated"
    )
    build_dir = (
        output_dir or configured_build_dir
    ).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    wrist_asset = prepare_integrated_32x32_uvc_robot_xml(
        output_dir=build_dir,
        rig_config=rig_config,
    )
    staged_assets: dict[str, str] = {}
    for filename in expected_hashes:
        destination = build_dir / filename
        shutil.copyfile(source_dir / filename, destination)
        staged_assets[filename] = str(destination)

    robot_xml = build_dir / "so101_new_calib_official_32x32_uvc_camera_rig.xml"
    _build_camera_rig_xml(
        Path(wrist_asset.robot_xml),
        staged_assets,
        robot_xml,
        rig_config=rig_config,
    )
    manifest_path = build_dir / "official_32x32_uvc_camera_rig_manifest.json"
    camera1_position_world = _camera1_position_world(rig_config)
    camera1 = rig_config.camera1 if rig_config is not None else None
    camera2 = rig_config.camera2 if rig_config is not None else None
    sensor = rig_config.sensor if rig_config is not None else None
    camera1_forward_world = (
        camera1.camera_forward_world if camera1 else OVERHEAD_CAMERA_FORWARD_WORLD
    )
    camera1_up_world = camera1.camera_up_world if camera1 else OVERHEAD_CAMERA_UP_WORLD
    camera1_fovy = (
        sensor.vertical_fov_degrees if sensor else OFFICIAL_OVERHEAD_CAMERA_FOVY_DEGREES
    )
    asset = SO101OfficialCameraRigAsset(
        preset=OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
        source_dir=str(source_dir),
        source_sha256=source_hashes,
        staged_assets=staged_assets,
        robot_xml=str(robot_xml),
        manifest_path=str(manifest_path),
        camera1_position_world=camera1_position_world,
        camera1_forward_world=camera1_forward_world,
        camera1_up_world=camera1_up_world,
        camera1_fovy_degrees=camera1_fovy,
    )
    payload = asdict(asset)
    payload.update(
        {
            "schema_version": 1,
            "status": "review_candidate_not_applied_to_existing_datasets",
            "camera_contract": {
                "observation.images.camera1": "egocentric_cam",
                "observation.images.camera2": "wrist_cam",
            },
            "camera1_body": "overhead_camera_tower",
            "camera1_pixel_postprocess_rotation_degrees": (
                camera1.pixel_postprocess_rotation_degrees if camera1 else 0
            ),
            "camera2_mount_preset": (
                camera2.preset if camera2 else INTEGRATED_32X32_UVC_PRESET
            ),
            "camera_model": (
                sensor.model_name
                if sensor
                else "InnoMaker U20CAM-1080P (Amazon B0CNCSFQC1)"
            ),
            "source_resolution": list(
                sensor.source_resolution if sensor else INNOMAKER_U20CAM_SOURCE_RESOLUTION
            ),
            "horizontal_fov_degrees": (
                sensor.horizontal_fov_degrees
                if sensor
                else INNOMAKER_U20CAM_HORIZONTAL_FOV_DEGREES
            ),
            "distortion": {
                "model": sensor.distortion.model if sensor else INNOMAKER_U20CAM_DISTORTION_MODEL,
                "coefficients": list(
                    sensor.distortion.coefficients
                    if sensor
                    else INNOMAKER_U20CAM_CANDIDATE_DISTORTION_COEFFICIENTS
                ),
                "calibration_status": (
                    sensor.distortion.calibration_status
                    if sensor
                    else INNOMAKER_U20CAM_DISTORTION_CALIBRATION_STATUS
                ),
                "applies_to": [
                    "observation.images.camera1",
                    "observation.images.camera2",
                ],
            },
            "camera2_optical_axis": {
                "position_gripper": list(
                    camera2.camera_position_gripper
                    if camera2
                    else INTEGRATED_32X32_UVC_CAMERA_POSITION
                ),
                "target_gripper": list(
                    camera2.optical_target_gripper
                    if camera2
                    else INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_GRIPPER
                ),
                "offset_degrees": (
                    abs(
                        camera2.optical_downward_angle_degrees
                        - camera2.mount_downward_angle_degrees
                    )
                    if camera2
                    else INTEGRATED_32X32_UVC_CAMERA_OPTICAL_AXIS_OFFSET_DEGREES
                ),
                "downward_angle_degrees": (
                    camera2.optical_downward_angle_degrees
                    if camera2
                    else INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES
                ),
                "rear_up_offset_m": (
                    camera2.rear_up_offset_m
                    if camera2
                    else INTEGRATED_32X32_UVC_CAMERA_REAR_UP_OFFSET_M
                ),
            },
            "camera1_optical_axis": {
                "downward_angle_degrees": (
                    camera1.camera_downward_angle_degrees
                    if camera1
                    else OFFICIAL_OVERHEAD_CAMERA_DOWNWARD_ANGLE_DEGREES
                ),
                "mount_face_normal_cad": list(
                    camera1.camera_mount_face_normal_cad
                    if camera1
                    else OVERHEAD_CAMERA_MOUNT_FACE_NORMAL_CAD
                ),
                "forward_cad": list(
                    _camera1_forward_cad(rig_config)
                ),
                "calibration_source": "installed_camera_frame",
                "self_mount_visible_at_home_pose": False,
            },
            "policy_resolution": [256, 256],
            "policy_resize": "center_crop_square_then_resize",
            "assembly": {
                "rig_world_position": list(
                    camera1.rig_world_position_m if camera1 else OVERHEAD_RIG_WORLD_POSITION
                ),
                "robot_base_world_position": list(
                    camera1.robot_base_world_position_m if camera1 else SO101_BASE_WORLD_POSITION
                ),
                "rig_quaternion_wxyz": list(
                    camera1.rig_quaternion_wxyz if camera1 else OVERHEAD_RIG_QUATERNION_WXYZ
                ),
                "tower_position_cad_m": list(
                    camera1.tower_position_cad_m if camera1 else OVERHEAD_TOWER_POSITION_CAD_M
                ),
                "tower_quaternion_cad_wxyz": list(
                    camera1.tower_quaternion_cad_wxyz
                    if camera1
                    else OVERHEAD_TOWER_QUATERNION_CAD_WXYZ
                ),
                "arm_base_to_lower_mast_insertion_depth_m": (
                    camera1.arm_base_to_lower_mast_insertion_depth_m
                    if camera1
                    else OVERHEAD_ARM_BASE_TO_LOWER_MAST_INSERTION_DEPTH_M
                ),
                "connector_insertion_depth_m": (
                    camera1.connector_insertion_depth_m
                    if camera1
                    else OVERHEAD_CONNECTOR_INSERTION_DEPTH_M
                ),
                "camera_pinhole_cad": list(
                    camera1.camera_pinhole_cad_m if camera1 else OVERHEAD_CAMERA_PINHOLE_CAD
                ),
                "camera_pinhole_protrusion_m": (
                    camera1.camera_pinhole_protrusion_m
                    if camera1
                    else OVERHEAD_CAMERA_PINHOLE_PROTRUSION_M
                ),
                "assembly_mode": "arm_base_slot_plus_connector_stack",
                "stl_parts_share_one_cad_frame": False,
                "part_translation_cad_m": {
                    name: list(translation)
                    for name, translation in (
                        camera1.mesh_translation_cad_m
                        if camera1
                        else OVERHEAD_MESH_TRANSLATION_CAD_M
                    ).items()
                },
                "part_quaternion_cad_wxyz": {
                    name: list(quaternion)
                    for name, quaternion in (
                        camera1.mesh_quaternion_cad_wxyz
                        if camera1
                        else OVERHEAD_MESH_QUATERNION_CAD_WXYZ
                    ).items()
                },
            },
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return asset


def make_pick_lift_env_with_official_32x32_uvc_camera_rig(
    *,
    config: Any,
    render_mode: str | None = None,
    overhead_source_dir: Path | None = None,
    rig_config: SO101CameraRigRenderConfig | None = None,
    **kwargs: Any,
) -> Any:
    import so101_nexus_mujoco.pick_env as pick_env

    asset = prepare_official_32x32_uvc_camera_rig_xml(
        overhead_source_dir,
        rig_config=rig_config,
    )
    with _ROBOT_XML_BUILD_LOCK:
        original_robot_xml = pick_env._SO101_XML
        try:
            pick_env._SO101_XML = Path(asset.robot_xml)
            env = pick_env.PickLiftEnv(config=config, render_mode=render_mode, **kwargs)
        finally:
            pick_env._SO101_XML = original_robot_xml
    return env


def _build_camera_rig_xml(
    wrist_robot_xml: Path,
    staged_assets: dict[str, str],
    output_xml: Path,
    *,
    rig_config: SO101CameraRigRenderConfig | None = None,
) -> None:
    tree = ET.parse(wrist_robot_xml)
    root = tree.getroot()
    asset = root.find("./asset")
    worldbody = root.find("./worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"SO101 robot XML is missing asset/worldbody: {wrist_robot_xml}")
    robot_base = worldbody.find('./body[@name="base"]')
    if robot_base is None:
        raise ValueError(f"SO101 robot XML is missing the base body: {wrist_robot_xml}")
    camera1 = rig_config.camera1 if rig_config is not None else None
    sensor = rig_config.sensor if rig_config is not None else None
    robot_base_position = (
        camera1.robot_base_world_position_m if camera1 else SO101_BASE_WORLD_POSITION
    )
    rig_world_position = camera1.rig_world_position_m if camera1 else OVERHEAD_RIG_WORLD_POSITION
    rig_quaternion = camera1.rig_quaternion_wxyz if camera1 else OVERHEAD_RIG_QUATERNION_WXYZ
    tower_position = camera1.tower_position_cad_m if camera1 else OVERHEAD_TOWER_POSITION_CAD_M
    tower_quaternion = (
        camera1.tower_quaternion_cad_wxyz
        if camera1
        else OVERHEAD_TOWER_QUATERNION_CAD_WXYZ
    )
    mesh_translations = (
        camera1.mesh_translation_cad_m if camera1 else OVERHEAD_MESH_TRANSLATION_CAD_M
    )
    mesh_quaternions = (
        camera1.mesh_quaternion_cad_wxyz
        if camera1
        else OVERHEAD_MESH_QUATERNION_CAD_WXYZ
    )
    mesh_scale = rig_config.assets.overhead_mesh_scale if rig_config else OVERHEAD_MESH_SCALE
    pinhole = camera1.camera_pinhole_cad_m if camera1 else OVERHEAD_CAMERA_PINHOLE_CAD
    mount_face_normal = (
        camera1.camera_mount_face_normal_cad
        if camera1
        else OVERHEAD_CAMERA_MOUNT_FACE_NORMAL_CAD
    )
    mount_quaternion = (
        camera1.camera_mount_quaternion_cad_wxyz
        if camera1
        else OVERHEAD_CAMERA_MOUNT_QUATERNION_CAD_WXYZ
    )
    camera_quaternion = (
        camera1.camera_quaternion_cad_wxyz
        if camera1
        else OVERHEAD_CAMERA_QUATERNION_CAD_WXYZ
    )
    camera_fovy = sensor.vertical_fov_degrees if sensor else OFFICIAL_OVERHEAD_CAMERA_FOVY_DEGREES
    board_distance = (
        sensor.board_distance_behind_pinhole_m
        if sensor
        else INNOMAKER_U20CAM_BOARD_DISTANCE_BEHIND_PINHOLE_M
    )
    lens_distance = (
        sensor.lens_distance_behind_pinhole_m
        if sensor
        else INNOMAKER_U20CAM_LENS_DISTANCE_BEHIND_PINHOLE_M
    )
    board_half_size = sensor.board_half_size_m if sensor else INNOMAKER_U20CAM_BOARD_HALF_SIZE_M
    lens_size = sensor.lens_size_m if sensor else INNOMAKER_U20CAM_LENS_SIZE_M
    robot_base.set("pos", _format_vector(robot_base_position))

    material_specs = {
        "overhead_yellow_pla": "0.70 0.78 0.03 1",
        "overhead_black_pla": "0.025 0.03 0.035 1",
        "overhead_camera_pcb": "0.018 0.025 0.022 1",
        "overhead_camera_lens": "0.008 0.010 0.014 1",
    }
    for name, rgba in material_specs.items():
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    mesh_names = {
        "arm_base.stl": "overhead_arm_base_32x32_uvc",
        "cam_mount_bottom.stl": "overhead_cam_mount_bottom_32x32_uvc",
        "cam_mount_middle.stl": "overhead_cam_mount_middle_32x32_uvc",
        "cam_mount_top.stl": "overhead_cam_mount_top_32x32_uvc",
    }
    for filename, mesh_name in mesh_names.items():
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": mesh_name,
                "file": str(Path(staged_assets[filename]).resolve()),
                "scale": " ".join(str(value) for value in mesh_scale),
            },
        )

    rig = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "overhead_camera_mount",
            "pos": _format_vector(rig_world_position),
            "quat": _format_vector(rig_quaternion),
        },
    )
    ET.SubElement(
        rig,
        "geom",
        {
            "name": "arm_base_visual",
            "class": "visual",
            "type": "mesh",
            "mesh": mesh_names["arm_base.stl"],
            "material": "overhead_yellow_pla",
            "pos": _format_vector(mesh_translations["arm_base.stl"]),
            "contype": "0",
            "conaffinity": "0",
        },
    )
    tower = ET.SubElement(
        rig,
        "body",
        {
            "name": "overhead_camera_tower",
            "pos": _format_vector(tower_position),
            "quat": _format_vector(tower_quaternion),
        },
    )
    for filename in (
        "cam_mount_bottom.stl",
        "cam_mount_middle.stl",
        "cam_mount_top.stl",
    ):
        material = (
            "overhead_yellow_pla" if filename == "cam_mount_bottom.stl" else "overhead_black_pla"
        )
        translation = mesh_translations[filename]
        attributes = {
            "name": f"{Path(filename).stem}_visual",
            "class": "visual",
            "type": "mesh",
            "mesh": mesh_names[filename],
            "material": material,
            "pos": _format_vector(translation),
            "contype": "0",
            "conaffinity": "0",
        }
        quaternion = mesh_quaternions[filename]
        if quaternion != (1.0, 0.0, 0.0, 0.0):
            attributes["quat"] = _format_vector(quaternion)
        ET.SubElement(
            tower,
            "geom",
            attributes,
        )

    plate_center = tuple(
        pinhole[index]
        - board_distance * mount_face_normal[index]
        for index in range(3)
    )
    lens_center = tuple(
        pinhole[index]
        - lens_distance * mount_face_normal[index]
        for index in range(3)
    )
    ET.SubElement(
        tower,
        "geom",
        {
            "name": "overhead_camera_board",
            "class": "visual",
            "type": "box",
            "size": _format_vector(board_half_size),
            "pos": _format_vector(plate_center),
            "quat": _format_vector(mount_quaternion),
            "material": "overhead_camera_pcb",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        tower,
        "geom",
        {
            "name": "overhead_camera_lens",
            "class": "visual",
            "type": "cylinder",
            "size": _format_vector(lens_size),
            "pos": _format_vector(lens_center),
            "quat": _format_vector(mount_quaternion),
            "material": "overhead_camera_lens",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        tower,
        "camera",
        {
            "name": "egocentric_cam",
            "pos": _format_vector(pinhole),
            "quat": _format_vector(camera_quaternion),
            "fovy": str(camera_fovy),
        },
    )

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="unicode")


def _verify_official_assets(
    source_dir: Path,
    *,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"SO101 overhead camera-mount directory is missing: {source_dir}")
    actual: dict[str, str] = {}
    for filename, expected_sha in (expected_hashes or OFFICIAL_OVERHEAD_SOURCE_SHA256).items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"SO101 overhead camera-mount STL is missing: {path}")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SO101 overhead STL hash mismatch for {filename}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        actual[filename] = actual_sha
    return actual


def _camera1_position_world(
    rig_config: SO101CameraRigRenderConfig | None = None,
) -> tuple[float, float, float]:
    # The tower preserves the source CAD orientation.
    camera1 = rig_config.camera1 if rig_config is not None else None
    tower_rotated = camera1.camera_pinhole_cad_m if camera1 else OVERHEAD_CAMERA_PINHOLE_CAD
    tower_position = camera1.tower_position_cad_m if camera1 else OVERHEAD_TOWER_POSITION_CAD_M
    rig_world_position = camera1.rig_world_position_m if camera1 else OVERHEAD_RIG_WORLD_POSITION
    rig_local = tuple(
        tower_position[index] + tower_rotated[index]
        for index in range(3)
    )
    # +90 degrees around rig-local X: (x, y, z) -> (x, -z, y).
    rig_rotated = (rig_local[0], -rig_local[2], rig_local[1])
    return tuple(
        rig_world_position[index] + rig_rotated[index]
        for index in range(3)
    )


def _camera1_forward_cad(
    rig_config: SO101CameraRigRenderConfig | None = None,
) -> tuple[float, float, float]:
    if rig_config is None:
        return OVERHEAD_CAMERA_FORWARD_CAD
    angle = math.radians(rig_config.camera1.camera_downward_angle_degrees)
    return (math.cos(angle), -math.sin(angle), 0.0)


def _format_vector(values: tuple[float, ...]) -> str:
    return " ".join(str(value) for value in values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
