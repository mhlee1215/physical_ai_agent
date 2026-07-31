from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from physical_ai_agent.sim.so101_camera_rig_render_config import (
    SO101CameraRigRenderConfig,
    resolve_repository_path,
)

INTEGRATED_32X32_UVC_PRESET = "integrated_32x32_uvc"
INTEGRATED_32X32_UVC_SOURCE_SHA256 = (
    "b4345ccf23f1f2ed3f4885c205cac5afbed6ddd1b183617c4801751e3bafb7b4"
)
INTEGRATED_32X32_UVC_MESH_SCALE = (0.001, 0.001, 0.001)

# The B0CNCSFQC1 listing labels 130 degrees as diagonal FOV (D) and 103
# degrees as horizontal FOV (H), not vertical FOV. A 16:9 rectilinear
# projection at H=103 degrees gives V=70.533 degrees. Barrel distortion accounts
# for additional diagonal coverage and remains an uncalibrated candidate below.
INNOMAKER_U20CAM_SOURCE_RESOLUTION = (1920, 1080)
INNOMAKER_U20CAM_HORIZONTAL_FOV_DEGREES = 103.0
INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES = 70.533
INNOMAKER_U20CAM_DIAGONAL_FOV_REPORTED_DEGREES = (120.0, 130.0)
# The product page does not publish calibrated coefficients. This weak barrel
# profile is therefore a replaceable preview candidate, not measured hardware
# calibration. OpenCV order: (k1, k2, p1, p2, k3).
INNOMAKER_U20CAM_DISTORTION_MODEL = "opencv_brown_conrady"
INNOMAKER_U20CAM_CANDIDATE_DISTORTION_COEFFICIENTS = (-0.08, 0.01, 0.0, 0.0, 0.0)
INNOMAKER_U20CAM_DISTORTION_CALIBRATION_STATUS = "uncalibrated_candidate"

# Shared low-detail visual envelope for the camera module mounted in both the
# wrist and overhead printed frames: a 32 x 32 mm board with one central lens.
INNOMAKER_U20CAM_BOARD_HALF_SIZE_M = (0.016, 0.016, 0.0015)
INNOMAKER_U20CAM_LENS_SIZE_M = (0.008, 0.005)
INNOMAKER_U20CAM_BOARD_DISTANCE_BEHIND_PINHOLE_M = 0.0115
INNOMAKER_U20CAM_LENS_DISTANCE_BEHIND_PINHOLE_M = 0.005

# Render the same full 16:9 field as the connected camera. Policy preprocessing
# center-crops that frame to a square before the 256x256 resize.
INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES = INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES
INTEGRATED_32X32_UVC_CAMERA_HORIZONTAL_FOV_DEGREES = (
    INNOMAKER_U20CAM_HORIZONTAL_FOV_DEGREES
)

# Measured from the four M2 board holes and the mounting plane in the official STL.
INTEGRATED_32X32_UVC_CAMERA_CENTER_SOURCE_MM = (
    2.52405,
    71.83424728,
    -3.20281401,
)
INTEGRATED_32X32_UVC_CAMERA_FORWARD_SOURCE = (
    0.0,
    -0.42261826174069944,
    0.9063077870366499,
)

# The source center is the optical/front face of the printed plate. The PCB
# mounts against the opposite face, 4 mm behind it. The 10 mm lens barrel starts
# at the PCB, passes through the plate opening, and ends at the optical pinhole.
INTEGRATED_32X32_UVC_MOUNT_FACE_CENTER_GRIPPER = (
    0.00252405,
    -0.07205246128,
    0.00415252001,
)
INTEGRATED_32X32_UVC_MOUNT_PLATE_THICKNESS_M = 0.004
INTEGRATED_32X32_UVC_LENS_PROTRUSION_M = 0.010

# The camera PCB is flush with the printed 65-degree mounting face, so its
# optical axis is the face normal. Framing calibration must move the physical
# pinhole or adjust the reviewed effective crop; tilting the lens independently
# would describe an impossible assembly.
INTEGRATED_32X32_UVC_CAMERA_MOUNT_DOWNWARD_ANGLE_DEGREES = 65.0
INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES = 65.0
_INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_RADIANS = math.radians(
    INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES
)
INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER = (
    0.0,
    math.cos(_INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_RADIANS),
    -math.sin(_INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_RADIANS),
)
INTEGRATED_32X32_UVC_CAMERA_UP_GRIPPER = (
    0.0,
    -math.sin(_INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_RADIANS),
    -math.cos(_INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_RADIANS),
)
INTEGRATED_32X32_UVC_BOARD_CONTACT_CENTER_GRIPPER = tuple(
    INTEGRATED_32X32_UVC_MOUNT_FACE_CENTER_GRIPPER[index]
    - INTEGRATED_32X32_UVC_MOUNT_PLATE_THICKNESS_M
    * INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER[index]
    for index in range(3)
)
INTEGRATED_32X32_UVC_CAMERA_POSITION = tuple(
    INTEGRATED_32X32_UVC_BOARD_CONTACT_CENTER_GRIPPER[index]
    + INTEGRATED_32X32_UVC_LENS_PROTRUSION_M
    * INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER[index]
    for index in range(3)
)
INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_DISTANCE_M = 0.15060335474204708
INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_GRIPPER = tuple(
    INTEGRATED_32X32_UVC_CAMERA_POSITION[index]
    + INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_DISTANCE_M
    * INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER[index]
    for index in range(3)
)
INTEGRATED_32X32_UVC_CAMERA_OPTICAL_AXIS_OFFSET_DEGREES = abs(
    INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES
    - INTEGRATED_32X32_UVC_CAMERA_MOUNT_DOWNWARD_ANGLE_DEGREES
)
_INTEGRATED_32X32_UVC_CAMERA_HALF_ROLL_PITCH_RADIANS = math.radians(
    (90.0 - INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES) / 2.0
)
INTEGRATED_32X32_UVC_CAMERA_QUATERNION_WXYZ = (
    0.0,
    0.0,
    -math.sin(_INTEGRATED_32X32_UVC_CAMERA_HALF_ROLL_PITCH_RADIANS),
    math.cos(_INTEGRATED_32X32_UVC_CAMERA_HALF_ROLL_PITCH_RADIANS),
)
INTEGRATED_32X32_UVC_BOARD_POSITION = tuple(
    INTEGRATED_32X32_UVC_CAMERA_POSITION[index]
    - INNOMAKER_U20CAM_BOARD_DISTANCE_BEHIND_PINHOLE_M
    * INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER[index]
    for index in range(3)
)
INTEGRATED_32X32_UVC_LENS_POSITION = tuple(
    INTEGRATED_32X32_UVC_CAMERA_POSITION[index]
    - INNOMAKER_U20CAM_LENS_DISTANCE_BEHIND_PINHOLE_M
    * INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER[index]
    for index in range(3)
)

_ROBOT_XML_BUILD_LOCK = threading.Lock()


@dataclass(frozen=True)
class SO101IntegratedWristCameraAsset:
    source_stl: str
    source_sha256: str
    binary_stl: str
    binary_sha256: str
    robot_xml: str
    manifest_path: str
    camera_position_gripper: tuple[float, float, float]
    camera_quaternion_wxyz: tuple[float, float, float, float]
    camera_forward_gripper: tuple[float, float, float]
    camera_up_gripper: tuple[float, float, float]


def default_integrated_32x32_uvc_source_path() -> Path:
    configured = os.environ.get("SO101_32X32_UVC_MOUNT_STL")
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "_workspace"
        / "so101_camera_mount_assets"
        / "Wrist_Cam_Mount_32x32_UVC_Module_SO101.stl"
    )


def prepare_integrated_32x32_uvc_robot_xml(
    mount_stl_path: Path | None = None,
    *,
    output_dir: Path | None = None,
    rig_config: SO101CameraRigRenderConfig | None = None,
) -> SO101IntegratedWristCameraAsset:
    from so101_nexus_core import get_so101_simulation_dir

    configured_source = (
        resolve_repository_path(rig_config.assets.wrist_stl_path)
        if rig_config is not None
        else default_integrated_32x32_uvc_source_path()
    )
    source_stl = (mount_stl_path or configured_source).resolve()
    if not source_stl.is_file():
        raise FileNotFoundError(
            f"SO101 integrated wrist-camera STL is missing: {source_stl}. "
            "Set SO101_32X32_UVC_MOUNT_STL or stage the reviewed STL in _workspace."
        )
    source_sha256 = _sha256(source_stl)
    expected_source_sha256 = (
        rig_config.assets.wrist_stl_sha256
        if rig_config is not None
        else INTEGRATED_32X32_UVC_SOURCE_SHA256
    )
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "SO101 integrated wrist-camera STL does not match the reviewed asset: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )

    repo_root = Path(__file__).resolve().parents[3]
    configured_build_dir = (
        resolve_repository_path(rig_config.assets.generated_asset_dir)
        if rig_config is not None
        else repo_root / "_workspace" / "so101_camera_mount_assets" / "generated"
    )
    build_dir = (output_dir or configured_build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_stl = build_dir / "Wrist_Cam_Mount_32x32_UVC_Module_SO101_binary.stl"
    robot_xml = build_dir / "so101_new_calib_32x32_uvc.xml"
    manifest_path = build_dir / "integrated_32x32_uvc_manifest.json"

    _convert_ascii_stl_to_binary(source_stl, binary_stl)
    source_robot_xml = get_so101_simulation_dir() / "so101_new_calib.xml"
    wrist = rig_config.camera2 if rig_config is not None else None
    sensor = rig_config.sensor if rig_config is not None else None
    camera_position = (
        wrist.camera_position_gripper
        if wrist is not None
        else INTEGRATED_32X32_UVC_CAMERA_POSITION
    )
    camera_quaternion = (
        wrist.camera_quaternion_wxyz
        if wrist is not None
        else INTEGRATED_32X32_UVC_CAMERA_QUATERNION_WXYZ
    )
    camera_forward = (
        wrist.camera_forward_gripper
        if wrist is not None
        else INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER
    )
    camera_up = (
        wrist.camera_up_gripper
        if wrist is not None
        else INTEGRATED_32X32_UVC_CAMERA_UP_GRIPPER
    )
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
    board_position = tuple(
        camera_position[index]
        - board_distance * camera_forward[index]
        for index in range(3)
    )
    lens_position = tuple(
        camera_position[index]
        - lens_distance * camera_forward[index]
        for index in range(3)
    )
    mesh_scale = (
        rig_config.assets.wrist_mesh_scale
        if rig_config is not None
        else INTEGRATED_32X32_UVC_MESH_SCALE
    )
    camera_fovy = (
        wrist.effective_vertical_fov_degrees
        if wrist is not None and wrist.effective_vertical_fov_degrees is not None
        else (
            sensor.vertical_fov_degrees
            if sensor is not None
            else INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES
        )
    )
    board_half_size = (
        sensor.board_half_size_m if sensor is not None else INNOMAKER_U20CAM_BOARD_HALF_SIZE_M
    )
    lens_size = sensor.lens_size_m if sensor is not None else INNOMAKER_U20CAM_LENS_SIZE_M
    _build_robot_xml(
        source_robot_xml,
        binary_stl,
        robot_xml,
        mesh_scale=mesh_scale,
        camera_position=camera_position,
        camera_quaternion=camera_quaternion,
        camera_fovy=camera_fovy,
        board_half_size=board_half_size,
        board_position=board_position,
        lens_size=lens_size,
        lens_position=lens_position,
    )

    asset = SO101IntegratedWristCameraAsset(
        source_stl=str(source_stl),
        source_sha256=source_sha256,
        binary_stl=str(binary_stl),
        binary_sha256=_sha256(binary_stl),
        robot_xml=str(robot_xml),
        manifest_path=str(manifest_path),
        camera_position_gripper=camera_position,
        camera_quaternion_wxyz=camera_quaternion,
        camera_forward_gripper=camera_forward,
        camera_up_gripper=camera_up,
    )
    payload = asdict(asset)
    payload.update(
        {
            "schema_version": 1,
            "preset": INTEGRATED_32X32_UVC_PRESET,
            "status": "review_candidate_not_applied_to_existing_datasets",
            "mesh_scale": list(mesh_scale),
            "camera_center_source_mm": list(
                wrist.source_center_mm if wrist else INTEGRATED_32X32_UVC_CAMERA_CENTER_SOURCE_MM
            ),
            "camera_forward_source": list(
                wrist.source_forward if wrist else INTEGRATED_32X32_UVC_CAMERA_FORWARD_SOURCE
            ),
            "camera_mount_face_center_gripper": list(
                wrist.mount_face_center_gripper_m
                if wrist
                else INTEGRATED_32X32_UVC_MOUNT_FACE_CENTER_GRIPPER
            ),
            "camera_board_contact_center_gripper": list(
                wrist.board_contact_center_gripper_m
                if wrist
                else INTEGRATED_32X32_UVC_BOARD_CONTACT_CENTER_GRIPPER
            ),
            "camera_mount_plate_thickness_m": (
                wrist.mount_plate_thickness_m
                if wrist and wrist.mount_plate_thickness_m is not None
                else INTEGRATED_32X32_UVC_MOUNT_PLATE_THICKNESS_M
            ),
            "camera_lens_protrusion_m": (
                wrist.lens_protrusion_m
                if wrist
                else INTEGRATED_32X32_UVC_LENS_PROTRUSION_M
            ),
            "camera_assembly_mode": (
                wrist.assembly_mode
                if wrist
                else "pcb_flush_lens_through_center_hole"
            ),
            "camera_downward_angle_degrees": (
                wrist.optical_downward_angle_degrees
                if wrist
                else INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES
            ),
            "camera_optical_target_gripper": list(
                wrist.optical_target_gripper
                if wrist
                else INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_GRIPPER
            ),
            "camera_optical_axis_offset_degrees": (
                abs(wrist.optical_downward_angle_degrees - wrist.mount_downward_angle_degrees)
                if wrist
                else INTEGRATED_32X32_UVC_CAMERA_OPTICAL_AXIS_OFFSET_DEGREES
            ),
            "camera_body": wrist.camera_body if wrist else "gripper",
            "camera_fovy_degrees": camera_fovy,
            "effective_horizontal_fov_degrees": (
                sensor.horizontal_fov_degrees
                if sensor
                else INTEGRATED_32X32_UVC_CAMERA_HORIZONTAL_FOV_DEGREES
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
            "reported_diagonal_fov_degrees": list(
                sensor.reported_diagonal_fov_degrees
                if sensor
                else INNOMAKER_U20CAM_DIAGONAL_FOV_REPORTED_DEGREES
            ),
            "fov_contract": "reviewed_camera2_effective_crop",
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
            },
            "policy_resize": "center_crop_square_then_resize_to_256x256",
            "camera_mount_roll_degrees": 180,
            "camera_visual": {
                "board_half_size_m": list(board_half_size),
                "lens_size_m": list(lens_size),
                "board_position_gripper": list(board_position),
                "lens_position_gripper": list(lens_position),
            },
            "collision_mesh": wrist.collision_mesh if wrist else "wrist_roll_follower_so101_v1",
            "pixel_postprocess_rotation_degrees": (
                wrist.pixel_postprocess_rotation_degrees if wrist else 0
            ),
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return asset


def make_pick_lift_env_with_integrated_32x32_uvc(
    *,
    config: Any,
    render_mode: str | None = None,
    mount_stl_path: Path | None = None,
    rig_config: SO101CameraRigRenderConfig | None = None,
    **kwargs: Any,
) -> Any:
    import so101_nexus_mujoco.pick_env as pick_env

    asset = prepare_integrated_32x32_uvc_robot_xml(
        mount_stl_path,
        rig_config=rig_config,
    )
    with _ROBOT_XML_BUILD_LOCK:
        original_robot_xml = pick_env._SO101_XML
        try:
            pick_env._SO101_XML = Path(asset.robot_xml)
            env = pick_env.PickLiftEnv(
                config=config,
                render_mode=render_mode,
                **kwargs,
            )
        finally:
            pick_env._SO101_XML = original_robot_xml
    return env


def _build_robot_xml(
    source_xml: Path,
    binary_stl: Path,
    output_xml: Path,
    *,
    mesh_scale: tuple[float, float, float],
    camera_position: tuple[float, float, float],
    camera_quaternion: tuple[float, float, float, float],
    camera_fovy: float,
    board_half_size: tuple[float, float, float],
    board_position: tuple[float, float, float],
    lens_size: tuple[float, float],
    lens_position: tuple[float, float, float],
) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find("./compiler")
    if compiler is None:
        raise ValueError(f"SO101 robot XML has no compiler element: {source_xml}")
    mesh_dir = compiler.attrib.pop("meshdir", "")
    for mesh in root.findall("./asset/mesh"):
        mesh_file = mesh.get("file")
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh.set("file", str((source_xml.parent / mesh_dir / mesh_file).resolve()))

    asset = root.find("./asset")
    if asset is None:
        raise ValueError(f"SO101 robot XML has no asset element: {source_xml}")
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "wrist_cam_mount_32x32_uvc",
            "file": str(binary_stl.resolve()),
            "scale": " ".join(str(value) for value in mesh_scale),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "wrist_camera_pcb", "rgba": "0.018 0.025 0.022 1"},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "wrist_camera_lens", "rgba": "0.008 0.010 0.014 1"},
    )

    visual_geoms = [
        geom
        for geom in root.findall(".//geom")
        if geom.get("class") == "visual" and geom.get("mesh") == "wrist_roll_follower_so101_v1"
    ]
    if len(visual_geoms) != 1:
        raise ValueError(
            "expected exactly one visual wrist_roll_follower_so101_v1 geom, "
            f"found {len(visual_geoms)}"
        )
    visual_geoms[0].set("mesh", "wrist_cam_mount_32x32_uvc")

    camera = root.find('.//body[@name="gripper"]/camera[@name="wrist_cam"]')
    if camera is None:
        raise ValueError("SO101 robot XML has no gripper-attached wrist_cam")
    camera.attrib.pop("euler", None)
    camera.set("pos", " ".join(str(value) for value in camera_position))
    camera.set(
        "quat",
        " ".join(str(value) for value in camera_quaternion),
    )
    camera.set("fovy", str(camera_fovy))

    gripper = root.find('.//body[@name="gripper"]')
    if gripper is None:
        raise ValueError("SO101 robot XML has no gripper body")
    ET.SubElement(
        gripper,
        "geom",
        {
            "name": "wrist_camera_board",
            "class": "visual",
            "type": "box",
            "size": " ".join(str(value) for value in board_half_size),
            "pos": " ".join(str(value) for value in board_position),
            "quat": " ".join(str(value) for value in camera_quaternion),
            "material": "wrist_camera_pcb",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        gripper,
        "geom",
        {
            "name": "wrist_camera_lens",
            "class": "visual",
            "type": "cylinder",
            "size": " ".join(str(value) for value in lens_size),
            "pos": " ".join(str(value) for value in lens_position),
            "quat": " ".join(str(value) for value in camera_quaternion),
            "material": "wrist_camera_lens",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="unicode")


def _convert_ascii_stl_to_binary(source: Path, output: Path) -> None:
    source_sha = _sha256(source)
    stamp = output.with_suffix(output.suffix + ".source_sha256")
    if output.is_file() and stamp.is_file() and stamp.read_text().strip() == source_sha:
        return

    triangles: list[tuple[tuple[float, float, float], list[tuple[float, float, float]]]] = []
    normal = (0.0, 0.0, 0.0)
    vertices: list[tuple[float, float, float]] = []
    with source.open("r", encoding="ascii") as stream:
        for raw_line in stream:
            tokens = raw_line.strip().split()
            if len(tokens) == 5 and tokens[:2] == ["facet", "normal"]:
                normal = tuple(float(value) for value in tokens[2:5])
                vertices = []
            elif len(tokens) == 4 and tokens[0] == "vertex":
                vertices.append(tuple(float(value) for value in tokens[1:4]))
            elif tokens == ["endfacet"]:
                if len(vertices) != 3:
                    raise ValueError(f"invalid ASCII STL facet in {source}: expected 3 vertices")
                triangles.append((normal, vertices))

    if not triangles:
        raise ValueError(f"no triangles found in ASCII STL: {source}")
    with output.open("wb") as stream:
        header = b"SO101 integrated 32x32 UVC wrist camera mount"
        stream.write(header.ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for facet_normal, facet_vertices in triangles:
            values = [*facet_normal]
            for vertex in facet_vertices:
                values.extend(vertex)
            stream.write(struct.pack("<12fH", *values, 0))
    stamp.write_text(source_sha + "\n", encoding="ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
