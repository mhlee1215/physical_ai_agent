import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

CAMERA_RIG_CONFIG = Path(
    "configs/so101/camera_rigs/official_32x32_uvc_photoreal_v1.json"
)
CAMERA_RIG_V4_CONFIG = Path(
    "configs/so101/camera_rigs/official_32x32_uvc_photoreal_v4.json"
)
CAMERA_RIG_V5_CONFIG = Path(
    "configs/so101/camera_rigs/"
    "official_32x32_uvc_photoreal_v5_camera_matched.json"
)
CAMERA_RIG_V7_CONFIG = Path(
    "configs/so101/camera_rigs/"
    "official_32x32_uvc_photoreal_v7_fixed_base_optical_yaw.json"
)
CAMERA_RIG_V8_CONFIG = Path(
    "configs/so101/camera_rigs/"
    "official_32x32_uvc_photoreal_v8_robot_base_geometry_aligned.json"
)
CAMERA_RIG_V9_CONFIG = Path(
    "configs/so101/camera_rigs/"
    "official_32x32_uvc_photoreal_v9_white_mount_locked.json"
)
CAMERA_RIG_V10_CONFIG = Path(
    "configs/so101/camera_rigs/"
    "official_32x32_uvc_photoreal_v10_fov_calibrated_direct_square.json"
)


class SO101OfficialCameraRigTest(unittest.TestCase):
    def test_v10_camera_axes_are_perpendicular_to_their_mounting_plates(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            _rotate_vector_wxyz,
            load_so101_camera_rig_render_config,
        )
        from so101_nexus_core import get_so101_simulation_dir

        config = load_so101_camera_rig_render_config(CAMERA_RIG_V10_CONFIG)

        self.assertEqual(config.camera1.effective_vertical_fov_degrees, 50.0)
        self.assertEqual(config.camera2.effective_vertical_fov_degrees, 50.0)
        self.assertEqual(config.camera1.virtual_optical_yaw_degrees, 0.0)
        self.assertEqual(config.camera1.mount_plate_thickness_m, 0.003)
        self.assertEqual(config.camera2.mount_plate_thickness_m, 0.004)
        self.assertEqual(config.camera2.mount_downward_angle_degrees, 65.0)
        self.assertEqual(
            config.camera2.optical_downward_angle_degrees,
            config.camera2.mount_downward_angle_degrees,
        )
        downward_mast_cad = (0.0, -1.0, 0.0)
        mast_to_optical_cosine = sum(
            actual * expected
            for actual, expected in zip(
                config.camera1.camera_forward_cad,
                downward_mast_cad,
                strict=True,
            )
        )
        mast_to_optical_degrees = math.degrees(
            math.acos(max(-1.0, min(1.0, mast_to_optical_cosine)))
        )
        self.assertAlmostEqual(mast_to_optical_degrees, 25.0)
        self.assertAlmostEqual(
            config.camera1.effective_vertical_fov_degrees,
            2.0 * mast_to_optical_degrees,
        )
        downward_wrist_support = (0.0, 0.0, -1.0)
        wrist_support_to_optical_cosine = sum(
            actual * expected
            for actual, expected in zip(
                config.camera2.camera_forward_gripper,
                downward_wrist_support,
                strict=True,
            )
        )
        wrist_support_to_optical_degrees = math.degrees(
            math.acos(
                max(-1.0, min(1.0, wrist_support_to_optical_cosine))
            )
        )
        self.assertAlmostEqual(wrist_support_to_optical_degrees, 25.0)
        self.assertAlmostEqual(
            config.camera2.effective_vertical_fov_degrees,
            2.0 * wrist_support_to_optical_degrees,
        )
        root = ET.parse(get_so101_simulation_dir() / "so101_new_calib.xml").getroot()
        wrist_mount = next(
            geom
            for geom in root.findall(".//geom")
            if geom.get("class") == "visual"
            and geom.get("mesh") == config.camera2.collision_mesh
        )
        source_to_gripper = tuple(
            float(value) for value in wrist_mount.get("quat", "").split()
        )
        mount_normal_gripper = _rotate_vector_wxyz(
            source_to_gripper,
            config.camera2.source_forward,
        )
        for actual, expected in zip(
            config.camera2.camera_forward_gripper,
            mount_normal_gripper,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

        self.assertEqual(
            config.camera1.assembly_mode,
            "pcb_flush_lens_through_center_hole",
        )
        self.assertEqual(
            config.camera2.assembly_mode,
            "pcb_flush_lens_through_center_hole",
        )

    def test_v10_camera_pcbs_are_behind_the_printed_plates_without_overlap(
        self,
    ) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_V10_CONFIG)
        with TemporaryDirectory() as temp_dir:
            asset = prepare_official_32x32_uvc_camera_rig_xml(
                output_dir=Path(temp_dir),
                rig_config=config,
            )
            root = ET.parse(asset.robot_xml).getroot()

            def position(name: str, kind: str) -> tuple[float, float, float]:
                element = root.find(f'.//{kind}[@name="{name}"]')
                self.assertIsNotNone(element)
                assert element is not None
                return tuple(float(value) for value in element.get("pos", "").split())

            camera_specs = (
                (
                    "camera1",
                    config.camera1.camera_mount_face_center_cad_m,
                    config.camera1.camera_board_contact_center_cad_m,
                    config.camera1.camera_mount_face_normal_cad,
                    config.camera1.mount_plate_thickness_m,
                    position("egocentric_cam", "camera"),
                    position("overhead_camera_board", "geom"),
                    position("overhead_camera_lens", "geom"),
                ),
                (
                    "camera2",
                    config.camera2.mount_face_center_gripper_m,
                    config.camera2.board_contact_center_gripper_m,
                    config.camera2.camera_forward_gripper,
                    config.camera2.mount_plate_thickness_m,
                    position("wrist_cam", "camera"),
                    position("wrist_camera_board", "geom"),
                    position("wrist_camera_lens", "geom"),
                ),
            )
            for (
                camera_name,
                optical_face,
                board_contact,
                forward,
                plate_thickness,
                pinhole,
                board_center,
                lens_center,
            ) in camera_specs:
                with self.subTest(camera=camera_name):
                    self.assertIsNotNone(plate_thickness)
                    assert plate_thickness is not None
                    measured_thickness = sum(
                        (optical_face[index] - board_contact[index]) * forward[index]
                        for index in range(3)
                    )
                    self.assertAlmostEqual(measured_thickness, plate_thickness)
                    board_front = tuple(
                        board_center[index]
                        + config.sensor.board_half_size_m[2] * forward[index]
                        for index in range(3)
                    )
                    lens_back = tuple(
                        lens_center[index]
                        - config.sensor.lens_size_m[1] * forward[index]
                        for index in range(3)
                    )
                    lens_tip = tuple(
                        lens_center[index]
                        + config.sensor.lens_size_m[1] * forward[index]
                        for index in range(3)
                    )
                    for actual, expected in zip(
                        board_front,
                        board_contact,
                        strict=True,
                    ):
                        self.assertAlmostEqual(actual, expected)
                    for actual, expected in zip(
                        lens_back,
                        board_contact,
                        strict=True,
                    ):
                        self.assertAlmostEqual(actual, expected)
                    for actual, expected in zip(lens_tip, pinhole, strict=True):
                        self.assertAlmostEqual(actual, expected)

    def test_camera_debug_axis_matrix_points_along_the_render_forward_vector(self) -> None:
        import numpy as np

        from render_so101_official_32x32_camera_rig_preview import (
            _matrix_with_z_axis,
        )

        forward = np.asarray([0.42, 0.10, -0.90], dtype=float)
        matrix = np.asarray(_matrix_with_z_axis(forward)).reshape(3, 3)

        np.testing.assert_allclose(
            matrix[:, 2],
            forward / np.linalg.norm(forward),
            atol=1e-9,
        )
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-9)

    def test_preview_defaults_to_the_locked_v9_assembly(self) -> None:
        from render_so101_official_32x32_camera_rig_preview import DEFAULT_CONFIG_PATH

        self.assertEqual(DEFAULT_CONFIG_PATH, CAMERA_RIG_V9_CONFIG)

    def test_policy_preview_center_crops_widescreen_without_stretching(self) -> None:
        import numpy as np
        from render_so101_official_32x32_camera_rig_preview import (
            _center_crop_policy_input,
        )

        pixels = np.zeros((100, 200, 3), dtype=np.uint8)
        pixels[:, :50] = 255
        cropped = _center_crop_policy_input(pixels, size=256)

        self.assertEqual(cropped.shape, (256, 256, 3))
        self.assertEqual(int(cropped.max()), 0)

    def test_camera_rig_preview_uses_neutral_hdri_direction(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        self.assertEqual(config.render.hdri_rotation_degrees, 90.0)

    def test_direct_square_render_preserves_the_full_render(self) -> None:
        import numpy as np

        from render_so101_official_32x32_camera_rig_preview import _policy_input

        pixels = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        resized = _policy_input(pixels, size=12, mode="direct_square_render")

        np.testing.assert_array_equal(resized, pixels)

    def test_direct_square_render_rejects_non_square_source(self) -> None:
        from pydantic import ValidationError

        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            SO101CameraRigRenderConfig,
            load_so101_camera_rig_render_config,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        payload = config.model_dump(mode="json")
        payload["render"]["policy_resize"] = "direct_square_render"

        with self.assertRaises(ValidationError):
            SO101CameraRigRenderConfig.model_validate(payload)

    def test_canonical_camera_rig_config_is_strict_and_complete(self) -> None:
        from pydantic import ValidationError

        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            SO101CameraRigRenderConfig,
            load_so101_camera_rig_render_config,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        self.assertEqual(config.render.source_width, 640)
        self.assertEqual(config.render.source_height, 360)
        self.assertEqual(config.sensor.horizontal_fov_degrees, 103.0)
        self.assertEqual(config.sensor.vertical_fov_degrees, 70.533)
        self.assertIsNone(config.camera1.effective_vertical_fov_degrees)
        self.assertIsNone(config.camera2.effective_vertical_fov_degrees)
        self.assertEqual(config.render.policy_size, 256)
        self.assertEqual(config.render.policy_resize, "center_crop_square_then_resize")
        self.assertAlmostEqual(
            config.camera2.camera_position_gripper[1],
            -0.06782627866259303,
        )
        self.assertEqual(
            config.camera1.assembly_mode,
            "pcb_flush_lens_through_center_hole",
        )
        self.assertEqual(config.camera1.camera_pinhole_protrusion_m, 0.010)

        payload = config.model_dump(mode="json")
        payload["unknown_render_setting"] = True
        with self.assertRaises(ValidationError):
            SO101CameraRigRenderConfig.model_validate(payload)

        floating_camera = config.model_dump(mode="json")
        floating_camera["camera1"]["camera_pinhole_protrusion_m"] = 0.020
        with self.assertRaises(ValidationError):
            SO101CameraRigRenderConfig.model_validate(floating_camera)

        tilted_camera = config.model_dump(mode="json")
        tilted_camera["camera1"]["camera_downward_angle_degrees"] = 50.0
        with self.assertRaises(ValidationError):
            SO101CameraRigRenderConfig.model_validate(tilted_camera)

        square_source = config.model_dump(mode="json")
        square_source["render"]["source_width"] = 512
        square_source["render"]["source_height"] = 512
        with self.assertRaises(ValidationError):
            SO101CameraRigRenderConfig.model_validate(square_source)

    def test_config_values_drive_generated_camera_xml(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        sensor = config.sensor.model_copy(update={"vertical_fov_degrees": 69.0})
        camera1 = config.camera1.model_copy(
            update={
                "camera_mount_face_center_top_part_cad_m": (
                    0.030523,
                    0.362249,
                    0.0,
                ),
                "effective_vertical_fov_degrees": 66.0,
            }
        )
        camera2 = config.camera2.model_copy(
            update={
                "mount_face_center_gripper_m": (
                    0.00352405,
                    -0.07205246128,
                    0.00415252001,
                ),
                "effective_vertical_fov_degrees": 67.0,
            }
        )
        config = config.model_copy(
            update={"sensor": sensor, "camera1": camera1, "camera2": camera2}
        )

        with TemporaryDirectory() as temp_dir:
            asset = prepare_official_32x32_uvc_camera_rig_xml(
                output_dir=Path(temp_dir),
                rig_config=config,
            )
            root = ET.parse(asset.robot_xml).getroot()
            camera1_xml = root.find('.//camera[@name="egocentric_cam"]')
            camera2_xml = root.find('.//camera[@name="wrist_cam"]')
            self.assertIsNotNone(camera1_xml)
            self.assertIsNotNone(camera2_xml)
            assert camera1_xml is not None
            assert camera2_xml is not None
            self.assertEqual(float(camera1_xml.get("fovy", "nan")), 66.0)
            self.assertEqual(float(camera2_xml.get("fovy", "nan")), 67.0)
            self.assertEqual(
                tuple(float(value) for value in camera1_xml.get("pos", "").split()),
                camera1.camera_pinhole_cad_m,
            )
            self.assertEqual(
                tuple(float(value) for value in camera2_xml.get("pos", "").split()),
                camera2.camera_position_gripper,
            )

    def test_overhead_camera_pcb_is_flush_and_lens_reaches_pinhole(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        with TemporaryDirectory() as temp_dir:
            asset = prepare_official_32x32_uvc_camera_rig_xml(
                output_dir=Path(temp_dir),
                rig_config=config,
            )
            root = ET.parse(asset.robot_xml).getroot()
            camera = root.find('.//camera[@name="egocentric_cam"]')
            board = root.find('.//geom[@name="overhead_camera_board"]')
            lens = root.find('.//geom[@name="overhead_camera_lens"]')
            self.assertIsNotNone(camera)
            self.assertIsNotNone(board)
            self.assertIsNotNone(lens)
            assert camera is not None and board is not None and lens is not None

            def position(element: ET.Element) -> tuple[float, float, float]:
                return tuple(float(value) for value in element.get("pos", "").split())

            forward = config.camera1.camera_mount_face_normal_cad
            camera_position = position(camera)
            board_position = position(board)
            lens_position = position(lens)
            board_front = tuple(
                board_position[index] + config.sensor.board_half_size_m[2] * forward[index]
                for index in range(3)
            )
            lens_back = tuple(
                lens_position[index] - config.sensor.lens_size_m[1] * forward[index]
                for index in range(3)
            )
            lens_tip = tuple(
                lens_position[index] + config.sensor.lens_size_m[1] * forward[index]
                for index in range(3)
            )
            for actual, expected in zip(
                board_front,
                config.camera1.camera_mount_face_center_cad_m,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(
                lens_back,
                config.camera1.camera_mount_face_center_cad_m,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(lens_tip, camera_position, strict=True):
                self.assertAlmostEqual(actual, expected)

    def test_wrist_camera_pcb_is_flush_and_lens_reaches_pinhole(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        with TemporaryDirectory() as temp_dir:
            asset = prepare_official_32x32_uvc_camera_rig_xml(
                output_dir=Path(temp_dir),
                rig_config=config,
            )
            root = ET.parse(asset.robot_xml).getroot()
            camera = root.find('.//camera[@name="wrist_cam"]')
            board = root.find('.//geom[@name="wrist_camera_board"]')
            lens = root.find('.//geom[@name="wrist_camera_lens"]')
            self.assertIsNotNone(camera)
            self.assertIsNotNone(board)
            self.assertIsNotNone(lens)
            assert camera is not None and board is not None and lens is not None

            def position(element: ET.Element) -> tuple[float, float, float]:
                return tuple(float(value) for value in element.get("pos", "").split())

            forward = config.camera2.camera_forward_gripper
            camera_position = position(camera)
            board_position = position(board)
            lens_position = position(lens)
            board_front = tuple(
                board_position[index] + config.sensor.board_half_size_m[2] * forward[index]
                for index in range(3)
            )
            lens_back = tuple(
                lens_position[index] - config.sensor.lens_size_m[1] * forward[index]
                for index in range(3)
            )
            lens_tip = tuple(
                lens_position[index] + config.sensor.lens_size_m[1] * forward[index]
                for index in range(3)
            )
            for actual, expected in zip(
                board_front,
                config.camera2.mount_face_center_gripper_m,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(
                lens_back,
                config.camera2.mount_face_center_gripper_m,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(lens_tip, camera_position, strict=True):
                self.assertAlmostEqual(actual, expected)

    def test_config_values_drive_blender_payload(self) -> None:
        from render_so101_official_32x32_camera_rig_preview import (
            _photoreal_config,
        )

        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        payload = _photoreal_config(config, evidence_renders=())

        self.assertEqual(payload["width"], 640)
        self.assertEqual(payload["height"], 360)
        self.assertEqual(payload["samples"], 32)
        self.assertEqual(payload["cycles_seed"], 98200)
        self.assertEqual(payload["compute_device_type"], "METAL")
        self.assertEqual(payload["hdri_rotation_deg"], 90.0)
        self.assertEqual(payload["scene_profile"], "black_table_clutter")
        self.assertEqual(
            Path(payload["material_profile"]),
            Path(
                "configs/so101/render_profiles/"
                "black_arm_green_white_gripper_official_camera_rig.json"
            ).resolve(),
        )

    def test_v4_preserves_camera_pose_and_scene_contract(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )

        v1 = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        v4 = load_so101_camera_rig_render_config(CAMERA_RIG_V4_CONFIG)

        for field in ("robot", "environment", "sensor", "camera1", "camera2"):
            self.assertEqual(
                getattr(v4, field),
                getattr(v1, field),
                f"V4 must not change the reviewed {field} contract",
            )
        self.assertEqual(v4.render.source_width, 912)
        self.assertEqual(v4.render.source_height, 512)
        self.assertEqual(v4.render.policy_size, 256)
        self.assertEqual(v4.render.policy_resize, "center_crop_square_then_resize")
        self.assertEqual(v4.render.samples, 256)
        self.assertFalse(v4.render.denoise)
        self.assertEqual(v4.render.color_management, "AgX")
        self.assertEqual(v4.render.bevel_width_mm_range, (0.24, 0.32))
        self.assertEqual(v4.render.bevel_segments, 3)
        self.assertEqual(
            [asset.object_name for asset in v4.render.scene_assets],
            ["plastic_thermos", "screwdriver", "Shelf_01", "Shelf_01"],
        )
        self.assertEqual(
            [light.name for light in v4.render.lights],
            ["v4_key", "v4_fill", "v4_rim", "v4_background"],
        )

    def test_v4_values_reach_blender_payload(self) -> None:
        from render_so101_official_32x32_camera_rig_preview import (
            _photoreal_config,
        )

        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_V4_CONFIG)
        payload = _photoreal_config(config, evidence_renders=())

        self.assertEqual(payload["width"], 912)
        self.assertEqual(payload["height"], 512)
        self.assertEqual(payload["samples"], 256)
        self.assertFalse(payload["denoise"])
        self.assertEqual(payload["color_management"], "AgX")
        self.assertAlmostEqual(payload["bevel_width_range_m"][0], 0.00024)
        self.assertAlmostEqual(payload["bevel_width_range_m"][1], 0.00032)
        self.assertEqual(payload["bevel_segments"], 3)
        self.assertEqual(
            [item["object_name"] for item in payload["visual_props"]],
            ["plastic_thermos", "screwdriver", "Shelf_01", "Shelf_01"],
        )
        self.assertEqual(
            [light["name"] for light in payload["lights"]],
            ["v4_key", "v4_fill", "v4_rim", "v4_background"],
        )

    def test_v7_optical_calibration_does_not_move_physical_assembly(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        v5 = load_so101_camera_rig_render_config(CAMERA_RIG_V5_CONFIG)
        v7 = load_so101_camera_rig_render_config(CAMERA_RIG_V7_CONFIG)
        physical_fields = (
            "rig_world_position_m",
            "rig_quaternion_wxyz",
            "robot_base_world_position_m",
            "tower_position_cad_m",
            "tower_quaternion_cad_wxyz",
            "mesh_translation_cad_m",
            "mesh_quaternion_cad_wxyz",
            "camera_mount_face_center_top_part_cad_m",
            "upper_mast_translation_cad_m",
            "camera_mount_face_normal_cad",
            "camera_mount_quaternion_cad_wxyz",
            "camera_quaternion_cad_wxyz",
            "camera_pinhole_protrusion_m",
            "connector_insertion_depth_m",
            "arm_base_to_lower_mast_insertion_depth_m",
        )
        for field in physical_fields:
            self.assertEqual(
                getattr(v7.camera1, field),
                getattr(v5.camera1, field),
                f"optical calibration must not move physical field {field}",
            )
        self.assertEqual(v7.robot, v5.robot)
        self.assertEqual(v7.camera2, v5.camera2)
        self.assertEqual(v5.camera1.virtual_optical_yaw_degrees, 0.0)
        self.assertEqual(v7.camera1.virtual_optical_yaw_degrees, 6.0)
        self.assertNotEqual(
            v7.camera1.effective_camera_quaternion_cad_wxyz,
            v7.camera1.camera_quaternion_cad_wxyz,
        )

        with TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            v5_asset = prepare_official_32x32_uvc_camera_rig_xml(
                output_dir=root_dir / "v5",
                rig_config=v5,
            )
            v7_asset = prepare_official_32x32_uvc_camera_rig_xml(
                output_dir=root_dir / "v7",
                rig_config=v7,
            )
            v5_root = ET.parse(v5_asset.robot_xml).getroot()
            v7_root = ET.parse(v7_asset.robot_xml).getroot()

            def values(root: ET.Element, xpath: str, attribute: str) -> tuple[float, ...]:
                element = root.find(xpath)
                self.assertIsNotNone(element)
                assert element is not None
                return tuple(float(value) for value in element.get(attribute, "").split())

            for xpath in (
                './worldbody/body[@name="base"]',
                './worldbody/body[@name="overhead_camera_mount"]',
                './worldbody/body[@name="overhead_camera_mount"]/'
                'body[@name="overhead_camera_tower"]',
            ):
                self.assertEqual(
                    values(v7_root, xpath, "pos"),
                    values(v5_root, xpath, "pos"),
                )
            camera_xpath = (
                './worldbody/body[@name="overhead_camera_mount"]/'
                'body[@name="overhead_camera_tower"]/'
                'camera[@name="egocentric_cam"]'
            )
            self.assertEqual(
                values(v7_root, camera_xpath, "pos"),
                values(v5_root, camera_xpath, "pos"),
            )
            self.assertNotEqual(
                values(v7_root, camera_xpath, "quat"),
                values(v5_root, camera_xpath, "quat"),
            )

    def test_v8_robot_base_fastener_holes_match_the_printed_mount(self) -> None:
        import numpy as np
        import trimesh
        from so101_nexus_core import get_so101_simulation_dir

        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
            resolve_repository_path,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_V8_CONFIG)
        camera1 = config.camera1

        arm_base = trimesh.load_mesh(
            resolve_repository_path(config.assets.overhead_stl_dir) / "arm_base.stl",
            process=False,
        )
        arm_base.apply_scale(config.assets.overhead_mesh_scale)
        arm_base_transform = trimesh.transformations.quaternion_matrix(
            camera1.rig_quaternion_wxyz
        )
        arm_base_transform[:3, 3] = camera1.rig_world_position_m
        arm_base.apply_transform(arm_base_transform)

        robot_base = trimesh.load_mesh(
            get_so101_simulation_dir() / "assets" / "base_so101_v2.stl",
            process=False,
        )
        robot_base_transform = trimesh.transformations.quaternion_matrix(
            (0.5, 0.5, 0.5, 0.5)
        )
        robot_base_transform[:3, 3] = (
            -0.00636471,
            -8.97657e-09,
            -0.0024,
        )
        robot_base.apply_transform(robot_base_transform)
        robot_base.apply_translation(camera1.robot_base_world_position_m)

        def screw_hole_centers(
            mesh: trimesh.Trimesh,
            *,
            z: float,
            x_extent: tuple[float, float],
            y_extent: tuple[float, float],
        ) -> np.ndarray:
            section = mesh.section(
                plane_origin=(0.0, 0.0, z),
                plane_normal=(0.0, 0.0, 1.0),
            )
            self.assertIsNotNone(section)
            assert section is not None
            centers = []
            for loop in section.discrete:
                xy = np.asarray(loop)[:, :2]
                extent = xy.max(axis=0) - xy.min(axis=0)
                center = xy.mean(axis=0)
                if (
                    x_extent[0] < extent[0] < x_extent[1]
                    and y_extent[0] < extent[1] < y_extent[1]
                    and abs(center[0]) > 0.02
                    and abs(center[1]) > 0.02
                ):
                    centers.append(center)
            self.assertEqual(len(centers), 4)
            return np.asarray(
                sorted(
                    centers,
                    key=lambda center: (round(float(center[0]), 3), center[1]),
                )
            )

        arm_holes = screw_hole_centers(
            arm_base,
            z=float(arm_base.bounds[:, 2].mean()),
            x_extent=(0.0045, 0.0055),
            y_extent=(0.0045, 0.0055),
        )
        robot_holes = screw_hole_centers(
            robot_base,
            z=0.010,
            x_extent=(0.0055, 0.0065),
            y_extent=(0.0045, 0.0055),
        )
        hole_error = np.linalg.norm(arm_holes - robot_holes, axis=1)
        self.assertLess(
            float(hole_error.max()),
            0.00005,
            "all four base screw holes must align within 0.05 mm",
        )
        self.assertGreaterEqual(
            float(robot_base.bounds[0, 0]),
            float(arm_base.bounds[0, 0]) - 0.001,
        )
        self.assertLessEqual(
            float(robot_base.bounds[1, 0]),
            float(arm_base.bounds[1, 0]) + 1e-9,
        )
        self.assertGreaterEqual(
            float(robot_base.bounds[0, 1]),
            float(arm_base.bounds[0, 1]),
        )
        self.assertLessEqual(
            float(robot_base.bounds[1, 1]),
            float(arm_base.bounds[1, 1]),
        )
        self.assertAlmostEqual(
            camera1.robot_base_world_position_m[2],
            float(arm_base.bounds[1, 2]),
            places=8,
        )

    def test_v9_assembly_lock_rejects_physical_transform_drift(self) -> None:
        from pydantic import ValidationError

        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            SO101CameraRigRenderConfig,
            load_so101_camera_rig_render_config,
        )

        v8 = load_so101_camera_rig_render_config(CAMERA_RIG_V8_CONFIG)
        v9 = load_so101_camera_rig_render_config(CAMERA_RIG_V9_CONFIG)
        self.assertIsNotNone(v9.assembly_lock)
        assert v9.assembly_lock is not None
        self.assertEqual(v9.assembly_lock.state, "locked")
        self.assertEqual(
            v9.camera1.robot_base_world_position_m,
            v8.camera1.robot_base_world_position_m,
        )
        self.assertEqual(v9.camera1.tower_position_cad_m, v8.camera1.tower_position_cad_m)
        self.assertEqual(
            v9.camera1.mesh_translation_cad_m,
            v8.camera1.mesh_translation_cad_m,
        )
        self.assertEqual(v9.camera2.model_dump(), v8.camera2.model_dump())
        self.assertEqual(
            v9.assembly_lock.fastener_alignment.robot_base_mesh_sha256,
            "bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d",
        )

        payload = v9.model_dump(mode="json")
        payload["camera1"]["robot_base_world_position_m"][0] += 0.001
        with self.assertRaisesRegex(ValidationError, "assembly lock mismatch"):
            SO101CameraRigRenderConfig.model_validate(payload)

        payload = v9.model_dump(mode="json")
        payload["camera1"]["mesh_translation_cad_m"]["cam_mount_middle.stl"][1] += 0.001
        with self.assertRaisesRegex(ValidationError, "assembly lock mismatch"):
            SO101CameraRigRenderConfig.model_validate(payload)

    def test_v9_whitens_only_the_printed_base_and_camera_tower(self) -> None:
        old_profile = json.loads(
            Path(
                "configs/so101/render_profiles/"
                "black_arm_green_white_gripper_official_camera_rig_v4.json"
            ).read_text(encoding="utf-8")
        )
        new_profile = json.loads(
            Path(
                "configs/so101/render_profiles/"
                "black_arm_green_white_gripper_white_mount_locked_v1.json"
            ).read_text(encoding="utf-8")
        )
        mount_parts = {
            "overhead_arm_base",
            "overhead_mount_bottom",
            "overhead_mount_middle",
            "overhead_mount_top",
        }
        for part_name in mount_parts:
            self.assertEqual(
                new_profile["parts"][part_name]["material"],
                "white_matte_pla",
            )
        for part_name, part in old_profile["parts"].items():
            if part_name not in mount_parts:
                self.assertEqual(new_profile["parts"][part_name], part)

    def test_v4_pbr_material_assets_are_hash_verified(self) -> None:
        from render_so101_dataset_blender_preview import (
            _load_robot_material_config,
        )

        profile_path = Path(
            "configs/so101/render_profiles/"
            "black_arm_green_white_gripper_official_camera_rig_v4.json"
        ).resolve()
        profile = _load_robot_material_config(profile_path)

        assert profile is not None
        black = profile["materials"]["black_matte_pla"]
        for key in (
            "roughness_texture",
            "normal_texture",
            "wear_mask_texture",
            "fingerprint_mask_texture",
        ):
            self.assertTrue(Path(black[key]).is_file())

    def test_official_stl_hashes_and_generated_manifest_are_reproducible(self) -> None:
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            OFFICIAL_OVERHEAD_SOURCE_SHA256,
            default_official_overhead_source_dir,
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        source_dir = default_official_overhead_source_dir()
        if not source_dir.is_dir():
            self.skipTest(f"reviewed overhead-camera STLs are not staged: {source_dir}")
        with TemporaryDirectory() as temp_dir:
            asset = prepare_official_32x32_uvc_camera_rig_xml(
                source_dir,
                output_dir=Path(temp_dir),
            )
            self.assertEqual(asset.source_sha256, OFFICIAL_OVERHEAD_SOURCE_SHA256)
            manifest = json.loads(Path(asset.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["camera_contract"],
                {
                    "observation.images.camera1": "egocentric_cam",
                    "observation.images.camera2": "wrist_cam",
                },
            )
            self.assertEqual(manifest["camera1_pixel_postprocess_rotation_degrees"], 0)
            self.assertEqual(
                manifest["camera2_optical_axis"]["assembly_mode"],
                "pcb_flush_lens_through_center_hole",
            )
            self.assertEqual(
                manifest["camera2_optical_axis"]["lens_protrusion_m"],
                0.010,
            )
            self.assertEqual(
                manifest["camera2_optical_axis"]["downward_angle_degrees"],
                65.0,
            )
            self.assertEqual(
                manifest["camera1_optical_axis"]["assembly_mode"],
                "pcb_flush_lens_through_center_hole",
            )
            self.assertEqual(
                manifest["camera1_optical_axis"]["lens_protrusion_m"],
                0.010,
            )
            self.assertEqual(
                manifest["camera1_optical_axis"]["downward_angle_degrees"],
                65.0,
            )

    def test_camera_rig_uses_static_overhead_camera_and_shared_hardware_fov(self) -> None:
        try:
            import mujoco
            import numpy as np
            from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

            from physical_ai_agent.sim.so101_camera_input import _make_camera
            from physical_ai_agent.sim.so101_overhead_camera_mount import (
                OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
                OFFICIAL_OVERHEAD_CAMERA_FOVY_DEGREES,
                OVERHEAD_CAMERA_FORWARD_WORLD,
                OVERHEAD_CAMERA_UP_WORLD,
            )
            from physical_ai_agent.sim.so101_wrist_camera_mount import (
                INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        env = make_high_contrast_picklift_env(
            target_object_color="green",
            object_half_sizes=(0.015,),
            camera_rig_preset=OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
        )
        try:
            env.reset(seed=0)
            model = env.unwrapped.model
            data = env.unwrapped.data
            ego_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "egocentric_cam")
            wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            self.assertEqual(
                model.body(int(model.cam_bodyid[ego_id])).name,
                "overhead_camera_tower",
            )
            self.assertEqual(float(model.cam_fovy[ego_id]), OFFICIAL_OVERHEAD_CAMERA_FOVY_DEGREES)
            self.assertEqual(
                float(model.cam_fovy[wrist_id]),
                INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES,
            )
            self.assertEqual(float(model.cam_fovy[ego_id]), 70.533)
            self.assertEqual(float(model.cam_fovy[wrist_id]), 70.533)
            rotation = np.asarray(data.cam_xmat[ego_id]).reshape(3, 3)
            np.testing.assert_allclose(
                rotation @ np.array([0.0, 0.0, -1.0]),
                OVERHEAD_CAMERA_FORWARD_WORLD,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                rotation @ np.array([0.0, 1.0, 0.0]),
                OVERHEAD_CAMERA_UP_WORLD,
                atol=1e-7,
            )
            self.assertEqual(_make_camera(env, "egocentric_cam"), "egocentric_cam")
        finally:
            env.close()

    def test_camera_rig_adds_only_visual_overhead_meshes(self) -> None:
        try:
            import mujoco
            from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

            from physical_ai_agent.sim.so101_overhead_camera_mount import (
                OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        env = make_high_contrast_picklift_env(
            target_object_color="green",
            object_half_sizes=(0.015,),
            camera_rig_preset=OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
        )
        try:
            model = env.unwrapped.model
            overhead_body_ids = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in (
                    "overhead_arm_base",
                    "overhead_camera_mount",
                    "overhead_camera_tower",
                )
            }
            overhead_geom_ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in overhead_body_ids
            ]
            self.assertGreaterEqual(len(overhead_geom_ids), 6)
            self.assertTrue(
                all(
                    int(model.geom_group[geom_id]) == 2
                    for geom_id in overhead_geom_ids
                )
            )
            mesh_names = {
                model.mesh(int(model.geom_dataid[geom_id])).name
                for geom_id in overhead_geom_ids
                if int(model.geom_dataid[geom_id]) >= 0
            }
            self.assertEqual(
                mesh_names,
                {
                    "overhead_arm_base_32x32_uvc",
                    "overhead_cam_mount_bottom_32x32_uvc",
                    "overhead_cam_mount_middle_32x32_uvc",
                    "overhead_cam_mount_top_32x32_uvc",
                },
            )
        finally:
            env.close()

    def test_camera1_does_not_render_its_own_board_or_lens(self) -> None:
        try:
            import mujoco
            import numpy as np
            from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

            from physical_ai_agent.sim.so101_overhead_camera_mount import (
                OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        env = make_high_contrast_picklift_env(
            target_object_color="green",
            object_half_sizes=(0.015,),
            camera_rig_preset=OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
        )
        renderer = None
        try:
            env.reset(seed=50_000_000)
            model = env.unwrapped.model
            data = env.unwrapped.data
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 320)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), 180)
            renderer = mujoco.Renderer(model, height=180, width=320)
            renderer.update_scene(data, camera="egocentric_cam")
            renderer.enable_segmentation_rendering()
            segmentation = np.asarray(renderer.render())

            camera_module_geom_ids = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("overhead_camera_board", "overhead_camera_lens")
            }
            is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            visible_camera_module_pixels = sum(
                int(np.count_nonzero(is_geom & (segmentation[..., 0] == geom_id)))
                for geom_id in camera_module_geom_ids
            )
            self.assertEqual(
                visible_camera_module_pixels,
                0,
                "the camera pinhole must sit beyond its own flush PCB and lens",
            )
        finally:
            if renderer is not None:
                renderer.close()
            env.close()

    def test_photoreal_profile_preserves_robot_colors_and_adds_mount_materials(self) -> None:
        profile_path = Path(
            "configs/so101/render_profiles/black_arm_green_white_gripper_official_camera_rig.json"
        )
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        self.assertEqual(profile["default_material"], "black_matte_pla")
        self.assertEqual(profile["parts"]["fixed_jaw"]["material"], "green_matte_pla")
        self.assertEqual(profile["parts"]["moving_jaw"]["material"], "white_matte_pla")
        self.assertEqual(profile["parts"]["overhead_mount_bottom"]["material"], "yellow_matte_pla")
        self.assertEqual(profile["parts"]["overhead_mount_top"]["material"], "black_matte_pla")
        self.assertEqual(profile["parts"]["wrist_camera_board"]["material"], "camera_pcb_black")
        self.assertEqual(profile["parts"]["wrist_camera_lens"]["material"], "camera_lens_black")

    def test_photoreal_camera_specs_overscan_before_distortion(self) -> None:
        from render_so101_dataset_blender_preview import (
            _camera_specs_with_distortion_overscan,
        )

        target = {
            "observation.images.camera2": {
                "fovy": 60.0,
                "intrinsics": {},
                "distortion": None,
            }
        }
        profile = {
            "model": "opencv_brown_conrady",
            "coefficients": [-0.08, 0.01, 0.0, 0.0, 0.0],
            "calibration_status": "uncalibrated_candidate",
        }
        rendered = _camera_specs_with_distortion_overscan(
            target,
            distortion_profiles={"observation.images.camera2": profile},
            width=320,
            height=180,
        )
        self.assertGreater(rendered["observation.images.camera2"]["fovy"], 60.0)
        self.assertEqual(target["observation.images.camera2"]["distortion"], profile)


if __name__ == "__main__":
    unittest.main()
