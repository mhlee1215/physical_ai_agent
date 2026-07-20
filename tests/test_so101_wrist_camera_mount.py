import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class SO101WristCameraMountTest(unittest.TestCase):
    def test_over_fixed_jaw_pose_changes_only_position(self) -> None:
        try:
            import mujoco
            import numpy as np
            from so101_nexus_core.config import PickConfig
            from so101_nexus_core.objects import CubeObject
            from so101_nexus_mujoco.pick_env import PickLiftEnv

            from physical_ai_agent.sim.so101_camera_input import (
                WRIST_CAMERA_OVER_FIXED_JAW_POSE,
                configure_wrist_camera_over_fixed_jaw,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        env = PickLiftEnv(
            config=PickConfig(objects=[CubeObject(half_size=0.015, mass=0.01, color="green")]),
            render_mode=None,
        )
        try:
            model = env.unwrapped.model
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            original_quaternion = np.asarray(model.cam_quat[camera_id]).copy()
            configure_wrist_camera_over_fixed_jaw(env)
            position = np.asarray(model.cam_pos[camera_id]).copy()
            expected_position = np.asarray(
                WRIST_CAMERA_OVER_FIXED_JAW_POSE["position"], dtype=np.float64
            )
            np.testing.assert_allclose(position, expected_position, atol=1e-12)
            self.assertAlmostEqual(
                float(position[1] - WRIST_CAMERA_OVER_FIXED_JAW_POSE["fixed_jaw_top_y"]),
                0.03,
            )
            np.testing.assert_array_equal(model.cam_quat[camera_id], original_quaternion)
        finally:
            env.close()

    def test_high_contrast_factory_applies_preset_only_when_requested(self) -> None:
        try:
            import mujoco
            import numpy as np
            from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

            from physical_ai_agent.sim.so101_camera_input import (
                WRIST_CAMERA_OVER_FIXED_JAW_POSE,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        default_env = make_high_contrast_picklift_env(target_object_color="green")
        preset_env = make_high_contrast_picklift_env(
            target_object_color="green",
            wrist_camera_mount_preset="over_fixed_jaw_rear_3cm",
        )
        try:
            default_model = default_env.unwrapped.model
            preset_model = preset_env.unwrapped.model
            default_id = mujoco.mj_name2id(default_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            preset_id = mujoco.mj_name2id(preset_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            self.assertFalse(
                np.allclose(default_model.cam_pos[default_id], preset_model.cam_pos[preset_id])
            )
            np.testing.assert_allclose(
                preset_model.cam_pos[preset_id],
                WRIST_CAMERA_OVER_FIXED_JAW_POSE["position"],
                atol=1e-12,
            )
        finally:
            preset_env.close()
            default_env.close()

    def test_wrist_pixels_are_not_rotated_in_postprocessing(self) -> None:
        import numpy as np

        from physical_ai_agent.sim.so101_camera_input import postprocess_camera_frame

        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        processed = postprocess_camera_frame("wrist_cam", pixels)
        self.assertIs(processed, pixels)

    def test_high_contrast_factory_accepts_integrated_mount_preset(self) -> None:
        try:
            import mujoco
            from train_so101_wrist_ego_visual_servo import (
                make_high_contrast_picklift_env,
            )

            from physical_ai_agent.sim.so101_wrist_camera_mount import (
                INTEGRATED_32X32_UVC_PRESET,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        env = make_high_contrast_picklift_env(
            target_object_color="green",
            object_half_sizes=(0.015,),
            wrist_camera_mount_preset=INTEGRATED_32X32_UVC_PRESET,
        )
        try:
            model = env.unwrapped.model
            mesh_names = {model.mesh(mesh_id).name for mesh_id in range(model.nmesh)}
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            self.assertIn("wrist_cam_mount_32x32_uvc", mesh_names)
            self.assertEqual(model.camera(camera_id).name, "wrist_cam")
        finally:
            env.close()

    def test_integrated_mount_replaces_only_visual_mesh_and_uses_hole_axis(self) -> None:
        try:
            import mujoco
            import numpy as np
            from so101_nexus_core.config import PickConfig
            from so101_nexus_core.objects import CubeObject

            from physical_ai_agent.sim.so101_wrist_camera_mount import (
                INNOMAKER_U20CAM_BOARD_HALF_SIZE_M,
                INNOMAKER_U20CAM_LENS_SIZE_M,
                INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES,
                INTEGRATED_32X32_UVC_BOARD_POSITION,
                INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES,
                INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER,
                INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES,
                INTEGRATED_32X32_UVC_CAMERA_MOUNT_POSITION,
                INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_GRIPPER,
                INTEGRATED_32X32_UVC_CAMERA_POSITION,
                INTEGRATED_32X32_UVC_CAMERA_QUATERNION_WXYZ,
                INTEGRATED_32X32_UVC_CAMERA_REAR_UP_DIRECTION_GRIPPER,
                INTEGRATED_32X32_UVC_CAMERA_REAR_UP_OFFSET_M,
                INTEGRATED_32X32_UVC_CAMERA_UP_GRIPPER,
                INTEGRATED_32X32_UVC_LENS_POSITION,
                make_pick_lift_env_with_integrated_32x32_uvc,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        env = make_pick_lift_env_with_integrated_32x32_uvc(
            config=PickConfig(objects=[CubeObject(half_size=0.015, mass=0.01, color="green")]),
            render_mode=None,
        )
        try:
            model = env.unwrapped.model
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
            self.assertEqual(int(model.cam_bodyid[camera_id]), gripper_id)
            np.testing.assert_allclose(
                model.cam_pos[camera_id], INTEGRATED_32X32_UVC_CAMERA_POSITION, atol=1e-8
            )
            np.testing.assert_allclose(
                np.asarray(INTEGRATED_32X32_UVC_CAMERA_MOUNT_POSITION)
                - np.asarray(INTEGRATED_32X32_UVC_CAMERA_POSITION),
                INTEGRATED_32X32_UVC_CAMERA_REAR_UP_OFFSET_M
                * np.asarray(INTEGRATED_32X32_UVC_CAMERA_REAR_UP_DIRECTION_GRIPPER),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                model.cam_quat[camera_id],
                INTEGRATED_32X32_UVC_CAMERA_QUATERNION_WXYZ,
                atol=1e-8,
            )

            camera_rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(camera_rotation, model.cam_quat[camera_id])
            rotation_matrix = camera_rotation.reshape(3, 3)
            camera_forward = rotation_matrix @ np.array([0.0, 0.0, -1.0])
            camera_up = rotation_matrix @ np.array([0.0, 1.0, 0.0])
            np.testing.assert_allclose(
                camera_forward,
                INTEGRATED_32X32_UVC_CAMERA_FORWARD_GRIPPER,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                camera_up,
                INTEGRATED_32X32_UVC_CAMERA_UP_GRIPPER,
                atol=1e-7,
            )
            target_direction = (
                np.asarray(INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_GRIPPER)
                - np.asarray(INTEGRATED_32X32_UVC_CAMERA_POSITION)
            )
            target_direction /= np.linalg.norm(target_direction)
            np.testing.assert_allclose(camera_forward, target_direction, atol=1e-7)
            self.assertAlmostEqual(
                float(
                    np.degrees(
                        np.arctan2(-camera_forward[2], camera_forward[1])
                    )
                ),
                INTEGRATED_32X32_UVC_CAMERA_DOWNWARD_ANGLE_DEGREES,
            )
            self.assertEqual(
                INTEGRATED_32X32_UVC_CAMERA_OPTICAL_TARGET_GRIPPER[0],
                INTEGRATED_32X32_UVC_CAMERA_POSITION[0],
            )
            self.assertEqual(
                INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES,
                INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES,
            )
            self.assertEqual(
                float(model.cam_fovy[camera_id]),
                INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES,
            )

            gripper_geoms = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == gripper_id
            ]
            visual_meshes = {
                model.mesh(int(model.geom_dataid[geom_id])).name
                for geom_id in gripper_geoms
                if int(model.geom_group[geom_id]) == 2 and int(model.geom_dataid[geom_id]) >= 0
            }
            collision_meshes = {
                model.mesh(int(model.geom_dataid[geom_id])).name
                for geom_id in gripper_geoms
                if int(model.geom_group[geom_id]) == 3 and int(model.geom_dataid[geom_id]) >= 0
            }
            self.assertIn("wrist_cam_mount_32x32_uvc", visual_meshes)
            self.assertIn("wrist_roll_follower_so101_v1", collision_meshes)
            board_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "wrist_camera_board"
            )
            lens_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "wrist_camera_lens"
            )
            for geom_id in (board_id, lens_id):
                self.assertEqual(int(model.geom_bodyid[geom_id]), gripper_id)
                self.assertEqual(int(model.geom_group[geom_id]), 2)
                self.assertEqual(int(model.geom_contype[geom_id]), 0)
                self.assertEqual(int(model.geom_conaffinity[geom_id]), 0)
                np.testing.assert_allclose(
                    model.geom_quat[geom_id],
                    INTEGRATED_32X32_UVC_CAMERA_QUATERNION_WXYZ,
                    atol=1e-8,
                )
            np.testing.assert_allclose(
                model.geom_pos[board_id], INTEGRATED_32X32_UVC_BOARD_POSITION, atol=1e-8
            )
            np.testing.assert_allclose(
                model.geom_size[board_id], INNOMAKER_U20CAM_BOARD_HALF_SIZE_M, atol=1e-8
            )
            np.testing.assert_allclose(
                model.geom_pos[lens_id], INTEGRATED_32X32_UVC_LENS_POSITION, atol=1e-8
            )
            np.testing.assert_allclose(
                model.geom_size[lens_id, :2], INNOMAKER_U20CAM_LENS_SIZE_M, atol=1e-8
            )
        finally:
            env.close()

    def test_integrated_mount_ascii_asset_conversion_is_deterministic(self) -> None:
        from physical_ai_agent.sim.so101_wrist_camera_mount import (
            INTEGRATED_32X32_UVC_SOURCE_SHA256,
            default_integrated_32x32_uvc_source_path,
            prepare_integrated_32x32_uvc_robot_xml,
        )

        source = default_integrated_32x32_uvc_source_path()
        if not source.is_file():
            self.skipTest(f"reviewed wrist-camera STL is not staged: {source}")
        with TemporaryDirectory() as temp_dir:
            first = prepare_integrated_32x32_uvc_robot_xml(
                source,
                output_dir=Path(temp_dir),
            )
            second = prepare_integrated_32x32_uvc_robot_xml(
                source,
                output_dir=Path(temp_dir),
            )
            self.assertEqual(first.source_sha256, INTEGRATED_32X32_UVC_SOURCE_SHA256)
            self.assertEqual(first.binary_sha256, second.binary_sha256)
            binary = Path(first.binary_stl).read_bytes()
            self.assertEqual(int.from_bytes(binary[80:84], "little"), 20_852)

    def test_brown_conrady_candidate_uses_overscan_without_changing_zero_profile(self) -> None:
        import numpy as np

        from physical_ai_agent.sim.so101_camera_input import (
            apply_brown_conrady_distortion,
            brown_conrady_overscan_fovy,
        )
        from physical_ai_agent.sim.so101_wrist_camera_mount import (
            INNOMAKER_U20CAM_CANDIDATE_DISTORTION_COEFFICIENTS,
        )

        candidate_fovy = brown_conrady_overscan_fovy(
            width=320,
            height=180,
            target_fovy_degrees=60.0,
            coefficients=INNOMAKER_U20CAM_CANDIDATE_DISTORTION_COEFFICIENTS,
        )
        self.assertGreater(candidate_fovy, 60.0)

        pixels = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
        identity = apply_brown_conrady_distortion(
            pixels,
            target_fovy_degrees=60.0,
            source_fovy_degrees=60.0,
            coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_array_equal(identity, pixels)


if __name__ == "__main__":
    unittest.main()
