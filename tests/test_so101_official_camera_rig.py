import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

CAMERA_RIG_CONFIG = Path(
    "configs/so101/camera_rigs/official_32x32_uvc_photoreal_v1.json"
)


class SO101OfficialCameraRigTest(unittest.TestCase):
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

    def test_canonical_camera_rig_config_is_strict_and_complete(self) -> None:
        from pydantic import ValidationError

        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            SO101CameraRigRenderConfig,
            load_so101_camera_rig_render_config,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        self.assertEqual(config.render.source_width, 640)
        self.assertEqual(config.render.source_height, 360)
        self.assertEqual(config.render.policy_size, 256)
        self.assertEqual(config.render.policy_resize, "center_crop_square_then_resize")
        self.assertEqual(config.camera2.camera_position_gripper[1], -0.07416756456598633)
        self.assertEqual(config.camera1.camera_pinhole_protrusion_m, 0.020)

        payload = config.model_dump(mode="json")
        payload["unknown_render_setting"] = True
        with self.assertRaises(ValidationError):
            SO101CameraRigRenderConfig.model_validate(payload)

    def test_config_values_drive_generated_camera_xml(self) -> None:
        from physical_ai_agent.sim.so101_camera_rig_render_config import (
            load_so101_camera_rig_render_config,
        )
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        config = load_so101_camera_rig_render_config(CAMERA_RIG_CONFIG)
        sensor = config.sensor.model_copy(update={"vertical_fov_degrees": 69.0})
        camera1 = config.camera1.model_copy(update={"camera_pinhole_protrusion_m": 0.021})
        camera2 = config.camera2.model_copy(update={"rear_up_offset_m": 0.007})
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
            self.assertEqual(float(camera1_xml.get("fovy", "nan")), 69.0)
            self.assertEqual(float(camera2_xml.get("fovy", "nan")), 69.0)
            self.assertEqual(
                tuple(float(value) for value in camera1_xml.get("pos", "").split()),
                camera1.camera_pinhole_cad_m,
            )
            self.assertEqual(
                tuple(float(value) for value in camera2_xml.get("pos", "").split()),
                camera2.camera_position_gripper,
            )

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
                manifest["camera2_optical_axis"]["rear_up_offset_m"],
                0.005,
            )
            self.assertEqual(
                manifest["camera2_optical_axis"]["downward_angle_degrees"],
                66.0,
            )

    def test_camera_rig_uses_static_overhead_camera_and_narrower_wrist_fov(self) -> None:
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
            self.assertLess(float(model.cam_fovy[wrist_id]), 75.0)
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

    def test_camera1_does_not_render_its_own_overhead_mount(self) -> None:
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

            overhead_body_ids = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("overhead_camera_mount", "overhead_camera_tower")
            }
            overhead_geom_ids = {
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in overhead_body_ids
            }
            is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            visible_overhead_pixels = sum(
                int(np.count_nonzero(is_geom & (segmentation[..., 0] == geom_id)))
                for geom_id in overhead_geom_ids
            )
            self.assertEqual(
                visible_overhead_pixels,
                0,
                "camera1 must match the installed feed, where its own mast is absent",
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
