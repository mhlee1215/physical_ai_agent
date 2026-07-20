import json
import unittest
from pathlib import Path


def _source_mesh_name(geom) -> str:
    mesh_name = geom.get("mesh", "")
    return f"{mesh_name.removeprefix('overhead_').removesuffix('_32x32_uvc')}.stl"


class SO101OverheadCameraMountTest(unittest.TestCase):
    def test_official_stl_parts_follow_bottom_middle_top_connector_stack(self) -> None:
        try:
            import mujoco
            import numpy as np
            import trimesh
            from train_so101_wrist_ego_visual_servo import make_high_contrast_picklift_env

            from physical_ai_agent.sim.so101_overhead_camera_mount import (
                OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
                OVERHEAD_ARM_BASE_FRONT_EDGE_CAD_X_M,
                OVERHEAD_ARM_BASE_JOINT_MAX_CAD_X_M,
                OVERHEAD_ARM_BASE_JOINT_MIN_CAD_X_M,
                OVERHEAD_ARM_BASE_SOCKET_INNER_CAD_Z_M,
                OVERHEAD_ARM_BASE_SOCKET_OUTER_CAD_Z_M,
                OVERHEAD_ARM_BASE_THICKNESS_M,
                OVERHEAD_ARM_BASE_TO_LOWER_MAST_INSERTION_DEPTH_M,
                OVERHEAD_CAMERA_FORWARD_WORLD,
                OVERHEAD_CAMERA_UP_WORLD,
                OVERHEAD_CONNECTOR_INSERTION_DEPTH_M,
                OVERHEAD_LOWER_MAST_JOINT_MAX_CAD_X_M,
                OVERHEAD_LOWER_MAST_JOINT_MIN_CAD_X_M,
                OVERHEAD_LOWER_MAST_TAB_INNER_CAD_Z_M,
                OVERHEAD_LOWER_MAST_TAB_OUTER_CAD_Z_M,
                OVERHEAD_MESH_QUATERNION_CAD_WXYZ,
                OVERHEAD_MESH_TRANSLATION_CAD_M,
                OVERHEAD_RIG_WORLD_POSITION,
                OVERHEAD_TOP_MOUNT_QUATERNION_CAD_WXYZ,
                OVERHEAD_TOWER_POSITION_CAD_M,
                OVERHEAD_TOWER_QUATERNION_CAD_WXYZ,
                SO101_BASE_SHELL_FRONT_EDGE_FROM_ROOT_X_M,
                SO101_BASE_WORLD_POSITION,
                prepare_official_32x32_uvc_camera_rig_xml,
            )
            from physical_ai_agent.sim.so101_wrist_camera_mount import (
                INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES,
                INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

        asset = prepare_official_32x32_uvc_camera_rig_xml()
        env = make_high_contrast_picklift_env(
            target_object_color="green",
            object_half_sizes=(0.015,),
            camera_rig_preset=OFFICIAL_32X32_UVC_CAMERA_RIG_PRESET,
        )
        try:
            env.reset(seed=50_000_000)
            model = env.unwrapped.model
            data = env.unwrapped.data
            mujoco.mj_forward(model, data)
            rig_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "overhead_camera_mount"
            )
            tower_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "overhead_camera_tower"
            )
            mesh_names = {
                "overhead_arm_base_32x32_uvc",
                "overhead_cam_mount_bottom_32x32_uvc",
                "overhead_cam_mount_middle_32x32_uvc",
                "overhead_cam_mount_top_32x32_uvc",
            }
            found = set()
            for geom_id in range(model.ngeom):
                mesh_id = int(model.geom_dataid[geom_id])
                if mesh_id < 0:
                    continue
                mesh_name = model.mesh(mesh_id).name
                if mesh_name in mesh_names:
                    found.add(mesh_name)
                    expected_body_id = (
                        rig_id
                        if mesh_name == "overhead_arm_base_32x32_uvc"
                        else tower_id
                    )
                    self.assertEqual(int(model.geom_bodyid[geom_id]), expected_body_id)
            self.assertEqual(found, mesh_names)
            import xml.etree.ElementTree as ET

            root = ET.parse(asset.robot_xml).getroot()
            rig = root.find('./worldbody/body[@name="overhead_camera_mount"]')
            self.assertIsNotNone(rig)
            tower = rig.find('./body[@name="overhead_camera_tower"]')
            self.assertIsNotNone(tower)
            self.assertEqual(
                tuple(float(value) for value in tower.get("pos").split()),
                OVERHEAD_TOWER_POSITION_CAD_M,
            )
            self.assertEqual(
                tuple(float(value) for value in tower.get("quat").split()),
                OVERHEAD_TOWER_QUATERNION_CAD_WXYZ,
            )
            mesh_geoms = [geom for geom in rig.findall("./geom") if geom.get("mesh") in mesh_names]
            mesh_geoms.extend(
                geom for geom in tower.findall("./geom") if geom.get("mesh") in mesh_names
            )
            self.assertEqual(len(mesh_geoms), 4)
            mesh_quaternions = {
                _source_mesh_name(geom): tuple(
                    float(value) for value in geom.get("quat", "1 0 0 0").split()
                )
                for geom in mesh_geoms
            }
            self.assertEqual(mesh_quaternions, OVERHEAD_MESH_QUATERNION_CAD_WXYZ)
            self.assertEqual(
                mesh_quaternions["cam_mount_top.stl"],
                OVERHEAD_TOP_MOUNT_QUATERNION_CAD_WXYZ,
            )
            actual_positions = {
                _source_mesh_name(geom): tuple(
                    float(value) for value in geom.get("pos", "0 0 0").split()
                )
                for geom in mesh_geoms
            }
            expected_positions = {
                "arm_base.stl": OVERHEAD_MESH_TRANSLATION_CAD_M["arm_base.stl"],
                "cam_mount_bottom.stl": OVERHEAD_MESH_TRANSLATION_CAD_M[
                    "cam_mount_bottom.stl"
                ],
                "cam_mount_middle.stl": OVERHEAD_MESH_TRANSLATION_CAD_M[
                    "cam_mount_middle.stl"
                ],
                "cam_mount_top.stl": OVERHEAD_MESH_TRANSLATION_CAD_M[
                    "cam_mount_top.stl"
                ],
            }
            self.assertEqual(actual_positions, expected_positions)
            self.assertEqual(
                actual_positions["cam_mount_bottom.stl"],
                (0.0, 0.0, 0.0),
            )
            # The source parts retain one CAD orientation. The lower floor's
            # keyed tab is aligned to and fully inserted through the arm-base
            # socket, which becomes the base's right side in the rendered
            # world frame.
            self.assertEqual(
                OVERHEAD_TOWER_QUATERNION_CAD_WXYZ,
                (1.0, 0.0, 0.0, 0.0),
            )
            self.assertAlmostEqual(
                OVERHEAD_ARM_BASE_JOINT_MIN_CAD_X_M,
                OVERHEAD_TOWER_POSITION_CAD_M[0]
                + OVERHEAD_LOWER_MAST_JOINT_MIN_CAD_X_M,
            )
            self.assertAlmostEqual(
                OVERHEAD_ARM_BASE_JOINT_MAX_CAD_X_M,
                OVERHEAD_TOWER_POSITION_CAD_M[0]
                + OVERHEAD_LOWER_MAST_JOINT_MAX_CAD_X_M,
            )
            self.assertAlmostEqual(
                OVERHEAD_ARM_BASE_SOCKET_OUTER_CAD_Z_M,
                OVERHEAD_TOWER_POSITION_CAD_M[2]
                + OVERHEAD_LOWER_MAST_TAB_OUTER_CAD_Z_M,
            )
            self.assertAlmostEqual(
                OVERHEAD_ARM_BASE_SOCKET_INNER_CAD_Z_M,
                OVERHEAD_TOWER_POSITION_CAD_M[2]
                + OVERHEAD_LOWER_MAST_TAB_INNER_CAD_Z_M,
            )
            self.assertAlmostEqual(
                OVERHEAD_ARM_BASE_SOCKET_INNER_CAD_Z_M
                - OVERHEAD_ARM_BASE_SOCKET_OUTER_CAD_Z_M,
                OVERHEAD_ARM_BASE_TO_LOWER_MAST_INSERTION_DEPTH_M,
            )
            self.assertGreater(
                OVERHEAD_ARM_BASE_TO_LOWER_MAST_INSERTION_DEPTH_M,
                0.0,
                "the mast floor must enter the arm-base socket, not edge-touch it",
            )
            arm_joint_world_x = (
                OVERHEAD_RIG_WORLD_POSITION[0]
                + OVERHEAD_ARM_BASE_JOINT_MIN_CAD_X_M
            )
            mast_joint_world_x = (
                OVERHEAD_RIG_WORLD_POSITION[0]
                + OVERHEAD_TOWER_POSITION_CAD_M[0]
                + OVERHEAD_LOWER_MAST_JOINT_MIN_CAD_X_M
            )
            self.assertAlmostEqual(arm_joint_world_x, mast_joint_world_x)
            joint_world_y = (
                OVERHEAD_RIG_WORLD_POSITION[1]
                - OVERHEAD_ARM_BASE_SOCKET_OUTER_CAD_Z_M
            )
            mast_outer_world_y = (
                OVERHEAD_RIG_WORLD_POSITION[1]
                - (OVERHEAD_TOWER_POSITION_CAD_M[2] - 0.010)
            )
            self.assertGreater(
                mast_outer_world_y,
                joint_world_y,
                "the mast must extend to the base's rendered right side, not behind it",
            )
            arm_base_min_y = -0.005
            arm_base_max_y = 0.0022
            bottom_min_y = 0.0
            bottom_plate_max_y = 0.0072
            self.assertAlmostEqual(
                arm_base_min_y,
                OVERHEAD_TOWER_POSITION_CAD_M[1] + bottom_min_y,
            )
            self.assertAlmostEqual(
                arm_base_max_y,
                OVERHEAD_TOWER_POSITION_CAD_M[1] + bottom_plate_max_y,
            )
            arm_base_mesh = trimesh.load_mesh(
                Path(asset.source_dir) / "arm_base.stl", process=False
            )
            bottom_mesh = trimesh.load_mesh(
                Path(asset.source_dir) / "cam_mount_bottom.stl", process=False
            )
            arm_base_mesh.apply_scale(0.001)
            bottom_mesh.apply_scale(0.001)
            tower_transform = trimesh.transformations.quaternion_matrix(
                OVERHEAD_TOWER_QUATERNION_CAD_WXYZ
            )
            tower_transform[:3, 3] = OVERHEAD_TOWER_POSITION_CAD_M
            bottom_mesh.apply_transform(tower_transform)
            _, arm_base_contact_distance, _ = trimesh.proximity.closest_point_naive(
                bottom_mesh, arm_base_mesh.vertices
            )
            _, bottom_contact_distance, _ = trimesh.proximity.closest_point_naive(
                arm_base_mesh, bottom_mesh.vertices
            )
            self.assertLess(
                min(
                    float(arm_base_contact_distance.min()),
                    float(bottom_contact_distance.min()),
                ),
                1e-6,
                "the lower mast must physically touch both arm-base fingers",
            )
            self.assertGreaterEqual(
                int((arm_base_contact_distance < 1e-5).sum())
                + int((bottom_contact_distance < 1e-5).sum()),
                300,
                "the keyed tab must mate across the full arm-base socket",
            )
            local_mast_meshes = {}
            for filename in (
                "cam_mount_bottom.stl",
                "cam_mount_middle.stl",
                "cam_mount_top.stl",
            ):
                mesh = trimesh.load_mesh(
                    Path(asset.source_dir) / filename, process=False
                )
                mesh.apply_scale(0.001)
                mesh.apply_translation(actual_positions[filename])
                local_mast_meshes[filename] = mesh
            for lower_name, upper_name in (
                ("cam_mount_bottom.stl", "cam_mount_middle.stl"),
                ("cam_mount_middle.stl", "cam_mount_top.stl"),
            ):
                _, connector_distance, _ = trimesh.proximity.closest_point_naive(
                    local_mast_meshes[lower_name],
                    local_mast_meshes[upper_name].vertices,
                )
                self.assertLess(
                    float(connector_distance.min()),
                    1e-5,
                    f"{lower_name} and {upper_name} must meet at their connectors",
                )

            # Each upper section is inserted into the section below rather than
            # merely touching it at a bounding-box plane.
            bottom_max_y = 0.2309499969482422
            middle_min_y = 0.035
            middle_max_y = 0.19785000610351562
            top_min_y = 0.190
            self.assertAlmostEqual(
                bottom_max_y
                - (middle_min_y + actual_positions["cam_mount_middle.stl"][1]),
                OVERHEAD_CONNECTOR_INSERTION_DEPTH_M,
                places=6,
            )
            self.assertAlmostEqual(
                (middle_max_y + actual_positions["cam_mount_middle.stl"][1])
                - (top_min_y + actual_positions["cam_mount_top.stl"][1]),
                OVERHEAD_CONNECTOR_INSERTION_DEPTH_M,
                places=6,
            )
            self.assertEqual(
                actual_positions["cam_mount_middle.stl"][1],
                actual_positions["cam_mount_top.stl"][1],
            )

            base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
            np.testing.assert_allclose(
                model.body_pos[base_id],
                SO101_BASE_WORLD_POSITION,
                atol=1e-10,
            )
            self.assertAlmostEqual(
                SO101_BASE_WORLD_POSITION[0]
                + SO101_BASE_SHELL_FRONT_EDGE_FROM_ROOT_X_M,
                OVERHEAD_RIG_WORLD_POSITION[0]
                + OVERHEAD_ARM_BASE_FRONT_EDGE_CAD_X_M,
            )
            self.assertAlmostEqual(
                SO101_BASE_WORLD_POSITION[2],
                OVERHEAD_ARM_BASE_THICKNESS_M,
            )

            ego_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "egocentric_cam"
            )
            wrist_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
            )
            self.assertAlmostEqual(
                float(model.cam_fovy[ego_id]),
                INNOMAKER_U20CAM_VERTICAL_FOV_DEGREES,
            )
            self.assertAlmostEqual(
                float(model.cam_fovy[wrist_id]),
                INTEGRATED_32X32_UVC_CAMERA_FOVY_DEGREES,
            )
            self.assertEqual(
                float(model.cam_fovy[wrist_id]),
                float(model.cam_fovy[ego_id]),
            )
            camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "egocentric_cam"
            )
            np.testing.assert_allclose(
                data.cam_xpos[camera_id], asset.camera1_position_world, atol=1e-8
            )
            rotation = data.cam_xmat[camera_id].reshape(3, 3)
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
            gripper_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
            )
            gripper_direction = data.xpos[gripper_id] - data.cam_xpos[camera_id]
            gripper_direction /= np.linalg.norm(gripper_direction)
            camera_forward = rotation @ np.array([0.0, 0.0, -1.0])
            self.assertGreater(
                float(camera_forward @ gripper_direction),
                0.8,
                "the overhead camera head must face the follower gripper side",
            )
        finally:
            env.close()

    def test_manifest_records_sensor_and_policy_resize_contract(self) -> None:
        from physical_ai_agent.sim.so101_overhead_camera_mount import (
            OVERHEAD_CAMERA_PINHOLE_PROTRUSION_M,
            prepare_official_32x32_uvc_camera_rig_xml,
        )

        asset = prepare_official_32x32_uvc_camera_rig_xml()
        manifest = json.loads(Path(asset.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(manifest["camera_model"], "InnoMaker U20CAM-1080P (Amazon B0CNCSFQC1)")
        self.assertEqual(manifest["source_resolution"], [1920, 1080])
        self.assertEqual(manifest["policy_resolution"], [256, 256])
        self.assertEqual(manifest["policy_resize"], "center_crop_square_then_resize")
        self.assertEqual(manifest["distortion"]["model"], "opencv_brown_conrady")
        self.assertEqual(manifest["distortion"]["calibration_status"], "uncalibrated_candidate")
        self.assertEqual(
            manifest["camera1_optical_axis"]["downward_angle_degrees"],
            50.0,
        )
        self.assertEqual(
            manifest["camera1_optical_axis"]["calibration_source"],
            "installed_camera_frame",
        )
        self.assertFalse(
            manifest["camera1_optical_axis"]["self_mount_visible_at_home_pose"]
        )
        self.assertFalse(manifest["assembly"]["stl_parts_share_one_cad_frame"])
        self.assertEqual(
            manifest["assembly"]["assembly_mode"],
            "arm_base_slot_plus_connector_stack",
        )
        self.assertEqual(
            manifest["assembly"]["arm_base_to_lower_mast_insertion_depth_m"],
            0.010,
        )
        self.assertEqual(
            manifest["assembly"]["robot_base_world_position"],
            [-0.019835293745632178, 0.0, 0.0072],
        )
        self.assertEqual(
            manifest["assembly"]["part_translation_cad_m"]["cam_mount_middle.stl"],
            [0.0187, 0.1881, 0.0365125],
        )
        self.assertEqual(
            manifest["assembly"]["part_translation_cad_m"]["cam_mount_top.stl"],
            [0.0187, 0.1881, 0.0365125],
        )
        self.assertEqual(manifest["assembly"]["connector_insertion_depth_m"], 0.00785)
        self.assertEqual(
            manifest["assembly"]["camera_pinhole_protrusion_m"],
            OVERHEAD_CAMERA_PINHOLE_PROTRUSION_M,
        )
        self.assertEqual(OVERHEAD_CAMERA_PINHOLE_PROTRUSION_M, 0.020)
        self.assertEqual(
            manifest["assembly"]["part_quaternion_cad_wxyz"]["cam_mount_top.stl"],
            [1.0, 0.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
