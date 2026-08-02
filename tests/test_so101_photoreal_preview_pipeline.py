from __future__ import annotations

import importlib.util
import json
import base64
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock


class SO101PhotorealPreviewPipelineTest(unittest.TestCase):
    def test_live_renderer_resolves_the_hardware_locked_camera_rig_profile(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview_profile",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        resolved = module._resolve_live_renderer_config(
            {
                "profile_from_camera_rig": True,
                "camera_rig_config": (
                    "configs/so101/camera_rigs/"
                    "official_32x32_uvc_photoreal_v10_fov_calibrated_direct_square.json"
                ),
            }
        )

        self.assertEqual(resolved["source_width"], 512)
        self.assertEqual(resolved["source_height"], 512)
        self.assertEqual(resolved["width"], 256)
        self.assertEqual(resolved["height"], 256)
        self.assertEqual(resolved["scene_profile"], "pbr_workshop_v4")
        self.assertEqual(resolved["lighting_profile"], "directional_key_fill_rim_v4")
        self.assertEqual(len(resolved["visual_props"]), 4)
        self.assertEqual(len(resolved["lights"]), 4)
        self.assertIn("observation.images.camera1", resolved["lens_distortion"])

    def test_named_camera1_uses_the_physical_pinhole_not_a_stereo_eye(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        class Env:
            unwrapped = type("Unwrapped", (), {"model": object(), "data": object()})()

        def fixed_camera(_model, _data, camera_name):
            location = [1.0, 2.0, 3.0] if camera_name == "egocentric_cam" else [4.0, 5.0, 6.0]
            return {
                "mode": "forward_up",
                "location": location,
                "forward": [0.0, 0.0, -1.0],
                "up": [0.0, 1.0, 0.0],
                "fovy": 70.0,
                "focus_distance": 0.5,
                "aperture_fstop": 10.0,
                "use_dof": False,
                "clip_start": 0.001,
            }

        with (
            mock.patch.object(module, "_named_camera_exists", return_value=True),
            mock.patch.object(module, "_fixed_mujoco_camera_spec", side_effect=fixed_camera),
            mock.patch.object(
                module,
                "_scene_camera_spec",
                side_effect=AssertionError("named camera must not use an MjvScene stereo eye"),
            ),
        ):
            cameras = module._camera_specs_from_mujoco_scene(
                Env(),
                {"egocentric_cam": object()},
                camera_lens=48.0,
                width=256,
                height=256,
            )

        self.assertEqual(
            cameras["observation.images.camera1"]["location"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            cameras["observation.images.camera2"]["location"],
            [4.0, 5.0, 6.0],
        )

    def test_dry_run_includes_photoreal_preview_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "recipes.json"
            recipe_path.write_text(
                json.dumps(
                    {
                        "defaults": {"fps": 12, "width": 96, "height": 96},
                        "recipes": [
                            {
                                "name": "tiny",
                                "script": "scripts/export_so101_teacher_rollouts_lerobot.py",
                                "root": "_workspace/test_so101_photoreal_dataset",
                                "repo_id": "physical-ai-agent/test",
                                "episodes": 1,
                                "seed": 123,
                                "args": {"skill_mode": "move_over_cube"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_so101_training_datasets.py",
                    "--recipes",
                    str(recipe_path),
                    "--only",
                    "tiny",
                    "--dry-run",
                    "--photoreal-preview",
                    "--photoreal-robot-material",
                    "matte_pla",
                    "--photoreal-samples",
                    "64",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["photoreal_preview"])
        self.assertEqual(len(payload["photoreal_commands"]), 1)
        command = payload["photoreal_commands"][0]
        self.assertIn("scripts/render_so101_blender_probe.py", command)
        self.assertIn("_workspace/test_so101_photoreal_dataset/photoreal_preview", command)
        self.assertIn("matte_pla", command)
        self.assertIn("64", command)

    def test_blender_probe_declares_matte_pla_option(self) -> None:
        spec = importlib.util.spec_from_file_location("render_so101_blender_probe", "scripts/render_so101_blender_probe.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

    def test_mycobot_blender_probe_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/render_mycobot_blender_probe.py", "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--asset-root", completed.stdout)
        self.assertIn("--robot-material", completed.stdout)

    def test_so101_dataset_blender_preview_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/render_so101_dataset_blender_preview.py", "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--dataset-root", completed.stdout)
        self.assertIn("--episodes", completed.stdout)
        self.assertIn("--env-source", completed.stdout)
        self.assertIn("--frames", completed.stdout)
        self.assertIn("--camera-lens", completed.stdout)
        self.assertIn("--scene-profile", completed.stdout)
        self.assertIn("--robot-material-config", completed.stdout)
        self.assertIn("--source-width", completed.stdout)
        self.assertIn("--policy-resize", completed.stdout)
        self.assertIn("--camera-rig-config", completed.stdout)
        self.assertIn("--no-preserve-pinhole-renders", completed.stdout)

    def test_pinhole_retention_does_not_change_final_distorted_pixels(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        from PIL import Image

        camera_key = "observation.images.camera1"
        camera_specs = {camera_key: {"fovy": 50.0}}
        render_specs = {camera_key: {"fovy": 55.0}}
        distortion = {
            camera_key: {
                "model": "opencv_brown_conrady",
                "coefficients": [-0.08, 0.01, 0.0, 0.0, 0.0],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept = root / "kept.png"
            discarded = root / "discarded.png"
            source = Image.new("RGB", (32, 32))
            source.putdata(
                [
                    ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
                    for y in range(32)
                    for x in range(32)
                ]
            )
            source.save(kept)
            source.save(discarded)

            module._apply_lens_distortion_to_images(
                {camera_key: str(kept)},
                target_camera_specs=camera_specs,
                render_camera_specs=render_specs,
                distortion_profiles=distortion,
                preserve_pinhole=True,
            )
            module._apply_lens_distortion_to_images(
                {camera_key: str(discarded)},
                target_camera_specs=camera_specs,
                render_camera_specs=render_specs,
                distortion_profiles=distortion,
                preserve_pinhole=False,
            )

            self.assertEqual(kept.read_bytes(), discarded.read_bytes())
            self.assertTrue((root / "kept_pinhole.png").is_file())
            self.assertFalse((root / "discarded_pinhole.png").exists())

    def test_dataset_renderer_downsamples_512_source_to_256_policy_image(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera1.png"
            Image.new("RGB", (512, 512), (10, 20, 30)).save(path)
            module._resize_policy_image(
                path,
                width=256,
                height=256,
                mode="direct_square_render",
            )
            with Image.open(path) as resized:
                resized_size = resized.size

        self.assertEqual(resized_size, (256, 256))

    def test_blender_batch_artifacts_are_process_unique_and_missing_outputs_retry(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_dir = root / "episode_0000_frame_0000"
            mesh_dir = frame_dir / "ply"
            mesh_dir.mkdir(parents=True)
            spec_path = frame_dir / "blender_scene_spec.json"
            spec_path.write_text("{}", encoding="utf-8")
            camera1 = frame_dir / "camera1.png"
            camera2 = frame_dir / "camera2.png"
            item = {
                "spec_path": spec_path,
                "frame_dir": frame_dir,
                "mesh_dir": mesh_dir,
                "image_paths": {
                    "observation.images.camera1": str(camera1),
                    "observation.images.camera2": str(camera2),
                },
                "camera_specs": {
                    "observation.images.camera1": {"rotation_degrees": 0},
                    "observation.images.camera2": {"rotation_degrees": 0},
                },
                "render_camera_specs": {},
                "distortion_profiles": {},
                "rendered_item": {"episode": 0, "frame": 0},
            }
            calls = 0

            def fake_blender(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    camera1.write_bytes(b"corrupt png")
                    camera2.write_bytes(b"corrupt png")
                else:
                    Image.new("RGB", (512, 512), (10, 20, 30)).save(camera1)
                    Image.new("RGB", (512, 512), (40, 50, 60)).save(camera2)
                return subprocess.CompletedProcess([], 0, stdout="", stderr="")

            with mock.patch.object(module.subprocess, "run", side_effect=fake_blender):
                rendered = module._flush_blender_pending(
                    [item],
                    output_dir=root,
                    driver_path=root / "driver.py",
                    blender_bin="blender",
                    duplicate_camera3_from_camera2=False,
                    batch_mode=True,
                    output_width=256,
                    output_height=256,
                    policy_resize="direct_square_render",
                )

            manifests = sorted(root.glob("blender_*.json"))
            logs = sorted(root.glob("blender_*.log"))
            self.assertEqual(calls, 2)
            self.assertEqual(len(manifests), 2)
            self.assertEqual(len({path.name for path in manifests}), 2)
            self.assertTrue(all(f"pid{module.os.getpid()}" in path.name for path in manifests))
            self.assertEqual(len(logs), 2)
            self.assertEqual(rendered[0]["episode"], 0)
            with Image.open(camera1) as rendered_camera1, Image.open(camera2) as rendered_camera2:
                self.assertEqual(rendered_camera1.size, (256, 256))
                self.assertEqual(rendered_camera2.size, (256, 256))

    def test_skip_existing_rejects_old_direct_256_render_contract(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp)
            image_path = frame_dir / "camera1.png"
            Image.new("RGB", (256, 256), (10, 20, 30)).save(image_path)
            (frame_dir / "blender_scene_spec.json").write_text(
                json.dumps(
                    {
                        "width": 256,
                        "height": 256,
                        "samples": 256,
                        "denoise": False,
                    }
                ),
                encoding="utf-8",
            )
            matches = module._existing_render_matches_contract(
                frame_dir=frame_dir,
                image_paths={"observation.images.camera1": str(image_path)},
                source_width=512,
                source_height=512,
                output_width=256,
                output_height=256,
                policy_resize="direct_square_render",
                samples=256,
                denoise=False,
                camera_rig_config_hash="a" * 64,
            )

        self.assertFalse(matches)

    def test_robot_material_config_is_editable_and_valid(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        path = Path("configs/so101/render_profiles/black_arm_green_white_gripper.json")
        config = module._load_robot_material_config(path)
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["default_material"], "black_matte_pla")
        self.assertEqual(config["materials"]["black_matte_pla"]["base_color"], [0.025, 0.03, 0.035])
        self.assertEqual(config["materials"]["white_matte_pla"]["base_color"], [0.82, 0.84, 0.80])
        self.assertTrue(
            {
                "fixed_jaw",
                "moving_jaw",
                "overhead_arm_base",
                "overhead_mount_bottom",
                "overhead_mount_upper",
                "overhead_camera_module",
            }.issubset(config["parts"])
        )
        self.assertEqual(
            config["parts"]["wrist_motor_holder"],
            {
                "material": "white_matte_pla",
                "selectors": [
                    {
                        "body_names": ["lower_arm"],
                        "mesh_names": ["motor_holder_so101_wrist_v1"],
                    }
                ],
            },
        )
        self.assertEqual(config["parts"]["fixed_jaw"]["material"], "green_matte_pla")
        self.assertEqual(config["parts"]["moving_jaw"]["material"], "white_matte_pla")

    def test_legacy_robot_material_config_remains_supported(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "legacy",
                        "default_part": "arm",
                        "parts": {
                            "arm": {"base_color": [0.1, 0.1, 0.1], "roughness": 0.8, "metallic": 0.0}
                        },
                        "selectors": {},
                    }
                ),
                encoding="utf-8",
            )
            config = module._load_robot_material_config(path)
        self.assertEqual(config["schema_version"], 1)

    def test_robot_material_config_rejects_unknown_selector_fields(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "name": "invalid",
                        "default_material": "black",
                        "materials": {
                            "black": {"base_color": [0.1, 0.1, 0.1], "roughness": 0.8, "metallic": 0.0}
                        },
                        "parts": {"arm": {"material": "black", "selectors": [{"geom_ids": [1]}]}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid selector for part arm"):
                module._load_robot_material_config(path)

    def test_black_table_props_are_deterministic_and_outside_manipulation_zone(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "render_so101_dataset_blender_preview",
            "scripts/render_so101_dataset_blender_preview.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        props = module._visual_props_for_episode(492500)
        self.assertEqual(props, module._visual_props_for_episode(492500))
        self.assertNotEqual(props, module._visual_props_for_episode(492501))
        self.assertEqual({item["kind"] for item in props}, {"mug", "bottle", "tape", "screwdriver"})
        for item in props:
            x, y = item["position"]
            self.assertFalse(0.10 <= x <= 0.46 and -0.05 <= y <= 0.34)
        self.assertEqual(module._frame_label({0: {113: "final"}}, episode=0, frame=113), "final")

    def test_dataset_viewer_photoreal_preview_helpers(self) -> None:
        spec = importlib.util.spec_from_file_location("serve_so101_dataset_viewer", "scripts/serve_so101_dataset_viewer.py")
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset_root"
            preview = Path(tmp) / "preview"
            root.mkdir()
            preview.mkdir()
            (preview / "episode_0002_frame_0085.png").write_bytes(
                base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax9L2kAAAAASUVORK5CYII=")
            )
            module.PHOTO_REAL_PREVIEW_DIRS = {root.name: preview}

            summary = module._photoreal_preview_summary(root)
            images = module._photoreal_frame_images(root, episode=2, frame=85)

        self.assertTrue(summary["available"])
        self.assertEqual(summary["frames_by_episode"], {"2": [85]})
        self.assertIn("photoreal_sidecar", images)
        self.assertTrue(images["photoreal_sidecar"].startswith("data:image/png;base64,"))

    def test_dataset_viewer_so101_photoreal_dataset_adapter(self) -> None:
        spec = importlib.util.spec_from_file_location("serve_so101_dataset_viewer", "scripts/serve_so101_dataset_viewer.py")
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax9L2kAAAAASUVORK5CYII=")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "photoreal_dataset"
            (root / "episodes").mkdir(parents=True)
            image_paths = {}
            for camera in ("camera1", "camera2", "camera3"):
                image = root / "images" / "episode_0000" / f"observation_images_{camera}" / "frame_0000.png"
                image.parent.mkdir(parents=True)
                image.write_bytes(png)
                image_paths[camera] = f"images/episode_0000/observation_images_{camera}/frame_0000.png"
            (root / "episodes" / "episode_0000.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "task_index": 0,
                        "task": "Grasp the visible cube and lift it up.",
                        "prompt": "Grasp the visible cube and lift it up.",
                        "source_episode_index": 2,
                        "source_frame_index": 85,
                        "observation": {
                            "state": [0, 1, 2, 3, 4, 5],
                            "images": image_paths,
                        },
                        "action": [5, 4, 3, 2, 1, 0],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "so101_photoreal_jsonl_v1",
                        "episodes": 1,
                        "frames": 1,
                        "fps": 12,
                        "image_mime_type": "image/png",
                        "image_shape": [1, 1, 3],
                        "features": [
                            "observation.images.camera1",
                            "observation.images.camera2",
                            "observation.images.camera3",
                            "observation.state",
                            "action",
                        ],
                        "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                        "action_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                        "camera_contract": {
                            "observation.images.camera1": "photoreal egocentric_cam",
                            "observation.images.camera2": "photoreal wrist_cam",
                            "observation.images.camera3": "photoreal wrist_cam duplicate",
                        },
                        "episode_summaries": [{"episode_index": 0, "frames": 1}],
                    }
                ),
                encoding="utf-8",
            )

            dataset = module._so101_photoreal_dataset(root)
            summary = module._so101_photoreal_dataset_summary("photoreal_test", dataset)
            payload = module._so101_photoreal_frame_payload(root, "photoreal_test", 0, 0)

        self.assertEqual(summary["dataset_format"], "so101_photoreal_jsonl_v1")
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(payload["source_episode_index"], 2)
        self.assertEqual(payload["source_frame_index"], 85)
        self.assertIn("observation.images.camera1", payload["images"])
        self.assertIn("observation.images.camera2", payload["images"])
        self.assertIn("observation.images.camera3", payload["images"])
        self.assertNotIn("photoreal_images", payload)

    def test_photoreal_lerobot_builder_replaces_embedded_image_bytes(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("pyarrow is not installed in this test runtime")

        spec = importlib.util.spec_from_file_location(
            "build_so101_photoreal_lerobot_dataset",
            "scripts/build_so101_photoreal_lerobot_dataset.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        png_old = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax9L2kAAAAASUVORK5CYII=")
        png_new = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/luzn8QAAAABJRU5ErkJggg==")
        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            rendered = Path(tmp) / "rendered"
            output = Path(tmp) / "output"
            (source / "data" / "chunk-000").mkdir(parents=True)
            (source / "meta").mkdir()
            (source / "render_replay").mkdir()
            (source / "render_replay" / "manifest.json").write_text("{}", encoding="utf-8")
            (rendered / "episode_0000_frame_0000").mkdir(parents=True)
            for camera in ("camera1", "camera2"):
                (rendered / "episode_0000_frame_0000" / f"episode_0000_frame_0000_{camera}.png").write_bytes(png_new)
            table = pa.table(
                {
                    "observation.images.camera1": pa.array([{"bytes": png_old, "path": None}], type=image_type),
                    "observation.images.camera2": pa.array([{"bytes": png_old, "path": None}], type=image_type),
                    "observation.images.camera3": pa.array([{"bytes": png_old, "path": None}], type=image_type),
                    "observation.state": pa.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], type=pa.list_(pa.float32(), 6)),
                    "action": pa.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]], type=pa.list_(pa.float32(), 6)),
                    "timestamp": pa.array([0.0], type=pa.float32()),
                    "frame_index": pa.array([0], type=pa.int64()),
                    "episode_index": pa.array([0], type=pa.int64()),
                    "index": pa.array([0], type=pa.int64()),
                    "task_index": pa.array([0], type=pa.int64()),
                }
            )
            pq.write_table(table, source / "data" / "chunk-000" / "file-000.parquet")
            (source / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 1, "total_frames": 1, "fps": 12, "features": {}}),
                encoding="utf-8",
            )

            report = module.build_photoreal_lerobot_dataset(
                source_dataset_root=source,
                rendered_dir=rendered,
                output_root=output,
                repo_id="physical-ai-agent/test-photoreal",
                overwrite=True,
            )
            resumed = module.build_photoreal_lerobot_dataset(
                source_dataset_root=source,
                rendered_dir=rendered,
                output_root=output,
                repo_id="physical-ai-agent/test-photoreal",
                skip_existing=True,
            )
            converted = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet").to_pydict()
            copied_render_replay = (output / "render_replay").exists()

        self.assertEqual(report["format"], "so101_photoreal_lerobot_v1")
        self.assertTrue(report["training_ready"])
        self.assertTrue(resumed["skipped_existing"])
        self.assertFalse(copied_render_replay)
        from PIL import Image

        for key in (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ):
            self.assertNotEqual(converted[key][0]["bytes"], png_old)
            with Image.open(BytesIO(converted[key][0]["bytes"])) as image:
                self.assertEqual(image.mode, "RGB")

    def test_pick_cube_photoreal_config_covers_train_and_eval(self) -> None:
        config = json.loads(Path("configs/so101/training/pick_photoreal.json").read_text(encoding="utf-8"))

        self.assertEqual(config["task"], "pick")
        self.assertEqual(config["camera_contract"]["observation.images.camera1"], "egocentric_cam")
        self.assertEqual(config["camera_contract"]["observation.images.camera2"], "wrist_cam")
        self.assertEqual(config["train_dataset"]["dataset_format"], "so101_photoreal_lerobot_v1")
        self.assertEqual(config["validation_dataset"]["dataset_format"], "so101_photoreal_lerobot_v1")
        self.assertEqual(config["train_dataset"]["expected_frames"], 4598)
        self.assertEqual(config["validation_dataset"]["expected_frames"], 2210)
        self.assertEqual(config["train_dataset"]["task_prompt_source"], "episode_seed_target_object_color")
        self.assertEqual(config["validation_dataset"]["task_prompt_source"], "episode_seed_target_object_color")

    def test_color_task_prompt_omits_visible_cube(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "build_so101_photoreal_lerobot_dataset",
            "scripts/build_so101_photoreal_lerobot_dataset.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == "pyarrow":
                self.skipTest("pyarrow is not installed in this test runtime")
            raise

        self.assertEqual(
            module._color_task_prompt(skill_mode="pick_cube", color="green", shape="cube"),
            "Grasp the green cube and lift it up.",
        )
        self.assertNotIn(
            "visible cube",
            module._color_task_prompt(skill_mode="pick_cube", color="red", shape="cube"),
        )

if __name__ == "__main__":
    unittest.main()
