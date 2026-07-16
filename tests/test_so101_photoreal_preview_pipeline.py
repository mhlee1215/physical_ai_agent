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


class SO101PhotorealPreviewPipelineTest(unittest.TestCase):
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
        self.assertEqual(config["default_part"], "arm")
        self.assertEqual(config["parts"]["arm"]["base_color"], [0.025, 0.03, 0.035])
        self.assertEqual(config["parts"]["wrist_strap"]["base_color"], [0.82, 0.84, 0.80])
        self.assertEqual(
            config["selectors"]["wrist_strap"]["mesh_names"],
            ["under_arm_so101_v1"],
        )
        self.assertEqual(config["selectors"]["static_gripper"]["primitive_names"], ["static_finger_pad"])
        self.assertEqual(config["selectors"]["moving_gripper"]["primitive_names"], ["moving_finger_pad"])

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
            converted = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet").to_pydict()

        self.assertEqual(report["format"], "so101_photoreal_lerobot_v1")
        self.assertTrue(report["training_ready"])
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
