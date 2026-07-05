from __future__ import annotations

import importlib.util
import json
import base64
import subprocess
import sys
import tempfile
import unittest
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
            (root / "images" / "episode_0000").mkdir(parents=True)
            image = root / "images" / "episode_0000" / "frame_0000.png"
            image.write_bytes(png)
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
                            "images": {"camera1": "images/episode_0000/frame_0000.png"},
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
                        "features": ["observation.images.camera1", "observation.state", "action"],
                        "joint_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                        "action_names": ["j0", "j1", "j2", "j3", "j4", "j5"],
                        "camera_contract": {"observation.images.camera1": "photoreal_render"},
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
        self.assertNotIn("photoreal_images", payload)


if __name__ == "__main__":
    unittest.main()
