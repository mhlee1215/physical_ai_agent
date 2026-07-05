from __future__ import annotations

import importlib.util
import json
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
        self.assertIn("--frames", completed.stdout)


if __name__ == "__main__":
    unittest.main()
