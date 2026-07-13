from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RECIPE_PATH = Path("configs/so101/dataset_generation/grip_the_cube_v2.json")


class SO101DatasetGenerationRecipeTests(unittest.TestCase):
    def test_grip_v2_dry_run_covers_full_pipeline(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/generate_so101_dataset_recipe.py",
                "--recipe",
                str(RECIPE_PATH),
                "--dry-run",
                "--overwrite",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        stages = {stage["name"]: stage["command"] for stage in payload["stages"]}

        self.assertEqual(
            [name for name in stages if name.startswith("export:train:")],
            ["export:train:bin5", "export:train:bin6", "export:train:bin9", "export:train:bin10"],
        )
        self.assertEqual(
            [name for name in stages if name.startswith("export:validation:")],
            [
                "export:validation:bin5",
                "export:validation:bin6",
                "export:validation:bin9",
                "export:validation:bin10",
            ],
        )
        train5 = stages["export:train:bin5"]
        validation5 = stages["export:validation:bin5"]
        self.assertEqual(_value_after(train5, "--seed"), "5000000")
        self.assertEqual(_value_after(validation5, "--seed"), "20000000")
        self.assertEqual(_value_after(validation5, "--grid-lookup-start-index"), "253")
        self.assertEqual(_value_after(train5, "--width"), "256")
        self.assertEqual(_value_after(train5, "--height"), "256")
        self.assertEqual(_value_after(train5, "--edge-contact-parallel-success-threshold-deg"), "3.0")
        self.assertIn("--deterministic-camera-bin-lookup", train5)
        self.assertIn("--overwrite", train5)
        self.assertIn("merge:train", stages)
        self.assertIn("sidecar:validation", stages)
        self.assertIn("closed-loop-starts:validation", stages)
        self.assertIn("audit:train-vs-validation", stages)
        self.assertEqual(
            _value_after(stages["audit:train-vs-validation"], "--expected-train-bins"),
            "5:75,6:75,9:75,10:75",
        )
        self.assertEqual(
            _value_after(stages["audit:train-vs-validation"], "--expected-validation-bins"),
            "5:13,6:13,9:12,10:12",
        )

    def test_recipe_rejects_overlapping_effective_seed_ranges(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from generate_so101_dataset_recipe import _validate_unique_seed_ranges

        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        broken = copy.deepcopy(recipe)
        broken["splits"]["validation"]["bins"][0].update(
            {"seed": 5_000_000, "lookup_start_index": 0}
        )
        with self.assertRaisesRegex(ValueError, "seed ranges overlap"):
            _validate_unique_seed_ranges(broken)

    def test_recipe_contract_is_train_300_validation_50(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sum(row["episodes"] for row in recipe["splits"]["train"]["bins"]), 300)
        self.assertEqual(
            sum(row["episodes"] for row in recipe["splits"]["validation"]["bins"]), 50
        )
        self.assertEqual(recipe["splits"]["validation"]["closed_loop"]["episodes"], 10)
        self.assertEqual(recipe["common"]["terminal_hold_steps"], 12)
        self.assertEqual(recipe["audit"]["expected_prompt"], "grip the green cube and lift")

    def test_split_audit_accepts_disjoint_hwc_lerobot_splits(self) -> None:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from audit_so101_dataset_splits import audit_splits

        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "train"
            validation = Path(tmpdir) / "validation"
            _write_fake_split(train, seed=100, spawn_xy=(0.1, 0.2), action_value=1.0)
            _write_fake_split(validation, seed=200, spawn_xy=(0.3, 0.4), action_value=2.0)
            report = audit_splits(
                train_root=train,
                validation_root=validation,
                expected_prompt="grip the green cube and lift",
                expected_resolution=(256, 256),
                expected_train_bins={5: 1},
                expected_validation_bins={5: 1},
                expected_terminal_hold_steps=12,
                max_pre_close_alignment_deg=3.0,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["overlap_counts"], {"seeds": 0, "spawn_xy": 0, "trajectory_hashes": 0})


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_fake_split(root: Path, *, seed: int, spawn_xy: tuple[float, float], action_value: float) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "so101_lerobot_export_report.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "seed": seed,
                        "forced_spawn_xy": list(spawn_xy),
                        "grid_balance_bin": 5,
                        "success": True,
                        "task_success": True,
                        "phase_counts": {"terminal_hold": 12},
                        "pre_close_cube_face_normal_parallel_error_deg": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.camera1": {
                        "shape": [256, 256, 3],
                        "names": ["height", "width", "channels"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"task_index": [0]}, index=pd.Index(["grip the green cube and lift"], name="task")
    ).to_parquet(root / "meta" / "tasks.parquet")
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "action": [np.full(6, action_value, dtype=np.float32)] * 2,
            "observation.state": [np.zeros(6, dtype=np.float32)] * 2,
            "task_index": [0, 0],
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


if __name__ == "__main__":
    unittest.main()
