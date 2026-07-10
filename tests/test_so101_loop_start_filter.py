from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.filter_so101_loop_start_report import filter_loop_start_report


class SO101LoopStartFilterTest(unittest.TestCase):
    def test_filter_keeps_only_train_bins_with_visible_policy_targets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source_report.json"
            source.write_text(
                json.dumps(
                    {
                        "episodes": [
                            _episode(seed=1, camera1_area=80, camera2_area=100, centroid=[80, 90]),  # bin 5
                            _episode(seed=2, camera1_area=4, camera2_area=100, centroid=[80, 90]),  # too small
                            _episode(seed=3, camera1_area=80, camera2_area=100, centroid=[230, 20]),  # bin 3
                            _episode(seed=4, camera1_area=90, camera2_area=20, centroid=[200, 100]),  # camera2 small
                            _episode(seed=5, camera1_area=90, camera2_area=100, centroid=[200, 100]),  # bin 7
                            _episode_with_stale_best_meta(seed=6, centroid=[80, 90]),  # start camera1 invisible
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "filtered_report.json"

            filtered = filter_loop_start_report(
                source_report=source,
                output_report=output,
                allowed_grid_bins=[5, 7],
                camera1_min_area=60,
                camera2_min_area=80,
            )

            self.assertEqual([episode["seed"] for episode in filtered["episodes"]], [1, 5])
            self.assertEqual([episode["loop_source_episode_index"] for episode in filtered["episodes"]], [0, 4])
            self.assertEqual([episode["loop_camera1_grid_bin_0_based"] for episode in filtered["episodes"]], [5, 7])
            self.assertEqual(filtered["loop_filter"]["allowed_grid_bins_0_based"], [5, 7])
            self.assertEqual(filtered["loop_filter"]["selected_episodes"], 2)
            self.assertTrue(output.exists())

    def test_filter_can_use_source_dataset_sidecars_for_actual_first_frame_bins(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source_report.json"
            source.write_text(
                json.dumps(
                    {
                        "episodes": [
                            _episode_with_stale_best_meta(seed=10, centroid=[230, 90]),
                            _episode_with_stale_best_meta(seed=11, centroid=[230, 90]),
                            _episode_with_stale_best_meta(seed=12, centroid=[230, 90]),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            camera1_sidecar = root / "camera1.parquet"
            camera2_sidecar = root / "camera2.parquet"
            pd.DataFrame(
                [
                    {"episode_index": 0, "visible": True, "grid_bin": 6, "area": 100},
                    {"episode_index": 1, "visible": True, "grid_bin": 7, "area": 100},
                    {"episode_index": 2, "visible": False, "grid_bin": -1, "area": 0},
                ]
            ).to_parquet(camera1_sidecar)
            pd.DataFrame(
                [
                    {"episode_index": 0, "visible": True, "grid_bin": 6, "area": 120},
                    {"episode_index": 1, "visible": True, "grid_bin": 6, "area": 120},
                    {"episode_index": 2, "visible": True, "grid_bin": 6, "area": 120},
                ]
            ).to_parquet(camera2_sidecar)

            filtered = filter_loop_start_report(
                source_report=source,
                output_report=root / "filtered_report.json",
                allowed_grid_bins=[6, 7],
                source_camera1_grid_bin_sidecar=camera1_sidecar,
                source_camera2_grid_bin_sidecar=camera2_sidecar,
                camera1_min_area=60,
                camera2_min_area=80,
            )

            self.assertEqual([episode["seed"] for episode in filtered["episodes"]], [10, 11])
            self.assertEqual([episode["loop_camera1_grid_bin_0_based"] for episode in filtered["episodes"]], [6, 7])
            decisions = [episode["loop_filter_decision"] for episode in filtered["episodes"]]
            self.assertEqual([decision["source_camera1_sidecar"]["grid_bin"] for decision in decisions], [6, 7])
            self.assertEqual(filtered["loop_filter"]["source_camera1_grid_bin_sidecar"], str(camera1_sidecar))

    def test_grip_from_above_config_uses_filtered_loop_report_when_artifact_exists(self) -> None:
        config_path = Path("configs/so101/training/qwen_edge_grip_from_above_edge_cube_only.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        test_case = config["closed_loop"]["test_cases"][0]
        self.assertEqual(test_case["episodes"], 6)
        self.assertIn("common_train_bins", test_case["start_report_path"])
        report_path = Path(test_case["start_report_path"])
        if not report_path.exists():
            self.skipTest(f"local loop report artifact is not available: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        loop_filter = report["loop_filter"]
        allowed = set(loop_filter["allowed_grid_bins_0_based"])
        self.assertEqual(set(allowed), {6, 7, 11})
        self.assertEqual(loop_filter["camera1_min_area"], 60)
        self.assertEqual(loop_filter["camera2_min_area"], 80)
        self.assertEqual(len(report["episodes"]), test_case["episodes"])
        for episode in report["episodes"]:
            decision = episode["loop_filter_decision"]
            self.assertTrue(decision["accepted"])
            self.assertGreaterEqual(decision["camera1_area"], loop_filter["camera1_min_area"])
            self.assertGreaterEqual(decision["camera2_area"], loop_filter["camera2_min_area"])
            self.assertIn(decision["grid_bin"], allowed)


def _episode(*, seed: int, camera1_area: int, camera2_area: int, centroid: list[float]) -> dict:
    return {
        "seed": seed,
        "q_start": [0, 0, 0, 0, 0, 0],
        "best_meta": {
            "preselected_policy_camera_visibility": {
                "camera1": {
                    "visible": camera1_area > 0,
                    "area": camera1_area,
                    "centroid": centroid,
                    "bbox": [centroid[0] - 1, centroid[1] - 1, centroid[0] + 1, centroid[1] + 1],
                },
                "camera2": {
                    "visible": camera2_area > 0,
                    "area": camera2_area,
                    "centroid": [128, 128],
                    "bbox": [120, 120, 136, 136],
                },
            }
        },
    }


def _episode_with_stale_best_meta(*, seed: int, centroid: list[float]) -> dict:
    episode = _episode(seed=seed, camera1_area=80, camera2_area=100, centroid=centroid)
    episode["best_meta"] = {
        "preselected_policy_camera_visibility": {
            "camera1": {"visible": True, "area": 80, "centroid": centroid},
            "camera2": {"visible": True, "area": 100, "centroid": [128, 128]},
        }
    }
    episode["start_policy_camera_visibility"] = {
        "camera1": {"visible": False, "area": 0, "centroid": None, "bbox": None},
        "camera2": {"visible": True, "area": 100, "centroid": [128, 128], "bbox": [120, 120, 136, 136]},
    }
    return episode


if __name__ == "__main__":
    unittest.main()
