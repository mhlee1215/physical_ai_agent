from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from physical_ai_agent.so101_lerobot_concat import GridBinBalancedDataset


class GridBinBalancedSamplerTest(unittest.TestCase):
    def test_weights_give_equal_total_mass_per_occupied_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar_path = Path(tmp) / "bins.parquet"
            pd.DataFrame(
                [
                    {"episode_index": 0, "visible": True, "grid_bin": 5},
                    {"episode_index": 1, "visible": True, "grid_bin": 5},
                    {"episode_index": 2, "visible": True, "grid_bin": 10},
                    {"episode_index": 3, "visible": False, "grid_bin": -1},
                ]
            ).to_parquet(sidecar_path, index=False)

            wrapped = GridBinBalancedDataset(_FakeDataset(), sidecar_path)
            weights = wrapped.grid_bin_sample_weights(drop_n_last_frames=1)

            self.assertAlmostEqual(float(weights.sum()), 1.0)
            self.assertAlmostEqual(float(weights[0:2].sum() + weights[3:5].sum()), 1.0 / 3.0)
            self.assertAlmostEqual(float(weights[6:8].sum()), 1.0 / 3.0)
            self.assertAlmostEqual(float(weights[9:11].sum()), 1.0 / 3.0)
            self.assertEqual(float(weights[2]), 0.0)
            self.assertEqual(float(weights[5]), 0.0)
            self.assertEqual(float(weights[8]), 0.0)
            self.assertEqual(float(weights[11]), 0.0)


class _FakeDataset:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(
            episodes=pd.DataFrame(
                [
                    {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 3},
                    {"episode_index": 1, "dataset_from_index": 3, "dataset_to_index": 6},
                    {"episode_index": 2, "dataset_from_index": 6, "dataset_to_index": 9},
                    {"episode_index": 3, "dataset_from_index": 9, "dataset_to_index": 12},
                ]
            )
        )

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> dict:
        return {"index": index}


if __name__ == "__main__":
    unittest.main()
