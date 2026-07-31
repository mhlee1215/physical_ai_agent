from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from physical_ai_agent.policies.mycobot280_smolvla_contract import (
    ACTION_KEY,
    CAMERA_KEY,
    CONTRACT_FILENAME,
    STATE_KEY,
    policy_feature_contract,
    stabilize_zero_variance_statistics,
    validate_saved_processor_contract,
)


class MyCobot280SmolVLAContractTest(unittest.TestCase):
    def test_policy_contract_reports_exact_7d_single_camera(self) -> None:
        config = SimpleNamespace(
            input_features={
                STATE_KEY: SimpleNamespace(shape=(7,)),
                CAMERA_KEY: SimpleNamespace(shape=(3, 256, 256)),
            },
            output_features={ACTION_KEY: SimpleNamespace(shape=(7,))},
        )

        report = policy_feature_contract(config)

        self.assertTrue(report["exact_7d_state_action"])
        self.assertEqual(report["state_shape"], [7])
        self.assertEqual(report["action_shape"], [7])
        self.assertEqual(report["camera_keys"], [CAMERA_KEY])

    def test_constant_action_dimension_is_centered_without_mutating_source(self) -> None:
        stats = {
            STATE_KEY: {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
            },
            ACTION_KEY: {
                "mean": [0.0, 0.0, 0.0, 0.0, -0.500001, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                "min": [-1.0, -1.0, -1.0, -1.0, -0.5, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0, 1.0, -0.5, 1.0, 1.0],
            },
        }

        stabilized, adjustments = stabilize_zero_variance_statistics(stats)

        self.assertEqual(stats[ACTION_KEY]["mean"][4], -0.500001)
        self.assertEqual(stabilized[ACTION_KEY]["mean"][4], -0.5)
        self.assertEqual(
            adjustments,
            [
                {
                    "feature": ACTION_KEY,
                    "dimension": 4,
                    "original_mean": -0.500001,
                    "replacement_mean": -0.5,
                    "std": 0.0,
                    "observed_range": 0.0,
                    "reason": "constant_dimension_centering",
                }
            ],
        )

    def test_saved_processor_contract_rejects_six_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            _write_processor_configs(checkpoint, state_dim=6, action_dim=6)

            with self.assertRaisesRegex(ValueError, "expected \\[7\\]"):
                validate_saved_processor_contract(checkpoint)

    def test_saved_processor_contract_accepts_exact_seven_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            _write_processor_configs(checkpoint, state_dim=7, action_dim=7)

            report = validate_saved_processor_contract(checkpoint)

        self.assertTrue(report["exact_7d_state_action"])
        self.assertEqual(report["state_shape"], [7])
        self.assertEqual(report["action_shape"], [7])
        self.assertEqual(report["camera_keys"], [CAMERA_KEY])


def _write_processor_configs(checkpoint: Path, *, state_dim: int, action_dim: int) -> None:
    preprocessor = {
        "steps": [
            {
                "config": {
                    "features": {
                        STATE_KEY: {"type": "STATE", "shape": [state_dim]},
                        CAMERA_KEY: {"type": "VISUAL", "shape": [3, 256, 256]},
                        ACTION_KEY: {"type": "ACTION", "shape": [action_dim]},
                    }
                }
            }
        ]
    }
    postprocessor = {
        "steps": [
            {
                "config": {
                    "features": {
                        ACTION_KEY: {"type": "ACTION", "shape": [action_dim]},
                    }
                }
            }
        ]
    }
    contract = {
        "feature_contract": {
            "exact_7d_state_action": state_dim == 7 and action_dim == 7,
        },
        "processor_source": "dataset_statistics",
        "normalization_adjustments": [],
    }
    (checkpoint / CONTRACT_FILENAME).write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(preprocessor), encoding="utf-8"
    )
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps(postprocessor), encoding="utf-8"
    )


if __name__ == "__main__":
    unittest.main()
