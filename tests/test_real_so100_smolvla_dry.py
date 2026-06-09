from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from physical_ai_agent.policies.smolvla_real import _select_policy_device
from scripts.real_so100_smolvla_dry import run_dry_inference


class RealSO100SmolVLADryTest(TestCase):
    def test_dry_inference_records_ten_step_action_chunk(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = _write_episode(tmp)
            output_dir = tmp / "out"

            with (
                patch("scripts.real_so100_smolvla_dry._read_rgb", return_value=[]),
                patch("physical_ai_agent.policies.smolvla_real._load_pretrained_policy", return_value=_fake_policy_with_device()),
                patch("physical_ai_agent.policies.smolvla_real._build_batch_for_policy", return_value=(_fake_batch(), {"0": "wrist_cam", "1": "egocentric_cam"})),
                patch("physical_ai_agent.policies.smolvla_real._clip_action", side_effect=lambda action, _dim: action),
            ):
                report = run_dry_inference(
                    episode=episode,
                    frame_index=0,
                    output_dir=output_dir,
                    instruction="Pick up the green Android figure and move it to the right.",
                    model_id="fake-smolvla",
                    local_files_only=True,
                    wrist_camera_index="0",
                    egocentric_camera_index="1",
                    observer_camera_indexes=["3"],
                    action_steps=10,
                    device="auto",
                )

            action_payload = json.loads((output_dir / "smolvla_action_chunk.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["raw_action_dim"], 6)
        self.assertEqual(report["raw_action_chunk_steps"], 10)
        self.assertEqual(report["predicted_chunk_size"], 12)
        self.assertEqual(report["planned_action_steps"], 10)
        self.assertEqual(report["executed_action_steps"], 10)
        self.assertEqual(report["device_requested"], "auto")
        self.assertIn(report["device_selected"], {"mps", "cpu"})
        self.assertEqual(len(action_payload["raw_action_chunk"]), 10)
        self.assertEqual(len(action_payload["raw_action_chunk"][0]), 6)
        self.assertEqual(action_payload["raw_action"], action_payload["raw_action_chunk"][0])
        self.assertEqual(action_payload["first_action"], action_payload["raw_action_chunk"][0])
        self.assertIn("not one isolated action", action_payload["action_chunk_semantics"])
        self.assertFalse(action_payload["safe_to_execute"])

    def test_dry_inference_blocks_when_requested_chunk_exceeds_prediction(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = _write_episode(tmp)
            output_dir = tmp / "out"

            with (
                patch("scripts.real_so100_smolvla_dry._read_rgb", return_value=[]),
                patch("physical_ai_agent.policies.smolvla_real._load_pretrained_policy", return_value=_fake_policy_with_device()),
                patch("physical_ai_agent.policies.smolvla_real._build_batch_for_policy", return_value=(_fake_batch(), {})),
            ):
                report = run_dry_inference(
                    episode=episode,
                    frame_index=0,
                    output_dir=output_dir,
                    instruction="Pick up the green Android figure.",
                    model_id="fake-smolvla",
                    local_files_only=True,
                    action_steps=13,
                )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("action_steps must be in [1, 12]", report["blocker"])

    def test_dry_inference_converts_raw_state_to_lerobot_position_with_calibration(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            episode = _write_episode(tmp)
            output_dir = tmp / "out"
            calibration = _write_calibration(tmp)

            with (
                patch("scripts.real_so100_smolvla_dry._read_rgb", return_value=[]),
                patch("physical_ai_agent.policies.smolvla_real._load_pretrained_policy", return_value=_fake_policy_with_device()),
                patch("physical_ai_agent.policies.smolvla_real._build_batch_for_policy", return_value=(_fake_batch(), {})) as build_batch,
                patch("physical_ai_agent.policies.smolvla_real._clip_action", side_effect=lambda action, _dim: action),
            ):
                report = run_dry_inference(
                    episode=episode,
                    frame_index=0,
                    output_dir=output_dir,
                    instruction="Pick up the green Android figure.",
                    model_id="fake-smolvla",
                    local_files_only=True,
                    action_steps=10,
                    calibration=calibration,
                )

        converted_state = build_batch.call_args.args[1]
        self.assertEqual(report["policy_state_units"], "lerobot_so100_position")
        self.assertAlmostEqual(converted_state[0], -47.3846, places=3)
        self.assertAlmostEqual(converted_state[5], -4.0, places=3)

    def test_auto_device_prefers_mps_when_available(self) -> None:
        fake_torch = _fake_torch(mps_built=True, mps_available=True, cuda_available=False)

        plan = _select_policy_device("auto", torch_module=fake_torch)

        self.assertEqual(plan["selected"], "mps")
        self.assertIsNone(plan["fallback_reason"])

    def test_auto_device_falls_back_to_cpu_when_mps_unavailable(self) -> None:
        fake_torch = _fake_torch(mps_built=True, mps_available=False, cuda_available=False)

        plan = _select_policy_device("auto", torch_module=fake_torch)

        self.assertEqual(plan["selected"], "cpu")
        self.assertIn("mps.is_available", plan["fallback_reason"])


class _FakePolicy:
    def predict_action_chunk(self, _batch):
        import torch

        return torch.tensor(
            [
                [
                    [float(step * 10 + joint) for joint in range(6)]
                    for step in range(12)
                ]
            ],
            dtype=torch.float32,
        )


def _fake_policy_with_device():
    policy = _FakePolicy()
    policy._physical_ai_agent_device_requested = "auto"
    policy._physical_ai_agent_device_selected = "cpu"
    policy._physical_ai_agent_device_probe = {"mps_available": False}
    policy._physical_ai_agent_device_fallback_reason = "MPS unavailable in test."
    return policy


def _fake_torch(*, mps_built: bool, mps_available: bool, cuda_available: bool):
    return SimpleNamespace(
        __version__="fake",
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_built=lambda: mps_built,
                is_available=lambda: mps_available,
            )
        ),
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )


def _fake_batch():
    import torch

    return {
        "observation.language.tokens": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        "observation.language.attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
    }


def _write_episode(tmp: Path) -> Path:
    episode = tmp / "episode.jsonl"
    payload = {
        "frame_index": 0,
        "observation": {
            "state": {
                "shoulder_pan": 1,
                "shoulder_lift": 2,
                "elbow_flex": 3,
                "wrist_flex": 4,
                "wrist_roll": 5,
                "gripper": 6,
            },
            "images": {"0": str(tmp / "wrist.png"), "1": str(tmp / "ego.png"), "3": str(tmp / "observer.png")},
        },
    }
    episode.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return episode


def _write_calibration(tmp: Path) -> Path:
    calibration = tmp / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "shoulder_pan": {"range_min": 0, "range_max": 1080},
                "shoulder_lift": {"range_min": 0, "range_max": 1080},
                "elbow_flex": {"range_min": 0, "range_max": 1080},
                "wrist_flex": {"range_min": 0, "range_max": 1080},
                "wrist_roll": {"range_min": 0, "range_max": 1080},
                "gripper": {"range_min": 10, "range_max": 110},
            }
        ),
        encoding="utf-8",
    )
    return calibration
