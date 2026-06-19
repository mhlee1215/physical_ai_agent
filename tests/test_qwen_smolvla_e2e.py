from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import run_so101_qwen_smolvla_e2e as e2e_runner


class QwenSmolVLAE2ETest(unittest.TestCase):
    def test_mock_qwen_output_drives_smolvla_prompt_without_live_qwen(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        mock_response = repo_root / "configs" / "agent" / "qwen3_so101_tool_planner_mock_response.json"

        captured = {}

        def fake_smolvla_probe(**kwargs):
            captured.update(kwargs)
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "smolvla_real_inference_report.json"
            trace_path = output_dir / "smolvla_real_rollout.jsonl"
            blocker_path = output_dir / "smolvla_real_inference_blocker.md"
            report_path.write_text("{}", encoding="utf-8")
            trace_path.write_text('{"step": 0}\n', encoding="utf-8")
            blocker_path.write_text("", encoding="utf-8")
            return SimpleNamespace(
                status="passed",
                report_path=str(report_path),
                trace_path=str(trace_path),
                blocker_path=str(blocker_path),
                action_shape=[1, 6],
                rollout_steps=1,
            )

        with TemporaryDirectory() as tmpdir, mock.patch.object(
            e2e_runner,
            "SO101NexusEnv",
            FakeSO101Env,
        ), mock.patch.object(
            e2e_runner,
            "run_real_smolvla_inference_probe",
            fake_smolvla_probe,
        ):
            report = e2e_runner.run_e2e(
                task="pick and lift the green cube",
                target_object="green cube",
                qwen_model="Qwen/Qwen3-8B",
                qwen_base_url=None,
                qwen_response_json=mock_response,
                qwen_api_key=None,
                smolvla_model_id="lerobot/smolvla_base",
                env_id="FakeSO101-v0",
                output_dir=Path(tmpdir),
                rollout_steps=1,
                device="cpu",
                allow_download=False,
                use_real_camera_inputs=False,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["qwen"]["source"], "saved_response_json")
        self.assertEqual(report["qwen"]["validated_order"], ["move", "align", "pick_up"])
        self.assertEqual(report["qwen"]["primitive_ids"], [
            "move_over_cube_edge",
            "align_fixed_jaw_cube_edge",
            "grip_from_edge_cube",
        ])
        self.assertEqual(captured["observation"], [0.0, 0.1, 0.2])
        self.assertEqual(captured["action_dim"], 6)
        self.assertIn("validated SO101 primitive chain", captured["task_prompt"])
        self.assertIn("Move the static finger pad above one visible green cube edge.", captured["task_prompt"])

    def test_real_qwen_plan_drives_real_smolvla_so101_rollout(self) -> None:
        if os.environ.get("RUN_QWEN_SMOLVLA_E2E") != "1":
            self.skipTest("set RUN_QWEN_SMOLVLA_E2E=1 to run the real Qwen + SmolVLA E2E")
        qwen_base_url = os.environ.get("QWEN_OPENAI_BASE_URL")
        if not qwen_base_url:
            self.skipTest("set QWEN_OPENAI_BASE_URL to an OpenAI-compatible Qwen endpoint")

        with TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "scripts/run_so101_qwen_smolvla_e2e.py",
                "--qwen-base-url",
                qwen_base_url,
                "--qwen-model",
                os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B"),
                "--smolvla-model-id",
                os.environ.get("SMOLVLA_MODEL_ID", "lerobot/smolvla_base"),
                "--output-dir",
                tmpdir,
                "--rollout-steps",
                os.environ.get("QWEN_SMOLVLA_E2E_ROLLOUT_STEPS", "1"),
                "--device",
                os.environ.get("SMOLVLA_DEVICE", "auto"),
                "--require-pass",
            ]
            if os.environ.get("QWEN_OPENAI_API_KEY"):
                cmd.extend(["--qwen-api-key", os.environ["QWEN_OPENAI_API_KEY"]])
            if os.environ.get("SMOLVLA_ALLOW_DOWNLOAD") == "1":
                cmd.append("--allow-download")
            if os.environ.get("SMOLVLA_USE_REAL_CAMERA_INPUTS") == "1":
                cmd.append("--use-real-camera-inputs")

            completed = subprocess.run(
                cmd,
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            report = json.loads(completed.stdout)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["qwen"]["validated_order"], ["move", "align", "pick_up"])
            self.assertEqual(report["smolvla"]["status"], "passed")
            self.assertGreaterEqual(report["smolvla"]["rollout_steps"], 1)
            self.assertTrue(Path(report["smolvla"]["trace_path"]).is_file())
            self.assertIn("validated SO101 primitive chain", report["smolvla"]["task_prompt"])


class FakeSO101Env:
    def __init__(self, env_id: str, render_mode: str | None = None) -> None:
        self.env_id = env_id
        self.render_mode = render_mode
        self.action_dim = 6

    def reset(self, seed: int):
        return [0.0, 0.1, 0.2], {"seed": seed}

    def close(self) -> None:
        pass
