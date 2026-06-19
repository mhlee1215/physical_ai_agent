from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class QwenSmolVLAE2ETest(unittest.TestCase):
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
