from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_ready_path_fixture import build_ready_path_fixture


class BuildRealSO100ReadyPathFixtureTest(TestCase):
    def test_builds_ready_path_manifest_without_robot_motion(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vla_prompt_packet = _write_json(tmp / "vla_prompt_packet.json", {"vla_prompt": {"target": "SmolVLA"}})

            manifest = build_ready_path_fixture(
                output_dir=tmp / "fixture",
                reports_dir=tmp / "reports",
                vla_prompt_packet=vla_prompt_packet,
                contact_output_dir="_workspace/real_so100/contact_probe_fixture_test",
            )
            plan = json.loads(Path(manifest["next_plan"]).read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "passed")
        self.assertFalse(manifest["physical_robot_motion"])
        self.assertFalse(manifest["send_action_called"])
        self.assertEqual(manifest["stage"], "minimal_contact_probe")
        self.assertEqual(
            manifest["next_step_types"],
            [
                "execute_video_backed_contact_probe",
                "materialize_relocation_verifier_packet",
                "run_relocation_verifier",
                "rebuild_agentic_contract",
            ],
        )
        self.assertTrue(all(check["passed"] for check in manifest["checks"]))
        self.assertEqual(plan["vla_prompt_packet"], str(vla_prompt_packet))
        self.assertIn("_workspace/real_so100/contact_probe_fixture_test/visual", plan["next_steps"][0]["command"])
        self.assertIn("scripts/build_real_so100_relocation_verifier_packet.py", plan["next_steps"][1]["command"])
        self.assertIn("_workspace/real_so100/contact_probe_fixture_test/visual/before.jpg", plan["next_steps"][2]["command"])


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
