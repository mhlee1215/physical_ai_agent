from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import json
import types

from physical_ai_agent.checkpoints import checkpoint_25


class _FakeActionSpec:
    shape = (12,)


class _FakeEnv:
    action_spec = (_FakeActionSpec(),)

    def __init__(self) -> None:
        self.closed = False

    def reset(self) -> dict[str, object]:
        return {"robot0_agentview_left_image": object()}

    def get_ep_meta(self) -> dict[str, str]:
        return {"lang": "Close the fridge."}

    def step(self, action: object) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        return {"robot0_agentview_left_image": object()}, 1.0, False, {"success": True}

    def close(self) -> None:
        self.closed = True


class Checkpoint25Test(TestCase):
    def _missing_robocasa_import(self, name: str) -> object:
        if name == "robocasa":
            raise ModuleNotFoundError("No module named 'robocasa'")
        if name == "robosuite":
            return types.SimpleNamespace()
        return __import__(name)

    def test_non_strict_checkpoint_writes_blocker_when_robocasa_missing(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.object(
            checkpoint_25.importlib,
            "import_module",
            self._missing_robocasa_import,
        ):
            report = checkpoint_25.run_checkpoint(output_dir=Path(tmpdir))

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.metrics["probe_status"], "blocked")
            self.assertIn("robocasa import failed", str(report.metrics["probe_blocker"]))
            self.assertTrue(Path(report.artifacts["robocasa_blocker"]).exists())
            self.assertTrue(Path(report.artifacts["robocasa_install_and_eval"]).exists())
            self.assertTrue(Path(report.artifacts["robocasa365_reference_table"]).exists())

    def test_require_robocasa_fails_when_import_missing(self) -> None:
        with TemporaryDirectory() as tmpdir, patch.object(
            checkpoint_25.importlib,
            "import_module",
            self._missing_robocasa_import,
        ):
            report = checkpoint_25.run_checkpoint(
                output_dir=Path(tmpdir),
                require_robocasa=True,
            )

            self.assertEqual(report.status, "failed")
            self.assertFalse(report.checks["cp25_require_robocasa_import"])

    def test_import_only_passes_when_dependencies_import(self) -> None:
        def fake_import(name: str) -> object:
            if name in {"robocasa", "robosuite"}:
                return types.SimpleNamespace()
            return __import__(name)

        with TemporaryDirectory() as tmpdir, patch.object(checkpoint_25.importlib, "import_module", fake_import):
            report = checkpoint_25.run_checkpoint(
                output_dir=Path(tmpdir),
                require_robocasa=True,
                probe_reset_step=False,
            )

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.metrics["probe_status"], "import_only")
            self.assertTrue(report.checks["cp25_require_robocasa_import"])

    def test_probe_reset_step_records_trace_and_success(self) -> None:
        def fake_import(name: str) -> object:
            if name in {"robocasa", "robosuite"}:
                return types.SimpleNamespace()
            if name == "robocasa.utils.env_utils":
                return types.SimpleNamespace(create_env=lambda **_: _FakeEnv())
            return __import__(name)

        with TemporaryDirectory() as tmpdir, patch.object(checkpoint_25.importlib, "import_module", fake_import):
            report = checkpoint_25.run_checkpoint(
                output_dir=Path(tmpdir),
                require_robocasa=True,
                probe_reset_step=True,
            )
            metrics = json.loads(Path(report.artifacts["robocasa_metrics"]).read_text(encoding="utf-8"))
            trace = Path(report.artifacts["robocasa_trace"]).read_text(encoding="utf-8")

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.metrics["probe_status"], "passed")
            self.assertEqual(metrics["language"], "Close the fridge.")
            self.assertEqual(metrics["success"], True)
            self.assertIn('"success": true', trace)

    def test_reference_table_contains_leaderboard_and_pending_row(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reference.md"
            checkpoint_25._write_reference_table(path)
            text = path.read_text(encoding="utf-8")

            self.assertIn("RLDX-1", text)
            self.assertIn("Diffusion Policy", text)
            self.assertIn("Our SmolVLA RoboCasa365 run", text)
            self.assertIn("20-episodes-per-task", text)
