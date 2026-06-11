from __future__ import annotations

from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[2]


class InstallScriptConsolidationTests(TestCase):
    def test_four_canonical_install_scripts_exist(self) -> None:
        expected = {
            "local_install.sh",
            "runpod_install.sh",
            "local_check.sh",
            "runpod_check.sh",
        }
        actual = {path.name for path in (ROOT / "scripts" / "install").glob("*.sh")}

        self.assertTrue(expected.issubset(actual))

    def test_checkpoint_install_scripts_are_compatibility_shims(self) -> None:
        mappings = {
            "bootstrap_checkpoint_01.sh": "--checkpoint 01",
            "bootstrap_checkpoint_05_06.sh": "--checkpoint 05-06",
            "bootstrap_checkpoint_07_13.sh": "--checkpoint 07-13",
            "bootstrap_checkpoint_14_15.sh": "--checkpoint 14-15",
            "bootstrap_checkpoint_24.sh": "--checkpoint 24",
        }
        for filename, flag in mappings.items():
            with self.subTest(filename=filename):
                text = (ROOT / "scripts" / "install" / filename).read_text(encoding="utf-8")
                self.assertIn("local_install.sh", text)
                self.assertIn(flag, text)

    def test_runpod_canonical_scripts_cover_install_and_check_components(self) -> None:
        install = (ROOT / "scripts" / "install" / "runpod_install.sh").read_text(encoding="utf-8")
        check = (ROOT / "scripts" / "install" / "runpod_check.sh").read_text(encoding="utf-8")

        self.assertIn("libero-smolvla", install)
        self.assertIn("risk1b-vlm", install)
        self.assertIn("libero-config", install)
        self.assertIn("libero-smolvla", check)
        self.assertIn("risk1b-vlm", check)
