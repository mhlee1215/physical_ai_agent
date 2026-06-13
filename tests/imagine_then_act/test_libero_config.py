from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from physical_ai_agent.imagine_then_act.libero_config import (
    ensure_noninteractive_libero_config,
    parse_simple_yaml,
)


ROOT = Path(__file__).resolve().parents[2]


def make_fake_libero_package(root: Path) -> Path:
    package_dir = root / "site-packages" / "libero" / "libero"
    (package_dir / "bddl_files").mkdir(parents=True)
    (package_dir / "init_files").mkdir(parents=True)
    (package_dir.parent / "datasets").mkdir(parents=True)
    return package_dir


class LiberoConfigTests(unittest.TestCase):
    def test_python_helper_writes_noninteractive_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package_dir = make_fake_libero_package(tmp_path)
            config_dir = tmp_path / "libero_config"
            assets_dir = tmp_path / "libero_assets"

            result = ensure_noninteractive_libero_config(
                config_dir=str(config_dir),
                assets_dir=str(assets_dir),
                libero_package_dir=str(package_dir),
            )

            config_path = Path(result.config_path)
            self.assertTrue(config_path.is_file())
            values = parse_simple_yaml(config_path)
            self.assertEqual(values["benchmark_root"], str(package_dir))
            self.assertEqual(values["assets"], str(assets_dir))
            self.assertEqual(values["bddl_files"], str(package_dir / "bddl_files"))
            self.assertEqual(values["init_states"], str(package_dir / "init_files"))
            self.assertEqual(os.environ["LIBERO_CONFIG_PATH"], str(config_dir))

    def test_runpod_prepare_libero_config_script_writes_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package_dir = make_fake_libero_package(tmp_path)
            config_dir = tmp_path / "libero_config"
            assets_dir = tmp_path / "libero_assets"
            env = os.environ.copy()
            env.update(
                {
                    "PROJECT_DIR": str(ROOT),
                    "PYTHON_BIN": sys.executable,
                    "PY312_VENV": str(tmp_path / "fake_venv"),
                    "LIBERO_PACKAGE_DIR": str(package_dir),
                    "LIBERO_CONFIG_PATH": str(config_dir),
                    "LIBERO_ASSETS_DIR": str(assets_dir),
                    "WORK_ROOT": str(tmp_path),
                }
            )
            completed = subprocess.run(
                ["sh", str(ROOT / "scripts" / "runpod_prepare_libero_config.sh")],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            config_path = config_dir / "config.yaml"
            self.assertTrue(config_path.is_file())
            values = parse_simple_yaml(config_path)
            self.assertEqual(values["benchmark_root"], str(package_dir))
            self.assertIn('"config_path"', completed.stdout)

    def test_runpod_gate_checks_libero_config_before_probe(self) -> None:
        shim = (ROOT / "scripts" / "runpod_check_libero_env.sh").read_text()
        self.assertIn("install/recipes/runpod_check_libero_env.sh", shim)
        gate = (ROOT / "scripts" / "install" / "recipes" / "runpod_check_libero_env.sh").read_text()
        self.assertIn("runpod_prepare_libero_config.sh", gate)
        self.assertIn("LIBERO_CONFIG_PATH", gate)
        self.assertIn("get_libero_path", gate)
        self.assertIn("libero_config_prompt", gate)


if __name__ == "__main__":
    unittest.main()
