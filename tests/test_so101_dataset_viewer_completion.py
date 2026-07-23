from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from physical_ai_agent.so101_dataset_registry import (
    DatasetRegistry,
    DatasetRegistryEntry,
)
from physical_ai_agent.so101_dataset_viewer_gate import (
    DatasetViewerGateError,
    verify_dataset_viewer_api,
)


class _ViewerHandler(BaseHTTPRequestHandler):
    include_dataset = True
    include_camera2 = True
    drop_catalog_connection = False

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/datasets":
            if self.drop_catalog_connection:
                self.close_connection = True
                return
            datasets = {
                "cube_v3": {"episodes": 4, "frames": 40}
            } if self.include_dataset else {}
            self._json({"datasets": datasets})
            return
        if self.path.startswith("/api/frame?"):
            images = {
                "observation.images.camera1": "data:image/png;base64,AA==",
            }
            if self.include_camera2:
                images["observation.images.camera2"] = "data:image/png;base64,AA=="
            self._json(
                {
                    "split": "cube_v3",
                    "episode": 0,
                    "frame": 0,
                    "prompt": "grip the green cube and lift",
                    "images": images,
                }
            )
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class SO101DatasetViewerCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        _ViewerHandler.include_dataset = True
        _ViewerHandler.include_camera2 = True
        _ViewerHandler.drop_catalog_connection = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ViewerHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_completion_gate_requires_catalog_and_real_policy_cameras(self) -> None:
        result = verify_dataset_viewer_api(
            self.base_url,
            [_entry()],
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )

        self.assertEqual(result.datasets[0]["split"], "cube_v3")
        self.assertEqual(
            result.datasets[0]["cameras"],
            ["observation.images.camera1", "observation.images.camera2"],
        )

    def test_completion_gate_fails_when_recipe_split_is_not_listed(self) -> None:
        _ViewerHandler.include_dataset = False

        with self.assertRaisesRegex(DatasetViewerGateError, "missing recipe split"):
            verify_dataset_viewer_api(
                self.base_url,
                [_entry()],
                timeout_seconds=0.05,
                poll_interval_seconds=0.005,
            )

    def test_completion_gate_fails_on_stale_viewer_empty_response(self) -> None:
        _ViewerHandler.drop_catalog_connection = True

        with self.assertRaisesRegex(DatasetViewerGateError, "completion contract"):
            verify_dataset_viewer_api(
                self.base_url,
                [_entry()],
                timeout_seconds=0.05,
                poll_interval_seconds=0.005,
            )

    def test_completion_gate_fails_when_frame_camera_contract_is_incomplete(self) -> None:
        _ViewerHandler.include_camera2 = False

        with self.assertRaisesRegex(
            DatasetViewerGateError, "observation.images.camera2"
        ):
            verify_dataset_viewer_api(
                self.base_url,
                [_entry()],
                timeout_seconds=0.05,
                poll_interval_seconds=0.005,
            )

    def test_recipe_generator_always_ends_with_live_viewer_completion_gate(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/generate_so101_dataset_recipe.py",
                "--recipe",
                "configs/so101/dataset_generation/grip_the_cube_v2_5_photoreal.json",
                "--dry-run",
            ],
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        stages = json.loads(completed.stdout)["stages"]
        self.assertEqual(stages[-1]["name"], "completion:registry-viewer")
        command = stages[-1]["command"]
        self.assertIn("scripts/verify_so101_dataset_completion.py", command)
        self.assertNotIn("--no-restart-viewer", command)
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command) if value == "--split"],
            ["train", "validation"],
        )

    def test_launchctl_viewer_script_is_valid_shell(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", "scripts/launch_so101_dataset_viewer.sh"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_validation_closed_loop_report_is_exposed_as_virtual_dataset(self) -> None:
        import tempfile

        sys.path.insert(0, str(Path("scripts").resolve()))
        import serve_so101_dataset_viewer as viewer

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "loop10.json"
            report.write_text('{"episodes": []}', encoding="utf-8")
            entry = _entry(closed_loop_start=str(report))
            registry = DatasetRegistry(
                repo_root=str(root),
                recipe_dir="configs/so101/dataset_generation",
                dataset_root="_workspace/so101_lerobot",
                entries=(entry,),
                issues=(),
            )
            with patch.object(viewer, "scan_dataset_registry", return_value=registry):
                views = viewer._generation_closed_loop_views(root)

        self.assertEqual(set(views), {"cube_v3_loop_test"})
        self.assertEqual(views["cube_v3_loop_test"]["report"], report.resolve())


def _entry(*, closed_loop_start: str | None = None) -> DatasetRegistryEntry:
    return DatasetRegistryEntry(
        dataset_id="cube_v3",
        split="train",
        catalog_name="cube_v3",
        recipe_path="configs/so101/dataset_generation/cube_v3.json",
        output_root="_workspace/so101_lerobot/cube_v3",
        absolute_root=str(Path("_workspace/so101_lerobot/cube_v3").resolve()),
        repo_id="physical-ai-agent/cube-v3",
        expected_episodes=4,
        status="available",
        episodes=4,
        frames=40,
        fps=12,
        size_bytes=1,
        audit_status="passed",
        grid_sidecar="meta/camera_grid_bins/camera1.parquet",
        closed_loop_start=closed_loop_start,
        training_ready=True,
        readiness_errors=(),
    )


if __name__ == "__main__":
    unittest.main()
