from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("scripts").resolve()))

from physical_ai_agent.so101_closed_loop_contract import (
    contract_path_for_start_report,
    load_executable_loop_test_contract,
    write_executable_loop_test_contract,
)
from scripts import build_so101_closed_loop_start_report as start_report_builder
from scripts import serve_so101_dataset_viewer as dataset_viewer
from scripts import verify_so101_dataset_completion as completion_gate


CAMERA_RIG = Path(
    "configs/so101/camera_rigs/"
    "official_32x32_uvc_photoreal_v10_fov_calibrated_direct_square.json"
)


class SO101ClosedLoopContractTests(unittest.TestCase):
    def test_generated_contract_is_executable_and_viewer_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "validation"
            info_path = dataset_root / "meta/info.json"
            info_path.parent.mkdir(parents=True)
            info_path.write_text(
                json.dumps({"total_episodes": 50, "total_frames": 9000}),
                encoding="utf-8",
            )
            start_report = dataset_root / "meta/closed_loop/start10.json"
            start_report.parent.mkdir(parents=True)
            start_report.write_text(json.dumps({"episodes": [{}] * 10}), encoding="utf-8")

            payload = start_report_builder.build_contract(
                start_report_path=start_report,
                dataset_root=dataset_root,
                dataset_name="validation",
                dataset_repo_id="example/validation",
                test_case_id="cube_loop_test",
                description="Held-out cube starts.",
                episodes=10,
                steps=200,
                seed=98100,
                task_prompt="grip the green cube and lift",
                success_metric="env_success",
                camera_rig_config=str(CAMERA_RIG),
                target_object_color="green",
                object_half_sizes=[0.015],
                spawn_center=(0.15, 0.0),
                spawn_min_radius=0.1,
                spawn_max_radius=0.3,
                spawn_angle_half_range_deg=90.0,
                source_recipe="configs/recipe.json",
                source_split="validation",
            )
            contract_path = write_executable_loop_test_contract(
                contract_path_for_start_report(start_report),
                payload,
            )
            loaded = load_executable_loop_test_contract(
                contract_path,
                repo_root=repo_root,
                expected_start_report=start_report,
            )
            selected = dataset_viewer._closed_loop_test_case_for_candidate(
                repo_root,
                dataset_root=dataset_root,
                start_report=start_report,
                test_cases=[],
            )
            recipe = SimpleNamespace(
                splits={
                    "validation": SimpleNamespace(
                        output_root=str(dataset_root),
                        closed_loop=SimpleNamespace(
                            output="meta/closed_loop/start10.json",
                            episodes=10,
                        ),
                    )
                }
            )
            verified = completion_gate.require_executable_loop_test_contracts(
                repo_root,
                recipe,
            )
            self.assertEqual(
                [Path(value).resolve() for value in verified],
                [contract_path.resolve()],
            )
            contract_path.unlink()
            with self.assertRaisesRegex(
                FileNotFoundError,
                "executable loop-test contract is missing",
            ):
                completion_gate.require_executable_loop_test_contracts(
                    repo_root,
                    recipe,
                )

        self.assertEqual(loaded["test_case"]["id"], "cube_loop_test")
        self.assertEqual(loaded["test_case"]["episodes"], 10)
        self.assertEqual(
            loaded["observation_renderer"]["camera_rig_config"],
            str(CAMERA_RIG),
        )
        self.assertEqual(selected["id"], "cube_loop_test")
        self.assertIn("_observation_renderer_contract", selected)
        self.assertEqual(
            Path(selected["_contract_path"]).resolve(),
            contract_path.resolve(),
        )

    def test_contract_loader_rejects_a_different_start_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / "start10.contract.json"
            contract_path.write_text(
                json.dumps(
                    {
                        "format": "so101_executable_loop_test_contract_v1",
                        "schema_version": 1,
                        "test_case": {
                            "id": "loop",
                            "episodes": 1,
                            "start_report_path": str(Path(tmpdir) / "start10.json"),
                            "start_dataset": {
                                "repo_id": "example/validation",
                                "root": str(Path(tmpdir) / "validation"),
                            },
                        },
                        "observation_renderer": start_report_builder.observation_renderer_from_camera_rig(
                            CAMERA_RIG,
                            camera_rig_config=str(CAMERA_RIG),
                        ),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "start report mismatch"):
                load_executable_loop_test_contract(
                    contract_path,
                    expected_start_report=Path(tmpdir) / "other.json",
                )


if __name__ == "__main__":
    unittest.main()
