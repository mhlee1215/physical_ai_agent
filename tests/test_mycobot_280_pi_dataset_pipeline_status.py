from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_mycobot_280_pi_dataset_pipeline_status import (
    check_mycobot_280_pi_dataset_pipeline_status,
)
from scripts.export_mycobot_280_pi_adaptive_lerobot_dataset import JOINT_NAMES


class MyCobot280PiDatasetPipelineStatusTest(unittest.TestCase):
    def test_missing_inputs_report_first_blocked_stage_and_next_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = check_mycobot_280_pi_dataset_pipeline_status(
                asset_root=tmp_path / "missing_mujoco",
                official_gripper_root=tmp_path / "missing_ros",
                input_trace=None,
                camera_manifest=None,
                jsonl_dataset_root=tmp_path / "jsonl",
                native_dataset_root=tmp_path / "native",
                smolvla_smoke_report=tmp_path / "native" / "smolvla_tiny_smoke.json",
                output=tmp_path / "status.json",
            )

            self.assertEqual(report.status, "blocked")
            self.assertEqual(report.first_blocked_stage, "profile_and_gate8_assets")
            self.assertEqual([stage.name for stage in report.stages], [
                "profile_and_gate8_assets",
                "capture_contract",
                "jsonl_dataset_export",
                "native_lerobot_dataset",
                "smolvla_tiny_smoke",
            ])
            self.assertIn("check_mycobot_280_pi_gate8_readiness.py", report.stages[0].next_command)
            self.assertIn("verify_mycobot_280_pi_capture_contract.py", report.stages[1].next_command)
            self.assertTrue((tmp_path / "status.json").exists())

    def test_synthetic_complete_pipeline_reports_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            asset_root, ros_root = _write_assets(tmp_path)
            trace_path, manifest_path = _write_capture(tmp_path)
            jsonl_root = _write_jsonl_dataset(tmp_path / "jsonl")
            native_root = _write_native_dataset(tmp_path / "native")
            smoke_report = native_root / "smolvla_tiny_smoke.json"
            smoke_report.write_text(
                json.dumps({"status": "passed", "loss_report": {"batches_evaluated": 1}}),
                encoding="utf-8",
            )

            with patch("importlib.util.find_spec", return_value=object()):
                report = check_mycobot_280_pi_dataset_pipeline_status(
                    asset_root=asset_root,
                    official_gripper_root=ros_root,
                    input_trace=trace_path,
                    camera_manifest=manifest_path,
                    jsonl_dataset_root=jsonl_root,
                    native_dataset_root=native_root,
                    smolvla_smoke_report=smoke_report,
                    output=None,
                )

            self.assertEqual(report.status, "passed")
            self.assertIsNone(report.first_blocked_stage)
            self.assertTrue(all(stage.status == "passed" for stage in report.stages))
            self.assertEqual(report.stages[-1].evidence["report"]["status"], "passed")


def _write_assets(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "mycobot_mujoco"
    ros_root = tmp_path / "mycobot_ros"
    (asset_root / "xml").mkdir(parents=True)
    (asset_root / "xml" / "mycobot_280jn_mujoco.xml").write_text("<mujoco />", encoding="utf-8")
    arm_dir = ros_root / "mycobot_description" / "urdf" / "mycobot_280_pi"
    gripper_dir = ros_root / "mycobot_description" / "urdf" / "adaptive_gripper"
    arm_dir.mkdir(parents=True)
    gripper_dir.mkdir(parents=True)
    (arm_dir / "mycobot_280_pi.urdf").write_text("<robot />", encoding="utf-8")
    (gripper_dir / "mycobot_adaptive_gripper.urdf").write_text("<robot />", encoding="utf-8")
    return asset_root, ros_root


def _write_capture(tmp_path: Path) -> tuple[Path, Path]:
    camera_dir = tmp_path / "camera"
    camera_dir.mkdir()
    frames = []
    records = []
    for index in range(2):
        top = camera_dir / f"top_{index}.ppm"
        wrist = camera_dir / f"wrist_{index}.ppm"
        _write_tiny_ppm(top, tint=20 + index)
        _write_tiny_ppm(wrist, tint=70 + index)
        frames.append({"top": str(top), "wrist": str(wrist), "timestamp": index / 12.0})
        state = [0.01 * (index + offset) for offset in range(7)]
        records.append(
            {
                "timestamp": index / 12.0,
                "joint_state": {"name": JOINT_NAMES, "position": state},
                "trajectory_point": {
                    "joint_names": JOINT_NAMES,
                    "positions": [value + 0.04 for value in state],
                },
                "object_pose": {"position": [0.12, -0.04, 0.02 + 0.03 * index]},
                "contacts": {"left_finger_pad": index >= 1, "right_finger_pad": False},
            }
        )
    trace_path = tmp_path / "trace.jsonl"
    manifest_path = tmp_path / "camera_manifest.json"
    trace_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return trace_path, manifest_path


def _write_jsonl_dataset(root: Path) -> Path:
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    for path in [
        root / "data" / "frames.jsonl",
        root / "data" / "episodes.jsonl",
        root / "meta" / "tasks.jsonl",
    ]:
        path.write_text("{}\n", encoding="utf-8")
    (root / "meta" / "info.json").write_text(json.dumps({"robot_type": "mycobot_280_pi_adaptive_gripper"}))
    (root / "meta" / "stats.json").write_text(json.dumps({"observation.state": {}}))
    (root / "meta" / "smolvla_tiny_smoke_plan.json").write_text(json.dumps({"status": "planned"}))
    return root


def _write_native_dataset(root: Path) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"robot_type": "mycobot_280_pi_adaptive_gripper", "fps": 12}),
        encoding="utf-8",
    )
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"PAR1fake")
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"PAR1fake")
    (root / "meta" / "tasks.parquet").write_bytes(b"PAR1fake")
    (root / "mycobot_280_pi_lerobot_convert_report.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    return root


def _write_tiny_ppm(path: Path, *, tint: int) -> None:
    pixels = bytes((tint, 30, 90, tint, 40, 100, tint, 50, 110, tint, 60, 120))
    path.write_bytes(b"P6\n2 2\n255\n" + pixels)


if __name__ == "__main__":
    unittest.main()
