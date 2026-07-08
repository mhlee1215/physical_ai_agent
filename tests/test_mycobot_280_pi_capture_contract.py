from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_mycobot_280_pi_adaptive_lerobot_dataset import JOINT_NAMES
from scripts.verify_mycobot_280_pi_capture_contract import (
    verify_mycobot_280_pi_capture_contract,
)


class MyCobot280PiCaptureContractTest(unittest.TestCase):
    def test_valid_capture_contract_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path, manifest_path = _write_capture_fixture(tmp_path, object_pose=True)

            report = verify_mycobot_280_pi_capture_contract(
                input_trace=trace_path,
                camera_manifest=manifest_path,
                output_dir=tmp_path / "verify",
            )

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.frame_count, 3)
            self.assertEqual(report.failed_frame_count, 0)
            self.assertEqual(report.joint_names, JOINT_NAMES)
            self.assertEqual(report.required_cameras, ["top", "wrist"])
            self.assertEqual(report.checks[-1].contact_count, 2)
            self.assertTrue(Path(report.artifacts["json"]).exists())
            self.assertTrue(Path(report.artifacts["markdown"]).exists())

    def test_capture_contract_rejects_missing_object_pose_and_camera_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path, manifest_path = _write_capture_fixture(
                tmp_path,
                object_pose=False,
                missing_camera=True,
            )

            report = verify_mycobot_280_pi_capture_contract(
                input_trace=trace_path,
                camera_manifest=manifest_path,
                output_dir=tmp_path / "verify",
            )

            self.assertEqual(report.status, "failed")
            self.assertGreater(report.failed_frame_count, 0)
            errors = "\n".join(error for check in report.checks for error in check.errors)
            self.assertIn("missing object_pose/object_state position", errors)
            self.assertIn("missing wrist camera file", errors)

    def test_capture_contract_rejects_mismatched_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path, manifest_path = _write_capture_fixture(tmp_path, object_pose=True)
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["frames"] = manifest_payload["frames"][:1]
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

            report = verify_mycobot_280_pi_capture_contract(
                input_trace=trace_path,
                camera_manifest=manifest_path,
                output_dir=tmp_path / "verify",
            )

            self.assertEqual(report.status, "failed")
            self.assertEqual(report.frame_count, 3)
            self.assertIn("missing camera manifest record", report.checks[-1].errors)


def _write_capture_fixture(
    tmp_path: Path,
    *,
    object_pose: bool,
    missing_camera: bool = False,
) -> tuple[Path, Path]:
    camera_dir = tmp_path / "camera"
    camera_dir.mkdir()
    manifest_frames = []
    trace_records = []
    for index in range(3):
        top = camera_dir / f"top_{index}.ppm"
        wrist = camera_dir / f"wrist_{index}.ppm"
        _write_tiny_ppm(top, tint=20 + index)
        if not missing_camera or index != 1:
            _write_tiny_ppm(wrist, tint=70 + index)
        manifest_frames.append({"top": str(top), "wrist": str(wrist), "timestamp": index / 12.0})
        state = [0.01 * (index + offset) for offset in range(7)]
        record: dict[str, object] = {
            "timestamp": index / 12.0,
            "joint_state": {"name": JOINT_NAMES, "position": state},
            "trajectory_point": {
                "joint_names": JOINT_NAMES,
                "positions": [value + 0.04 for value in state],
            },
            "contacts": {"left_finger_pad": index >= 1, "right_finger_pad": index >= 2},
            "task": "Pick up the test cube.",
        }
        if object_pose:
            record["object_pose"] = {"position": [0.12, -0.04, 0.02 + 0.02 * index]}
        trace_records.append(record)
    trace_path = tmp_path / "trace.jsonl"
    manifest_path = tmp_path / "camera_manifest.json"
    trace_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in trace_records) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"frames": manifest_frames}), encoding="utf-8")
    return trace_path, manifest_path


def _write_tiny_ppm(path: Path, *, tint: int) -> None:
    pixels = bytes((tint, 30, 90, tint, 40, 100, tint, 50, 110, tint, 60, 120))
    path.write_bytes(b"P6\n2 2\n255\n" + pixels)


if __name__ == "__main__":
    unittest.main()
