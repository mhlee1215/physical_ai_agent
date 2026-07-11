from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.convert_mycobot_280_pi_adaptive_jsonl_to_lerobot import (
    convert_mycobot_280_pi_adaptive_jsonl_to_lerobot,
)
from scripts.export_mycobot_280_pi_adaptive_lerobot_dataset import (
    JOINT_NAMES,
    export_mycobot_280_pi_adaptive_lerobot_dataset,
)


class MyCobot280PiLeRobotNativeConverterTest(unittest.TestCase):
    def test_missing_lerobot_writes_blocked_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = _write_jsonl_export(tmp_path)
            output_root = tmp_path / "native"

            real_import = __import__

            def fake_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "lerobot.datasets.lerobot_dataset":
                    raise ImportError("no lerobot here")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                report = convert_mycobot_280_pi_adaptive_jsonl_to_lerobot(
                    source_root=source_root,
                    output_root=output_root,
                    repo_id="physical-ai-agent/mycobot-280pi-adaptive-test",
                    fps=None,
                    use_videos=False,
                    overwrite=False,
                )

            self.assertEqual(report["status"], "blocked")
            self.assertIn("lerobot import failed", report["blocker"])
            self.assertEqual(report["features"]["observation.state"]["shape"], (7,))
            report_path = output_root / "mycobot_280_pi_lerobot_convert_report.json"
            self.assertTrue(report_path.exists())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "blocked")

    def test_fake_lerobot_dataset_receives_native_frames_and_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_root = _write_jsonl_export(tmp_path)
            output_root = tmp_path / "native"

            FakeLeRobotDataset.instances.clear()
            report = convert_mycobot_280_pi_adaptive_jsonl_to_lerobot(
                source_root=source_root,
                output_root=output_root,
                repo_id="physical-ai-agent/mycobot-280pi-adaptive-test",
                fps=None,
                use_videos=False,
                overwrite=False,
                lerobot_dataset_cls=FakeLeRobotDataset,
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["exported_episodes"], 1)
            self.assertEqual(report["exported_frames"], 3)
            self.assertEqual(report["robot_type"], "mycobot_280_pi_adaptive_gripper")
            instance = FakeLeRobotDataset.instances[-1]
            self.assertEqual(instance.repo_id, "physical-ai-agent/mycobot-280pi-adaptive-test")
            self.assertEqual(instance.fps, 12)
            self.assertEqual(instance.robot_type, "mycobot_280_pi_adaptive_gripper")
            self.assertEqual(instance.features["action"]["names"], JOINT_NAMES)
            self.assertEqual(len(instance.frames), 3)
            self.assertEqual(instance.save_episode_calls, 1)
            first = instance.frames[0]
            self.assertEqual(_shape_of(first["observation.images.camera1"]), (2, 2, 3))
            self.assertEqual(_shape_of(first["observation.images.camera2"]), (2, 2, 3))
            self.assertEqual(_shape_of(first["observation.state"]), (7,))
            self.assertEqual(_shape_of(first["action"]), (7,))
            self.assertEqual(_shape_of(first["object_position"]), (3,))
            self.assertEqual(_shape_of(first["contact_count"]), (1,))
            self.assertEqual(first["task"], "Pick up the test cube.")
            self.assertTrue(instance.finalized)
            self.assertTrue((output_root / "mycobot_280_pi_lerobot_convert_report.json").exists())


class FakeLeRobotDataset:
    instances: list["FakeLeRobotDataset"] = []

    def __init__(
        self,
        *,
        repo_id: str,
        fps: int,
        features: dict[str, object],
        root: Path,
        robot_type: str,
        use_videos: bool,
    ) -> None:
        self.repo_id = repo_id
        self.fps = fps
        self.features = features
        self.root = root
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.frames: list[dict[str, object]] = []
        self.save_episode_calls = 0
        self.finalized = False
        self.instances.append(self)

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        fps: int,
        features: dict[str, object],
        root: Path,
        robot_type: str,
        use_videos: bool,
        image_writer_processes: int,
        image_writer_threads: int,
    ) -> "FakeLeRobotDataset":
        assert image_writer_processes == 0
        assert image_writer_threads == 0
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
        )

    def add_frame(self, frame: dict[str, object]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.save_episode_calls += 1

    def finalize(self) -> None:
        self.finalized = True


def _shape_of(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    dims: list[int] = []
    current = value
    while isinstance(current, list):
        dims.append(len(current))
        current = current[0] if current else []
    return tuple(dims)

def _write_jsonl_export(tmp_path: Path) -> Path:
    trace_path = tmp_path / "trace.jsonl"
    manifest_path = tmp_path / "camera_manifest.json"
    camera_root = tmp_path / "camera"
    camera_root.mkdir()
    manifest_frames = []
    trace_records = []
    for index in range(3):
        top = camera_root / f"top_{index}.ppm"
        wrist = camera_root / f"wrist_{index}.ppm"
        _write_tiny_ppm(top, tint=20 + index)
        _write_tiny_ppm(wrist, tint=70 + index)
        manifest_frames.append({"top": str(top), "wrist": str(wrist), "timestamp": index / 12.0})
        trace_records.append(_trace_record(index, z=0.02 + 0.02 * index, contacts=index))
    trace_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in trace_records) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"frames": manifest_frames}), encoding="utf-8")
    source_root = tmp_path / "jsonl_export"
    export_mycobot_280_pi_adaptive_lerobot_dataset(
        root=source_root,
        input_trace=trace_path,
        camera_manifest=manifest_path,
        episode_index=0,
        fps=12,
        repo_id="physical-ai-agent/mycobot-280pi-adaptive-test",
        task="Pick up the test cube.",
        overwrite=False,
    )
    return source_root


def _trace_record(index: int, *, z: float, contacts: int) -> dict[str, object]:
    state = [0.01 * (index + offset) for offset in range(7)]
    return {
        "timestamp": index / 12.0,
        "joint_state": {"name": JOINT_NAMES, "position": state},
        "trajectory_point": {
            "joint_names": JOINT_NAMES,
            "positions": [value + 0.04 for value in state],
        },
        "object_pose": {"position": [0.12, -0.04, z]},
        "contacts": {"left_finger_pad": contacts >= 1, "right_finger_pad": contacts >= 2},
        "task": "Pick up the test cube.",
    }


def _write_tiny_ppm(path: Path, *, tint: int) -> None:
    pixels = bytes((tint, 30, 90, tint, 40, 100, tint, 50, 110, tint, 60, 120))
    path.write_bytes(b"P6\n2 2\n255\n" + pixels)


if __name__ == "__main__":
    unittest.main()
