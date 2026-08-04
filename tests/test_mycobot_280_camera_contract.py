from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

import scripts.export_mycobot_280_ground_pickup_teacher_dataset as teacher
from scripts.audit_mycobot_280_camera_contract import (
    _sample_indices,
    _target_stats,
)


class MyCobot280CameraContractTest(unittest.TestCase):
    def test_deterministic_export_defaults_preserve_existing_camera_and_bmp(self) -> None:
        args = teacher.build_parser().parse_args([])

        self.assertEqual(args.render_camera_profile, "full_robot")
        self.assertEqual(args.image_format, "bmp")

    def test_closeup_contract_records_fixed_camera_geometry(self) -> None:
        contract = teacher._camera_contract("ground_pickup_closeup", width=256, height=256)

        self.assertEqual(contract["profile"], "ground_pickup_closeup")
        self.assertEqual(contract["resolution_hw"], [256, 256])
        self.assertEqual(contract["mode"], "free_camera")
        self.assertEqual(contract["target"], "initial_cube_xyz_plus_[0,0,0.035]_m")
        self.assertAlmostEqual(contract["distance_m"], 0.24)
        self.assertAlmostEqual(contract["azimuth_deg"], 215.0)
        self.assertAlmostEqual(contract["elevation_deg"], -10.0)

    def test_dependency_free_png_writer_round_trips_exact_rgb_pixels(self) -> None:
        rows, columns = 12, 16
        yy, xx = np.mgrid[:rows, :columns]
        rgb = np.stack(
            (
                (xx * 13 + yy * 3) % 256,
                (xx * 5 + yy * 17) % 256,
                (xx * 7 + yy * 11) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            teacher._write_dataset_image(path, rgb, "png")
            decoded = _read_unfiltered_rgb_png(path)

        np.testing.assert_array_equal(decoded, rgb)

    def test_phase_sampler_includes_first_middle_and_last_frames(self) -> None:
        self.assertEqual(_sample_indices(10, 3), [0, 4, 9])
        self.assertEqual(_sample_indices(2, 3), [0, 1])
        self.assertEqual(_sample_indices(0, 3), [])

    def test_target_stats_detect_readable_red_cube(self) -> None:
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[20:40, 30:55, 0] = 220

        stats = _target_stats(
            rgb,
            min_target_pixels=64,
            min_target_fraction=0.001,
        )

        self.assertEqual(stats["target_pixels"], 500)
        self.assertAlmostEqual(stats["target_fraction"], 0.05)
        self.assertEqual(stats["target_bbox_xywh"], [30, 20, 25, 20])
        self.assertTrue(stats["target_visible"])

    def test_camera_contract_rejects_unknown_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported render camera profile"):
            teacher._camera_contract("unknown", width=256, height=256)


def _read_unfiltered_rgb_png(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("missing PNG signature")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    raw = zlib.decompress(bytes(compressed))
    stride = width * 3
    rows = []
    for start in range(0, len(raw), stride + 1):
        if raw[start] != 0:
            raise AssertionError("test decoder only supports PNG filter 0")
        rows.append(raw[start + 1 : start + stride + 1])
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, 3)
