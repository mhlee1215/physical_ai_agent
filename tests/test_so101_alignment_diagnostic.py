from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_so101_alignment_diagnostic import (  # noqa: E402
    alignment_frame_window,
    alignment_success,
)


class So101AlignmentDiagnosticTest(unittest.TestCase):
    def test_v3_alignment_window_starts_after_home_approach_and_ends_before_close(self) -> None:
        start, end = alignment_frame_window(
            {
                "open_from_hardware_start": 10,
                "move_to_cube": 34,
                "roll_align_with_cube_edge": 17,
                "gripper_descend": 17,
                "settle_aligned": 10,
                "close": 42,
                "lift": 35,
                "terminal_hold": 12,
            },
            phases_before=["open_from_hardware_start", "move_to_cube"],
            phases=[
                "roll_align_with_cube_edge",
                "gripper_descend",
                "settle_aligned",
            ],
        )

        self.assertEqual(start, 44)
        self.assertEqual(end, 88)

    def test_alignment_success_requires_both_position_and_angle(self) -> None:
        contract = {
            "static_edge_xy_error_max_m": 0.012,
            "jaw_face_parallel_error_max_deg": 3.0,
        }

        self.assertTrue(
            alignment_success(
                static_edge_xy_error_m=0.011,
                jaw_face_parallel_error_deg=2.9,
                success_config=contract,
            )
        )
        self.assertFalse(
            alignment_success(
                static_edge_xy_error_m=0.013,
                jaw_face_parallel_error_deg=2.9,
                success_config=contract,
            )
        )
        self.assertFalse(
            alignment_success(
                static_edge_xy_error_m=0.011,
                jaw_face_parallel_error_deg=3.1,
                success_config=contract,
            )
        )

    def test_alignment_window_rejects_missing_phase(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing phases"):
            alignment_frame_window(
                {"move_to_cube": 34},
                phases_before=["move_to_cube"],
                phases=["roll_align_with_cube_edge"],
            )


if __name__ == "__main__":
    unittest.main()
