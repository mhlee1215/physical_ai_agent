import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.sim.so101_interactive_control import (
    SO101_JOINT_ORDER,
    build_parser,
    _gui_html,
    hardware_alignment_contract,
    make_interactive_action,
    run_scripted_session,
    validate_action_values,
)


class SO101InteractiveControlTest(TestCase):
    def test_hardware_alignment_contract_blocks_real_execution(self) -> None:
        contract = hardware_alignment_contract()

        self.assertEqual(contract["joint_order"], SO101_JOINT_ORDER)
        self.assertEqual(contract["real_robot_execution"], "disabled")
        self.assertFalse(contract["send_action_called"])
        self.assertIn(
            "home-return and torque-off report",
            contract["requires_before_real_execution"],
        )

    def test_validate_action_values_checks_bounds_and_dim(self) -> None:
        blockers = validate_action_values([0.0, 2.0], low=[-1.0, -1.0], high=[1.0, 1.0])

        self.assertEqual(len(blockers), 1)
        dim_blockers = validate_action_values([0.0], low=[-1.0, -1.0], high=[1.0, 1.0])
        self.assertIn("Expected 2 action values", dim_blockers[0])
        self.assertIn("shoulder_lift", blockers[0])

    def test_make_action_marks_sim_only(self) -> None:
        action = make_interactive_action([0, 0, 0, 0, 0, 0], source="test")

        self.assertFalse(action.real_robot_safe_to_execute)
        self.assertEqual(action.hardware_aligned_joint_order, SO101_JOINT_ORDER)

    def test_parser_accepts_script_and_command(self) -> None:
        args = build_parser().parse_args(
            [
                "--env-id",
                "MuJoCoReach-v1",
                "--seed",
                "7",
                "--output-dir",
                "_workspace/tmp",
                "--script",
                "commands.txt",
                "--command",
                "observe",
                "--gui",
                "--port",
                "8766",
            ]
        )

        self.assertEqual(args.env_id, "MuJoCoReach-v1")
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.command, ["observe"])
        self.assertEqual(str(args.script), "commands.txt")
        self.assertTrue(args.gui)
        self.assertEqual(args.port, 8766)

    def test_dry_contract_writes_artifact_without_so101_dependency(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(["--dry-contract", "--output-dir", tmpdir])

        self.assertEqual(run_scripted_session(args), 0)
        payload = json.loads((Path(tmpdir) / "hardware_alignment_contract.json").read_text())
        self.assertEqual(payload["real_robot_execution"], "disabled")

    def test_gui_html_includes_scene_image(self) -> None:
        html = _gui_html()

        self.assertIn("sceneCanvas", html)
        self.assertIn("SO101 3D simulation viewer", html)
        self.assertIn("drawScene3D", html)
