from unittest import TestCase

from physical_ai_agent.sim.so101_live_viewer import LiveViewerConfig, build_parser


class SO101LiveViewerTest(TestCase):
    def test_parser_accepts_headless_smoke_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--env-id",
                "MuJoCoReach-v1",
                "--fps",
                "12",
                "--max-steps",
                "2",
                "--show-inputs",
                "--input-width",
                "160",
                "--input-height",
                "120",
                "--input-port",
                "8877",
            ]
        )

        self.assertEqual(args.env_id, "MuJoCoReach-v1")
        self.assertEqual(args.fps, 12)
        self.assertEqual(args.max_steps, 2)
        self.assertTrue(args.show_inputs)
        self.assertEqual(args.input_width, 160)
        self.assertEqual(args.input_height, 120)
        self.assertEqual(args.input_port, 8877)

    def test_live_viewer_config_defaults_to_so101_reach(self) -> None:
        config = LiveViewerConfig()

        self.assertEqual(config.env_id, "MuJoCoReach-v1")
        self.assertEqual(config.fps, 30.0)
        self.assertFalse(config.show_inputs)
        self.assertEqual(config.input_width, 320)
        self.assertEqual(config.input_height, 240)
        self.assertEqual(config.input_port, 8765)
