from unittest import TestCase

from physical_ai_agent.sim.so101_live_viewer import LiveViewerConfig, build_parser


class SO101LiveViewerTest(TestCase):
    def test_parser_accepts_headless_smoke_options(self) -> None:
        args = build_parser().parse_args(["--env-id", "MuJoCoReach-v1", "--fps", "12", "--max-steps", "2"])

        self.assertEqual(args.env_id, "MuJoCoReach-v1")
        self.assertEqual(args.fps, 12)
        self.assertEqual(args.max_steps, 2)

    def test_live_viewer_config_defaults_to_so101_reach(self) -> None:
        config = LiveViewerConfig()

        self.assertEqual(config.env_id, "MuJoCoReach-v1")
        self.assertEqual(config.fps, 30.0)
