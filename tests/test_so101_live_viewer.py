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
                "--policy",
                "visual-rl",
                "--allow-download",
                "--smolvla-action-steps",
                "15",
                "--smolvla-worker-python",
                ".venv/bin/python",
                "--visual-policy-checkpoint",
                "_workspace/test_policy.pt",
                "--visual-policy-camera",
                "egocentric_cam",
                "--visual-reach-checkpoint",
                "_workspace/test_delta.pt",
                "--visual-reach-camera",
                "top_down",
                "--browser-only",
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
        self.assertEqual(args.policy, "visual-rl")
        self.assertTrue(args.allow_download)
        self.assertEqual(args.smolvla_action_steps, 15)
        self.assertEqual(args.smolvla_worker_python, ".venv/bin/python")
        self.assertEqual(args.visual_policy_checkpoint, "_workspace/test_policy.pt")
        self.assertEqual(args.visual_policy_camera, "egocentric_cam")
        self.assertEqual(args.visual_reach_checkpoint, "_workspace/test_delta.pt")
        self.assertEqual(args.visual_reach_camera, "top_down")
        self.assertTrue(args.browser_only)
        self.assertTrue(args.show_inputs)
        self.assertEqual(args.input_width, 160)
        self.assertEqual(args.input_height, 120)
        self.assertEqual(args.input_port, 8877)

    def test_live_viewer_config_defaults_to_so101_reach(self) -> None:
        config = LiveViewerConfig()

        self.assertEqual(config.env_id, "MuJoCoReach-v1")
        self.assertEqual(config.fps, 30.0)
        self.assertEqual(config.policy, "sample")
        self.assertFalse(config.allow_download)
        self.assertEqual(config.smolvla_action_steps, 15)
        self.assertEqual(config.smolvla_worker_python, "")
        self.assertEqual(
            config.visual_policy_checkpoint,
            "_workspace/so101_visual_rl/train/so101_visual_rl_policy.pt",
        )
        self.assertEqual(config.visual_policy_camera, "wrist_cam")
        self.assertEqual(
            config.visual_reach_checkpoint,
            "_workspace/so101_visual_rl/reach_delta/so101_visual_reach_delta.pt",
        )
        self.assertEqual(config.visual_reach_camera, "top_down")
        self.assertFalse(config.browser_only)
        self.assertFalse(config.show_inputs)
        self.assertEqual(config.input_width, 320)
        self.assertEqual(config.input_height, 240)
        self.assertEqual(config.input_port, 8765)
