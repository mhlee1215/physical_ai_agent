from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.sim.so101_visual_rl import (
    SO101VisualRLConfig,
    _image_hwc,
    make_so101_visual_rl_env,
    run_visual_rl_smoke,
)


class SO101VisualRLTest(TestCase):
    def test_image_hwc_converts_channel_first(self) -> None:
        import numpy as np

        image = np.zeros((3, 8, 9), dtype=np.uint8)
        converted = _image_hwc(image, channel_first=True)

        self.assertEqual(converted.shape, (8, 9, 3))

    def test_visual_wrapper_exposes_image_and_state_spaces(self) -> None:
        config = SO101VisualRLConfig(env_id="MuJoCoReach-v1", width=32, height=24)
        try:
            env = make_so101_visual_rl_env(config)
        except RuntimeError as exc:
            self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")
        try:
            obs, _info = env.reset(seed=0)
            self.assertEqual(env.observation_space["image"].shape, (3, 24, 32))
            self.assertEqual(env.observation_space["state"].shape, (6,))
            self.assertEqual(obs["image"].shape, (3, 24, 32))
            self.assertEqual(obs["state"].shape, (6,))
        finally:
            env.close()

    def test_visual_rl_smoke_writes_frames_and_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            try:
                report = run_visual_rl_smoke(
                    output_dir=Path(tmpdir),
                    config=SO101VisualRLConfig(env_id="MuJoCoReach-v1", width=32, height=24),
                    steps=2,
                )
            except RuntimeError as exc:
                self.skipTest(f"SO101-Nexus dependencies are not available: {exc}")

            self.assertEqual(report["steps"], 2)
            self.assertEqual(report["observation_space"]["image"]["shape"], [3, 24, 32])
            self.assertTrue(Path(report["manifest_path"]).exists())
            for record in report["records"]:
                self.assertTrue(Path(record["image_path"]).exists())
