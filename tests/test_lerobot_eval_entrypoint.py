from __future__ import annotations

import tempfile
import json
from pathlib import Path
from unittest import TestCase

from physical_ai_agent.evaluation.agentic_layers import build_agentic_layer, write_debug_artifacts
from physical_ai_agent.evaluation.lerobot_eval import build_config, build_parser


class LeRobotEvalEntrypointTest(TestCase):
    def parse(self, *args: str):
        return build_parser().parse_args([*args])

    def test_libero_command_keeps_smolvla_camera_mapping(self) -> None:
        config = build_config(
            self.parse(
                "--benchmark",
                "libero",
                "--output-dir",
                "/tmp/libero/eval_logs",
                "--task-ids",
                "[0,1]",
                "--max-parallel-tasks",
                "1",
                "--extra-args",
                "--policy.num_steps=10 --policy.n_action_steps=1 --policy.device=cuda",
            )
        )

        argv = config.build_argv()
        command = config.build_shell_command()

        self.assertIn("--env.type=libero", argv)
        self.assertIn("--policy.path=lerobot/smolvla_libero", argv)
        self.assertIn("--env.max_parallel_tasks=1", argv)
        self.assertIn(
            '--env.camera_name_mapping={"agentview_image": "camera1", '
            '"robot0_eye_in_hand_image": "camera2"}',
            argv,
        )
        self.assertIn("--policy.num_steps=10", argv)
        self.assertIn("'--env.camera_name_mapping={", command)

    def test_metaworld_command_keeps_rename_map_and_action_steps(self) -> None:
        config = build_config(
            self.parse(
                "--benchmark",
                "metaworld",
                "--output-dir",
                "/tmp/metaworld/eval_logs",
                "--n-action-steps",
                "15",
            )
        )

        argv = config.build_argv()

        self.assertIn("--env.type=metaworld", argv)
        self.assertIn("--policy.path=lerobot/smolvla_metaworld", argv)
        self.assertIn('--rename_map={"observation.image":"observation.images.camera1"}', argv)
        self.assertIn("--policy.empty_cameras=0", argv)
        self.assertIn("--policy.device=cuda", argv)
        self.assertIn("--policy.use_amp=false", argv)
        self.assertIn("--policy.n_action_steps=15", argv)
        self.assertIn("--seed=0", argv)

    def test_command_script_uses_egl_and_exec(self) -> None:
        config = build_config(
            self.parse(
                "--benchmark",
                "metaworld",
                "--output-dir",
                "/tmp/metaworld/eval_logs",
            )
        )

        script = config.build_shell_script()

        self.assertTrue(script.startswith("#!/bin/sh\nset -eu\n\n"))
        self.assertIn('MUJOCO_GL="${MUJOCO_GL:-egl}" exec lerobot-eval', script)

    def test_agentic_layer_registry_separates_baseline_and_retry(self) -> None:
        baseline = build_agentic_layer("baseline")
        retry = build_agentic_layer("episode_retry", retry_budget=2)

        self.assertEqual(baseline.name, "baseline")
        self.assertEqual(retry.name, "episode_retry")
        self.assertEqual(retry.debug_spec("libero").retry_budget, 2)
        self.assertTrue(retry.debug_spec("libero").runnable)
        self.assertFalse(retry.debug_spec("metaworld").runnable)

    def test_debug_artifacts_capture_eval_config_and_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            config = build_config(
                self.parse(
                    "--benchmark",
                    "libero",
                    "--output-dir",
                    str(output_root / "eval_logs"),
                )
            )
            layer = build_agentic_layer("baseline")
            paths = write_debug_artifacts(
                output_root=output_root,
                config=config,
                layer=layer,
                command=config.build_shell_command(),
            )

            manifest = json.loads(Path(paths["eval_manifest"]).read_text(encoding="utf-8"))
            argv = json.loads(Path(paths["command_argv"]).read_text(encoding="utf-8"))
            layer_payload = json.loads(Path(paths["agentic_layer"]).read_text(encoding="utf-8"))
            events = Path(paths["events"]).read_text(encoding="utf-8")

            self.assertEqual(manifest["benchmark"], "libero")
            self.assertEqual(manifest["agentic_layer"], "baseline")
            self.assertIn("--env.type=libero", argv)
            self.assertEqual(layer_payload["name"], "baseline")
            self.assertIn("eval_command_prepared", events)

    def test_write_command_creates_executable_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_command.sh"
            config = build_config(
                self.parse(
                    "--benchmark",
                    "libero",
                    "--output-dir",
                    "/tmp/libero/eval_logs",
                )
            )

            path.write_text(config.build_shell_script(), encoding="utf-8")
            path.chmod(0o755)

            self.assertTrue(path.read_text(encoding="utf-8").startswith("#!/bin/sh"))
            self.assertTrue(path.stat().st_mode & 0o111)

    def test_runpod_runner_has_strict_environment_defaults(self) -> None:
        runner = Path("scripts/eval_smolvla_lerobot_linux.sh").read_text(encoding="utf-8")

        self.assertIn('PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"', runner)
        self.assertIn('REQUIRE_CUDA="${REQUIRE_CUDA:-1}"', runner)
        self.assertIn("runpod_preflight.txt", runner)
        self.assertIn("refusing CPU fallback", runner)
        self.assertIn("Python >=3.12 is required", runner)
