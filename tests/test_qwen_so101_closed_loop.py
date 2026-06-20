from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from physical_ai_agent.agent_core.qwen_so101_closed_loop import (
    LoopArtifactConfig,
    parse_primitive_policy_routes,
    resolve_policy_routes,
    run_closed_loop_plan,
)
from physical_ai_agent.agent_core.qwen_so101_tool_planner import (
    SO101PrimitiveCall,
    SO101ToolPlan,
)


class QwenSO101ClosedLoopTest(unittest.TestCase):
    def test_plan_routes_three_separate_primitive_policies(self) -> None:
        plan = _plan()
        routes = resolve_policy_routes(
            plan,
            default_policy_path=None,
            primitive_policy_paths=parse_primitive_policy_routes(
                [
                    "move_over_cube_edge=/ckpts/move",
                    "align_fixed_jaw_cube_edge=/ckpts/align",
                    "grip_from_edge_cube=/ckpts/grip",
                ]
            ),
        )

        self.assertEqual(
            [route.policy_path for route in routes],
            ["/ckpts/move", "/ckpts/align", "/ckpts/grip"],
        )

    def test_mock_closed_loop_executes_plan_in_one_env(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report = run_closed_loop_plan(
                plan=_plan(),
                output_dir=Path(tmpdir),
                default_policy_path=None,
                primitive_policy_paths={
                    "move_over_cube_edge": "move_policy",
                    "align_fixed_jaw_cube_edge": "align_policy",
                    "grip_from_edge_cube": "grip_policy",
                },
                episodes=1,
                seed=7,
                device="cpu",
                local_files_only=True,
                max_steps_per_primitive=2,
                artifact_config=LoopArtifactConfig(enabled=True, render_media=False),
                env_factory=FakeEnv,
                policy_loader=fake_policy_loader,
                batch_builder=fake_batch_builder,
            )
            trace_rows = _read_jsonl(Path(report["episodes"][0]["trace_path"]))
            primitive_ids = [
                row["primitive_id"]
                for row in trace_rows
            ]

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["episodes_completed"], 1)
        self.assertEqual(report["episodes"][0]["steps"], 6)
        self.assertIsNone(report["episodes"][0]["media_root"])
        self.assertEqual(report["loop_artifact_config"]["enabled"], True)
        self.assertEqual(report["loop_artifact_config"]["render_media"], False)
        self.assertEqual(trace_rows[0]["media"]["render_mode"], "deferred")
        self.assertEqual(trace_rows[0]["media"]["policy_input_images"], {})
        self.assertIsNone(trace_rows[0]["media"]["robot_frame"])
        self.assertEqual(report["success_rate"], 1.0)
        self.assertEqual(report["policy_rollout_config"]["chunk_size"], 50)
        self.assertEqual(report["policy_rollout_config"]["n_action_steps"], 15)
        self.assertEqual(report["policy_rollout_config"]["num_steps"], 10)
        self.assertEqual(
            primitive_ids,
            [
                "move_over_cube_edge",
                "move_over_cube_edge",
                "align_fixed_jaw_cube_edge",
                "align_fixed_jaw_cube_edge",
                "grip_from_edge_cube",
                "grip_from_edge_cube",
            ],
        )


def _plan() -> SO101ToolPlan:
    return SO101ToolPlan(
        task="pick and lift the green cube",
        model="qwen3-vl-8b-instruct-mlx",
        thinking_mode="non-thinking",
        calls=[
            SO101PrimitiveCall(0, "move", "green cube", "move_over_cube_edge", "move prompt", 90),
            SO101PrimitiveCall(1, "align", "green cube", "align_fixed_jaw_cube_edge", "align prompt", 75),
            SO101PrimitiveCall(2, "pick_up", "green cube", "grip_from_edge_cube", "grip prompt", 90),
        ],
    )


class FakeConfig:
    device = "cpu"
    image_features = {}
    robot_state_feature = None
    chunk_size = 50
    n_action_steps = 50
    num_steps = 50


class FakePolicy:
    def __init__(self, policy_path: str) -> None:
        self.policy_path = policy_path
        self.config = FakeConfig()
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, batch):
        del batch
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def fake_policy_loader(policy_path: str, local_files_only: bool, device: str) -> FakePolicy:
    del local_files_only, device
    return FakePolicy(policy_path)


def fake_batch_builder(policy, observation, camera_pixels=None, instruction=None, local_files_only=True):
    del policy, observation, camera_pixels, instruction, local_files_only
    return {}, {}


class FakeEnv:
    def __init__(self, env_id: str, render_mode: str | None = None) -> None:
        self.env_id = env_id
        self.render_mode = render_mode
        self.action_dim = 6
        self.step_count = 0
        self.closed = False

    def reset(self, seed: int):
        self.step_count = 0
        return [float(seed), 0.0, 0.0], {"seed": seed}

    def step(self, action):
        self.step_count += 1
        obs = [float(self.step_count), *[float(item) for item in action[:2]]]
        info = {"success": self.step_count >= 6}
        return obs, 1.0, False, False, info

    def close(self) -> None:
        self.closed = True


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
