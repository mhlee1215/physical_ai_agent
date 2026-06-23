from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from physical_ai_agent.agent_core.qwen_so101_closed_loop import (
    LoopArtifactConfig,
    _apply_start_contract_to_env,
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
            seen_camera_batches = []
            with (
                patch(
                    "physical_ai_agent.agent_core.qwen_so101_closed_loop._make_renderers_or_none",
                    return_value={"egocentric_cam": object(), "wrist_cam": object()},
                ),
                patch(
                    "physical_ai_agent.agent_core.qwen_so101_closed_loop._render_policy_cameras",
                    return_value={"egocentric_cam": "ego_pixels", "wrist_cam": "wrist_pixels"},
                ),
            ):
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
                    valid_mask_head=FakeValidMaskHead(),
                    artifact_config=LoopArtifactConfig(enabled=True, render_media=False),
                    env_config={"object_shape": "cube", "object_color": "green", "n_distractors": 0},
                    start_contract="pick_up_reset",
                    env_factory=FakeEnv,
                    policy_loader=fake_policy_loader,
                    batch_builder=lambda *args, **kwargs: fake_batch_builder(
                        *args,
                        seen_camera_batches=seen_camera_batches,
                        **kwargs,
                    ),
                )
            trace_rows = _read_jsonl(Path(report["episodes"][0]["trace_path"]))
            primitive_ids = [
                row["primitive_id"]
                for row in trace_rows
            ]

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["episodes_completed"], 1)
        self.assertEqual(report["episodes"][0]["steps"], 3)
        self.assertIsNone(report["episodes"][0]["media_root"])
        self.assertEqual(report["loop_artifact_config"]["enabled"], True)
        self.assertEqual(report["loop_artifact_config"]["render_media"], False)
        self.assertEqual(report["env_config"]["object_color"], "green")
        self.assertEqual(report["start_contract"], "pick_up_reset")
        self.assertEqual(report["episodes"][0]["start_contract"], "pick_up_reset")
        self.assertEqual(report["episodes"][0]["start_contract_state"]["mode"], "unsupported_env")
        self.assertEqual(trace_rows[0]["render_replay"]["env_config"]["object_color"], "green")
        self.assertEqual(trace_rows[0]["render_replay"]["start_contract"], "pick_up_reset")
        self.assertEqual(trace_rows[0]["policy_input_camera_names"], ["egocentric_cam", "wrist_cam"])
        self.assertEqual(trace_rows[0]["media"]["render_mode"], "deferred")
        self.assertEqual(trace_rows[0]["media"]["policy_input_images"], {})
        self.assertIsNone(trace_rows[0]["media"]["robot_frame"])
        self.assertTrue(seen_camera_batches)
        self.assertTrue(all("egocentric_cam" in batch for batch in seen_camera_batches))
        self.assertTrue(all("wrist_cam" in batch for batch in seen_camera_batches))
        self.assertEqual(report["success_rate"], 0.0)
        self.assertEqual(report["valid_mask"]["required_for_loop_test"], True)
        self.assertEqual(trace_rows[0]["valid_mask"]["budget"], 1)
        self.assertEqual(trace_rows[0]["valid_mask"]["reason"], "valid_mask_stop")
        self.assertEqual(report["policy_rollout_config"]["chunk_size"], 50)
        self.assertEqual(report["policy_rollout_config"]["n_action_steps"], 15)
        self.assertEqual(report["policy_rollout_config"]["num_steps"], 10)
        self.assertEqual(
            primitive_ids,
            [
                "move_over_cube_edge",
                "align_fixed_jaw_cube_edge",
                "grip_from_edge_cube",
            ],
        )

    def test_closed_loop_blocks_when_policy_cameras_are_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "physical_ai_agent.agent_core.qwen_so101_closed_loop._make_renderers_or_none",
                    return_value={},
                ),
                patch(
                    "physical_ai_agent.agent_core.qwen_so101_closed_loop._render_policy_cameras",
                    return_value={},
                ),
            ):
                report = run_closed_loop_plan(
                    plan=_plan(),
                    output_dir=Path(tmpdir),
                    default_policy_path="policy",
                    episodes=1,
                    seed=7,
                    device="cpu",
                    local_files_only=True,
                    max_steps_per_primitive=2,
                    valid_mask_head=FakeValidMaskHead(),
                    artifact_config=LoopArtifactConfig(enabled=True, render_media=False),
                    env_factory=FakeEnv,
                    policy_loader=fake_policy_loader,
                    batch_builder=fake_batch_builder,
                )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("policy camera render failed", report["blocker"])

    def test_precondition_primitives_run_before_recorded_plan(self) -> None:
        with TemporaryDirectory() as tmpdir:
            seen_instructions = []
            precondition = SO101ToolPlan(
                task="move then align",
                model="qwen3-vl-8b-instruct-mlx",
                thinking_mode="non-thinking",
                calls=[
                    SO101PrimitiveCall(0, "move", "green cube", "move_over_cube_edge", "move prompt", 90),
                    SO101PrimitiveCall(1, "align", "green cube", "align_fixed_jaw_cube_edge", "align prompt", 75),
                ],
            )
            with (
                patch(
                    "physical_ai_agent.agent_core.qwen_so101_closed_loop._make_renderers_or_none",
                    return_value={"egocentric_cam": object(), "wrist_cam": object()},
                ),
                patch(
                    "physical_ai_agent.agent_core.qwen_so101_closed_loop._render_policy_cameras",
                    return_value={"egocentric_cam": "ego_pixels", "wrist_cam": "wrist_pixels"},
                ),
            ):
                report = run_closed_loop_plan(
                    plan=SO101ToolPlan(
                        task="lift the green cube",
                        model="qwen3-vl-8b-instruct-mlx",
                        thinking_mode="non-thinking",
                        calls=[
                            SO101PrimitiveCall(
                                0,
                                "pick_up",
                                "green cube",
                                "grip_from_edge_cube",
                                "grip prompt",
                                90,
                            )
                        ],
                    ),
                    output_dir=Path(tmpdir),
                    default_policy_path="policy",
                    episodes=1,
                    seed=7,
                    device="cpu",
                    local_files_only=True,
                    max_steps_per_primitive=2,
                    valid_mask_head=FakeValidMaskHead(),
                    artifact_config=LoopArtifactConfig(enabled=True, render_media=False),
                    start_contract="pick_up_reset",
                    precondition_plan=precondition,
                    env_factory=FakeEnv,
                    policy_loader=fake_policy_loader,
                    batch_builder=lambda *args, **kwargs: fake_batch_builder_record_instruction(
                        *args,
                        seen_instructions=seen_instructions,
                        **kwargs,
                    ),
                )
            trace_rows = _read_jsonl(Path(report["episodes"][0]["trace_path"]))

        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            [row["primitive_id"] for row in trace_rows],
            ["grip_from_edge_cube"],
        )
        self.assertEqual(
            [row["primitive_id"] for row in report["episodes"][0]["precondition_summaries"]],
            ["move_over_cube_edge", "align_fixed_jaw_cube_edge"],
        )
        self.assertLess(seen_instructions.index("move prompt"), seen_instructions.index("align prompt"))
        self.assertLess(seen_instructions.index("align prompt"), seen_instructions.index("grip prompt"))
        self.assertEqual(report["precondition_plan"]["task"], "move then align")

    def test_pick_up_start_contract_sets_teacher_ik_qpos(self) -> None:
        class ActionSpace:
            low = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        class GymEnv:
            action_space = ActionSpace()

            def __init__(self) -> None:
                self.unwrapped = self
                self.qpos = None

        class Wrapper:
            def __init__(self) -> None:
                self.env = GymEnv()

        wrapper = Wrapper()

        def set_qpos(env, qpos):
            env.qpos = [float(value) for value in qpos]

        helpers = {
            "_make_fast_fixed_jaw_teacher_targets": lambda env: [
                {"q_open": [0.2, 0.3, 0.4, 0.5, 0.6, 1.0], "meta": {"score": 1.0}}
            ],
            "_make_fixed_jaw_edge_qpos": lambda env, q_open, meta: [0.2, 0.3, 0.4, 0.5, 0.6, 0.9],
            "_make_fixed_jaw_above_qpos": lambda env, q_edge, meta, move_target_z_offset: [
                0.2,
                0.3,
                0.7,
                0.5,
                0.6,
                0.9,
            ],
            "_open_gripper_value": lambda env: 1.0,
            "_set_qpos": set_qpos,
            "_static_finger_edge_error": lambda env, meta: {"xy_error": 0.0},
            "_tcp_to_object_delta": lambda env: [0.0, 0.0, 0.01],
            "_current_qpos": lambda env: env.qpos,
        }

        with patch(
            "physical_ai_agent.agent_core.qwen_so101_closed_loop._fixed_jaw_export_helpers",
            return_value=helpers,
        ):
            state = _apply_start_contract_to_env(
                env=wrapper,
                start_contract="pick_up_reset",
                seed=98300,
            )

        self.assertTrue(state["applied"])
        self.assertEqual(state["mode"], "deterministic_teacher_ik_qpos")
        self.assertEqual(state["phase"], "edge_contact_open_gripper")
        self.assertEqual(wrapper.env.qpos[-1], 1.0)
        self.assertEqual(state["observation"], wrapper.env.qpos)

    def test_start_contracts_prefer_their_matching_exported_validation_q_start(self) -> None:
        class ActionSpace:
            low = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            high = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]

        class GymEnv:
            action_space = ActionSpace()

            def __init__(self) -> None:
                self.unwrapped = self
                self.qpos = None
                self.reset_seed = None

            def reset(self, *, seed=None):
                self.reset_seed = seed
                return [0.0]

        class Wrapper:
            def __init__(self) -> None:
                self.env = GymEnv()

        def set_qpos(env, qpos):
            env.qpos = [float(value) for value in qpos]

        helpers = {
            "_set_qpos": set_qpos,
            "_current_qpos": lambda env: env.qpos,
        }
        cases = [
            ("full_chain_reset", "move_over_cube_edge", "move_over_cube_edge_q_start", 98100),
            ("align_pick_reset", "align_fixed_jaw_cube_edge", "align_fixed_jaw_q_start", 98200),
            ("pick_up_reset", "grip_from_edge_cube", "grip_from_edge_q_start", 98300),
        ]
        for contract, skill, phase, seed in cases:
            with self.subTest(contract=contract, skill=skill):
                report_path = Path(
                    "_workspace/hf_datasets/mhlee1215__so101-nexus-sim-dataset/"
                    f"datasets/{skill}/validation/so101_lerobot_export_report.json"
                )
                if not report_path.exists():
                    self.skipTest(f"local SO101 {skill} validation report is not available")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                green = [episode for episode in report["episodes"] if episode.get("object_color") == "green"]
                candidates = green or report["episodes"]
                expected = candidates[seed % len(candidates)]
                wrapper = Wrapper()

                with patch(
                    "physical_ai_agent.agent_core.qwen_so101_closed_loop._fixed_jaw_export_helpers",
                    return_value=helpers,
                ):
                    state = _apply_start_contract_to_env(
                        env=wrapper,
                        start_contract=contract,
                        seed=seed,
                    )

                self.assertEqual(state["mode"], "exported_dataset_qpos")
                self.assertEqual(state["source"], "exported_validation_dataset_q_start")
                self.assertEqual(state["phase"], phase)
                self.assertEqual(state["dataset_skill"], skill)
                self.assertEqual(state["dataset_split"], "validation")
                self.assertEqual(state["dataset_object_color"], "green")
                self.assertEqual(wrapper.env.reset_seed, expected["seed"])
                self.assertEqual(
                    wrapper.env.qpos,
                    [float(value) for value in expected["q_start"]],
                )

    def test_start_contracts_do_not_fallback_to_wrong_skill_when_export_exists(self) -> None:
        class ActionSpace:
            low = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            high = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]

        class GymEnv:
            action_space = ActionSpace()

            def __init__(self) -> None:
                self.unwrapped = self
                self.qpos = None
                self.reset_seed = None

            def reset(self, *, seed=None):
                self.reset_seed = seed
                return [0.0]

        class Wrapper:
            def __init__(self) -> None:
                self.env = GymEnv()

        def set_qpos(env, qpos):
            env.qpos = [float(value) for value in qpos]

        helpers = {
            "_set_qpos": set_qpos,
            "_current_qpos": lambda env: env.qpos,
        }

        def unexpected_fixed_jaw_start(*args, **kwargs):
            del args, kwargs
            raise AssertionError("exported validation q_start should be used before IK fallback")

        for contract in ("full_chain_reset", "align_pick_reset", "pick_up_reset"):
            with self.subTest(contract=contract):
                wrapper = Wrapper()
                with (
                    patch(
                        "physical_ai_agent.agent_core.qwen_so101_closed_loop._fixed_jaw_export_helpers",
                        return_value=helpers,
                    ),
                    patch(
                        "physical_ai_agent.agent_core.qwen_so101_closed_loop._fixed_jaw_edge_start_qpos",
                        side_effect=unexpected_fixed_jaw_start,
                    ),
                ):
                    state = _apply_start_contract_to_env(
                        env=wrapper,
                        start_contract=contract,
                        seed=98100,
                    )
                self.assertEqual(state["mode"], "exported_dataset_qpos")
                self.assertTrue(state["applied"])


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

    def predict_action_chunk(self, batch):
        del batch
        return [[[0.0] * 6 for _index in range(50)]]


class FakeValidMaskHead:
    def predict_valid_probs(self, state, action_chunk):
        del state, action_chunk
        return [[1.0, 0.1, 0.1, *([0.0] * 47)]]


def fake_policy_loader(policy_path: str, local_files_only: bool, device: str) -> FakePolicy:
    del local_files_only, device
    return FakePolicy(policy_path)


def fake_batch_builder(
    policy,
    observation,
    camera_pixels=None,
    instruction=None,
    local_files_only=True,
    *,
    seen_camera_batches=None,
):
    del policy, observation, instruction, local_files_only
    if seen_camera_batches is not None:
        seen_camera_batches.append(dict(camera_pixels or {}))
    return {}, {}


def fake_batch_builder_record_instruction(
    policy,
    observation,
    camera_pixels=None,
    instruction=None,
    local_files_only=True,
    *,
    seen_instructions=None,
):
    if seen_instructions is not None:
        seen_instructions.append(instruction)
    return fake_batch_builder(
        policy,
        observation,
        camera_pixels=camera_pixels,
        instruction=instruction,
        local_files_only=local_files_only,
    )


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
