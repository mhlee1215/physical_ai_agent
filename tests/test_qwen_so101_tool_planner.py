from __future__ import annotations

from unittest import TestCase

from physical_ai_agent.agent_core.qwen_so101_tool_planner import (
    QwenSO101ToolPlanner,
    plan_to_dict,
)


class FakeToolClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[dict] = []

    def create_tool_plan(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def _tool_call(name: str, obj: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": f'{{"object": "{obj}"}}',
        },
    }


class QwenSO101ToolPlannerTest(TestCase):
    def test_tool_calls_build_edge_chain_primitive_prompts(self) -> None:
        client = FakeToolClient(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                _tool_call("move", "green cube"),
                                _tool_call("align", "green cube"),
                                _tool_call("pick_up", "green cube"),
                            ]
                        }
                    }
                ]
            }
        )

        plan = QwenSO101ToolPlanner(client=client, model="Qwen/Qwen3-8B").plan(
            task="pick and lift the green cube",
            target_object="green cube",
        )

        self.assertEqual([call.fn for call in plan.calls], ["move", "align", "pick_up"])
        self.assertEqual(
            [call.primitive_id for call in plan.calls],
            ["move_over_cube_edge", "align_fixed_jaw_cube_edge", "grip_from_edge_cube"],
        )
        self.assertIn("green cube edge", plan.calls[0].prompt)
        self.assertEqual(plan.thinking_mode, "non-thinking")
        self.assertEqual(client.requests[0]["temperature"], 0.0)
        self.assertEqual(client.requests[0]["tool_choice"], "auto")
        system_prompt = client.requests[0]["messages"][0]["content"]
        self.assertIn("/no_think", system_prompt)

    def test_json_plan_fallback_is_validated(self) -> None:
        client = FakeToolClient(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"plan": ['
                                '{"fn": "move", "args": {"object": "green cube"}},'
                                '{"fn": "align", "args": {"object": "green cube"}},'
                                '{"fn": "pick_up", "args": {"object": "green cube"}}'
                                "]}"
                            )
                        }
                    }
                ]
            }
        )

        plan = QwenSO101ToolPlanner(client=client).plan()

        self.assertEqual(plan_to_dict(plan)["calls"][2]["primitive_id"], "grip_from_edge_cube")

    def test_wrong_tool_order_is_rejected(self) -> None:
        client = FakeToolClient(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                _tool_call("align", "green cube"),
                                _tool_call("move", "green cube"),
                                _tool_call("pick_up", "green cube"),
                            ]
                        }
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "expected the narrow SO101 order"):
            QwenSO101ToolPlanner(client=client).plan()

    def test_mixed_target_objects_are_rejected(self) -> None:
        client = FakeToolClient(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                _tool_call("move", "green cube"),
                                _tool_call("align", "red cube"),
                                _tool_call("pick_up", "green cube"),
                            ]
                        }
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "one target object"):
            QwenSO101ToolPlanner(client=client).plan()
