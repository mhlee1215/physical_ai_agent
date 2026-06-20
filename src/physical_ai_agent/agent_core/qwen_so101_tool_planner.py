from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_OBJECT = "green cube"
DEFAULT_TASK = f"pick and lift the {DEFAULT_OBJECT}"


@dataclass(frozen=True)
class SO101ToolSpec:
    name: str
    description: str
    primitive_id: str
    prompt_template: str
    max_steps: int

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object": {
                            "type": "string",
                            "description": "Visible target object, e.g. green cube.",
                        }
                    },
                    "required": ["object"],
                    "additionalProperties": False,
                },
            },
        }


@dataclass(frozen=True)
class SO101PrimitiveCall:
    index: int
    fn: str
    object: str
    primitive_id: str
    prompt: str
    max_steps: int


@dataclass(frozen=True)
class SO101ToolPlan:
    task: str
    model: str
    thinking_mode: str
    calls: list[SO101PrimitiveCall]


class ChatToolClient(Protocol):
    def create_tool_plan(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        temperature: float,
    ) -> dict[str, Any]:
        ...


SO101_EDGE_TOOLS: tuple[SO101ToolSpec, ...] = (
    SO101ToolSpec(
        name="move",
        description=(
            "Move the robot from its current pose until the gripper is above "
            "one visible edge of the target cube. Use this before alignment."
        ),
        primitive_id="move_over_cube_edge",
        prompt_template="Move the gripper above one visible {object} edge.",
        max_steps=90,
    ),
    SO101ToolSpec(
        name="align",
        description=(
            "Refine the pose so the gripper jaws are aligned around one visible edge "
            "of the target cube. Use this after move and before pick_up."
        ),
        primitive_id="align_fixed_jaw_cube_edge",
        prompt_template="Align the gripper jaws around one visible {object} edge.",
        max_steps=75,
    ),
    SO101ToolSpec(
        name="pick_up",
        description=(
            "Keep the gripper at the target cube edge, close the gripper, "
            "grasp the cube, and lift it."
        ),
        primitive_id="grip_from_edge_cube",
        prompt_template="Close the gripper on the {object} edge and lift.",
        max_steps=90,
    ),
)


class OpenAICompatibleQwenClient:
    """Tiny OpenAI-compatible client for vLLM/SGLang/Ollama-like Qwen servers.

    The planner keeps dependencies out of the repo. A RunPod/local vLLM server can
    expose Qwen3-8B at /v1/chat/completions, and this client sends normal tool
    schemas plus Qwen's non-thinking hint through extra_body where supported.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        default_url = "http://127.0.0.1:8000/v1"
        self.base_url = (base_url or os.environ.get("QWEN_OPENAI_BASE_URL") or default_url).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("QWEN_OPENAI_API_KEY") or "EMPTY"
        self.timeout_s = timeout_s

    def create_tool_plan(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        temperature: float,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen OpenAI-compatible endpoint is not reachable: {exc}") from exc


class QwenSO101ToolPlanner:
    def __init__(
        self,
        *,
        client: ChatToolClient | None = None,
        model: str = DEFAULT_MODEL,
        tools: tuple[SO101ToolSpec, ...] = SO101_EDGE_TOOLS,
        expected_order: tuple[str, ...] = ("move", "align", "pick_up"),
    ) -> None:
        self.client = client or OpenAICompatibleQwenClient()
        self.model = model
        self.tools = tools
        self.expected_order = expected_order
        self._tool_by_name = {tool.name: tool for tool in tools}

    def plan(
        self,
        *,
        task: str = DEFAULT_TASK,
        target_object: str = DEFAULT_OBJECT,
    ) -> SO101ToolPlan:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot task planner. Use only the provided tools. "
                    "Plan short SO101 primitive calls for the task. "
                    "Qwen3 non-thinking mode: do not emit hidden reasoning, prose, or markdown. "
                    "Return tool calls only. /no_think"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n"
                    f"Target object: {target_object}\n"
                    "Use the narrow SO101 edge-grasp primitive set in order when appropriate: "
                    "move, align, pick_up."
                ),
            },
        ]
        response = self.client.create_tool_plan(
            model=self.model,
            messages=messages,
            tools=[tool.to_openai_tool() for tool in self.tools],
            tool_choice="auto",
            temperature=0.0,
        )
        calls = self._extract_tool_calls(response=response, fallback_object=target_object)
        self._validate_calls(calls)
        primitive_calls = [
            self._to_primitive_call(index=index, fn=fn, target_object=obj)
            for index, (fn, obj) in enumerate(calls)
        ]
        return SO101ToolPlan(
            task=task,
            model=self.model,
            thinking_mode="non-thinking",
            calls=primitive_calls,
        )

    def _extract_tool_calls(
        self,
        *,
        response: dict[str, Any],
        fallback_object: str,
    ) -> list[tuple[str, str]]:
        message = response.get("choices", [{}])[0].get("message", {})
        raw_tool_calls = message.get("tool_calls") or []
        calls: list[tuple[str, str]] = []
        for raw_call in raw_tool_calls:
            function = raw_call.get("function", {})
            name = str(function.get("name") or "").strip()
            args = _loads_object(function.get("arguments"))
            obj = str(args.get("object") or fallback_object).strip()
            if name:
                calls.append((name, obj))
        if calls:
            return calls
        content = str(message.get("content") or "").strip()
        return self._extract_json_plan(content=content, fallback_object=fallback_object)

    def _extract_json_plan(self, *, content: str, fallback_object: str) -> list[tuple[str, str]]:
        parsed = _loads_object(content)
        raw_plan = parsed.get("plan") if isinstance(parsed, dict) else None
        if not isinstance(raw_plan, list):
            raise ValueError("Qwen planner response had neither tool_calls nor a JSON plan list.")
        calls: list[tuple[str, str]] = []
        for item in raw_plan:
            if not isinstance(item, dict):
                raise ValueError("Qwen planner JSON plan entries must be objects.")
            fn = str(item.get("fn") or "").strip()
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            obj = str(args.get("object") or item.get("object") or fallback_object).strip()
            calls.append((fn, obj))
        return calls

    def _validate_calls(self, calls: list[tuple[str, str]]) -> None:
        names = tuple(name for name, _ in calls)
        if names != self.expected_order:
            raise ValueError(
                "Qwen planner returned "
                f"{names}; expected the narrow SO101 order {self.expected_order}."
            )
        unknown = [name for name in names if name not in self._tool_by_name]
        if unknown:
            raise ValueError(f"Qwen planner returned unsupported tools: {unknown}")
        objects = {obj for _, obj in calls}
        if len(objects) != 1:
            raise ValueError(
                "Qwen planner must keep one target object across the chain; "
                f"got {sorted(objects)}"
            )

    def _to_primitive_call(self, *, index: int, fn: str, target_object: str) -> SO101PrimitiveCall:
        tool = self._tool_by_name[fn]
        return SO101PrimitiveCall(
            index=index,
            fn=fn,
            object=target_object,
            primitive_id=tool.primitive_id,
            prompt=tool.prompt_template.format(object=target_object),
            max_steps=tool.max_steps,
        )


def plan_to_dict(plan: SO101ToolPlan) -> dict[str, Any]:
    return asdict(plan)


def _loads_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"expected JSON object, got: {value[:200]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed
