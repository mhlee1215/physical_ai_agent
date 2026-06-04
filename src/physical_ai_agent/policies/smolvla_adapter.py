from __future__ import annotations

import importlib
from dataclasses import dataclass

from physical_ai_agent.policies.base import ActionChunk


DEFAULT_SMOLVLA_MODEL_ID = "lerobot/smolvla_base"


@dataclass(frozen=True)
class SmolVLAProbe:
    model_id: str
    ready: bool
    blockers: list[str]
    imports: dict[str, bool]
    policy_class_importable: bool
    notes: list[str]


def probe_smolvla(model_id: str = DEFAULT_SMOLVLA_MODEL_ID) -> SmolVLAProbe:
    imports = {
        "torch": _can_import("torch"),
        "lerobot": _can_import("lerobot"),
        "transformers": _can_import("transformers"),
        "huggingface_hub": _can_import("huggingface_hub"),
    }
    policy_class_importable = _has_smolvla_policy_class()
    blockers = [
        f"missing Python package: {name}"
        for name, ok in imports.items()
        if not ok
    ]
    if not policy_class_importable:
        blockers.append("could not import lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy")
    notes = [
        "SmolVLA is loaded through LeRobot's SmolVLAPolicy.",
        "The base checkpoint usually needs task-specific fine-tuning for useful behavior.",
        "Ready means the LeRobot SmolVLA import path exists, not that model weights were downloaded.",
        "This probe does not download model weights unless a later real-inference gate is added.",
    ]
    return SmolVLAProbe(
        model_id=model_id,
        ready=not blockers,
        blockers=blockers,
        imports=imports,
        policy_class_importable=policy_class_importable,
        notes=notes,
    )


class SmolVLAPolicyAdapter:
    name = "smolvla"

    def __init__(self, model_id: str = DEFAULT_SMOLVLA_MODEL_ID) -> None:
        self.model_id = model_id
        self.probe = probe_smolvla(model_id)

    @property
    def ready(self) -> bool:
        return self.probe.ready

    def action_chunk(self, _observation: object, instruction: str) -> ActionChunk:
        if not self.ready:
            blockers = "; ".join(self.probe.blockers)
            raise RuntimeError(f"SmolVLA adapter is not ready: {blockers}")

        raise NotImplementedError(
            "Real SmolVLA inference requires loading LeRobot's SmolVLAPolicy with "
            f"model_id={self.model_id!r} and mapping observations/actions to the target robot."
        )

    def blocker_markdown(self) -> str:
        blockers = self.probe.blockers or ["None"]
        return "\n".join(
            [
                "# SmolVLA Adapter Blocker",
                "",
                f"- Model id: `{self.model_id}`",
                f"- Ready: `{self.ready}`",
                "",
                "## Blockers",
                "",
                *[f"- {blocker}" for blocker in blockers],
                "",
            ]
        )


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return False
    return True


def _has_smolvla_policy_class() -> bool:
    try:
        module = importlib.import_module("lerobot.policies.smolvla.modeling_smolvla")
    except Exception:  # noqa: BLE001
        return False
    return hasattr(module, "SmolVLAPolicy")
