from __future__ import annotations

from dataclasses import dataclass

from physical_ai_agent.policies.base import ActionChunk
from physical_ai_agent.policies.smolvla_adapter import SmolVLAPolicyAdapter


@dataclass(frozen=True)
class SmolVLADryRunInput:
    state: list[float]
    image_shape: tuple[int, int, int]
    instruction: str
    action_dim: int


class SmolVLADryRunBridge:
    """Validates the SO101->SmolVLA mapping without downloading model weights."""

    def __init__(self, model_id: str = "lerobot/smolvla_base") -> None:
        self.adapter = SmolVLAPolicyAdapter(model_id=model_id)

    def build_input(self, observation: list[float], instruction: str, action_dim: int) -> SmolVLADryRunInput:
        return SmolVLADryRunInput(
            state=[float(value) for value in observation[:32]],
            image_shape=(3, 512, 512),
            instruction=instruction,
            action_dim=action_dim,
        )

    def dry_action_chunk(self, dry_input: SmolVLADryRunInput, chunk_size: int = 8) -> ActionChunk:
        if not self.adapter.ready:
            raise RuntimeError("SmolVLA import path is not ready")
        action = [0.0 for _ in range(dry_input.action_dim)]
        return ActionChunk(
            actions=[action[:] for _ in range(chunk_size)],
            policy_name="smolvla_dry_run",
            metadata={
                "instruction": dry_input.instruction,
                "state_dim": len(dry_input.state),
                "image_shape": dry_input.image_shape,
                "note": "Dry run validates mapping only; no model weights are executed.",
            },
        )

