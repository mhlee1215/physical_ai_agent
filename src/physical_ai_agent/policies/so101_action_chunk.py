from __future__ import annotations

from dataclasses import dataclass

from physical_ai_agent.policies.base import ActionChunk


@dataclass(frozen=True)
class SO101ActionChunkConfig:
    action_dim: int
    chunk_size: int = 8
    low: list[float] | None = None
    high: list[float] | None = None


class SO101CenterActionChunkPolicy:
    name = "so101_center_action_chunk"

    def __init__(self, config: SO101ActionChunkConfig) -> None:
        self.config = config

    def action_chunk(self, _observation: object, instruction: str) -> ActionChunk:
        if self.config.low is not None and self.config.high is not None:
            action = [
                (float(low) + float(high)) / 2.0
                for low, high in zip(self.config.low, self.config.high, strict=True)
            ]
        else:
            action = [0.0 for _ in range(self.config.action_dim)]
        return ActionChunk(
            actions=[action[:] for _ in range(self.config.chunk_size)],
            policy_name=self.name,
            metadata={"instruction": instruction, "action_dim": self.config.action_dim},
        )

