from __future__ import annotations

import random
from dataclasses import dataclass

from physical_ai_agent.policies.base import ActionChunk


@dataclass(frozen=True)
class ChunkedRandomPolicyConfig:
    action_dim: int
    chunk_size: int = 4
    seed: int = 0
    scale: float = 1.0


class ChunkedRandomPolicy:
    name = "chunked_random"

    def __init__(self, config: ChunkedRandomPolicyConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def action_chunk(self, _observation: object, instruction: str) -> ActionChunk:
        actions = [
            [
                self._rng.uniform(-self.config.scale, self.config.scale)
                for _ in range(self.config.action_dim)
            ]
            for _ in range(self.config.chunk_size)
        ]
        return ActionChunk(
            actions=actions,
            policy_name=self.name,
            metadata={
                "instruction": instruction,
                "action_dim": self.config.action_dim,
                "chunk_size": self.config.chunk_size,
            },
        )

