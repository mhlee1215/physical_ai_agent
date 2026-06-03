from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RandomPolicyConfig:
    action_dim: int
    seed: int = 0
    scale: float = 1.0


class RandomPolicy:
    def __init__(self, config: RandomPolicyConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def act(self, _observation: object) -> list[float]:
        return [
            self._rng.uniform(-self.config.scale, self.config.scale)
            for _ in range(self.config.action_dim)
        ]

