from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ActionChunk:
    actions: list[list[float]]
    policy_name: str
    metadata: dict[str, object]

    @property
    def chunk_size(self) -> int:
        return len(self.actions)

    def first_action(self) -> list[float]:
        if not self.actions:
            raise ValueError("action chunk is empty")
        return self.actions[0]


class PolicyAdapter(Protocol):
    name: str

    def action_chunk(self, observation: object, instruction: str) -> ActionChunk:
        ...

