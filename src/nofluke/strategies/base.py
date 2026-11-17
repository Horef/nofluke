"""Strategy interface: the `History` a strategy sees and the `Strategy` base class that proposes one candidate.

A strategy proposes one candidate at a time from (history, rng, index), where `rng` is derived for that candidate
index (D7) and the history lists completed candidates sorted by id plus pending ones, so proposals within one ask
do not collide and are independent of the order in which results arrived. Scores are in "higher is better" units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from nofluke.space import Space


@dataclass(frozen=True)
class History:
    """What a strategy may use: `completed` is [(candidate id, params, mean score, n)] sorted by id; `failed` and
    `pending` are the params of candidates without results (failed with nothing running; running)."""

    space: Space
    direction: str
    completed: list[tuple[int, dict[str, Any], float, int]]
    failed: list[dict[str, Any]]
    pending: list[dict[str, Any]]


class Strategy(ABC):
    """Proposes candidates; subclasses set `name` and implement `propose`."""

    name: str

    @abstractmethod
    def propose(self, history: History, rng: np.random.Generator, index: int) -> dict[str, Any]:
        """Return the parameters of candidate number `index`, drawing all randomness from `rng`."""
