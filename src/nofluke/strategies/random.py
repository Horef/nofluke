"""Random-search strategy: every candidate is a uniform draw from the search space."""

from __future__ import annotations

from typing import Any

import numpy as np

from nofluke.strategies.base import History, Strategy


class RandomStrategy(Strategy):
    """Decode a uniform point of the unit cube; ignores the history."""

    name = "random"

    def propose(self, history: History, rng: np.random.Generator, index: int) -> dict[str, Any]:
        return history.space.decode(rng.random(len(history.space)))
