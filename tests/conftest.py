"""Shared test helpers: a config factory and `Sim`, an in-memory ask/tell driver over the pure planner."""

from __future__ import annotations

from typing import Any

import pytest

from nofluke.config import normalise
from nofluke.noise import estimate
from nofluke.planner import plan_ask
from nofluke.state import StudyState

SPACE = {
    "x": {"type": "float", "low": -2.0, "high": 2.0},
    "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "kind": {"type": "choice", "values": ["a", "b", "c"]},
    "deep": {"type": "bool"},
    "layers": {"type": "int", "low": 1, "high": 6, "when": {"deep": True}},
}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Normalised config for a small mixed space; keyword overrides replace top-level keys."""
    return normalise({"direction": "maximize", "budget": 40, "seed": 3, "space": SPACE, **overrides})


@pytest.fixture
def config_factory():
    return make_config


class Sim:
    """Drives `plan_ask` against an in-memory `StudyState` like `Study` does, without touching disk."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.state = StudyState(config=config)
        self.clock = 0

    def ask(self, n: int = 1) -> list[dict[str, Any]]:
        candidates, evals = plan_ask(self.config, self.state, n)
        if evals:
            self.clock += 1
            self.state.apply({"event": "ask", "ask": self.state.n_asks + 1, "time": f"t{self.clock}",
                              "candidates": candidates, "evals": evals})  # fmt: skip
        return evals

    def tell(self, eval_id: int, value: float) -> None:
        self.state.apply_tell(eval_id, "ok", value, None, None, "t")

    def fail(self, eval_id: int, reason: str = "boom") -> None:
        self.state.apply_tell(eval_id, "failed", None, reason, None, "t")

    def model(self):
        return estimate(self.state, self.config)
