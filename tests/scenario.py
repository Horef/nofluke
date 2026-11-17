"""A fixed sequence of study operations used by the determinism and runner tests (module-level, so picklable)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from nofluke import Study
from nofluke.rng import derive

CONFIG: dict[str, Any] = {
    "direction": "maximize",
    "budget": 36,
    "seed": 11,
    "max_failures": 10,
    "strategy": {"n_init": 8},
    "noise": {"min_candidates": 4},
    "space": {
        "x": {"type": "float", "low": -2.0, "high": 2.0},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        "kind": {"type": "choice", "values": ["a", "b", "c"]},
        "deep": {"type": "bool"},
        "layers": {"type": "int", "low": 1, "high": 6, "when": {"deep": True}},
    },
}


def evaluate(params: dict[str, Any], seed: int) -> float:
    """Noisy quadratic with its peak at x = 1 (module-level so process pools can pickle it)."""
    return -(params["x"] - 1.0) ** 2 + 0.4 * derive(seed, "eval-noise").normal()


def run_operations(directory: str | Path) -> list[dict[str, Any]]:
    """Create a study and drive it: batches of 3 told in reverse order, a failure, an abandon; returns all asks."""
    study = Study.create(directory, CONFIG)
    seen: list[dict[str, Any]] = []
    step = 0
    while not study.is_done():
        asks = study.ask(3)
        seen += [a.to_dict() for a in asks]
        step += 1
        for a in reversed(asks):
            if step == 2 and a is asks[0]:
                study.tell_failed(a.eval, "simulated failure")
            elif step == 4 and a is asks[1]:
                study.abandon(a.eval, "simulated loss")
            else:
                study.tell(a.eval, evaluate(a.params, a.seed))
    return seen


if __name__ == "__main__":
    print(json.dumps(run_operations(sys.argv[1])))
