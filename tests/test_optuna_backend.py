"""Tests for nofluke.strategies.optuna_backend (`tpe`, `gp`): valid proposals, determinism, pending candidates, options."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("optuna")

from conftest import SPACE, make_config
from nofluke import Study
from nofluke.errors import NoflukeError
from nofluke.rng import derive
from nofluke.space import Space
from nofluke.strategies import get_strategy
from nofluke.strategies.base import History
from nofluke.strategies.lhs import lhs_design
from nofluke.strategies.optuna_backend import OptunaStrategy

TESTS = Path(__file__).parent
HAS_GP = all(importlib.util.find_spec(m) is not None for m in ("torch", "scipy"))
NAMES = ["tpe", pytest.param("gp", marks=pytest.mark.skipif(not HAS_GP, reason="gp needs torch and scipy: pip install 'nofluke[gp]'"))]


def mixed_history(n_completed: int, n_failed: int = 0, n_pending: int = 0) -> History:
    """A fixed history over the mixed conftest space: random completed candidates scored by -(x - 1)^2."""
    space = Space.from_dict(SPACE)
    rng = derive(5, "test-history")
    draw = lambda: space.decode(rng.random(len(space)))  # noqa: E731
    completed = []
    for cid in range(1, n_completed + 1):
        params = draw()
        completed.append((cid, params, -((params["x"] - 1.0) ** 2), 1))
    return History(space, "maximize", completed, [draw() for _ in range(n_failed)], [draw() for _ in range(n_pending)])


def propose(name: str, history: History, index: int = 13, options: dict[str, Any] | None = None) -> dict[str, Any]:
    return OptunaStrategy(name, options or {}).propose(history, derive(1, "candidate", index), index)


@pytest.mark.parametrize("name", NAMES)
def test_proposals_are_valid_for_a_mixed_space_with_a_conditional(name):
    history = mixed_history(14, n_failed=2, n_pending=2)
    strategy = OptunaStrategy(name, {})
    seen = [strategy.propose(history, derive(1, "candidate", k), k) for k in range(15, 21)]
    for params in seen:
        assert history.space.validate_params(params) == params
        assert ("layers" in params) == params["deep"]
    assert len({json.dumps(p, sort_keys=True) for p in seen}) == len(seen)


@pytest.mark.parametrize("name", NAMES)
def test_few_or_no_completed_candidates_still_propose(name):
    for n in (0, 1, 3):
        assert propose(name, mixed_history(n, n_pending=2))


@pytest.mark.parametrize("name", NAMES)
def test_choice_values_that_compare_equal_stay_distinct(name):
    space = Space.from_dict({
        "c": {"type": "choice", "values": [1, 1.0, True, None, "a"]},
        "n": {"type": "int", "low": 1, "high": 64, "log": True},
    })  # fmt: skip
    rng = derive(2, "test-history")
    completed = [(i, p, float(i % 5), 1) for i, p in enumerate((space.decode(rng.random(2)) for _ in range(12)), 1)]
    history = History(space, "maximize", completed, [], [])
    kinds = {type(propose(name, history, k, {"n_startup_trials": 100})["c"]).__name__ for k in range(13, 60)}
    assert kinds == {"int", "float", "bool", "NoneType", "str"}


def test_history_encodes_choice_values_that_compare_equal_as_different_categories():
    from nofluke.strategies.optuna_backend import _to_trial_params

    space = Space.from_dict({"c": {"type": "choice", "values": [1, 1.0, True, None, "a"]}, "b": {"type": "bool"}})
    encoded = [_to_trial_params(space, {"c": v, "b": b})[0] for v, b in ((1, False), (1.0, True), (True, False), (None, True))]
    assert encoded == [{"c": 0, "b": 0}, {"c": 1, "b": 1}, {"c": 2, "b": 0}, {"c": 3, "b": 1}]


@pytest.mark.parametrize("name", NAMES)
def test_same_inputs_give_the_same_proposal(name):
    history = mixed_history(14, n_failed=1, n_pending=2)
    assert propose(name, history) == propose(name, history)
    assert propose(name, history, 14) != propose(name, history, 13)


SUBPROCESS_SCRIPT = """
import json, sys
from test_optuna_backend import mixed_history, propose
print(json.dumps(propose(sys.argv[1], mixed_history(14, n_failed=1, n_pending=2))))
"""


@pytest.mark.parametrize("name", NAMES)
def test_a_subprocess_reproduces_the_proposal(name):
    done = subprocess.run([sys.executable, "-c", SUBPROCESS_SCRIPT, name],
                          capture_output=True, text=True, check=True, cwd=TESTS)  # fmt: skip
    assert json.loads(done.stdout) == propose(name, mixed_history(14, n_failed=1, n_pending=2))


@pytest.mark.parametrize("name", NAMES)
def test_pending_candidates_change_the_proposal(name):
    history = mixed_history(14)
    best_three = [c[1] for c in sorted(history.completed, key=lambda c: c[2])[-3:]]
    pending = replace(history, pending=best_three)
    assert propose(name, pending) != propose(name, history)


@pytest.mark.parametrize("name", NAMES)
def test_failed_candidates_are_accepted(name):
    assert propose(name, mixed_history(14, n_failed=4))


def test_options_reach_the_sampler():
    history = mixed_history(14)
    assert propose("tpe", history, options={"n_startup_trials": 100}) != propose("tpe", history)


def test_bad_options_are_rejected_at_creation():
    with pytest.raises(NoflukeError, match="'seed' is not an option"):
        OptunaStrategy("tpe", {"seed": 3})
    with pytest.raises(NoflukeError, match="bad options"):
        OptunaStrategy("tpe", {"no_such_option": 1})
    with pytest.raises(NoflukeError, match="'seed' is not an option"):
        Study.create(Path("unused"), make_config(strategy={"name": "tpe", "options": {"seed": 3}}))


@pytest.mark.parametrize(("name", "extra"), [("tpe", "optuna"), ("gp", "gp")])
def test_missing_optuna_names_the_extra_to_install(monkeypatch, name, extra):
    monkeypatch.setitem(sys.modules, "optuna", None)
    monkeypatch.delitem(sys.modules, "nofluke.strategies.optuna_backend")
    with pytest.raises(NoflukeError, match=rf"nofluke\[{extra}\]"):
        get_strategy(make_config(strategy={"name": name}))


def test_the_core_does_not_import_optuna_until_asked():
    code = "import sys, nofluke; from nofluke.strategies import get_strategy; " \
           "from conftest import make_config; get_strategy(make_config()); assert 'optuna' not in sys.modules"  # fmt: skip
    subprocess.run([sys.executable, "-c", code], check=True, cwd=TESTS)


@pytest.mark.parametrize("name", NAMES)
def test_a_study_runs_to_done_and_verifies(tmp_path, name):
    config = make_config(budget=20, strategy={"name": name, "n_init": 8})
    study = Study.create(tmp_path, config)
    while not study.is_done():
        asks = study.ask(3)
        for a in asks:
            study.tell(a.eval, -((a.params["x"] - 1.0) ** 2) + 0.05 * derive(a.seed, "noise").normal())
    assert study.verify() == []
    origins = [json.loads(line).get("candidates", []) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert {c["origin"]["strategy"] for cs in origins for c in cs} == {"lhs", name}


def final_distances(name: str, seed: int, budget: int = 36, n_init: int = 10) -> float:
    """Mean distance to the optimum (0.3, -0.2) of the last 12 proposals of a sequential noiseless 2-D quadratic search."""
    space = Space.from_dict({"a": {"type": "float", "low": -1.0, "high": 1.0}, "b": {"type": "float", "low": -1.0, "high": 1.0}})
    score = lambda p: -((p["a"] - 0.3) ** 2 + (p["b"] + 0.2) ** 2)  # noqa: E731
    design = lhs_design(n_init, 2, derive(seed, "lhs"))
    strategy = OptunaStrategy("tpe", {}) if name == "tpe" else None
    completed, distances = [], []
    for k in range(1, budget + 1):
        rng = derive(seed, "candidate", k)
        if k <= n_init:
            params = space.decode(design[k - 1])
        elif strategy is None:
            params = space.decode(rng.random(2))
        else:
            params = strategy.propose(History(space, "maximize", completed, [], []), rng, k)
        completed.append((k, params, score(params), 1))
        if k > budget - 12:
            distances.append(float(np.hypot(params["a"] - 0.3, params["b"] + 0.2)))
    return float(np.mean(distances))


def test_tpe_concentrates_near_the_optimum_better_than_random():
    seeds = range(1, 7)
    tpe = np.mean([final_distances("tpe", s) for s in seeds])
    uniform = np.mean([final_distances("random", s) for s in seeds])
    assert tpe < 0.8 * uniform, (tpe, uniform)
