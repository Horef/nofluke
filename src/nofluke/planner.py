"""Planner: decides one ask, as a pure function of (normalised config, replayed state, number of slots).

Slots are filled in the order of D8: retries of failed evaluations, then for each remaining slot a re-evaluation of a
plausible-best candidate or a new candidate. New candidates come from the Latin-hypercube design for the first
`n_init` and from the configured strategy afterwards (D11). Seeds follow D10, all randomness is derived from the study
seed (D7). The result never exceeds the budget; `Study.ask` turns it into an ask event, `Study.verify` replays it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from nofluke import noise
from nofluke.errors import NoflukeError
from nofluke.rng import derive, derive_int
from nofluke.space import Space
from nofluke.state import FAILED, OK, PENDING, StudyState
from nofluke.strategies import History, Strategy, get_strategy
from nofluke.strategies.lhs import lhs_design

_MAX_DUPLICATE_RETRIES = 20


@lru_cache(maxsize=8)
def _lhs(seed: int, n: int, d: int) -> np.ndarray:
    design = lhs_design(n, d, derive(seed, "lhs"))
    design.setflags(write=False)
    return design


def _eval_seed(config: Mapping[str, Any], candidate: int, replicate: int) -> int:
    if config["noise"]["seed_mode"] == "per_candidate":
        return derive_int(config["seed"], "eval-seed", candidate, replicate)
    return derive_int(config["seed"], "eval-seed", replicate)


def _params_key(params: Mapping[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _history(space: Space, config: Mapping[str, Any], state: StudyState, model: noise.NoiseModel,
             planned: list[dict[str, Any]]) -> History:  # fmt: skip
    statuses: dict[int, set[str]] = {}
    for ev in state.evals.values():
        statuses.setdefault(ev.candidate, set()).add(ev.status)
    completed = [(cid, state.candidates[cid].params, s.mean, s.n) for cid, s in sorted(model.stats.items())]
    failed, pending = [], []
    for cid in sorted(statuses):
        if cid in model.stats:
            continue
        if PENDING in statuses[cid]:
            pending.append(state.candidates[cid].params)
        elif FAILED in statuses[cid]:
            failed.append(state.candidates[cid].params)
    return History(space, config["direction"], completed, failed, pending + planned)


def plan_ask(config: Mapping[str, Any], state: StudyState, n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (new candidates, new evaluations) for up to `n` slots, as the dicts of an ask event."""
    if state.n_failed() >= config["max_failures"]:
        raise NoflukeError(f"{state.n_failed()} evaluations have failed (max_failures = {config['max_failures']}); not asking")
    slots = min(n, config["budget"] - state.budget_used())
    if slots <= 0:
        return [], []
    space = Space.from_dict(config["space"])
    strategy = get_strategy(config)
    cfg_noise = config["noise"]
    model = noise.estimate(state, config)

    next_replicate: dict[int, int] = {}
    counts: dict[int, int] = {}
    n_issued = n_reeval = 0
    for ev in state.evals.values():
        next_replicate[ev.candidate] = max(next_replicate.get(ev.candidate, 0), ev.replicate + 1)
        if ev.status in (OK, PENDING):
            counts[ev.candidate] = counts.get(ev.candidate, 0) + 1
        if ev.kind != "retry":
            n_issued += 1
            n_reeval += ev.kind == "reeval"

    new_candidates: list[dict[str, Any]] = []
    new_evals: list[dict[str, Any]] = []
    known_keys = {_params_key(c.params) for c in state.candidates.values()}
    used = state.budget_used()
    next_eval = state.next_eval

    def add_eval(candidate: int, replicate: int, seed: int, kind: str, attempt: int) -> None:
        nonlocal next_eval, used
        new_evals.append({"eval": next_eval, "candidate": candidate, "replicate": replicate, "seed": seed,
                          "kind": kind, "attempt": attempt})  # fmt: skip
        next_eval += 1
        used += 1

    for failed in state.retry_queue()[:slots]:
        add_eval(failed.candidate, failed.replicate, failed.seed, "retry", failed.attempt + 1)

    while len(new_evals) < slots:
        after = used + 1
        target = None
        if cfg_noise["reeval_share"] > 0:
            in_finalist_zone = after > config["budget"] * (1 - cfg_noise["finalist_share"])
            within_share = (n_reeval + 1) / (n_issued + 1) <= cfg_noise["reeval_share"]
            if in_finalist_zone or (within_share and len(model.stats) >= cfg_noise["min_candidates"]):
                target = noise.choose_reeval(model, counts, cfg_noise["max_replicates"])
        if target is not None:
            replicate = next_replicate.get(target, 0)
            add_eval(target, replicate, _eval_seed(config, target, replicate), "reeval", 1)
            next_replicate[target] = replicate + 1
            counts[target] = counts.get(target, 0) + 1
            n_issued += 1
            n_reeval += 1
            continue

        candidate = state.next_candidate + len(new_candidates)
        history = _history(space, config, state, model, [c["params"] for c in new_candidates])
        params, strategy_name = _propose(config, space, strategy, history, candidate, known_keys)
        known_keys.add(_params_key(params))
        new_candidates.append({"candidate": candidate, "params": params,
                               "origin": {"strategy": strategy_name, "index": candidate}})  # fmt: skip
        add_eval(candidate, 0, _eval_seed(config, candidate, 0), "new", 1)
        next_replicate[candidate] = 1
        counts[candidate] = 1
        n_issued += 1
    return new_candidates, new_evals


def _propose(config: Mapping[str, Any], space: Space, strategy: Strategy, history: History, candidate: int,
             known_keys: set[str]) -> tuple[dict[str, Any], str]:  # fmt: skip
    """First `n_init` candidates from the LHS design, later ones from the strategy; re-draw exact duplicates."""
    seed, n_init = config["seed"], config["strategy"]["n_init"]
    if candidate <= n_init:
        params = space.decode(_lhs(seed, n_init, len(space))[candidate - 1])
        name = "lhs"
    else:
        params = strategy.propose(history, derive(seed, "candidate", candidate), candidate)
        name = strategy.name
    for attempt in range(1, _MAX_DUPLICATE_RETRIES + 1):
        if _params_key(params) not in known_keys:
            break
        params, name = strategy.propose(history, derive(seed, "candidate", candidate, attempt), candidate), strategy.name
    return space.validate_params(params), name
