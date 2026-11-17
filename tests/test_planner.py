"""Tests for nofluke.planner through an in-memory `Sim`: D8 slot order, re-evaluation share, finalists, retries, seeds."""

from __future__ import annotations

import pytest

from conftest import Sim, make_config
from nofluke.errors import NoflukeError
from nofluke.rng import derive
from nofluke.space import Space


def noisy_value(config, evaluation, params, sd=1.0):
    """Deterministic noisy score: a weak signal in x plus noise keyed by (candidate, replicate)."""
    return -0.1 * params["x"] ** 2 + sd * derive(99, "noise", evaluation["candidate"], evaluation["replicate"]).normal()


def run(sim, n_total, batch=1, sd=1.0):
    """Ask `batch` at a time and tell everything immediately, until `n_total` evaluations were issued."""
    issued = []
    while len(issued) < n_total:
        evals = sim.ask(min(batch, n_total - len(issued)))
        if not evals:
            break
        for e in evals:
            params = sim.state.candidates[e["candidate"]].params
            sim.tell(e["eval"], noisy_value(sim.config, e, params, sd))
        issued += evals
    return issued


def test_reeval_share_converges_to_a_third():
    sim = Sim(make_config(budget=300, noise={"max_replicates": 100, "finalist_share": 0.0}))
    issued = run(sim, 300)
    share = sum(e["kind"] == "reeval" for e in issued) / len(issued)
    assert 0.30 <= share <= 0.34


def test_reeval_share_respects_the_configured_value():
    sim = Sim(make_config(budget=200, noise={"reeval_share": 0.1, "max_replicates": 100, "finalist_share": 0.0}))
    issued = run(sim, 200)
    assert 0.08 <= sum(e["kind"] == "reeval" for e in issued) / len(issued) <= 0.11


def test_no_reevaluations_before_min_candidates_have_results():
    sim = Sim(make_config(budget=100, noise={"min_candidates": 8}))
    first = sim.ask(20)  # nothing told yet
    assert [e["kind"] for e in first] == ["new"] * 20
    for e in first[:7]:
        sim.tell(e["eval"], 1.0)
    assert [e["kind"] for e in sim.ask(5)] == ["new"] * 5  # only 7 candidates have results
    sim.tell(first[7]["eval"], 2.0)
    kinds = [e["kind"] for e in sim.ask(10)]
    assert "reeval" in kinds


def test_slots_within_one_ask_spread_over_candidates_and_replicates_are_sequential():
    sim = Sim(make_config(budget=100, noise={"min_candidates": 2, "finalist_share": 0.0}))
    for e in sim.ask(3):
        sim.tell(e["eval"], 0.0)  # three tied candidates: all plausible
    evals = sim.ask(12)
    reevals = [e for e in evals if e["kind"] == "reeval"]
    assert len(reevals) >= 3
    pairs = [(e.candidate, e.replicate) for e in sim.state.evals.values()]
    assert len(pairs) == len(set(pairs))  # no (candidate, replicate) issued twice
    assert len({e["candidate"] for e in reevals[:3]}) == 3  # fewest evaluations first


def test_finalist_zone_spends_the_last_share_on_reevaluations_only():
    sim = Sim(make_config(budget=40, noise={"finalist_share": 0.25, "reeval_share": 0.05, "max_replicates": 40}))
    issued = run(sim, 40, sd=5.0)
    zone_start = 30  # slots whose used count after issue exceeds 40 * 0.75 = 30 are slots 31..40
    assert all(e["kind"] == "reeval" for e in issued[zone_start:])
    assert any(e["kind"] == "new" for e in issued[:zone_start])


def test_finalist_zone_ends_when_every_plausible_candidate_is_capped():
    sim = Sim(make_config(budget=30, noise={"finalist_share": 0.5, "max_replicates": 1, "min_candidates": 2}))
    issued = run(sim, 30)
    assert all(e["kind"] == "new" for e in issued)  # cap 1: nothing can be re-evaluated, new candidates fill in


def test_max_replicates_caps_evaluations_per_candidate():
    sim = Sim(make_config(budget=120, noise={"max_replicates": 3, "finalist_share": 0.2}))
    run(sim, 120)
    per_candidate = {}
    for e in sim.state.evals.values():
        per_candidate[e.candidate] = per_candidate.get(e.candidate, 0) + 1
    assert max(per_candidate.values()) <= 3


def test_reeval_share_zero_turns_the_protocol_off_including_finalists():
    sim = Sim(make_config(budget=40, noise={"reeval_share": 0.0, "finalist_share": 0.3}))
    issued = run(sim, 40)
    assert all(e["kind"] == "new" for e in issued)


def test_retries_come_first_with_the_same_candidate_replicate_and_seed():
    sim = Sim(make_config(budget=50, noise={"min_candidates": 100}))
    first = sim.ask(4)
    sim.fail(first[1]["eval"])
    sim.fail(first[3]["eval"])
    sim.tell(first[0]["eval"], 1.0)
    nxt = sim.ask(3)
    assert [e["kind"] for e in nxt] == ["retry", "retry", "new"]
    for retry, original in zip(nxt, (first[1], first[3])):
        assert (retry["candidate"], retry["replicate"], retry["seed"]) == (
            original["candidate"], original["replicate"], original["seed"])  # fmt: skip
        assert retry["attempt"] == 2
    sim.fail(nxt[0]["eval"])  # second failure: retries = 1, so no third attempt
    again = sim.ask(5)
    assert all(not (e["kind"] == "retry" and e["candidate"] == nxt[0]["candidate"]) for e in again)


def test_retries_option_allows_more_attempts_and_zero_disables():
    sim = Sim(make_config(retries=2, budget=50))
    e = sim.ask(1)[0]
    sim.fail(e["eval"])
    r1 = sim.ask(1)[0]
    sim.fail(r1["eval"])
    r2 = sim.ask(1)[0]
    assert (r1["kind"], r1["attempt"], r2["kind"], r2["attempt"]) == ("retry", 2, "retry", 3)
    assert r2["seed"] == e["seed"]
    sim.fail(r2["eval"])
    assert sim.ask(1)[0]["kind"] != "retry"
    sim0 = Sim(make_config(retries=0, budget=50))
    sim0.fail(sim0.ask(1)[0]["eval"])
    assert sim0.ask(1)[0]["kind"] == "new"


def test_budget_is_never_exceeded_and_abandon_frees_a_slot():
    sim = Sim(make_config(budget=10))
    evals = sim.ask(100)
    assert len(evals) == 10
    assert sim.ask(1) == []
    sim.state.apply_abandon(evals[0]["eval"], "gone")
    assert len(sim.ask(5)) == 1
    assert sim.state.budget_used() == 10


def test_failed_evaluations_free_their_slot_but_count_towards_max_failures():
    sim = Sim(make_config(budget=10, max_failures=3, retries=0))
    for _ in range(3):
        sim.fail(sim.ask(1)[0]["eval"])
    with pytest.raises(NoflukeError, match="max_failures = 3"):
        sim.ask(1)


def test_first_n_init_candidates_come_from_one_latin_hypercube():
    sim = Sim(make_config(budget=60, strategy={"n_init": 12}))
    evals = sim.ask(20)
    cands = [sim.state.candidates[e["candidate"]] for e in evals]
    assert [c.origin["strategy"] for c in cands] == ["lhs"] * 12 + ["random"] * 8
    assert [c.origin["index"] for c in cands] == list(range(1, 21))
    space = Space.from_dict(sim.config["space"])
    for name in ("x", "lr"):  # one point per stratum in every continuous dimension
        param = next(p for p in space.params if p.name == name)
        strata = sorted(int(param.encode(c.params[name]) * 12) for c in cands[:12])
        assert strata == list(range(12))


def test_seed_modes():
    shared = Sim(make_config(budget=60, noise={"min_candidates": 2, "finalist_share": 0.0}))
    issued = run(shared, 40)
    by_replicate = {}
    for e in issued:
        by_replicate.setdefault(e["replicate"], set()).add(e["seed"])
    assert all(len(seeds) == 1 for seeds in by_replicate.values()) and max(by_replicate) >= 1
    assert len({next(iter(s)) for s in by_replicate.values()}) == len(by_replicate)

    per = Sim(make_config(seed=3, budget=60, noise={"seed_mode": "per_candidate"}))
    first = per.ask(6)
    assert len({e["seed"] for e in first}) == 6
    other = Sim(make_config(seed=3, budget=60, noise={"seed_mode": "per_candidate"}))
    assert [e["seed"] for e in other.ask(6)] == [e["seed"] for e in first]


def test_duplicate_proposals_are_redrawn():
    config = make_config(budget=6, strategy={"n_init": 0}, space={"flag": {"type": "bool"}}, noise={"reeval_share": 0.0})
    sim = Sim(config)
    evals = sim.ask(2)
    keys = {str(sim.state.candidates[e["candidate"]].params) for e in evals}
    assert len(keys) == 2  # only two distinct candidates exist; the second draw must not repeat the first
    assert len(sim.ask(3)) == 3  # afterwards duplicates are accepted rather than looping forever


def test_asks_are_pure_functions_of_state():
    a, b = Sim(make_config()), Sim(make_config())
    ea, eb = run(a, 25, batch=4), run(b, 25, batch=4)
    assert ea == eb
    assert {k: c.params for k, c in a.state.candidates.items()} == {k: c.params for k, c in b.state.candidates.items()}


def test_one_big_ask_equals_several_small_ones_when_nothing_is_told_between():
    big = Sim(make_config(budget=100))
    small = Sim(make_config(budget=100))
    assert big.ask(8) == small.ask(3) + small.ask(3) + small.ask(2)
