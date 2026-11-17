"""Tests for nofluke.noise against hand-computed D9 numbers, fallbacks, tie-breaks and the plausible-best set."""

from __future__ import annotations

import math

import pytest

from conftest import Sim, make_config
from nofluke.noise import CandidateStats, NoiseModel, choose_reeval, estimate, plausible_best, rank


def sim_with(values: dict[int, list[float]], **overrides) -> Sim:
    """A Sim in which candidate k has completed the given values (evals are issued by `ask`, then told).

    The D9 formula tests switch the D14 seed-effect correction off unless a test asks for it."""
    overrides["noise"] = {"seed_effects": "ignore", **overrides.get("noise", {})}
    sim = Sim(make_config(budget=500, strategy={"n_init": 500}, **overrides))
    sim.ask(len(values))  # candidate ids 1..K, replicate 0
    cand_evals = {}
    for cid, vals in values.items():
        cand_evals[cid] = [next(e for e in sim.state.evals.values() if e.candidate == cid).id]
    for cid, vals in values.items():
        sim.tell(cand_evals[cid][0], vals[0])
        for v in vals[1:]:
            e = sim.state.next_eval
            sim.state.apply({"event": "ask", "ask": sim.state.n_asks + 1, "time": "t", "candidates": [], "evals": [
                {"eval": e, "candidate": cid, "replicate": 1, "seed": 0, "kind": "reeval", "attempt": 1}]})
            sim.tell(e, v)
    return sim


DATA = {1: [0, 2], 2: [4, 6], 3: [1, 3, 2], 4: [10]}
# Hand computation (D9): pooled s_p^2 = (1*2 + 1*2 + 2*1) / (1+1+2) = 1.5; nu0 = 4;
# sigma_i^2 = (4*1.5 + (n-1)*s_i^2)/(4 + n-1) -> 1.6, 1.6, 4/3, 1.5; variances of the means 0.8, 0.8, 4/9, 1.5;
# m = 4.5; Var(ybar) = 49/3; tau^2 = 49/3 - 0.886111... = 15.447222...
EXPECTED = {
    1: dict(mean=1.0, sd=math.sqrt(2), shrunk=1.1723371516498542, se=0.8721288041633884),
    2: dict(mean=5.0, sd=math.sqrt(2), shrunk=4.975380406907163, se=0.8721288041633884),
    3: dict(mean=2.0, sd=1.0, shrunk=2.069917846530327, se=0.6572781793588428),
    4: dict(mean=10.0, sd=None, shrunk=9.513194558269134, se=1.1692881935296986),
}


def test_d9_formulas_match_hand_computation():
    sim = sim_with(DATA)
    model = estimate(sim.state, sim.config)
    assert model.sigma_source == "pooled" and model.sigma == pytest.approx(math.sqrt(1.5))
    assert not model.tie_break
    for cid, want in EXPECTED.items():
        s = model.stats[cid]
        assert s.n == len(DATA[cid])
        assert s.mean == pytest.approx(want["mean"])
        assert (s.sd is None) if want["sd"] is None else s.sd == pytest.approx(want["sd"])
        assert s.shrunk_mean == pytest.approx(want["shrunk"], rel=1e-12)
        assert s.se == pytest.approx(want["se"], rel=1e-12)
        assert s.lcb == pytest.approx(want["shrunk"] - 1.5 * want["se"], rel=1e-12)
        assert s.ucb == pytest.approx(want["shrunk"] + 1.5 * want["se"], rel=1e-12)
    assert model.stats[1].sigma_i == pytest.approx(math.sqrt(1.6))
    assert model.stats[3].sigma_i == pytest.approx(math.sqrt(4 / 3))
    assert model.stats[4].sigma_i == pytest.approx(math.sqrt(1.5))
    assert rank(model) == [4, 2, 3, 1]
    assert plausible_best(model) == [4]


def test_minimize_negates_scores_internally():
    sim = sim_with(DATA, direction="minimize")
    model = estimate(sim.state, sim.config)
    assert model.stats[4].mean == pytest.approx(-10.0)
    assert rank(model) == [1, 3, 2, 4]


def test_sigma_falls_back_to_prior_sd_then_to_the_spread_of_the_means():
    two_replicated = {1: [0, 2], 2: [4, 6], 3: [1], 4: [10]}  # only 2 candidates with n >= 2: no pooling
    sim = sim_with(two_replicated, noise={"prior_sd": 2.0})
    model = estimate(sim.state, sim.config)
    assert (model.sigma_source, model.sigma) == ("prior_sd", 2.0)
    assert model.stats[3].sigma_i == pytest.approx(2.0)  # n=1: (nu0*sp2 + 0)/nu0
    assert model.stats[1].sigma_i == pytest.approx(math.sqrt((4 * 4.0 + 1 * 2.0) / 5))

    sim = sim_with(two_replicated)
    model = estimate(sim.state, sim.config)
    ybar = [1.0, 5.0, 1.0, 10.0]
    m = sum(ybar) / 4
    var = sum((y - m) ** 2 for y in ybar) / 3
    assert model.sigma_source == "spread" and model.sigma == pytest.approx(math.sqrt(var))


def test_tau_zero_shrinks_everything_and_ties_are_broken_by_the_lower_bound_rule():
    sim = sim_with({1: [1.0], 2: [1.5]})  # no replicates, no prior: all spread is treated as noise
    model = estimate(sim.state, sim.config)
    assert model.tie_break
    assert model.stats[1].shrunk_mean == model.stats[2].shrunk_mean == pytest.approx(1.25)
    assert all(s.se == 0.0 for s in model.stats.values())
    assert rank(model) == [2, 1]  # mean - z*sigma_i/sqrt(n): same sigma, higher mean first
    assert plausible_best(model) == [1, 2]


def test_single_candidate_has_no_population():
    sim = sim_with({1: [1.0, 3.0, 5.0]})
    model = estimate(sim.state, sim.config)
    s = model.stats[1]
    assert model.sigma_source == "own" and s.shrunk_mean == 3.0 and s.se == pytest.approx(2.0 / math.sqrt(3))
    sim = sim_with({1: [1.0]}, noise={"prior_sd": 0.4})
    s = estimate(sim.state, sim.config).stats[1]
    assert s.shrunk_mean == 1.0 and s.se == pytest.approx(0.4)
    s = estimate(sim_with({1: [1.0]}).state, make_config()).stats[1]
    assert s.se == 0.0 and s.lcb == s.ucb == 1.0


def test_no_results_gives_an_empty_model():
    model = estimate(Sim(make_config()).state, make_config())
    assert model.stats == {} and rank(model) == [] and plausible_best(model) == []
    assert choose_reeval(model, {}, 6) is None


def test_ties_in_the_ranking_fall_back_to_the_candidate_id():
    sim = sim_with({1: [2.0], 2: [2.0], 3: [2.0]})
    assert rank(estimate(sim.state, sim.config)) == [1, 2, 3]


def _stat(cid, shrunk, se, n=1, mean=None):
    m = shrunk if mean is None else mean
    return CandidateStats(cid, n, m, m, None, 1.0, shrunk, se, shrunk - 1.5 * se, shrunk + 1.5 * se)


def test_plausible_best_set_and_reeval_choice():
    stats = {i: _stat(i, shrunk, se) for i, shrunk, se in
             [(1, 5.0, 1.0), (2, 4.0, 1.0), (3, 2.0, 1.0), (4, 5.5, 0.1), (5, 3.5, 0.5)]}  # fmt: skip
    model = NoiseModel(stats, 1.5, 1.0, "pooled", False)
    # max LCB = 5.5 - 0.15 = 5.35; UCB >= 5.35 for: 1 (6.5), 2 (5.5), 4, but not 3 (3.5) or 5 (4.25)
    assert plausible_best(model) == [1, 2, 4]
    counts = {1: 2, 2: 1, 4: 1, 3: 1, 5: 1}
    assert choose_reeval(model, counts, 6) == 4  # fewest evaluations: 2 and 4 (one each); 4 has the better shrunk mean
    counts[4] = 3
    assert choose_reeval(model, counts, 6) == 2
    counts[2] = 6
    assert choose_reeval(model, counts, 6) == 1
    counts[1] = 6
    assert choose_reeval(model, counts, 6) == 4
    counts[4] = 6
    assert choose_reeval(model, counts, 6) is None


def test_reeval_ties_prefer_the_higher_shrunk_mean_then_lower_id():
    stats = {i: _stat(i, 1.0, 1.0) for i in (1, 2)} | {3: _stat(3, 1.2, 1.0)}
    model = NoiseModel(stats, 1.5, 1.0, "pooled", False)
    assert choose_reeval(model, {1: 1, 2: 1, 3: 1}, 6) == 3
    assert choose_reeval(model, {1: 1, 2: 1, 3: 2}, 6) == 1


# --- D14: seed-effect correction ----------------------------------------------------------------------------------

def _sim_replicates(table: dict[int, dict[int, float]], **noise_overrides) -> Sim:
    """A Sim where candidate c has value table[c][r] on replicate r (replicate 0 asked first, then re-evaluations)."""
    sim = Sim(make_config(budget=500, strategy={"n_init": 500}, noise=noise_overrides))
    sim.ask(len(table))
    for cid, by_rep in table.items():
        first = next(e for e in sim.state.evals.values() if e.candidate == cid)
        sim.tell(first.id, by_rep[0])
        for rep in sorted(r for r in by_rep if r > 0):
            e = sim.state.next_eval
            sim.state.apply({"event": "ask", "ask": sim.state.n_asks + 1, "time": "t", "candidates": [], "evals": [
                {"eval": e, "candidate": cid, "replicate": rep, "seed": 0, "kind": "reeval", "attempt": 1}]})
            sim.tell(e, by_rep[rep])
    return sim


# true values f = (1, 2, 3, 4); seed effects b = (0, +10, -4); candidate 1 got the lucky seed 1 twice over the others
TRUE = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
EFFECT = {0: 0.0, 1: 10.0, 2: -4.0}
REPS = {1: (0, 1), 2: (0, 1, 2), 3: (0, 2), 4: (0,)}


def test_seed_effects_are_recovered_and_removed_exactly_without_noise():
    table = {c: {r: TRUE[c] + EFFECT[r] for r in REPS[c]} for c in TRUE}
    model = estimate(_sim_replicates(table).state, make_config(budget=500))
    effects = model.seed_effects
    assert effects[1] - effects[0] == pytest.approx(10.0) and effects[2] - effects[0] == pytest.approx(-4.0)
    means = {c: model.stats[c].mean for c in TRUE}
    for c in TRUE:  # adjusted means differ exactly by the true differences
        assert means[c] - means[1] == pytest.approx(TRUE[c] - TRUE[1])
    assert rank(model)[0] == 4
    assert model.stats[1].raw_mean == pytest.approx((1 + 0) / 2 + (1 + 10) / 2)


def test_ignoring_seed_effects_lets_a_lucky_seed_win():
    table = {c: {r: TRUE[c] + EFFECT[r] for r in REPS[c]} for c in TRUE}
    model = estimate(_sim_replicates(table, seed_effects="ignore").state,
                     make_config(budget=500, noise={"seed_effects": "ignore"}))  # fmt: skip
    assert model.seed_effects == {} and model.stats[1].mean > model.stats[4].mean  # 6.0 > 4.0: the fluke


def test_no_correction_with_one_replicate_or_per_candidate_seeds():
    one = {c: {0: TRUE[c]} for c in TRUE}
    assert estimate(_sim_replicates(one).state, make_config(budget=500)).seed_effects == {}
    table = {c: {r: TRUE[c] + EFFECT[r] for r in REPS[c]} for c in TRUE}
    cfg = make_config(budget=500, noise={"seed_mode": "per_candidate"})
    assert estimate(_sim_replicates(table, seed_mode="per_candidate").state, cfg).seed_effects == {}


def test_journals_without_the_key_keep_unadjusted_statistics():
    table = {c: {r: TRUE[c] + EFFECT[r] for r in REPS[c]} for c in TRUE}
    sim = _sim_replicates(table)
    legacy = {**sim.config, "noise": {k: v for k, v in sim.config["noise"].items() if k != "seed_effects"}}
    assert estimate(sim.state, legacy).seed_effects == {}
