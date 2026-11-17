"""Tests for benchmarks/problems.py and benchmarks/harness.py: known optima, noise determinism and a tiny end-to-end run."""

from __future__ import annotations

import json

import numpy as np
import pytest

import harness
import problems as P
import summarise
from nofluke.space import Space


def test_branin_optima():
    branin = P.get_problem("branin")
    for point in P.BRANIN_ARGMIN:
        assert branin.true(point) == pytest.approx(-0.397887, abs=1e-5)
    assert branin.optimum == pytest.approx(-0.397887)
    rng = np.random.default_rng(0)
    space = Space.from_dict(branin.space)
    assert max(branin.true(space.decode(u)) for u in rng.random((2000, 2))) < branin.optimum + 1e-9


def test_hartmann6_optimum():
    hartmann = P.get_problem("hartmann6")
    assert hartmann.true(hartmann.optimum_params) == pytest.approx(3.32237, abs=1e-4)
    assert hartmann.optimum == pytest.approx(3.32237)
    rng = np.random.default_rng(0)
    space = Space.from_dict(hartmann.space)
    assert max(hartmann.true(space.decode(u)) for u in rng.random((2000, 6))) < hartmann.optimum


def test_mixed_hpo_optimum_is_inside_the_space():
    problem = P.get_problem("mixed_hpo")
    space = Space.from_dict(problem.space)
    best = space.validate_params(problem.optimum_params)
    assert problem.true(best) == pytest.approx(problem.optimum)
    for name in ("loss_w", "lr", "aux_w"):  # continuous optima strictly inside their ranges
        spec = problem.space[name]
        assert spec["low"] < best[name] < spec["high"]
    assert problem.space["epochs"]["low"] < best["epochs"] < problem.space["epochs"]["high"]
    rng = np.random.default_rng(1)
    assert max(problem.true(space.decode(u)) for u in rng.random((3000, len(space)))) < problem.optimum


def test_noise_is_deterministic_and_has_the_crn_structure():
    problem = P.get_problem("branin")
    params = {"x1": 1.0, "x2": 2.0}
    a = problem.observe(params, eval_seed=7, candidate=3, seed=0, noise_ratio=0.5)
    assert a == problem.observe(params, eval_seed=7, candidate=3, seed=0, noise_ratio=0.5)
    assert a != problem.observe(params, eval_seed=8, candidate=3, seed=0, noise_ratio=0.5)
    assert a != problem.observe(params, eval_seed=7, candidate=4, seed=0, noise_ratio=0.5)
    # the same noise draw scaled by the ratio: observation - true is proportional to noise_ratio
    b = problem.observe(params, eval_seed=7, candidate=3, seed=0, noise_ratio=1.0)
    assert (b - problem.true(params)) == pytest.approx(2 * (a - problem.true(params)))
    # noise SD over many (eval seed, candidate) draws is about noise_ratio x true_sd
    draws = [problem.observe(params, s, c, 0, 1.0) - problem.true(params) for s in range(60) for c in range(60)]
    assert np.std(draws) == pytest.approx(problem.true_sd, rel=0.15)
    # candidates sharing an eval seed share an offset: their noise is positively correlated across eval seeds
    first = [problem.observe(params, s, 1, 0, 1.0) - problem.true(params) for s in range(400)]
    second = [problem.observe(params, s, 2, 0, 1.0) - problem.true(params) for s in range(400)]
    assert np.corrcoef(first, second)[0, 1] == pytest.approx(P.SHARED_VARIANCE_SHARE, abs=0.15)


def test_true_sd_is_deterministic():
    assert P.get_problem("mixed_hpo").true_sd == pytest.approx(P.get_problem("mixed_hpo").true_sd, rel=0)
    assert 0.03 < P.get_problem("mixed_hpo").true_sd < 0.15


def test_arm_expansion_and_gp_skip():
    assert harness.expand_arms(["random", "tpe-on"]) == ["random-on", "random-off", "tpe-on"]
    specs, notes = harness.make_specs(["branin", "mixed_hpo"], [0.5], ["gp-on"], [0], 16, None)
    assert [s["problem"] for s in specs] == ["branin"] and notes
    assert harness.parse_seeds("0-2,5") == [0, 1, 2, 5]


def test_end_to_end_harness_run(tmp_path, capsys):
    out = tmp_path / "res"
    argv = ["--problems", "branin", "--noise-ratios", "0.5", "--arms", "random", "--seeds", "0", "--budget", "12",
            "--out", str(out)]  # fmt: skip
    assert harness.main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["runs"] == 2 and manifest["arms"] == ["random-on", "random-off"] and manifest["budget"] == 12
    records = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    on, off = records
    assert on["n_evals"] == off["n_evals"] == 12 and off["n_reevals"] == 0 and on["n_reevals"] > 0
    assert off["same_pick"] and on["d9_regret"] >= 0
    # determinism: the same run again gives identical records except for the volatile timing fields
    again = [harness.run_one(s) for s in harness.make_specs(["branin"], [0.5], ["random-on", "random-off"], [0], 12, None)[0]]
    for old, new in zip(records, again):
        assert {k: v for k, v in old.items() if k not in harness.VOLATILE} == {k: v for k, v in new.items() if k not in harness.VOLATILE}
    assert "regret D9" in summarise.tables(records)
