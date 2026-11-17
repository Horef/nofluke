"""Noise statistics: shrunk means, confidence bounds, ranking, plausible-best set and re-evaluation choice (D8, D9).

All numbers are in "higher is better" units (values are negated for `minimize`; the caller converts back). With shared
evaluation seeds the values are first corrected for an additive effect per replicate seed (D14).
`estimate(state, config)` builds a `NoiseModel` with one `CandidateStats` per candidate that has a completed value;
`rank`, `plausible_best` and `choose_reeval` are pure functions of that model. After the seed-effect correction, replicates are treated as independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from nofluke.state import OK, StudyState


@dataclass(frozen=True)
class CandidateStats:
    """Summary of one candidate's completed values (higher is better)."""

    candidate: int
    n: int
    mean: float  # after the seed-effect correction (D14); equals raw_mean when there is none
    raw_mean: float
    sd: float | None  # sample SD, None for n = 1
    sigma_i: float  # per-evaluation noise SD estimate (D9)
    shrunk_mean: float
    se: float  # posterior standard error
    lcb: float
    ucb: float


@dataclass(frozen=True)
class NoiseModel:
    """Per-candidate statistics plus the population-level quantities they were computed from.

    `sigma_source` is "pooled", "prior_sd", "spread" (variance of the means, D9 fallback), "own" (a single
    candidate's sample SD) or "none". `tie_break` is True when tau^2 = 0, so all shrunk means are equal and the
    secondary ranking key decides.
    """

    stats: dict[int, CandidateStats]
    z: float
    sigma: float
    sigma_source: str
    tie_break: bool
    seed_effects: dict[int, float] = field(default_factory=dict)


def _scores(state: StudyState, sign: float, adjust: bool) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Completed values per candidate in eval-id order, for candidates with at least one, plus seed effects.

    With `adjust` (shared seeds, D14) the values are corrected for an additive effect per replicate seed, so that
    candidates measured on different sets of seeds compare fairly; the effects are returned by replicate.
    """
    observations = [(ev.candidate, ev.replicate, sign * float(ev.value))  # type: ignore[arg-type]
                    for ev in state.evals.values() if ev.status == OK]  # fmt: skip
    effects = _seed_effects(observations) if adjust else {}
    values: dict[int, list[float]] = {}
    for cid, replicate, y in observations:
        values.setdefault(cid, []).append(y - effects.get(replicate, 0.0))
    return {cid: np.array(values[cid], dtype=float) for cid in sorted(values)}, effects


def _seed_effects(observations: list[tuple[int, int, float]], iterations: int = 200) -> dict[int, float]:
    """Least-squares additive effects b_r of the model y_ir = f_i + b_r + e (D14), by alternating means.

    Anchored so that the effects average to zero over the observations, which keeps adjusted values on the scale of
    the raw ones. Returns {} when only one replicate seed has results (nothing to separate).
    """
    replicates = sorted({r for _, r, _ in observations})
    if len(replicates) < 2:
        return {}
    candidates = sorted({c for c, _, _ in observations})
    c_index = {c: k for k, c in enumerate(candidates)}
    r_index = {r: k for k, r in enumerate(replicates)}
    ci = np.array([c_index[c] for c, _, _ in observations])
    ri = np.array([r_index[r] for _, r, _ in observations])
    y = np.array([v for _, _, v in observations], dtype=float)
    n_c = np.bincount(ci, minlength=len(candidates)).astype(float)
    n_r = np.bincount(ri, minlength=len(replicates)).astype(float)
    b = np.zeros(len(replicates))
    for _ in range(iterations):
        f = np.bincount(ci, weights=y - b[ri], minlength=len(candidates)) / n_c
        new_b = np.bincount(ri, weights=y - f[ci], minlength=len(replicates)) / n_r
        new_b -= float((new_b[ri]).mean())
        if np.max(np.abs(new_b - b)) < 1e-12:
            b = new_b
            break
        b = new_b
    return {r: float(b[r_index[r]]) for r in replicates}


def estimate(state: StudyState, config: Mapping[str, Any]) -> NoiseModel:
    """Compute the D9 empirical-Bayes model from the completed evaluations of `state`."""
    noise = config["noise"]
    z, nu0, prior_sd = float(noise["z"]), float(noise["nu0"]), noise["prior_sd"]
    # Journals written before D14 have no `seed_effects` key and keep their original (unadjusted) statistics.
    adjust = noise.get("seed_effects", "ignore") == "adjust" and noise["seed_mode"] == "shared"
    sign = 1.0 if config["direction"] == "maximize" else -1.0
    scores, effects = _scores(state, sign, adjust)
    raw = _scores(state, sign, False)[0] if effects else scores
    ids = list(scores)
    if not ids:
        return NoiseModel({}, z, 0.0, "none", False)
    n = np.array([len(scores[i]) for i in ids], dtype=float)
    ybar = np.array([scores[i].mean() for i in ids])
    s2 = np.array([scores[i].var(ddof=1) if len(scores[i]) > 1 else 0.0 for i in ids])

    if len(ids) == 1:  # no population: the mean is its own estimate
        if prior_sd is not None:
            sigma, source = float(prior_sd), "prior_sd"
        elif n[0] > 1:
            sigma, source = math.sqrt(s2[0]), "own"
        else:
            sigma, source = 0.0, "none"
        sigma_i = np.array([sigma])
        shrunk, se = ybar.copy(), sigma / np.sqrt(n)
        tie_break = False
    else:
        replicated = n >= 2
        if replicated.sum() >= 3:
            pooled, source = float(((n - 1) * s2)[replicated].sum() / (n - 1)[replicated].sum()), "pooled"
        elif prior_sd is not None:
            pooled, source = float(prior_sd) ** 2, "prior_sd"
        else:
            pooled, source = float(ybar.var(ddof=1)), "spread"
        sigma = math.sqrt(pooled)
        sigma_i = np.sqrt((nu0 * pooled + (n - 1) * s2) / (nu0 + n - 1))
        noise_var = sigma_i**2 / n
        m = float(ybar.mean())
        tau2 = max(0.0, float(ybar.var(ddof=1)) - float(noise_var.mean()))
        denominator = tau2 + noise_var
        weight = np.where(denominator > 0, tau2 / np.where(denominator > 0, denominator, 1.0), 1.0)
        shrunk = m + weight * (ybar - m)
        se = np.sqrt(weight * noise_var)
        tie_break = tau2 == 0.0

    stats = {
        cid: CandidateStats(
            candidate=cid,
            n=int(n[k]),
            mean=float(ybar[k]),
            raw_mean=float(raw[cid].mean()),
            sd=math.sqrt(s2[k]) if n[k] > 1 else None,
            sigma_i=float(sigma_i[k]),
            shrunk_mean=float(shrunk[k]),
            se=float(se[k]),
            lcb=float(shrunk[k] - z * se[k]),
            ucb=float(shrunk[k] + z * se[k]),
        )
        for k, cid in enumerate(ids)
    }
    return NoiseModel(stats, z, float(sigma), source, bool(tie_break), effects)


def _rank_key(model: NoiseModel, s: CandidateStats) -> tuple[float, float, int]:
    return (-s.shrunk_mean, -(s.mean - model.z * s.sigma_i / math.sqrt(s.n)), s.candidate)


def rank(model: NoiseModel) -> list[int]:
    """Candidate ids best first: by shrunk mean, ties by mean - z*sigma_i/sqrt(n), then id (D9)."""
    return sorted(model.stats, key=lambda cid: _rank_key(model, model.stats[cid]))


def plausible_best(model: NoiseModel) -> list[int]:
    """The set S = {i : UCB_i >= max_j LCB_j} (D8), by candidate id."""
    if not model.stats:
        return []
    best_lcb = max(s.lcb for s in model.stats.values())
    return [cid for cid, s in model.stats.items() if s.ucb >= best_lcb]


def choose_reeval(model: NoiseModel, counts: Mapping[int, int], max_replicates: int) -> int | None:
    """Pick the member of S with the fewest evaluations (completed + pending) below the cap, or None.

    `counts` maps candidate id to its completed + pending evaluations. Ties go to the better-ranked candidate:
    highest shrunk mean, then the D9 secondary key, then the lowest id.
    """
    eligible = [cid for cid in plausible_best(model) if counts.get(cid, 0) < max_replicates]
    if not eligible:
        return None
    return min(eligible, key=lambda cid: (counts.get(cid, 0), _rank_key(model, model.stats[cid])))
