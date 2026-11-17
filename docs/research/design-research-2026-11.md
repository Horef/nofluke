# Design research: noise, small budgets, determinism (2026-11)
Status: current (verified 2026-11-17)

State-of-the-art notes behind nofluke's noise protocol, selection rule, seed handling, strategy defaults and
determinism design (D8-D11). Compiled by a research agent on 2026-11-17; items marked UNVERIFIED were from memory,
not from a source fetched that day. Labels: established / plausible / unresolved.

## 1. Allocating re-evaluations (budget ~128, noise ≈ effect)
- Separating two candidates with true gap d and per-evaluation SD σ at one-sided z needs n ≈ 2(zσ/d)² each:
  d = σ, z = 1.64 → n ≈ 5.4; d = σ/2 → n ≈ 22. At B = 128 only gaps of about 1σ or more can be resolved, and only
  for a handful of finalists.
- **F-race / irace** (López-Ibáñez et al. 2016, Oper. Res. Perspectives): survivors evaluated on shared seed blocks,
  Friedman/t-test elimination after ~5 blocks (T_first = 5, UNVERIFIED). Too costly early at B = 128.
- **Hoeffding / Bernstein races** (Maron & Moore 1994; Mnih et al. 2008, UNVERIFIED): ε(n) = R√(ln(2/δ)/2n) ≈ 0.6
  at n = 5 for accuracy - nothing is ever eliminated. Rejected.
- **SMAC intensification** (Lindauer et al. 2022, JMLR): challengers run on the incumbent's (instance, seed) pairs,
  rejected when worse; sequential and stateful.
  https://automl.github.io/SMAC3/main/_modules/smac/intensifier/intensifier.html
- **OCBA** (Chen et al. 2000, UNVERIFIED): N_i ∝ (σ_i/δ_bi)², N_b = σ_b √Σ(N_i/σ_i)². Stateless (recompute
  targets from history, fill the largest deficit), but the asymptotic rule is poor at tiny budgets (LaPorte &
  Branke, WSC 2012, https://www.informs-sim.org/wsc12papers/includes/files/con371.pdf).
- **Knowledge gradient** (Frazier, Powell & Dayanik 2008); Pearce, Poloczek & Branke 2022 (Oper. Res.,
  arXiv 1910.09259) trade off reused vs fresh seeds. More complex than OCBA for modest gain.
- **Fixed-budget best-arm identification**: Successive Rejects (Audibert et al. 2010), Successive Halving (Karnin
  et al. 2013), LUCB (Kalyanakrishnan et al. 2012) (all UNVERIFIED). SR/SH are batch-synchronous; LUCB is
  incremental and fits out-of-order results.
- **Noisy evolutionary optimisation**: Jin & Branke 2005 (IEEE TEVC 9(3)); Rakshit, Konar & Das 2017 (Swarm Evol.
  Comput. 33). Explicit averaging (fixed/adaptive resampling), implicit averaging (bigger populations), modified
  selection, surrogates.
- **Recommended (plausible):** a rolling, stateless LUCB/OCBA-flavoured policy: ~1/3 of evaluations are
  re-evaluations; each ask fills its re-evaluation share from the plausible-best set S = {i : UCB_i ≥ max_j LCB_j}
  (z ≈ 1.5), smallest n_i first, ties by highest mean, cap n_i ≤ 6-8; reserve the last 5-10 % for finalists.

## 2. Final selection under noise
- Winner's curse: for K equal candidates the expected max exceeds the mean by σ·E[max of K normals] ≈ 2.65σ at
  K = 128 (≈ +0.08 accuracy at σ = 0.03) - established (order statistics).
- Overtuning (Schneider, Bischl & Feurer 2025, AutoML Conf, arXiv 2506.19540): mostly mild, severe in ~10 % of runs;
  grows with budget and small data; reshuffling splits helps; recommend the posterior mean, not the empirical best,
  as incumbent - established.
- **Default best (plausible):** empirical-Bayes shrunk mean μ̂_i = m + B_i(ȳ_i − m), B_i = τ²/(τ² + σ²/n_i), with m
  and τ² = max(0, Var(ȳ) − mean(σ²/n_i)) by method of moments over candidates; posterior SD SE_i = √(B_i σ²/n_i);
  report ȳ, n, μ̂ and LCB = μ̂ − z·SE.
- Variance with n = 1: pool σ² over candidates with n ≥ 2; per-candidate s_i² shrunk toward it with prior weight
  ν0 ≈ 4: (ν0 s_p² + (n_i − 1)s_i²)/(ν0 + n_i − 1). Until enough replicates exist, use a user-given prior SD.
- Validation-set sampling error (≈ √(p(1−p)/N_val)) is shared by all seeds and candidates: re-seeding does not
  reduce it, hence the untouched test set - established.

## 3. Common random numbers and seeds
- F-race, irace and SMAC compare on shared instance-seed blocks (CRN). Pearce et al. formalise reuse vs fresh seeds.
- No source tests CRN for neural-network HPO; seed effects may decorrelate across hyperparameters - unresolved,
  measure the paired correlation from data.
- **Recommended (plausible):** pass an explicit evaluation seed; seed of replicate r = f(master seed, r), shared by
  all candidates (paired data for free; each replicate of one candidate still gets a new seed); log it.

## 4. Strategies at 100-200 evaluations in mixed spaces
- carps (Benjamins et al. 2025, arXiv 2506.06143; 3,336 tasks incl. HPOBench, YAHPO, BBOB): black-box subset mean
  ranks SMAC3-BB 1.50, Nevergrad CMA-ES 1.82, random 2.67; rankings settle only after 60-80 % of the budget.
- Koster & Sigrist 2026 (arXiv 2602.05786, 59 tabular datasets): SMAC best on average vs random, TPE, GP-BO,
  Hyperband; tuning typically needed > 100 trials.
- Optuna GPSampler: Matérn-5/2 ARD; 2nd to Vizier on BBOB in the 4.4 release notes; no GP-vs-TPE mixed-space
  benchmark found (UNVERIFIED). OptunaHub CMA-ES page advises TPE with categoricals.
- **Recommended (plausible):** LHS start of ~10-20 points, then Optuna TPE (multivariate, constant liar); GP opt-in
  for continuous spaces under ~15 dimensions. Any claim must beat LHS/random with the same budget and the same
  re-evaluation and selection policy - established.

## 5. Batch / asynchronous proposal
- Constant liar (Ginsbourger et al. 2010) is the simplest; Optuna TPE `constant_liar` penalises running trials.
  Kriging believer needs a GP refit per slot; local penalisation is GP-only; Thompson sampling (Kandasamy et al.
  2018) is per-slot independent - cleanest for GP. At batch ≤ 8 of 128 the differences are small.

## 6. Deterministic ask/tell
- Optuna's sampler RNG is a sequential stream; parallel runs are documented as non-deterministic (FAQ).
- **Recommended (established practice for RNG derivation; log design plausible):** rng_k =
  default_rng(SeedSequence([seed, k])) with k the candidate counter; proposal = f(seed, k, H_k) where H_k is the set
  of completed tells visible at ask time, sorted by id; record which tells each ask saw; rebuild Optuna studies per
  ask with sampler seed derived from (seed, k).

## 7. Binary/subset spaces (later)
- PBIL / UMDA / compact GA average noise implicitly; λ = 8-16 for ~15 generations ≈ 200 evaluations.
- BOCS (Baptista & Poloczek 2018, arXiv 1806.08838), COMBO (Oh et al. 2019), scalable BOCS (Deshwal et al. 2020,
  arXiv 2008.08177). At a few hundred noisy evaluations, main effects only: a sparse linear model on 0/1 indicators
  with Thompson sampling is a plausible first step - unresolved.

## 8. Multi-fidelity under noise
- Successive halving is error-prone with noisy, crossing learning curves (Egelé et al. 2023, arXiv 2307.15422). At
  noise ≈ effect a η = 3 cut often drops the true best (judgement). Defer; have evaluators log per-epoch curves so
  rank correlation can be measured later - unresolved.
