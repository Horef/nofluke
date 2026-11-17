# Plan: benchmark harness
Status: current (verified 2026-11-17; in progress)

## Objective
Measure, at budgets of 64-128 noisy evaluations, (a) whether the noise protocol (D8, D9) picks better true optima
than "no re-evaluation, best single observation", and (b) how the strategies (D11) compare with random search at an
equal budget. Results go to `docs/RESULTS.md` as numbered R entries with run ids; the harness must make every number
reproducible from (problem, config, seed).

## Non-goals
Large-scale benchmarking (HPOBench/YAHPO suites), multi-fidelity, wall-clock comparisons of strategies.

## Design
- Folder `benchmarks/` (not part of the installed package; may import nofluke and, optionally, scikit-learn):
  `problems.py`, `harness.py` (CLI: `python benchmarks/harness.py --problems ... --arms ... --seeds 0-19 --budget
  128 --jobs 4 --out benchmarks/results/<run_id>`), `summarise.py` (tables from a results folder), `README.md`.
- **Problems** (all maximised; each has a noise-free `true(params)` and a noisy `observe(params, seed, rng)`):
  1. `branin` (2-D continuous, negated and rescaled), 2. `hartmann6` (6-D continuous), 3. `mixed_hpo`: a synthetic
     "training run" over the shape of the first client (log-float loss weight and learning rate with optima inside
     the range, int epochs with an overfitting optimum, choice batch {16, 32, 64} with small offsets, bool `aux` with a
     conditional log-float `aux_w`), scaled like accuracy (≈ 0.6-0.8), 4. optional `sklearn_svm` (an SVM or
     gradient-boosting model on a small built-in dataset, scored by a seed-dependent train/validation split; real
     noise; runs only if scikit-learn is installed).
  Noise for 1-3: Gaussian with SD = `noise_ratio` × SD of `true` over 10,000 uniform random points (ratios 0.25,
  0.5, 1.0); the noise RNG is derived from (problem seed, eval seed, candidate) so replicates differ and the same
  evaluation seed is shared across candidates as D10 intends (common random numbers: add a seed-level offset term
  shared by all candidates with the same eval seed, plus independent noise; document the split).
- **Arms** (each a nofluke config): `random-noisefree-pick` is not an arm; arms are `random` (LHS start + random),
  `lhs-only` (n_init = budget), `tpe`, `gp` (continuous problems only), each with the noise protocol on (defaults)
  and off (`reeval_share: 0`). Selection reported both ways for every arm: nofluke's D9 pick and the best single
  observation.
- **Execution:** simulate 4 parallel workers: steady-state `ask(1)` whenever a worker frees, durations drawn from a
  seeded log-normal, so results arrive out of order like Slurm jobs. Deterministic per (problem, arm, seed).
- **Metrics per run:** true value of the D9 pick; true value of the best-single-observation pick; true value of the
  best candidate ever evaluated (search quality); optimism = reported value of the pick − its true value (for both
  selection rules); evaluations spent on re-evaluation; wall time.
- **Output:** `<out>/manifest.json` (run id, UTC time, nofluke version and git revision with dirty flag, config hash,
  arms, problems, seeds, budget) and `<out>/results.jsonl` (one line per run); `benchmarks/results/` is git-ignored
  except its README. Summary: per problem × noise ratio × arm, the median and interquartile range over seeds of
  each metric, and the paired win rate of protocol-on vs protocol-off and of each strategy vs random.
- **Compute:** a smoke configuration (2 seeds, budget 32) runs in under a minute; the full sweep is a few CPU-hours at
  most and is run only after the owner approves (AGENTS.md compute rule), within `memory_budget`.

## As built (2026-11-17)
Implemented in `benchmarks/` (see its README). Noise split 50/50 by variance between a seed-level offset shared by
all candidates evaluated with the same eval seed and independent noise. Branin and Hartmann6 are negated, not
rescaled. `sklearn_svm` (digits) uses the best true value seen as its reference optimum and ignores the noise
ratio. Arms gained `<strategy>-raw` (protocol on, no D14 correction). Smoke: 56 runs in 17 s; the full sweep
(about 1,900 runs with `-raw` arms for random and tpe) is estimated at 15-20 min wall with 4 jobs, < 1 GB.

## Verification
`python benchmarks/harness.py --smoke` finishes and `summarise.py` prints a table; unit tests for problems
(`true` optima where known: Branin, Hartmann6) in `tests/test_benchmark_problems.py`; determinism: the same run
twice gives identical `results.jsonl` apart from wall time.

## Risks
- Synthetic noise may flatter or penalise the protocol relative to real training noise; the sklearn problem is the
  check. The first client's real search (simple_complex_v1) is the final test.
