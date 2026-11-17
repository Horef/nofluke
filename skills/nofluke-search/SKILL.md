---
name: nofluke-search
description: Set up and run a nofluke search in any repository - a black-box search over parameters whose evaluation is slow, noisy or runs outside Python (training jobs, simulations, cluster runs). Covers declaring the search space, keeping a test set the search never sees, estimating the budget and asking before compute, wiring the evaluator (local function, shell command, or Slurm via files), re-evaluating winners, and recording the search in the project's run log. Use when asked to tune hyperparameters or other knobs with nofluke, to start, resume or inspect a nofluke study, or to choose parameters by search when each evaluation costs minutes or more.
---

# Running a nofluke search

nofluke hands out candidates (params + an evaluation seed), takes scores back, and keeps an append-only journal in a
study folder. It re-evaluates plausible winners and ranks by a shrunk mean with a lower bound, so the pick is not the
luckiest single run. Repository: https://github.com/Horef/nofluke (formats: `docs/CONTRACTS.md`; cluster pattern:
`docs/guides/cluster.md`).

## 1. Frame the search (ask the person before anything runs)
1. **What is scored, and on what data.** `value` must be a validation score. Identify a test set the search will
   never see; if none exists, say so and propose one before starting. Test scores may go in `info` for logging
   only - never into `value`.
2. **Noise.** Ask (or measure with 3-5 repeated runs of one config) how much the score moves between seeds. If the
   seed-to-seed SD is known, set `noise.prior_sd`.
3. **Budget.** Evaluations × time per evaluation ÷ parallel workers = wall time; also the cost per run (GPU
   hours, cluster share). Rules of thumb: below ~40 evaluations, a hand-picked grid or random search is as good;
   100-200 is the regime nofluke is built for; about a third goes to re-evaluations. **Stop and get approval for
   the budget and where it runs** before the first `ask`.
4. **Space.** Prefer few parameters (≤ 6-8 at 100-200 evaluations). Log scale for rates, weights and anything
   spanning decades; `choice` for discrete options; `when` for parameters that exist only under another's value.

## 2. Create the study
```yaml
# configs/search-<name>.yaml (commit it)
name: <name>
direction: maximize            # or minimize
metric: val_acc
budget: 128
seed: 0
space:
  lr:     {type: float, low: 1.0e-4, high: 3.0e-3, log: true}
  epochs: {type: int, low: 30, high: 200}
  batch:  {type: choice, values: [16, 32, 64]}
strategy: {name: tpe}          # random (numpy only) | tpe (pip install "nofluke[optuna]") | gp (continuous)
noise: {prior_sd: 0.03}        # optional; other defaults are documented in CONTRACTS §1
```
```bash
pip install "nofluke[optuna,yaml] @ git+https://github.com/Horef/nofluke"
nofluke init configs/search-<name>.yaml --dir <state folder>
```
Keep the state folder out of git (it is data) but record its path, the config file and the seed in the project's
run log or worklog.

## 3. Wire the evaluator
- **Python function, local:** `nofluke.optimize(study, lambda params, seed: score, n_workers=k)`.
- **Shell command, local:** `nofluke run --dir S --cmd 'python train.py --params {params_file} --out {result_file}' -j 4`;
  the command writes `{"value": x, "info": {...}}` to `{result_file}`.
- **Cluster or anything else:** the file loop - `nofluke ask --dir S -n K --split DIR` (one `eval-<id>.json` with
  params and seed per job), submit the jobs with the project's own tooling, then `nofluke tell --dir S results/*.json`
  with records `{"eval": id, "value": x, "info": {...}}` or `{"eval": id, "status": "failed", "reason": "..."}`.
  Repeat until `nofluke done --dir S` exits 0. nofluke never submits jobs or touches credentials.
- Always pass the evaluation's `seed` into the run (training seed, and the data split if the project varies it).

## 4. While it runs
- `nofluke status --dir S`; `nofluke pending --dir S` re-prints outstanding evaluations; abandon lost jobs with
  `nofluke abandon --dir S --older-than 2h`.
- Never edit `journal.jsonl` by hand. `nofluke verify --dir S` checks it replays exactly.

## 5. Finish
1. `nofluke best --dir S --top 5 --json`. Prefer a pick with n ≥ 2; report `shrunk_mean`, `n`, `lcb` and
   `n_candidates` (how many candidates the pick was made over), not the best single value.
2. Evaluate the top 1-3 picks **once** on the held-out test set, outside nofluke, and report those numbers as the
   result. Do not go back and search again on test results.
3. Compare with a baseline: the default configuration, and if the claim is "the search helped", random search at
   the same budget.
4. Record in the project's log: config path, state folder, seed, budget used, the pick, its validation statistics and
   its test score, and `nofluke export --format csv` if the project keeps run tables.
