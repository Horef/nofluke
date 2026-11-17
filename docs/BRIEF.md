# Brief
Status: current (verified 2026-11-17)

## Goal
nofluke lets any project, in any language, run a black-box search over parameters whose evaluation is slow, noisy and
happens outside Python (a Slurm job, a simulator run, a lab script). A project declares a search space in a file,
asks for a batch of candidates, runs them however it likes, and tells the scores back from files; nofluke keeps an
append-only, resumable, deterministic record and decides what to try next. Its distinctive part is the noise
protocol: evaluation seeds handed to the evaluator, re-evaluation of the leading candidates, and selection by a
lower confidence bound, so that the reported winner is not a fluke. Success means a client project can run a
100-200-evaluation search on a cluster with no Python glue beyond its own job submission, and trust the winner.

## Non-goals
- New optimisation algorithms. Strategies are simple built-ins (random, Latin hypercube) or wrapped from Optuna; a
  small evolutionary strategy for binary/subset spaces may come later (D1, D3).
- Running jobs. nofluke never submits cluster jobs, holds credentials or talks ssh; the calling project does (D2).
- Knowing about any client project (simple_complex_v1, TwinProp, torch, NEURON); adapters live in the clients (D2).
- Large-budget optimisation of cheap functions (thousands of evaluations per second): use pycma, Nevergrad, evosax.
- A dashboard. `nofluke status` / `best` / `export` and the JSONL journal are the interface.

## Users / stakeholders
Sergiy Horef (owner; approves scope, releases and pushes). First client: `simple_complex_v1` (TwinProp
hyperparameter search on the ELSC Slurm cluster); its wish list is `docs/requests/simple_complex_v1.md`, answered
point by point in D6. Coding agents in other repositories, through a planned agent skill.

## Success criteria
- `nofluke init / ask / tell / status / best` drive a full search from files, with failures and out-of-order
  results, and `pip install -e .` needs only numpy (YAML and Optuna optional).
- Replaying the same config, seed and tells reproduces the same candidates exactly (tested).
- Killing the process at any point and resuming loses at most the in-flight command (tested).
- The noise protocol re-evaluates leaders and reports mean, n and a lower confidence bound; on noisy benchmark
  functions at 100-200 evaluations it selects a better true optimum than "best single observation" (benchmarked).
- Benchmarks compare every strategy with random search at an equal budget over several seeds (docs/RESULTS.md).
- simple_complex_v1 runs its first real search (128 evaluations) through the CLI (acceptance test, run there).

## Constraints
Python ≥ 3.11 (developed on 3.12); macOS, Linux and Windows (the owner also works on Windows); numpy is the only
required dependency. State folders may live on network file systems shared with a cluster login node.

## Open questions (for the owner)
- [ ] Public or private GitHub repository `Horef/nofluke`, and publish to PyPI once stable? - decides licence
  visibility and whether the name is reserved now.
- [ ] Should the agent skill ("how to set up a nofluke search in any repo") live in the claude-plugins marketplace
  repo or in this repo? - decides where it is versioned.
