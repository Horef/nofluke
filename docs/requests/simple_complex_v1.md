# What simple_complex_v1 would like from GOE

Written 17 November 2026 by the Claude Code session working on `~/Programming/Neuroscience/simple_complex_v1` (TwinProp
stage, circle 1), for the session that revives this repository. It is a wish list from one user project, not a spec:
weigh it against what you find (other projects' needs, existing libraries) and push back where it is wrong.

## 1. Why we want it

TwinProp optimises a neuron's synaptic weights by gradient descent through a frozen DNN twin of the neuron. The
optimisation itself has knobs that change the outcome a lot (simple_complex_v1 `docs/RESULTS.md` §92-§94). One of
them is the weight of one loss term: values 1, 3, 10 and 30 gave held-out accuracy 0.726, 0.739, 0.772 and 0.750.
The learning rate and the number of epochs decide whether a run overfits. Tuning these by hand is slow and easy to
bias. We want a search over them, and later possibly over discrete structure (which dendritic sites get synapses)
scored directly in the NEURON simulator, where gradients do not exist.

The shape of our problem, which is probably typical for scientific users:

- **Expensive evaluations.** One evaluation is a GPU training job on a Slurm cluster (ELSC): 4-10 minutes, with about
  4 running at once. A realistic budget is 100-200 evaluations in total, not 10,000 (the current defaults of
  100 agents x 100 generations).
- **Noisy fitness.** Validation accuracy on 132-308 held-out items has a standard error of about 0.03, the same size
  as the differences a search chases. A different training seed moves it by an unknown amount (we are measuring
  that now). Selection on a single noisy evaluation rewards luck.
- **Evaluations happen outside Python.** A candidate becomes a config file and a `sbatch` job; its fitness arrives
  minutes later from a log or JSON file. Jobs fail (node problems, a guard that stops a run whose code checkout
  changed) and finish out of order.
- **Selection and reporting must be separate.** We choose on a validation set and report on a test set the search
  never saw. The final winners are then checked in NEURON, a far more expensive evaluation.

## 2. What is wrong or missing in the current code (from reading `GeneticOptimizationEngine.py`, Oct 2024)

- `run_evolution()` owns the loop and evaluates every agent itself (local processes via `DillProcess`). There is no
  way to hand out a batch and take its results back later.
- Line ~214: `self.variability = min(self.variability, np.std(self.fitness))` compares a mutation scale (parameter
  units) with a fitness spread (fitness units). With accuracies near 0.7 this collapses mutation to about 0.03 times
  each value after the first generation.
- No elitism: `_reproduce` replaces the whole population, so the best agent can be lost (`best_agent` is remembered,
  but no longer bred from).
- Default mutation is normal with sd = variability x value. It cannot move a parameter that is exactly 0, and it is
  wrong for parameters that live on a log scale (learning rates, loss weights).
- `if seed:` skips seed 0, and `np.random.seed` changes the global generator of the host program. Use an
  engine-owned `np.random.Generator`.
- No persistence or resume. The text log (`goe_log.txt`, 3.4 MB in the repo folder) is the only record and is not
  machine readable.
- No packaging (no `pyproject.toml`), no tests beyond `testing.py`, no CI, and `dill` is a hard import.

## 3. Must have

1. **Ask/tell interface.** For example `engine.ask(n) -> list[Candidate]` (stable id, params, generation) and
   `engine.tell(id, fitness, info=None)`, plus `tell_failed(id, reason)`. `run()` with a local function becomes a thin
   wrapper over these. It must handle partial generations, out-of-order results, failures (retry once, then mark),
   and optionally a steady-state mode, where a new candidate is created whenever one finishes instead of waiting for
   a whole generation.
2. **A declared, serialisable search space.** Float and int with bounds and `log=True`, choice, bool, and ideally
   conditional parameters (a parameter that exists only when another has some value). It should load from JSON or YAML
   so a project keeps it in its configs and commits it.
3. **Persistence and resume.** The state goes to disk after every tell (JSON lines or SQLite), and the engine resumes
   from it after a crash. The same seed and the same tells give the same candidates.
4. **Handling noisy fitness.**
   - Elitism.
   - Re-evaluation of the best candidates with new seeds, so the engine aggregates several evaluations of one
     candidate (mean and n, optionally a lower confidence bound for selection).
   - An option to pass the evaluation seed to the evaluator as part of the candidate.
5. **Proper mutation operators.**
   - Log-space Gaussian for log parameters.
   - Bounded operators (reflection or clipping) instead of after-the-fact projections.
   - A mutation scale that adapts to the parameter's range, not to fitness.
6. **A machine-readable history.** One row per evaluation: candidate id, generation, params, seed, fitness, status,
   wall time, evaluator info. Plus a provenance header: GOE version and commit, the engine seed, and a hash of the
   search space.
7. **Packaging.**
   - A `pyproject.toml` that installs with `pip install -e`.
   - numpy as the only required dependency; dill optional.
   - Python 3.12, type hints, pytest, CI.
   - A CHANGELOG and semantic versions, so a project can pin a version.

## 4. Connections so other projects can use it

- **A command-line interface for evaluators outside Python** (what we need first):
  - `goe init --space space.yaml --seed 0 --state run_dir/` starts a search;
  - `goe ask --n 16 > batch.json` hands out a batch of candidates (ids and params);
  - `goe tell results.json` returns the scores;
  - `goe status` and `goe best` report progress and the winner.

  Any project, in any language, can then drive it with files: our side would turn each candidate into a run config,
  submit a Slurm array, and collect the scores.
- **Evaluator backends**, behind one small interface:
  - local function with a process pool;
  - subprocess command template (`{params_json}` in, a JSON score out);
  - Slurm.
  
  For Slurm, consider the `submitit` library rather than writing sbatch handling. Never put credentials in GOE: ssh and
  cluster access belong to the calling project (ours is key-only and must never fall back to passwords).
- **Optuna integration**, worth deciding early: GOE as an Optuna sampler (`optuna.samplers.BaseSampler`) would give
  storage, resume, pruning, a dashboard and many users for free. GOE would then be the evolutionary strategy itself.
  Compare that with keeping GOE standalone; both can coexist (a standalone core plus a thin sampler adapter).
- **Multi-fidelity search** (optional, high value for training jobs): evaluate many candidates briefly, for example
  with a few epochs, and continue only the promising ones (successive halving, ASHA, Hyperband). Also consider
  population-based training (Jaderberg et al. 2017, in Ray Tune), which evolves hyperparameters during training. It
  is the closest existing idea to "a GA over training knobs" and a natural comparison.
- **Multi-objective search** (optional, later): for example accuracy against the number of weights changed, with a
  Pareto front (NSGA-II; see pymoo).
- **An agent skill:** a small skill in Sergiy's plugin marketplace that tells coding agents how to set up a GOE search
  in any repo. It would cover:
  - declaring the space;
  - keeping a test set the search never sees;
  - estimating the budget and asking before compute;
  - re-evaluating the winners;
  - recording the search's state folder and seed in the project's run log.

## 5. Benchmarks and the literature (also what a paper would need)

- **Survey existing implementations before building more:**
  - DEAP and PyGAD (classic genetic algorithms);
  - pymoo (multi-objective);
  - Nevergrad (Meta; many gradient-free optimisers, including noisy settings);
  - CMA-ES (`cma`, also in Optuna);
  - Optuna (TPE, CMA-ES and NSGA-II samplers);
  - EvoTorch;
  - Ray Tune (population-based training, ASHA, BOHB);
  - SMAC3.

  Record what each offers for: ask/tell, noise, resume, mixed and conditional spaces, and cluster execution. Put it in
  a table in the repo. If one already does what GOE would do, say so plainly; the honest outcome may be "GOE is a
  thin, well-tested strategy on top of Optuna".
- **A benchmark harness:**
  - noisy test functions under a small budget (100-200 evaluations);
  - mixed spaces;
  - one cheap real task (for example tuning a small scikit-learn model).

  Compare GOE with random search, Sobol or Latin-hypercube sampling, TPE and CMA-ES at an equal number of
  evaluations, over several seeds. Random search with an equal budget is the baseline any claim must beat.

## 6. Our first real use (acceptance test, run from simple_complex_v1, not from GOE)

- **Space:**
  - loss-term weights on a log scale: `near_spike_binary`, `spike_event` and `output_spike_prob_well`, each in
    [0.1, 100];
  - learning rate, log scale in [1e-4, 3e-3];
  - epochs, int in [30, 200] (or as a fidelity);
  - batch size, a choice of {16, 32, 64}.
- **Fitness:** mean validation accuracy over 2 training seeds. The test-set accuracy is logged but never used for
  selection.
- **Budget:** about 128 evaluations, against a random search of the same size.
- **Evaluator:** a Slurm GPU job per candidate. The scv1 side writes the run config and submits. GOE needs only
  ask/tell, files and resume.

We can do our side as soon as the command-line ask/tell exists and is tested. Everything else in §3-§5 makes GOE
useful beyond us.

## 7. Boundaries

GOE must not import or know about simple_complex_v1, TwinProp, torch or NEURON. It stays a general library; project
adapters live in the projects.
