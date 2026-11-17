# nofluke

Ask/tell search over slow, noisy evaluations that run outside Python, without rewarding flukes.

You declare a search space in a file, ask nofluke for a batch of candidates, run them however you like (a Slurm job
each, a simulator, a lab script), and tell the scores back. nofluke keeps an append-only, resumable, deterministic
journal, re-evaluates the leading candidates with new seeds, and ranks them by a shrunk mean with a lower
confidence bound, so the winner it reports is not just the luckiest single run.

> Status: early development (0.1.0.dev0). The CLI, the Python API and the strategies below work and are tested;
> the noise-protocol defaults are not yet benchmarked. See `docs/STATUS.md`.

## Why
Existing optimisers (Optuna, SMAC3, Nevergrad, pymoo, ...) are excellent at proposing candidates, but none offers a
file-driven batch ask/tell for evaluators outside Python, deterministic resume, and a protocol for noisy scores
(`docs/research/survey-2026-11.md`). With noise as large as the differences you chase, the best of 128 single
evaluations is biased upwards by about 2.6 standard deviations; nofluke spends about a third of the budget
re-evaluating plausible winners instead.

## Quick start (command line)
```bash
pip install -e .            # numpy only; add [yaml] for YAML configs, [optuna] for TPE/GP strategies
nofluke init study.yaml --dir runs/s1
nofluke ask --dir runs/s1 -n 4 --split runs/s1/batch     # one JSON file per evaluation: params + seed
# ... run each evaluation with its params and seed, collect {"eval": id, "value": score} records ...
nofluke tell --dir runs/s1 results/*.json
nofluke status --dir runs/s1
nofluke best --dir runs/s1 --top 5
```

```yaml
# study.yaml
direction: maximize
budget: 128
seed: 0
space:
  lr:     {type: float, low: 1.0e-4, high: 3.0e-3, log: true}
  epochs: {type: int, low: 30, high: 200}
  batch:  {type: choice, values: [16, 32, 64]}
```

## Strategies
The first `n_init` candidates come from a Latin-hypercube design; after that `strategy.name` decides: `random`
(default, numpy only), `tpe` (Optuna TPE; `pip install "nofluke[optuna]"`; best for mixed and conditional spaces)
or `gp` (Optuna GP; `pip install "nofluke[gp]"`, needs torch; continuous spaces). Running a cluster search:
`docs/guides/cluster.md`.

## Python
```python
from nofluke import Study, optimize

study = Study.create("runs/s1", "study.yaml")
for a in study.ask(4):
    study.tell(a.eval, train_and_score(a.params, seed=a.seed))
optimize(study, lambda params, seed: train_and_score(params, seed), n_workers=4)
print(study.best(top=3))
```

## Good practice
Select on a validation score, then evaluate only the final 1-3 picks once on a test set nofluke never sees. Compare
any strategy with random search at the same budget. Formats: `docs/CONTRACTS.md`; design: `docs/ARCHITECTURE.md`.

## Licence
MIT.
