# Changelog
All notable changes to nofluke. Format: Keep a Changelog; versions: semantic versioning. A change in which
candidates a given (config, seed, tells) produces is a breaking change.

## [Unreleased]
### Added
- Project scaffolding, design decisions D1-D12, file-format contracts (`docs/CONTRACTS.md`).
- `nofluke.Study` (create/open, ask, tell, tell_failed, tell_many, abandon, pending, best, status, export_rows,
  is_done, verify) over an append-only, locked JSONL journal with deterministic replay.
- Search space: float/int (optionally log), choice, bool, conditional parameters (`when`); JSON or YAML configs.
- Strategies: Latin-hypercube start, then random, Optuna TPE (`nofluke[optuna]`) or Optuna GP (`nofluke[gp]`,
  needs torch); `strategy.options` pass sampler kwargs (D13).
- Noise protocol: evaluation seeds shared per replicate, rolling re-evaluation of the plausible-best set,
  empirical-Bayes shrunk-mean ranking with lower confidence bounds, corrected for shared seed effects (D14).
- Local evaluators: `optimize` (Python function, threads or processes) and `run_command` (shell template).
- Benchmark harness (`benchmarks/`): noisy Branin, Hartmann6, a mixed HPO-like problem and an optional
  scikit-learn task, simulated asynchronous workers, protocol on/off/raw arms, `overview.py` cross-cell tables (`nofluke[bench]`); first results in R1.
- CLI: `nofluke init / ask / tell / pending / abandon / status / best / export / done / verify / run`;
  `ask` and `pending` take `--split DIR`.
