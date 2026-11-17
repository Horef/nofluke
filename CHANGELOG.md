# Changelog
All notable changes to nofluke. Format: Keep a Changelog; versions: semantic versioning. A change in which
candidates a given (config, seed, tells) produces is a breaking change.

## [Unreleased]
### Added
- Project scaffolding, design decisions D1-D12, file-format contracts (`docs/CONTRACTS.md`).
- `nofluke.Study` (create/open, ask, tell, tell_failed, tell_many, abandon, pending, best, status, export_rows,
  is_done, verify) over an append-only, locked JSONL journal with deterministic replay.
- Search space: float/int (optionally log), choice, bool, conditional parameters (`when`); JSON or YAML configs.
- Strategies: Latin-hypercube start, then random.
- Noise protocol: evaluation seeds shared per replicate, rolling re-evaluation of the plausible-best set,
  empirical-Bayes shrunk-mean ranking with lower confidence bounds.
- Local evaluators: `optimize` (Python function, threads or processes) and `run_command` (shell template).
- CLI: `nofluke init / ask / tell / pending / abandon / status / best / export / done / verify / run`.
