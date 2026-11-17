# Architecture
Status: current (verified 2026-11-17). Only what rarely changes; the file-level map is generated in docs/MAP.md.

## Overview
```
study.yaml ──init──▶ <dir>/journal.jsonl ◀── append-only events (D5)
                          │ replay
                          ▼
                    StudyState (pure)  ──▶  planner: retries → re-evaluations (noise, D8) → new candidates
                                                               │                               │
                                                               │                strategy.propose(space, history, rng_k)
                                                               ▼                               ▼
                                     ask event appended ──▶ evals printed / returned ──▶ client runs them anywhere
                                                                                               │
                                     tell / abandon events appended ◀──────────────────────────┘
```
Layers, bottom up: `rng` (seed derivation) and `space` (parameters, encoding) → `config` (validation, defaults,
hashes) → `journal` (file I/O, lock) and `state` (replay into candidates and evaluations) → `noise` (statistics,
selection, re-evaluation choice) and `strategies` (proposals) → `planner` (one ask, pure) → `study` (the Python
API: lock, replay, plan, append) → `runner` (local function / subprocess evaluators) and `cli`.

## Boundaries and invariants
- The core imports only numpy and the standard library; Optuna only in `nofluke.strategies.optuna_backend`,
  imported lazily when a study uses `tpe`/`gp` (D3) - enforced by `[tool.docmap.forbid]`.
- No client code, ML frameworks, simulators or cluster clients (D2) - enforced by `[tool.docmap.forbid]`.
- The journal is the only state; anything else is derived by replay (D5) - tests in `tests/test_study.py`.
- Planning is a pure function of (normalised config, replayed state): no I/O, clocks or global RNG (D7) - tests in
  `tests/test_determinism.py` and `nofluke verify`.
- Every random draw comes from `nofluke.rng.derive(seed, purpose, *indices)` (D7).

## Cross-cutting concerns
- **Errors:** user mistakes raise `nofluke.NoflukeError` with a message naming the field or id; the CLI prints it and
  exits with status 2. Bugs propagate.
- **Concurrency:** every mutating operation takes the folder lock, replays the journal, appends, releases. Several
  processes (a login-node loop and a result collector) may therefore share one study.
- **Units:** values are stored in the user's units; `noise` and strategies work in "higher is better" units.
- **Provenance:** the `study` event records versions, commit and host; each evaluation keeps its seed, replicate,
  kind, attempt and ask/tell times; `info` carries whatever the evaluator reports.
