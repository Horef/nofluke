# Status
Updated 2026-11-17. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md`.

## Workstreams
| Workstream | Where | State | Next |
|---|---|---|---|
| Core MVP (CLI, journal, noise protocol) | `docs/plans/mvp.md` | implemented and reviewed | close the plan after the first push |
| Optuna backend (TPE, GP) | D3, D11 | designed | implement the Optuna backend module |
| Benchmark harness | D11, `docs/BRIEF.md` | designed | noisy functions, random vs LHS vs TPE vs noise-off |
| Agent skill | D12 | not started | `skills/` after the backend |
| simple_complex_v1 first search | `docs/requests/simple_complex_v1.md`, D6 | can start | hand over CLI usage notes |

## Works (verified)
| Capability | How verified |
|---|---|
| Library + CLI (init/ask/tell/pending/abandon/status/best/export/done/verify/run) | `python -m pytest` → 191 passed, 2026-11-17 |
| Full file-driven loop with failures, re-evaluation, replay check | manual CLI session, budget 48, mock noisy evaluator (SD 0.03): done in 13 rounds, 18 re-evaluations, 2 failures retried, `verify` ok, 2026-11-17 |
| Docs and import boundaries | `python tools/docmap.py check` → 0 errors, 2026-11-17 |

## Doesn't work yet / known issues
- `tpe` / `gp` strategies are rejected at `init` until the Optuna backend exists.
- D8/D9 defaults (re-evaluation share 1/3, z 1.5, cap 6) are not yet benchmarked.
- CSV export writes booleans as `True`/`False` (Python spelling).
- CI never run (no GitHub remote yet); lock files on NFS untested.

## Next actions
1. Create the public GitHub repository `Horef/nofluke` and push - owner confirmation.

## Blockers
- None.
