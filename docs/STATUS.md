# Status
Updated 2026-11-17. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md`.

## Workstreams
| Workstream | Where | State | Next |
|---|---|---|---|
| Core library + CLI | `docs/plans/mvp.md` (done) | released on main, CI green on 3 OS | tag 0.1.0 after the first client search |
| Noise protocol tuning | R1, D8, D9, D14 | first benchmark done | try a lower re-evaluation share when noise is small |
| Agent skill | `skills/nofluke-search/SKILL.md` | first version | test it in simple_complex_v1 |
| simple_complex_v1 first search | `docs/requests/simple_complex_v1.md`, D6, `docs/guides/cluster.md` | ready to start (client side) | client writes its sbatch wrapper around ask/tell |

## Works (verified)
| Capability | How verified |
|---|---|
| Library + CLI + tpe/gp strategies | `python -m pytest` → 225 passed locally; GitHub CI green on ubuntu/macos/windows × Python 3.11-3.13 (gp tests skipped in CI), 2026-11-17 |
| Full file-driven loop with failures and replay check | manual CLI session, budget 48, 2026-11-17 (WORKLOG) |
| Benchmark harness and full sweep | run 20261117T163106Z, 1,840 runs, 0 errors (R1), 2026-11-17 |
| Docs and import boundaries | `python tools/docmap.py check` → 0 errors, 2026-11-17 |

## Doesn't work yet / known issues
- D14 seed-effect correction shows no measurable effect in R1 (unresolved; kept as default).
- GP models only unconditional parameters; GP reproducibility holds per environment (D13).
- CSV export writes booleans as `True`/`False` (Python spelling).
- Lock files on network file systems untested.

## Next actions
1. Hand simple_complex_v1 the cluster guide and the skill; its first 128-evaluation search is the acceptance test.

## Blockers
- None.
