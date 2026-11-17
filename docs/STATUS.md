# Status
Updated 2026-11-17. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md`.

## Workstreams
| Workstream | Where | State | Next |
|---|---|---|---|
| Core MVP (CLI, journal, noise protocol) | `docs/plans/mvp.md` | spec written, no code yet | implement modules and tests to the spec |
| Optuna backend (TPE, GP) | D3, D11 | designed | after the MVP |
| Benchmark harness | D11, `docs/BRIEF.md` | designed | after the Optuna backend |
| simple_complex_v1 first search | `docs/requests/simple_complex_v1.md`, D6 | waiting on the MVP | hand over CLI usage notes |

## Works (verified)
| Capability | How verified |
|---|---|
| Scaffolding and docs check | `python tools/docmap.py check` → 0 errors, 2026-11-17 |

## Doesn't work yet / known issues
- No package code yet; `pip install -e .` installs an empty package.
- CI workflow added but never run (no GitHub remote yet).

## Next actions
1. Create the GitHub repository `Horef/nofluke` - needs the owner's choice of public or private (BRIEF open questions).

## Blockers
- None.
