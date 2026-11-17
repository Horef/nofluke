# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier months: `docs/worklog/YYYY-MM.md`.

## 2026-11-17 - Core MVP implemented and reviewed

Changed: Implemented the core MVP to `docs/plans/mvp.md` (implementation agent), then reviewed it. Review fixes:
`is_done` no longer reports done while evaluations are pending after the failure limit; `optimize`/`run` collect
in-flight results before stopping at the failure limit; `ask --split` files hold the evaluation object alone, as
CONTRACTS §4 says (the test had checked the implementation's own shape). CONTRACTS and D8 clarified (event key,
order-preserving space hash, `best --json` shape, tie-break rules). Decisions: D12 (public repo, skill in `skills/`).

| Verification command | Result |
|---|---|
| `python -m pytest` | 191 passed in 4.4 s |
| `python -m pytest tests/test_runner.py` x5 | all passed (threaded failure-limit test not flaky) |
| manual CLI loop, budget 48, 6-parameter mixed space with a conditional, mock evaluator SD 0.03, eval ids divisible by 17 fail | done after 13 rounds; 18 re-evaluations; noise SD estimate 0.023 (pooled); winner n = 2; `verify` ok |
| `python tools/docmap.py check` | 0 errors |

Next: Optuna backend, then the benchmark harness.

## 2026-11-17 - Project founded: survey, decisions, scaffolding
Changed: Surveyed about 20 optimisation tools and the literature on noisy small-budget search
(`docs/research/survey-2026-11.md`, `docs/research/design-research-2026-11.md`); concluded that the old GOE should
not become another GA engine (D1) and started nofluke instead (D4; GOE frozen with a README pointer, GOE commit
4e435a4). Set up project foundations (AGENTS.md, shims, BRIEF, STATUS, DECISIONS, ARCHITECTURE, CONTRACTS, docmap,
CI) and wrote the MVP spec (`docs/plans/mvp.md`). Decisions: D1-D11.

| Verification command | Result |
|---|---|
| `python tools/docmap.py check` | 0 errors |

Next: implement the MVP to `docs/plans/mvp.md`.
