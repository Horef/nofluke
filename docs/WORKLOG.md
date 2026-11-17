# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier months: `docs/worklog/YYYY-MM.md`.

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
