Follow [AGENTS.md](../AGENTS.md) at the repository root; its rules are mandatory. Read it and `AGENTS.local.md` (this machine), then `docs/STATUS.md` and `docs/MAP.md`, before any change. Most important if it is not loaded:
- Run `python -m pytest` and `python tools/docmap.py check` before committing; never commit study state folders (`runs/`), benchmark outputs or environments.
- All randomness comes from the study seed through `nofluke.rng`; never change which candidates a (config, seed, tells) produces without a CHANGELOG entry. The journal is append-only.
- Never force-push, rewrite pushed history or amend pushed commits.
