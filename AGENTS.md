# nofluke - agent instructions

nofluke is a small Python library and command line for ask/tell search over slow, noisy evaluations that run outside Python (for example one Slurm job per candidate), with a protocol that keeps lucky single runs from winning. It is **not** a new optimisation-algorithm library: algorithms come from simple built-ins or from Optuna (D1).
Goal, scope and open questions: `docs/BRIEF.md`. Scaffolding: the project-foundations skill (version in `tools/docmap.py`, `DOCMAP_VERSION`).

## Start of every session
1. `git status` (and `git pull --rebase` if there is a remote). Do not commit, discard or overwrite changes you did not make; ask.
2. Read `AGENTS.local.md` (git-ignored): this machine's interpreter, paths, limits and quirks. If it is missing, create it from `AGENTS.local.example.md` with `python tools/docmap.py local` and the user's answers.
3. Read `docs/STATUS.md`; skim the latest entry in `docs/WORKLOG.md`.
4. Before changing behavior covered by a decision, find it with `python tools/docmap.py find <topic>` and read it with `python tools/docmap.py get D<n>`. Never read `docs/DECISIONS.md` whole: it only grows. Likewise `docs/RESULTS.md`.

## Finding your way
1. `docs/MAP.md` (generated): every folder and code file with a one-line summary.
2. The folder's `README.md`: purpose, folder rules, and a generated table of its files and who imports them.
3. The file's header docstring; `python tools/docmap.py show <path>` lists its symbols, signatures and importers.
4. Before changing a function or class: `python tools/docmap.py where <Name>` (definition, importers, every use); run the tests it lists. `calls <Name>` shows its callers and callees, `deps <folder>` how modules depend on each other, `map --budget 2000` the repo's most-used symbols.
5. File formats (study config, state folder, journal events, CLI JSON): `docs/CONTRACTS.md`. Design and boundaries: `docs/ARCHITECTURE.md`.

## Environment and commands
- `python` means the interpreter named by `python` in `AGENTS.local.md` (a Python 3.12 environment; on a new machine: `conda create -p .venv python=3.12` or `python3.12 -m venv .venv`); never the base interpreter.
- Install: `python -m pip install -e '.[dev]'` (add `,optuna` for the Optuna backend). Tests (offline, fast): `python -m pytest`. Docs check: `python tools/docmap.py check`.
- Limits: stay within `memory_budget` (AGENTS.local.md); ask before running benchmark sweeps longer than a few minutes. nofluke itself never submits cluster jobs.

## Boundaries
- Read-only: `docs/requests/` (wish lists written by client projects; answer them in DECISIONS, do not edit them). Never commit: study state folders (`runs/`), benchmark outputs, environments (in `.gitignore`; never `git add -f`).
- Tracked files name AGENTS.local.md keys, never their values: no absolute paths, user or host names (`docmap.py check` flags home paths and the local file's paths).
- Import boundaries are enforced by `docmap.py check` (`[tool.docmap.forbid]` in `pyproject.toml`; listed in `docs/MAP.md`): the core imports only numpy and the standard library; Optuna only inside `nofluke.strategies.optuna_backend`.
- nofluke never knows about a client project (no simple_complex_v1, TwinProp, torch, NEURON) and never handles credentials, ssh or cluster access: those stay in the calling project.

## Conventions
- **Design for what will vary next.** Before building something that will keep being used (a config schema, a file format, a function's signature, a report layout), ask what it will have to hold next: new parameter types, strategies, event kinds. Make it handle them from the data rather than from a hand-written list of today's cases; anything a special case does not list must fail loudly, never vanish.
- **One id, one meaning.** D decisions and R results are numbered once project-wide; ids local to one folder carry that folder's short tag when cited from outside it.
- **Determinism is a contract.** All randomness comes from the study seed through `nofluke.rng`; never `np.random.*` globals, `random`, time or dict/set iteration order. A change that alters which candidates a given (config, seed, tells) produces is a breaking change: say so in CHANGELOG and bump the journal format version if old journals would replay differently.
- **The journal is append-only.** Never rewrite or truncate `journal.jsonl`; corrections are new events.
- Public API and CLI changes need tests through the public surface (`nofluke.Study` or `nofluke.cli.main`), not only unit tests of internals.

## Keeping the project maintained
Update in the same commit as the change that requires it.

| When you… | Update |
|---|---|
| add a code file | its header docstring: first line = what it is for (shown in the map), then how to use it |
| add a folder with code | its `README.md`: a purpose paragraph whose first sentence stands alone |
| change what a file or folder is for | that first line / paragraph |
| add, move, rename or delete files | run `python tools/docmap.py update` |
| learn a fact about this machine (path, interpreter, limit, quirk) | `AGENTS.local.md` only; if every machine needs that key, add it to `AGENTS.local.example.md` |
| finish a unit of work | `docs/STATUS.md` (overwrite) and a new entry at the top of `docs/WORKLOG.md` with commands and results (`python tools/docmap.py worklog "TITLE" body.md --status` inserts it dated, rotates old months and sets STATUS's date) |
| choose between alternatives, or fill a gap | the next D entry in `docs/DECISIONS.md`; to reverse one, a new entry that supersedes it |
| add or change a module boundary | `docs/ARCHITECTURE.md` and `[tool.docmap.forbid]` |
| change a file format, event kind or CLI JSON shape | `docs/CONTRACTS.md` (and the journal `format` version if old files would read differently) |
| change user-visible behavior, CLI or config | `README.md` and `CHANGELOG.md` |
| run a benchmark | `python tools/docmap.py runs`; a numbered section in `docs/RESULTS.md` citing the run id |
| start a new line of work (multi-session) | a plan in `docs/plans/<slug>.md` with a Status line, and a row in STATUS's Workstreams table |
| write or substantially revise a doc in `docs/` | its Status line: `current (verified <today>)`, `proposal (<today>)`, `superseded by <id>` or `historical`; a decided proposal is superseded by its D entry |
| supersede a decision | fully: the old entry's status says `superseded by D<n>`; partly: keep it accepted and add `partly superseded by D<n>: <which part>`. Fix the live docs that `check` then lists |
| find that this scaffolding itself (a template rule, the docs layout, a `docmap.py` check or command) misleads you, is missing something or costs too much | nothing here: send it to the skill that installed it, `python3 "$(cat ~/.config/strange-loop/genome)/plugins/strange-loop/skills/strange-loop/scripts/loop.py" note project-foundations <kind> "<what happened, what would have helped>" --context nofluke` (if that fails and the person takes part in improving their skills, an `Observations for project-foundations` block in your reply) |
| reach a milestone (a release, a phase closed) | a docs audit: `python tools/docmap.py docs`, re-read the oldest or affected docs against code, DECISIONS and RESULTS, fix them and re-date their Status lines |

- Before every commit: `python tools/docmap.py update`, then `python tools/docmap.py check` (CI runs it too). When `check` warns about old WORKLOG months, run `python tools/docmap.py rotate`.
- Never describe planned or partial work as done; say how it was verified.
- Keep this file short; add a rule only when an agent got something wrong.

## Git
- One logical, tested change per commit: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), summary ≤72 chars, body says why; cite D/R ids.
- Before committing: tests and docs check green, `git diff --check` clean, staged diff reviewed for never-commit files.
- After committing: `git push origin main` (run unpiped: the SSH key asks for a passphrase; never ask for or store it). If you can't push, say so.
- Never force-push, rewrite pushed history or amend pushed commits. Ask before anything destructive.
- A separate worktree (`git worktree add ../nofluke_<topic> -b <branch>`) is for parallel work only. Removing it is part of the merge: right after merging, run `python tools/docmap.py worktrees --remove` (removes clean merged worktrees and their branches; keeps any with uncommitted changes).
