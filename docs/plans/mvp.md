# Plan: core MVP (ask/tell CLI, journal, noise protocol)
Status: historical (done 2026-11-17)

## Objective
The smallest nofluke that simple_complex_v1 can drive from Slurm: `init / ask / tell / pending / abandon / status /
best / export / done / verify`, the Python `Study` API, random and LHS strategies, the noise protocol (D8-D10),
local evaluators, and tests for determinism and crash safety. Formats: `docs/CONTRACTS.md`. Design:
`docs/ARCHITECTURE.md`, D5, D7-D11.

## Non-goals
Optuna backend (next plan step), benchmark harness, agent skill, PyPI release, multi-objective, multi-fidelity.

## Modules (`src/nofluke/`)
Each file gets a module docstring (first line = purpose). Type hints everywhere; `from __future__ import annotations`.

1. `errors.py` - `NoflukeError(Exception)`.
2. `rng.py` - `derive(seed: int, purpose: str, *indices: int) -> np.random.Generator` using
   `np.random.SeedSequence(entropy=seed, spawn_key=(_purpose_code(purpose), *indices))` where `_purpose_code` is a
   stable int (first 8 bytes of sha256 of the purpose string, as int); `derive_int(seed, purpose, *indices) -> int`
   in [0, 2**31). Never use global numpy/random state anywhere in the package.
3. `space.py` - parameter classes `FloatParam`, `IntParam`, `ChoiceParam`, `BoolParam` (frozen dataclasses with
   `name`, `when`), `Space` (ordered). `Space.from_dict(d)`, `to_dict()`, validation per CONTRACTS §1 (raise
   `NoflukeError` naming the parameter). `active(params_so_far)` logic for `when`. Unit-cube codec:
   `decode(u: np.ndarray) -> dict` (one coordinate per parameter in declared order; inactive parameters dropped;
   float: linear or log interpolation; int: `low + floor(u*(high-low+1))` clipped, log-int:
   `floor(exp(log(low) + u*(log(high+1)-log(low))))` clipped; choice: `values[min(floor(u*k), k-1)]`; bool: `u >= 0.5`)
   and `encode(params) -> np.ndarray` (inverse, cell centre for discrete; NaN for inactive). `validate_params(params)`
   for values coming back from strategies. Output values are plain Python types (float/int/str/bool/None), never
   numpy scalars.
4. `config.py` - `load_config(path_or_dict) -> dict` (JSON; YAML if suffix .yaml/.yml, error naming the `yaml` extra
   if PyYAML is missing), `normalise(dict) -> dict` filling defaults (CONTRACTS §1), rejecting unknown keys,
   validating ranges; `config_hash` / `space_hash` = sha256 hex of `json.dumps(obj, sort_keys=True,
   separators=(",", ":"))`.
5. `journal.py` - `Journal(dir)`: `exists()`, `create(study_event)`, `read() -> list[dict]` (validates `format == 1`;
   raises on a malformed line that is not the last one; an incomplete last line without `\n` is ignored on read),
   `append(events: list[dict])` (moves an incomplete last line to `journal.torn-<time>.jsonl` first, writes, flush,
   `os.fsync`), and `lock()` context manager: `os.open(.lock, O_CREAT|O_EXCL|O_WRONLY)` writing pid/host/time, retry
   every 0.1 s up to `timeout` (default 30 s), break a lock older than 300 s with a `warnings.warn`; works on
   macOS, Linux, Windows.
6. `state.py` - `StudyState.from_events(events)`: pure replay into `candidates: dict[int, Candidate]` (`id`,
   `params`, `origin`) and `evals: dict[int, Evaluation]` (`id`, `candidate`, `replicate`, `seed`, `kind`, `attempt`,
   `status`, `value`, `reason`, `info`, `asked`, `told`, `ask`), counters (`next_candidate`, `next_eval`, `n_asks`),
   `budget_used()` (ok + pending), `n_failed()`, `retry_queue()` (failed evals whose attempt ≤ `retries` and which
   have not yet been re-issued: a retry is a new evaluation with the same candidate, replicate and seed and
   `attempt + 1`), `apply_tell(...)` validation rules from CONTRACTS §3 (unknown eval, conflicting re-tell,
   non-finite value → `NoflukeError`; identical re-tell → returns "duplicate").
7. `noise.py` - pure statistics on "higher is better" scores: `CandidateStats` per candidate (n, mean, sd, sigma_i,
   shrunk_mean, se, lcb, ucb); `estimate(state, config) -> NoiseModel` implementing D9 exactly (pooled variance
   needs ≥ 3 candidates with n ≥ 2; fallback `prior_sd`, then variance of means; with a single candidate there is
   no population, so μ̂ = ȳ and SE = σ_i/√n, using `prior_sd` or, failing that, its own sd or 0); `rank(...)` per D9 ties; `plausible_best(...)` = S;
   `choose_reeval(stats, counts_incl_pending, max_replicates)` per D8.
8. `strategies/base.py` - `History` (frozen: `space`, `direction`, `completed`: list of (candidate id, params,
   score mean in higher-is-better units, n) sorted by id, `failed`: list of params, `pending`: list of params),
   `Strategy` ABC with `propose(history, rng, index) -> dict` and `name`. `strategies/random.py`: decode a uniform
   draw. `strategies/lhs.py`: `lhs_design(n, d, rng) -> np.ndarray` (one stratified permutation per dimension,
   jittered within cells), cached per (seed, n, d). `strategies/__init__.py`: `get_strategy(config) -> Strategy`
   registry; `tpe`/`gp` raise `NoflukeError("... install nofluke[optuna]")` until the backend exists.
9. `planner.py` - `plan_ask(config, state, n) -> (new_candidates, new_evals)`, pure. Order per D8: retries, then for
   each remaining slot decide re-eval vs new: finalist zone when `budget_used_after_slot > budget*(1-finalist_share)`;
   otherwise re-eval iff `(n_reeval_issued + 1) / (n_issued + 1) <= reeval_share` and ≥ `min_candidates` candidates
   have results and `choose_reeval` returns a candidate; counts include evaluations planned earlier in the same ask.
   New candidates: index k = next candidate id; the first `n_init` come from the LHS design (row k-1, design drawn
   with `derive(seed, "lhs")`), later ones from `strategy.propose(history, derive(seed, "candidate", k), k)` with the
   candidates proposed earlier in this ask added to `pending`. Reject a proposal identical to an existing candidate
   by retrying with `derive(seed, "candidate", k, attempt)` up to 20 times, then accept the duplicate. Seeds per
   D10. Never exceeds the budget; returns fewer slots when it is exhausted. Raises `NoflukeError` if `n_failed >=
   max_failures`.
10. `study.py` - `Study.create(dir, config, *, exist_ok=False)`, `Study.open(dir)`; methods `ask(n=1) ->
    list[Ask]` (dataclass with `eval`, `candidate`, `replicate`, `seed`, `kind`, `attempt`, `params`; `to_dict()`),
    `tell(eval, value, info=None) -> bool` (False when duplicate), `tell_failed(eval, reason, info=None)`,
    `tell_many(records) -> summary`, `abandon(eval, reason)`, `abandon_older_than(seconds)`, `pending() ->
    list[Ask]`, `best(top=None) -> list[dict]` (CONTRACTS §4 shape), `status() -> dict`, `export_rows() ->
    list[dict]`, `is_done()` (budget used up and nothing pending, or failures exhausted), `verify() -> list[str]`
    (replays every ask event against the state before it with `plan_ask` and reports differences; empty = ok).
    Every mutating method: lock → read → replay → plan/validate → append → unlock. Provenance per CONTRACTS §3
    (`nofluke_commit`: `git rev-parse HEAD` in the package's source tree if it is a git checkout, else None; never
    fail on it).
11. `runner.py` - `optimize(study, func, n_workers=1, executor="process"|"thread")`: steady state; keeps n_workers
    evaluations in flight (`ask(1)` per free slot) until `is_done()`; `func(params, seed)` returns a float or
    `(float, info_dict)`; an exception → `tell_failed` with the exception text. `run_command(study, template,
    n_workers, workdir)`: per evaluation writes `<workdir>/eval-<id>/params.json` (the CONTRACTS §4 object), formats
    the template with `{params_file}`, `{result_file}`, `{seed}`, `{eval}`, `{dir}` (paths quoted for the shell),
    runs it with `subprocess.run(..., shell=True)`; result = JSON `{"value":..., "info":...}` in `{result_file}`,
    else the last stdout line parsed as float; nonzero exit or no result → failed with stderr tail as reason.
12. `cli.py` - `main(argv=None) -> int`, argparse subcommands `init CONFIG --dir`, `ask [-n] [--out] [--split DIR]`,
    `tell FILE... | --eval ID (--value X [--info JSON] | --failed REASON)`, `pending`, `abandon (--eval ID |
    --older-than DURATION) [--reason]`, `status [--json]`, `best [--top N] [--json]`, `export [--format csv|jsonl]
    [--out]`, `done` (exit 0 if done, else 1), `verify` (exit 0 if reproducible), `run --cmd TEMPLATE [-j N]
    [--workdir]`, `--version`. `--dir` defaults to `$NOFLUKE_DIR` or `.`. `NoflukeError` → message on stderr, exit 2.
    Human tables on stdout without `--json`; machine JSON with it. Durations like `90s`, `30m`, `2h`.
13. `__init__.py` - exports `Study`, `Space`, `NoflukeError`, `optimize`, `__version__` (from
    `importlib.metadata`, fallback "0+unknown").

## Tests (`tests/`, pytest, offline, < 30 s total)
`test_space.py` (validation errors, conditionals, codec round trip, log bounds, plain types), `test_config.py`
(defaults, unknown keys, hashes stable, YAML load), `test_rng.py` (stable values across calls, independence of
purposes), `test_journal.py` (append/read, torn last line moved aside, lock contention and stale lock),
`test_noise.py` (D9 formulas against hand-computed numbers; pooled vs prior vs fallback σ; τ² = 0 tie-break;
plausible-best set), `test_planner.py` (re-eval share converges to ~1/3; no re-evals before `min_candidates`;
finalist zone; retries first with same seed; budget never exceeded; `max_failures`; LHS first `n_init`; seeds
shared per replicate; per_candidate mode), `test_study.py` (create/open, ask/tell, out-of-order tells, duplicate
and conflicting tells, abandon + late tell, pending, best/status/export shapes, minimize direction),
`test_determinism.py` (same op sequence in two fresh study folders → identical journals except times/host;
op sequence run in a subprocess vs in-process; `verify()` passes, and fails after a hand-edited journal),
`test_cli.py` (full loop through `cli.main` with files: init → ask --split → tell results → status/best/export/done),
`test_runner.py` (optimize with threads on a noisy quadratic; run_command with a tiny Python script; failures).

## Verification
`python -m pytest` green; `python tools/docmap.py check` green; a manual CLI session in a temp folder on a noisy
toy function reaches `done`, and `best` reports n ≥ 2 for the winner.

## Risks
- Lock files on network file systems: O_EXCL is atomic on NFSv3+; documented, not tested here.
- D8/D9 defaults are untested on benchmarks until the harness exists (plan step after this one).
