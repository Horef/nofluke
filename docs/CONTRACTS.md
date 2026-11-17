# Contracts: files and formats
Status: current (verified 2026-11-17)

Every format another program reads or writes: the study config, the state folder, journal events and the CLI's JSON.
Changing any of them is a user-visible change (CHANGELOG); changing how an existing journal replays bumps
`format`. Decisions: D5 (journal), D7 (determinism), D8-D11 (noise, selection, seeds, strategies).

## 1. Study config (`study.yaml` or `study.json`)
JSON, or YAML with the `yaml` extra. Unknown keys are an error. Defaults are filled in and the full normalised
config is stored in the journal, so later default changes never alter an existing study.

```yaml
name: lossweights-v1          # optional, free text
direction: maximize           # maximize | minimize (required)
metric: val_acc               # optional display name of the value
budget: 128                   # max evaluations counted as ok + pending (required, >= 1)
seed: 0                       # study seed, any int >= 0 (required)
max_failures: 64              # default budget // 2; ask raises once this many evaluations have failed
retries: 1                    # times a failed evaluation is re-issued with the same seed (default 1)
space:                        # ordered; a parameter's `when` may only name earlier parameters
  lr:      {type: float, low: 1.0e-4, high: 3.0e-3, log: true}
  epochs:  {type: int, low: 30, high: 200}            # log: false by default; bounds inclusive
  batch:   {type: choice, values: [16, 32, 64]}         # JSON scalars (str, int, float, bool, null)
  use_aux: {type: bool}
  aux_w:   {type: float, low: 0.1, high: 100, log: true, when: {use_aux: true}}
strategy:
  name: random                # random | tpe | gp (tpe/gp need the optuna extra)
  n_init: 16                  # default clip(budget // 8, 8, 20); LHS design for the first n_init candidates
  options: {}                 # passed to the backend (e.g. TPE kwargs); must be JSON
noise:
  reeval_share: 0.333         # target share of issued evaluations that are re-evaluations; 0 = off
  min_candidates: 8           # candidates with a result before re-evaluation starts
  max_replicates: 6           # cap on evaluations per candidate (completed + pending)
  finalist_share: 0.08        # final share of the budget spent only on re-evaluating S
  z: 1.5                      # width of the confidence bounds (D8, D9)
  nu0: 4                      # prior weight of the pooled variance (D9)
  prior_sd: null              # optional per-evaluation noise SD used until replicates exist
  seed_mode: shared           # shared | per_candidate (D10)
```

`when: {parent: value}` or `{parent: [v1, v2]}`: the parameter is active only if `parent` is active and its value
is (one of) the given. Inactive parameters are absent from `params`. Float bounds need `low < high` (`low > 0`
if `log`); int bounds `low <= high` (`low >= 1` if `log`); choice needs ≥ 1 distinct value.

## 2. State folder
```
<dir>/journal.jsonl     the only authoritative state (append-only)
<dir>/.lock             present while a command mutates the study (pid, host, time); stale after 300 s
<dir>/journal.torn-<UTC time>.jsonl   an incomplete last line moved aside before an append (D5)
```

## 3. Journal events (`journal.jsonl`, format 1)
One JSON object per line, UTF-8, `\n`-terminated, fsynced after each append; the key `event` names its kind. Times
are UTC ISO 8601 with `Z`.
Ids are integers counted from 1: candidate ids and evaluation ids are separate sequences.

| event | fields |
|---|---|
| `study` (first line only) | `format` (1), `time`, `config` (normalised), `config_hash` (sha256 of canonical JSON; the space is hashed as an ordered list of (name, spec) pairs because its order fixes LHS columns and decoding), `space_hash`, `provenance` {`nofluke_version`, `nofluke_commit` or null, `python`, `numpy`, `optuna` or null, `host`} |
| `ask` | `ask` (int, from 1), `time`, `candidates`: [{`candidate`, `params`, `origin` {`strategy`, `index`}}] (new ones only), `evals`: [{`eval`, `candidate`, `replicate`, `seed`, `kind` (`new`/`reeval`/`retry`), `attempt`}] |
| `tell` | `time`, `eval`, `status` (`ok` or `failed`), `value` (finite number, `ok` only), `reason` (`failed` only, string), `info` (object, optional) |
| `abandon` | `time`, `eval`, `reason`: a pending evaluation given up; no retry; frees its budget slot. A later `tell` for it is still accepted. |

Rules: a `tell` for an evaluation that already has the identical outcome is a no-op (not appended); a different
outcome is an error. Evaluation status: `pending` (asked) → `ok` | `failed` | `abandoned` (→ `ok`/`failed` by a late
tell). Budget used = evaluations `ok` + `pending`. An evaluation's `value` is in the user's units; nofluke works
internally in "higher is better" units (negated for `minimize`).

## 4. CLI JSON
`nofluke ask` and `nofluke pending` print (or write with `--out`):
```json
{"study": "lossweights-v1", "dir": "runs/s1", "evals": [
  {"eval": 7, "candidate": 5, "replicate": 1, "seed": 1817263, "kind": "reeval", "attempt": 1,
   "params": {"lr": 0.00041, "epochs": 120, "batch": 32, "use_aux": false}}]}
```
`--split DIR` also writes one file per evaluation, `DIR/eval-<eval>.json`, holding that object alone.

`nofluke tell FILE...` reads JSON (one object or a list) or JSONL; each record:
`{"eval": 7, "value": 0.771, "info": {"test_acc": 0.70, "runtime_s": 412}}` or
`{"eval": 7, "status": "failed", "reason": "node failure"}`. `info` is stored, never used for selection.

`nofluke best --json` prints `{"n_candidates", "sigma": {"value", "source"}, "tie_break", "best": [...]}`; `best` is
ranked as in D9 with `rank`, `candidate`, `params`, `n`, `mean`, `sd`, `shrunk_mean`, `se`, `lcb`. `sigma.source` is
`pooled`, `prior_sd`, `spread`, `own` or `none`; `tie_break` is true when τ² = 0 and the secondary key decided.
Values are in the user's units (for `minimize`, `lcb` becomes the upper bound and is named `ucb`).

`nofluke export --format csv|jsonl` writes one row per evaluation: `eval`, `candidate`, `replicate`, `seed`,
`kind`, `attempt`, `status`, `value`, `reason`, `asked`, `told`, `ask`, every parameter as `p.<name>`, and every
`info` key as `info.<key>`.
