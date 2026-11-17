# Guide: running a nofluke search on a Slurm cluster
Status: current (verified 2026-11-17)

How a client project drives nofluke from a login node or its own machine when every evaluation is a cluster job.
nofluke never submits jobs or touches ssh (D2); everything below lives in the client project. File formats:
`docs/CONTRACTS.md`.

## 1. One-time setup
```bash
pip install "nofluke @ git+https://github.com/Horef/nofluke"   # or pip install -e <checkout>
nofluke init configs/search.yaml --dir searches/lossweights-v1
```
Commit `configs/search.yaml` and record the state folder and the study seed in the project's run log. The state
folder can live on a shared file system; commands lock it, so a submit loop and a result collector may run at once.

## 2. The loop
```bash
S=searches/lossweights-v1
until nofluke done --dir $S; do
  nofluke ask --dir $S -n 4 --split $S/batch-$(date +%s)      # writes eval-<id>.json: params + seed
  # client: turn each eval-<id>.json into a run config, submit (e.g. one sbatch array), wait or poll
  # client: for each finished job write {"eval": <id>, "value": <val_acc>, "info": {"test_acc": ..., "job": ...}}
  nofluke tell --dir $S results/*.json                         # re-telling the same file is harmless
done
nofluke best --dir $S --top 5
```
- `ask` returns fewer evaluations (or none) when the budget is used up; keep collecting until `done` exits 0.
- Pass the evaluation's `seed` to the training run. Replicate r of every candidate gets the same seed (D10).
- A failed job: tell `{"eval": <id>, "status": "failed", "reason": "..."}`. nofluke re-issues it once with the same
  seed at the next ask. A job that will never report: `nofluke abandon --dir $S --eval <id>` or
  `--older-than 2h`; a result arriving later is still accepted.
- Lost the output of an `ask` (e.g. the shell died)? `nofluke pending --dir $S --split DIR` re-prints the
  outstanding evaluations.
- Steady state instead of batches: ask `-n 1` whenever a job finishes.

## 3. Selection and reporting
- `value` is what nofluke selects on (a validation score). Put held-out test scores only in `info`: they are stored
  and exported, never used.
- `best` ranks by a shrunk mean (D9) and shows n and a lower bound; prefer a pick with n ≥ 2. Then evaluate the
  final 1-3 picks once on the test set, outside nofluke, and report how many candidates the pick was made over
  (`n_candidates` in `best --json`).
- `nofluke verify --dir $S` checks that the journal replays exactly (determinism, D7); `nofluke export --format csv`
  gives one row per evaluation for analysis.
