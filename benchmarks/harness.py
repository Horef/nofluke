"""Benchmark harness: run nofluke arms on the benchmark problems with simulated asynchronous workers.

For every (problem, noise ratio, arm, seed) it runs one study to its budget with `WORKERS` simulated workers
(steady-state `ask(1)`, log-normal durations, results told in finishing order), then records the true value of two
selection rules (nofluke's D9 pick and the best single observation) and of the best candidate ever evaluated. A run
is a pure function of its arguments: `results.jsonl` is identical between repeats except for the `wall_s` and
`peak_rss_mb` fields. See `docs/plans/benchmarks.md` and `benchmarks/README.md`.

Usage: `python benchmarks/harness.py --smoke --jobs 4`, or choose `--problems --noise-ratios --arms --seeds
--budget --out`. Arms are `random`, `lhs`, `tpe`, `gp` (each with protocol on and off) or one of them like `tpe-on`; `<strategy>-raw`
is the protocol on without the D14 seed-effect correction.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import platform
try:
    import resource  # Unix only; peak memory is reported as None elsewhere
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nofluke
from nofluke import Study
from nofluke.rng import derive

import problems as P

WORKERS = 4  # simulated parallel workers
DURATION_SIGMA = 0.5  # sigma of the log-normal evaluation duration (median 1.0 time unit)
STRATEGIES = ("random", "lhs", "tpe", "gp")
SMOKE = {"seeds": [0, 1], "budget": 32, "noise_ratios": [0.5]}
VOLATILE = ("wall_s", "peak_rss_mb")  # run-record fields that differ between repeats
REPO = Path(__file__).resolve().parents[1]


def arm_config(arm: str, problem: P.Problem, budget: int, seed: int) -> dict[str, Any]:
    """The nofluke study config of `arm` (`<strategy>-on|off|raw`) on `problem`.

    Protocol off = `reeval_share: 0`; raw = protocol on with `seed_effects: ignore` (no D14 correction).
    """
    strategy, protocol = arm.rsplit("-", 1)
    config: dict[str, Any] = {"name": f"bench-{problem.name}-{arm}", "direction": "maximize", "budget": budget,
                              "seed": seed, "space": problem.space, "strategy": {}, "noise": {}}  # fmt: skip
    if strategy == "lhs":
        config["strategy"] = {"name": "random", "n_init": budget}  # every new candidate from the LHS design
    else:
        config["strategy"] = {"name": strategy}
    if protocol == "off":
        config["noise"] = {"reeval_share": 0}
    elif protocol == "raw":
        config["noise"] = {"seed_effects": "ignore"}
    return config


def expand_arms(tokens: list[str]) -> list[str]:
    """Turn `random`, `tpe-on`, ... into full arm names (a bare strategy gives both protocol states)."""
    arms: list[str] = []
    for token in tokens:
        strategy = token.rsplit("-", 1)[0] if token.endswith(("-on", "-off", "-raw")) else token
        if strategy not in STRATEGIES:
            raise SystemExit(f"unknown arm '{token}' (strategies: {', '.join(STRATEGIES)})")
        for arm in ([token] if token != strategy else [f"{token}-on", f"{token}-off"]):
            if arm not in arms:
                arms.append(arm)
    return arms


def parse_seeds(text: str) -> list[int]:
    """'0-19' or '0,3,5-7' to a list of ints."""
    seeds: list[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        seeds += list(range(int(low), int(high or low) + 1))
    return seeds


def make_specs(problem_names: list[str], noise_ratios: list[float], arms: list[str], seeds: list[int], budget: int,
               keep_dir: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    """The run list in a fixed order, and notes about skipped combinations (gp on non-continuous problems)."""
    specs: list[dict[str, Any]] = []
    notes: list[str] = []
    for name in problem_names:
        problem = P.get_problem(name)
        ratios: list[float | None] = [None] if problem.noisy_natively else list(noise_ratios)
        for ratio in ratios:
            for arm in arms:
                if arm.startswith("gp-") and not problem.continuous:
                    note = f"skipped gp on {name}: the gp strategy models continuous spaces only (D13)"
                    if note not in notes:
                        notes.append(note)
                    continue
                for seed in seeds:
                    specs.append({"problem": name, "noise_ratio": ratio, "arm": arm, "seed": seed, "budget": budget,
                                  "keep_dir": keep_dir})  # fmt: skip
    return specs, notes


def simulate(study: Study, problem: P.Problem, noise_ratio: float | None, seed: int) -> float:
    """Drive `study` to completion with `WORKERS` simulated workers; returns the simulated makespan."""
    clock = 0.0
    in_flight: list[tuple[float, int, Any]] = []  # (finish time, eval id, ask)
    while True:
        while len(in_flight) < WORKERS:
            asked = study.ask(1)
            if not asked:
                break
            a = asked[0]
            duration = math.exp(DURATION_SIGMA * derive(seed, "bench-duration", a.eval).normal())
            heapq.heappush(in_flight, (clock + duration, a.eval, a))
        if not in_flight:
            break
        clock, _, a = heapq.heappop(in_flight)
        study.tell(a.eval, problem.observe(a.params, a.seed, a.candidate, seed, noise_ratio))
    if not study.is_done():
        raise RuntimeError("simulation stopped before the study was done")
    return clock


def run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one (problem, noise ratio, arm, seed) study and return its metrics record (never raises: errors are recorded)."""
    record = {k: spec[k] for k in ("problem", "noise_ratio", "arm", "seed", "budget")}
    started = time.perf_counter()
    try:
        record.update(_run(spec))
    except Exception as exc:  # noqa: BLE001 - one failed run must not kill a sweep
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_s"] = round(time.perf_counter() - started, 3)
    record["peak_rss_mb"] = None if resource is None else round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1)
    return record


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    problem = P.get_problem(spec["problem"])
    seed, ratio = spec["seed"], spec["noise_ratio"]
    config = arm_config(spec["arm"], problem, spec["budget"], seed)
    if spec["keep_dir"]:
        label = f"{spec['problem']}_nr{ratio}_{spec['arm']}_s{seed}"
        tmp = None
        folder = Path(spec["keep_dir"]) / label
    else:
        tmp = tempfile.TemporaryDirectory(prefix="nofluke-bench-")
        folder = Path(tmp.name) / "study"
    try:
        study = Study.create(folder, config)
        makespan = simulate(study, problem, ratio, seed)
        status = study.status()
        report = study.best_report()["best"]
        rows = [r for r in study.export_rows() if r["status"] == "ok"]
    finally:
        if tmp is not None:
            tmp.cleanup()
    by_candidate = {r["candidate"]: r for r in report}
    true_of = {cid: problem.true(r["params"]) for cid, r in by_candidate.items()}
    d9 = report[0]
    single = max(rows, key=lambda r: (r["value"], -r["eval"]))
    out: dict[str, Any] = {
        "optimum": problem.optimum, "true_sd": None if math.isnan(problem.true_sd) else problem.true_sd,
        "n_evals": status["ok"], "n_candidates": status["candidates"], "n_reevals": status["reevaluations"],
        "sim_time": round(makespan, 6),
        "d9_candidate": d9["candidate"], "d9_n": d9["n"], "d9_reported": d9["mean"], "d9_shrunk": d9["shrunk_mean"],
        "d9_true": true_of[d9["candidate"]],
        "bs_candidate": single["candidate"], "bs_n": by_candidate[single["candidate"]]["n"],
        "bs_reported": single["value"], "bs_true": true_of[single["candidate"]],
        "ever_true": max(true_of.values()),
    }  # fmt: skip
    out["d9_optimism"] = out["d9_reported"] - out["d9_true"]
    out["d9_optimism_shrunk"] = out["d9_shrunk"] - out["d9_true"]
    out["bs_optimism"] = out["bs_reported"] - out["bs_true"]
    out["same_pick"] = out["d9_candidate"] == out["bs_candidate"]
    for key in ("d9", "bs", "ever"):
        out[f"{key}_regret"] = None if problem.optimum is None else problem.optimum - out[f"{key}_true"]
    return out


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"revision": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _versions() -> dict[str, Any]:
    import numpy

    out: dict[str, Any] = {"nofluke": nofluke.__version__, "python": platform.python_version(), "numpy": numpy.__version__}
    for name in ("optuna", "torch", "scipy", "sklearn"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def build_manifest(run_id: str, specs: list[dict[str, Any]], args: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """The manifest dict: run id, UTC time, versions, git state, harness config (with its hash), arms, problems, seeds."""
    arms = sorted({s["arm"] for s in specs}, key=lambda a: (STRATEGIES.index(a.rsplit("-", 1)[0]), ("on", "raw", "off").index(a.rsplit("-", 1)[1])))
    config = {
        "workers": WORKERS, "duration_sigma": DURATION_SIGMA, "shared_variance_share": P.SHARED_VARIANCE_SHARE,
        "sd_points": P.SD_POINTS, "arm_configs": {arm: arm_config(arm, P.get_problem("branin"), args["budget"], 0)
                                                   for arm in arms},
    }  # fmt: skip
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "run_id": run_id, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nofluke_version": nofluke.__version__, "git": _git_state(), "versions": _versions(),
        "config": config, "config_hash": config_hash, "arms": arms,
        "problems": sorted({s["problem"] for s in specs}), "noise_ratios": args["noise_ratios"],
        "seeds": args["seeds"], "budget": args["budget"], "runs": len(specs), "smoke": args["smoke"], "notes": notes,
        "command": " ".join(sys.argv),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--problems", default="all", help="comma list of branin,hartmann6,mixed_hpo,sklearn_svm (default: all available)")
    ap.add_argument("--noise-ratios", default="0.25,0.5,1.0", help="comma list; ignored by problems with native noise")
    ap.add_argument("--arms", default="all", help="comma list of random,lhs,tpe,gp or e.g. tpe-on (default all)")
    ap.add_argument("--seeds", default="0-19", help="e.g. 0-19 or 0,3,5-7")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (each run is single-threaded)")
    ap.add_argument("--out", default=None, help="output folder (default benchmarks/results/<run_id>)")
    ap.add_argument("--smoke", action="store_true", help="2 seeds, budget 32, noise ratio 0.5, every problem and arm")
    ap.add_argument("--keep", action="store_true", help="keep each run's study folder under <out>/studies")
    args = ap.parse_args(argv)

    problem_names = P.available_problems() if args.problems == "all" else args.problems.split(",")
    skipped = [n for n in ("sklearn_svm",) if args.problems == "all" and n not in problem_names]
    notes = [f"skipped {n}: scikit-learn is not installed (pip install 'nofluke[bench]')" for n in skipped]
    for n in problem_names:
        if n not in P.available_problems():
            raise SystemExit(f"problem '{n}' is unknown or unavailable here (available: {', '.join(P.available_problems())})")
    seeds, budget, ratios = parse_seeds(args.seeds), args.budget, [float(x) for x in args.noise_ratios.split(",")]
    if args.smoke:
        seeds, budget, ratios = SMOKE["seeds"], SMOKE["budget"], SMOKE["noise_ratios"]
    arms = expand_arms(["random", "lhs", "tpe", "gp"] if args.arms == "all" else args.arms.split(","))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{'smoke-' if args.smoke else ''}{stamp}"
    out = Path(args.out) if args.out else REPO / "benchmarks" / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    keep_dir = str(out / "studies") if args.keep else None
    specs, more = make_specs(problem_names, ratios, arms, seeds, budget, keep_dir)
    notes += more
    for note in notes:
        print(note, file=sys.stderr)
    manifest = build_manifest(run_id, specs, {"budget": budget, "seeds": seeds, "noise_ratios": ratios, "smoke": args.smoke}, notes)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"  # one thread per run: parallelism comes from --jobs
    print(f"{len(specs)} runs, {args.jobs} jobs, output {out}", file=sys.stderr)
    started = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    partial = out / "results.partial.jsonl"
    with partial.open("w") as sink:
        if args.jobs <= 1:
            for i, spec in enumerate(specs):
                results[i] = run_one(spec)
                sink.write(json.dumps(results[i]) + "\n")
                sink.flush()
                print(f"[{len(results)}/{len(specs)}] {spec['problem']} {spec['arm']} s{spec['seed']}", file=sys.stderr)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(run_one, spec): i for i, spec in enumerate(specs)}
                for future in as_completed(futures):
                    results[futures[future]] = record = future.result()
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    print(f"[{len(results)}/{len(specs)}] {record['problem']} {record['arm']} s{record['seed']}", file=sys.stderr)
    (out / "results.jsonl").write_text("".join(json.dumps(results[i]) + "\n" for i in sorted(results)))
    partial.unlink()
    errors = sum(1 for r in results.values() if "error" in r)
    print(f"done in {time.perf_counter() - started:.1f} s; {errors} runs with errors; results in {out}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
