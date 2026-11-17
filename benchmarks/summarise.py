"""Summarise a benchmark results folder into markdown tables (medians and IQRs over seeds, paired win rates).

Usage: `python benchmarks/summarise.py benchmarks/results/<run_id>`. Per problem x noise ratio and arm it prints the
median [Q1, Q3] over seeds of the regret of each selection rule (nofluke's D9 pick, the best single observation, and
the best candidate ever evaluated) and of the optimism (reported - true) of the two picks; then paired win rates
(same problem, noise ratio, strategy and seed): protocol on vs off for each rule, the headline "protocol on + D9 pick"
vs "protocol off + best single observation", and each strategy vs random (protocol on). Regret = optimum - true
value; where the optimum is unknown (`sklearn_svm`) the best true value seen in the run is used instead.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RULES = (("d9", "D9 pick"), ("bs", "best single"), ("ever", "best ever evaluated"))
STRATEGY_ORDER = ["random", "lhs", "tpe", "gp"]


def load(folder: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(manifest, run records) of a results folder; records with an error are returned too (see `split_errors`)."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines() if line.strip()]
    return manifest, records


def add_regret(records: list[dict[str, Any]]) -> None:
    """Fill each rule's `<rule>_regret` where the optimum is unknown, from the best true value seen per problem."""
    best: dict[str, float] = {}
    for r in records:
        if "error" not in r and r["optimum"] is None:
            best[r["problem"]] = max(best.get(r["problem"], -np.inf), r["d9_true"], r["bs_true"], r["ever_true"])
    for r in records:
        if "error" not in r and r["optimum"] is None:
            for key, _ in RULES:
                r[f"{key}_regret"] = best[r["problem"]] - r[f"{key}_true"]


def _fmt(values: list[float]) -> str:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return f"{med:.4g} [{q1:.3g}, {q3:.3g}]"


def _arm_key(arm: str) -> tuple[int, int]:
    strategy, protocol = arm.rsplit("-", 1)
    return STRATEGY_ORDER.index(strategy), 0 if protocol == "on" else 1


def _win(mine: list[float], other: list[float]) -> tuple[float, int]:
    """Win rate of `mine` (lower is better) over paired `other`; ties count 1/2. Also the number of ties."""
    a, b = np.array(mine), np.array(other)
    ties = int(np.sum(a == b))
    return (float(np.sum(a < b)) + 0.5 * ties) / len(a), ties


def tables(records: list[dict[str, Any]]) -> str:
    """The markdown report for already-loaded records."""
    good = [r for r in records if "error" not in r]
    add_regret(good)
    out: list[str] = []
    if len(good) < len(records):
        out.append(f"**{len(records) - len(good)} runs failed and are excluded.**\n")
    groups: dict[tuple, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for r in good:
        groups[(r["problem"], r["noise_ratio"])][r["arm"]][r["seed"]] = r
    for (problem, ratio), arms in groups.items():
        sample = next(iter(next(iter(arms.values())).values()))
        scale = f", SD of true {sample['true_sd']:.4g}" if sample["true_sd"] else ""
        optimum = "unknown (best true seen)" if sample["optimum"] is None else f"{sample['optimum']:.6g}"
        label = "native noise" if ratio is None else f"noise ratio {ratio}"
        out.append(f"### {problem}, {label} (budget {sample['budget']}, optimum {optimum}{scale})\n")
        out.append("Median [Q1, Q3] over seeds. Regret = optimum - true value (lower is better); optimism = reported - true.\n")
        out.append("| arm | seeds | regret D9 | regret best-single | regret best-ever | optimism D9 | optimism best-single "
                   "| re-evals | same pick |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for arm in sorted(arms, key=_arm_key):
            runs = list(arms[arm].values())
            cells = [_fmt([r[f"{k}_regret"] for r in runs]) for k, _ in RULES]
            cells += [_fmt([r["d9_optimism"] for r in runs]), _fmt([r["bs_optimism"] for r in runs]),
                      f"{np.median([r['n_reevals'] for r in runs]):.0f}",
                      f"{np.mean([r['same_pick'] for r in runs]):.0%}"]  # fmt: skip
            out.append(f"| {arm} | {len(runs)} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Paired win rates (same seed; share of seeds with lower regret, ties count 1/2; `n` pairs, `t` ties).\n")
        out.append("| comparison | D9 pick | best-single pick |")
        out.append("|---|---|---|")
        strategies = [s for s in STRATEGY_ORDER if f"{s}-on" in arms]
        for s in strategies:
            on, off = arms.get(f"{s}-on", {}), arms.get(f"{s}-off", {})
            seeds = sorted(set(on) & set(off))
            if seeds:
                cells = []
                for key in ("d9", "bs"):
                    w, t = _win([on[i][f"{key}_regret"] for i in seeds], [off[i][f"{key}_regret"] for i in seeds])
                    cells.append(f"{w:.0%} (n={len(seeds)}, t={t})")
                out.append(f"| {s}: protocol on vs off | {cells[0]} | {cells[1]} |")
                w, t = _win([on[i]["d9_regret"] for i in seeds], [off[i]["bs_regret"] for i in seeds])
                out.append(f"| {s}: on + D9 pick vs off + best single | {w:.0%} (n={len(seeds)}, t={t}) | |")
            raw = arms.get(f"{s}-raw", {})
            seeds = sorted(set(on) & set(raw))
            if seeds:
                w, t = _win([on[i]["d9_regret"] for i in seeds], [raw[i]["d9_regret"] for i in seeds])
                out.append(f"| {s}: seed-effect correction on vs raw (D14) | {w:.0%} (n={len(seeds)}, t={t}) | |")
        base = arms.get("random-on", {})
        for s in strategies:
            if s == "random":
                continue
            seeds = sorted(set(base) & set(arms[f"{s}-on"]))
            if seeds:
                cells = []
                for key in ("d9", "bs"):
                    w, t = _win([arms[f"{s}-on"][i][f"{key}_regret"] for i in seeds], [base[i][f"{key}_regret"] for i in seeds])
                    cells.append(f"{w:.0%} (n={len(seeds)}, t={t})")
                out.append(f"| {s} vs random (protocol on) | {cells[0]} | {cells[1]} |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    manifest, records = load(args[0])
    print(f"# Benchmark summary: {manifest['run_id']}\n")
    print(f"nofluke {manifest['nofluke_version']}, git {str(manifest['git']['revision'])[:10]}"
          f"{' (dirty)' if manifest['git']['dirty'] else ''}, {len(records)} runs, seeds {manifest['seeds']}, "
          f"budget {manifest['budget']}.\n")  # fmt: skip
    for note in manifest.get("notes", []):
        print(f"Note: {note}\n")
    print(tables(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
