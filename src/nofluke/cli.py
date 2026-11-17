"""Command line: `nofluke init | ask | tell | pending | abandon | status | best | export | done | verify | run`.

`main(argv)` returns the exit status (0 ok; 1 for `done`/`verify` answering "no"; 2 for a `NoflukeError` or bad
arguments). The study folder is `--dir`, else `$NOFLUKE_DIR`, else the current directory. Commands print human
tables, or the JSON of CONTRACTS section 4 where a program is meant to read them (`ask`, `pending`, `--json`).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import nofluke
from nofluke.errors import NoflukeError
from nofluke.runner import run_command
from nofluke.study import Ask, Study

_DURATION = re.compile(r"^(\d+(?:\.\d+)?)([smh])$")
_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(text: str) -> float:
    """Seconds in a duration like `90s`, `30m` or `2h`."""
    match = _DURATION.match(text.strip())
    if not match:
        raise NoflukeError(f"duration must look like 90s, 30m or 2h, got {text!r}")
    return float(match.group(1)) * _UNITS[match.group(2)]


def _emit(text: str, out: str | None) -> None:
    """Print `text`, or write it to the file `out`."""
    if out is None:
        print(text, end="" if text.endswith("\n") else "\n")
    else:
        Path(out).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _asks_json(study: Study, asks: list[Ask]) -> dict[str, Any]:
    return {"study": study.config["name"], "dir": str(study.dir), "evals": [a.to_dict() for a in asks]}


def _read_records(path: str) -> list[Any]:
    """Records of a tell file: one JSON object, a JSON list, or JSON lines."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise NoflukeError(f"tell: cannot read '{path}': {exc.strerror or exc}") from None
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise NoflukeError(f"tell: '{path}' line {number} is not valid JSON ({exc})") from None
    return records


def _cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return format(value, f".{digits}g") if isinstance(value, float) else str(value)


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
    lines = ["  ".join(h.rjust(w) for h, w in zip(header, widths))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(lines)


def _cmd_init(args: argparse.Namespace) -> int:
    study = Study.create(args.dir, args.config)
    cfg = study.config
    print(f"created study{' ' + cfg['name'] if cfg['name'] else ''} in {study.dir} "
          f"(budget {cfg['budget']}, {cfg['direction']}, seed {cfg['seed']})")  # fmt: skip
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    study = Study.open(args.dir)
    asks = study.ask(args.n)
    if args.split:
        split = Path(args.split)
        split.mkdir(parents=True, exist_ok=True)
        for a in asks:
            (split / f"eval-{a.eval}.json").write_text(json.dumps(a.to_dict(), indent=2) + "\n", encoding="utf-8")
    _emit(json.dumps(_asks_json(study, asks), indent=2), args.out)
    return 0


def _cmd_pending(args: argparse.Namespace) -> int:
    study = Study.open(args.dir)
    _emit(json.dumps(_asks_json(study, study.pending()), indent=2), args.out)
    return 0


def _cmd_tell(args: argparse.Namespace) -> int:
    study = Study.open(args.dir)
    if args.files:
        if args.eval is not None:
            raise NoflukeError("tell: give either FILE... or --eval, not both")
        records = [r for path in args.files for r in _read_records(path)]
    elif args.eval is not None:
        if (args.value is None) == (args.failed is None):
            raise NoflukeError("tell: --eval needs exactly one of --value or --failed")
        record: dict[str, Any] = {"eval": args.eval}
        if args.value is not None:
            record["value"] = args.value
        else:
            record.update(status="failed", reason=args.failed)
        if args.info is not None:
            try:
                record["info"] = json.loads(args.info)
            except json.JSONDecodeError as exc:
                raise NoflukeError(f"tell: --info is not valid JSON ({exc})") from None
        records = [record]
    else:
        raise NoflukeError("tell: give FILE... or --eval ID")
    summary = study.tell_many(records)
    print(f"told {summary['applied']} evaluation(s), {summary['duplicate']} already recorded")
    return 0


def _cmd_abandon(args: argparse.Namespace) -> int:
    study = Study.open(args.dir)
    if (args.eval is None) == (args.older_than is None):
        raise NoflukeError("abandon: give exactly one of --eval or --older-than")
    if args.eval is not None:
        study.abandon(args.eval, args.reason or "abandoned")
        print(f"abandoned evaluation {args.eval}")
    else:
        ids = study.abandon_older_than(parse_duration(args.older_than), args.reason or f"abandoned: older than {args.older_than}")
        print(f"abandoned {len(ids)} evaluation(s)" + (f": {', '.join(map(str, ids))}" if ids else ""))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    status = Study.open(args.dir).status()
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    label = status["name"] or "study"
    metric = f" ({status['metric']})" if status["metric"] else ""
    print(f"{label}: {status['direction']}{metric}, {'done' if status['done'] else 'running'}")
    print(f"budget {status['used']}/{status['budget']} used ({status['remaining']} left)")
    print(f"evaluations: {status['ok']} ok, {status['pending']} pending, {status['failed']} failed, "
          f"{status['abandoned']} abandoned; {status['reevaluations']} re-evaluations")  # fmt: skip
    print(f"{status['candidates']} candidates, {status['asks']} asks")
    return 0


def _cmd_best(args: argparse.Namespace) -> int:
    report = Study.open(args.dir).best_report(args.top)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    rows = report["best"]
    bound = "lcb" if rows and "lcb" in rows[0] else "ucb"
    header = ["rank", "cand", "n", "mean", "sd", "shrunk", "se", bound, "params"]
    body = [[_cell(r["rank"]), _cell(r["candidate"]), _cell(r["n"]), _cell(r["mean"]), _cell(r["sd"]),
             _cell(r["shrunk_mean"]), _cell(r["se"]), _cell(r[bound]), json.dumps(r["params"])] for r in rows]  # fmt: skip
    print(_table(header, body))
    sigma = report["sigma"]
    print(f"{report['n_candidates']} candidates with results; noise SD {_cell(sigma['value'])} ({sigma['source']})")
    if report["tie_break"]:
        print("note: all shrunk means are equal (no spread beyond noise); ranking falls back to mean - z*sigma/sqrt(n)")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    rows = Study.open(args.dir).export_rows()
    if args.format == "jsonl":
        text = "".join(json.dumps(r) + "\n" for r in rows)
    else:
        columns: list[str] = []
        for row in rows:
            columns += [c for c in row if c not in columns]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        text = buffer.getvalue()
    if args.out is None:
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8", newline="")
    return 0


def _cmd_done(args: argparse.Namespace) -> int:
    done = Study.open(args.dir).is_done()
    print("done" if done else "not done")
    return 0 if done else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    problems = Study.open(args.dir).verify()
    for problem in problems:
        print(problem)
    print("ok: replay reproduces the journal" if not problems else f"{len(problems)} problem(s) found")
    return 0 if not problems else 1


def _cmd_run(args: argparse.Namespace) -> int:
    study = Study.open(args.dir)
    run_command(study, args.cmd, n_workers=args.jobs, workdir=args.workdir or study.dir / "evals")
    status = study.status()
    print(f"{status['ok']} ok, {status['failed']} failed; {'done' if status['done'] else 'not done'}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nofluke", description="Ask/tell search over slow, noisy evaluations.")
    parser.add_argument("--version", action="version", version=f"nofluke {nofluke.__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default=os.environ.get("NOFLUKE_DIR", "."), help="study folder (default: $NOFLUKE_DIR or .)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, handler: Any, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=help, description=help)
        p.set_defaults(handler=handler)
        return p

    p = command("init", _cmd_init, "create a study from a config file")
    p.add_argument("config", help="study.yaml or study.json")
    p = command("ask", _cmd_ask, "hand out evaluations (JSON)")
    p.add_argument("-n", type=int, default=1, help="number of evaluations (default 1)")
    p.add_argument("--out", help="write the JSON to this file instead of printing it")
    p.add_argument("--split", metavar="DIR", help="also write DIR/eval-<id>.json, one file per evaluation")
    p = command("tell", _cmd_tell, "report results from files or one evaluation")
    p.add_argument("files", nargs="*", help="JSON, JSON list or JSONL files of {eval, value|status+reason, info}")
    p.add_argument("--eval", type=int, help="evaluation id (with --value or --failed)")
    p.add_argument("--value", type=float)
    p.add_argument("--failed", metavar="REASON")
    p.add_argument("--info", help="JSON object stored with the result")
    p = command("pending", _cmd_pending, "list evaluations asked but not yet told (JSON)")
    p.add_argument("--out", help="write the JSON to this file instead of printing it")
    p = command("abandon", _cmd_abandon, "give up pending evaluations")
    p.add_argument("--eval", type=int)
    p.add_argument("--older-than", metavar="DURATION", help="e.g. 90s, 30m, 2h")
    p.add_argument("--reason")
    p = command("status", _cmd_status, "show progress")
    p.add_argument("--json", action="store_true")
    p = command("best", _cmd_best, "rank candidates (shrunk mean, with n and a bound)")
    p.add_argument("--top", type=int)
    p.add_argument("--json", action="store_true")
    p = command("export", _cmd_export, "write one row per evaluation")
    p.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    p.add_argument("--out")
    command("done", _cmd_done, "exit 0 if the study is done, else 1")
    command("verify", _cmd_verify, "check that replaying the journal reproduces every ask (exit 0 if so)")
    p = command("run", _cmd_run, "evaluate with a local shell command until done")
    p.add_argument("--cmd", required=True, help="template with {params_file} {result_file} {seed} {eval} {dir}")
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--workdir", help="where eval-<id> folders go (default: <dir>/evals)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns the exit status instead of exiting."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits for --help, --version and usage errors
        return exc.code if isinstance(exc.code, int) else 2
    try:
        return args.handler(args)
    except NoflukeError as exc:
        print(f"nofluke: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
