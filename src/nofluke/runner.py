"""Local evaluators: `optimize` runs a Python function, `run_command` runs a shell command, per evaluation.

Both keep `n_workers` evaluations in flight in steady state (`ask(1)` per free slot) until the study is done, tell
each outcome as it arrives, and turn exceptions, bad return values, nonzero exit codes and missing results into
`tell_failed`. For work on a cluster use the command line (`nofluke ask` / `nofluke tell`) instead.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from functools import partial
from pathlib import Path
from typing import Any, Callable

from nofluke.errors import NoflukeError
from nofluke.study import Ask, Study

Outcome = tuple[str, Any, dict[str, Any] | None]  # ("ok", value, info) or ("failed", reason, None)


def _outcome(value: Any, info: Any) -> Outcome:
    """Validate an evaluator's result into an Outcome; anything unusable becomes a failure with a reason."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ("failed", f"result is not a number: {value!r}", None)
    if not math.isfinite(number):
        return ("failed", f"result is not finite: {value!r}", None)
    if info is not None:
        try:
            json.dumps(info, allow_nan=False)
        except (TypeError, ValueError):
            return ("failed", "info is not a JSON-serialisable object", None)
        if not isinstance(info, dict):
            return ("failed", f"info must be an object, got {info!r}", None)
    return ("ok", number, info)


def _call(func: Callable[..., Any], params: dict[str, Any], seed: int) -> Outcome:
    """Run `func(params, seed)` and classify the result; never raises (the worker may be another process)."""
    try:
        result = func(params, seed)
    except Exception as exc:  # noqa: BLE001 - any evaluator error is a failed evaluation
        return ("failed", f"{type(exc).__name__}: {exc}", None)
    value, info = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
    return _outcome(value, info)


def _drive(study: Study, pool: Executor, n_workers: int, work: Callable[[Ask], Future[Outcome]]) -> None:
    """Steady-state loop: refill free slots with `ask(1)`, tell results as they finish, stop when done.

    If asking fails (e.g. the failure limit is reached), evaluations already running are still collected and told
    first, so no result is lost; the error is raised only if the study is not done after that.
    """
    in_flight: dict[Future[Outcome], Ask] = {}
    stopped: NoflukeError | None = None
    while True:
        while stopped is None and len(in_flight) < n_workers and not study.is_done():
            try:
                asked = study.ask(1)
            except NoflukeError as exc:
                stopped = exc
                break
            if not asked:
                break
            in_flight[work(asked[0])] = asked[0]
        if not in_flight:
            if stopped is not None and not study.is_done():
                raise stopped
            return
        finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in sorted(finished, key=lambda f: in_flight[f].eval):
            ask = in_flight.pop(future)
            status, payload, info = future.result()
            if status == "ok":
                study.tell(ask.eval, payload, info)
            else:
                study.tell_failed(ask.eval, payload)


def optimize(study: Study, func: Callable[[dict[str, Any], int], Any], n_workers: int = 1,
             executor: str = "thread") -> None:  # fmt: skip
    """Evaluate `func(params, seed)` until the study is done, `n_workers` at a time.

    `func` returns a float or `(float, info_dict)`; an exception marks the evaluation failed. `executor` is
    "thread" (any callable) or "process" (`func` must be picklable, e.g. a module-level function).
    """
    if executor not in ("thread", "process"):
        raise NoflukeError(f"optimize: executor must be 'thread' or 'process', got {executor!r}")
    if n_workers < 1:
        raise NoflukeError(f"optimize: n_workers must be >= 1, got {n_workers!r}")
    pool_type = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor
    with pool_type(max_workers=n_workers) as pool:
        _drive(study, pool, n_workers, lambda ask: pool.submit(_call, func, ask.params, ask.seed))


def _quote(path: Path | str) -> str:
    return subprocess.list2cmdline([str(path)]) if os.name == "nt" else shlex.quote(str(path))


def _tail(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _read_result(result_file: Path, stdout: str) -> Outcome:
    if result_file.is_file():
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            return _outcome(data["value"], data.get("info"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return ("failed", f"unusable result file {result_file.name}: {exc!r}", None)
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return ("failed", "no result: no result file and no output", None)
    try:
        return _outcome(float(lines[-1]), None)
    except ValueError:
        return ("failed", f"no result: last output line is not a number: {_tail(lines[-1], 100)!r}", None)


def _run_one(ask: Ask, template: str, workdir: Path) -> Outcome:
    eval_dir = workdir / f"eval-{ask.eval}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    params_file, result_file = eval_dir / "params.json", eval_dir / "result.json"
    params_file.write_text(json.dumps(ask.to_dict(), indent=2), encoding="utf-8")
    result_file.unlink(missing_ok=True)
    command = template.format(params_file=_quote(params_file), result_file=_quote(result_file), seed=ask.seed,
                              eval=ask.eval, dir=_quote(eval_dir))  # fmt: skip
    done = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return ("failed", f"exit status {done.returncode}: {_tail(done.stderr)}", None)
    return _read_result(result_file, done.stdout)


def run_command(study: Study, template: str, n_workers: int = 1, workdir: str | Path = ".") -> None:
    """Evaluate each asked candidate by running a shell command until the study is done.

    The template may use `{params_file}`, `{result_file}`, `{seed}`, `{eval}` and `{dir}`; per evaluation a folder
    `<workdir>/eval-<id>` gets `params.json`. The result is `{"value": x, "info": {...}}` in `{result_file}`, else the
    last stdout line as a number; a nonzero exit or no result marks the evaluation failed.
    """
    if n_workers < 1:
        raise NoflukeError(f"run_command: n_workers must be >= 1, got {n_workers!r}")
    try:
        template.format(params_file="", result_file="", seed=0, eval=0, dir="")
    except (KeyError, IndexError, ValueError) as exc:
        raise NoflukeError(
            f"command template: bad placeholder {exc!r} (use {{params_file}}, {{result_file}}, {{seed}}, {{eval}}, {{dir}})"
        ) from None
    root = Path(workdir)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        _drive(study, pool, n_workers, lambda ask: pool.submit(_run_one, ask, template, root))
