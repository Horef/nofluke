"""Tests for nofluke.runner: `optimize` with threads and processes, failures, and `run_command` with a tiny script."""

from __future__ import annotations

import json
import sys
import threading
import time

import pytest

from conftest import make_config
from nofluke import NoflukeError, Study, optimize
from nofluke.runner import run_command
from scenario import evaluate


def test_optimize_threads_on_a_noisy_quadratic(tmp_path):
    study = Study.create(tmp_path, make_config(budget=60, name="quad"))
    optimize(study, lambda params, seed: (evaluate(params, seed), {"x": params["x"]}), n_workers=4)
    status = study.status()
    assert status["done"] and status["ok"] == 60 and status["pending"] == 0
    best = study.best(top=1)[0]
    assert best["n"] >= 2 and abs(best["params"]["x"] - 1.0) < 1.0
    assert study.verify() == []
    assert "info.x" in study.export_rows()[0]


def test_optimize_keeps_n_workers_in_flight_and_no_more(tmp_path):
    study = Study.create(tmp_path, make_config(budget=24))
    full = threading.Condition()
    running = peak = 0

    def func(params, seed):
        nonlocal running, peak
        with full:
            running += 1
            peak = max(peak, running)
            full.notify_all()
            # Wait (bounded) until the pool is full, so the test does not depend on how fast `ask` is (slow on Windows).
            full.wait_for(lambda: running >= 3 or study_is_draining(), timeout=2.0)
        with full:
            running -= 1
        return evaluate(params, seed)

    def study_is_draining():
        return study.status()["remaining"] == 0

    optimize(study, func, n_workers=3)
    assert peak == 3 and study.status()["ok"] == 24


def test_optimize_single_worker_is_sequential_and_deterministic(tmp_path):
    a, b = Study.create(tmp_path / "a", make_config(budget=20)), Study.create(tmp_path / "b", make_config(budget=20))
    optimize(a, evaluate)
    optimize(b, evaluate)
    values = [r["value"] for r in a.export_rows()]
    assert len(values) == 20 and values == [r["value"] for r in b.export_rows()]


def test_optimize_with_processes(tmp_path):
    study = Study.create(tmp_path, make_config(budget=12))
    optimize(study, evaluate, n_workers=2, executor="process")
    assert study.status()["ok"] == 12


def test_exceptions_and_bad_values_become_failures_that_are_retried(tmp_path):
    study = Study.create(tmp_path, make_config(budget=10, max_failures=20))
    calls = {"n": 0}

    def flaky(params, seed):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("node failure")
        if calls["n"] == 3:
            return float("nan")
        if calls["n"] == 4:
            return "oops"
        return evaluate(params, seed)

    optimize(study, flaky)
    rows = study.export_rows()
    failed = [r for r in rows if r["status"] == "failed"]
    assert [r["reason"] for r in failed][0] == "RuntimeError: node failure"
    assert any("not finite" in r["reason"] for r in failed) and any("not a number" in r["reason"] for r in failed)
    assert any(r["kind"] == "retry" and r["attempt"] == 2 for r in rows)
    assert study.status()["ok"] == 10 and study.is_done()


def test_optimize_stops_when_failures_are_exhausted(tmp_path):
    study = Study.create(tmp_path, make_config(budget=30, max_failures=3, retries=0))

    def always_fails(params, seed):
        raise ValueError("bad")

    optimize(study, always_fails, n_workers=2)
    status = study.status()
    assert status["failed"] >= 3 and status["ok"] == 0 and status["done"]
    assert status["pending"] == 0  # evaluations in flight when the limit hit were still told


def test_optimize_collects_in_flight_results_after_the_failure_limit(tmp_path):
    study = Study.create(tmp_path, make_config(budget=30, max_failures=1, retries=0))
    first_failed = threading.Event()

    def first_fails_others_wait(params, seed):
        if not first_failed.is_set():
            first_failed.set()
            raise ValueError("bad")
        time.sleep(0.2)  # still running when the failure is told and the next ask is refused
        return 1.0

    optimize(study, first_fails_others_wait, n_workers=3)
    status = study.status()
    assert status["failed"] == 1 and status["pending"] == 0 and status["ok"] >= 1 and status["done"]


def test_optimize_validates_arguments(tmp_path):
    study = Study.create(tmp_path, make_config())
    with pytest.raises(NoflukeError, match="executor"):
        optimize(study, evaluate, executor="fork")
    with pytest.raises(NoflukeError, match="n_workers"):
        optimize(study, evaluate, n_workers=0)


EVAL_SCRIPT = """\
import json, sys
params_file, result_file = sys.argv[1], sys.argv[2]
ask = json.load(open(params_file))
mode = sys.argv[3]
if mode == "json":
    json.dump({"value": -(ask["params"]["x"] - 1) ** 2, "info": {"eval": ask["eval"]}}, open(result_file, "w"))
elif mode == "stdout":
    print("some log line"); print(-(ask["params"]["x"] - 1) ** 2)
elif mode == "exit":
    sys.stderr.write("CUDA out of memory"); sys.exit(3)
elif mode == "silent":
    pass
elif mode == "badjson":
    open(result_file, "w").write("{")
elif mode == "text":
    print("not a number")
"""


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "evalone.py"
    path.write_text(EVAL_SCRIPT, encoding="utf-8")
    return path


def command(script, mode):
    return f'"{sys.executable}" "{script}" {{params_file}} {{result_file}} {mode}'


def test_run_command_with_result_files(tmp_path, script):
    study = Study.create(tmp_path / "s", make_config(budget=10))
    run_command(study, command(script, "json"), n_workers=3, workdir=tmp_path / "work")
    assert study.status()["ok"] == 10
    params = json.loads((tmp_path / "work" / "eval-1" / "params.json").read_text(encoding="utf-8"))
    assert params["eval"] == 1 and "x" in params["params"] and "seed" in params
    assert "info.eval" in study.export_rows()[0]
    assert study.verify() == []


def test_run_command_falls_back_to_the_last_stdout_line(tmp_path, script):
    study = Study.create(tmp_path / "s", make_config(budget=4))
    run_command(study, command(script, "stdout"), workdir=tmp_path / "work")
    assert study.status()["ok"] == 4


@pytest.mark.parametrize(
    "mode, fragment",
    [("exit", "exit status 3: CUDA out of memory"), ("silent", "no result"), ("badjson", "unusable result file"),
     ("text", "last output line is not a number")],
)  # fmt: skip
def test_run_command_failures_carry_a_reason(tmp_path, script, mode, fragment):
    study = Study.create(tmp_path / "s", make_config(budget=3, max_failures=4, retries=0))
    run_command(study, command(script, mode), workdir=tmp_path / "work")
    rows = study.export_rows()
    assert rows and all(r["status"] == "failed" and fragment in r["reason"] for r in rows)
    assert study.is_done()


def test_run_command_rejects_unknown_placeholders(tmp_path):
    study = Study.create(tmp_path / "s", make_config())
    with pytest.raises(NoflukeError, match="bad placeholder"):
        run_command(study, "echo {nonsense}", workdir=tmp_path)
    assert study.status()["asks"] == 0
