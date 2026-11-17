"""Determinism contract (D7): same operations give the same candidates, in separate folders and in a subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import make_config
from nofluke import Study
from scenario import evaluate, run_operations

TESTS = Path(__file__).parent


def stable_events(directory: Path) -> list[dict]:
    """Journal events without wall-clock times and host-specific provenance."""
    events = [json.loads(line) for line in (directory / "journal.jsonl").read_text().splitlines()]
    for event in events:
        event.pop("time", None)
        event.pop("provenance", None)
    return events


def test_identical_operations_in_separate_folders_give_identical_journals(tmp_path):
    asks_a = run_operations(tmp_path / "a")
    asks_b = run_operations(tmp_path / "b")
    assert asks_a == asks_b and len(asks_a) > 36
    assert stable_events(tmp_path / "a") == stable_events(tmp_path / "b")
    assert any(a["kind"] == "reeval" for a in asks_a) and any(a["kind"] == "retry" for a in asks_a)


def test_a_subprocess_reproduces_the_in_process_run(tmp_path):
    in_process = run_operations(tmp_path / "local")
    done = subprocess.run([sys.executable, str(TESTS / "scenario.py"), str(tmp_path / "remote")],
                          capture_output=True, text=True, check=True, cwd=TESTS)  # fmt: skip
    assert json.loads(done.stdout) == in_process
    assert stable_events(tmp_path / "local") == stable_events(tmp_path / "remote")


def test_reopening_between_operations_changes_nothing(tmp_path):
    continuous = Study.create(tmp_path / "a", make_config(budget=24))
    reopening = Study.create(tmp_path / "b", make_config(budget=24))
    for _ in range(8):
        a = continuous.ask(3)
        b = Study.open(tmp_path / "b").ask(3)
        assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
        for x in a:
            continuous.tell(x.eval, evaluate(x.params, x.seed))
        Study.open(tmp_path / "b").tell_many([{"eval": x.eval, "value": evaluate(x.params, x.seed)} for x in b])
    assert stable_events(tmp_path / "a") == stable_events(tmp_path / "b")


def test_order_of_results_within_one_tell_does_not_matter(tmp_path):
    studies = [Study.create(tmp_path / name, make_config(budget=40)) for name in ("fwd", "rev")]
    for study, order in zip(studies, (1, -1)):
        asks = study.ask(10)
        records = [{"eval": a.eval, "value": evaluate(a.params, a.seed)} for a in asks]
        study.tell_many(records[::order])
    forward = [a.to_dict() for a in studies[0].ask(6)]
    reverse = [a.to_dict() for a in studies[1].ask(6)]
    assert forward == reverse and any(a["kind"] == "reeval" for a in forward)


def test_verify_passes_on_untouched_journals(tmp_path):
    run_operations(tmp_path)
    assert Study.open(tmp_path).verify() == []


def test_verify_detects_a_hand_edited_journal(tmp_path):
    run_operations(tmp_path)
    path = tmp_path / "journal.jsonl"
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    ask_line = next(i for i, line in enumerate(lines) if json.loads(line)["event"] == "ask")

    edited = json.loads(lines[ask_line])
    edited["candidates"][0]["params"]["x"] = 0.123456
    lines[ask_line] = json.dumps(edited, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = Study.open(tmp_path).verify()
    assert any("different candidates" in p for p in problems)

    edited = json.loads(original.splitlines()[ask_line])
    edited["evals"][0]["seed"] += 1
    lines = original.splitlines()
    lines[ask_line] = json.dumps(edited, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert any("different evaluations" in p for p in Study.open(tmp_path).verify())

    path.write_text(original.replace('"seed":11', '"seed":12', 1), encoding="utf-8")
    assert any("config_hash" in p for p in Study.open(tmp_path).verify())


def test_verify_reports_an_impossible_event_sequence(tmp_path):
    run_operations(tmp_path)
    path = tmp_path / "journal.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.append(json.dumps({"event": "tell", "time": "t", "eval": 9999, "status": "ok", "value": 1.0}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert any("unknown evaluation 9999" in p for p in Study.open(tmp_path).verify())
