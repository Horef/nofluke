"""Tests for nofluke.Study: create/open, ask/tell, duplicates and conflicts, abandon, shapes of best/status/export."""

from __future__ import annotations

import json

import pytest

from conftest import make_config
from nofluke import NoflukeError, Study
from nofluke.rng import derive


def score(a, sd=0.5):
    """Noisy score of an Ask: peak at x = 1, noise keyed by (candidate, replicate)."""
    return -(a.params["x"] - 1) ** 2 + sd * derive(5, "noise", a.candidate, a.replicate).normal()


def test_create_and_open(tmp_path):
    study = Study.create(tmp_path / "s", make_config())
    assert (tmp_path / "s" / "journal.jsonl").exists() and not (tmp_path / "s" / ".lock").exists()
    reopened = Study.open(tmp_path / "s")
    assert reopened.config == study.config
    with pytest.raises(NoflukeError, match="already exists"):
        Study.create(tmp_path / "s", make_config())
    assert Study.create(tmp_path / "s", make_config(), exist_ok=True).config == study.config
    with pytest.raises(NoflukeError, match="different config"):
        Study.create(tmp_path / "s", make_config(seed=99), exist_ok=True)
    with pytest.raises(NoflukeError, match="no study in"):
        Study.open(tmp_path / "nowhere")


def test_create_validates_config_and_strategy(tmp_path):
    with pytest.raises(NoflukeError, match="budget"):
        Study.create(tmp_path / "a", {**make_config(), "budget": 0})
    with pytest.raises(NoflukeError, match="unknown name 'nope'"):
        Study.create(tmp_path / "c", make_config(strategy={"name": "nope"}))
    with pytest.raises(NoflukeError, match="takes no options"):
        Study.create(tmp_path / "d", make_config(strategy={"options": {"a": 1}}))
    assert not (tmp_path / "c" / "journal.jsonl").exists()


def test_study_event_has_provenance(tmp_path):
    Study.create(tmp_path, make_config())
    first = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[0])
    assert first["event"] == "study" and first["format"] == 1
    assert set(first["provenance"]) == {"nofluke_version", "nofluke_commit", "python", "numpy", "optuna", "host"}
    assert first["config"] == make_config() and len(first["config_hash"]) == 64 and len(first["space_hash"]) == 64


def test_ask_returns_plain_asks_and_pending_lists_them(tmp_path):
    study = Study.create(tmp_path, make_config())
    asks = study.ask(3)
    assert [a.eval for a in asks] == [1, 2, 3] and [a.candidate for a in asks] == [1, 2, 3]
    d = asks[0].to_dict()
    assert set(d) == {"eval", "candidate", "replicate", "seed", "kind", "attempt", "params"}
    json.dumps(d)  # plain types only
    assert [a.eval for a in study.pending()] == [1, 2, 3]
    assert [a.params for a in study.pending()] == [a.params for a in asks]
    study.tell(2, 0.5)
    assert [a.eval for a in study.pending()] == [1, 3]


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_ask_rejects_bad_n(tmp_path, n):
    with pytest.raises(NoflukeError, match="positive integer"):
        Study.create(tmp_path, make_config()).ask(n)


def test_out_of_order_tells_and_final_state(tmp_path):
    study = Study.create(tmp_path, make_config(budget=6, noise={"reeval_share": 0.0}))
    asks = study.ask(6)
    for a in reversed(asks):
        assert study.tell(a.eval, score(a), info={"runtime_s": 3}) is True
    assert study.is_done() and study.ask(1) == []
    assert study.status()["ok"] == 6


def test_duplicate_tell_is_a_noop_and_conflicts_are_refused(tmp_path):
    study = Study.create(tmp_path, make_config())
    a = study.ask(1)[0]
    assert study.tell(a.eval, 1.5) is True
    size = (tmp_path / "journal.jsonl").stat().st_size
    assert study.tell(a.eval, 1.5, info={"other": 1}) is False
    assert (tmp_path / "journal.jsonl").stat().st_size == size
    with pytest.raises(NoflukeError, match="eval 1 already has a different outcome"):
        study.tell(a.eval, 2.5)
    with pytest.raises(NoflukeError, match="eval 1 already has a different outcome"):
        study.tell_failed(a.eval, "x")
    assert study.tell_many([{"eval": 1, "value": 1.5}]) == {"applied": 0, "duplicate": 1}


@pytest.mark.parametrize(
    "record, fragment",
    [
        ({"eval": 99, "value": 1.0}, "unknown evaluation 99"),
        ({"eval": 1, "value": float("nan")}, "finite"),
        ({"eval": 1, "value": float("inf")}, "finite"),
        ({"eval": 1, "value": "high"}, "finite"),
        ({"eval": 1, "value": True}, "finite"),
        ({"eval": 1}, "needs 'value'"),
        ({"eval": 1, "status": "failed"}, "needs 'reason'"),
        ({"eval": 1, "status": "ok", "value": 1, "reason": "x"}, "no 'reason'"),
        ({"eval": 1, "status": "weird", "value": 1}, "status must be"),
        ({"eval": 1, "value": 1, "extra": 2}, "unknown key 'extra'"),
        ({"value": 1}, "integer 'eval'"),
        ({"eval": 1, "value": 1, "info": [1]}, "info must be an object"),
        ("nope", "must be an object"),
    ],
)
def test_bad_tells_are_refused_and_nothing_is_written(tmp_path, record, fragment):
    study = Study.create(tmp_path, make_config())
    study.ask(1)
    size = (tmp_path / "journal.jsonl").stat().st_size
    with pytest.raises(NoflukeError, match=fragment):
        study.tell_many([{"eval": 1, "value": 0.0}, record] if record != "nope" else [record])
    assert (tmp_path / "journal.jsonl").stat().st_size == size
    assert study.status()["ok"] == 0


def test_tell_many_is_atomic_and_counts(tmp_path):
    study = Study.create(tmp_path, make_config())
    study.ask(3)
    study.tell(1, 1.0)
    summary = study.tell_many([{"eval": 1, "value": 1.0}, {"eval": 2, "value": 2.0},
                               {"eval": 3, "status": "failed", "reason": "oom"}])  # fmt: skip
    assert summary == {"applied": 2, "duplicate": 1}
    assert study.status()["failed"] == 1


def test_failed_evaluations_are_retried_once_with_the_same_seed(tmp_path):
    study = Study.create(tmp_path, make_config(max_failures=5))
    a = study.ask(1)[0]
    study.tell_failed(a.eval, "node failure")
    retry = study.ask(1)[0]
    assert (retry.kind, retry.attempt, retry.seed, retry.candidate) == ("retry", 2, a.seed, a.candidate)
    study.tell_failed(retry.eval, "node failure")
    assert study.ask(1)[0].kind == "new"


def test_max_failures_stops_asking_and_marks_done(tmp_path):
    study = Study.create(tmp_path, make_config(max_failures=2, retries=0))
    for _ in range(2):
        study.tell_failed(study.ask(1)[0].eval, "bad")
    assert study.is_done()
    with pytest.raises(NoflukeError, match="max_failures"):
        study.ask(1)


def test_failure_limit_with_pending_results_is_not_done_until_they_are_collected(tmp_path):
    study = Study.create(tmp_path, make_config(max_failures=2, retries=0))
    first, second, still_running = study.ask(3)
    study.tell_failed(first.eval, "bad")
    study.tell_failed(second.eval, "bad")
    assert not study.is_done()  # a `until nofluke done` loop must keep collecting
    with pytest.raises(NoflukeError, match="max_failures"):
        study.ask(1)
    study.tell(still_running.eval, 0.5)
    assert study.is_done()


def test_abandon_frees_the_slot_and_a_late_tell_is_accepted(tmp_path):
    study = Study.create(tmp_path, make_config(budget=2, noise={"reeval_share": 0.0}))
    a, b = study.ask(2)
    assert study.ask(1) == []
    study.abandon(a.eval, "lost job")
    assert [p.eval for p in study.pending()] == [b.eval]
    assert study.status()["abandoned"] == 1
    c = study.ask(2)
    assert len(c) == 1 and c[0].kind == "new"  # no retry for an abandoned evaluation
    assert study.tell(a.eval, 3.0) is True  # the job finished after all
    rows = {r["eval"]: r for r in study.export_rows()}
    assert rows[a.eval]["status"] == "ok" and rows[a.eval]["value"] == 3.0
    with pytest.raises(NoflukeError, match="only pending evaluations can be abandoned"):
        study.abandon(a.eval)
    with pytest.raises(NoflukeError, match="unknown evaluation"):
        study.abandon(99)


def test_late_failed_tell_of_an_abandoned_evaluation_is_not_retried(tmp_path):
    study = Study.create(tmp_path, make_config(budget=5))
    a = study.ask(1)[0]
    study.abandon(a.eval)
    study.tell_failed(a.eval, "died")
    assert study.ask(1)[0].kind == "new"


def test_abandon_older_than(tmp_path):
    study = Study.create(tmp_path, make_config())
    study.ask(3)
    study.tell(1, 1.0)
    assert study.abandon_older_than(3600) == []
    assert study.abandon_older_than(0, reason="timeout") == [2, 3]
    assert study.pending() == [] and study.status()["abandoned"] == 2


def test_best_status_and_export_shapes(tmp_path):
    study = Study.create(tmp_path, make_config(budget=30, name="demo", metric="acc"))
    while not study.is_done():
        for a in study.ask(3):
            study.tell(a.eval, score(a), info={"runtime_s": 2.5, "tag": "x"})
    best = study.best(top=3)
    assert [r["rank"] for r in best] == [1, 2, 3]
    assert set(best[0]) == {"rank", "candidate", "params", "n", "mean", "sd", "shrunk_mean", "se", "lcb"}
    assert best[0]["shrunk_mean"] >= best[1]["shrunk_mean"] >= best[2]["shrunk_mean"]
    assert all(r["lcb"] <= r["shrunk_mean"] for r in best)
    json.dumps(best)
    assert len(study.best()) == study.best_report()["n_candidates"]
    report = study.best_report(2)
    assert set(report) == {"n_candidates", "sigma", "tie_break", "best"} and len(report["best"]) == 2
    assert report["sigma"]["source"] in {"pooled", "prior_sd", "spread", "own"}
    with pytest.raises(NoflukeError, match="top"):
        study.best(top=0)

    status = study.status()
    assert status["name"] == "demo" and status["metric"] == "acc" and status["done"] is True
    assert (status["budget"], status["used"], status["remaining"], status["ok"]) == (30, 30, 0, 30)
    assert status["reevaluations"] > 0 and status["candidates"] + status["reevaluations"] == 30

    rows = study.export_rows()
    assert len(rows) == 30
    first = rows[0]
    assert list(first)[:12] == ["eval", "candidate", "replicate", "seed", "kind", "attempt", "status", "value", "reason",
                                "asked", "told", "ask"]  # fmt: skip
    assert {"p.x", "p.lr", "p.kind", "p.deep", "p.layers", "info.runtime_s", "info.tag"} <= set(first)
    assert any(r["p.layers"] is None for r in rows) and any(r["p.layers"] is not None for r in rows)
    json.dumps(rows)


def test_minimize_reports_user_units_and_an_upper_bound(tmp_path):
    study = Study.create(tmp_path, make_config(direction="minimize", budget=30))
    while not study.is_done():
        for a in study.ask(3):
            study.tell(a.eval, -score(a))  # a loss: smaller is better
    best = study.best(top=3)
    assert "ucb" in best[0] and "lcb" not in best[0]
    assert best[0]["shrunk_mean"] <= best[1]["shrunk_mean"] <= best[2]["shrunk_mean"]
    assert all(r["ucb"] >= r["shrunk_mean"] for r in best)
    top = best[0]
    values = [r["value"] for r in study.export_rows() if r["candidate"] == top["candidate"]]
    assert top["mean"] == pytest.approx(sum(values) / len(values))  # user units, not negated


def test_journal_is_append_only(tmp_path):
    study = Study.create(tmp_path, make_config())
    snapshots = [(tmp_path / "journal.jsonl").read_bytes()]
    for step in (lambda: study.ask(2), lambda: study.tell(1, 1.0), lambda: study.tell_failed(2, "x"),
                 lambda: study.ask(2), lambda: study.abandon(3)):  # fmt: skip
        step()
        snapshots.append((tmp_path / "journal.jsonl").read_bytes())
    for before, after in zip(snapshots, snapshots[1:]):
        assert after.startswith(before) and len(after) > len(before)
    assert not list(tmp_path.glob("journal.torn-*"))


def test_state_survives_reopening_and_a_torn_line(tmp_path):
    study = Study.create(tmp_path, make_config())
    study.ask(2)
    with open(tmp_path / "journal.jsonl", "ab") as handle:
        handle.write(b'{"event":"tell","eval":1,"val')  # crash in the middle of a write
    reopened = Study.open(tmp_path)
    assert [a.eval for a in reopened.pending()] == [1, 2]
    reopened.tell(1, 1.0)
    assert len(list(tmp_path.glob("journal.torn-*.jsonl"))) == 1
    assert [a.eval for a in Study.open(tmp_path).pending()] == [2]
