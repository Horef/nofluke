"""Tests for nofluke.cli through `main([...])`: full ask/tell loops with files, outputs, errors and exit codes."""

from __future__ import annotations

import csv
import io
import json
import sys

import pytest

import nofluke
from nofluke.cli import main, parse_duration
from nofluke.errors import NoflukeError
from scenario import CONFIG, evaluate


@pytest.fixture
def study_dir(tmp_path):
    """An initialised study folder (budget 36) and the path of its config file."""
    config = tmp_path / "study.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    folder = tmp_path / "runs" / "s1"
    assert main(["init", str(config), "--dir", str(folder)]) == 0
    return folder


def cli(capsys, *argv):
    """Run main and return (exit code, stdout, stderr)."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_full_loop_with_split_files(study_dir, tmp_path, capsys):
    results = tmp_path / "results"
    results.mkdir()
    batch = 0
    while cli(capsys, "done", "--dir", str(study_dir))[0] != 0:
        batch += 1
        split = tmp_path / f"batch{batch}"
        code, out, _ = cli(capsys, "ask", "--dir", str(study_dir), "-n", "4", "--split", str(split))
        assert code == 0
        asked = json.loads(out)
        assert set(asked) == {"study", "dir", "evals"}
        files = sorted(split.glob("eval-*.json"))
        assert len(files) == len(asked["evals"]) and files
        for path in files:
            ev = json.loads(path.read_text(encoding="utf-8"))  # CONTRACTS section 4: the evaluation object alone
            assert ev in asked["evals"] and path.name == f"eval-{ev['eval']}.json"
            (results / path.name).write_text(
                json.dumps({"eval": ev["eval"], "value": evaluate(ev["params"], ev["seed"]), "info": {"batch": batch}}),
                encoding="utf-8")  # fmt: skip
        code, out, _ = cli(capsys, "tell", "--dir", str(study_dir), *map(str, sorted(results.glob("eval-*.json"))))
        assert code == 0 and "already recorded" in out
        for path in files:
            (results / path.name).unlink()
    assert cli(capsys, "done", "--dir", str(study_dir))[0] == 0

    code, out, _ = cli(capsys, "status", "--dir", str(study_dir), "--json")
    status = json.loads(out)
    assert code == 0 and status["ok"] == 36 and status["done"] and status["pending"] == 0
    code, out, _ = cli(capsys, "status", "--dir", str(study_dir))
    assert "36/36" in out and "done" in out

    code, out, _ = cli(capsys, "best", "--dir", str(study_dir), "--top", "3", "--json")
    best = json.loads(out)
    assert set(best) == {"n_candidates", "sigma", "tie_break", "seed_effects", "best"} and len(best["best"]) == 3
    assert best["best"][0]["rank"] == 1 and best["best"][0]["n"] >= 2
    code, out, _ = cli(capsys, "best", "--dir", str(study_dir), "--top", "2")
    assert code == 0 and "rank" in out and "noise SD" in out and len(out.strip().splitlines()) == 4

    code, out, _ = cli(capsys, "export", "--dir", str(study_dir))
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 36 and rows[0]["eval"] == "1" and "p.x" in rows[0] and "info.batch" in rows[0]
    jsonl = tmp_path / "rows.jsonl"
    assert cli(capsys, "export", "--dir", str(study_dir), "--format", "jsonl", "--out", str(jsonl))[0] == 0
    parsed = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == 36 and parsed[0]["p.x"] == pytest.approx(float(rows[0]["p.x"]))

    code, out, _ = cli(capsys, "verify", "--dir", str(study_dir))
    assert code == 0 and out.startswith("ok")


def test_ask_out_pending_and_budget_exhaustion(study_dir, tmp_path, capsys):
    out_file = tmp_path / "ask.json"
    code, out, _ = cli(capsys, "ask", "--dir", str(study_dir), "-n", "2", "--out", str(out_file))
    assert code == 0 and out == ""
    asked = json.loads(out_file.read_text(encoding="utf-8"))
    assert [e["eval"] for e in asked["evals"]] == [1, 2]
    assert asked["evals"][0].keys() >= {"eval", "candidate", "replicate", "seed", "kind", "attempt", "params"}
    code, out, _ = cli(capsys, "pending", "--dir", str(study_dir))
    assert [e["eval"] for e in json.loads(out)["evals"]] == [1, 2]
    pending_file = tmp_path / "pending.json"
    cli(capsys, "pending", "--dir", str(study_dir), "--out", str(pending_file))
    assert json.loads(pending_file.read_text(encoding="utf-8")) == asked
    split = tmp_path / "re-split"
    cli(capsys, "pending", "--dir", str(study_dir), "--split", str(split))
    assert sorted(json.loads(f.read_text(encoding="utf-8"))["eval"] for f in split.glob("eval-*.json")) == sorted(
        e["eval"] for e in asked["evals"])  # fmt: skip
    code, out, _ = cli(capsys, "ask", "--dir", str(study_dir), "-n", "100")
    assert len(json.loads(out)["evals"]) == 34
    code, out, _ = cli(capsys, "ask", "--dir", str(study_dir))
    assert code == 0 and json.loads(out)["evals"] == []
    assert cli(capsys, "done", "--dir", str(study_dir))[0] == 1


def test_tell_single_evaluation_flags_and_formats(study_dir, tmp_path, capsys):
    cli(capsys, "ask", "--dir", str(study_dir), "-n", "6")
    assert cli(capsys, "tell", "--dir", str(study_dir), "--eval", "1", "--value", "0.5", "--info", '{"t": 3}')[0] == 0
    assert cli(capsys, "tell", "--dir", str(study_dir), "--eval", "2", "--failed", "node failure")[0] == 0
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text('{"eval": 3, "value": 1.0}\n\n{"eval": 4, "status": "failed", "reason": "oom"}\n', encoding="utf-8")
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps([{"eval": 5, "value": 2.0}]), encoding="utf-8")
    code, out, _ = cli(capsys, "tell", "--dir", str(study_dir), str(jsonl), str(listing))
    assert code == 0 and "told 3" in out
    code, out, _ = cli(capsys, "tell", "--dir", str(study_dir), str(listing))  # repeating is harmless
    assert code == 0 and "told 0 evaluation(s), 1 already" in out
    status = json.loads(cli(capsys, "status", "--dir", str(study_dir), "--json")[1])
    assert (status["ok"], status["failed"], status["pending"]) == (3, 2, 1)
    rows = list(csv.DictReader(io.StringIO(cli(capsys, "export", "--dir", str(study_dir))[1])))
    assert rows[0]["info.t"] == "3" and rows[1]["reason"] == "node failure"


def test_abandon_by_id_and_by_age(study_dir, capsys):
    cli(capsys, "ask", "--dir", str(study_dir), "-n", "4")
    code, out, _ = cli(capsys, "abandon", "--dir", str(study_dir), "--eval", "1", "--reason", "lost")
    assert code == 0 and "abandoned evaluation 1" in out
    assert cli(capsys, "abandon", "--dir", str(study_dir), "--older-than", "1h")[1].startswith("abandoned 0")
    code, out, _ = cli(capsys, "abandon", "--dir", str(study_dir), "--older-than", "0s")
    assert code == 0 and "abandoned 3 evaluation(s): 2, 3, 4" in out
    assert json.loads(cli(capsys, "pending", "--dir", str(study_dir))[1])["evals"] == []


@pytest.mark.parametrize(
    "argv, fragment",
    [
        (["tell", "--eval", "99", "--value", "1"], "unknown evaluation 99"),
        (["tell", "--eval", "1"], "exactly one of --value or --failed"),
        (["tell"], "give FILE... or --eval"),
        (["tell", "--eval", "1", "--value", "1", "x.json"], "not both"),
        (["tell", "--eval", "1", "--value", "1", "--info", "{"], "--info is not valid JSON"),
        (["tell", "missing.json"], "cannot read"),
        (["abandon"], "exactly one of"),
        (["abandon", "--older-than", "soon"], "duration must look like"),
        (["abandon", "--eval", "1", "--older-than", "1h"], "exactly one of"),
        (["best", "--top", "0"], "positive integer"),
        (["ask", "-n", "0"], "positive integer"),
    ],
)
def test_user_errors_exit_2_with_a_message_on_stderr(study_dir, capsys, argv, fragment):
    cli(capsys, "ask", "--dir", str(study_dir), "-n", "1")
    code, out, err = cli(capsys, *argv, "--dir", str(study_dir))
    assert code == 2 and out == "" and err.startswith("nofluke: error: ") and fragment in err


def test_conflicting_tell_is_an_error(study_dir, capsys):
    cli(capsys, "ask", "--dir", str(study_dir))
    assert cli(capsys, "tell", "--dir", str(study_dir), "--eval", "1", "--value", "1")[0] == 0
    code, _, err = cli(capsys, "tell", "--dir", str(study_dir), "--eval", "1", "--value", "2")
    assert code == 2 and "different outcome" in err


def test_init_errors_and_yaml(tmp_path, capsys):
    folder = tmp_path / "s"
    (tmp_path / "bad.json").write_text(json.dumps({**CONFIG, "budget": 0}), encoding="utf-8")
    code, _, err = cli(capsys, "init", str(tmp_path / "bad.json"), "--dir", str(folder))
    assert code == 2 and "'budget'" in err and not folder.exists()
    (tmp_path / "study.yaml").write_text(
        "name: demo\ndirection: minimize\nbudget: 5\nseed: 1\nspace:\n  x: {type: float, low: 0, high: 1}\n", encoding="utf-8")
    code, out, _ = cli(capsys, "init", str(tmp_path / "study.yaml"), "--dir", str(folder))
    assert code == 0 and "created study demo" in out
    code, _, err = cli(capsys, "init", str(tmp_path / "study.yaml"), "--dir", str(folder))
    assert code == 2 and "already exists" in err
    code, _, err = cli(capsys, "status", "--dir", str(tmp_path / "nothing"))
    assert code == 2 and "no study in" in err


def test_dir_defaults_to_environment_variable_then_cwd(study_dir, capsys, monkeypatch):
    monkeypatch.setenv("NOFLUKE_DIR", str(study_dir))
    assert cli(capsys, "ask")[0] == 0
    assert json.loads(cli(capsys, "status", "--json")[1])["asks"] == 1
    monkeypatch.delenv("NOFLUKE_DIR")
    monkeypatch.chdir(study_dir)
    assert json.loads(cli(capsys, "pending")[1])["evals"][0]["eval"] == 1


def test_verify_exit_status_follows_the_journal(study_dir, capsys):
    cli(capsys, "ask", "--dir", str(study_dir), "-n", "3")
    path = study_dir / "journal.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"kind":"new"', '"kind":"reeval"', 1), encoding="utf-8")
    code, out, _ = cli(capsys, "verify", "--dir", str(study_dir))
    assert code == 1 and "different evaluations" in out


def test_minimize_best_output_names_the_upper_bound(tmp_path, capsys):
    (tmp_path / "s.json").write_text(json.dumps({**CONFIG, "direction": "minimize", "budget": 12}), encoding="utf-8")
    folder = str(tmp_path / "m")
    assert cli(capsys, "init", str(tmp_path / "s.json"), "--dir", folder)[0] == 0
    for ev in json.loads(cli(capsys, "ask", "--dir", folder, "-n", "12")[1])["evals"]:
        cli(capsys, "tell", "--dir", folder, "--eval", str(ev["eval"]), "--value", str(-evaluate(ev["params"], ev["seed"])))
    best = json.loads(cli(capsys, "best", "--dir", folder, "--json")[1])["best"]
    assert "ucb" in best[0] and "lcb" not in best[0]
    assert " ucb " in cli(capsys, "best", "--dir", folder)[1]


def test_run_command_subcommand(study_dir, tmp_path, capsys):
    script = tmp_path / "evalone.py"
    script.write_text("import sys; print(sys.argv[1])\n", encoding="utf-8")
    cmd = f'"{sys.executable}" "{script}" {{seed}}'
    code, out, _ = cli(capsys, "run", "--dir", str(study_dir), "--cmd", cmd, "-j", "3", "--workdir", str(tmp_path / "w"))
    assert code == 0 and "36 ok" in out and "done" in out
    assert (tmp_path / "w" / "eval-1" / "params.json").is_file()


def test_version_and_usage(capsys):
    code, out, _ = cli(capsys, "--version")
    assert code == 0 and nofluke.__version__ in out
    code, _, err = cli(capsys, "bogus")
    assert code == 2 and "usage" in err
    assert cli(capsys, "--help")[0] == 0


@pytest.mark.parametrize("text, seconds", [("90s", 90), ("30m", 1800), ("2h", 7200), ("1.5h", 5400)])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "10", "5x", "-5m", "m"])
def test_parse_duration_rejects_garbage(text):
    with pytest.raises(NoflukeError, match="duration"):
        parse_duration(text)
