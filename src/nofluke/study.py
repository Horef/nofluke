"""The `Study` class: the Python API over one study folder (ask, tell, abandon, best, status, export, verify).

Every mutating method takes the folder lock, reads and replays the journal, validates or plans against that state,
appends events, and releases the lock, so several processes can share a study. Read-only methods replay without
locking. The journal is the only state (D5); nothing is cached between calls.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

import nofluke
from nofluke import noise
from nofluke.config import config_hash, load_config, normalise, space_hash
from nofluke.errors import NoflukeError
from nofluke.journal import FORMAT, Journal, utc_now
from nofluke.planner import plan_ask
from nofluke.space import Space
from nofluke.state import ABANDONED, FAILED, OK, PENDING, StudyState
from nofluke.strategies import get_strategy

_RECORD_KEYS = {"eval", "value", "status", "reason", "info"}


@dataclass(frozen=True)
class Ask:
    """One evaluation handed out by `Study.ask`: what to run (`params`, `seed`) and how to report it (`eval`)."""

    eval: int
    candidate: int
    replicate: int
    seed: int
    kind: str
    attempt: int
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """The CONTRACTS section 4 object for one evaluation."""
        return {"eval": self.eval, "candidate": self.candidate, "replicate": self.replicate, "seed": self.seed,
                "kind": self.kind, "attempt": self.attempt, "params": dict(self.params)}  # fmt: skip


def _git_commit() -> str | None:
    """HEAD of the git checkout this package was imported from, or None (installed copy, no git, any failure)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        return None
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5, check=False)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def _provenance() -> dict[str, Any]:
    try:
        optuna = metadata.version("optuna")
    except metadata.PackageNotFoundError:
        optuna = None
    return {"nofluke_version": nofluke.__version__, "nofluke_commit": _git_commit(), "python": platform.python_version(),
            "numpy": np.__version__, "optuna": optuna, "host": platform.node()}  # fmt: skip


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _parse_record(record: Any) -> tuple[int, str, Any, str | None, dict[str, Any] | None]:
    """Validate one tell record into (eval, status, value, reason, info)."""
    if not isinstance(record, Mapping):
        raise NoflukeError(f"tell: a record must be an object, got {record!r}")
    unknown = sorted(set(record) - _RECORD_KEYS)
    if unknown:
        raise NoflukeError(f"tell: unknown key '{unknown[0]}' in record {dict(record)!r} (allowed: {sorted(_RECORD_KEYS)})")
    eval_id = record.get("eval")
    if isinstance(eval_id, bool) or not isinstance(eval_id, int):
        raise NoflukeError(f"tell: record needs an integer 'eval', got {eval_id!r}")
    status = record.get("status", OK)
    if status == OK and ("value" not in record or "reason" in record):
        raise NoflukeError(f"tell: eval {eval_id}: an ok record needs 'value' and no 'reason'")
    if status == FAILED and ("reason" not in record or "value" in record):
        raise NoflukeError(f"tell: eval {eval_id}: a failed record needs 'reason' and no 'value'")
    return eval_id, status, record.get("value"), record.get("reason"), record.get("info")


class Study:
    """A study folder: create or open it, then ask, tell and inspect."""

    def __init__(self, directory: str | Path, config: dict[str, Any]) -> None:
        self.dir = Path(directory)
        self.config = config
        self.journal = Journal(self.dir)

    @classmethod
    def create(cls, directory: str | Path, config: str | Path | Mapping[str, Any], *, exist_ok: bool = False) -> Study:
        """Start a study in `directory` from a config file or dict; with `exist_ok`, reopen an identical study."""
        normalised = normalise(load_config(config))
        get_strategy(normalised)  # fail now, not at the first ask
        journal = Journal(directory)
        digest = config_hash(normalised)
        with journal.lock():
            if journal.exists():
                if not exist_ok:
                    raise NoflukeError(f"study already exists in '{directory}'")
                if journal.read()[0]["config_hash"] != digest:
                    raise NoflukeError(f"a study with a different config already exists in '{directory}'")
            else:
                journal.create({"event": "study", "format": FORMAT, "time": utc_now(), "config": normalised,
                                "config_hash": digest, "space_hash": space_hash(normalised["space"]),
                                "provenance": _provenance()})  # fmt: skip
        return cls(directory, normalised)

    @classmethod
    def open(cls, directory: str | Path) -> Study:
        """Open the study in `directory`."""
        return cls(directory, Journal(directory).read()[0]["config"])

    # -- internals ---------------------------------------------------------------------------------------------

    def _state(self) -> StudyState:
        return StudyState.from_events(self.journal.read())

    def _mutate(self, change: Callable[[StudyState], tuple[list[dict[str, Any]], Any]]) -> Any:
        """lock -> read -> replay -> `change(state)` -> append its events -> unlock; returns its result."""
        with self.journal.lock():
            state = self._state()
            events, result = change(state)
            if events:
                self.journal.append(events)
        return result

    def _asks(self, state: StudyState, eval_ids: list[int]) -> list[Ask]:
        out = []
        for eid in eval_ids:
            e = state.evals[eid]
            out.append(Ask(e.id, e.candidate, e.replicate, e.seed, e.kind, e.attempt, dict(state.candidates[e.candidate].params)))
        return out

    # -- ask / tell ---------------------------------------------------------------------------------------------

    def ask(self, n: int = 1) -> list[Ask]:
        """Hand out up to `n` evaluations (fewer, or none, when the budget is used up)."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise NoflukeError(f"ask: n must be a positive integer, got {n!r}")

        def change(state: StudyState) -> tuple[list[dict[str, Any]], list[Ask]]:
            candidates, evals = plan_ask(self.config, state, n)
            if not evals:
                return [], []
            event = {"event": "ask", "ask": state.n_asks + 1, "time": utc_now(), "candidates": candidates, "evals": evals}
            state.apply(event)
            return [event], self._asks(state, [e["eval"] for e in evals])

        return self._mutate(change)

    def tell_many(self, records: list[Mapping[str, Any]]) -> dict[str, int]:
        """Record several outcomes atomically (all valid or none). Returns {"applied": n, "duplicate": n}."""
        parsed = [_parse_record(r) for r in records]

        def change(state: StudyState) -> tuple[list[dict[str, Any]], dict[str, int]]:
            events: list[dict[str, Any]] = []
            summary = {"applied": 0, "duplicate": 0}
            for eval_id, status, value, reason, info in parsed:
                when = utc_now()
                outcome = state.apply_tell(eval_id, status, value, reason, info, when)
                summary[outcome] += 1
                if outcome == "applied":
                    event: dict[str, Any] = {"event": "tell", "time": when, "eval": eval_id, "status": status}
                    if status == OK:
                        event["value"] = float(value)
                    else:
                        event["reason"] = reason
                    if info is not None:
                        event["info"] = info
                    events.append(event)
            return events, summary

        return self._mutate(change)

    def tell(self, eval: int, value: float, info: Mapping[str, Any] | None = None) -> bool:
        """Report a result; False if the identical result was already recorded. Conflicts raise NoflukeError."""
        record: dict[str, Any] = {"eval": eval, "value": value}
        if info is not None:
            record["info"] = dict(info)
        return self.tell_many([record])["applied"] == 1

    def tell_failed(self, eval: int, reason: str, info: Mapping[str, Any] | None = None) -> bool:
        """Report a failed evaluation (it may be retried with the same seed); False if already recorded."""
        record: dict[str, Any] = {"eval": eval, "status": FAILED, "reason": reason}
        if info is not None:
            record["info"] = dict(info)
        return self.tell_many([record])["applied"] == 1

    def abandon(self, eval: int, reason: str = "abandoned") -> None:
        """Give up a pending evaluation: frees its budget slot, no retry. A late tell is still accepted."""

        def change(state: StudyState) -> tuple[list[dict[str, Any]], None]:
            state.apply_abandon(eval, reason)
            return [{"event": "abandon", "time": utc_now(), "eval": eval, "reason": reason}], None

        self._mutate(change)

    def abandon_older_than(self, seconds: float, reason: str = "abandoned: too old") -> list[int]:
        """Abandon every pending evaluation asked more than `seconds` ago; returns their eval ids."""

        def change(state: StudyState) -> tuple[list[dict[str, Any]], list[int]]:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
            old = [e.id for e in state.evals.values()
                   if e.status == PENDING and datetime.fromisoformat(e.asked) < cutoff]  # fmt: skip
            for eid in old:
                state.apply_abandon(eid, reason)
            return [{"event": "abandon", "time": utc_now(), "eval": eid, "reason": reason} for eid in old], old

        return self._mutate(change)

    # -- inspection ---------------------------------------------------------------------------------------------

    def pending(self) -> list[Ask]:
        """Evaluations that were asked and not yet told or abandoned, by eval id."""
        state = self._state()
        return self._asks(state, [e.id for e in state.evals.values() if e.status == PENDING])

    def is_done(self) -> bool:
        """True when nothing is pending and either the budget is used up or the failure limit is reached."""
        return self._done(self._state())

    def _done(self, state: StudyState) -> bool:
        if any(e.status == PENDING for e in state.evals.values()):
            return False  # results still to collect, even after the failure limit stops new asks
        return state.budget_used() >= self.config["budget"] or state.n_failed() >= self.config["max_failures"]

    def best_report(self, top: int | None = None) -> dict[str, Any]:
        """Ranked candidates (D9) in the user's units, with `n_candidates`, `sigma` and `tie_break` (CONTRACTS section 4)."""
        if top is not None and (isinstance(top, bool) or not isinstance(top, int) or top < 1):
            raise NoflukeError(f"best: top must be a positive integer, got {top!r}")
        state = self._state()
        model = noise.estimate(state, self.config)
        sign = 1.0 if self.config["direction"] == "maximize" else -1.0
        bound = "lcb" if sign > 0 else "ucb"
        rows = []
        for position, cid in enumerate(noise.rank(model)[:top], start=1):
            s = model.stats[cid]
            rows.append({
                "rank": position, "candidate": cid, "params": dict(state.candidates[cid].params), "n": s.n,
                "mean": sign * s.mean, "raw_mean": sign * s.raw_mean, "sd": s.sd, "shrunk_mean": sign * s.shrunk_mean, "se": s.se,
                bound: sign * s.lcb,
            })  # fmt: skip
        return {"n_candidates": len(model.stats), "sigma": {"value": model.sigma, "source": model.sigma_source},
                "tie_break": model.tie_break,
                "seed_effects": {str(r): sign * b for r, b in sorted(model.seed_effects.items())},
                "best": rows}  # fmt: skip

    def best(self, top: int | None = None) -> list[dict[str, Any]]:
        """The `top` best candidates (all if None), best first; see `best_report`."""
        return self.best_report(top)["best"]

    def status(self) -> dict[str, Any]:
        """Counts and progress as a plain dict."""
        state = self._state()
        by_status = {s: sum(1 for e in state.evals.values() if e.status == s) for s in (OK, PENDING, FAILED, ABANDONED)}
        budget = self.config["budget"]
        return {
            "name": self.config["name"], "direction": self.config["direction"], "metric": self.config["metric"],
            "budget": budget, "used": state.budget_used(), "remaining": max(0, budget - state.budget_used()),
            "ok": by_status[OK], "pending": by_status[PENDING], "failed": by_status[FAILED],
            "abandoned": by_status[ABANDONED], "asks": state.n_asks, "candidates": len(state.candidates),
            "reevaluations": sum(1 for e in state.evals.values() if e.kind == "reeval"),
            "done": self._done(state),
        }  # fmt: skip

    def export_rows(self) -> list[dict[str, Any]]:
        """One flat row per evaluation (CONTRACTS section 4): fixed columns, `p.<name>` and `info.<key>`."""
        state = self._state()
        space = Space.from_dict(self.config["space"])
        rows = []
        for e in state.evals.values():
            row: dict[str, Any] = {
                "eval": e.id, "candidate": e.candidate, "replicate": e.replicate, "seed": e.seed, "kind": e.kind,
                "attempt": e.attempt, "status": e.status, "value": e.value, "reason": e.reason, "asked": e.asked,
                "told": e.told, "ask": e.ask,
            }  # fmt: skip
            params = state.candidates[e.candidate].params
            row.update({f"p.{name}": params.get(name) for name in space.names})
            row.update({f"info.{key}": value for key, value in (e.info or {}).items()})
            rows.append(row)
        return rows

    def verify(self) -> list[str]:
        """Replay every ask against the state before it; returns the differences found (empty = reproducible)."""
        events = self.journal.read()
        study = events[0]
        problems = []
        if study.get("config_hash") != config_hash(study["config"]):
            problems.append("study event: config_hash does not match the stored config")
        if study.get("space_hash") != space_hash(study["config"]["space"]):
            problems.append("study event: space_hash does not match the stored space")
        state = StudyState(config=study["config"])
        for number, event in enumerate(events[1:], start=2):
            label = f"event {number} ({event.get('event')})"
            try:
                if event["event"] == "ask":
                    candidates, evals = plan_ask(study["config"], state, len(event["evals"]))
                    if _canonical(candidates) != _canonical(event["candidates"]):
                        problems.append(f"{label}: replay proposes different candidates than the journal records")
                    if _canonical(evals) != _canonical(event["evals"]):
                        problems.append(f"{label}: replay plans different evaluations than the journal records")
                state.apply(event)
            except (NoflukeError, KeyError, TypeError) as exc:
                problems.append(f"{label}: {exc}")
                break
        return problems
