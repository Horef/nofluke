"""Study state rebuilt by pure replay of journal events (candidates, evaluations, counters, budget).

`StudyState.from_events(events)` replays a journal (CONTRACTS section 3); `apply` adds one more event, which is how
`Study` validates a tell before appending it and how `verify` replays asks one at a time. Nothing here does I/O.
Values are stored in the user's units; the noise model converts to "higher is better".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from nofluke.errors import NoflukeError

OK, FAILED, PENDING, ABANDONED = "ok", "failed", "pending", "abandoned"


@dataclass
class Candidate:
    """A parameter assignment with the strategy and index that produced it."""

    id: int
    params: dict[str, Any]
    origin: dict[str, Any]


@dataclass
class Evaluation:
    """One run of a candidate: replicate, seed, kind, attempt, status and outcome."""

    id: int
    candidate: int
    replicate: int
    seed: int
    kind: str
    attempt: int
    status: str
    asked: str
    ask: int
    value: float | None = None
    reason: str | None = None
    info: dict[str, Any] | None = None
    told: str | None = None
    abandoned: bool = False


@dataclass
class StudyState:
    """Replayed state of a study."""

    config: dict[str, Any]
    candidates: dict[int, Candidate] = field(default_factory=dict)
    evals: dict[int, Evaluation] = field(default_factory=dict)
    next_candidate: int = 1
    next_eval: int = 1
    n_asks: int = 0

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> StudyState:
        """Replay a full journal; the first event must be the study event."""
        if not events or events[0].get("event") != "study":
            raise NoflukeError("journal: first event must be a study event")
        state = cls(config=events[0]["config"])
        for number, event in enumerate(events[1:], start=2):
            try:
                state.apply(event)
            except NoflukeError as exc:
                raise NoflukeError(f"journal event {number} ({event.get('event')}): {exc}") from None
            except (KeyError, TypeError) as exc:
                raise NoflukeError(f"journal event {number} ({event.get('event')}): malformed, missing {exc}") from None
        return state

    def apply(self, event: Mapping[str, Any]) -> str:
        """Apply one ask, tell or abandon event; returns "applied" or, for a repeated identical tell, "duplicate"."""
        kind = event["event"]
        if kind == "ask":
            self._apply_ask(event)
        elif kind == "tell":
            return self.apply_tell(
                event["eval"], event["status"], event.get("value"), event.get("reason"), event.get("info"), event["time"]
            )
        elif kind == "abandon":
            self.apply_abandon(event["eval"], event["reason"])
        else:
            raise NoflukeError(f"unknown event kind {kind!r}")
        return "applied"

    def _apply_ask(self, event: Mapping[str, Any]) -> None:
        if event["ask"] != self.n_asks + 1:
            raise NoflukeError(f"ask number {event['ask']} out of sequence (expected {self.n_asks + 1})")
        for c in event["candidates"]:
            if c["candidate"] != self.next_candidate:
                raise NoflukeError(f"candidate id {c['candidate']} out of sequence (expected {self.next_candidate})")
            self.candidates[c["candidate"]] = Candidate(c["candidate"], c["params"], c["origin"])
            self.next_candidate += 1
        for e in event["evals"]:
            if e["eval"] != self.next_eval:
                raise NoflukeError(f"eval id {e['eval']} out of sequence (expected {self.next_eval})")
            if e["candidate"] not in self.candidates:
                raise NoflukeError(f"eval {e['eval']} names unknown candidate {e['candidate']}")
            self.evals[e["eval"]] = Evaluation(
                id=e["eval"], candidate=e["candidate"], replicate=e["replicate"], seed=e["seed"], kind=e["kind"],
                attempt=e["attempt"], status=PENDING, asked=event["time"], ask=event["ask"],
            )  # fmt: skip
            self.next_eval += 1
        self.n_asks += 1

    def _eval(self, eval_id: Any) -> Evaluation:
        if eval_id not in self.evals:
            raise NoflukeError(f"unknown evaluation {eval_id!r}")
        return self.evals[eval_id]

    def apply_tell(
        self, eval_id: int, status: str, value: float | None, reason: str | None, info: dict[str, Any] | None, time: str
    ) -> str:
        """Record an outcome. Identical repeat -> "duplicate"; conflicting repeat, unknown eval or bad value -> error."""
        ev = self._eval(eval_id)
        if info is not None and not isinstance(info, dict):
            raise NoflukeError(f"eval {eval_id}: info must be an object, got {info!r}")
        if status == OK:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise NoflukeError(f"eval {eval_id}: value must be a finite number, got {value!r}")
            value = float(value)
        elif status == FAILED:
            if not isinstance(reason, str):
                raise NoflukeError(f"eval {eval_id}: a failed tell needs a string reason, got {reason!r}")
        else:
            raise NoflukeError(f"eval {eval_id}: status must be 'ok' or 'failed', got {status!r}")
        if ev.status in (OK, FAILED):
            if ev.status == status and ev.value == value and ev.reason == reason:
                return "duplicate"
            raise NoflukeError(
                f"eval {eval_id} already has a different outcome ({ev.status}, "
                f"{ev.value if ev.status == OK else ev.reason!r}); conflicting tell refused"
            )
        ev.status, ev.value, ev.reason, ev.info, ev.told = status, value, reason, info, time
        return "applied"

    def apply_abandon(self, eval_id: int, reason: str) -> None:
        """Give up a pending evaluation (frees its budget slot, no retry)."""
        ev = self._eval(eval_id)
        if ev.status != PENDING:
            raise NoflukeError(f"eval {eval_id} is {ev.status}, only pending evaluations can be abandoned")
        ev.status, ev.reason, ev.abandoned = ABANDONED, reason, True

    def budget_used(self) -> int:
        """Evaluations that are ok or pending."""
        return sum(1 for e in self.evals.values() if e.status in (OK, PENDING))

    def n_failed(self) -> int:
        return sum(1 for e in self.evals.values() if e.status == FAILED)

    def retry_queue(self) -> list[Evaluation]:
        """Failed, never-abandoned evaluations with attempts left that have not been re-issued, by eval id."""
        retries = self.config["retries"]
        issued = {(e.candidate, e.replicate, e.attempt) for e in self.evals.values()}
        return [
            e
            for e in self.evals.values()
            if e.status == FAILED
            and not e.abandoned
            and e.attempt <= retries
            and (e.candidate, e.replicate, e.attempt + 1) not in issued
        ]
