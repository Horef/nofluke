"""The append-only JSONL journal of one study folder, with a cross-platform lock file.

`Journal(dir)` reads and appends events (CONTRACTS section 3). A line without a trailing newline was never
acknowledged: reads ignore it and the next append moves it to `journal.torn-<UTC time>.jsonl` before writing (D5).
Writers wrap read-modify-append in `with journal.lock():`, an `O_EXCL` lock file that works on macOS, Linux and
Windows and is broken with a warning after 300 s. This is the only module that touches the study folder's files.
"""

from __future__ import annotations

import json
import os
import platform
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from nofluke.errors import NoflukeError

FORMAT = 1
_POLL_SECONDS = 0.1


def utc_now() -> str:
    """Current UTC time as ISO 8601 with a trailing Z (millisecond precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _dumps(event: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(event, separators=(",", ":"), allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NoflukeError(f"journal: event is not JSON-serialisable ({exc}): {event!r}") from None


class Journal:
    """Reader/appender for `<dir>/journal.jsonl`."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.path = self.dir / "journal.jsonl"
        self.lock_path = self.dir / ".lock"

    def exists(self) -> bool:
        return self.path.is_file()

    def create(self, study_event: dict[str, Any]) -> None:
        """Create the folder and the journal with its first (study) event; fail if a journal exists."""
        self.dir.mkdir(parents=True, exist_ok=True)
        data = _dumps(study_event)
        try:
            with open(self.path, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise NoflukeError(f"study already exists in '{self.dir}'") from None

    def read(self) -> list[dict[str, Any]]:
        """All acknowledged events in order; validates the format version of the first one."""
        if not self.exists():
            raise NoflukeError(f"no study in '{self.dir}' (journal.jsonl not found)")
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        lines.pop()  # the segment after the last newline: empty, or an unacknowledged torn line
        events: list[dict[str, Any]] = []
        for number, line in enumerate(lines, start=1):
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NoflukeError(f"journal '{self.path}': line {number} is not valid JSON ({exc})") from None
            if not isinstance(event, dict):
                raise NoflukeError(f"journal '{self.path}': line {number} is not a JSON object")
            events.append(event)
        if not events:
            raise NoflukeError(f"journal '{self.path}' has no study event")
        first = events[0]
        if first.get("event") != "study" or first.get("format") != FORMAT:
            raise NoflukeError(
                f"journal '{self.path}': first event must be a study event with format {FORMAT}, "
                f"got event={first.get('event')!r} format={first.get('format')!r}"
            )
        return events

    def append(self, events: list[dict[str, Any]]) -> None:
        """Append events durably (flush + fsync); first moves an unacknowledged last line aside. Hold the lock."""
        data = b"".join(_dumps(e) for e in events)
        self._move_torn_tail()
        with open(self.path, "ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def _move_torn_tail(self) -> None:
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        keep = raw.rfind(b"\n") + 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        torn = self.dir / f"journal.torn-{stamp}.jsonl"
        with open(torn, "xb") as handle:
            handle.write(raw[keep:])
            handle.flush()
            os.fsync(handle.fileno())
        with open(self.path, "r+b") as handle:  # the one sanctioned truncation (D5)
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def lock(self, timeout: float = 30.0, stale_after: float = 300.0) -> Iterator[None]:
        """Exclusive lock on the study folder; waits up to `timeout` s, breaks locks older than `stale_after` s."""
        self.dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._break_if_stale(stale_after):
                    continue
                if time.monotonic() >= deadline:
                    raise NoflukeError(
                        f"could not lock study '{self.dir}' within {timeout:g} s; held by {self._holder()} "
                        f"(delete '{self.lock_path}' if that process is gone)"
                    ) from None
                time.sleep(_POLL_SECONDS)
                continue
            break
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "host": platform.node(), "time": utc_now()}))
            yield
        finally:
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass

    def _break_if_stale(self, stale_after: float) -> bool:
        try:
            age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return True  # released meanwhile: just retry
        if age <= stale_after:
            return False
        warnings.warn(f"breaking stale lock '{self.lock_path}' ({age:.0f} s old, held by {self._holder()})", stacklevel=3)
        try:
            os.unlink(self.lock_path)
        except OSError:
            return False
        return True

    def _holder(self) -> str:
        try:
            return self.lock_path.read_text(encoding="utf-8").strip() or "unknown"
        except OSError:
            return "unknown"
