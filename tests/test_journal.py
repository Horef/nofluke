"""Tests for nofluke.journal: append/read, torn lines, malformed lines, lock contention and stale locks."""

from __future__ import annotations

import os
import threading
import time

import pytest

from nofluke.errors import NoflukeError
from nofluke.journal import Journal, utc_now

STUDY = {"event": "study", "format": 1, "time": "t", "config": {}}


def test_create_append_read(tmp_path):
    j = Journal(tmp_path / "s")
    assert not j.exists()
    j.create(STUDY)
    assert j.exists()
    j.append([{"event": "tell", "eval": 1}, {"event": "tell", "eval": 2}])
    j.append([{"event": "tell", "eval": 3}])
    assert [e.get("eval") for e in j.read()] == [None, 1, 2, 3]
    assert j.path.read_bytes().endswith(b"\n")
    with pytest.raises(NoflukeError, match="already exists"):
        j.create(STUDY)


def test_read_without_study_names_the_folder(tmp_path):
    with pytest.raises(NoflukeError, match="no study in"):
        Journal(tmp_path).read()


def test_torn_last_line_is_ignored_then_moved_aside(tmp_path):
    j = Journal(tmp_path)
    j.create(STUDY)
    j.append([{"event": "tell", "eval": 1}])
    good = j.path.read_bytes()
    with open(j.path, "ab") as handle:
        handle.write(b'{"event":"tell","ev')
    assert len(j.read()) == 2  # the incomplete line was never acknowledged
    j.append([{"event": "tell", "eval": 2}])
    torn = list(tmp_path.glob("journal.torn-*.jsonl"))
    assert len(torn) == 1 and torn[0].read_bytes() == b'{"event":"tell","ev'
    assert j.path.read_bytes().startswith(good)
    assert [e.get("eval") for e in j.read()] == [None, 1, 2]


def test_complete_but_unterminated_json_line_is_also_unacknowledged(tmp_path):
    j = Journal(tmp_path)
    j.create(STUDY)
    with open(j.path, "ab") as handle:
        handle.write(b'{"event":"tell","eval":1}')
    assert len(j.read()) == 1


def test_malformed_complete_line_raises_with_line_number(tmp_path):
    j = Journal(tmp_path)
    j.create(STUDY)
    with open(j.path, "ab") as handle:
        handle.write(b"not json\n" + b'{"event":"tell","eval":1}\n')
    with pytest.raises(NoflukeError, match="line 2"):
        j.read()


@pytest.mark.parametrize("first", [b'{"event":"study","format":2}\n', b'{"event":"tell","format":1}\n', b"[1]\n"])
def test_first_event_must_be_a_format_1_study(tmp_path, first):
    (tmp_path / "journal.jsonl").write_bytes(first)
    with pytest.raises(NoflukeError):
        Journal(tmp_path).read()


def test_non_serialisable_events_are_refused_before_writing(tmp_path):
    import numpy as np

    j = Journal(tmp_path)
    j.create(STUDY)
    before = j.path.read_bytes()
    for bad in ({"event": "tell", "value": np.int64(1)}, {"event": "tell", "value": float("nan")}):
        with pytest.raises(NoflukeError, match="not JSON-serialisable"):
            j.append([{"event": "tell", "eval": 1}, bad])
    assert j.path.read_bytes() == before


def test_lock_excludes_and_releases(tmp_path):
    j = Journal(tmp_path)
    with j.lock():
        assert j.lock_path.exists()
        with pytest.raises(NoflukeError, match="could not lock"):
            with Journal(tmp_path).lock(timeout=0.3):
                pass
    assert not j.lock_path.exists()
    with j.lock(timeout=0.3):
        pass


def test_lock_waits_for_a_holder_that_releases(tmp_path):
    j = Journal(tmp_path)
    order = []

    def hold():
        with j.lock():
            order.append("holder in")
            time.sleep(0.4)
            order.append("holder out")

    thread = threading.Thread(target=hold)
    thread.start()
    time.sleep(0.15)
    with Journal(tmp_path).lock(timeout=5):
        order.append("waiter in")
    thread.join()
    assert order == ["holder in", "holder out", "waiter in"]


def test_stale_lock_is_broken_with_a_warning(tmp_path):
    j = Journal(tmp_path)
    j.dir.mkdir(exist_ok=True)
    j.lock_path.write_text("{}", encoding="utf-8")
    old = time.time() - 1000
    os.utime(j.lock_path, (old, old))
    with pytest.warns(UserWarning, match="stale lock"):
        with j.lock(timeout=1):
            pass
    assert not j.lock_path.exists()


def test_utc_now_format():
    stamp = utc_now()
    assert stamp.endswith("Z") and "T" in stamp and "+" not in stamp
