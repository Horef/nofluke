"""Tests for nofluke.rng: stable, independent, index-sensitive streams."""

from __future__ import annotations

import numpy as np

from nofluke.rng import derive, derive_int


def test_same_arguments_same_stream():
    assert np.array_equal(derive(5, "lhs").random(8), derive(5, "lhs").random(8))
    assert derive_int(5, "eval-seed", 2) == derive_int(5, "eval-seed", 2)


def test_values_are_pinned():
    # SeedSequence/PCG64 streams are stable across numpy versions; a change here is a breaking change (D7).
    assert derive_int(0, "eval-seed", 0) == 193365190
    assert derive_int(7, "candidate", 3, 1) == 509999033


def test_purposes_indices_and_seeds_are_independent():
    streams = {
        derive(1, "a").random(),
        derive(1, "b").random(),
        derive(1, "a", 0).random(),
        derive(1, "a", 1).random(),
        derive(2, "a").random(),
        derive(1, "a", 0, 1).random(),
    }
    assert len(streams) == 6


def test_derive_int_range_and_type():
    values = [derive_int(3, "x", i) for i in range(200)]
    assert all(type(v) is int and 0 <= v < 2**31 for v in values)
    assert len(set(values)) == 200


def test_global_numpy_state_is_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    derive(1, "x").random(10)
    assert np.array_equal(before, np.random.get_state()[1])
