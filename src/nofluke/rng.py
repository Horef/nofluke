"""Seed derivation: every random draw in nofluke comes from `derive(study_seed, purpose, *indices)`.

A generator is a pure function of the study seed, a purpose string (for example "lhs", "candidate", "eval-seed") and
integer indices, so replaying a journal reproduces every draw without storing generator state (D7). Never use
`np.random.*` module functions or `random` anywhere in the package.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _purpose_code(purpose: str) -> int:
    """Stable integer for a purpose string: the first 8 bytes of its sha256 digest."""
    return int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:8], "big")


def derive(seed: int, purpose: str, *indices: int) -> np.random.Generator:
    """Return a fresh generator for (seed, purpose, indices); equal arguments give equal streams."""
    sequence = np.random.SeedSequence(entropy=seed, spawn_key=(_purpose_code(purpose), *indices))
    return np.random.default_rng(sequence)


def derive_int(seed: int, purpose: str, *indices: int) -> int:
    """Return an integer in [0, 2**31) derived like `derive`, as a plain Python int."""
    return int(derive(seed, purpose, *indices).integers(0, 2**31))
