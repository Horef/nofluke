"""Latin-hypercube design for the first `n_init` candidates of every study (D11)."""

from __future__ import annotations

import numpy as np


def lhs_design(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """An n x d array in [0, 1): each column holds one point per stratum [i/n, (i+1)/n), jittered, in random order."""
    columns = [(rng.permutation(n) + rng.random(n)) / n for _ in range(d)]
    return np.column_stack(columns) if columns else np.empty((n, 0))
