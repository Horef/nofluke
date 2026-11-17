"""Benchmark problems: noise-free `true(params)` plus a noisy `observe(...)` with common-random-number structure.

Every problem is maximised (nofluke maximises, so Branin and Hartmann6 are negated) and offers:
`space` (a nofluke space dict), `true(params)` (noise-free value), `optimum` (known or computed value, None if
unknown), `true_sd` (SD of `true` over 10,000 uniform points, drawn deterministically) and
`observe(params, eval_seed, candidate, seed, noise_ratio)`.

Noise (synthetic problems): total SD = `noise_ratio` x `true_sd`, split by variance into a *seed-level offset* shared
by every candidate evaluated with the same eval seed (`SHARED_VARIANCE_SHARE` of the variance; this is what
common random numbers, D10, share in a real training run) and an *independent* per-evaluation term keyed by
(seed, eval seed, candidate). Both come from `nofluke.rng.derive`, so a run is a pure function of its arguments.
`sklearn_svm` has real noise instead (a seed-dependent train/validation split) and needs scikit-learn.

Use `get_problem(name)`; `available_problems()` lists the names that can run here.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from nofluke.rng import derive
from nofluke.space import Space

SHARED_VARIANCE_SHARE = 0.5  # fraction of the noise variance that is a seed-level offset shared across candidates
SD_POINTS = 10_000  # uniform points used to measure the SD of `true`
NOISE_RATIOS = (0.25, 0.5, 1.0)


class Problem:
    """Base class: subclasses set `name`, `space`, `continuous`, `optimum` and implement `true`."""

    name: str = ""
    space: dict[str, Any] = {}
    continuous: bool = True  # only continuous problems run with the gp strategy
    noisy_natively: bool = False  # True when `observe` has its own noise and ignores `noise_ratio`
    optimum: float | None = None
    optimum_params: dict[str, Any] | None = None

    def true(self, params: dict[str, Any]) -> float:
        raise NotImplementedError

    @property
    def true_sd(self) -> float:
        """SD of `true` over `SD_POINTS` uniform points of the space (deterministic)."""
        return _true_sd(self.name)

    def observe(self, params: dict[str, Any], eval_seed: int, candidate: int, seed: int, noise_ratio: float | None) -> float:
        """One noisy observation: `true` plus the shared-offset and independent noise terms described above."""
        if noise_ratio is None:
            raise ValueError(f"problem '{self.name}' needs a noise ratio")
        sd = noise_ratio * self.true_sd
        shared = derive(seed, "bench-noise-shared", eval_seed).normal()
        own = derive(seed, "bench-noise-eval", eval_seed, candidate).normal()
        noise = math.sqrt(SHARED_VARIANCE_SHARE) * shared + math.sqrt(1.0 - SHARED_VARIANCE_SHARE) * own
        return self.true(params) + sd * float(noise)


# -- Branin ---------------------------------------------------------------------------------------------------

BRANIN_MIN = 0.397887
BRANIN_ARGMIN = ({"x1": -math.pi, "x2": 12.275}, {"x1": math.pi, "x2": 2.275}, {"x1": 9.42478, "x2": 2.475})


def branin(x1: float, x2: float) -> float:
    """Standard Branin-Hoo function (minimum 0.397887 at three points) on x1 in [-5, 10], x2 in [0, 15]."""
    b, c, r = 5.1 / (4 * math.pi**2), 5 / math.pi, 6.0
    return (x2 - b * x1**2 + c * x1 - r) ** 2 + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1) + 10


class Branin(Problem):
    """Branin, negated: 2-D continuous, maximum -0.397887 at three points."""

    name = "branin"
    space = {"x1": {"type": "float", "low": -5.0, "high": 10.0}, "x2": {"type": "float", "low": 0.0, "high": 15.0}}
    optimum = -BRANIN_MIN
    optimum_params = BRANIN_ARGMIN[0]

    def true(self, params: dict[str, Any]) -> float:
        return -branin(params["x1"], params["x2"])


# -- Hartmann6 ------------------------------------------------------------------------------------------------

_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])  # fmt: skip
HARTMANN6_MIN = -3.32237
HARTMANN6_ARGMIN = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)


def hartmann6(x: Any) -> float:
    """Standard Hartmann 6-D function on [0, 1]^6 (minimum about -3.32237)."""
    x = np.asarray(x, dtype=float)
    return float(-np.sum(_H_ALPHA * np.exp(-np.sum(_H_A * (x - _H_P) ** 2, axis=1))))


class Hartmann6(Problem):
    """Hartmann6, negated: 6-D continuous, maximum 3.32237."""

    name = "hartmann6"
    space = {f"x{i}": {"type": "float", "low": 0.0, "high": 1.0} for i in range(1, 7)}
    optimum = -HARTMANN6_MIN
    optimum_params = {f"x{i}": v for i, v in enumerate(HARTMANN6_ARGMIN, start=1)}

    def true(self, params: dict[str, Any]) -> float:
        return -hartmann6([params[f"x{i}"] for i in range(1, 7)])


# -- mixed_hpo ------------------------------------------------------------------------------------------------

_BATCH_OFFSET = {16: -0.005, 32: 0.004, 64: 0.0}
_EPOCH_RANGE = (20, 200)


def _epochs_term(epochs: float) -> float:
    """Learning gain that saturates, minus an overfitting penalty growing with epochs (interior optimum near 65)."""
    return 0.06 * (1 - math.exp(-epochs / 25)) - 0.07 * ((epochs - 20) / 180) ** 2


def _lr_term(lr: float, batch: int) -> float:
    """Quadratic in log10(lr) around an optimum that shifts with the batch size (0.25 decades per doubling)."""
    return -0.10 * (math.log10(lr) - (-3.0 + 0.25 * math.log2(batch / 32))) ** 2


def _w_term(loss_w: float) -> float:
    return -0.012 * (math.log10(loss_w) - math.log10(0.3)) ** 2


def _aux_term(aux_w: float) -> float:
    """The auxiliary loss helps only when its weight is near 10^0.5 and hurts at the ends of its range."""
    return 0.015 - 0.01 * (math.log10(aux_w) - 0.5) ** 2


class MixedHpo(Problem):
    """A synthetic training run shaped like the first client: log floats, int, choice, bool and a conditional float.

    Accuracy-like (about 0.6-0.8): `loss_w` and `lr` (log floats, interior optima, `lr`'s optimum moves with
    `batch`), `epochs` (int, overfitting optimum), `batch` (choice with small offsets), `aux` (bool) and `aux_w`
    (log float, only when `aux` is true). The optimum is computed numerically over the structure (see `_solve`).
    """

    name = "mixed_hpo"
    continuous = False
    space = {
        "loss_w": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
        "lr": {"type": "float", "low": 1e-4, "high": 3e-2, "log": True},
        "epochs": {"type": "int", "low": _EPOCH_RANGE[0], "high": _EPOCH_RANGE[1]},
        "batch": {"type": "choice", "values": [16, 32, 64]},
        "aux": {"type": "bool"},
        "aux_w": {"type": "float", "low": 0.1, "high": 100.0, "log": True, "when": {"aux": True}},
    }
    _BASE = 0.72

    def true(self, params: dict[str, Any]) -> float:
        value = (self._BASE + _w_term(params["loss_w"]) + _lr_term(params["lr"], params["batch"])
                 + _epochs_term(params["epochs"]) + _BATCH_OFFSET[params["batch"]])  # fmt: skip
        if params["aux"]:
            value += _aux_term(params["aux_w"])
        return value

    def __init__(self) -> None:
        self.optimum, self.optimum_params = self._solve()

    def _solve(self) -> tuple[float, dict[str, Any]]:
        """Maximise numerically: the terms separate given (batch, aux), so grid each one densely and enumerate."""
        grid = lambda low, high: np.exp(np.linspace(math.log(low), math.log(high), 20001))  # noqa: E731
        spec = self.space

        def best_of(term: Callable[[float], float], values: Any) -> tuple[float, Any]:
            scored = [(term(float(v)), v) for v in values]
            return max(scored, key=lambda s: s[0])

        w_val, w_best = best_of(_w_term, grid(spec["loss_w"]["low"], spec["loss_w"]["high"]))
        e_val, e_best = best_of(_epochs_term, range(_EPOCH_RANGE[0], _EPOCH_RANGE[1] + 1))
        a_val, a_best = best_of(_aux_term, grid(spec["aux_w"]["low"], spec["aux_w"]["high"]))
        lr_values = grid(spec["lr"]["low"], spec["lr"]["high"])
        best: tuple[float, dict[str, Any]] | None = None
        for batch in spec["batch"]["values"]:
            lr_val, lr_best = best_of(lambda lr, b=batch: _lr_term(lr, b), lr_values)
            for aux in (False, True):
                total = self._BASE + w_val + lr_val + e_val + _BATCH_OFFSET[batch] + (a_val if aux else 0.0)
                if best is None or total > best[0]:
                    params = {"loss_w": float(w_best), "lr": float(lr_best), "epochs": int(e_best), "batch": batch, "aux": aux}
                    if aux:
                        params["aux_w"] = float(a_best)
                    best = (total, params)
        assert best is not None
        return best


# -- sklearn_svm ----------------------------------------------------------------------------------------------

_SVM_TRAIN, _SVM_VAL, _SVM_TRUE_SPLITS = 400, 400, 12


class SklearnSvm(Problem):
    """An SVM on the scikit-learn digits data, scored by validation accuracy on a seed-dependent split (real noise).

    `observe(...)` draws a train/validation split from the eval seed (shared by all candidates, so it is a true
    common-random-numbers design) and ignores `noise_ratio`. `true` is the mean accuracy over `_SVM_TRUE_SPLITS`
    fixed other splits. The optimum is unknown (`optimum` is None); `summarise.py` uses the best true value seen.
    """

    name = "sklearn_svm"
    continuous = False
    noisy_natively = True
    space = {
        "C": {"type": "float", "low": 1e-2, "high": 1e3, "log": True},
        "gamma": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
        "kernel": {"type": "choice", "values": ["rbf", "poly", "sigmoid"]},
        "degree": {"type": "int", "low": 2, "high": 4, "when": {"kernel": "poly"}},
    }

    @property
    def true_sd(self) -> float:
        return math.nan  # not measured: the noise is native, and 10,000 points would cost hours

    def __init__(self) -> None:
        from sklearn.datasets import load_digits  # imported here so the module loads without scikit-learn

        digits = load_digits()
        self._x = digits.data / 16.0
        self._y = digits.target
        self._true_cache: dict[tuple, float] = {}

    def _accuracy(self, params: dict[str, Any], split_seed: int) -> float:
        from sklearn.svm import SVC

        order = derive(split_seed, "bench-split").permutation(len(self._y))
        train, val = order[:_SVM_TRAIN], order[_SVM_TRAIN : _SVM_TRAIN + _SVM_VAL]
        model = SVC(C=params["C"], gamma=params["gamma"], kernel=params["kernel"], degree=params.get("degree", 3))
        model.fit(self._x[train], self._y[train])
        return float(np.mean(model.predict(self._x[val]) == self._y[val]))

    def true(self, params: dict[str, Any]) -> float:
        key = tuple(sorted(params.items()))
        if key not in self._true_cache:
            self._true_cache[key] = float(np.mean([self._accuracy(params, 10**6 + i) for i in range(_SVM_TRUE_SPLITS)]))
        return self._true_cache[key]

    def observe(self, params: dict[str, Any], eval_seed: int, candidate: int, seed: int, noise_ratio: float | None) -> float:
        return self._accuracy(params, eval_seed)


# -- registry -------------------------------------------------------------------------------------------------

_BUILDERS: dict[str, Callable[[], Problem]] = {
    "branin": Branin, "hartmann6": Hartmann6, "mixed_hpo": MixedHpo, "sklearn_svm": SklearnSvm,
}  # fmt: skip


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def available_problems() -> list[str]:
    """Names of the problems that can run in this environment (`sklearn_svm` needs scikit-learn)."""
    return [n for n in _BUILDERS if n != "sklearn_svm" or sklearn_available()]


@lru_cache(maxsize=None)
def get_problem(name: str) -> Problem:
    """The (cached) problem instance for `name`; raises KeyError for unknown names, ImportError for missing extras."""
    if name not in _BUILDERS:
        raise KeyError(f"unknown problem '{name}' (known: {', '.join(_BUILDERS)})")
    return _BUILDERS[name]()


@lru_cache(maxsize=None)
def _true_sd(name: str) -> float:
    """SD of `true` over `SD_POINTS` points drawn uniformly (in the space's own encoding) with a fixed seed."""
    problem = get_problem(name)
    space = Space.from_dict(problem.space)
    points = derive(0, f"bench-true-sd-{name}").random((SD_POINTS, len(space)))
    return float(np.std([problem.true(space.decode(u)) for u in points], ddof=1))
