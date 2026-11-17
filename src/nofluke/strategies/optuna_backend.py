"""Optuna strategy backend (`tpe`, `gp`): proposes with an Optuna sampler on a study rebuilt from the history (D3, D7, D11).

This is the only module that imports Optuna. At each `propose` it builds a fresh in-memory study, replays the history
into it (completed candidates as COMPLETE trials with their mean score, pending ones as RUNNING so the samplers'
constant liar / pending-aware acquisition sees them) and asks for one trial. Failed candidates are not replayed:
Optuna's TPE and GP samplers ignore FAIL trials. The sampler is seeded from
the candidate's own rng, so a proposal is a pure function of (history, rng, index). `strategy.options` are extra
keyword arguments of the Optuna sampler (`seed` excluded). Choices and booleans are given to Optuna as category
indices, so values that compare equal in Python (1, 1.0, True) stay distinct.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import Trial, TrialState, create_trial

from nofluke.errors import NoflukeError
from nofluke.space import BoolParam, ChoiceParam, FloatParam, IntParam, Param, Space
from nofluke.strategies.base import History, Strategy

# The Latin-hypercube start already explores, and the models need a few real results to be better than random.
_N_STARTUP = 4

_DEFAULTS: dict[str, dict[str, Any]] = {
    "tpe": {"multivariate": True, "constant_liar": True, "n_startup_trials": _N_STARTUP},
    "gp": {"n_startup_trials": _N_STARTUP},
}


def _categories(param: Param) -> tuple[Any, ...]:
    return param.values if isinstance(param, ChoiceParam) else (False, True)


def _distribution(param: Param) -> BaseDistribution:
    """The Optuna distribution of one parameter (categories are indices into `_categories`)."""
    if isinstance(param, FloatParam):
        return FloatDistribution(param.low, param.high, log=param.log)
    if isinstance(param, IntParam):
        return IntDistribution(param.low, param.high, log=param.log)
    if isinstance(param, (ChoiceParam, BoolParam)):
        return CategoricalDistribution(tuple(range(len(_categories(param)))))
    raise NoflukeError(f"strategy: the Optuna backend does not support parameter type {param.type!r}")


def _to_trial_params(space: Space, params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, BaseDistribution]]:
    """Trial params and distributions of the parameters that are active in `params`."""
    values: dict[str, Any] = {}
    distributions: dict[str, BaseDistribution] = {}
    for param in space.params:
        if param.name in params:
            value = params[param.name]
            if isinstance(param, ChoiceParam):
                value = param.index(value)  # type-aware: 1, 1.0 and True are different categories
            elif isinstance(param, BoolParam):
                value = int(bool(value))
            values[param.name] = value
            distributions[param.name] = _distribution(param)
    return values, distributions


def _suggest(trial: Trial, space: Space) -> dict[str, Any]:
    """Define-by-run suggestion in declared order, skipping parameters whose `when` condition fails."""
    params: dict[str, Any] = {}
    for param in space.params:
        if not param.active(params):
            continue
        if isinstance(param, FloatParam):
            params[param.name] = trial.suggest_float(param.name, param.low, param.high, log=param.log)
        elif isinstance(param, IntParam):
            params[param.name] = trial.suggest_int(param.name, param.low, param.high, log=param.log)
        elif isinstance(param, (ChoiceParam, BoolParam)):
            categories = _categories(param)
            params[param.name] = categories[trial.suggest_categorical(param.name, list(range(len(categories))))]
        else:
            raise NoflukeError(f"strategy: the Optuna backend does not support parameter type {param.type!r}")
    return params


class OptunaStrategy(Strategy):
    """TPE or GP proposals from a study rebuilt on every call."""

    def __init__(self, name: str, options: Mapping[str, Any]) -> None:
        if name not in _DEFAULTS:
            raise NoflukeError(f"strategy: the Optuna backend knows {sorted(_DEFAULTS)}, not {name!r}")
        if "seed" in options:
            raise NoflukeError(f"strategy '{name}': 'seed' is not an option; it is derived from the study seed (D7)")
        self.name = name
        self._kwargs = {**_DEFAULTS[name], **options}
        self._sampler(0, group=False)  # fail now on unknown options or a missing package
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _sampler(self, seed: int, group: bool) -> optuna.samplers.BaseSampler:
        kwargs = {**self._kwargs, "seed": seed}
        if self.name == "tpe":
            kwargs.setdefault("group", group)
        cls = optuna.samplers.TPESampler if self.name == "tpe" else optuna.samplers.GPSampler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            try:
                return cls(**kwargs)
            except TypeError as exc:
                raise NoflukeError(f"strategy '{self.name}': bad options: {exc}") from None

    def propose(self, history: History, rng: np.random.Generator, index: int) -> dict[str, Any]:
        space = history.space
        seed = int(rng.integers(0, 2**31))
        has_conditions = any(param.when for param in space.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            study = optuna.create_study(sampler=self._sampler(seed, group=has_conditions), direction="maximize")
            for _, params, score, _ in history.completed:
                values, distributions = _to_trial_params(space, params)
                study.add_trial(create_trial(value=score, params=values, distributions=distributions))
            for params in history.pending:
                values, distributions = _to_trial_params(space, params)
                study.add_trial(create_trial(state=TrialState.RUNNING, params=values, distributions=distributions))
            return space.validate_params(_suggest(study.ask(), space))
