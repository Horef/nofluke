"""Strategy registry: `get_strategy(config)` returns the candidate-proposing strategy a study asked for."""

from __future__ import annotations

from typing import Any, Mapping

from nofluke.errors import NoflukeError
from nofluke.strategies.base import History, Strategy
from nofluke.strategies.random import RandomStrategy

_BUILTIN: dict[str, type[Strategy]] = {"random": RandomStrategy}
_NEEDS_OPTUNA = ("tpe", "gp")

__all__ = ["History", "Strategy", "get_strategy"]


def get_strategy(config: Mapping[str, Any]) -> Strategy:
    """Instantiate the strategy named by `config["strategy"]`; unknown or unavailable names raise NoflukeError."""
    name = config["strategy"]["name"]
    options = config["strategy"]["options"]
    if name in _NEEDS_OPTUNA:
        raise NoflukeError(f"strategy '{name}' needs the Optuna backend, which is not available yet: install nofluke[optuna]")
    if name not in _BUILTIN:
        known = ", ".join([*_BUILTIN, *_NEEDS_OPTUNA])
        raise NoflukeError(f"strategy: unknown name {name!r} (known: {known})")
    if options:
        raise NoflukeError(f"strategy '{name}' takes no options, got {sorted(options)}")
    return _BUILTIN[name]()
