"""Strategy registry: `get_strategy(config)` returns the candidate-proposing strategy a study asked for."""

from __future__ import annotations

from typing import Any, Mapping

from nofluke.errors import NoflukeError
from nofluke.strategies.base import History, Strategy
from nofluke.strategies.random import RandomStrategy

_BUILTIN: dict[str, type[Strategy]] = {"random": RandomStrategy}
_OPTUNA_EXTRAS = {"tpe": "optuna", "gp": "gp"}

__all__ = ["History", "Strategy", "get_strategy"]


def get_strategy(config: Mapping[str, Any]) -> Strategy:
    """Instantiate the strategy named by `config["strategy"]`; unknown or unavailable names raise NoflukeError."""
    name = config["strategy"]["name"]
    options = config["strategy"]["options"]
    if name in _OPTUNA_EXTRAS:
        try:
            from nofluke.strategies.optuna_backend import OptunaStrategy

            return OptunaStrategy(name, options)
        except ImportError as exc:
            raise NoflukeError(
                f"strategy '{name}' needs the optional packages of nofluke[{_OPTUNA_EXTRAS[name]}]: "
                f"pip install 'nofluke[{_OPTUNA_EXTRAS[name]}]' ({exc})"
            ) from None
    if name not in _BUILTIN:
        known = ", ".join([*_BUILTIN, *_OPTUNA_EXTRAS])
        raise NoflukeError(f"strategy: unknown name {name!r} (known: {known})")
    if options:
        raise NoflukeError(f"strategy '{name}' takes no options, got {sorted(options)}")
    return _BUILTIN[name]()
