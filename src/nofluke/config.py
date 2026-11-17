"""Study configuration: loading JSON/YAML, filling defaults, validating, and canonical hashes.

`load_config` reads a file or copies a dict; `normalise` returns the full config with every default filled in
(CONTRACTS section 1), rejecting unknown keys and out-of-range values. The normalised config is what the journal
stores, so later changes to the defaults never alter an existing study. `config_hash` and `space_hash` are sha256
digests of canonical JSON; the order of the space is part of the hash because it fixes the unit-cube coordinates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from nofluke.errors import NoflukeError
from nofluke.space import Space

_TOP_KEYS = ("name", "direction", "metric", "budget", "seed", "max_failures", "retries", "space", "strategy", "noise")
_STRATEGY_KEYS = ("name", "n_init", "options")
_NOISE_DEFAULTS: dict[str, Any] = {
    "reeval_share": 0.333,
    "min_candidates": 8,
    "max_replicates": 6,
    "finalist_share": 0.08,
    "z": 1.5,
    "nu0": 4.0,
    "prior_sd": None,
    "seed_mode": "shared",
}


def load_config(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Read a config file (JSON, or YAML for .yaml/.yml) or copy a dict; the result is not yet normalised."""
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NoflukeError(f"config: cannot read '{path}': {exc.strerror or exc}") from None
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise NoflukeError(f"config: '{path}' is YAML; install PyYAML with `pip install nofluke[yaml]`") from None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise NoflukeError(f"config: '{path}' is not valid YAML: {exc}") from None
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise NoflukeError(f"config: '{path}' is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise NoflukeError(f"config: '{path}' must contain a mapping at the top level")
    return data


def _int(section: str, key: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NoflukeError(f"config: '{section}{key}' must be an integer >= {minimum}, got {value!r}")
    return value


def _float(section: str, key: str, value: Any, low: float, high: float = math.inf, open_low: bool = False) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not ok or (value <= low if open_low else value < low) or value > high:
        bounds = f"{'>' if open_low else '>='} {low}" + (f" and <= {high}" if high != math.inf else "")
        raise NoflukeError(f"config: '{section}{key}' must be a number {bounds}, got {value!r}")
    return float(value)


def _unknown(section: str, given: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    for key in given:
        if key not in allowed:
            where = f" in '{section.rstrip('.')}'" if section else ""
            raise NoflukeError(f"config: unknown key '{key}'{where} (allowed: {', '.join(allowed)})")


def _section(name: str, value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise NoflukeError(f"config: '{name}' must be a mapping, got {value!r}")
    return value


def _text_or_none(key: str, value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise NoflukeError(f"config: '{key}' must be a string, got {value!r}")
    return value


def normalise(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the full normalised config (defaults filled in, values validated); raise NoflukeError otherwise."""
    _unknown("", config, _TOP_KEYS)
    for required in ("direction", "budget", "seed", "space"):
        if required not in config:
            raise NoflukeError(f"config: missing required key '{required}'")
    direction = config["direction"]
    if direction not in ("maximize", "minimize"):
        raise NoflukeError(f"config: 'direction' must be 'maximize' or 'minimize', got {direction!r}")
    budget = _int("", "budget", config["budget"], 1)
    out: dict[str, Any] = {
        "name": _text_or_none("name", config.get("name")),
        "direction": direction,
        "metric": _text_or_none("metric", config.get("metric")),
        "budget": budget,
        "seed": _int("", "seed", config["seed"], 0),
        "max_failures": _int("", "max_failures", config.get("max_failures", budget // 2), 1),
        "retries": _int("", "retries", config.get("retries", 1), 0),
        "space": Space.from_dict(config["space"]).to_dict(),
    }

    strategy = _section("strategy", config.get("strategy"))
    _unknown("strategy.", strategy, _STRATEGY_KEYS)
    name = strategy.get("name", "random")
    if not isinstance(name, str):
        raise NoflukeError(f"config: 'strategy.name' must be a string, got {name!r}")
    options = strategy.get("options", {})
    if not isinstance(options, Mapping):
        raise NoflukeError(f"config: 'strategy.options' must be a mapping, got {options!r}")
    try:
        json.dumps(options, allow_nan=False)
    except (TypeError, ValueError):
        raise NoflukeError("config: 'strategy.options' must be JSON-serialisable") from None
    out["strategy"] = {
        "name": name,
        "n_init": _int("strategy.", "n_init", strategy.get("n_init", min(max(budget // 8, 8), 20)), 0),
        "options": copy.deepcopy(dict(options)),
    }

    noise = _section("noise", config.get("noise"))
    _unknown("noise.", noise, tuple(_NOISE_DEFAULTS))
    given = {**_NOISE_DEFAULTS, **noise}
    prior_sd = given["prior_sd"]
    seed_mode = given["seed_mode"]
    if seed_mode not in ("shared", "per_candidate"):
        raise NoflukeError(f"config: 'noise.seed_mode' must be 'shared' or 'per_candidate', got {seed_mode!r}")
    out["noise"] = {
        "reeval_share": _float("noise.", "reeval_share", given["reeval_share"], 0.0, 1.0),
        "min_candidates": _int("noise.", "min_candidates", given["min_candidates"], 1),
        "max_replicates": _int("noise.", "max_replicates", given["max_replicates"], 1),
        "finalist_share": _float("noise.", "finalist_share", given["finalist_share"], 0.0, 1.0),
        "z": _float("noise.", "z", given["z"], 0.0),
        "nu0": _float("noise.", "nu0", given["nu0"], 0.0, open_low=True),
        "prior_sd": None if prior_sd is None else _float("noise.", "prior_sd", prior_sd, 0.0),
        "seed_mode": seed_mode,
    }
    return out


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def space_hash(space: Mapping[str, Any]) -> str:
    """sha256 hex of the canonical JSON of a normalised `space` dict (declared order is part of the hash)."""
    return hashlib.sha256(_canonical_json([[name, spec] for name, spec in space.items()]).encode("utf-8")).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """sha256 hex of the canonical JSON of a normalised config (the space is hashed in declared order)."""
    ordered = {**config, "space": [[name, spec] for name, spec in config["space"].items()]}
    return hashlib.sha256(_canonical_json(ordered).encode("utf-8")).hexdigest()
