"""Search space: typed parameters, conditional activation and the unit-cube codec.

A `Space` is an ordered list of parameters (`FloatParam`, `IntParam`, `ChoiceParam`, `BoolParam`). `Space.from_dict`
parses the `space:` section of a study config (CONTRACTS section 1) and `to_dict` writes the canonical form back.
`decode` maps a point of the unit cube (one coordinate per declared parameter) to a parameter dict, dropping the
parameters whose `when` condition fails; `encode` is its inverse. All values handed out are plain Python types.
To add a parameter type, subclass `Param` and register it in `_PARAM_TYPES`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np

from nofluke.errors import NoflukeError

Scalar = str | int | float | bool | None


def _is_number(value: Any) -> bool:
    """True for a finite int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _key(value: Any) -> tuple[str, Any]:
    """Identity of a scalar that keeps 1, 1.0 and True apart."""
    return (type(value).__name__, value)


def _check_keys(name: str, spec: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise NoflukeError(f"parameter '{name}': unknown key '{unknown[0]}' (allowed: {', '.join(sorted(allowed))})")


class Param(ABC):
    """One searchable parameter; subclasses are frozen dataclasses with `name` and `when` fields."""

    type: ClassVar[str]
    name: str
    when: dict[str, tuple[Scalar, ...]] | None

    def active(self, params_so_far: Mapping[str, Any]) -> bool:
        """True if every `when` condition holds for the (already decided) earlier parameters."""
        if not self.when:
            return True
        for parent, allowed in self.when.items():
            if parent not in params_so_far:
                return False
            if _key(params_so_far[parent]) not in {_key(v) for v in allowed}:
                return False
        return True

    @abstractmethod
    def decode(self, u: float) -> Any:
        """Map a coordinate in [0, 1] to a value."""

    @abstractmethod
    def encode(self, value: Any) -> float:
        """Map a value to a coordinate in [0, 1] (cell centre for discrete parameters)."""

    @abstractmethod
    def coerce(self, value: Any) -> Any:
        """Return `value` as a plain Python value of this parameter, or raise ValueError if it is not one."""

    @abstractmethod
    def _spec(self) -> dict[str, Any]:
        """The type-specific part of the canonical dict form."""

    def to_dict(self) -> dict[str, Any]:
        """Canonical dict form (what `Space.from_dict` accepts)."""
        spec: dict[str, Any] = {"type": self.type, **self._spec()}
        if self.when:
            spec["when"] = {p: (vs[0] if len(vs) == 1 else list(vs)) for p, vs in self.when.items()}
        return spec

    def accepts_condition(self, value: Any) -> bool:
        """True if `value` is a legal right-hand side of a `when` condition on this parameter."""
        try:
            self.coerce(value)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class FloatParam(Param):
    """Real parameter in [low, high], linear or logarithmic."""

    type: ClassVar[str] = "float"
    name: str
    low: float
    high: float
    log: bool = False
    when: dict[str, tuple[Scalar, ...]] | None = None

    def decode(self, u: float) -> float:
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            value = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            value = self.low + u * (self.high - self.low)
        return min(max(value, self.low), self.high)

    def encode(self, value: Any) -> float:
        if self.log:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def coerce(self, value: Any) -> float:
        if not _is_number(value) or not self.low <= value <= self.high:
            raise ValueError(f"expected a number in [{self.low}, {self.high}], got {value!r}")
        return float(value)

    def _spec(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high, "log": self.log}


@dataclass(frozen=True)
class IntParam(Param):
    """Integer parameter in [low, high] (both inclusive), linear or logarithmic."""

    type: ClassVar[str] = "int"
    name: str
    low: int
    high: int
    log: bool = False
    when: dict[str, tuple[Scalar, ...]] | None = None

    def decode(self, u: float) -> int:
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            span = math.log(self.high + 1) - math.log(self.low)
            value = math.floor(math.exp(math.log(self.low) + u * span))
        else:
            value = self.low + math.floor(u * (self.high - self.low + 1))
        return min(max(int(value), self.low), self.high)

    def encode(self, value: Any) -> float:
        if self.log:
            centre = (math.log(value) + math.log(value + 1)) / 2
            return (centre - math.log(self.low)) / (math.log(self.high + 1) - math.log(self.low))
        return (value - self.low + 0.5) / (self.high - self.low + 1)

    def coerce(self, value: Any) -> int:
        if not _is_int(value) or not self.low <= value <= self.high:
            raise ValueError(f"expected an integer in [{self.low}, {self.high}], got {value!r}")
        return int(value)

    def _spec(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high, "log": self.log}


@dataclass(frozen=True)
class ChoiceParam(Param):
    """One of a fixed list of JSON scalars."""

    type: ClassVar[str] = "choice"
    name: str
    values: tuple[Scalar, ...]
    when: dict[str, tuple[Scalar, ...]] | None = None

    def decode(self, u: float) -> Scalar:
        k = len(self.values)
        return self.values[min(math.floor(min(max(float(u), 0.0), 1.0) * k), k - 1)]

    def encode(self, value: Any) -> float:
        return (self.index(value) + 0.5) / len(self.values)

    def index(self, value: Any) -> int:
        """Position of `value` in `values`, matching type as well as value (1, 1.0 and True are distinct)."""
        for i, v in enumerate(self.values):
            if _key(v) == _key(value):
                return i
        raise ValueError(f"expected one of {list(self.values)}, got {value!r}")

    def coerce(self, value: Any) -> Scalar:
        return self.values[self.index(value)]

    def _spec(self) -> dict[str, Any]:
        return {"values": list(self.values)}


@dataclass(frozen=True)
class BoolParam(Param):
    """Boolean parameter."""

    type: ClassVar[str] = "bool"
    name: str
    when: dict[str, tuple[Scalar, ...]] | None = None

    def decode(self, u: float) -> bool:
        return bool(float(u) >= 0.5)

    def encode(self, value: Any) -> float:
        return 0.75 if value else 0.25

    def coerce(self, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"expected true or false, got {value!r}")
        return value

    def _spec(self) -> dict[str, Any]:
        return {}


def _parse_float_like(name: str, spec: Mapping[str, Any], integer: bool) -> dict[str, Any]:
    kind = "int" if integer else "float"
    _check_keys(name, spec, {"type", "low", "high", "log", "when"})
    for bound in ("low", "high"):
        if bound not in spec:
            raise NoflukeError(f"parameter '{name}': {kind} needs '{bound}'")
        value = spec[bound]
        if not (_is_int(value) if integer else _is_number(value)):
            expected = "an integer" if integer else "a finite number"
            raise NoflukeError(f"parameter '{name}': '{bound}' must be {expected}, got {value!r}")
    low, high = spec["low"], spec["high"]
    log = spec.get("log", False)
    if not isinstance(log, bool):
        raise NoflukeError(f"parameter '{name}': 'log' must be true or false, got {log!r}")
    if (low > high) if integer else (low >= high):
        relation = "<=" if integer else "<"
        raise NoflukeError(f"parameter '{name}': needs low {relation} high, got low={low}, high={high}")
    if log and ((low < 1) if integer else (low <= 0)):
        need = "low >= 1" if integer else "low > 0"
        raise NoflukeError(f"parameter '{name}': a log scale needs {need}, got low={low}")
    return {"low": low if integer else float(low), "high": high if integer else float(high), "log": log}


def _parse_choice(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    _check_keys(name, spec, {"type", "values", "when"})
    values = spec.get("values")
    if not isinstance(values, (list, tuple)) or not values:
        raise NoflukeError(f"parameter '{name}': choice needs a non-empty list 'values'")
    for v in values:
        if not (v is None or isinstance(v, (str, bool)) or _is_number(v)):
            raise NoflukeError(f"parameter '{name}': choice values must be JSON scalars, got {v!r}")
    if len({_key(v) for v in values}) != len(values):
        raise NoflukeError(f"parameter '{name}': choice values must be distinct, got {list(values)}")
    return {"values": tuple(values)}


def _parse_bool(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    _check_keys(name, spec, {"type", "when"})
    return {}


_PARAM_TYPES: dict[str, tuple[type[Param], Any]] = {
    "float": (FloatParam, lambda n, s: _parse_float_like(n, s, integer=False)),
    "int": (IntParam, lambda n, s: _parse_float_like(n, s, integer=True)),
    "choice": (ChoiceParam, _parse_choice),
    "bool": (BoolParam, _parse_bool),
}


def _parse_when(name: str, when: Any, earlier: dict[str, Param]) -> dict[str, tuple[Scalar, ...]]:
    if not isinstance(when, Mapping) or not when:
        raise NoflukeError(f"parameter '{name}': 'when' must be a non-empty mapping {{parent: value or [values]}}")
    parsed: dict[str, tuple[Scalar, ...]] = {}
    for parent, wanted in when.items():
        if parent not in earlier:
            raise NoflukeError(f"parameter '{name}': 'when' names '{parent}', which is not an earlier parameter")
        if isinstance(earlier[parent], FloatParam):
            raise NoflukeError(f"parameter '{name}': 'when' cannot test the float parameter '{parent}'")
        values = tuple(wanted) if isinstance(wanted, (list, tuple)) else (wanted,)
        if not values:
            raise NoflukeError(f"parameter '{name}': 'when' lists no values for '{parent}'")
        for v in values:
            if not earlier[parent].accepts_condition(v):
                raise NoflukeError(f"parameter '{name}': 'when' value {v!r} is not a possible value of '{parent}'")
        parsed[str(parent)] = values
    return parsed


class Space:
    """An ordered collection of parameters; the order fixes the unit-cube coordinates."""

    def __init__(self, params: list[Param]) -> None:
        if not params:
            raise NoflukeError("space: needs at least one parameter")
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise NoflukeError("space: parameter names must be unique")
        self.params: tuple[Param, ...] = tuple(params)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def __len__(self) -> int:
        return len(self.params)

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> Space:
        """Parse `{name: {type: ..., ...}}` (declared order is kept); raise NoflukeError naming the parameter."""
        if not isinstance(spec, Mapping) or not spec:
            raise NoflukeError("space: must be a non-empty mapping of parameter name to its definition")
        params: dict[str, Param] = {}
        for name, definition in spec.items():
            if not isinstance(name, str) or not name:
                raise NoflukeError(f"space: parameter names must be non-empty strings, got {name!r}")
            if not isinstance(definition, Mapping) or "type" not in definition:
                raise NoflukeError(f"parameter '{name}': needs a mapping with a 'type'")
            kind = definition["type"]
            if kind not in _PARAM_TYPES:
                raise NoflukeError(f"parameter '{name}': unknown type {kind!r} (known: {', '.join(_PARAM_TYPES)})")
            cls_, parse = _PARAM_TYPES[kind]
            fields = parse(name, definition)
            when = _parse_when(name, definition["when"], params) if "when" in definition else None
            params[name] = cls_(name=name, when=when, **fields)
        return cls(list(params.values()))

    def to_dict(self) -> dict[str, Any]:
        """Canonical dict form in declared order."""
        return {p.name: p.to_dict() for p in self.params}

    def decode(self, u: np.ndarray) -> dict[str, Any]:
        """Map a unit-cube point (one coordinate per parameter, declared order) to parameters; inactive ones are dropped."""
        if len(u) != len(self.params):
            raise NoflukeError(f"space: expected {len(self.params)} coordinates, got {len(u)}")
        params: dict[str, Any] = {}
        for coordinate, param in zip(u, self.params):
            if param.active(params):
                params[param.name] = param.decode(float(coordinate))
        return params

    def encode(self, params: Mapping[str, Any]) -> np.ndarray:
        """Inverse of `decode`: coordinates in [0, 1], NaN for parameters absent from `params`."""
        return np.array(
            [param.encode(params[param.name]) if param.name in params else math.nan for param in self.params],
            dtype=float,
        )

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Check a parameter dict against the space and return it with plain Python values in declared order."""
        unknown = [k for k in params if k not in self.names]
        if unknown:
            raise NoflukeError(f"params: unknown parameter '{unknown[0]}'")
        clean: dict[str, Any] = {}
        for param in self.params:
            if not param.active(clean):
                if param.name in params:
                    raise NoflukeError(f"params: '{param.name}' is given but its 'when' condition does not hold")
                continue
            if param.name not in params:
                raise NoflukeError(f"params: missing active parameter '{param.name}'")
            try:
                clean[param.name] = param.coerce(params[param.name])
            except ValueError as exc:
                raise NoflukeError(f"params: '{param.name}': {exc}") from None
        return clean
