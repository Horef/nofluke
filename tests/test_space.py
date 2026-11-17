"""Tests for nofluke.space: validation, conditionals, the unit-cube codec, plain output types."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nofluke.errors import NoflukeError
from nofluke.rng import derive
from nofluke.space import BoolParam, ChoiceParam, FloatParam, IntParam, Space

SPEC = {
    "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "epochs": {"type": "int", "low": 30, "high": 200},
    "width": {"type": "int", "low": 1, "high": 1000, "log": True},
    "batch": {"type": "choice", "values": [16, 32, "big", None, 1.5, True]},
    "aux": {"type": "bool"},
    "aux_w": {"type": "float", "low": 0.1, "high": 100, "log": True, "when": {"aux": True}},
    "mode": {"type": "choice", "values": ["a", "b", "c"], "when": {"batch": [16, 32]}},
}


@pytest.mark.parametrize(
    "spec, fragment",
    [
        ({"p": {"type": "float", "low": 1, "high": 1}}, "parameter 'p'"),
        ({"p": {"type": "float", "low": 0, "high": 1, "log": True}}, "log scale needs low > 0"),
        ({"p": {"type": "float", "low": 0, "high": 1, "bogus": 1}}, "unknown key 'bogus'"),
        ({"p": {"type": "float", "low": "a", "high": 1}}, "'low' must be a finite number"),
        ({"p": {"type": "float", "high": 1}}, "needs 'low'"),
        ({"p": {"type": "int", "low": 5, "high": 4}}, "needs low <= high"),
        ({"p": {"type": "int", "low": 0, "high": 4, "log": True}}, "low >= 1"),
        ({"p": {"type": "int", "low": 1.5, "high": 4}}, "must be an integer"),
        ({"p": {"type": "choice", "values": []}}, "non-empty list"),
        ({"p": {"type": "choice", "values": [1, 1]}}, "distinct"),
        ({"p": {"type": "choice", "values": [[1]]}}, "JSON scalars"),
        ({"p": {"type": "tensor"}}, "unknown type 'tensor'"),
        ({"p": {"low": 0}}, "needs a mapping with a 'type'"),
        ({"p": {"type": "bool", "when": {"q": True}}}, "not an earlier parameter"),
        ({"q": {"type": "bool"}, "p": {"type": "bool", "when": {"q": 3}}}, "not a possible value of 'q'"),
        ({"q": {"type": "float", "low": 0, "high": 1}, "p": {"type": "bool", "when": {"q": 0.5}}}, "float parameter 'q'"),
        ({"p": {"type": "bool", "when": {}}}, "non-empty mapping"),
        ({}, "non-empty mapping"),
    ],
)
def test_validation_errors_name_the_parameter(spec, fragment):
    with pytest.raises(NoflukeError, match=fragment):
        Space.from_dict(spec)


def test_when_must_name_an_earlier_parameter():
    with pytest.raises(NoflukeError, match="'p'.*'q'"):
        Space.from_dict({"p": {"type": "bool", "when": {"q": True}}, "q": {"type": "bool"}})


def test_round_trip_to_dict_is_canonical_and_ordered():
    space = Space.from_dict(SPEC)
    again = Space.from_dict(space.to_dict())
    assert again.to_dict() == space.to_dict()
    assert space.names == list(SPEC)
    assert space.to_dict()["mode"]["when"] == {"batch": [16, 32]}
    assert space.to_dict()["aux_w"]["when"] == {"aux": True}
    assert isinstance(space.params[0], FloatParam) and isinstance(space.params[1], IntParam)
    assert isinstance(space.params[3], ChoiceParam) and isinstance(space.params[4], BoolParam)


def test_conditionals_drop_inactive_parameters():
    space = Space.from_dict(SPEC)
    u = np.full(len(space), 0.1)
    u[space.names.index("aux")] = 0.2  # False -> aux_w inactive
    u[space.names.index("batch")] = 0.0  # 16 -> mode active
    params = space.decode(u)
    assert "aux_w" not in params and "mode" in params
    u[space.names.index("aux")] = 0.9
    u[space.names.index("batch")] = 0.5  # index 3 -> None -> mode inactive
    params = space.decode(u)
    assert "aux_w" in params and params["aux"] is True
    assert params["batch"] is None and "mode" not in params


def test_decode_endpoints_and_log_scale():
    space = Space.from_dict(SPEC)
    low = space.decode(np.zeros(len(space)))
    high = space.decode(np.ones(len(space)))
    assert low["lr"] == pytest.approx(1e-4) and high["lr"] == pytest.approx(1e-1)
    assert (low["epochs"], high["epochs"]) == (30, 200)
    assert (low["width"], high["width"]) == (1, 1000)
    assert low["batch"] == 16 and high["batch"] is True
    mid = space.decode(np.full(len(space), 0.5))
    assert mid["lr"] == pytest.approx(math.sqrt(1e-4 * 1e-1))


def test_log_int_follows_the_spec_formula():
    p = IntParam("n", 1, 100, log=True)
    for u in (0.0, 0.1, 0.37, 0.8, 0.999999, 1.0):
        expected = min(max(math.floor(math.exp(math.log(1) + u * (math.log(101) - math.log(1)))), 1), 100)
        assert p.decode(u) == expected


def test_codec_round_trip_for_random_points():
    space = Space.from_dict(SPEC)
    rng = derive(0, "test-codec")
    for _ in range(300):
        params = space.decode(rng.random(len(space)))
        again = space.decode(space.encode(params))
        assert again == params
        assert [type(v) for v in again.values()] == [type(v) for v in params.values()]


def test_encode_marks_inactive_with_nan_and_floats_continuously():
    space = Space.from_dict(SPEC)
    params = {"lr": 1e-2, "epochs": 30, "width": 10, "batch": 16, "aux": False, "mode": "b"}
    u = space.encode(params)
    assert np.isnan(u[space.names.index("aux_w")])
    assert u[0] == pytest.approx(2 / 3)
    assert all(0 <= x <= 1 for x in u if not np.isnan(x))


def test_decoded_values_are_plain_python_types():
    space = Space.from_dict(SPEC)
    rng = derive(1, "types")
    for _ in range(50):
        for value in space.decode(rng.random(len(space))).values():
            assert type(value) in (float, int, str, bool, type(None))


def test_decode_wrong_length_is_an_error():
    with pytest.raises(NoflukeError, match="coordinates"):
        Space.from_dict(SPEC).decode(np.zeros(2))


def test_validate_params():
    space = Space.from_dict(SPEC)
    good = {"lr": 0.01, "epochs": 50, "width": 3, "batch": 16, "aux": True, "aux_w": 5, "mode": "a"}
    clean = space.validate_params(good)
    assert list(clean) == space.names and type(clean["aux_w"]) is float
    for bad, fragment in [
        ({**good, "lr": 1.0}, "'lr'"),
        ({**good, "epochs": 50.0}, "'epochs'"),
        ({**good, "batch": 17}, "'batch'"),
        ({**good, "aux": 1}, "'aux'"),
        ({**good, "zzz": 1}, "unknown parameter 'zzz'"),
        ({k: v for k, v in good.items() if k != "aux_w"}, "missing active parameter 'aux_w'"),
        ({**good, "aux": False}, "'aux_w' is given but its 'when'"),
        ({**good, "lr": float("nan")}, "'lr'"),
    ]:
        with pytest.raises(NoflukeError, match=fragment):
            space.validate_params(bad)


def test_bool_and_int_are_not_interchangeable_in_choices():
    p = ChoiceParam("c", (1, True, 1.0), None)
    assert p.coerce(True) is True and p.coerce(1) == 1 and type(p.coerce(1)) is int
    assert type(p.coerce(1.0)) is float
