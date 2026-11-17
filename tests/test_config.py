"""Tests for nofluke.config: defaults, unknown keys, validation, hashes, YAML loading."""

from __future__ import annotations

import json
import sys

import pytest

from nofluke.config import config_hash, load_config, normalise, space_hash
from nofluke.errors import NoflukeError

RAW = {
    "direction": "maximize",
    "budget": 128,
    "seed": 0,
    "space": {"a": {"type": "float", "low": 0, "high": 1}, "b": {"type": "bool"}},
}


def test_defaults_are_filled_in():
    cfg = normalise(RAW)
    assert cfg["max_failures"] == 64 and cfg["retries"] == 1
    assert cfg["name"] is None and cfg["metric"] is None
    assert cfg["strategy"] == {"name": "random", "n_init": 16, "options": {}}
    assert cfg["noise"] == {
        "reeval_share": 0.333, "min_candidates": 8, "max_replicates": 6, "finalist_share": 0.08,
        "z": 1.5, "nu0": 4.0, "prior_sd": None, "seed_mode": "shared",
    }  # fmt: skip
    assert normalise(cfg) == cfg  # idempotent


@pytest.mark.parametrize("budget, n_init", [(8, 8), (64, 8), (128, 16), (400, 20)])
def test_n_init_default_is_clipped(budget, n_init):
    assert normalise({**RAW, "budget": budget})["strategy"]["n_init"] == n_init


def test_explicit_values_override_defaults():
    cfg = normalise({**RAW, "noise": {"z": 2, "prior_sd": 0.5}, "strategy": {"n_init": 3}, "retries": 0, "max_failures": 5})
    assert cfg["noise"]["z"] == 2.0 and cfg["noise"]["prior_sd"] == 0.5
    assert (cfg["strategy"]["n_init"], cfg["retries"], cfg["max_failures"]) == (3, 0, 5)


@pytest.mark.parametrize(
    "patch, fragment",
    [
        ({"bogus": 1}, "unknown key 'bogus'"),
        ({"noise": {"zz": 1}}, "unknown key 'zz' in 'noise'"),
        ({"strategy": {"x": 1}}, "unknown key 'x' in 'strategy'"),
        ({"direction": "up"}, "'direction'"),
        ({"budget": 0}, "'budget'"),
        ({"budget": 1.5}, "'budget'"),
        ({"seed": -1}, "'seed'"),
        ({"retries": -1}, "'retries'"),
        ({"max_failures": 0}, "'max_failures'"),
        ({"noise": {"reeval_share": 1.5}}, "'noise.reeval_share'"),
        ({"noise": {"nu0": 0}}, "'noise.nu0'"),
        ({"noise": {"seed_mode": "x"}}, "'noise.seed_mode'"),
        ({"noise": {"prior_sd": -1}}, "'noise.prior_sd'"),
        ({"noise": {"min_candidates": 0}}, "'noise.min_candidates'"),
        ({"strategy": {"options": {"f": float("nan")}}}, "JSON-serialisable"),
        ({"space": {"a": {"type": "nope"}}}, "parameter 'a'"),
        ({"name": 3}, "'name'"),
    ],
)
def test_invalid_configs_are_rejected(patch, fragment):
    with pytest.raises(NoflukeError, match=fragment):
        normalise({**RAW, **patch})


@pytest.mark.parametrize("missing", ["direction", "budget", "seed", "space"])
def test_required_keys(missing):
    raw = {k: v for k, v in RAW.items() if k != missing}
    with pytest.raises(NoflukeError, match=f"missing required key '{missing}'"):
        normalise(raw)


def test_hashes_are_stable_and_sensitive():
    cfg = normalise(RAW)
    assert config_hash(cfg) == config_hash(normalise(json.loads(json.dumps(RAW))))
    assert len(config_hash(cfg)) == 64 and len(space_hash(cfg["space"])) == 64
    assert config_hash(normalise({**RAW, "seed": 1})) != config_hash(cfg)
    # keys inside a definition may be written in any order ...
    reordered = {"a": {"high": 1, "low": 0, "type": "float"}, "b": {"type": "bool"}}
    assert space_hash(normalise({**RAW, "space": reordered})["space"]) == space_hash(cfg["space"])
    # ... but the order of the parameters fixes the unit-cube coordinates, so it is part of the hash
    swapped = {"b": RAW["space"]["b"], "a": RAW["space"]["a"]}
    other = normalise({**RAW, "space": swapped})
    assert space_hash(other["space"]) != space_hash(cfg["space"])
    assert config_hash(other) != config_hash(cfg)


def test_load_config_json_dict_and_errors(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    assert load_config(path) == RAW and load_config(str(path)) == RAW
    copy = load_config(RAW)
    copy["budget"] = 1
    assert RAW["budget"] == 128
    (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(NoflukeError, match="not valid JSON"):
        load_config(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(NoflukeError, match="mapping"):
        load_config(tmp_path / "list.json")
    with pytest.raises(NoflukeError, match="cannot read"):
        load_config(tmp_path / "missing.json")


def test_load_config_yaml(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(
        "direction: minimize\nbudget: 10\nseed: 2\nspace:\n  lr: {type: float, low: 1.0e-4, high: 1.0e-1, log: true}\n"
        "  k: {type: choice, values: [1, 2]}\n",
        encoding="utf-8",
    )
    cfg = normalise(load_config(path))
    assert cfg["direction"] == "minimize" and list(cfg["space"]) == ["lr", "k"]
    assert cfg["space"]["lr"]["low"] == 1e-4


def test_yaml_without_pyyaml_names_the_extra(tmp_path, monkeypatch):
    path = tmp_path / "study.yml"
    path.write_text("a: 1\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(NoflukeError, match=r"nofluke\[yaml\]"):
        load_config(path)
