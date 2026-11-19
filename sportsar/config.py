"""YAML config loading with dotted-key CLI overrides (``--set train.epochs=10``)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # PyYAML reads "3e-4" as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(raw.strip())
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def add_config_args(parser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config value (repeatable)")
