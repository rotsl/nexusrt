"""Config loader — reads the YAML files under config/."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def _config_root() -> Path:
    env = os.environ.get("NEXUSRT_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for cand in (here.parent.parent / "config",
                 Path("/etc/nexusrt"),
                 Path.cwd() / "config"):
        if cand.exists():
            return cand
    return Path("config")


def load(name: str) -> Dict[str, Any]:
    """Load a YAML config by name (without .yaml extension)."""
    p = _config_root() / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"NexusRT config not found: {p}")
    if yaml is None:
        raise RuntimeError("PyYAML not installed — `pip install pyyaml`")
    with p.open() as f:
        return yaml.safe_load(f)


def hardware() -> Dict[str, Any]:
    return load("hardware")


def runtime() -> Dict[str, Any]:
    return load("runtime")


def pipeline() -> Dict[str, Any]:
    return load("pipeline")
