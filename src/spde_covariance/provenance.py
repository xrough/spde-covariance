"""Source and environment records for configured numerical benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git(root, *arguments):
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _source_root():
    package = Path(__file__).resolve().parent
    candidate = _git(package, "rev-parse", "--show-toplevel")
    if candidate is None:
        return None
    root = Path(candidate)
    return root if (root / "src" / "spde_covariance").resolve() == package else None


def _manifest(config, root):
    package = Path(__file__).resolve().parent
    hashes = {
        f"spde_covariance/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.glob("*.py"))
    }
    project_hashes = (
        {}
        if root is None
        else {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("uv.lock", "pyproject.toml")
            if (root / name).is_file()
        }
    )
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "cvxpy", "scs", "spde-covariance"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git_status = _git(root, "status", "--porcelain")
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "git_revision": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_status) if git_status is not None else None,
        "source_sha256": hashes,
        "project_file_sha256": project_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "solver": {
            "name": "accelerated_projected_gradient",
            "relative_tolerance": 1e-8,
            "absolute_tolerance": 1e-9,
            "max_iterations": 20000,
            "numerical_error_budget": config.numerical_error_budget,
        },
        "experiment": "independent centered Gaussian innovations; deterministic designs",
        "space": {"metric": "f(0)g(0)+integral exp(x) f' g'", "maturity_unit": "year"},
    }
