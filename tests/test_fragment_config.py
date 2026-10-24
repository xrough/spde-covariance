"""Declared fragment-study scope rejects silently skipped comparisons."""

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spde_covariance.fragment_config import FragmentBenchmarkConfig


def test_defaults_lock_fixed_step_sources_and_disjoint_data_blocks():
    config = FragmentBenchmarkConfig()
    assert config.profile == "spde"
    assert config.delta == 1 / 52
    assert config.training_counts == (128, 512, 2048)
    assert config.total_count == 2048 + 128 + 512
    assert config.models == ("powerlaw", "rotated")
    assert config.reference_dimension == 512
    assert config.covers == (3, 9)
    assert config.ranks == (1, 2, 4, 8)
    assert config.ridge_comparison


@pytest.mark.parametrize(
    "changes",
    [
        {"study": "other"},
        {"profile": "other"},
        {"name": " "},
        {"seed": True},
        {"replications": 0},
        {"training_counts": ()},
        {"training_counts": (512, 128)},
        {"training_counts": (128, 128)},
        {"training_counts": (4, 512)},
        {"covers": (4,)},
        {"grids": (10, 81)},
        {"ranks": (0, 2)},
        {"candidate_dimensions": (4, 32)},
        {"models": ("analytic",)},
        {"models": ("powerlaw", "powerlaw")},
        {"reference_dimension": 511},
        {"reference_dimension": 510},
        {"source_exponent": 2},
        {"source_trace": 0},
        {"delta": 0},
        {"noise_variances": (-0.1,)},
        {"noise_variances": (True,)},
        {"noise_variances": (np.nan,)},
        {"relative_regularizations": (0,)},
        {"relative_regularizations": (True,)},
        {"support_rtol": 1},
        {"numerical_error_budget": np.inf},
        {"rotation_angle": np.nan},
        {"include_fve": 1},
        {"ridge_comparison": 1},
        {"population_checks": 1},
        {"grids": (41, 161)},
        {"training_counts": (128, 2048)},
    ],
)
def test_invalid_or_vacuous_declared_scope_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(FragmentBenchmarkConfig(), **changes)


def test_pair_complete_prefix_can_skip_half_forcing_sensitivity_explicitly():
    config = replace(FragmentBenchmarkConfig(), reference_dimension=510, population_checks=False)
    assert config.reference_dimension == 510
    assert not config.population_checks


def test_reference_profile_never_silently_fits_an_unrelated_h_basis():
    with pytest.raises(ValueError, match="H-coordinate"):
        replace(FragmentBenchmarkConfig(), profile="reference", models=("finite_rank",))
    config = replace(
        FragmentBenchmarkConfig(),
        profile="reference",
        models=("finite_rank", "brownian", "analytic"),
        ridge_comparison=False,
    )
    assert config.models == ("finite_rank", "brownian", "analytic")


def test_committed_reference_full_and_smoke_configs_are_parseable():
    root = Path(__file__).resolve().parents[1]
    configs = {}
    for name in ("canonical-smoke", "canonical-reference", "canonical-spde"):
        with (root / "experiments" / "configs" / f"{name}.toml").open("rb") as stream:
            configs[name] = FragmentBenchmarkConfig(**tomllib.load(stream))
    assert configs["canonical-spde"] == FragmentBenchmarkConfig()
    reference = configs["canonical-reference"]
    assert reference.profile == "reference"
    assert reference.replications == 100
    assert reference.training_counts == (100, 500)
    assert reference.grids == (100,)
    assert not reference.ridge_comparison
    assert configs["canonical-smoke"].replications == 2
    assert not configs["canonical-smoke"].ridge_comparison
