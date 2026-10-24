"""Selection consumes chronological observed statistics, never evaluation truth."""

from dataclasses import replace

import numpy as np
import pytest

from spde_covariance.coherent_benchmark import fit_coherent_candidates
from spde_covariance.coherent_config import CoherentBenchmarkConfig
from spde_covariance.observation_statistics import partial_kernel_from_arrays


def test_config_is_validated_without_changing_historical_profile():
    config = CoherentBenchmarkConfig()
    assert config.study == "coherent"
    assert config.replications == 20
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(config, partial_regularizations=(0.01, 0.001))
    with pytest.raises(ValueError, match="positive"):
        replace(config, eigenvalue_floor=0)
    with pytest.raises(ValueError, match="SPDE fragments"):
        replace(config, population_checks=True)


def test_validation_changes_selection_only_not_training_candidates():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    rows = tuple(cliques[i % 2] for i in range(40))
    rng = np.random.default_rng(17)
    values = tuple(rng.standard_normal(2) for _ in rows)
    train = partial_kernel_from_arrays(values, rows, size=3)
    validation = partial_kernel_from_arrays(values[::-1], rows, size=3)
    settings = {
        "relative_regularizations": (0.001, 0.1),
        "eigenvalue_floor": 1e-4,
        "numerical_error_budget": 1e-4,
        "rho": 0.2,
        "max_iterations": 20_000,
        "support_rtol": 1e-12,
    }
    fits, audit, selected = fit_coherent_candidates(
        train, validation, cliques=cliques, weights=np.ones(3), settings=settings
    )
    altered = partial_kernel_from_arrays(tuple(v * 100 for v in values), rows, size=3)
    again, other, _ = fit_coherent_candidates(
        train, altered, cliques=cliques, weights=np.ones(3), settings=settings
    )
    assert selected is not None
    for (kernel, fit), (new_kernel, _), first, second in zip(
        fits, again, audit, other, strict=True
    ):
        np.testing.assert_array_equal(kernel, new_kernel)
        assert fit.converged
        assert first["validation_score"] != second["validation_score"]
    expected = min(
        range(len(audit)),
        key=lambda i: (audit[i]["validation_score"], -audit[i]["relative_regularization"]),
    )
    assert selected == expected
