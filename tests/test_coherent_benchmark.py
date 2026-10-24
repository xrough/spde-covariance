"""Selection consumes chronological observed statistics, never evaluation truth."""

from dataclasses import replace

import numpy as np
import pytest

from spde_covariance.coherent_benchmark import _run_replications, fit_coherent_candidates
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


def test_parallel_replicates_preserve_seed_order_and_measurements(tmp_path):
    from spde_covariance.fragment_benchmark import _prepare
    from spde_covariance.provenance import _jsonable

    config = CoherentBenchmarkConfig(
        replications=2,
        workers=1,
        training_counts=(16,),
        covers=(3,),
        grids=(11,),
        models=("powerlaw",),
        noise_variances=(0.001,),
        reference_dimension=4,
        candidate_dimensions=(4,),
        validation_count=6,
        test_count=6,
        partial_regularizations=(0.001,),
        numerical_error_budget=0.01,
        admissibility_dimensions=(2,),
        admissibility_max_iterations=50,
    )
    settings = {
        "relative_regularizations": config.partial_regularizations,
        "eigenvalue_floor": config.eigenvalue_floor,
        "numerical_error_budget": config.numerical_error_budget,
        "rho": config.admm_rho,
        "max_iterations": config.admm_max_iterations,
        "support_rtol": config.support_rtol,
    }
    prepared = _prepare(config)
    left, right = tmp_path / "serial", tmp_path / "parallel"
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        (directory / "manifest.json").write_text('{"test_source": "fixed"}')
    serial = _run_replications(config, left, prepared, settings)
    parallel = _run_replications(replace(config, workers=2), right, prepared, settings)

    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k != "runtime_seconds"}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value

    assert [r["replication"] for r in parallel] == [0, 1]
    assert stable(_jsonable(serial)) == stable(_jsonable(parallel))
    assert len(list(right.glob("replication-*.json"))) == 2


def test_failed_replication_retains_manifest_bound_checkpoint(tmp_path):
    import hashlib
    import json

    from spde_covariance.fragment_benchmark import _prepare

    config = CoherentBenchmarkConfig(
        replications=1,
        training_counts=(16,),
        covers=(3,),
        grids=(11,),
        models=("powerlaw",),
        noise_variances=(0.001,),
        reference_dimension=4,
        candidate_dimensions=(4,),
        validation_count=6,
        test_count=6,
        partial_regularizations=(0.001,),
        admissibility_dimensions=(2,),
    )
    manifest = '{"source": "fixed-failure-control"}'
    (tmp_path / "manifest.json").write_text(manifest)
    settings = {
        "relative_regularizations": config.partial_regularizations,
        "eigenvalue_floor": config.eigenvalue_floor,
        "numerical_error_budget": 1e-12,
        "rho": config.admm_rho,
        "max_iterations": 1,
        "support_rtol": config.support_rtol,
    }
    with pytest.raises(RuntimeError, match="checkpoint retained"):
        _run_replications(config, tmp_path, _prepare(config), settings)
    checkpoint = json.loads((tmp_path / "replication-0000.json").read_text())
    assert checkpoint["error"] is not None
    assert checkpoint["parent_manifest_sha256"] == hashlib.sha256(manifest.encode()).hexdigest()
    assert not checkpoint["candidates"][0]["converged"]
