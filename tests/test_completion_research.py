"""Controls for controlled ablations, fixed truth and inverse amplification."""

from dataclasses import replace

import numpy as np
import pytest

from spde_covariance.completion_research import (
    _population,
    _replication,
    fit_strategy_grid,
    separator_perturbation_bound,
    strategy_grid,
)
from spde_covariance.completion_research_config import CompletionResearchConfig
from spde_covariance.fragment_benchmark import _array_statistics, _prepare


def small_config(**kwargs):
    defaults = dict(
        replications=1,
        workers=1,
        models=("powerlaw",),
        grids=(21,),
        covers=(3,),
        training_counts=(32,),
        reference_dimension=16,
        validation_count=18,
        test_count=27,
        population_grids=(21, 41),
        forcing_dimensions=(8, 16),
        output_dimensions=(2, 4),
        source_dimensions=(),
        stability=False,
    )
    return CompletionResearchConfig(**(defaults | kwargs))


def test_floor_policy_and_factorial_scope():
    config = small_config(experiment="ablation")
    strategies = strategy_grid(config)
    assert len(strategies) == 16
    settings = strategies["coherent_eta0.001_resolution_support"]
    assert settings.eigenvalue_floor(46) == pytest.approx(1e-5)
    assert 92 * settings.eigenvalue_floor(92) < 46 * settings.eigenvalue_floor(46)
    assert len(strategy_grid(replace(config, experiment="refinement"))) == 3


def test_reused_partial_matches_independent_completion(monkeypatch):
    import spde_covariance.completion_research as research
    from spde_covariance.completion_strategy import (
        CompletionStrategySettings,
        fit_completion_strategy,
    )

    cliques = (np.array([0, 1]), np.array([1, 2]))
    data = np.random.default_rng(41).normal(size=(40, 3))
    statistics = _array_statistics(data, np.arange(40) % 2, cliques, 0.001)
    settings = CompletionStrategySettings(
        partial_method="coherent", floor_policy="constant", relative_regularization=0.001
    )
    first = replace(settings, completion_rule="rank2")
    original = research.fit_completion_strategy
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(research, "fit_completion_strategy", counted)
    fits = fit_strategy_grid(
        statistics,
        cliques=cliques,
        weights=np.ones(3),
        strategies={"rank": first, "support": settings},
    )
    independent = fit_completion_strategy(
        statistics, cliques=cliques, weights=np.ones(3), settings=settings
    )
    assert len(calls) == 1
    assert fits["support"][2]
    assert fits["support"][0].available == independent.available
    np.testing.assert_allclose(fits["support"][0].partial_kernel, independent.partial_kernel)
    if independent.available:
        np.testing.assert_allclose(fits["support"][0].kernel, independent.kernel)


@pytest.mark.parametrize("gap", [1.0, 0.01, 0.0001])
def test_finite_separator_perturbation_certificate(gap):
    a, b = np.array([[0.3, 0.4]]), np.array([[0.2], [0.5]])
    s = np.diag([1.0, gap])
    eps = gap / 10
    audit = separator_perturbation_bound(a, s, b, a + eps, s + eps * np.eye(2), b - eps)
    assert audit["available"]
    assert 0 <= audit["actual"] <= audit["bound"]
    unavailable = separator_perturbation_bound(a, s, b, a, s - gap * np.eye(2), b)
    assert not unavailable["available"]


def test_fixed_reference_decomposes_without_renormalizing_source():
    config = small_config()
    prepared = _prepare(config)
    population, perturbations = _population(config, prepared)
    forcing = [r for r in population if r["record_type"] == "forcing"]
    assert forcing[-1]["forcing_error"] < 1e-12
    assert forcing[0]["forcing_error"] <= forcing[0]["tail_bound"]
    assert all(p["actual"] <= p["bound"] for p in perturbations if p["available"])
    assert any(p["record_type"] == "population_first_join" for p in perturbations)
    assert prepared[1]["powerlaw"]["parent"].source_covariance.trace() < config.source_trace


def test_replication_reproducibility_and_truth_separation():
    config = small_config()
    prepared = _prepare(config)
    first = _replication(0, config, prepared)
    second = _replication(0, config, prepared)
    for a, b in zip(first["rows"], second["rows"], strict=True):
        assert {k: v for k, v in a.items() if k != "runtime_seconds"} == {
            k: v for k, v in b.items() if k != "runtime_seconds"
        }
    # Replacing evaluation truth cannot alter an issued fit or its observed score.
    altered_geometry = {
        key: {**geo, "truth": 10 * geo["truth"], "canonical": 10 * geo["canonical"]}
        for key, geo in prepared[2].items()
    }
    altered = _replication(0, config, (prepared[0], prepared[1], altered_geometry))
    for a, b in zip(first["rows"], altered["rows"], strict=True):
        assert a["test_observed_score"] == b["test_observed_score"]
        assert a["available"] == b["available"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(workers=5),
        dict(forcing_dimensions=(7, 16)),
        dict(source_dimensions=(32,)),
        dict(noise_variance=-1),
    ],
)
def test_invalid_research_configuration(kwargs):
    with pytest.raises(ValueError):
        small_config(**kwargs)
