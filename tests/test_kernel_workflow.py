"""The kernel adapter preserves coordinates, chronology and truth isolation."""

from dataclasses import replace

import numpy as np
import pytest

from spde_covariance import (
    ExponentialTransport,
    FragmentLayout,
    ObservedInnovationBatch,
    extract_innovations,
    fit_covariance_kernel,
    fit_innovation_covariance,
    simulate_transport,
)
from spde_covariance.observations import Observation


def example_data():
    model = ExponentialTransport()
    simulation = simulate_transport(
        model, np.arange(49) / 52, np.diag([0.4, 0.3, 0.2, 0.1]), seed=91
    )
    innovations = extract_innovations(simulation.trajectory, model)
    nodes = np.linspace(0, 3, 13)
    E = model.evaluate(nodes)
    intervals = ((0.0, 1.8), (1.2, 3.0))
    cliques = tuple(np.flatnonzero((nodes >= a) & (nodes <= b)) for a, b in intervals)
    rows = tuple(cliques[i % 2] for i in range(48))
    observations = tuple(
        Observation(
            innovations.times[i + 1],
            E[ids] @ innovations.values[i],
            E[ids],
            np.zeros((len(ids), len(ids))),
        )
        for i, ids in enumerate(rows)
    )
    data = ObservedInnovationBatch(innovations.times, observations, model.basis_id, model.kind)
    layout = FragmentLayout(nodes, np.ones(13) / 13, intervals, rows, E, model.basis_id)
    return data, layout


def test_ridge_adapter_is_exact_legacy_parity():
    data, layout = example_data()
    options = dict(regularization=0.001, trace_cap=2.0, numerical_error_budget=1e-4)
    legacy = fit_innovation_covariance(data, covariance_assumption="common", **options)
    fit = fit_covariance_kernel(
        data, layout=layout, method="ridge", method_options=options, covariance_assumption="common"
    )
    np.testing.assert_array_equal(fit.operator_fit.covariance, legacy.covariance)
    np.testing.assert_array_equal(
        fit.kernel, layout.evaluation_matrix @ legacy.covariance @ layout.evaluation_matrix.T
    )
    assert not hasattr(fit, "covariance")
    assert fit.operator_fit.estimate.covariance_error_bound <= 1e-4
    assert not fit.kernel.flags.writeable


def test_canonical_is_kernel_only_and_truth_is_not_input():
    data, layout = example_data()
    options = dict(rank=2, psd_projection=True)
    fit = fit_covariance_kernel(
        data,
        layout=layout,
        method="canonical",
        method_options=options,
        covariance_assumption="common",
    )
    changed = replace(data, metadata={"latent_truth": np.eye(4) * 1e6})
    again = fit_covariance_kernel(
        changed,
        layout=layout,
        method="canonical",
        method_options=options,
        covariance_assumption="common",
    )
    np.testing.assert_array_equal(fit.kernel, again.kernel)
    assert fit.operator_fit is None
    assert not hasattr(fit, "covariance")
    assert fit.projected_kernel is not None
    assert np.linalg.eigvalsh(fit.projected_kernel)[0] >= -1e-10
    assert "canonical" in fit.estimand


def test_canonical_rejects_irregular_local_or_bad_point_maps():
    data, layout = example_data()
    with pytest.raises(ValueError, match="common"):
        fit_covariance_kernel(
            data,
            layout=layout,
            method="canonical",
            method_options={},
            covariance_assumption="local",
        )
    times = data.times.copy()
    times[1] += 0.001
    observations = tuple(
        replace(item, time=times[i + 1]) for i, item in enumerate(data.observations)
    )
    irregular = replace(data, times=times, observations=observations)
    with pytest.raises(ValueError, match="equal"):
        fit_covariance_kernel(
            irregular,
            layout=layout,
            method="canonical",
            method_options={},
            covariance_assumption="common",
        )
    bad = replace(data, basis_id="different-basis")
    with pytest.raises(ValueError, match="basis"):
        fit_covariance_kernel(
            bad,
            layout=layout,
            method="canonical",
            method_options={},
            covariance_assumption="common",
        )


def test_zero_date_weight_coverage_is_not_filled_as_observed_zero():
    data, layout = example_data()
    weights = np.ones(len(data.observations))
    weights[::2] = 0
    with pytest.raises(ValueError, match="positive date-weight coverage"):
        fit_covariance_kernel(
            data,
            layout=layout,
            method="canonical",
            method_options={"rank": 1, "weights": weights},
            covariance_assumption="common",
        )


def test_touching_fragment_intervals_are_not_positive_measure_overlaps():
    E = np.ones((3, 1))
    rows = (np.array([0, 1]), np.array([1, 2]))
    observations = tuple(
        Observation(i + 1.0, np.ones(2), E[ids], np.zeros((2, 2))) for i, ids in enumerate(rows)
    )
    data = ObservedInnovationBatch(np.arange(3.0), observations, "constant", "exact_transport")
    layout = FragmentLayout(
        np.arange(3.0), np.ones(3), ((0.0, 1.0), (1.0, 2.0)), rows, E, "constant"
    )
    with pytest.raises(ValueError, match="positive-length"):
        fit_covariance_kernel(
            data,
            layout=layout,
            method="canonical",
            method_options={"rank": 1},
            covariance_assumption="common",
        )
