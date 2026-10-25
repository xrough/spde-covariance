"""Statistical completion preserves data boundaries, graph and target units."""

from dataclasses import replace

import numpy as np
import pytest

from spde_covariance.fragments import FragmentLayout
from spde_covariance.gaussian_workflow import (
    GaussianCompletionConvergenceError,
    gaussian_kernel_validation_score,
    select_gaussian_completion,
)
from spde_covariance.kernel_workflow import fit_covariance_kernel
from spde_covariance.observations import Observation
from spde_covariance.workflow import ObservedInnovationBatch


def sample(start, count, *, full=False, seed=46):
    kernel = np.array([[1, 0.4, 0.16], [0.4, 1, 0.4], [0.16, 0.4, 1]])
    rng = np.random.default_rng(seed)
    parents = rng.multivariate_normal(np.zeros(3), kernel + 0.02 * np.eye(3), count)
    times = np.arange(start, start + count + 1) / 52
    intervals = ((0.0, 2.0),) if full else ((0.0, 1.1), (0.9, 2.0))
    rows = tuple(np.arange(3) if full else np.array([i % 2, i % 2 + 1]) for i in range(count))
    observations = tuple(
        Observation(times[i + 1], parents[i, row], np.eye(3)[row], 0.02 * np.eye(len(row)))
        for i, row in enumerate(rows)
    )
    batch = ObservedInnovationBatch(times, observations, "grid-control", "direct Gaussian control")
    layout = FragmentLayout(np.arange(3), np.ones(3), intervals, rows, np.eye(3), batch.basis_id)
    return batch, layout


def fit(data, layout, **options):
    return fit_covariance_kernel(
        data,
        layout=layout,
        method="gaussian",
        covariance_assumption="common",
        method_options={"reference_kernel": np.eye(3), "regularization": 0.02, **options},
    )


def test_gaussian_adapter_preserves_kernel_target_and_truth_isolation():
    data, layout = sample(0, 80)
    own = fit(data, layout)
    changed = fit(replace(data, metadata={"latent_truth": np.ones((3, 3)) * 1e9}), layout)
    np.testing.assert_array_equal(own.kernel, changed.kernel)
    assert np.linalg.eigvalsh(own.kernel)[0] > 0
    assert abs(own.diagnostics["precision"][0, 2]) < 1e-12
    assert own.operator_fit is None and own.projected_kernel is None
    assert not hasattr(own, "covariance")
    assert not own.kernel.flags.writeable
    assert "Gaussian Markov" in own.estimand
    assert "local stationarity" in own.diagnostics["certificate_scope"]
    assert own.diagnostics["gaussian_model"]["distinct_initial_kernel_count"] == 3


def test_bridge_observation_layout_does_not_silently_change_the_model_graph():
    data, layout = sample(0, 80, full=True)
    own = fit(data, layout, model_cliques=((0, 1), (1, 2)))
    assert abs(own.diagnostics["precision"][0, 2]) < 1e-12
    unrestricted = fit(data, layout)
    assert abs(unrestricted.diagnostics["precision"][0, 2]) > 1e-3


def test_selection_is_chronological_observed_only_and_does_not_refit_validation():
    training, train_layout = sample(0, 80)
    validation, valid_layout = sample(80, 32, seed=47)
    selected = select_gaussian_completion(
        training,
        validation,
        training_layout=train_layout,
        validation_layout=valid_layout,
        reference_kernel=np.eye(3),
        regularizations=[0.01, 0.1],
        covariance_assumption="common",
    )
    assert selected.available
    assert selected.selected_regularization in (0.01, 0.1)
    assert selected.fit.metadata["times"][-1] == training.times[-1]
    assert not selected.diagnostics["refitted_on_validation"]
    assert not selected.diagnostics["unobserved_conditional_independence_validated"]
    scores = [r["validation_negative_loglikelihood"] for r in selected.diagnostics["candidates"]]
    assert gaussian_kernel_validation_score(selected.fit, validation, layout=valid_layout) == min(
        scores
    )
    truth_changed = replace(validation, metadata={"future_truth": -np.eye(3) * 1e6})
    assert gaussian_kernel_validation_score(
        selected.fit, truth_changed, layout=valid_layout
    ) == min(scores)
    future, future_layout = sample(112, 20, full=True, seed=48)
    original = selected.fit.kernel.copy()
    gaussian_kernel_validation_score(selected.fit, future, layout=future_layout)
    np.testing.assert_array_equal(selected.fit.kernel, original)
    with pytest.raises(ValueError, match="training endpoint"):
        select_gaussian_completion(
            training,
            training,
            training_layout=train_layout,
            validation_layout=train_layout,
            reference_kernel=np.eye(3),
            regularizations=[0.1],
            covariance_assumption="common",
        )


def test_equal_native_likelihood_cannot_validate_invisible_completion_choice():
    data, layout = sample(0, 60)
    own = fit(data, layout)
    changed = own.kernel.copy()
    changed[0, 2] = changed[2, 0] = own.kernel[0, 2] + 0.1
    assert np.linalg.eigvalsh(changed)[0] > 0
    other = replace(own, kernel=changed)
    assert gaussian_kernel_validation_score(
        own, data, layout=layout
    ) == gaussian_kernel_validation_score(other, data, layout=layout)
    bridge, bridge_layout = sample(60, 100, full=True, seed=49)
    assert gaussian_kernel_validation_score(
        own, bridge, layout=bridge_layout
    ) != gaussian_kernel_validation_score(other, bridge, layout=bridge_layout)


def test_failed_local_fit_is_explicit_and_selection_retains_the_attempt():
    train, tl = sample(0, 80)
    valid, vl = sample(80, 32, seed=47)
    with pytest.raises(GaussianCompletionConvergenceError):
        fit(train, tl, max_iterations=1, stationarity_tolerance=1e-12)
    selected = select_gaussian_completion(
        train,
        valid,
        training_layout=tl,
        validation_layout=vl,
        reference_kernel=np.eye(3),
        regularizations=[0.02],
        covariance_assumption="common",
        fit_options={"max_iterations": 1, "stationarity_tolerance": 1e-12},
    )
    assert not selected.available and selected.fit is None
    assert not selected.diagnostics["candidates"][0]["available"]


def test_gaussian_adapter_rejects_implicit_prior_local_target_and_wrong_units():
    data, layout = sample(0, 20)
    with pytest.raises(ValueError, match="explicit"):
        fit_covariance_kernel(
            data,
            layout=layout,
            method="gaussian",
            method_options={},
            covariance_assumption="common",
        )
    with pytest.raises(ValueError, match="common"):
        fit_covariance_kernel(
            data, layout=layout, method="gaussian", method_options={}, covariance_assumption="local"
        )
    with pytest.raises(ValueError, match="unknown"):
        fit(data, layout, psd_projection=True)
    own = fit(data, layout)
    with pytest.raises(ValueError, match="units"):
        gaussian_kernel_validation_score(own, replace(data, curve_unit="different"), layout=layout)
