import inspect

import numpy as np
import pytest

from spde_covariance.baselines import (
    empirical_covariance,
    linear_shrinkage_covariance,
    projected_ridge_covariance,
    refit_selection,
    select_covariance,
    validation_score,
)
from spde_covariance.estimation import fit_covariance
from spde_covariance.observations import Observation


def _data(values, operator=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    operator = np.eye(values.shape[1]) if operator is None else np.asarray(operator)
    return tuple(
        Observation(
            float(index), operator @ value, operator, np.zeros((len(operator), len(operator)))
        )
        for index, value in enumerate(values)
    )


def _balanced(covariance, operator=None):
    dimension = len(covariance)
    return _data(np.sqrt(dimension) * np.linalg.cholesky(covariance).T, operator)


def test_empirical_weights_and_trace_preserving_linear_shrinkage():
    data = _data([[1, 0], [0, 2]])
    expected = np.diag([0.25, 3])
    np.testing.assert_allclose(empirical_covariance(data, [1, 3]), expected)
    shrunk = linear_shrinkage_covariance(data, 0.5, [1, 3])
    np.testing.assert_allclose(shrunk, 0.5 * expected + 0.5 * np.trace(expected) / 2 * np.eye(2))
    assert np.trace(shrunk) == pytest.approx(np.trace(expected))
    np.testing.assert_allclose(
        empirical_covariance(data, [1e308, 1e308]), empirical_covariance(data)
    )


def test_native_full_baselines_do_not_clip_noise_corrected_measurements():
    data = [Observation(0, np.zeros(2), np.eye(2), np.eye(2))]
    np.testing.assert_array_equal(empirical_covariance(data), -np.eye(2))
    np.testing.assert_array_equal(linear_shrinkage_covariance(data, 0.5), -np.eye(2))


@pytest.mark.parametrize(
    "operator", [np.array([[1.0, 0.0]]), np.diag([1.0, 2.0]), np.array([[0.0, 1.0], [1.0, 0.0]])]
)
def test_full_observation_baselines_reject_partial_or_transformed_data(operator):
    data = _data([[1, 0], [0, 1]], operator)
    with pytest.raises(ValueError, match="identity"):
        empirical_covariance(data)
    with pytest.raises(ValueError, match="identity"):
        linear_shrinkage_covariance(data, 0.5)


def test_full_identity_postprojected_ridge_equals_constrained_optimizer():
    data = _balanced(np.array([[0.8, 0.2], [0.2, 0.5]]))
    projected = projected_ridge_covariance(data, regularization=0.2, trace_cap=0.8)
    constrained = fit_covariance(
        data, regularization=0.2, trace_cap=0.8, absolute_tolerance=1e-12, relative_tolerance=0
    )
    assert constrained.converged
    np.testing.assert_allclose(projected, constrained.covariance, atol=1e-10)


def test_partial_design_euclidean_postprojection_is_not_constrained_optimizer():
    data = _balanced(np.array([[1.0, 0.8], [0.8, 1.0]]), np.diag([1.0, 0.2]))
    regularization, trace_cap = 0.1, 0.7
    projected = projected_ridge_covariance(data, regularization=regularization, trace_cap=trace_cap)
    constrained = fit_covariance(
        data,
        regularization=regularization,
        trace_cap=trace_cap,
        absolute_tolerance=1e-12,
        relative_tolerance=0,
    )

    def objective(matrix):
        return (
            sum(
                np.sum((item.operator @ matrix @ item.operator.T - item.measurement) ** 2)
                for item in data
            )
            / (2 * len(data))
            + regularization * np.sum(matrix**2) / 2
        )

    assert np.trace(projected) <= trace_cap + 1e-12
    assert np.linalg.eigvalsh(projected)[0] >= -1e-12
    assert objective(projected) > constrained.objective + 1e-4


def test_validation_score_omits_only_a_candidate_independent_constant():
    data = _data([[1, 2], [0, 1]], np.array([[1.0, -0.5]]))
    candidates = [np.eye(2), np.diag([0.5, 2.0])]
    omitted = np.mean([np.sum(item.measurement**2) for item in data])
    full_losses = []
    scores = []
    for covariance in candidates:
        loss = np.mean(
            [
                np.sum((item.operator @ covariance @ item.operator.T - item.measurement) ** 2)
                for item in data
            ]
        )
        score = validation_score(covariance, data)
        assert loss - score == pytest.approx(omitted)
        full_losses.append(loss)
        scores.append(score)
    assert np.argmin(full_losses) == np.argmin(scores)


def test_empirical_selection_has_only_window_candidates_and_is_deterministic():
    training = _data([4, 4, 1, 1])
    validation = _data([1, 1])
    options = dict(
        method="empirical", relative_candidates=None, window_candidates=[2, 4], trace_cap=0.1
    )
    selected = select_covariance(training, validation, **options)
    repeated = select_covariance(training, validation, **options)
    assert selected.selected_parameters == {"window": 2}
    assert selected.estimate is None
    assert len(selected.candidates) == 2
    assert selected.validation_score == pytest.approx(-1)
    assert selected.candidates == repeated.candidates
    np.testing.assert_array_equal(selected.covariance, [[1]])
    np.testing.assert_array_equal(selected.covariance, repeated.covariance)
    assert np.trace(selected.covariance) > options["trace_cap"]
    assert not selected.covariance.flags.writeable


@pytest.mark.parametrize("method", ["constrained_ridge", "projected_ridge"])
def test_relative_ridge_selection_and_refit_recalibrate_actual_regularization(method):
    training = _data([np.sqrt(2)] * 4)
    validation = _data([1, 1])
    selection = select_covariance(
        training,
        validation,
        method=method,
        relative_candidates=[0.1, 1.0],
        window_candidates=[2, 4],
        trace_cap=3,
    )
    assert selection.selected_parameters["window"] == 2
    assert selection.selected_parameters["relative_regularization"] == 1
    assert selection.selected_parameters["gram_operator_norm"] == pytest.approx(1)
    assert selection.selected_parameters["regularization"] == pytest.approx(1)
    scaled_data = _data([np.sqrt(2)] * 4, np.array([[2.0]]))
    refitted = refit_selection(scaled_data, selection, trace_cap=3)
    assert refitted.selected_parameters["relative_regularization"] == 1
    assert refitted.selected_parameters["gram_operator_norm"] == pytest.approx(16)
    assert refitted.selected_parameters["regularization"] == pytest.approx(16)
    assert refitted.validation_score == selection.validation_score
    assert refitted.candidates == selection.candidates
    np.testing.assert_allclose(refitted.covariance, selection.covariance, atol=1e-10)


def test_shrinkage_selection_records_alpha_without_ridge_parameters():
    training = _data([[np.sqrt(8), 0], [0, 0]])
    validation = _balanced(2 * np.eye(2))
    selected = select_covariance(
        training,
        validation,
        method="linear_shrinkage",
        relative_candidates=[0, 1],
        window_candidates=[2],
        trace_cap=0.1,
    )
    assert selected.selected_parameters == {"window": 2, "shrinkage": 1.0}
    np.testing.assert_allclose(selected.covariance, 2 * np.eye(2))
    refitted = refit_selection(training, selected, trace_cap=0.1)
    np.testing.assert_allclose(refitted.covariance, selected.covariance)


def test_failed_constrained_candidate_is_skipped_and_all_failure_raises():
    training = _data([np.sqrt(2)] * 2)
    validation = _data([1])
    selected = select_covariance(
        training,
        validation,
        method="constrained_ridge",
        relative_candidates=[1, 2],
        window_candidates=[2],
        trace_cap=3,
        solver_options={"max_iterations": 0, "initial": np.eye(1)},
    )
    assert [item["status"] for item in selected.candidates] == ["converged", "failed"]
    assert selected.selected_parameters["relative_regularization"] == 1
    with pytest.raises(RuntimeError, match="all covariance candidates failed"):
        select_covariance(
            training,
            validation,
            method="constrained_ridge",
            relative_candidates=[0.1],
            window_candidates=[2],
            trace_cap=3,
            solver_options={"max_iterations": 0},
        )


def test_selection_and_refit_have_no_latent_truth_or_model_inputs():
    for function in (select_covariance, refit_selection):
        assert not {"truth", "model", "source_covariance"}.intersection(
            inspect.signature(function).parameters
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        select_covariance(
            _data([1, 2]),
            _data([1]),
            method="empirical",
            relative_candidates=None,
            window_candidates=[2],
            trace_cap=3,
            truth=np.eye(1),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "unknown"},
        {"relative_candidates": [0]},
        {"window_candidates": [0]},
        {"window_candidates": [3]},
        {"numerical_error_budget": 0},
        {"solver_options": {"regularization": 1}},
        {"solver_options": {"max_iterations": -1}},
        {"solver_options": {"relative_tolerance": -1}},
    ],
)
def test_invalid_selection_configuration(kwargs):
    options = dict(
        method="constrained_ridge", relative_candidates=[0.1], window_candidates=[2], trace_cap=3
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        select_covariance(_data([1, 2]), _data([1]), **options)
