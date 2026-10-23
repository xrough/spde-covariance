import json

import numpy as np
import pytest

from spde_covariance.benchmark_stress import run_stress_suite


@pytest.fixture(scope="module")
def smoke_results():
    return run_stress_suite(smoke=True)


def test_smoke_results_are_finite_json_and_include_reproduction_settings(smoke_results):
    encoded = json.dumps(smoke_results, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["settings"]["seed"] == 1729
    assert decoded["settings"]["smoke"] is True
    assert decoded["settings"]["exact_rates"] == [0.6, 1.0, 2.0, 4.0]
    assert decoded["settings"]["time_unit"] == "year"
    assert decoded["settings"]["numerical_error_budget"] == 1e-4


def test_transport_bridge_refines_and_agrees_with_independent_quadrature(smoke_results):
    rows = smoke_results["dt_bridge"]
    np.testing.assert_allclose([row["dt"] for row in rows], [1 / 12, 1 / 52, 1 / 252])
    errors = [row["v_minus_c_hs"] for row in rows]
    assert errors[0] > errors[1] > errors[2] > 0
    assert max(row["analytic_vs_gauss_hs"] for row in rows) < 1e-10


def test_irregular_covariance_targets_are_distinct_and_pooling_is_rejected(smoke_results):
    result = smoke_results["irregular_intervals"]
    assert result["pooling_rejected"]
    assert "interval-specific V_i" in result["pooling_rejection_reason"]
    assert not result["diagnostics"]["equal_intervals"]
    assert result["diagnostics"]["delta_min"] == pytest.approx(0.01)
    assert result["diagnostics"]["delta_max"] == pytest.approx(0.03)
    assert result["innovation_covariance_span_hs"] > 0


def test_projection_refinement_separates_interior_tail_and_quadrature(smoke_results):
    result = smoke_results["spline_projection"]
    assert result["quadrature_orders"] == [12, 24]
    for length in [4, 8]:
        rows = [row for row in result["rows"] if row["padding_length"] == length]
        errors = [row["projection_error_h"] for row in rows]
        assert errors[0] > errors[1] > errors[2]
        shifted_errors = [row["shift_function_rmse"] for row in rows]
        assert shifted_errors[0] > shifted_errors[1] > shifted_errors[2]
        tail_errors = [row["omitted_tail_error_h"] for row in rows]
        np.testing.assert_allclose(tail_errors, tail_errors[0], atol=1e-14)
        for row in rows:
            np.testing.assert_allclose(
                row["projection_error_h"] ** 2,
                row["anchor_error"] ** 2
                + row["interior_derivative_error_h"] ** 2
                + row["omitted_tail_error_h"] ** 2,
                rtol=1e-12,
            )
            assert row["projection_identity_discrepancy"] < 1e-10
            assert row["metric_orthogonality_error"] < 1e-9
            assert row["raw_gram_condition"] > 1
            assert row["semigroup_defect"] > 0
            assert row["propagation_method"] == "projected_mild_recursion"
    short = next(row for row in result["rows"] if row["padding_length"] == 4)
    long = next(row for row in result["rows"] if row["padding_length"] == 8)
    assert long["omitted_tail_error_h"] < short["omitted_tail_error_h"]
    assert max(row["projection_quadrature_coefficient_error"] for row in result["rows"]) < 1e-6


def test_conditioning_stress_rejects_unreliable_exponential_whitening(smoke_results):
    result = smoke_results["conditioning"]
    assert result["default_raw_gram_condition"] > 1
    assert result["close_rates_rejected"]
    assert "ill-conditioned" in result["rejection_reason"]


def test_temporal_regime_change_distinguishes_terminal_and_post_change_bias(smoke_results):
    result = smoke_results["temporal_variation"]
    assert result["interval_count"] == 128
    assert result["change_after_interval"] == 64
    assert result["source_regime_change_hs"] > 0
    terminal = [row for row in result["rows"] if row["endpoint_label"] == "terminal"]
    post_change = [row for row in result["rows"] if row["endpoint_label"] == "post_change"]
    assert max(row["window_covariance_bias_hs"] for row in terminal) < 1e-12
    assert post_change[0]["window_covariance_bias_hs"] < 1e-12
    assert post_change[-1]["window_covariance_bias_hs"] > 0.1
    for row in result["rows"]:
        assert row["selected_intervals"] == row["window_intervals"]
        assert row["covariance_assumption"] == "local"
        assert row["converged"]
        assert row["numerical_covariance_error_bound"] <= 1e-4
        assert "local approximation" in row["target"]
        assert row["observed_covariance_rmse"] == row["actual_covariance_error_hs"]
        for metric in (
            "observed_covariance_rmse",
            "off_grid_kernel_rmse",
            "spread_risk_absolute_error",
            "actual_covariance_error_hs",
        ):
            assert np.isfinite(row[metric]) and row[metric] >= 0


def test_performance_uses_common_information_and_meets_the_certificate_budget(smoke_results):
    result = smoke_results["performance"]
    assert [row["basis_size"] for row in result["rows"]] == [4]
    for row in result["rows"]:
        assert row["samples"] == 64
        assert row["converged"]
        assert row["numerical_covariance_error_bound"] <= 1e-4
        assert row["runtime_seconds"] > 0
        assert row["python_tracemalloc_peak_bytes"] > 0
        assert row["solver_iterations"] > 0
        assert row["q_min"] == pytest.approx(0.5**4)
        assert row["q_max"] == pytest.approx(1)
        for metric in (
            "observed_covariance_rmse",
            "off_grid_kernel_rmse",
            "spread_risk_absolute_error",
            "operator_norm_error",
            "relative_operator_error",
        ):
            assert np.isfinite(row[metric]) and row[metric] >= 0
    assert "native BLAS" in result["memory_scope"]
    assert "data" not in result["timing_scope"].split(";")[0]


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": -1},
        {"seed": True},
        {"smoke": 1},
        {"numerical_error_budget": 0},
        {"numerical_error_budget": np.nan},
        {"numerical_error_budget": True},
    ],
)
def test_stress_settings_are_validated_before_running(settings):
    with pytest.raises(ValueError):
        run_stress_suite(**settings)
