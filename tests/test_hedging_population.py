"""Nested population observations restrict one fixed hedge ambiguity set."""

import numpy as np
import pytest

from spde_covariance.hedging import CovarianceFunctionalInterval, CovarianceFunctionalResult
from spde_covariance.hedging_controls import fixed_power_hedge_control
from spde_covariance.hedging_population import run_population_hedge_study


@pytest.fixture(scope="module", params=["spread", "point"])
def study(request):
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("CLARABEL is required for the population source intervals")
    control = fixed_power_hedge_control(
        source_dimension=8,
        output_dimension=8,
        reference_dimension=128,
        exposure=request.param,
    )
    return control, run_population_hedge_study(control, solver_tolerance=1e-8)


def test_population_designs_keep_the_population_source_family_fixed(study):
    control, result = study
    assert result["all_source_intervals_available"]
    assert result["statistical_confidence_interval"] is False
    assert result["infinite_dimensional_identification_claim"] is False
    np.testing.assert_array_equal(result["baseline_covariance"], control["baseline_covariance"])
    counts = [row["observed_pair_count"] for row in result["designs"].values()]
    assert counts == [19, 22, 22, 28]
    for row in result["designs"].values():
        assert row["source_trace_lower"] == row["source_trace_cap"] == control["trace_value"]
        assert row["baseline_source_feasibility"]["scaled_max_violation"] < 1e-12
    for check in result["nesting_diagnostics"].values():
        assert check["observed_mask_contains_baseline"]
        assert check["source_interval_inside_baseline"]
        assert check["maximum_interval_nesting_violation"] <= 1e-5


def test_source_endpoints_have_feasible_witnesses_and_observed_known_variances(study):
    control, result = study
    target, hedge = control["target"], control["hedge"]
    for row in result["designs"].values():
        source = row["source_interval"]
        assert source["available"]
        assert np.max(np.abs(source["endpoint_known_variance_mismatches"])) < 2e-7
        for witness in (source["minimizer"], source["maximizer"]):
            assert witness["available"]
            assert witness["diagnostics"]["status"] == "optimal"
            assert witness["diagnostics"]["feasibility"]["scaled_max_violation"] <= 1e-7
            covariance, block = witness["covariance"], witness["source_covariance"]
            assert np.trace(block) == pytest.approx(control["trace_value"], abs=1e-7)
            mask = row["observed_mask"]
            np.testing.assert_allclose(
                covariance[mask], control["baseline_covariance"][mask], atol=2e-7
            )
            assert witness["candidate_value"] == pytest.approx(
                target @ covariance @ hedge, abs=1e-14
            )
        interval = source["hedge_interval"]
        free = row["free_psd_fixed_tail_interval"]["hedge_interval"]
        assert interval[0] >= free[0] - 1e-5
        assert interval[1] <= free[1] + 1e-5
    complete = result["designs"]["complete"]["source_interval"]
    assert complete["observed_functional_identified"]
    assert np.ptp(complete["hedge_interval"]) == 0
    assert complete["hedge_interval"][0] == pytest.approx(result["population_oracle"]["hedge"])


def test_population_scalar_decisions_separate_variance_from_regret(study):
    control, result = study
    baseline = result["designs"]["baseline"]
    source = baseline["source_interval"]
    low, high = source["hedge_interval"]
    decisions = source["scalar_decisions"]
    variance = baseline["known_variances"]["hedge"]
    assert decisions["minimax_regret_hedge"] == pytest.approx((low + high) / 2)
    assert decisions["minimax_regret"] == pytest.approx(variance * (high - low) ** 2 / 4)
    assert decisions["minimax_variance_hedge"] == pytest.approx(np.clip(0, low, high))
    if control["config"]["exposure"] == "spread":
        np.testing.assert_allclose([low, high], [0.94185511, 7.21001956], atol=2e-4)
        assert decisions["minimax_regret_hedge"] != pytest.approx(
            decisions["minimax_variance_hedge"]
        )
        narrower = result["designs"]["add_maturity_3_to_left"]["source_interval"]
        assert narrower["scalar_decisions"]["minimax_regret"] < decisions["minimax_regret"] / 100
    complete = result["designs"]["complete"]["source_interval"]["scalar_decisions"]
    assert complete["minimax_regret"] == 0
    if control["config"]["exposure"] == "point":
        identified = result["designs"]["add_maturity_0_to_right"]["source_interval"]
        assert identified["observed_functional_identified"]
        assert np.ptp(identified["hedge_interval"]) == 0


def test_canonical_completion_is_an_explicit_population_estimate_without_a_source_claim(study):
    control, result = study
    for row in result["designs"].values():
        canonical = row["canonical_estimate"]
        assert canonical["available"]
        assert canonical["source_compatibility"].startswith("unverified")
        mask = row["observed_mask"]
        np.testing.assert_array_equal(
            canonical["kernel"][mask], control["baseline_covariance"][mask]
        )
        lo, hi = row["free_psd_interval"]["hedge_interval"]
        assert lo - 1e-6 <= canonical["hedge"] <= hi + 1e-6
        residual = control["target"] - canonical["hedge"] * control["hedge"]
        assert canonical["population_residual_variance"] == pytest.approx(
            residual @ control["baseline_covariance"] @ residual,
        )


def test_inaccurate_solver_candidate_is_preserved_but_never_promoted_to_an_interval(monkeypatch):
    control = fixed_power_hedge_control(
        source_dimension=8, output_dimension=8, reference_dimension=32
    )
    candidate = CovarianceFunctionalResult(
        123.0,
        np.eye(8),
        np.eye(7),
        {"available": False, "status": "optimal_inaccurate", "numerical_certificate": False},
    )
    failed = CovarianceFunctionalInterval(
        None,
        None,
        candidate,
        candidate,
        {"available": False, "numerical_certificate": False},
    )
    monkeypatch.setattr(
        "spde_covariance.hedging_population.covariance_functional_interval",
        lambda *args, **kwargs: failed,
    )
    result = run_population_hedge_study(control)
    assert not result["all_source_intervals_available"]
    for row in result["designs"].values():
        source = row["source_interval"]
        assert not source["available"]
        assert source["hedge_interval"] is None
        assert source["scalar_decisions"] is None
        assert source["minimizer"]["candidate_value"] == 123.0
        assert row["canonical_estimate"]["available"]


@pytest.mark.parametrize("value", [0, -1, np.nan, True])
def test_population_solver_tolerance_is_validated(value):
    with pytest.raises(ValueError):
        run_population_hedge_study({}, solver_tolerance=value)
