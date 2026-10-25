"""Finite-sample formulas, structural uncertainty and dual outer-bound controls."""

import numpy as np
import pytest
from scipy.stats import chi2

from spde_covariance.completion_uncertainty import (
    _maximize_region,
    _repair_identity_dual,
    completion_risk_interval,
    gaussian_completion_region,
)
from spde_covariance.hedging import CovarianceAmbiguity
from spde_covariance.hedging_statistics import gaussian_pairwise_bands


def test_exact_zero_mean_chi_square_formula_and_derived_trace_cap():
    y = np.array([[1.0], [2.0], [-1.0], [0.0]])
    region = gaussian_completion_region(
        y, (np.array([0]),) * 4, noise_covariances=0.2, size=1, confidence_level=0.9
    )
    expected = np.array([6 / chi2.ppf(0.95, 4), 6 / chi2.ppf(0.05, 4)]) - 0.2
    np.testing.assert_allclose([region.lower[0, 0], region.upper[0, 0]], expected)
    assert region.corrected_covariance[0, 0] == pytest.approx(1.3)
    assert region.available and region.ambiguity.calibrated_bands
    assert region.ambiguity.trace_cap == region.upper[0, 0]
    assert region.diagnostics["scalar_intervals"] == 1
    assert region.diagnostics["canonical_assumption"] is False


def test_homogeneous_regression_matches_existing_gaussian_bands():
    y = np.array(
        [[1.0, 2.0, np.nan], [-1.0, 1.0, np.nan], [np.nan, 3.0, -2.0], [np.nan, -2.0, 1.0]]
    )
    rows = (np.array([0, 1]), np.array([0, 1]), np.array([1, 2]), np.array([1, 2]))
    noise = 0.1 * np.eye(3)
    old = gaussian_pairwise_bands(y, noise_covariance=noise)
    new = gaussian_completion_region(y, rows, noise_covariances=noise, size=3)
    ids = tuple(old.entry_pairs.T)
    np.testing.assert_allclose(new.lower[ids], old.lower, rtol=1e-12)
    np.testing.assert_allclose(new.upper[ids], old.upper, rtol=1e-12)
    np.testing.assert_allclose(new.corrected_covariance, old.corrected_covariance, equal_nan=True)
    assert np.isnan(new.lower[0, 2]) and not new.observed_mask[0, 2]
    assert new.diagnostics["scalar_intervals"] == old.diagnostics["scalar_intervals"]


def test_heterogeneous_noise_groups_by_projected_probe_variance():
    rows = tuple(np.array([0, 1]) for _ in range(4))
    values = ((1.0, 2.0), (-1.0, 1.0), (2.0, -1.0), (-2.0, -2.0))
    plus = np.array([[0.1, 0.05], [0.05, 0.1]])
    minus = np.array([[0.1, -0.05], [-0.05, 0.1]])
    region = gaussian_completion_region(
        values, rows, noise_covariances=(plus, plus, minus, minus), size=2
    )
    groups = region.diagnostics["scalar_group_intervals"]
    assert region.diagnostics["scalar_probes"] == 4
    assert (
        region.diagnostics["scalar_intervals"] == 6
    )  # two node groups, four sum/difference groups
    assert sorted(g["count"] for g in groups) == [2, 2, 2, 2, 4, 4]
    tail = 0.05 / (2 * 6)
    assert region.diagnostics["one_sided_tail_probability"] == pytest.approx(tail)
    assert region.diagnostics["allocated_miscoverage"] == pytest.approx(0.05)
    for group in groups:
        assert group["lower"] == pytest.approx(
            group["sum_squares"] / chi2.isf(tail, group["count"]) - group["noise_variance"]
        )
        assert group["upper"] == pytest.approx(
            group["sum_squares"] / chi2.ppf(tail, group["count"]) - group["noise_variance"]
        )
    probe = {(x["probe"], tuple(x["nodes"])): x for x in region.diagnostics["probe_intervals"]}
    for kind in ("sum", "difference"):
        relevant = [g for g in groups if g["probe"] == kind]
        assert probe[kind, (0, 1)]["lower"] == max(g["lower"] for g in relevant)
        assert probe[kind, (0, 1)]["upper"] == min(g["upper"] for g in relevant)
    p, m = probe["sum", (0, 1)], probe["difference", (0, 1)]
    assert region.lower[0, 1] == (p["lower"] - m["upper"]) / 4
    assert region.upper[0, 1] == (p["upper"] - m["lower"]) / 4


def test_bonferroni_allocation_depends_on_masks_and_noise_not_values():
    rows = (np.array([0]), np.array([0]), np.array([1]), np.array([1]))
    first = gaussian_completion_region(
        ((1.0,), (2.0,), (3.0,), (4.0,)), rows, noise_covariances=(0.1, 0.2, 0.1, 0.1), size=2
    )
    second = gaussian_completion_region(
        ((2.0,), (-1.0,), (7.0,), (1.0,)), rows, noise_covariances=(0.1, 0.2, 0.1, 0.1), size=2
    )
    assert first.diagnostics["scalar_intervals"] == second.diagnostics["scalar_intervals"] == 3
    assert (
        first.diagnostics["one_sided_tail_probability"]
        == second.diagnostics["one_sided_tail_probability"]
    )
    assert first.pair_counts[0, 1] == 0
    assert np.isnan(first.lower[0, 1])


def test_dense_mask_ignores_unobserved_nan_and_hidden_truth():
    rows = (np.array([0, 1]), np.array([1, 2]), np.array([0, 1]), np.array([1, 2]))
    visible = np.array(
        [[1.0, 2.0, np.nan], [np.nan, -1.0, 3.0], [-1.0, 1.0, np.nan], [np.nan, 2.0, -2.0]]
    )
    changed = visible.copy()
    changed[np.isnan(changed)] = 1e200
    a = gaussian_completion_region(visible, rows, noise_covariances=np.eye(3) * 0.1, size=3)
    b = gaussian_completion_region(changed, rows, noise_covariances=np.eye(3) * 0.1, size=3)
    np.testing.assert_array_equal(a.lower, b.lower)
    np.testing.assert_array_equal(a.upper, b.upper)
    np.testing.assert_array_equal(a.corrected_covariance, b.corrected_covariance)


def test_row_reordering_and_curve_unit_scaling():
    values = (np.array([1.0, 2.0]), np.array([-2.0, 1.0]), np.array([3.0, -1.0]))
    rows = tuple(np.array([0, 1]) for _ in values)
    noise = np.array([[0.2, 0.04], [0.04, 0.1]])
    a = gaussian_completion_region(values, rows, noise_covariances=(noise,) * 3, size=2)
    b = gaussian_completion_region(
        tuple(x[::-1] for x in values),
        tuple(r[::-1] for r in rows),
        noise_covariances=(noise[::-1, ::-1],) * 3,
        size=2,
    )
    np.testing.assert_allclose(a.lower, b.lower)
    np.testing.assert_allclose(a.upper, b.upper)
    c = gaussian_completion_region(
        tuple(7 * x for x in values), rows, noise_covariances=(49 * noise,) * 3, size=2
    )
    np.testing.assert_allclose(c.lower, 49 * a.lower, rtol=1e-12)
    np.testing.assert_allclose(c.upper, 49 * a.upper, rtol=1e-12)


def test_negative_corrected_bounds_are_not_clipped_or_projected():
    region = gaussian_completion_region(
        np.array([[0.0], [0.1]]), (np.array([0]),) * 2, noise_covariances=0.1, size=1
    )
    assert region.lower[0, 0] < 0 < region.upper[0, 0]
    assert region.corrected_covariance[0, 0] < 0
    assert region.available
    zero = gaussian_completion_region(
        np.zeros((5, 1)), (np.array([0]),) * 5, noise_covariances=1.0, size=1
    )
    assert zero.lower[0, 0] == zero.upper[0, 0] == -1
    assert not zero.available and zero.ambiguity is None
    assert "negative upper" in zero.reason


def test_empty_noise_group_intersection_stays_empty_and_risk_unavailable():
    region = gaussian_completion_region(
        np.array([[0.0], [0.0], [10.0], [10.0]]),
        (np.array([0]),) * 4,
        noise_covariances=(0.0, 0.0, 0.1, 0.1),
        size=1,
    )
    assert region.lower[0, 0] > region.upper[0, 0]
    assert not region.available and "empty noise-group" in region.reason
    result = completion_risk_interval(region, [1.0])
    assert not result.available and result.lower is None and result.upper is None
    assert result.minimizer.diagnostics["status"] == "region_unavailable"


def test_missing_node_cannot_acquire_an_arbitrary_trace_cap():
    region = gaussian_completion_region(
        ((1.0,), (-1.0,)), (np.array([0]),) * 2, noise_covariances=0.0, size=2
    )
    assert not region.available and "unobserved nodes" in region.reason
    assert region.ambiguity is None and region.diagnostics["trace_cap"] is None
    assert np.isnan(region.upper[1, 1])


def test_immutable_arrays_nested_diagnostics_and_no_input_alias():
    values = np.array([[1.0], [-1.0]])
    region = gaussian_completion_region(values, (np.array([0]),) * 2, noise_covariances=0.0, size=1)
    original = region.upper.copy()
    values[:] = 100
    np.testing.assert_array_equal(region.upper, original)
    with pytest.raises(ValueError):
        region.upper.setflags(write=True)
    with pytest.raises(TypeError):
        region.diagnostics["scalar_group_intervals"][0]["count"] = 99


def test_identity_dual_repair_is_an_outer_bound_and_counts_offdiagonals_once():
    model = CovarianceAmbiguity(
        (np.eye(2),), np.array([[0, 0], [1, 1]]), np.ones(2), np.ones(2), 2.0
    )
    a = np.array([1.0, -1.0])
    objective = np.outer(a, a)
    repaired = _repair_identity_dual(
        objective, model, -1.0, np.array([-2.0, 0.0]), np.array([-3.0, 0.0])
    )
    assert repaired["available"]
    assert repaired["upper_bound"] >= 4
    assert repaired["nonnegative_multiplier_correction"] == 3
    assert repaired["minimum_slack_eigenvalue"] >= 0
    model2 = CovarianceAmbiguity(
        (np.eye(2),), np.array([[0, 1]]), np.array([-0.5]), np.array([0.5]), 2.0
    )
    repaired2 = _repair_identity_dual(
        np.zeros((2, 2)), model2, 2.0, np.array([0.0]), np.array([2.0])
    )
    expected = np.array(
        [[repaired2["trace_multiplier"], 1.0], [1.0, repaired2["trace_multiplier"]]]
    )
    np.testing.assert_allclose(repaired2["dual_slack"], expected)


def test_no_joint_pair_has_nontrivial_outer_risk_range_not_canonical_CI():
    pytest.importorskip("cvxpy")
    rows = tuple(np.array([i % 2]) for i in range(40))
    values = tuple(np.array([1.0 if i % 4 < 2 else -1.0]) for i in range(40))
    region = gaussian_completion_region(values, rows, noise_covariances=0.0, size=2)
    assert region.available and not region.observed_mask[0, 1]
    result = completion_risk_interval(region, [1.0, -1.0], tolerance=1e-9)
    assert result.available
    exact_upper = (np.sqrt(region.upper[0, 0]) + np.sqrt(region.upper[1, 1])) ** 2
    assert result.lower <= 1e-10
    assert result.lower == 0
    assert result.diagnostics["raw_dual_lower"] <= 0
    assert result.upper >= exact_upper - 1e-9
    assert result.upper == pytest.approx(exact_upper, rel=1e-6)
    assert result.lower <= result.inner_lower + 1e-7
    assert result.upper >= result.inner_upper - 1e-7
    assert result.diagnostics["outer_bounds"] and not result.diagnostics["numerical_certificate"]
    assert not result.diagnostics["canonical_assumption"]
    assert result.minimizer.diagnostics["dual"]["available"]
    assert result.maximizer.diagnostics["dual"]["available"]


@pytest.mark.parametrize("solver", ["SCS", "CLARABEL"])
def test_dual_outer_endpoints_match_analytic_bounded_cross_covariance(solver):
    pytest.importorskip("cvxpy")
    model = CovarianceAmbiguity(
        (np.eye(2),),
        np.array([[0, 0], [0, 1], [1, 1]]),
        np.array([1.0, 0.2, 1.0]),
        np.array([1.0, 0.4, 1.0]),
        2.0,
    )
    a = np.array([1.0, -1.0])
    options = dict(solver=solver, tolerance=1e-9, max_iterations=100000, feasibility_tolerance=1e-6)
    _, maximum, high = _maximize_region(model, np.outer(a, a), **options)
    _, negative_minimum, low = _maximize_region(model, -np.outer(a, a), **options)
    assert high["available"] and low["available"]
    assert maximum == pytest.approx(1.6, abs=1e-7)
    assert -negative_minimum == pytest.approx(1.2, abs=1e-7)
    assert high["dual"]["upper_bound"] >= 1.6 - 1e-10
    assert -low["dual"]["upper_bound"] <= 1.2 + 1e-10
    assert high["dual"]["upper_bound"] == pytest.approx(1.6, abs=1e-6)
    assert low["dual"]["upper_bound"] == pytest.approx(-1.2, abs=1e-6)


@pytest.mark.parametrize(
    "change",
    [
        {"confidence_level": 1.0},
        {"confidence_level": True},
        {"size": False},
        {"values": ((np.nan,),)},
        {"row_to_nodes": (np.array([0, 0]),)},
        {"noise_covariances": -1.0},
        {"noise_covariances": (np.array([[-1.0]]),)},
    ],
)
def test_invalid_data_rejected(change):
    arguments = dict(values=((1.0,),), row_to_nodes=(np.array([0]),), noise_covariances=0.0, size=1)
    arguments.update(change)
    with pytest.raises(ValueError):
        gaussian_completion_region(**arguments)
