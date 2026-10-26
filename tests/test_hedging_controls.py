"""A fixed infinite-rank control links missing covariance to hedge decisions."""

import numpy as np
import pytest

from spde_covariance.hedging_controls import (
    fixed_power_hedge_control,
    free_psd_clique_hedge_interval,
)
from spde_covariance.laguerre import PowerLawSource


@pytest.fixture(scope="module")
def control():
    return fixed_power_hedge_control(reference_dimension=128)


def test_control_has_two_equal_trace_psd_sources_with_one_fixed_infinite_tail(control):
    source = PowerLawSource(exponent=3, trace=1)
    np.testing.assert_array_equal(control["finite_source_base"], source.finite_matrix(16))
    assert np.trace(control["finite_source_base"]) < 1
    for name in ("source_minus", "source_plus"):
        covariance = control[name]
        assert np.linalg.eigvalsh(covariance)[0] > 1e-4
        assert np.trace(covariance) + source.tail_trace(16) == pytest.approx(1, abs=3e-15)
        # The unmodified q_k > 0 for every omitted mode makes each full source infinite rank.
        assert source.eigenvalues(256)[-1] > 0
    assert abs(np.trace(control["perturbation"])) < 1e-15
    assert np.max(np.abs(np.linalg.eigvalsh(control["whitened_perturbation"]))) == pytest.approx(1)
    assert control["construction"]["constraint_rank"] == 20
    assert control["construction"]["source_coordinate_dimension"] == 136
    assert control["construction"]["target_null_relative_residual"] > 0.2
    assert control["infinite_dimensional_identification_claim"] is False
    with pytest.raises(ValueError):
        control["source_plus"][0, 0] = 0


def test_all_observed_covariances_agree_while_optimal_hedges_and_losses_differ(control):
    mask = control["observed_mask"]
    assert np.count_nonzero(np.triu(mask)) == 19
    minus, plus = control["kernels"]
    np.testing.assert_allclose(minus[mask], plus[mask], atol=3e-15, rtol=0)
    assert np.linalg.norm((plus - minus)[~mask]) > 1e-3
    target, hedge = control["target"], control["hedge"]
    decisions = control["paired_decisions"]
    positions = decisions["optimal_hedges"]
    np.testing.assert_allclose(positions, [1.67708044, 5.27662739], atol=2e-6)
    assert decisions["hedge_gap"] > 3.5
    for index, kernel in enumerate((minus, plus)):
        position = positions[index]
        residual = target - position * hedge
        # First-order orthogonality checks the actual hedge decision.
        assert abs(hedge @ kernel @ residual) < 1e-14
        assert residual @ kernel @ residual == pytest.approx(
            decisions["optimal_risks"][index],
            abs=2e-15,
        )
    expected_excess = decisions["hedge_variances"][0] * np.diff(positions)[0] ** 2
    np.testing.assert_allclose(decisions["wrong_hedge_excess"], expected_excess, atol=2e-14)
    assert expected_excess > 0.14 * decisions["target_variances"][0]
    assert decisions["minimax_regret"] == pytest.approx(expected_excess / 4, abs=2e-14)
    assert decisions["minimax_regret_hedge"] == pytest.approx(np.mean(positions))
    # Positive cross covariances give distinct variance and regret decisions.
    assert decisions["minimax_variance_hedge"] == pytest.approx(np.min(positions))
    assert decisions["minimax_regret_hedge"] > decisions["minimax_variance_hedge"] + 1


def test_positive_factors_and_independent_block_check_match_the_witness(control):
    offset = control["offset_covariance"]
    assert np.linalg.eigvalsh(offset)[0] >= -1e-16
    for source, kernel in zip(
        (control["source_minus"], control["source_plus"]),
        control["kernels"],
        strict=True,
    ):
        image = offset + sum(factor @ source @ factor.T for factor in control["factors"])
        np.testing.assert_allclose(image, kernel, atol=2e-15, rtol=0)
    checks = control["integration_diagnostics"]
    assert checks["refined_quadrature_order"] == 48
    assert checks["map_frobenius_refinement_discrepancy"] < 1e-13
    assert np.max(checks["fixed_source_kernel_block_discrepancies"]) < 1e-13
    assert np.max(checks["fixed_source_kernel_refinement_discrepancies"]) < 1e-13
    assert checks["refined_observed_image_difference"] < 1e-13
    assert checks["numerical_error_certificate"] is False


def test_reference_refinement_preserves_the_sources_and_is_covered_by_fixed_tail_bound(control):
    coarse = fixed_power_hedge_control(reference_dimension=32)
    for name in ("finite_source_base", "perturbation", "source_minus", "source_plus"):
        np.testing.assert_array_equal(coarse[name], control[name])
    difference = control["offset_covariance"] - coarse["offset_covariance"]
    bound = coarse["tail_diagnostics"]["kernel_psd_upper_bound"]
    assert np.linalg.eigvalsh(difference)[0] > -1e-15
    assert np.linalg.eigvalsh(bound - difference)[0] > -1e-15
    assert (
        control["tail_diagnostics"]["omitted_trace_beyond_reference"]
        < (coarse["tail_diagnostics"]["omitted_trace_beyond_reference"])
    )
    hedge_change = np.abs(
        control["paired_decisions"]["optimal_hedges"] - coarse["paired_decisions"]["optimal_hedges"]
    )
    assert np.all(hedge_change <= coarse["tail_diagnostics"]["optimal_hedge_absolute_error_bounds"])
    assert control["tail_diagnostics"]["uniform_source_class_certificate"] is False


def test_one_extra_maturity_contracts_psd_interval_for_the_same_population(control):
    covariance, target, hedge = control["baseline_covariance"], control["target"], control["hedge"]
    original = free_psd_clique_hedge_interval(covariance, target, hedge, control["cliques"])
    extra = free_psd_clique_hedge_interval(
        covariance,
        target,
        hedge,
        ((0, 1, 2, 3, 6), (3, 4, 5, 6)),
    )
    np.testing.assert_allclose(original["hedge_interval"], [-8.620115, 9.314290], atol=3e-5)
    np.testing.assert_allclose(extra["hedge_interval"], [2.318524, 6.856235], atol=3e-5)
    assert original["hedge_interval"][0] < extra["hedge_interval"][0]
    assert extra["hedge_interval"][1] < original["hedge_interval"][1]
    assert extra["minimax_regret"] < original["minimax_regret"] / 10
    # Revealing target maturity in the second clique identifies this point hedge functional.
    point = np.eye(7)[0]
    identified = free_psd_clique_hedge_interval(
        covariance,
        point,
        hedge,
        ((0, 1, 2, 3), (0, 3, 4, 5, 6)),
    )
    assert np.ptp(identified["hedge_interval"]) < 1e-7
    assert identified["hedge_interval"][0] == pytest.approx(
        point @ covariance @ hedge / (hedge @ covariance @ hedge), abs=1e-7
    )


def test_identified_functional_cannot_be_normalized_into_a_spurious_witness():
    with pytest.raises(ValueError, match="identified"):
        fixed_power_hedge_control(
            exposure="point",
            cliques=((0, 1, 2, 3), (0, 3, 4, 5, 6)),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"safety": 1},
        {"safety": 0},
        {"rank_rtol": 1},
        {"reference_dimension": 8},
        {"source_dimension": 4},
        {"grid": [0, 1]},
        {"grid": [0, 0, 1, 2, 3, 4, 5]},
        {"exposure": "invalid"},
        {"cliques": ((0, 1, 2, 3), (3, 4, 5, 5, 6))},
    ],
)
def test_control_rejects_invalid_declared_models(arguments):
    with pytest.raises(ValueError):
        fixed_power_hedge_control(**arguments)


def test_analytic_free_psd_interval_handles_a_disjoint_overlap_and_known_offset():
    covariance = np.diag([2.0, 3.0])
    offset = np.eye(2)
    target, hedge = np.eye(2)
    result = free_psd_clique_hedge_interval(
        covariance,
        target,
        hedge,
        ((0,), (1,)),
        offset_covariance=offset,
    )
    np.testing.assert_allclose(result["cross_covariance_interval"], [-np.sqrt(2), np.sqrt(2)])
    np.testing.assert_allclose(result["hedge_interval"], [-np.sqrt(2) / 3, np.sqrt(2) / 3])
    assert result["minimax_variance_hedge"] == 0
