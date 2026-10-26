"""True point aliases require a separately constructed source perturbation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from spde_covariance.hedging_true_points import true_point_hedge_control
from spde_covariance.laguerre import LaguerreTransport, PowerLawSource


@pytest.fixture(scope="module")
def control():
    return true_point_hedge_control()


def test_true_point_pair_is_psd_equal_trace_and_observationally_equivalent(control):
    assert control["observation_model"] == "true point evaluation at grid+r; no output projection"
    pair = control["true_point_pair"]
    assert min(pair["minimum_finite_source_eigenvalues"]) > 8e-5
    np.testing.assert_allclose(pair["full_source_traces"], 1, atol=3e-15, rtol=0)
    background = PowerLawSource(exponent=3, trace=1)
    for name in ("source_minus", "source_plus"):
        finite = np.asarray(pair[name])
        assert np.trace(finite) + background.tail_trace(16) == pytest.approx(1, abs=3e-15)
    np.testing.assert_array_equal(control["finite_source_base"], background.finite_matrix(16))
    assert control["construction"]["constraint_rank"] == 20
    assert control["construction"]["source_coordinate_dimension"] == 136
    assert control["construction"]["target_null_relative_residual"] > 0.3
    assert abs(control["construction"]["perturbation_trace"]) < 1e-15
    mask = np.asarray(control["observed_mask"], dtype=bool)
    assert np.count_nonzero(np.triu(mask)) == 19
    minus, plus = np.asarray(pair["kernels"])
    np.testing.assert_allclose(minus[mask], plus[mask], atol=3e-15, rtol=0)
    assert pair["observed_svec_difference"] < 3e-15
    assert np.linalg.norm((plus - minus)[~mask]) > 0.005


def test_true_point_hedge_gap_is_material_and_wrong_hedge_excess_is_recomputed(control):
    pair = control["true_point_pair"]
    np.testing.assert_allclose(pair["optimal_hedges"], [0.8167802083, 6.0791741523], atol=3e-6)
    assert pair["hedge_gap"] > 5
    assert min(pair["wrong_hedge_excess_fraction_unhedged"]) > 0.31
    target, hedge = np.asarray(control["target"]), np.asarray(control["hedge"])
    for world, kernel in enumerate(np.asarray(pair["kernels"])):
        own, other = pair["optimal_hedges"][world], pair["optimal_hedges"][1 - world]
        own_residual, wrong_residual = target - own * hedge, target - other * hedge
        assert abs(hedge @ kernel @ own_residual) < 1e-14
        excess = wrong_residual @ kernel @ wrong_residual - own_residual @ kernel @ own_residual
        assert excess == pytest.approx(pair["wrong_hedge_excess"][world], abs=1e-14)
    assert control["paired_decisions"]["minimax_regret"] == pytest.approx(
        pair["wrong_hedge_excess"][0] / 4,
        abs=3e-14,
    )


def test_original_j8_witness_does_not_remain_an_alias_under_true_point_observations(control):
    original = control["original_projected_witness"]
    assert original["projected_observed_difference"] < 1e-14
    assert not original["is_true_point_alias_at_1e_minus_10"]
    reinterpreted = original["evaluated_as_true_points"]
    assert reinterpreted["observed_svec_difference"] > 2e-4
    assert reinterpreted["observed_entry_max_difference"] > 9e-5
    np.testing.assert_allclose(reinterpreted["optimal_hedges"], [1.67187307, 5.22125403], atol=3e-6)
    assert np.linalg.norm(np.asarray(control["perturbation"]) - original["perturbation"]) > 1e-3
    assert control["main_projected_benchmark_reinterpreted"] is False
    assert control["monte_carlo_performance_claim"] is False
    assert control["uniform_infinite_source_identification_claim"] is False


def test_true_point_factors_agree_with_full_finite_transport_and_independent_integration(control):
    grid = np.asarray(control["grid"])
    model = LaguerreTransport(16)
    evaluation = model.evaluate(grid)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    for factor, node, weight in zip(control["factors"], nodes, weights, strict=True):
        expected = np.sqrt(weight / 2) * evaluation @ model.matrix((node + 1) / 104)
        np.testing.assert_allclose(factor, expected, atol=1e-15, rtol=0)
    checks = control["integration_diagnostics"]
    assert checks["refined_quadrature_order"] == 48
    assert checks["map_frobenius_refinement_discrepancy"] < 1e-13
    assert checks["fixed_tail_kernel_refinement_discrepancy"] < 1e-15
    assert checks["same_witness_refined_observed_difference"] < 1e-14
    assert max(checks["independent_block_finite_kernel_discrepancies"]) < 1e-13
    assert max(checks["positive_factor_kernel_discrepancies"]) < 1e-13
    assert checks["numerical_error_certificate"] is False


def test_fixed_true_point_tail_bound_covers_reference_refinement_without_changing_sources(control):
    coarse = true_point_hedge_control(reference_dimension=128)
    np.testing.assert_array_equal(coarse["perturbation"], control["perturbation"])
    np.testing.assert_array_equal(
        coarse["true_point_pair"]["source_plus"], control["true_point_pair"]["source_plus"]
    )
    tail_difference = np.asarray(control["offset_covariance"]) - coarse["offset_covariance"]
    tail_bound = np.asarray(coarse["tail_diagnostics"]["kernel_psd_upper_bound"])
    assert np.linalg.eigvalsh(tail_difference)[0] >= -1e-15
    assert np.linalg.eigvalsh(tail_bound - tail_difference)[0] >= -1e-15
    actual = np.abs(
        np.asarray(control["true_point_pair"]["optimal_hedges"])
        - coarse["true_point_pair"]["optimal_hedges"]
    )
    assert np.all(actual <= coarse["tail_diagnostics"]["optimal_hedge_absolute_error_bounds"])
    tails = control["tail_diagnostics"]
    assert max(tails["optimal_hedge_absolute_error_bounds"]) < 9e-6
    # This continuous-time allowance is obtained from q_R I_H, not by treating
    # agreement of the two Gauss rules as an integration error certificate.
    q = PowerLawSource().eigenvalues(513)[-1]
    gamma = -np.expm1(-1 / 52) / (1 / 52)
    assert tails["target_variance_upper_bound"] == pytest.approx(q * gamma * (1 - np.exp(-0.5)))
    assert tails["hedge_variance_upper_bound"] == pytest.approx(
        q * gamma * (np.exp(-2.5) - np.exp(-3))
    )
    assert tails["uniform_source_class_certificate"] is False


def test_result_is_json_safe_and_example_gate_has_no_io(control):
    roundtrip = json.loads(json.dumps(control, allow_nan=False))
    assert roundtrip == control
    path = Path(__file__).resolve().parents[1] / "examples" / "true_point_identification.py"
    spec = importlib.util.spec_from_file_location("true_point_annex_example", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolves its declaring module when inspecting postponed annotations.
    import sys

    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        acceptance = module._acceptance(control, 1e-10)
        assert acceptance["passed"]
        report = module._report(control, acceptance)
        assert "not** an observational alias" in report
        assert "remains a projected-observation study" in report
        assert "[results](results.json)" in report
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize(
    "arguments",
    [
        {"source_dimension": 7},
        {"reference_dimension": 8},
        {"quadrature_order": 0},
        {"safety": 1},
        {"safety": 0},
        {"rank_rtol": 1},
        {"delta": 0},
        {"grid": [0, 1]},
        {"grid": [0, 0, 1, 2, 3, 4, 5]},
    ],
)
def test_true_point_control_rejects_invalid_declared_models(arguments):
    with pytest.raises(ValueError):
        true_point_hedge_control(**arguments)
