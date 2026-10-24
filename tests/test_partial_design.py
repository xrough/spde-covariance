"""Independent functional geometry, common-data and source-approximation controls."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from spde_covariance.infinite_benchmark import fixed_exposures as old_exposures
from spde_covariance.laguerre import LaguerreTransport, PowerLawSource
from spde_covariance.partial_design import (
    DESIGN_NAMES,
    Contract,
    contract_pool,
    design_groups,
    design_indices,
    fixed_exposures,
    fixed_grid_witness,
    functional_coefficients,
    functional_gram,
    observation_omission,
    prepare_parent_model,
    simulate_parent_pool,
    span_diagnostics,
)


def test_point_gram_is_full_sobolev_reproducing_kernel():
    points = np.array([0.0, 0.2, 0.75, 2.25, 6.0])
    contracts = [Contract("point", point) for point in points]
    expected = 2 - np.exp(-np.minimum.outer(points, points))
    np.testing.assert_allclose(functional_gram(contracts), expected, rtol=2e-15, atol=2e-15)


def test_delivery_gram_matches_independent_kernel_integration():
    point = 0.8
    first, second = (0.5, 1.0), (1.5, 2.0)
    contracts = [
        Contract("point", point),
        Contract("delivery", *first),
        Contract("delivery", *second),
    ]
    gram = functional_gram(contracts)
    point_delivery = quad(
        lambda x: 2 - np.exp(-min(point, x)), *first, points=[point], epsabs=1e-13
    )[0] / (first[1] - first[0])
    delivery_delivery = quad(
        lambda x: quad(lambda y: 2 - np.exp(-min(x, y)), *second, epsabs=1e-13)[0],
        *first,
        epsabs=1e-13,
    )[0] / ((first[1] - first[0]) * (second[1] - second[0]))
    np.testing.assert_allclose(gram[0, 1], point_delivery, rtol=1e-14)
    np.testing.assert_allclose(gram[1, 2], delivery_delivery, rtol=1e-14)
    assert np.linalg.eigvalsh(gram)[0] > 0


def test_unit_exposures_match_existing_normalization_and_stay_fixed_across_prefixes():
    exposure = fixed_exposures()
    np.testing.assert_allclose(np.diag(functional_gram(list(exposure.values()))), 1.0, atol=1e-14)
    for name, value in old_exposures(LaguerreTransport(32)).items():
        np.testing.assert_allclose(
            functional_coefficients([exposure[name]], LaguerreTransport(32))[0], value, atol=4e-15
        )
    low = functional_coefficients(list(exposure.values()), LaguerreTransport(8))
    high = functional_coefficients(list(exposure.values()), LaguerreTransport(128))
    np.testing.assert_array_equal(low, high[:, :8])
    assert all(value.anchor == 0 for value in exposure.values())


def test_delivery_coefficients_match_direct_integration_at_high_mode_and_thin_interval():
    model = LaguerreTransport(128)
    for interval in [(0.5, 1.0), (1.5, 2.0), (0.75, 0.750000001)]:
        actual = functional_coefficients([Contract("delivery", *interval)], model)[0]
        nodes, weights = np.polynomial.legendre.leggauss(100)
        points = interval[0] + (interval[1] - interval[0]) * (nodes + 1) / 2
        expected = weights @ model.evaluate(points) / 2
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)


def test_pool_nesting_and_rotating_groups_are_predeclared():
    pool = contract_pool()
    assert len(pool) == len(set(pool)) == 16
    assert len(design_groups("rotating4")) == 16
    assert all(len(group) == 4 for group in design_groups("rotating4"))
    assert design_indices("rotating4", 15) == (15, 0, 1, 2)
    assert design_indices("rotating4", 16) == (0, 1, 2, 3)
    assert set(design_indices("nested8", 0)) < set(design_indices("nested16", 0))
    assert not any(value.kind == "point" and value.start in (0.75, 2.25) for value in pool)


def test_full_h_proxy_spans_distinguish_per_date_availability_and_union():
    pool, exposures = contract_pool(), fixed_exposures()
    expected = {
        "fixed_points": (True, False),
        "fixed_deliveries": (False, True),
        "nested8": (True, True),
        "nested16": (True, True),
    }
    for design, availability in expected.items():
        contracts = [pool[index] for index in design_indices(design, 0)]
        for name, available in zip(("point_spread", "delivery_spread"), availability, strict=True):
            result = span_diagnostics(exposures[name], contracts)
            assert result["available"] == available
            if available:
                assert result["relative_residual"] < 1e-10
    assert span_diagnostics(exposures["point_spread"], pool)["available"]
    offgrid = span_diagnostics(exposures["offgrid_spread"], pool)
    assert not offgrid["available"]
    assert offgrid["relative_residual"] > 0.2
    rotating = [
        span_diagnostics(exposures["point_spread"], [pool[index] for index in group])
        for group in design_groups("rotating4")
    ]
    assert 0 < sum(row["available"] for row in rotating) < len(rotating)


def test_finite_sieve_identification_does_not_imply_full_h_proxy_availability():
    model = LaguerreTransport(2)
    contracts = contract_pool()[:2]
    operator = functional_coefficients(contracts, model)
    assert np.linalg.matrix_rank(operator) == 2
    target = fixed_exposures()["offgrid_spread"]
    # Every target vector in this two-dimensional sieve is in the row span.
    finite = functional_coefficients([target], model)[0]
    np.testing.assert_allclose(np.linalg.solve(operator.T, finite) @ operator, finite)
    assert not span_diagnostics(target, contracts)["available"]


def test_fixed_grid_witness_is_invisible_but_not_an_spde_admissibility_claim():
    model = LaguerreTransport(256)
    pool = contract_pool()
    target = fixed_exposures()["offgrid_spread"]
    witness = fixed_grid_witness(pool, model, target)
    operator = functional_coefficients(pool, model)
    np.testing.assert_allclose(np.linalg.norm(witness["vector"]), 1.0, atol=1e-14)
    assert witness["observation_residual"] < 1e-13
    assert np.linalg.norm(operator @ witness["rank_one_covariance"] @ operator.T) < 1e-13
    assert witness["exposure_risk_difference"] > 0.001
    assert "unrestricted" in witness["scope"]
    assert "no SPDE" in witness["scope"]


def test_observation_omission_includes_signed_cross_terms():
    covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
    operator = np.array([[1.0, -1.0]])
    result = observation_omission(covariance, operator, 1)
    np.testing.assert_allclose(
        result["omission"],
        result["tail_tail"] + result["cross"] + result["cross_transpose"],
        atol=1e-15,
    )
    np.testing.assert_allclose(result["omission"], [[-0.6]], atol=1e-15)
    assert result["tail_tail_hs"] == 1.0
    assert result["cross_sum_hs"] > result["omission_hs"]


def test_adjacent_rotation_has_fixed_spectrum_nested_prefixes_and_source_id():
    small = prepare_parent_model(
        "rotated", 8, source_exponent=2.7, source_trace=0.4, rotation_angle=0.2
    )
    large = prepare_parent_model(
        "rotated", 16, source_exponent=2.7, source_trace=0.4, rotation_angle=0.2
    )
    spectrum = PowerLawSource(2.7, 0.4)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(small.source_covariance)[::-1], spectrum.eigenvalues(8)
    )
    np.testing.assert_array_equal(small.source_covariance, large.source_covariance[:8, :8])
    assert small.source_metadata["source_family_id"] == large.source_metadata["source_family_id"]
    np.testing.assert_allclose(
        np.trace(small.source_covariance) + small.tail_bounds["full_source_trace_tail"], 0.4
    )
    angle_zero = prepare_parent_model("rotated", 8, rotation_angle=0)
    diagonal = prepare_parent_model("powerlaw", 8)
    np.testing.assert_array_equal(angle_zero.source_covariance, diagonal.source_covariance)
    assert small.numerical_checks["absolute_hs_discrepancy"] < 1e-13


@pytest.mark.parametrize("source", ["powerlaw", "rotated"])
def test_sharp_full_functional_tail_bounds_dominate_larger_prefix(source):
    small, large = prepare_parent_model(source, 16), prepare_parent_model(source, 64)
    lower = small.operator @ small.innovation_covariance @ small.operator.T
    upper = large.operator @ large.innovation_covariance @ large.operator.T
    remainder = small.tail_bounds["observed_matrix"] - (upper - lower)
    assert np.linalg.eigvalsh(remainder)[0] >= -1e-13
    for name in small.exposures:
        a, b = small.exposure_coefficients[name], large.exposure_coefficients[name]
        difference = b @ large.innovation_covariance @ b - a @ small.innovation_covariance @ a
        assert -1e-13 <= difference <= small.tail_bounds["risk_bounds"][name] + 1e-13


def test_parent_measurements_and_noise_are_identical_across_sieve_views():
    prepared = prepare_parent_model(dimension=16)
    parent = simulate_parent_pool(prepared, 20, seed=1729, noise_std=np.sqrt(0.001))
    repeat = simulate_parent_pool(prepared, 20, seed=1729, noise_std=np.sqrt(0.001))
    np.testing.assert_array_equal(parent.y, repeat.y)
    for design in DESIGN_NAMES:
        low, high = parent.observations(design, 4), parent.observations(design, 16)
        for index, (first, second) in enumerate(zip(low, high, strict=True)):
            np.testing.assert_array_equal(first.y, second.y)
            np.testing.assert_array_equal(first.noise_covariance, second.noise_covariance)
            np.testing.assert_array_equal(first.operator, second.operator[:, :4])
            np.testing.assert_array_equal(
                first.y, parent.y[index, list(design_indices(design, index))]
            )
    for array in (parent.y, parent.operator, parent.noise_covariance, parent.times, low[0].y):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        parent.y[0, 0] = 1.0


def test_parent_gaussian_law_and_fresh_noise_correction():
    prepared = prepare_parent_model(dimension=8)
    n = 8000
    parent = simulate_parent_pool(prepared, n, seed=314159, noise_std=np.sqrt(0.001))
    truth = prepared.operator @ prepared.innovation_covariance @ prepared.operator.T
    total = truth + parent.noise_covariance
    empirical = parent.y.T @ parent.y / n
    expected_squared_error = (np.trace(total) ** 2 + np.sum(total**2)) / n
    assert np.linalg.norm(empirical - total) ** 2 < 8 * expected_squared_error
    corrected = empirical - parent.noise_covariance
    np.testing.assert_allclose(corrected - truth, empirical - total, atol=1e-14)
    noise_only = simulate_parent_pool(
        replace(prepared, operator=np.zeros_like(prepared.operator)),
        n,
        seed=314159,
        noise_std=np.sqrt(0.001),
    )
    np.testing.assert_allclose(np.diag(noise_only.y.T @ noise_only.y / n), 0.001, rtol=0.05)


@pytest.mark.parametrize(
    "call",
    [
        lambda: Contract("point", -1),
        lambda: Contract("point", True),
        lambda: Contract("point", 1.0, 2.0),
        lambda: Contract("delivery", 1.0, 1.0),
        lambda: Contract("other", 1.0),
        lambda: design_indices("unknown", 0),
        lambda: design_indices("rotating4", True),
        lambda: prepare_parent_model("rotated", 5),
        lambda: prepare_parent_model(rotation_angle=np.nan),
        lambda: prepare_parent_model(source_trace=0),
    ],
)
def test_invalid_geometry_or_source_inputs_are_rejected(call):
    with pytest.raises((ValueError, TypeError)):
        call()
