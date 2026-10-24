"""Separate forcing support from retained transport coordinates."""

import numpy as np
import pytest

from spde_covariance.admissibility import covariance_integration_matrix
from spde_covariance.geometry import lifted_matrix, smat, svec
from spde_covariance.laguerre import LaguerreTransport, PowerLawSource
from spde_covariance.source_classes import (
    _rectangular_lift,
    embed_source_covariance,
    evaluate_source_witness,
    fixed_source_tail_allowance,
    integration_map_comparison_allowance,
    projected_covariance_integration_matrix,
    projected_integration_diagnostics,
)


def test_fast_rectangular_lift_preserves_the_svec_frobenius_metric():
    rng = np.random.default_rng(64)
    operator = rng.normal(size=(3, 7))
    np.testing.assert_allclose(_rectangular_lift(operator), lifted_matrix(operator), atol=2e-15)
    source = rng.normal(size=(7, 7))
    source = (source + source.T) / 2
    np.testing.assert_allclose(
        smat(_rectangular_lift(operator) @ svec(source)), operator @ source @ operator.T, atol=1e-14
    )


@pytest.mark.parametrize(
    "source_dimension,output_dimension,delta", [(2, 2, 1.0), (8, 3, 0.2), (16, 8, 1 / 52)]
)
def test_direct_positive_map_agrees_with_independent_existing_covariance_integration(
    source_dimension,
    output_dimension,
    delta,
):
    rng = np.random.default_rng(source_dimension)
    factor = rng.normal(size=(source_dimension, 3))
    source = factor @ factor.T / source_dimension
    matrix = projected_covariance_integration_matrix(source_dimension, output_dimension, delta)
    expected = LaguerreTransport(source_dimension).innovation_covariance(
        source, delta, method="block"
    )
    np.testing.assert_allclose(
        smat(matrix @ svec(source)), expected[:output_dimension, :output_dimension], atol=2e-14
    )
    assert matrix.shape == (
        output_dimension * (output_dimension + 1) // 2,
        source_dimension * (source_dimension + 1) // 2,
    )
    assert not matrix.flags.writeable
    if source_dimension == output_dimension:
        old = covariance_integration_matrix(LaguerreTransport(source_dimension), delta)
        np.testing.assert_allclose(matrix, old, atol=5e-16)


def test_forcing_outside_output_prefix_feeds_retained_coordinates():
    delta = 0.7
    source = np.diag([0.0, 1.0])
    projected = projected_covariance_integration_matrix(2, 1, delta)
    actual = smat(projected @ svec(source))
    average = -np.expm1(-delta) / delta
    squared_average = -np.expm1(-2 * delta) / (2 * delta)
    assert actual[0, 0] == pytest.approx(1 - 2 * average + squared_average, abs=2e-16)
    assert actual[0, 0] > 0.09
    matched = projected_covariance_integration_matrix(1, 1, delta)
    np.testing.assert_array_equal(matched @ svec(source[:1, :1]), 0)


def test_witness_embedding_preserves_trace_psd_and_projected_image():
    rng = np.random.default_rng(19)
    factor = rng.normal(size=(4, 2))
    source = factor @ factor.T
    embedded = embed_source_covariance(source, 9)
    assert not embedded.flags.writeable
    np.testing.assert_allclose(np.trace(embedded), np.trace(source), atol=0)
    assert np.linalg.eigvalsh(embedded)[0] >= -1e-14
    small = projected_covariance_integration_matrix(4, 3, 0.8)
    large = projected_covariance_integration_matrix(9, 3, 0.8)
    np.testing.assert_allclose(large @ svec(embedded), small @ svec(source), atol=5e-15)
    np.testing.assert_array_equal(source, embedded[:4, :4])


def test_witness_prediction_is_immutable_and_does_not_accept_an_infeasible_upper_bound():
    matrix = projected_covariance_integration_matrix(3, 2, 0.3)
    source = np.diag([0.2, 0.3, 0.1])
    evaluation = LaguerreTransport(2).evaluate([0.0, 0.5, 1.0])
    prediction = evaluate_source_witness(
        source, matrix, evaluation, target_kernel=np.eye(3), trace_cap=1
    )
    assert prediction["source_feasibility"]["feasible"]
    assert prediction["weighted_distance"] == pytest.approx(
        np.linalg.norm(prediction["projected_innovation_kernel"] - np.eye(3), "fro")
    )
    with pytest.raises(ValueError):
        prediction["projected_innovation"][0, 0] = 0
    with pytest.raises(ValueError, match="trace_cap"):
        evaluate_source_witness(source, matrix, evaluation, trace_cap=0.5)
    # This is within the general covariance validator's loose input tolerance,
    # but cannot serve as a feasible optimization upper-bound witness.
    with pytest.raises(ValueError, match="not PSD"):
        evaluate_source_witness(np.diag([1.0, 0.0, -1e-11]), matrix, evaluation, trace_cap=2)


def test_64_mode_map_refinement_and_fixed_low_rank_block_crosscheck():
    rng = np.random.default_rng(1264)
    factor = rng.normal(size=(64, 3))
    source = factor @ factor.T
    source /= np.trace(source)
    evaluation = LaguerreTransport(16).evaluate([0.0, 0.2, 1.0, 3.0])
    checks = projected_integration_diagnostics(
        64,
        16,
        1 / 52,
        source_covariance=source,
        evaluation_matrix=evaluation,
        weights=[0.1, 0.3, 0.4, 0.2],
    )
    assert checks["map_frobenius_refinement_discrepancy"] < 1e-13
    assert checks["output_hs_block_discrepancy"] < 2e-15
    assert checks["weighted_kernel_block_discrepancy"] < 2e-15
    assert checks["minimum_output_eigenvalue"] >= -2e-15
    assert checks["numerical_error_certificate"] is False
    assert checks["uniform_source_class_certificate"] is False
    with pytest.raises(ValueError):
        checks["quadrature_output_covariance"][0, 0] = 0


def test_two_numerical_maps_have_an_explicit_uniform_trace_capped_comparison():
    coarse = projected_covariance_integration_matrix(8, 3, 1, quadrature_order=2)
    refined = projected_covariance_integration_matrix(8, 3, 1, quadrature_order=24)
    evaluation = LaguerreTransport(3).evaluate([0.0, 0.5, 1.0, 2.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    comparison = integration_map_comparison_allowance(
        coarse,
        refined,
        trace_cap=2,
        evaluation_matrix=evaluation,
        weights=weights,
    )
    assert comparison["weighted_kernel_allowance"] > 1e-4
    rng = np.random.default_rng(336)
    for _ in range(10):
        factor = rng.normal(size=(8, 2))
        source = factor @ factor.T
        source *= 2 / np.trace(source)
        output = smat((coarse - refined) @ svec(source))
        weighted = np.sqrt(weights)[:, None] * evaluation
        error = np.linalg.norm(weighted @ output @ weighted.T, "fro")
        assert error <= comparison["weighted_kernel_allowance"] + 1e-14
    assert comparison["exact_integration_error_certificate"] is False


@pytest.mark.parametrize("angle", [0.0, np.pi / 6])
def test_fixed_source_tail_allowance_bounds_actual_omission_without_an_infinite_class_claim(angle):
    source, cutoff, reference, output = PowerLawSource(exponent=3, trace=0.7), 8, 32, 4
    covariance = source.finite_matrix(reference)
    if angle:
        block = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotation = np.kron(np.eye(reference // 2), block)
        covariance = rotation @ covariance @ rotation.T
    fine = LaguerreTransport(reference).innovation_covariance(covariance, 1 / 52)[:output, :output]
    coarse = LaguerreTransport(cutoff).innovation_covariance(covariance[:cutoff, :cutoff], 1 / 52)[
        :output, :output
    ]
    evaluation = LaguerreTransport(output).evaluate([0.0, 0.25, 0.5, 1.0, 3.0])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    bound = fixed_source_tail_allowance(
        source,
        cutoff,
        output,
        1 / 52,
        evaluation_matrix=evaluation,
        weights=weights,
        rotation_angle=angle,
    )
    omission = fine - coarse
    assert np.linalg.eigvalsh(omission)[0] >= -1e-15
    assert np.linalg.eigvalsh(bound["output_covariance_upper_matrix"] - omission)[0] >= -1e-15
    weighted = np.sqrt(weights)[:, None] * evaluation
    assert (
        np.linalg.norm(weighted @ omission @ weighted.T, "fro")
        <= bound["weighted_kernel_tail_bound"]
    )
    assert bound["full_innovation_trace_bound"] == source.innovation_tail_bound(cutoff, 1 / 52)
    assert bound["infinite_dimensional_exclusion"] is False
    assert bound["uniform_source_class_certificate"] is False


@pytest.mark.parametrize(
    "arguments", [(0, 1, 1.0), (2, 3, 1.0), (True, 1, 1.0), (2, 1, 0), (2, 1, np.inf)]
)
def test_invalid_projected_map_dimensions_and_intervals_are_rejected(arguments):
    with pytest.raises(ValueError):
        projected_covariance_integration_matrix(*arguments)


def test_invalid_witness_embedding_and_tail_declarations_are_rejected():
    with pytest.raises(ValueError):
        embed_source_covariance(np.eye(3), 2)
    with pytest.raises(ValueError):
        embed_source_covariance(np.diag([1.0, -1.0]), 3)
    with pytest.raises(ValueError, match="even"):
        fixed_source_tail_allowance(PowerLawSource(), 7, 3, 1 / 52, rotation_angle=0.1)
    with pytest.raises(TypeError):
        fixed_source_tail_allowance(np.eye(4), 4, 2, 1 / 52)
    with pytest.raises(ValueError):
        projected_integration_diagnostics(4, 2, 1 / 52, source_covariance=np.eye(3))
    with pytest.raises(ValueError):
        integration_map_comparison_allowance(np.eye(3), np.eye(4), trace_cap=1)


def _study_config(**updates):
    from types import SimpleNamespace

    settings = dict(
        source_dimensions=(2,),
        source_multipliers=(1, 2),
        reference_dimension=16,
        delta=1 / 52,
        source_exponent=3.0,
        source_trace=0.7,
        rotation_angle=np.pi / 6,
        quadrature_order=24,
        support_rtol=1e-12,
        trace_cap=2.0,
        source_max_iterations=25,
        source_gap=1e-8,
        models=("powerlaw", "rotated"),
        covers=(3, 9),
    )
    return SimpleNamespace(**(settings | updates))


def test_population_study_keeps_fixed_projected_targets_and_nested_fitted_witnesses(monkeypatch):
    from spde_covariance import completion_source_study as study

    config = _study_config()
    original = study.diagnose_constant_source_admissibility
    initializations = []

    def audited(kernel, evaluation, **options):
        if options["delta"] == config.delta:
            initializations.append(options["initial_source"].copy())
        return original(kernel, evaluation, **options)

    monkeypatch.setattr(study, "diagnose_constant_source_admissibility", audited)
    rows, controls = study.run_source_class_study(config)
    assert len(rows) == 16 and controls["passed"]
    assert all(row["valid"] for row in controls["rows"])
    assert all(not np.any(initial) for initial in initializations if len(initial) == 2)
    assert all(
        not np.any(initial[2:, :]) and not np.any(initial[:, 2:])
        for initial in initializations
        if len(initial) == 4
    )
    for name in config.models:
        covariance = PowerLawSource(3.0, 0.7).finite_matrix(config.reference_dimension)
        if name == "rotated":
            angle = config.rotation_angle
            rotation = np.kron(
                np.eye(config.reference_dimension // 2),
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            )
            covariance = rotation @ covariance @ rotation.T
        expected = LaguerreTransport(config.reference_dimension).innovation_covariance(
            covariance,
            config.delta,
            method="block",
        )[:2, :2]
        for cover in config.covers:
            for target in ("generating", "canonical"):
                group = [
                    r
                    for r in rows
                    if r["model"] == name and r["cover"] == cover and r["target"] == target
                ]
                assert [r["M"] for r in group] == [2, 4]
                assert group[1]["nested_witness_upper"] <= group[0]["nested_witness_upper"] + 1e-12
                for row in group:
                    assert row["available"] and row["source_feasibility"]["feasible"]
                    assert row["source_multiplier"] == row["M"] // row["J"]
                    evaluation = LaguerreTransport(row["J"]).evaluate(row["nodes"])
                    if target == "generating":
                        np.testing.assert_allclose(
                            row["target_kernel"], evaluation @ expected @ evaluation.T, atol=2e-15
                        )
                    root = np.sqrt(row["weights"])
                    residual = (
                        root[:, None]
                        * (row["projected_innovation_kernel"] - row["target_kernel"])
                        * root[None, :]
                    )
                    assert np.linalg.norm(residual) == pytest.approx(
                        row["distance_upper"], abs=2e-15
                    )
                    witness = evaluate_source_witness(
                        row["nested_witness_source"],
                        projected_covariance_integration_matrix(row["M"], row["J"], config.delta),
                        evaluation,
                        target_kernel=row["target_kernel"],
                        weights=row["weights"],
                        trace_cap=config.trace_cap,
                    )
                    assert witness["weighted_distance"] == pytest.approx(
                        row["nested_witness_upper"]
                    )


def test_skipped_population_still_checks_the_cheap_projected_forcing_controls(monkeypatch):
    from spde_covariance import completion_source_study as study

    def forbidden(*args, **kwargs):
        raise AssertionError("large reference preparation must be skipped")

    monkeypatch.setattr(study, "prepare_parent_model", forbidden)
    rows, controls = study.run_source_class_study(_study_config(source_dimensions=()))
    assert rows == [] and controls["passed"]
    assert [r["M"] for r in controls["rows"]] == [2, 3, 4]


def test_unavailable_population_completion_remains_explicit_in_source_study(monkeypatch):
    from types import SimpleNamespace

    from spde_covariance import completion_source_study as study

    monkeypatch.setattr(
        study,
        "complete_kernel",
        lambda *a, **kw: SimpleNamespace(
            kernel=None,
            diagnostics={"available": False, "reason": "controlled unavailable"},
        ),
    )
    rows, controls = study.run_source_class_study(_study_config(models=("powerlaw",), covers=(3,)))
    assert controls["passed"] and len(rows) == 4
    unavailable = [r for r in rows if not r["available"]]
    assert len(unavailable) == 2
    assert all(
        r["target"] == "canonical"
        and r["distance_upper"] is None
        and r["unavailable_reason"] == "controlled unavailable"
        for r in unavailable
    )
