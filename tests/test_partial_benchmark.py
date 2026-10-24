"""Information boundaries, corrected scores and complete partial-data audits."""

import csv
import json
from dataclasses import fields, replace

import numpy as np
import pytest

from spde_covariance import partial_benchmark as benchmark
from spde_covariance.estimation import fit_covariance_reference
from spde_covariance.geometry import diagnose_design
from spde_covariance.observations import Observation
from spde_covariance.partial_benchmark import (
    FitSettings,
    choose_candidate,
    direct_risk_estimate,
    fit_candidate,
    observed_quadratic_score,
    run_partial_benchmark,
    score_risk,
)
from spde_covariance.partial_config import PartialBenchmarkConfig
from spde_covariance.partial_design import (
    prepare_parent_model,
    simulate_parent_pool,
    span_diagnostics,
)


def tiny_config(**changes):
    settings = dict(
        replications=2,
        reference_dimension=16,
        training_counts=(8, 16),
        validation_count=8,
        test_count=8,
        candidate_dimensions=(2, 4),
        geometry_dimensions=(2, 4),
        relative_regularizations=(0.01, 0.1),
        scenarios=("powerlaw",),
        designs=("fixed_points", "nested16"),
        reference_checks=False,
    )
    settings.update(changes)
    return PartialBenchmarkConfig(**settings)


@pytest.fixture
def no_figures(monkeypatch):
    # Plot validation is covered independently in test_partial_reporting.py.
    monkeypatch.setattr("spde_covariance.partial_reporting.write_report", lambda *args: None)


@pytest.fixture(scope="module")
def audited_run(tmp_path_factory):
    config = tiny_config()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("spde_covariance.partial_reporting.write_report", lambda *args: None)
        output = run_partial_benchmark(config, tmp_path_factory.mktemp("partial-audit"))
    return config, output, json.loads((output / "results.json").read_text())


def test_identity_fit_is_scaled_corrected_second_moment_and_immutable():
    samples = np.array([[0.3, 0.1], [-0.2, 0.4], [0.1, -0.1]])
    noise = np.eye(2) * 0.002
    observations = tuple(
        Observation(float(i + 1), y, np.eye(2), noise) for i, y in enumerate(samples)
    )
    settings = FitSettings(2.0, 1e-7)
    candidate = fit_candidate(
        observations,
        settings,
        dimension=2,
        relative_regularization=0.1,
        gram_operator_norm=1.0,
        basis_id="identity-test",
    )
    expected = (samples.T @ samples / len(samples) - noise) / 1.1
    assert np.linalg.eigvalsh(expected)[0] > 0
    np.testing.assert_allclose(candidate.covariance, expected, atol=1e-7, rtol=0)
    assert candidate.estimate.converged
    assert candidate.estimate.covariance_error_bound <= settings.numerical_error_budget
    assert candidate.estimate.regularization == pytest.approx(0.1)
    assert {field.name for field in fields(settings)} == {"trace_cap", "numerical_error_budget"}
    for array in (candidate.covariance, candidate.estimate.weights):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99


def test_small_partial_fit_agrees_with_independent_convex_reference():
    pytest.importorskip("cvxpy")
    parent = prepare_parent_model(dimension=16)
    data = simulate_parent_pool(parent, 16, seed=52, noise_std=0.03).observations("rotating4", 4)
    norm = np.linalg.eigvalsh(benchmark.gram_operator([item.operator for item in data]))[-1]
    own = fit_candidate(
        data,
        FitSettings(2.0, 1e-5),
        dimension=4,
        relative_regularization=0.01,
        gram_operator_norm=norm,
        basis_id="partial-reference",
    )
    reference = fit_covariance_reference(
        data,
        regularization=0.01 * norm,
        trace_cap=2.0,
        numerical_error_budget=1e-5,
        solver_tolerance=1e-11,
    )
    assert own.estimate.converged and reference.converged
    assert np.linalg.norm(own.covariance - reference.covariance) <= (
        own.estimate.covariance_error_bound + reference.covariance_error_bound
    )


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"dimension": 3}, ValueError),
        ({"relative_regularization": 0}, ValueError),
        ({"gram_operator_norm": np.nan}, ValueError),
        ({"basis_id": ""}, ValueError),
        ({"settings": tiny_config()}, TypeError),
    ],
)
def test_candidate_rejects_inconsistent_or_truth_bearing_inputs(changes, error):
    kwargs = dict(
        observations=[Observation(1.0, [0.1, 0.2], np.eye(2), np.zeros((2, 2)))],
        settings=FitSettings(2.0, 1e-4),
        dimension=2,
        relative_regularization=0.01,
        gram_operator_norm=1.0,
        basis_id="test",
    )
    kwargs.update(changes)
    with pytest.raises(error):
        fit_candidate(**kwargs)
    with pytest.raises(ValueError, match="times"):
        fit_candidate(
            kwargs["observations"] * 2,
            FitSettings(2.0, 1e-4),
            dimension=2,
            relative_regularization=0.01,
            gram_operator_norm=1.0,
            basis_id="test",
        )


def test_corrected_observed_score_has_exact_loss_difference_with_known_noise():
    truth, noise = 0.7, 0.3
    observations = [
        Observation(float(index + 1), [sign * np.sqrt(truth + noise)], [[1]], [[noise]])
        for index, sign in enumerate((-1, 1))
    ]
    a, b = np.array([[0.2]]), np.array([[0.9]])
    difference = observed_quadratic_score(a, observations) - observed_quadratic_score(
        b, observations
    )
    assert difference == pytest.approx((0.2 - truth) ** 2 - (0.9 - truth) ** 2)
    full_loss = np.mean([np.sum((item.measurement - a) ** 2) for item in observations])
    constant = np.mean([np.sum(item.measurement**2) for item in observations])
    assert observed_quadratic_score(a, observations) + constant == pytest.approx(full_loss)
    assert observed_quadratic_score(b, observations) < 0
    with pytest.raises(ValueError):
        observed_quadratic_score(np.eye(2), observations)


def test_ragged_score_grouping_preserves_each_actual_observation_map():
    matrix = np.array([[0.4, 0.1], [0.1, 0.3]])
    data = [
        Observation(1, [0.2], [[1, 0]], [[0.1]]),
        Observation(2, [0.3, -0.1], np.eye(2), np.eye(2) * 0.02),
        Observation(3, [0.4], [[1, 0]], [[0.2]]),
    ]
    expected = np.mean(
        [
            np.sum((item.operator @ matrix @ item.operator.T) ** 2)
            - 2 * np.sum(item.measurement * (item.operator @ matrix @ item.operator.T))
            for item in data
        ]
    )
    assert observed_quadratic_score(matrix, data) == pytest.approx(expected)


def test_direct_risk_uses_full_h_proxy_and_keeps_negative_noise_corrections():
    item = Observation(1, [0.0], [[3.0, 0.0]], [[0.2]])
    spans = [{"available": True, "coefficients": np.array([2.0])}]
    direct = direct_risk_estimate([item], spans)
    assert direct["estimate"] == pytest.approx(-0.8)
    assert direct["available_train_dates"] == 1
    assert score_risk(0.3, [item], spans)["risk_score"] == pytest.approx(0.3**2 + 0.48)
    unavailable = [{"available": False, "coefficients": np.array([2.0])}]
    assert direct_risk_estimate([item], unavailable)["estimate"] is None
    assert score_risk(0.3, [item], unavailable) == {"available_test_dates": 0, "risk_score": None}
    with pytest.raises(ValueError):
        direct_risk_estimate([item], [])


def test_finite_sieve_full_rank_does_not_create_a_full_h_offgrid_proxy():
    parent = prepare_parent_model(dimension=16)
    a = parent.exposure_coefficients["offgrid_spread"][:4]
    finite = diagnose_design([parent.operator[:, :4]], targets={"risk": np.outer(a, a)})
    full_h = span_diagnostics(parent.exposures["offgrid_spread"], parent.contracts)
    assert finite["rank"] == finite["dimension"]
    assert finite["targets"]["risk"]["identified"]
    assert not full_h["available"]
    assert full_h["relative_residual"] == pytest.approx(0.2758782411718531)


def test_joint_selection_reads_observed_score_only_and_has_declared_ties():
    rows = [
        dict(J=j, relative_regularization=eta, validation_score=-2.0, latent_error=object())
        for j in (8, 4)
        for eta in (0.001, 0.1)
    ]
    assert choose_candidate(rows)["J"] == 4
    assert choose_candidate(rows)["relative_regularization"] == 0.1
    rows[0]["validation_score"] = -3.0
    assert choose_candidate(rows) is rows[0]
    with pytest.raises(ValueError):
        choose_candidate([rows[0], rows[0]])
    with pytest.raises(ValueError):
        choose_candidate([dict(J=4, relative_regularization=0.1, validation_score=np.nan)])


def test_run_retains_complete_candidate_geometry_and_numerical_evidence(audited_run):
    config, output, results = audited_run
    acceptance = results["acceptance"]
    assert acceptance["passed"]
    assert acceptance["candidate_fits"] == 32
    assert acceptance["failed_fits"] == 0
    assert not acceptance["reference_resolution_passed"]  # M16 is an honest unresolved smoke.
    assert acceptance["maximum_covariance_error_bound"] <= config.numerical_error_budget
    assert len(results["candidate_summary"]) == 16
    assert len(results["geometry"]) == 8
    assert len(results["observation_approximation"]) == 8
    assert results["numerical_checks"]["powerlaw"]["absolute_hs_discrepancy"] < 1e-10
    assert results["numerical_checks"]["powerlaw"]["numerical_error_certificate"] is False
    assert json.loads((output / "geometry.json").read_text()) == results["geometry"]
    assert json.loads((output / "full_h_geometry.json").read_text()) == results["full_h_geometry"]
    offgrid = [
        row
        for row in results["full_h_geometry"]
        if row["target"] == "offgrid_spread" and row["kind"] == "same_date"
    ]
    assert offgrid and all(not row["available"] for row in offgrid)
    for row in results["selections"]:
        assert row["validation_first_interval"] == 17
        assert row["validation_last_interval"] == 24
        assert row["test_first_interval"] == 25
        assert row["test_last_interval"] == 32
        assert row["selection_valid"] and not row["refit_on_validation"]
    for row in results["risk_summary"]:
        if row["target"] == "offgrid_spread":
            assert row["available_test_dates"] == 0
            assert row["risk_score_mean"] is None
            if row["rule"] == "direct":
                assert row["estimate_mean"] is None
    with (output / "candidates.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 32
    assert all(int(row["training_last_interval"]) == int(row["N"]) for row in records)
    manifest = json.loads((output / "manifest.json").read_text())
    assert "partial linear observations" in manifest["experiment"]
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        run_partial_benchmark(config, output)


def test_full_h_rows_keep_rotating_same_date_and_union_statements_separate():
    prepared = benchmark._prepare(tiny_config(designs=("rotating4",), training_counts=(16,)))
    rows = prepared["full_h_geometry"]
    assert len([row for row in rows if row["kind"] == "same_date"]) == 16 * 3
    union = [row for row in rows if row["kind"] == "training_union_functional_span"]
    assert len(union) == 3
    assert all(len(row["contract_indices"]) == 16 for row in union)
    assert all("not pooled covariance identification" in row["scope"] for row in union)


def test_parent_simulation_is_once_per_path_and_shared_across_j(audited_run):
    config, _, _ = audited_run
    parent = prepare_parent_model(dimension=config.reference_dimension)
    pool = simulate_parent_pool(
        parent, config.total_count, seed=config.seed, noise_std=np.sqrt(config.noise_variance)
    )
    for design in config.designs:
        small, large = [pool.observations(design, j) for j in config.candidate_dimensions]
        for left, right in zip(small, large, strict=True):
            np.testing.assert_array_equal(left.y, right.y)
            np.testing.assert_array_equal(left.noise_covariance, right.noise_covariance)
            np.testing.assert_array_equal(left.operator, right.operator[:, :2])


def test_future_and_latent_tampering_cannot_change_training_or_selection(
    tmp_path, monkeypatch, no_figures
):
    config = tiny_config(
        replications=1,
        designs=("fixed_points",),
        training_counts=(8,),
        candidate_dimensions=(2,),
        geometry_dimensions=(2,),
    )
    original_simulate, original_prepare = (
        benchmark.simulate_parent_pool,
        benchmark.prepare_parent_model,
    )
    first = run_partial_benchmark(config, tmp_path / "first")
    original_fits = json.loads((first / "candidate_fits.json").read_text())
    original_choice = json.loads((first / "selection_audit.json").read_text())
    test_start = max(config.training_counts) + config.validation_count

    def changed_future(*args, **kwargs):
        pool = original_simulate(*args, **kwargs)
        values = pool.y.copy()
        values[test_start:] *= 20
        return replace(pool, y=values)

    def changed_latent(*args, **kwargs):
        parent = original_prepare(*args, **kwargs)
        return replace(parent, innovation_covariance=parent.innovation_covariance * 3)

    # Keep the actual observations fixed while replacing population evaluation truth.
    parent = original_prepare(dimension=config.reference_dimension)

    def fixed_observed_source(_parent, *args, **kwargs):
        return changed_future(parent, *args, **kwargs)

    monkeypatch.setattr(benchmark, "prepare_parent_model", changed_latent)
    monkeypatch.setattr(benchmark, "simulate_parent_pool", fixed_observed_source)
    second = run_partial_benchmark(config, tmp_path / "tampered")
    assert json.loads((second / "candidate_fits.json").read_text()) == original_fits
    assert json.loads((second / "selection_audit.json").read_text()) == original_choice
    old = json.loads((first / "results.json").read_text())["selected_summary"]
    new = json.loads((second / "results.json").read_text())["selected_summary"]
    assert old[0]["test_observed_score_mean"] != new[0]["test_observed_score_mean"]
    assert old[0]["operator_hs_error_mean"] != new[0]["operator_hs_error_mean"]


def test_identical_seed_reproduces_numerical_results(tmp_path, no_figures):
    config = tiny_config(
        replications=1,
        designs=("fixed_deliveries",),
        training_counts=(8,),
        candidate_dimensions=(2,),
        geometry_dimensions=(2,),
    )
    first = run_partial_benchmark(config, tmp_path / "a")
    second = run_partial_benchmark(config, tmp_path / "b")
    for name in ("candidate_fits.json", "selection_audit.json", "results.json"):
        assert (first / name).read_text() == (second / name).read_text()


def test_failed_candidate_keeps_audit_and_fails_run_without_relaxing_budget(
    tmp_path, monkeypatch, no_figures
):
    config = tiny_config(
        replications=1,
        designs=("fixed_points",),
        training_counts=(8,),
        candidate_dimensions=(2,),
        geometry_dimensions=(2,),
    )
    original = benchmark.fit_candidate

    def failed_fit(*args, **kwargs):
        candidate = original(*args, **kwargs)
        estimate = replace(
            candidate.estimate,
            converged=False,
            status="maximum iterations reached",
            covariance_error_bound=2 * config.numerical_error_budget,
        )
        return replace(candidate, estimate=estimate)

    monkeypatch.setattr(benchmark, "fit_candidate", failed_fit)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="numerical checks failed"):
        run_partial_benchmark(config, output)
    results = json.loads((output / "results.json").read_text())
    assert not results["acceptance"]["passed"]
    assert results["acceptance"]["failed_fits"] == len(config.relative_regularizations)
    assert not results["selections"][0]["selection_valid"]
    assert len(json.loads((output / "candidate_fits.json").read_text())) == 2
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert results["acceptance"]["numerical_error_budget"] == config.numerical_error_budget


def test_reference_checks_cover_each_scenario(tmp_path, no_figures):
    pytest.importorskip("cvxpy")
    config = tiny_config(
        replications=1,
        scenarios=("powerlaw", "rotated"),
        designs=("fixed_points",),
        training_counts=(8,),
        candidate_dimensions=(2,),
        geometry_dimensions=(2,),
        relative_regularizations=(0.01,),
        reference_checks=True,
    )
    output = run_partial_benchmark(config, tmp_path / "reference")
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["reference_cases"] == 2
    assert results["acceptance"]["reference_passed"]
    assert {row["scenario"] for row in results["reference_checks"]} == set(config.scenarios)


def test_paired_mcse_uses_within_replication_differences(audited_run):
    _, output, results = audited_run
    with (output / "selected.csv").open() as stream:
        raw = list(csv.DictReader(stream))
    for paired in results["paired_differences"]:
        if paired["metric"] != "observed_score":
            continue
        rows = [
            row
            for row in raw
            if row["scenario"] == paired["scenario"]
            and row["design"] == paired["design"]
            and int(row["N"]) == paired["N"]
        ]
        fixed = {
            int(row["replication"]): float(row["test_observed_score"])
            for row in rows
            if row["rule"] == "fixed"
        }
        differences = [
            float(row["test_observed_score"]) - fixed[int(row["replication"])]
            for row in rows
            if row["rule"] == "selected"
        ]
        assert paired["mean_difference"] == pytest.approx(np.mean(differences))
        assert paired["mcse"] == pytest.approx(np.std(differences, ddof=1) / np.sqrt(2))


def test_nondefault_source_configuration_is_honored():
    config = tiny_config(
        scenarios=("rotated",),
        source_exponent=4.0,
        source_trace=0.7,
        rotation_angle=0.0,
        designs=("fixed_points",),
        training_counts=(8,),
    )
    parent = benchmark._prepare(config)["parents"]["rotated"]
    assert parent.source_metadata["exponent"] == 4.0
    assert parent.source_metadata["infinite_trace"] == 0.7
    assert parent.source_metadata["rotation_angle"] == 0.0
    np.testing.assert_allclose(parent.source_covariance, np.diag(np.diag(parent.source_covariance)))
