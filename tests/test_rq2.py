"""Independent algebraic checks for the RQ2 local estimation experiment."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from spde_covariance import rq2, rq2_reporting
from spde_covariance.experiments import main
from spde_covariance.observations import Observation
from spde_covariance.rq2 import direct_risk_diagnostics, fit_window, moment_bias, run_rq2_benchmark
from spde_covariance.rq2_design import RQ2Config, generate_data


def test_direct_risk_gaussian_bias_and_variance_with_known_noise():
    exposure = np.array([1.0, -0.5])
    target = np.outer(exposure, exposure)
    truths = np.array([[[0.8, 0.2], [0.2, 0.4]], [[0.4, -0.1], [-0.1, 0.7]]])
    noises = np.array([np.diag([0.1, 0.2]), np.diag([0.3, 0.1])])
    values = np.array([[0.7, -0.3], [0.2, 0.8]])
    data = tuple(
        Observation(float(i), value, np.eye(2), noise)
        for i, (value, noise) in enumerate(zip(values, noises, strict=True))
    )
    result = direct_risk_diagnostics(data, truths, target)
    expected = np.mean([exposure @ truth @ exposure for truth in truths])
    terminal = exposure @ truths[-1] @ exposure
    estimate = np.mean(
        [
            (exposure @ value) ** 2 - exposure @ noise @ exposure
            for value, noise in zip(values, noises, strict=True)
        ]
    )
    # For independent Gaussian scalar exposures, Var(z^2) = 2 Var(z)^2.
    variance = 0.5 * sum(
        (exposure @ (truth + noise) @ exposure) ** 2
        for truth, noise in zip(truths, noises, strict=True)
    )
    assert result["identified"]
    assert result["unavailable_reason"] is None
    assert result["truth"] == pytest.approx(terminal)
    assert result["estimate"] == pytest.approx(estimate)
    assert result["error"] == pytest.approx(estimate - terminal)
    assert result["squared_error"] == pytest.approx((estimate - terminal) ** 2)
    assert result["expected_direct"] == pytest.approx(expected)
    assert result["analytic_bias"] == pytest.approx(expected - terminal)
    assert result["analytic_variance"] == pytest.approx(variance)
    assert result["analytic_mse"] == pytest.approx(variance + (expected - terminal) ** 2)


def test_changing_observation_maps_weight_temporal_bias_in_observation_geometry():
    # Scalar maps 1 and 2 give squared lift strengths 1 and 16. Their local
    # population fit weights V1 and V2 by 1:16 rather than taking their mean.
    data = (
        Observation(0.0, [0.7], [[1.0]], [[0.2]]),
        Observation(1.0, [1.2], [[2.0]], [[0.3]]),
    )
    truths = np.array([[[1.0]], [[3.0]]])
    result = direct_risk_diagnostics(data, truths, np.ones((1, 1)))
    assert result["expected_direct"] == pytest.approx(49 / 17)
    assert result["expected_direct"] != pytest.approx(2.0)
    assert result["analytic_bias"] == pytest.approx(-2 / 17)
    # Q = 17/2, so the direct influence matrices are 2/17 and 8/17.
    variance = 0.5 * ((2 / 17 * 1.2) ** 2 + (8 / 17 * 12.3) ** 2)
    assert result["analytic_variance"] == pytest.approx(variance)
    np.testing.assert_allclose(moment_bias(data, truths), [[-1.0]])


def test_constant_covariance_direct_risk_has_zero_temporal_bias():
    data = (
        Observation(0.0, [0.7], [[1.0]], [[0.2]]),
        Observation(1.0, [1.2], [[2.0]], [[0.3]]),
    )
    truths = np.array([[[0.8]], [[0.8]]])
    result = direct_risk_diagnostics(data, truths, np.ones((1, 1)))
    assert result["expected_direct"] == pytest.approx(0.8)
    assert result["analytic_bias"] == pytest.approx(0.0, abs=1e-14)
    assert result["analytic_mse"] == pytest.approx(result["analytic_variance"])
    np.testing.assert_allclose(moment_bias(data, truths), 0.0, atol=1e-14)


def test_changing_evaluation_truth_cannot_change_direct_estimate():
    observations = (
        Observation(0.0, [0.7], [[1.0]], [[0.2]]),
        Observation(1.0, [1.2], [[2.0]], [[0.3]]),
    )
    first = direct_risk_diagnostics(observations, np.array([[[1.0]], [[3.0]]]), [[1.0]])
    second = direct_risk_diagnostics(observations, np.array([[[2.0]], [[1.0]]]), [[1.0]])
    assert first["estimate"] == second["estimate"]
    assert first["truth"] != second["truth"]
    assert first["expected_direct"] != second["expected_direct"]
    assert first["analytic_variance"] != second["analytic_variance"]


def test_unobservable_direct_risk_is_explicitly_unavailable_and_json_safe():
    data = (
        Observation(0.0, [0.7], [[1.0, 0.0]], [[0.1]]),
        Observation(1.0, [1.2], [[0.0, 1.0]], [[0.1]]),
    )
    truths = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    target = np.array([[1.0, -1.0], [-1.0, 1.0]])
    result = direct_risk_diagnostics(data, truths, target)
    assert not result["identified"]
    assert result["unavailable_reason"]
    assert result["truth"] == pytest.approx(2.0)
    for field in (
        "estimate",
        "error",
        "squared_error",
        "expected_direct",
        "analytic_bias",
        "analytic_variance",
        "analytic_mse",
    ):
        assert result[field] is None
    json.dumps(result, allow_nan=False)


def test_identity_control_matches_known_ridge_and_empirical_solutions():
    truth = np.diag([0.4, 0.3, 0.2, 0.1])
    samples = 2 * np.linalg.cholesky(truth)
    data = tuple(
        Observation(float(index), samples[:, index], np.eye(4), np.zeros((4, 4)))
        for index in range(4)
    )
    config = RQ2Config()
    fits = fit_window(data, config, config.relative_regularization)
    constrained = fits["constrained_ridge"]
    assert constrained.converged
    assert constrained.covariance_error_bound <= config.numerical_error_budget
    assert constrained.min_eigenvalue >= -1e-12
    assert constrained.trace <= config.trace_cap + 1e-12
    exact = truth / (1 + config.relative_regularization)
    np.testing.assert_allclose(
        constrained.covariance, exact, atol=config.numerical_error_budget, rtol=0
    )
    np.testing.assert_allclose(fits["projected_ridge"], exact, atol=1e-12)
    np.testing.assert_allclose(fits["empirical"], truth, atol=1e-12)


@pytest.fixture(scope="module")
def tiny_rq2_run(tmp_path_factory):
    config = RQ2Config(
        intervals=80,
        replications=2,
        scenarios=("constant",),
        windows=(16,),
        endpoints=(64,),
        sensitivity_window=16,
        sensitivity_relatives=(),
        reference_checks=False,
    )
    output = tmp_path_factory.mktemp("rq2") / "run"
    return config, output, run_rq2_benchmark(config, output)


def test_tiny_run_writes_complete_finite_artifacts(tiny_rq2_run):
    _, output, returned = tiny_rq2_run
    assert returned == output
    expected = {
        "manifest.json",
        "config.json",
        "results.json",
        "replications.csv",
        "risks.csv",
        "estimates.json",
        "geometry.json",
        "report.md",
    }
    assert expected.issubset({item.name for item in output.iterdir()})
    assert len(list(output.glob("*.png"))) == 3
    for filename in expected:
        path = output / filename
        assert path.stat().st_size > 0
        if path.suffix == ".json":
            parsed = json.loads(path.read_text())
            json.dumps(parsed, allow_nan=False)


def test_saved_identity_estimate_uses_exact_past_window(tiny_rq2_run):
    config, output, _ = tiny_rq2_run
    estimates = json.loads((output / "estimates.json").read_text())
    for replication in range(config.replications):
        data = generate_data(config, replication)
        identity = next(
            item
            for item in estimates
            if item["replication"] == replication
            and item["design"] == "full"
            and item["method"] == "empirical"
        )
        # The target date is interval 64 although simulation continues to 80.
        past = data.latent["constant"][48:64]
        expected = past.T @ past / 16
        np.testing.assert_allclose(identity["covariance"], expected, atol=1e-14)
        future = data.latent["constant"][64:80]
        assert not np.allclose(identity["covariance"], future.T @ future / 16)
    geometry = json.loads((output / "geometry.json").read_text())
    assert len(geometry) == 5
    for item in geometry:
        assert item["first_interval"] == 49
        assert item["last_interval"] == 64
        assert item["effective_sample_size"] == 16
        assert item["trailing_window_years"] == pytest.approx(16 / 52)
        assert item["regularization"] == pytest.approx(
            config.relative_regularization * max(item["eigenvalues"])
        )


def test_reported_monte_carlo_mse_and_paired_uncertainty_match_raw_replications(tiny_rq2_run):
    config, output, _ = tiny_rq2_run
    results = json.loads((output / "results.json").read_text())
    with (output / "risks.csv").open(newline="") as stream:
        risks = list(csv.DictReader(stream))
    assert results["acceptance"]["passed"]
    assert results["acceptance"]["failed_fits"] == 0
    assert results["acceptance"]["maximum_covariance_error_bound"] <= config.numerical_error_budget
    assert results["acceptance"]["statistical_performance_gate"] is None
    for summary in results["risk_summary"]:
        matching = [
            row
            for row in risks
            if row["design"] == summary["design"]
            and row["target"] == summary["target"]
            and row["method"] == summary["method"]
            and row["estimate"] != ""
        ]
        assert summary["available_replications"] == len(matching)
        if not matching:
            assert summary["mse"] is None
            assert summary["mse_mcse"] is None
            continue
        errors = np.array([float(row["error"]) for row in matching])
        squared_errors = errors**2
        assert summary["bias"] == pytest.approx(errors.mean())
        assert summary["mse"] == pytest.approx(squared_errors.mean())
        assert summary["mse_mcse"] == pytest.approx(
            squared_errors.std(ddof=1) / np.sqrt(len(squared_errors))
        )
    for paired in results["paired_differences"]:
        matching = [
            row
            for row in risks
            if row["design"] == paired["design"] and row["target"] == paired["target"]
        ]
        baseline = {
            row["replication"]: float(row["squared_error"])
            for row in matching
            if row["method"] == "constrained_ridge"
        }
        differences = np.array(
            [
                float(row["squared_error"]) - baseline[row["replication"]]
                for row in matching
                if row["method"] == paired["comparison_method"] and row["squared_error"] != ""
            ]
        )
        if len(differences):
            assert paired["mse_difference"] == pytest.approx(differences.mean())
            assert paired["mcse"] == pytest.approx(
                differences.std(ddof=1) / np.sqrt(len(differences))
            )
        else:
            assert paired["mse_difference"] is None
            assert paired["mcse"] is None


def test_existing_run_is_preserved(tiny_rq2_run):
    config, output, _ = tiny_rq2_run
    existing = {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()}
    with pytest.raises(FileExistsError):
        run_rq2_benchmark(config, output)
    assert {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()} == existing


def test_failed_numerical_certificate_saves_diagnostics_and_rejects_run(tmp_path, monkeypatch):
    config = RQ2Config(
        intervals=16,
        replications=1,
        windows=(4,),
        endpoints=(16,),
        sensitivity_window=4,
        sensitivity_relatives=(),
        scenarios=("constant",),
        reference_checks=False,
    )
    data = tuple(
        Observation(float(index), np.eye(4)[index], np.eye(4), np.zeros((4, 4)))
        for index in range(4)
    )
    fit = fit_window(data, config, config.relative_regularization)["constrained_ridge"]
    unfinished = replace(fit, converged=False, status="maximum_iterations")

    def unfinished_fit(*args, **kwargs):
        kwargs["timings"]["constrained_ridge"] = 0.0
        return {"constrained_ridge": unfinished}

    def cheap_report(output, *_):
        (output / "report.md").write_text("Numerical acceptance failed.\n")

    monkeypatch.setattr(rq2, "fit_window", unfinished_fit)
    monkeypatch.setattr(rq2_reporting, "write_report", cheap_report)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="numerical checks failed"):
        run_rq2_benchmark(config, output)
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    results = json.loads((output / "results.json").read_text())
    assert not results["acceptance"]["passed"]
    assert results["acceptance"]["failed_fits"] == 5
    assert (output / "report.md").exists()
    assert (output / "replications.csv").exists()


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [(RuntimeError("generation failed"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_unexpected_generation_failure_leaves_honest_status(
    tmp_path, monkeypatch, exception, expected_status
):
    def failed_generation(*_):
        raise exception

    monkeypatch.setattr(rq2, "generate_data", failed_generation)
    output = tmp_path / "interrupted"
    with pytest.raises(type(exception)):
        run_rq2_benchmark(RQ2Config(reference_checks=False), output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == expected_status
    assert status["error_type"] == type(exception).__name__
    assert status["error"] == str(exception)
    assert (output / "manifest.json").exists()
    assert (output / "config.json").exists()


def test_unconverged_reference_rejects_run_despite_perfect_matrix_agreement(tmp_path, monkeypatch):
    config = RQ2Config(
        intervals=16,
        replications=1,
        windows=(4,),
        endpoints=(16,),
        sensitivity_window=4,
        sensitivity_relatives=(),
        scenarios=("constant",),
        reference_checks=True,
    )
    observations = tuple(
        Observation(float(index), np.eye(4)[index], np.eye(4), np.zeros((4, 4)))
        for index in range(4)
    )
    fit = fit_window(observations, config, config.relative_regularization)["constrained_ridge"]
    reference = replace(fit, converged=False, status="unconverged_reference")

    def cheap_fit(*args, **kwargs):
        kwargs["timings"]["constrained_ridge"] = 0.0
        return {"constrained_ridge": fit}

    def cheap_report(output, *_):
        (output / "report.md").write_text("Reference acceptance failed.\n")

    monkeypatch.setattr(rq2, "fit_window", cheap_fit)
    monkeypatch.setattr(rq2, "fit_covariance_reference", lambda *args, **kwargs: reference)
    monkeypatch.setattr(rq2_reporting, "write_report", cheap_report)
    output = tmp_path / "reference_failure"
    with pytest.raises(RuntimeError, match="numerical checks failed"):
        run_rq2_benchmark(config, output)
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["failed_fits"] == 0
    assert results["acceptance"]["feasibility_passed"]
    assert not results["acceptance"]["reference_passed"]
    assert not results["acceptance"]["passed"]
    assert results["acceptance"]["reference_cases"] == 2
    for item in results["reference_checks"]:
        assert item["matrix_discrepancy"] == 0.0
        assert item["reference_covariance_error_bound"] <= config.numerical_error_budget
        assert not item["reference_converged"]
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_rq2_cli_help_and_missing_configuration(capsys, tmp_path):
    with pytest.raises(SystemExit) as help_exit:
        main(["benchmark-rq2", "--help"])
    assert help_exit.value.code == 0
    help_output = capsys.readouterr().out
    assert "config" in help_output
    assert "--output" in help_output
    with pytest.raises(SystemExit) as missing_exit:
        main(["benchmark-rq2", str(tmp_path / "missing.toml")])
    assert missing_exit.value.code == 2
    assert "error:" in capsys.readouterr().err
