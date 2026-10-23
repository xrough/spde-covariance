"""Chronology, operational scoring and independent replication checks for RQ3."""

import csv
import json
from dataclasses import fields, replace

import numpy as np
import pytest

from spde_covariance import rq3
from spde_covariance.estimation import fit_covariance_reference
from spde_covariance.geometry import diagnose_design
from spde_covariance.observations import Observation
from spde_covariance.rq2_design import generate_data
from spde_covariance.rq3 import (
    RQ3Config,
    fit_forecast,
    observed_score,
    run_rq3_benchmark,
    score_exposure,
    select_forecast_window,
    select_window,
    summarize_scores,
)


def tiny_config(**changes):
    settings = dict(
        intervals=96,
        replications=2,
        windows=(4, 8, 16),
        validation_origins=(16, 32),
        test_origins=(64, 72),
        horizon=4,
        static_window=16,
        fixed_window=16,
        scenarios=("constant",),
        designs=("full", "split"),
        reference_checks=False,
    )
    settings.update(changes)
    return RQ3Config(**settings)


def test_defaults_and_adapter_reuse_the_rq2_innovation_paths():
    config = RQ3Config()
    assert config.windows == (16, 32, 64, 128)
    assert config.validation_origins == (128, 192)
    assert config.test_origins == (256, 272, 304, 448)
    assert config.horizon == 16
    adapter = config.simulation_config()
    assert adapter.seed == config.seed
    assert adapter.intervals == 512
    assert adapter.delta == 1 / 52
    assert adapter.scenarios == config.scenarios
    assert adapter.sensitivity_relatives == ()
    assert {field.name for field in fields(config.fit_settings())} == {
        "delta",
        "relative_regularization",
        "trace_cap",
        "numerical_error_budget",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"replications": 0},
        {"horizon": 0},
        {"intervals": 50},
        {"windows": ()},
        {"windows": (4, 4)},
        {"windows": (17,)},
        {"validation_origins": (32, 16)},
        {"test_origins": (64, 66)},
        {"validation_origins": (16, 64)},
        {"test_origins": (64, 96)},
        {"static_window": 17},
        {"fixed_window": 17},
        {"rank_rtol": 1},
        {"relative_regularization": 0},
        {"noise_variance": float("nan")},
        {"reference_checks": 1},
        {"scenarios": ("unknown",)},
        {"designs": ("unknown",)},
        {"designs": ("full", "full")},
    ],
)
def test_invalid_protocol_is_rejected_before_simulation(change):
    with pytest.raises(ValueError):
        tiny_config(**change)


def test_identity_forecast_is_known_scaled_second_moment_and_snapshot():
    config = tiny_config()
    data = generate_data(config.simulation_config(), 0)
    observations = data.designs["constant"]["full"]
    settings = config.fit_settings()
    forecast = fit_forecast(
        observations[:16], settings, origin=16, window=8, basis_id=data.basis_id
    )
    samples = data.latent["constant"][8:16]
    exact = samples.T @ samples / 8 / (1 + config.relative_regularization)
    # The trace cap is inactive for these seeded samples.
    assert np.trace(exact) < config.trace_cap
    np.testing.assert_allclose(
        forecast.covariance, exact, atol=config.numerical_error_budget, rtol=0
    )
    assert forecast.training_first_interval == 9
    assert forecast.training_last_interval == 16
    assert forecast.training_times == tuple(item.time for item in observations[8:16])
    assert forecast.estimate.converged
    for array in (forecast.covariance, forecast.estimate.covariance, forecast.estimate.weights):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 1
    with pytest.raises(ValueError, match="exactly"):
        fit_forecast(observations[:17], settings, origin=16, window=8, basis_id=data.basis_id)
    with pytest.raises(ValueError, match="times"):
        fit_forecast(observations[1:17], settings, origin=16, window=8, basis_id=data.basis_id)
    with pytest.raises(TypeError, match="scenario-free"):
        fit_forecast(observations[:16], config, origin=16, window=8, basis_id=data.basis_id)


def test_corrected_score_difference_targets_visible_covariance_loss_exactly():
    truth, noise = 0.7, 0.3
    first, second = np.array([[0.2]]), np.array([[0.9]])
    # Symmetric two-point quadrature integrates the needed Gaussian second moment exactly.
    observations = [
        Observation(1.0, [sign * np.sqrt(truth + noise)], [[1.0]], [[noise]]) for sign in (-1, 1)
    ]
    difference = np.mean(
        [observed_score(first, item) - observed_score(second, item) for item in observations]
    )
    assert difference == pytest.approx((0.2 - truth) ** 2 - (0.9 - truth) ** 2)
    item = observations[0]
    full = np.sum((item.measurement - first) ** 2)
    assert observed_score(first, item) + np.sum(item.measurement**2) == pytest.approx(full)
    assert observed_score(second, item) < 0  # Reduced scores need not be nonnegative.
    with pytest.raises(ValueError):
        observed_score(np.eye(2), item)


def test_exposure_scoring_corrects_noise_and_requires_future_joint_observation():
    covariance = np.diag([0.3, 0.2])
    observation = Observation(1.0, [3.0], [[2.0, 0.0]], [[0.4]])
    available = score_exposure(covariance, [1.0, 0.0], observation)
    assert available["available"]
    assert available["risk_proxy_noise_correction"] == pytest.approx(0.1)
    assert available["realized_risk_proxy"] == pytest.approx(2.15)
    assert available["risk_score"] == pytest.approx(0.3**2 - 2 * 0.3 * 2.15)
    unavailable = score_exposure(covariance, [1.0, -1.0], observation)
    assert not unavailable["available"]
    assert unavailable["risk_score"] is None
    assert unavailable["realized_risk_proxy"] is None
    assert unavailable["unavailable_reason"]
    with pytest.raises(ValueError):
        score_exposure(covariance, [0.0, 0.0], observation)
    # Historical pooled range can identify a diagonal exposure but does not make
    # that exposure observable from a different future row.
    past = diagnose_design(
        [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])], targets={"second": np.diag([0.0, 1.0])}
    )
    assert past["targets"]["second"]["identified"]
    assert not score_exposure(covariance, [0.0, 1.0], observation)["available"]


def test_indefinite_corrected_measurement_and_negative_risk_proxy_are_retained():
    observation = Observation(1.0, [0.0], [[1.0]], [[0.4]])
    assert observed_score([[0.2]], observation) == pytest.approx(0.04 + 0.16)
    risk = score_exposure([[0.2]], [1.0], observation)
    assert risk["realized_risk_proxy"] == pytest.approx(-0.4)
    assert risk["risk_score"] == pytest.approx(0.04 + 0.16)


def test_window_selection_uses_mean_and_longest_exact_tie():
    selected, candidates = select_window({4: [1.0, 3.0], 16: [2.0, 2.0], 8: [3.0, 3.0]})
    assert selected == 16
    assert [row["window"] for row in candidates] == [4, 8, 16]
    selected, _ = select_window({4: [-5.0, 0.0], 16: [0.0, 0.0]})
    assert selected == 4
    for invalid in ({}, {4: []}, {4: [0.0], 8: [0.0, 0.0]}, {4: [float("nan")]}):
        with pytest.raises(ValueError):
            select_window(invalid)


def test_validation_accepts_only_its_prefix_and_future_data_cannot_change_forecasts():
    config = tiny_config(replications=1, designs=("full",))
    data = generate_data(config.simulation_config(), 0)
    observations = data.designs["constant"]["full"]
    endpoint = config.validation_origins[-1] + config.horizon
    first = select_forecast_window(observations[:endpoint], config, basis_id=data.basis_id)
    changed = tuple(
        item
        if index < endpoint
        else Observation(item.time, item.y * 1000, item.operator, item.noise_covariance)
        for index, item in enumerate(observations)
    )
    second = select_forecast_window(changed[:endpoint], config, basis_id=data.basis_id)
    assert first == second
    with pytest.raises(ValueError, match="exactly"):
        select_forecast_window(observations, config, basis_id=data.basis_id)
    origin = config.test_origins[0]
    before = fit_forecast(
        observations[:origin],
        config.fit_settings(),
        origin=origin,
        window=first["selected_window"],
        basis_id=data.basis_id,
    )
    # Alter only outcomes after the issue origin, never the training prefix.
    after_data = tuple(
        item
        if index < origin
        else Observation(item.time, item.y * -999, item.operator, item.noise_covariance)
        for index, item in enumerate(observations)
    )
    after = fit_forecast(
        after_data[:origin],
        config.fit_settings(),
        origin=origin,
        window=first["selected_window"],
        basis_id=data.basis_id,
    )
    np.testing.assert_array_equal(before.covariance, after.covariance)


def test_clustered_and_paired_mcse_use_replications_not_dates():
    scores = []
    for replication, fixed, selected in (
        (0, [0.0, 2.0], [1.0, 5.0]),
        (1, [10.0, 14.0], [6.0, 8.0]),
    ):
        for rule, values in (("fixed", fixed), ("selected", selected), ("static", fixed)):
            scores.extend(
                dict(
                    scenario="constant",
                    design="dense",
                    rule=rule,
                    replication=replication,
                    origin=10,
                    horizon=horizon,
                    observed_score=value,
                )
                for horizon, value in enumerate(values, start=1)
            )
    result = summarize_scores(scores, [], [])
    fixed = next(row for row in result["summary"] if row["rule"] == "fixed")
    assert fixed["observed_score_mean"] == pytest.approx(6.5)  # Cluster means 1 and 12.
    assert fixed["observed_score_mcse"] == pytest.approx(5.5)
    difference = next(
        row for row in result["paired_differences"] if row["comparison_rule"] == "selected"
    )
    assert difference["mean_difference"] == pytest.approx(-1.5)  # Paired means 2 and -5.
    assert difference["mcse"] == pytest.approx(3.5)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    config = tiny_config()
    output = tmp_path_factory.mktemp("rq3") / "run"
    run_rq3_benchmark(config, output)
    return config, output


def test_run_saves_complete_finite_evidence_and_strict_chronology(tiny_run):
    config, output = tiny_run
    expected = {
        "config.json",
        "manifest.json",
        "status.json",
        "results.json",
        "scores.csv",
        "risks.csv",
        "truth_metrics.csv",
        "validation_scores.csv",
        "forecasts.json",
        "fit_diagnostics.json",
        "selection_audit.json",
        "geometry.json",
        "observation_schedule.json",
        "report.md",
    }
    assert expected.issubset({item.name for item in output.iterdir()})
    for path in output.glob("*.json"):
        json.dumps(json.loads(path.read_text()), allow_nan=False)
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["passed"]
    assert results["acceptance"]["maximum_covariance_error_bound"] <= config.numerical_error_budget
    assert results["acceptance"]["statistical_performance_gate"] is None
    forecasts = json.loads((output / "forecasts.json").read_text())
    for forecast in forecasts:
        assert forecast["training_last_interval"] <= forecast["origin"]
        assert forecast["issue_origin"] == forecast["training_last_interval"]
        assert forecast["basis_id"] == results["model"]["basis_id"]
        if forecast["rule"] == "static":
            assert forecast["training_first_interval"] == 1
            assert forecast["training_last_interval"] == config.static_window
        else:
            assert forecast["training_last_interval"] == forecast["origin"]
            assert (
                forecast["training_first_interval"] == forecast["origin"] - forecast["window"] + 1
            )
    with (output / "scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == config.replications * len(config.designs) * len(config.test_origins) * 3 * 4
    assert {int(row["observed_dimension"]) for row in rows} == {2, 4}
    assert all(
        int(row["target_interval"]) == int(row["origin"]) + int(row["horizon"]) for row in rows
    )
    # Each deterministic training geometry is saved once, despite paired paths/replications.
    geometry = json.loads((output / "geometry.json").read_text())
    assert len(geometry) == len({row["geometry_id"] for row in geometry})
    assert all("replication" not in row and "scenario" not in row for row in geometry)


def test_saved_static_and_selected_forecasts_and_raw_score_aggregation(tiny_run):
    config, output = tiny_run
    records = json.loads((output / "forecasts.json").read_text())
    selections = json.loads((output / "selection_audit.json").read_text())
    for selection in selections:
        selected, candidates = select_window(
            {row["window"]: row["validation_scores"] for row in selection["candidates"]}
        )
        assert selected == selection["selected_window"]
        assert candidates == selection["candidates"]
        assert selection["selection_complete_interval"] < min(config.test_origins)
    for replication in range(config.replications):
        data = generate_data(config.simulation_config(), replication)
        matching = [
            row
            for row in records
            if row["replication"] == replication
            and row["design"] == "full"
            and row["rule"] == "static"
        ]
        expected = fit_forecast(
            data.designs["constant"]["full"][:16],
            config.fit_settings(),
            origin=16,
            window=16,
            basis_id=data.basis_id,
        )
        for row in matching:
            np.testing.assert_array_equal(row["covariance"], expected.covariance)
    with (output / "scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    results = json.loads((output / "results.json").read_text())
    for summary in results["summary"]:
        clusters = [
            np.mean(
                [
                    float(row["observed_score"])
                    for row in rows
                    if row["design"] == summary["design"]
                    and row["rule"] == summary["rule"]
                    and int(row["replication"]) == replication
                ]
            )
            for replication in range(config.replications)
        ]
        assert summary["observed_score_mean"] == pytest.approx(np.mean(clusters))
        assert summary["observed_score_mcse"] == pytest.approx(
            np.std(clusters, ddof=1) / np.sqrt(2)
        )


def test_modifying_latent_truth_cannot_change_observed_selection_or_issued_forecasts(
    tiny_run,
    tmp_path,
    monkeypatch,
):
    config, baseline = tiny_run
    original = rq3.generate_data

    def changed_truth(settings, replication):
        generated = original(settings, replication)
        return replace(
            generated, truths={key: value * 1.7 for key, value in generated.truths.items()}
        )

    monkeypatch.setattr(rq3, "generate_data", changed_truth)
    changed = tmp_path / "changed"
    run_rq3_benchmark(config, changed)
    for name in ("forecasts.json", "selection_audit.json", "scores.csv", "risks.csv"):
        assert (baseline / name).read_bytes() == (changed / name).read_bytes()
    assert (baseline / "truth_metrics.csv").read_bytes() != (
        changed / "truth_metrics.csv"
    ).read_bytes()


def test_failed_certificate_rejects_run_and_retains_fit_diagnostics(tmp_path, monkeypatch):
    from spde_covariance import rq3_reporting

    config = tiny_config(replications=1, designs=("full",))
    original = rq3.fit_forecast

    def failed(*args, **kwargs):
        forecast = original(*args, **kwargs)
        return replace(
            forecast,
            estimate=replace(forecast.estimate, converged=False, status="maximum_iterations"),
        )

    monkeypatch.setattr(rq3, "fit_forecast", failed)
    monkeypatch.setattr(
        rq3_reporting,
        "write_report",
        lambda output, *_: (output / "report.md").write_text("Failed numerical acceptance.\n"),
    )
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="numerical checks failed"):
        run_rq3_benchmark(config, output)
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["failed_fits"] > 0
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert all(
        not row["solver_converged"]
        for row in json.loads((output / "fit_diagnostics.json").read_text())
    )


def test_reference_fallback_is_nonempty_and_agrees_with_independent_solver(tmp_path):
    pytest.importorskip("cvxpy")
    config = tiny_config(
        replications=1, scenarios=("abrupt",), designs=("full",), reference_checks=True
    )
    output = tmp_path / "reference"
    run_rq3_benchmark(config, output)
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["reference_cases"] == 1
    assert results["acceptance"]["reference_cases_expected"] == 1
    assert results["acceptance"]["reference_passed"]
    data = generate_data(config.simulation_config(), 0)
    selected = data.designs["abrupt"]["full"][48:64]
    own = fit_forecast(
        data.designs["abrupt"]["full"][:64],
        config.fit_settings(),
        origin=64,
        window=16,
        basis_id=data.basis_id,
    )
    reference = fit_covariance_reference(
        selected,
        regularization=own.estimate.regularization,
        trace_cap=config.trace_cap,
        solver_tolerance=1e-10,
    )
    assert (
        np.linalg.norm(reference.covariance - own.covariance) <= 2 * config.numerical_error_budget
    )


def test_existing_outputs_are_preserved_and_configuration_path_supported(tiny_run, tmp_path):
    config, output = tiny_run
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        run_rq3_benchmark(config, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(TypeError):
        run_rq3_benchmark({}, tmp_path / "invalid")
    path = tmp_path / "invalid.toml"
    path.write_text("horizon = 0\n")
    with pytest.raises(ValueError):
        run_rq3_benchmark(path, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()
