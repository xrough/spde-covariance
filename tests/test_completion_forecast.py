"""Chronology, source targets and honest scoring for completion forecasts."""

import hashlib
import json

import numpy as np
import pytest

from spde_covariance.completion_forecast import (
    METHODS,
    RULES,
    FragmentHistory,
    _aggregate,
    _fit_replication,
    exposure_proxy,
    fit_completion_forecast,
    observed_kernel_score,
    prepare_forecast_model,
    run_completion_forecast,
    select_forecast_window,
    source_path_parameters,
)
from spde_covariance.completion_forecast_config import CompletionForecastConfig
from spde_covariance.completion_strategy import CompletionStrategySettings
from spde_covariance.laguerre import LaguerreTransport, PowerLawSource
from spde_covariance.provenance import _jsonable


def tiny_config(**kwargs):
    values = dict(
        replications=2,
        intervals=48,
        reference_dimension=4,
        grid=5,
        covers=(3,),
        scenarios=("constant", "abrupt"),
        windows=(9, 16),
        validation_origins=(16, 24),
        test_origins=(32, 40),
        horizon=2,
        static_window=16,
        fixed_window=16,
        workers=1,
    )
    values.update(kwargs)
    return CompletionForecastConfig(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"workers": 5},
        {"seed": True},
        {"noise_variance": float("nan")},
        {"source_exponent": 2},
        {"reference_dimension": 1},
        {"windows": (16, 9)},
        {"test_origins": (24, 40)},
        {"covers": (3, 17)},
    ],
)
def test_config_rejects_invalid_protocol(change):
    with pytest.raises(ValueError):
        tiny_config(**change)


def test_source_paths_are_transport_of_time_varying_source_not_posthoc_V():
    config = tiny_config(scenarios=("constant", "scale", "rotating", "abrupt"))
    prepared = prepare_forecast_model(config)
    model = LaguerreTransport(config.reference_dimension)
    eigen = PowerLawSource(config.source_exponent, config.source_trace).eigenvalues(model.dimension)
    evaluation = model.evaluate(prepared["nodes"])
    for scenario in config.scenarios:
        scales, angles = source_path_parameters(config, scenario)
        for index in (0, config.intervals // 2, config.intervals - 1):
            c, s = np.cos(angles[index]), np.sin(angles[index])
            rotation = np.eye(model.dimension)
            rotation[:2, :2] = [[c, -s], [s, c]]
            source = scales[index] * (rotation @ np.diag(eigen) @ rotation.T)
            expected = evaluation @ model.innovation_covariance(source, config.delta) @ evaluation.T
            np.testing.assert_allclose(
                prepared["paths"][scenario][index], expected, atol=3e-14, rtol=3e-14
            )
    assert prepared["metadata"]["source_prefix_trace"] < config.source_trace
    assert prepared["metadata"]["source_tail_trace_max"] > 0
    assert prepared["metadata"]["integration_numerical_error_certificate"] is False


def test_reduced_observed_score_and_exact_proxy_noise_correction():
    kernel = np.array([[1, 0.3, -0.2], [0.3, 2, 0.1], [-0.2, 0.1, 1.0]])
    row = np.array([2, 0])
    value = np.array([2.0, -1.0])
    noise = 0.2
    z = np.outer(value, value) - noise * np.eye(2)
    pred = kernel[np.ix_(row, row)]
    expected = np.sum((pred - z) ** 2) - np.sum(z**2)
    assert observed_kernel_score(kernel, value, row, noise) == pytest.approx(expected)
    exposure = np.array([1.0, 0.0, -2.0])
    assert exposure_proxy(value, row, exposure, noise) == pytest.approx((-5) ** 2 - noise * 5)
    assert exposure_proxy(value, row, np.array([0.0, 1.0, 0.0]), noise) is None


def test_population_score_expectation_differences_equal_prediction_loss():
    truth = np.array([[2.0, 0.4], [0.4, 1.0]])
    a = np.eye(2)
    b = np.array([[2.0, 0.0], [0.0, 3.0]])
    # Integrate yy' exactly using a two-point quadrature of each Gaussian mode.
    eigen, vectors = np.linalg.eigh(truth + 0.2 * np.eye(2))
    samples = [sign * np.sqrt(2 * eigen[j]) * vectors[:, j] for j in range(2) for sign in (-1, 1)]
    observed = np.mean(
        [
            observed_kernel_score(a, y, np.arange(2), 0.2)
            - observed_kernel_score(b, y, np.arange(2), 0.2)
            for y in samples
        ]
    )
    assert observed == pytest.approx(np.sum((a - truth) ** 2) - np.sum((b - truth) ** 2))


def test_window_tie_longest_and_any_failed_candidate_blocks_selection():
    assert select_forecast_window({16: [1.0, 2.0], 128: [1.0, 2.0]}) == (128, None)
    window, reason = select_forecast_window({16: [1.0, None], 128: [100.0, 100.0]})
    assert window is None and "unavailable" in reason
    with pytest.raises(ValueError, match="same number"):
        select_forecast_window({16: [1.0], 128: [1.0, 2.0]})


def test_fit_accepts_only_exact_past_prefix_and_copies_observations():
    rng = np.random.default_rng(7)
    rows = tuple(np.arange(3) for _ in range(16))
    values = tuple(rng.standard_normal(3) for _ in rows)
    history = FragmentHistory(values, rows, 3, 0.1)
    values[0][:] = 123
    assert np.max(history.values[0]) < 123
    settings = CompletionStrategySettings(
        partial_method="coherent",
        relative_regularization=0.001,
        floor_policy="resolution",
        numerical_error_budget=1e-4,
    )
    issue = fit_completion_forecast(
        history, origin=16, window=9, cliques=(np.arange(3),), weights=np.ones(3), settings=settings
    )
    assert issue.available
    assert issue.training_first_interval == 8 and issue.training_last_interval == 16
    assert not issue.kernel.flags.writeable
    assert issue.diagnostics["partial_error_bound"] <= 1e-4
    assert np.linalg.eigvalsh(issue.kernel)[0] >= -1e-10
    with pytest.raises(ValueError, match="exactly"):
        fit_completion_forecast(
            history,
            origin=15,
            window=9,
            cliques=(np.arange(3),),
            weights=np.ones(3),
            settings=settings,
        )
    with pytest.raises(TypeError, match="scenario-free"):
        fit_completion_forecast(
            history,
            origin=16,
            window=9,
            cliques=(np.arange(3),),
            weights=np.ones(3),
            settings=tiny_config(),
        )


def test_seeded_generation_chronology_and_latent_truth_isolation():
    config = tiny_config(replications=1, scenarios=("constant",))
    prepared = prepare_forecast_model(config)
    first = _fit_replication(0, config, prepared)
    altered = {**prepared, "paths": {"constant": prepared["paths"]["constant"] * 7}}
    second = _fit_replication(0, config, altered)

    def stable(rows):
        return _jsonable([{k: v for k, v in row.items() if k != "runtime_seconds"} for row in rows])

    assert stable(first["fits"]) == stable(second["fits"])
    assert first["validations"] == second["validations"]
    assert first["selections"] == second["selections"]
    assert first["parent_values_sha256"] == second["parent_values_sha256"]
    assert first["evaluation"][0]["kernel_error"] != second["evaluation"][0]["kernel_error"]
    assert all(r["selection_endpoint"] == 26 for r in first["selections"])
    for row in first["fits"]:
        assert row["training_last_interval"] == row["origin"]
        assert row["training_first_interval"] == row["origin"] - row["window"] + 1
    # One observed training prefix is identical across methods.
    by_window = {}
    for row in first["fits"]:
        by_window.setdefault((row["origin"], row["window"]), set()).add(row["training_sha256"])
    assert all(len(hashes) == 1 for hashes in by_window.values())
    future_roots = prepared["roots"]["constant"].copy()
    future_roots[40:] *= 5
    future = _fit_replication(0, config, {**prepared, "roots": {"constant": future_roots}})
    assert stable(first["fits"]) == stable(future["fits"])
    assert first["selections"] == future["selections"]
    assert first["evaluation"][-1]["observed_score"] != future["evaluation"][-1]["observed_score"]


def test_replication_means_define_paired_MCSE_not_individual_dates():
    records = []
    for rep, value in enumerate((1.0, 5.0)):
        evaluations = []
        risks = []
        for method in METHODS:
            for rule in RULES:
                for origin in (32, 40):
                    for horizon in (1, 2):
                        row = dict(
                            replication=rep,
                            scenario="constant",
                            cover=3,
                            method=method,
                            rule=rule,
                            origin=origin,
                            horizon=horizon,
                            available=True,
                            observed_score=value,
                            kernel_error=value,
                            missing_kernel_error=value,
                        )
                        evaluations.append(row)
                        risks.append(
                            {
                                **row,
                                "target": "long_spread",
                                "risk_score": None,
                                "risk_squared_error": value,
                            }
                        )
        records.append({"evaluation": evaluations, "risks": risks})
    results, averages, _ = _aggregate(records)
    assert len(averages) == 18
    assert results["summary"][0]["observed_score_mean"] == 3
    assert results["summary"][0]["observed_score_mcse"] == 2
    assert results["risk_summary"][0]["risk_score_available"] == 0
    assert results["risk_summary"][0]["risk_score_mean"] is None
    assert all(r["observed_score_mean"] == 0 for r in results["paired_differences"])


def test_failed_forecast_cannot_disappear_from_replication_average():
    from spde_covariance.completion_forecast import _replication_averages

    rows = [
        {"replication": 0, "available": True, "score": 42.0},
        {"replication": 0, "available": False, "score": None},
    ]
    result = _replication_averages(rows, ("replication",), ("score",))[0]
    assert not result["available"]
    assert result["score"] is None
    assert result["score_available_dates"] == 1


def test_failed_validation_window_keeps_static_fixed_and_blocks_selected(monkeypatch):
    from dataclasses import replace

    import spde_covariance.completion_forecast as module

    original = module.fit_completion_forecast

    def controlled_failure(*args, **kwargs):
        fit = original(*args, **kwargs)
        if (
            kwargs["origin"] == 16
            and kwargs["window"] == 9
            and kwargs["settings"].partial_method == "coherent"
        ):
            return replace(
                fit, kernel=None, available=False, reason="controlled certificate failure"
            )
        return fit

    monkeypatch.setattr(module, "fit_completion_forecast", controlled_failure)
    config = tiny_config(replications=1, scenarios=("constant",))
    record = module._fit_replication(0, config, module.prepare_forecast_model(config))
    selected = next(x for x in record["selections"] if x["method"] == "coherent")
    assert selected["selected_window"] is None
    assert "unavailable" in selected["reason"]
    rows = [x for x in record["evaluation"] if x["method"] == "coherent"]
    assert all(
        not x["available"] and x["observed_score"] is None for x in rows if x["rule"] == "selected"
    )
    assert all(x["available"] for x in rows if x["rule"] in ("static", "fixed"))


def test_small_end_to_end_report_and_manifest_bound_checkpoints(tmp_path):
    output = tmp_path / "run"
    config = tiny_config(replications=1, scenarios=("constant",))
    assert run_completion_forecast(config, output) == output
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["passed"]
    checkpoint = json.loads((output / "replication-0000.json").read_text())
    assert (
        checkpoint["parent_manifest_sha256"]
        == hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    )
    assert len(checkpoint["evaluation"]) == 3 * 3 * 2 * 2
    assert all(r["risk_score"] is None for r in checkpoint["risks"] if r["target"] == "long_spread")
    for name in (
        "forecast-comparison.png",
        "forecast-horizons.png",
        "forecast-risks.png",
        "report.md",
    ):
        assert (output / name).stat().st_size > 100
    with pytest.raises(FileExistsError):
        run_completion_forecast(config, output)


def test_spawned_workers_preserve_seed_order_and_all_scientific_fields():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from spde_covariance.completion_forecast import _initialize_worker, _worker_replication

    config = tiny_config(
        scenarios=("constant",), validation_origins=(16,), test_origins=(32,), horizon=1
    )
    prepared = prepare_forecast_model(config)
    serial = [_fit_replication(i, config, prepared) for i in range(2)]
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        parallel = list(executor.map(_worker_replication, range(2)))

    def stable(value):
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items() if key != "runtime_seconds"}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    assert [record["replication"] for record in parallel] == [0, 1]
    assert stable(_jsonable(serial)) == stable(_jsonable(parallel))
