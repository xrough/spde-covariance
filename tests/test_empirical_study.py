"""Past-only calibration, actual contract P&L and aggregate publication boundary."""

import copy
import hashlib
import json
import warnings

import numpy as np
import pytest

import spde_covariance.empirical_study as study
from spde_covariance.empirical_calibration import EmpiricalSourceFit
from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.laguerre import LaguerreTransport


@pytest.fixture(scope="module")
def synthetic_panel():
    """Four synthetic years, with actual contract identities and lifecycle."""
    months = np.arange(np.datetime64("2022-01", "M"), np.datetime64("2026-01", "M"))
    dates = months.astype("datetime64[D]") + np.timedelta64(3, "D")
    rows = []
    basis = LaguerreTransport(3)
    for i, day in enumerate(dates):
        contracts = months[i] + np.arange(1, 14).astype("timedelta64[M]")
        maturities = contracts.astype("datetime64[D]") + np.timedelta64(14, "D")
        coefficients = np.array(
            [60000 + 45 * i + 180 * np.sin(i), 700 * np.cos(i / 4), -250 + 50 * np.sin(i / 3)]
        )
        prices = basis.evaluate(
            (maturities - day).astype(int) / 365.25
        ) @ coefficients + 4 * np.cos(np.arange(13) * 1.7)
        for j, (month, maturity, price) in enumerate(
            zip(contracts, maturities, prices, strict=True)
        ):
            identifier = "cu" + str(month)[2:4] + str(month)[5:7]
            rows.append((day, identifier, maturity, price, 100 + j, 1000 + j))
    return ShfeCuPanel(
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        [r[4] for r in rows],
        [r[5] for r in rows],
        dates,
        provenance={"synthetic": True, "archives": [{"sha256": "a" * 64, "year": 2022}]},
    )


@pytest.fixture
def config():
    return study.EmpiricalConfig(
        windows=(3, 5),
        min_direct_count=2,
        origin_stride=2,
        bootstrap_replications=40,
        bootstrap_block=3,
        sensitivities=False,
    )


@pytest.fixture
def fast_source_fit(monkeypatch):
    """Exercise the workflow without repeating the separately tested optimizer."""
    calls = []

    def fit(coefficients, times, **options):
        values = np.asarray(coefficients)
        times = np.asarray(times)
        assert options["drift"] == "zero"
        calls.append((values.copy(), times.copy(), options))
        scale = 1e5 + float(np.mean(np.diff(values, axis=0) ** 2))
        covariance = np.diag(scale * np.arange(1, values.shape[1] + 1))
        return EmpiricalSourceFit(
            covariance,
            np.zeros(values.shape[1]),
            covariance,
            np.zeros((len(values) - 1, values.shape[1])),
            {
                "converged": True,
                "covariance_error_bound": 1e-7,
                "relative_covariance_error_bound": 1e-12,
                "transport_adjustment": options["transport_adjustment"],
                "scope": "synthetic workflow fixture",
            },
        )

    monkeypatch.setattr(study, "fit_source_covariance", fit)
    return calls


def _changed_panel(panel, mask):
    prices = np.array(panel.settlement)
    prices[mask] += 1e6 * (1 + np.arange(np.count_nonzero(mask)) % 7)
    return ShfeCuPanel(
        panel.dates,
        panel.contract_ids,
        panel.contract_month_dates,
        prices,
        panel.volume,
        panel.open_interest,
        panel.session_dates,
        panel.provenance,
    )


def test_validation_window_is_frozen_before_test_and_ignores_future_prices(
    synthetic_panel,
    config,
    fast_source_fit,
):
    daily = study.prepare_daily_curves(synthetic_panel, config)
    selected, audit = study.select_window(daily, config)
    assert selected in config.windows
    assert audit["selection_last_outcome"] <= "2023-12-31"
    assert all(
        "2023-01-01" <= r["origin"] <= r["scored"] <= "2023-12-31" for r in audit["boundaries"]
    )
    modified = _changed_panel(synthetic_panel, synthetic_panel.dates > np.datetime64("2023-12-31"))
    other = study.prepare_daily_curves(modified, config)
    other_selected, other_audit = study.select_window(other, config)
    assert selected == other_selected
    assert audit == other_audit
    baseline_selected, baseline_audit = study.select_baseline_windows(daily, config)
    other_baseline_selected, other_baseline_audit = study.select_baseline_windows(other, config)
    assert baseline_selected == other_baseline_selected
    assert baseline_audit == other_baseline_audit
    assert baseline_audit["selection_last_outcome"] <= "2023-12-31"
    initial_times = daily["times"][daily["dates"] <= np.datetime64(config.validation_end)]
    assert all(times[-1] <= initial_times[-1] for _, times, _ in fast_source_fit)


def test_forecast_issuance_cannot_read_future_curves_or_outcomes(
    synthetic_panel,
    config,
    fast_source_fit,
):
    daily = study.prepare_daily_curves(synthetic_panel, config)
    origin = int(np.flatnonzero(daily["dates"] > np.datetime64("2023-12-31"))[2])
    issued = study.issue_risk_forecast(
        daily, origin, 5, config, next_date=daily["dates"][origin + 1]
    )
    modified = copy.deepcopy(daily)
    modified["coefficients"][origin + 1 :] = 1e12
    for section in (
        modified["sections"][origin + 1 :] + modified["settlement_sections"][origin + 1 :]
    ):
        section["prices"] = np.full(len(section["prices"]), 1e10)
    repeated = study.issue_risk_forecast(
        modified, origin, 5, config, next_date=daily["dates"][origin + 1]
    )
    assert issued["variances"] == repeated["variances"]
    assert issued["contract_ids"] == repeated["contract_ids"]
    np.testing.assert_array_equal(issued["origin_prices"], repeated["origin_prices"])
    assert issued["training_end"] == issued["origin"]
    assert all(times[-1] <= daily["times"][origin] for _, times, _ in fast_source_fit)


def test_same_contract_transport_identity_and_single_annualization(synthetic_panel, config):
    daily = study.prepare_daily_curves(synthetic_panel, config)
    origin = 15
    next_date = daily["dates"][origin + 1]
    source = np.diag([120000.0, 80000.0, 50000.0])
    fit = EmpiricalSourceFit(
        source,
        np.zeros(3),
        source,
        np.zeros((2, 3)),
        {"converged": True, "transport_adjustment": True, "covariance_error_bound": 1e-8},
    )
    issued = study.issue_risk_forecast(
        daily, origin, 5, config, next_date=next_date, source_fit=fit
    )
    chosen = study.select_contracts(daily["sections"][origin], daily["dates"][origin])
    model = LaguerreTransport(3)
    start_tenors = (chosen["maturity"] - daily["dates"][origin]).astype(int) / 365.25
    end_tenors = (chosen["maturity"] - next_date).astype(int) / 365.25
    delta = float((next_date - daily["dates"][origin]).astype(int)) / 365.25
    np.testing.assert_allclose(
        model.evaluate(end_tenors) @ model.matrix(delta), model.evaluate(start_tenors), atol=2e-15
    )
    covariance = (
        model.evaluate(end_tenors)
        @ model.cumulative_covariance(source, delta, method="block")
        @ model.evaluate(end_tenors).T
    )
    for name, exposure in study.EXPOSURES.items():
        expected = config.multiplier**2 * exposure @ covariance @ exposure
        assert issued["variances"][name] == pytest.approx(expected, rel=1e-10)
        assert issued["risk_numerical_error_bounds"][name] >= 0


def test_outcome_anchor_is_actual_origin_price_and_missing_legs_are_separate(config):
    forecast = {
        "contract_ids": ["cu2403", "cu2406", "cu2409"],
        "origin_prices": np.array([100.0, 200.0, 300.0]),
        "fitted_prices": np.array([90.0, 170.0, 350.0]),
    }
    partial = {"ids": np.array(["cu2403", "cu2406"]), "prices": np.array([103.0, 205.0])}
    scores = study.score_issued_forecast(forecast, partial, config)
    assert scores["outright_3m"] == 225
    assert scores["spread_3m_6m"] == 100
    assert scores["butterfly_3m_6m_9m"] is None
    complete = {
        "ids": np.array(forecast["contract_ids"]),
        "prices": np.array([103.0, 205.0, 306.0]),
    }
    assert study.score_issued_forecast(forecast, complete, config)["butterfly_3m_6m_9m"] == 25
    assert (
        study.score_issued_forecast(forecast, {"ids": np.array([]), "prices": np.array([])}, config)
        is None
    )


def test_direct_history_uses_same_ids_adjacent_quotes_and_actual_calendar_gaps():
    dates = np.array(
        ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], dtype="datetime64[D]"
    )
    raw = [
        {"ids": np.array(["cu2403", "cu2406"]), "prices": np.array([100.0, 50.0])},
        {"ids": np.array(["cu2406"]), "prices": np.array([51.0])},
        {"ids": np.array(["cu2403", "cu2406"]), "prices": np.array([106.0, 53.0])},
        {"ids": np.array(["cu2403", "cu2406"]), "prices": np.array([108.0, 56.0])},
    ]
    daily = {
        "dates": dates,
        "times": (dates - dates[0]).astype(int) / 365.25,
        "sections": [{"ids": np.array([]), "prices": np.array([])}] * 4,
        "settlement_sections": raw,
    }
    missing, count = study._historical_contract_covariance(daily, 3, ["cu2403"], 3, 2)
    assert missing is None and count == 1  # Jan4→Jan8 must not bridge the missing Jan5 quote.
    covariance, count = study._historical_contract_covariance(daily, 3, ["cu2406"], 3, 2)
    assert count == 3
    assert covariance[0, 0] == pytest.approx(365.25 * (1 + 4 / 3 + 9) / 3)


def test_missing_curve_is_not_dropped_into_a_longer_coefficient_change(synthetic_panel, config):
    daily = study.prepare_daily_curves(synthetic_panel, config)
    daily["coefficients"][10] = np.nan
    with pytest.raises(ValueError, match="no gap bridging"):
        study._source_fit(daily, 12, 5, config)


def test_curve_reconstruction_preserves_actual_quote_sections(synthetic_panel, config):
    daily = study.prepare_daily_curves(synthetic_panel, config)
    first = daily["sections"][0]
    for identifier, price in zip(first["ids"], first["prices"], strict=True):
        raw = synthetic_panel.settlement[
            (synthetic_panel.dates == daily["dates"][0])
            & (synthetic_panel.contract_ids == identifier)
        ]
        assert len(raw) == 1 and price == raw[0]
    assert daily["diagnostics"][0]["rmse_cny_per_tonne"] > 0
    model_prices = (
        LaguerreTransport(3).evaluate((first["maturity"] - daily["dates"][0]).astype(int) / 365.25)
        @ daily["coefficients"][0]
    )
    assert np.max(np.abs(model_prices - first["prices"])) > 0.1


def test_quadratic_score_identity_and_qlike_proper_scale_behavior():
    forecast = np.array([0.0, 2.0, 8.0])
    proxy = np.array([0.0, 9.0, 1.0])
    np.testing.assert_allclose(
        study.quadratic_risk_score(forecast, proxy), (forecast - proxy) ** 2 - proxy**2
    )
    assert study.qlike(0.0, 0.0, floor=25.0) == np.log(25.0)
    # Both scores are minimized at the conditional second moment, without a Gaussian assumption.
    outcomes = np.array([1.0, 9.0])
    assert np.mean(study.qlike(5.0, outcomes, floor=0.1)) < np.mean(
        study.qlike(2.0, outcomes, floor=0.1)
    )
    assert np.mean(study.quadratic_risk_score(5.0, outcomes)) < np.mean(
        study.quadratic_risk_score(8.0, outcomes)
    )
    np.testing.assert_allclose(
        study.qlike(25 * forecast, 25 * proxy, floor=25) - study.qlike(forecast, proxy, floor=1),
        np.log(25),
    )


@pytest.mark.parametrize(
    "variance,proxy,floor", [(-1, 1, 1), (1, -1, 1), (np.nan, 1, 1), (1, np.inf, 1), (1, 1, 0)]
)
def test_qlike_rejects_invalid_inputs(variance, proxy, floor):
    with pytest.raises(ValueError):
        study.qlike(variance, proxy, floor=floor)


def test_block_bootstrap_is_paired_deterministic_and_not_iid_mcse():
    differences = np.repeat(np.array([-2.0, 2.0, -1.0, 1.0]), 12)
    first = study.paired_block_summary(differences, block=10, replications=500, seed=91)
    second = study.paired_block_summary(differences, block=10, replications=500, seed=91)
    assert first == second
    assert first["mean"] == np.mean(differences)
    assert first["bootstrap_standard_error"] > np.std(differences, ddof=1) / np.sqrt(
        len(differences)
    )
    constant = study.paired_block_summary(np.ones(12), block=4, replications=30, seed=1)
    assert constant["bootstrap_standard_error"] == 0
    assert constant["bootstrap_percentile_95"] == [1, 1]


def test_aggregate_summary_pairs_by_origin_not_by_individual_contract(config):
    records = []
    for i in range(6):
        for method, variance in (("direct", 25 + i), ("spde_local", 30 + i)):
            for exposure in study.EXPOSURES:
                records.append(
                    {
                        "origin": f"2024-01-{i + 1:02d}",
                        "year": "2024",
                        "method": method,
                        "exposure": exposure,
                        "variance": variance,
                        "squared_pnl": 40 + 2 * i,
                        "signed_pnl": np.sqrt(40 + 2 * i),
                    }
                )
    summary = study.summarize_predictions(records, config)
    overall = [r for r in summary["paired_qlike"] if r["year"] == "all"]
    assert len(overall) == len(study.EXPOSURES)
    for row in overall:
        assert row["count"] == 6
        expected = np.mean(
            [
                study.qlike(30 + i, 40 + 2 * i, floor=25)
                - study.qlike(25 + i, 40 + 2 * i, floor=25)
                for i in range(6)
            ]
        )
        assert row["mean"] == pytest.approx(expected)


def _all_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(v) for v in value))
    return set()


def test_local_run_exports_only_aggregate_evidence(
    tmp_path,
    synthetic_panel,
    config,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    local = tmp_path / "local"
    summary = study.run_empirical_study(synthetic_panel, config, local)
    assert summary["selection"]["selection_last_outcome"] <= "2023-12-31"
    assert summary["test_period"][0] >= "2024-01-01"
    predictions = json.loads((local / "private_predictions.json").read_text())
    audit = json.loads((local / "private_forecast_audit.json").read_text())
    assert predictions and audit
    assert all(r["training_end"] <= r["origin"] < r["next_date"] for r in audit)
    forbidden = {
        "origin",
        "contract_ids",
        "origin_prices",
        "settlement",
        "signed_pnl",
        "squared_pnl",
        "coefficients",
        "session_dates",
    }
    assert not forbidden.intersection(_all_keys(summary))
    destination = tmp_path / "public"
    study.export_empirical_evidence(local, destination)
    assert not any(path.name.startswith("private_") for path in destination.iterdir())
    assert {p.name for p in destination.glob("*.json")} == {
        "summary.json",
        "manifest.json",
        "evidence-sha256.json",
    }
    for path in destination.glob("*.json"):
        if path.name != "evidence-sha256.json":
            assert not forbidden.intersection(_all_keys(json.loads(path.read_text())))
    checksums = json.loads((destination / "evidence-sha256.json").read_text())
    assert checksums
    for name, digest in checksums.items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    for name in ("risk-calibration.png", "paired-risk-scores.png", "standardized-pnl.png"):
        assert (destination / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FileExistsError):
        study.run_empirical_study(synthetic_panel, config, local)
    with pytest.raises(FileExistsError):
        study.export_empirical_evidence(local, destination)


def test_all_test_spde_failures_preserve_baselines_and_export(
    tmp_path,
    synthetic_panel,
    config,
    monkeypatch,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    original_issue = study.issue_risk_forecast

    def fail_test_only(daily, origin, *args, **kwargs):
        if daily["dates"][origin] > np.datetime64(config.validation_end):
            raise RuntimeError("declared test-period SPDE failure")
        return original_issue(daily, origin, *args, **kwargs)

    monkeypatch.setattr(study, "issue_risk_forecast", fail_test_only)
    local = tmp_path / "all-spde-failed-local"
    public = tmp_path / "all-spde-failed-public"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = study.run_empirical_study(synthetic_panel, config, local)
        study.export_empirical_evidence(local, public)
    assert summary["selection"]["selection_last_outcome"] <= config.validation_end
    assert summary["numerical_diagnostics"]["available"] is False
    assert summary["numerical_diagnostics"]["all_converged"] is False
    assert summary["mean_source_covariance"] is None
    assert summary["source_eigenvalue_quantiles"] is None
    assert summary["failure_count"] > 0
    assert "declared test-period SPDE failure" in summary["failure_reasons"]
    assert {r["method"] for r in summary["risk_summary"]} == {"direct", "shrinkage"}
    for row in summary["availability"]:
        assert row["attempted"] == summary["attempted_test_origin_count"] > 0
        if row["method"] in {"direct", "shrinkage"}:
            assert row["scored"] > 0
        else:
            assert row["issued"] == row["scored"] == 0
    for row in summary["standardized_pnl_histograms"]:
        assert row["count"] == sum(row["counts"]) == row["outside_count"] == 0
        assert row["mean"] is row["second_moment"] is row["fourth_moment"] is None
    assert (public / "report.md").is_file()
    assert (public / "standardized-pnl.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(public.glob("private_*"))
