"""Matched mechanisms, chronology and aggregate-only attribution evidence."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from spde_covariance import empirical_attribution as attribution
from spde_covariance import empirical_study as study
from spde_covariance.empirical_calibration import EmpiricalSourceFit
from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.laguerre import LaguerreTransport


@pytest.fixture(scope="module")
def panel():
    months = np.arange(np.datetime64("2022-01", "M"), np.datetime64("2026-01", "M"))
    dates = months.astype("datetime64[D]") + np.timedelta64(3, "D")
    model = LaguerreTransport(3)
    rows = []
    for i, day in enumerate(dates):
        contracts = months[i] + np.arange(1, 14).astype("timedelta64[M]")
        maturities = contracts.astype("datetime64[D]") + np.timedelta64(14, "D")
        state = [60000 + 45 * i + 180 * np.sin(i), 700 * np.cos(i / 4), -250]
        prices = model.evaluate((maturities - day).astype(int) / 365.25) @ state
        prices += 4 * np.cos(np.arange(13) * 1.7 + i)
        for month, maturity, price in zip(contracts, maturities, prices, strict=True):
            rows.append((day, "cu" + str(month)[2:4] + str(month)[5:7], maturity, price))
    return ShfeCuPanel(
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        np.full(len(rows), 100),
        np.full(len(rows), 1000),
        dates,
        provenance={"synthetic": True, "raw_prices": [123456.789]},
    )


@pytest.fixture
def config():
    return study.EmpiricalConfig(
        windows=(3, 5),
        min_direct_count=2,
        origin_stride=4,
        bootstrap_replications=20,
        bootstrap_block=3,
        sensitivities=False,
    )


@pytest.fixture
def fake_fit(monkeypatch):
    calls = []

    def fit(coefficients, times, **options):
        values, times = np.asarray(coefficients), np.asarray(times)
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
            },
        )

    monkeypatch.setattr(study, "fit_source_covariance", fit)
    return calls


def test_maturity_masks_are_nested_deterministic_and_price_independent():
    for size in range(25):
        full = attribution.maturity_mask(size, "all")
        alternate = attribution.maturity_mask(size, "alternate")
        sparse = attribution.maturity_mask(size, "sparse4")
        assert set(sparse) <= set(alternate) <= set(full)
        assert len(sparse) <= 4
        assert len(set(sparse)) == len(sparse)
        np.testing.assert_array_equal(sparse, attribution.maturity_mask(size, "sparse4"))
        if len(alternate) > 4:
            assert len(sparse) == 4 and sparse[0] == alternate[0] and sparse[-1] == alternate[-1]
    for value in (-1, 2.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            attribution.maturity_mask(value, "all")
    with pytest.raises(ValueError, match="mask"):
        attribution.maturity_mask(4, "optimized")


def test_masking_changes_factor_information_but_not_exposure_universe(panel, config):
    daily = study.prepare_daily_curves(panel, config)
    original = daily["coefficients"].copy()
    for mask in attribution.MASKS:
        view = attribution.prepare_masked_curves(daily, config, mask)
        assert view["sections"] is daily["sections"]
        assert view["settlement_sections"] is daily["settlement_sections"]
        for origin in (10, 25, 40):
            expected = study.select_contracts(daily["sections"][origin], daily["dates"][origin])
            actual = study.select_contracts(view["sections"][origin], view["dates"][origin])
            np.testing.assert_array_equal(actual["ids"], expected["ids"])
        if mask != "all":
            assert not np.array_equal(view["coefficients"], original)
    np.testing.assert_array_equal(daily["coefficients"], original)


def test_mask_uses_each_cross_section_independently(panel, config):
    daily = study.prepare_daily_curves(panel, config)
    origin = 26
    view = attribution.prepare_masked_curves(daily, config, "sparse4")
    changed = copy.deepcopy(daily)
    for section in changed["sections"][origin + 1 :]:
        section["prices"] *= 100
    other = attribution.prepare_masked_curves(changed, config, "sparse4")
    np.testing.assert_array_equal(
        view["coefficients"][: origin + 1], other["coefficients"][: origin + 1]
    )


def test_actual_issuer_cannot_read_future_states_or_prices(panel, config, fake_fit):
    daily = study.prepare_daily_curves(panel, config)
    origin = 30
    issued = attribution.issue_attribution_forecasts(
        daily, origin, 5, config, next_date=daily["dates"][origin + 1]
    )
    changed = copy.deepcopy(daily)
    changed["coefficients"][origin + 1 :] = 1e12
    for section in changed["sections"][origin + 1 :] + changed["settlement_sections"][origin + 1 :]:
        section["prices"] *= 100
    repeated = attribution.issue_attribution_forecasts(
        changed, origin, 5, config, next_date=daily["dates"][origin + 1]
    )
    assert issued["methods"] == repeated["methods"]
    assert issued["contract_ids"] == repeated["contract_ids"]
    assert issued["training_end"] == issued["origin"]
    assert issued["training_intervals"] == 5
    assert all(times[-1] <= daily["times"][origin] for _, times, _ in fake_fit)
    np.testing.assert_array_equal(fake_fit[0][0], fake_fit[1][0])
    np.testing.assert_array_equal(fake_fit[0][1], fake_fit[1][1])
    assert fake_fit[0][2]["transport_adjustment"] is True
    assert fake_fit[1][2]["transport_adjustment"] is False


def test_matched_baseline_is_same_window_raw_contract_estimator(panel, config, fake_fit):
    daily = study.prepare_daily_curves(panel, config)
    origin, window = 30, 5
    issued = attribution.issue_attribution_forecasts(
        daily, origin, window, config, next_date=daily["dates"][origin + 1]
    )
    chosen = study.select_contracts(daily["sections"][origin], daily["dates"][origin])
    dt = daily["times"][origin + 1] - daily["times"][origin]
    for name, a in study.EXPOSURES.items():
        cov, count = study._historical_contract_covariance(
            daily, origin, chosen["ids"][a != 0], window, config.min_direct_count
        )
        assert issued["direct_joint_counts"][name] == count
        expected = config.multiplier**2 * dt * a[a != 0] @ cov @ a[a != 0]
        assert issued["methods"]["direct"][name] == pytest.approx(expected)
    masked = attribution.prepare_masked_curves(daily, config, "sparse4")
    other = attribution.issue_attribution_forecasts(
        masked, origin, window, config, next_date=daily["dates"][origin + 1]
    )
    assert issued["methods"]["direct"] == other["methods"]["direct"]
    assert issued["methods"]["shrinkage"] == other["methods"]["shrinkage"]


def test_missing_future_leg_does_not_remove_unheld_exposures(panel, config, fake_fit):
    daily = study.prepare_daily_curves(panel, config)
    issued = attribution.issue_attribution_forecasts(
        daily, 30, 5, config, next_date=daily["dates"][31]
    )
    section = {
        "ids": np.array(issued["contract_ids"][:2]),
        "prices": issued["origin_prices"][:2] + [2, 4],
    }
    scores = attribution.score_attribution_forecast(issued, section, config)
    assert scores["outright_3m"] == 100
    assert scores["spread_3m_6m"] == 100
    assert scores["butterfly_3m_6m_9m"] is None


def test_missing_historical_leg_is_not_bridged(panel, config, fake_fit):
    daily = study.prepare_daily_curves(panel, config)
    chosen = study.select_contracts(daily["sections"][30], daily["dates"][30])
    for i in range(25, 31):
        section = daily["settlement_sections"][i]
        keep = section["ids"] != chosen["ids"][-1]
        daily["settlement_sections"][i] = {k: v[keep] for k, v in section.items()}
    issued = attribution.issue_attribution_forecasts(
        daily, 30, 5, config, next_date=daily["dates"][31]
    )
    assert "outright_3m" in issued["methods"]["direct"]
    assert "spread_3m_6m" in issued["methods"]["direct"]
    assert "butterfly_3m_6m_9m" not in issued["methods"]["direct"]
    assert issued["direct_joint_counts"]["butterfly_3m_6m_9m"] == 0


def test_all_method_common_sample_and_exact_paired_score_identities(config):
    rows = [
        {
            "exposure": "outright_3m",
            "year": "2024",
            "proxy": z,
            "variances": dict(zip(attribution.METHODS, [v, v + 1, v + 2, v + 3], strict=True)),
        }
        for z, v in ((40.0, 30.0), (60.0, 45.0), (100.0, 55.0))
    ]
    rows.append(
        {"exposure": "outright_3m", "year": "2024", "proxy": 1e10, "variances": {"direct": 50.0}}
    )
    rows.append(
        {
            "exposure": "outright_3m",
            "year": "2024",
            "proxy": None,
            "variances": dict.fromkeys(attribution.METHODS, 50.0),
        }
    )
    scores, marginal, paired, available = attribution._score_rows(
        rows, window=3, mask="all", config=config, bootstrap_replications=20
    )
    assert all(row["count"] == 3 for row in scores)
    assert next(r for r in marginal if r["method"] == "direct")["count"] == 4
    direct_avail = next(
        r for r in available if r["method"] == "direct" and r["exposure"] == "outright_3m"
    )
    assert direct_avail["candidate_origins"] == 5
    assert direct_avail["issued_count"] == 5 and direct_avail["scored_count"] == 4
    item = next(
        r
        for r in paired
        if r["year"] == "all"
        and r["method"] == "factor_adjusted"
        and r["reference"] == "direct"
        and r["score"] == "quadratic"
    )
    expected = np.mean(
        [
            (v - (v + 2)) * (v + v + 2 - 2 * z)
            for z, v in ((40.0, 30.0), (60.0, 45.0), (100.0, 55.0))
        ]
    )
    assert item["mean"] == pytest.approx(expected)


def test_full_runner_returns_only_aggregate_evidence_and_frozen_protocol(panel, config, fake_fit):
    result = attribution.run_attribution_study(panel, config, bootstrap_replications=20)
    assert result["scope"]["no_test_selection"] is True
    assert result["scope"]["unchanged_original_benchmark"] is True
    assert result["scope"]["settings"]["windows"] == list(config.windows)
    assert len(result["scope"]["configuration_sha256"]) == 64
    assert {r["mask"] for r in result["scores"]} == set(attribution.MASKS)
    assert set(config.windows) <= {r["window"] for r in result["scores"]}
    assert result["scope"]["settings"]["short_windows"] == [16, 32]
    assert result["scope"]["settings"]["all_study_windows"] == [3, 5, 16, 32]
    assert result["scope"]["evaluation_start_after"] == config.validation_end
    assert result["scope"]["evaluation_end"] == config.test_end
    assert all(
        row["reconstruction"]["design_condition_min_median_p95_max"] is not None
        for row in result["representation"]
    )
    assert len(result["numerical_diagnostics"]) == len(attribution.MASKS) * (
        len(config.windows) + 2
    )
    assert all(
        r["relative_covariance_error_bound_max"] <= 1e-5 for r in result["numerical_diagnostics"]
    )
    json.dumps(result, allow_nan=False)

    def inspect_public_structure(value):
        if isinstance(value, dict):
            assert not {"raw_prices", "contract_ids", "origin_prices"}.intersection(value)
            for item in value.values():
                inspect_public_structure(item)
        elif isinstance(value, list):
            for item in value:
                inspect_public_structure(item)
        elif isinstance(value, str):
            assert value != "cu2404"
        elif isinstance(value, (int, float)):
            # Match the private provenance sentinel as a number, never digits
            # that could coincidentally appear inside runtime measurements.
            assert value != 123456.789

    inspect_public_structure(result)
    assert result["scope"]["settings"]["smoothing"] == config.smoothing


def test_deterministic_nonruntime_results(panel, config, fake_fit):
    first = attribution.run_attribution_study(panel, config, bootstrap_replications=20)
    second = attribution.run_attribution_study(panel, config, bootstrap_replications=20)
    first["scope"].pop("runtime_seconds")
    second["scope"].pop("runtime_seconds")
    for result in (first, second):
        for row in result["numerical_diagnostics"]:
            row.pop("runtime_seconds_min_median_p95_max")
    assert first == second


@pytest.mark.parametrize("unavailable", [False, True])
def test_year_groups_follow_outcome_session_including_unavailable_records(
    panel, config, fake_fit, monkeypatch, unavailable
):
    # December 2024 origin scores the next observed January 2025 session.
    config = replace(config, validation_end="2024-12-01", test_end="2025-01-04", origin_stride=1)
    captured = []
    original = attribution._score_rows

    def capture(records, **options):
        captured.extend(records)
        return original(records, **options)

    monkeypatch.setattr(attribution, "_score_rows", capture)
    if unavailable:
        monkeypatch.setattr(attribution, "issue_attribution_forecasts", lambda *a, **kw: None)
    result = attribution.run_attribution_study(panel, config, bootstrap_replications=20)
    assert captured and {row["year"] for row in captured} == {"2025"}
    assert result["scope"]["year_grouping"] == "outcome_session_calendar_year"


def test_unavailable_reconstructions_are_reported_not_imputed(panel, config, fake_fit):
    high_dimension = replace(config, dimension=8)
    result = attribution.run_attribution_study(panel, high_dimension, bootstrap_replications=20)
    sparse = [r for r in result["availability"] if r["mask"] == "sparse4"]
    assert all(r["issued_count"] == 0 for r in sparse if r["method"].startswith("factor"))
    assert any(r["issued_count"] > 0 for r in sparse if r["method"] == "direct")
    assert not any(row["mask"] == "sparse4" for row in result["scores"])
    assert result["representation"][-1]["reconstruction"]["available_sessions"] == 0


def test_short_history_does_not_relax_baseline_sample_requirement(panel, config, fake_fit):
    config = replace(config, min_direct_count=20)
    result = attribution.run_attribution_study(panel, config, bootstrap_replications=20)
    short = [row for row in result["availability"] if row["window"] == 16]
    assert all(row["issued_count"] == 0 for row in short if row["method"] == "direct")
    assert any(row["issued_count"] > 0 for row in short if row["method"] == "factor_adjusted")
    assert not any(row["window"] == 16 for row in result["scores"])
    assert any(row["window"] == 16 for row in result["available_scores"])
    assert any(
        row["window"] == 16 and row["reference"] == "factor_unadjusted"
        for row in result["paired_comparisons"]
    )


def test_real_solver_issuance_is_psd_and_certified(panel, config):
    daily = study.prepare_daily_curves(panel, config)
    issued = attribution.issue_attribution_forecasts(
        daily, 30, 5, config, next_date=daily["dates"][31]
    )
    assert set(issued["methods"]) == set(attribution.METHODS)
    for method in ("factor_adjusted", "factor_unadjusted"):
        assert issued["diagnostics"][method]["converged"]
        assert issued["diagnostics"][method]["relative_covariance_error_bound"] <= 1e-5
        assert all(v >= 0 for v in issued["methods"][method].values())


def test_numerical_failure_does_not_hide_raw_baseline_availability(panel, config, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic convergence failure")

    monkeypatch.setattr(study, "fit_source_covariance", fail)
    daily = study.prepare_daily_curves(panel, config)
    issued = attribution.issue_attribution_forecasts(
        daily, 30, 5, config, next_date=daily["dates"][31]
    )
    assert set(issued["methods"]) == {"direct", "shrinkage"}
    assert issued["unavailable"]["factor_adjusted"] == "numerical_budget_not_met"
    assert issued["unavailable"]["factor_unadjusted"] == "numerical_budget_not_met"
