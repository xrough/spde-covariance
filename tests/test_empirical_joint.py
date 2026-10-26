"""Joint residual risk, diffusion clocks and genuinely past-only hedge decisions."""

import copy
import json

import numpy as np
import pytest

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_joint import (
    JointRiskConfig,
    bounded_minimum_variance_hedge,
    daily_training_prefix,
    diffusion_duration,
    issue_joint_risk_forecast,
    joint_training_statistics,
    run_joint_risk_study,
    score_joint_risk_forecast,
    select_joint_policies,
)
from spde_covariance.empirical_study import EmpiricalConfig, prepare_daily_curves, select_contracts
from spde_covariance.laguerre import LaguerreTransport


@pytest.fixture(scope="module")
def panel():
    months = np.arange(np.datetime64("2022-01", "M"), np.datetime64("2026-01", "M"))
    dates = months.astype("datetime64[D]") + np.timedelta64(3, "D")
    model = LaguerreTransport(3)
    rows = []
    for i, day in enumerate(dates):
        contracts = months[i] + np.arange(1, 14).astype("timedelta64[M]")
        maturity = contracts.astype("datetime64[D]") + np.timedelta64(14, "D")
        states = np.array(
            [60000 + 70 * i + 100 * np.sin(i), 450 * np.cos(i / 2), 200 * np.sin(i / 3)]
        )
        prices = model.evaluate((maturity - day).astype(int) / 365.25) @ states + 25 * np.sin(
            np.arange(13) + i
        )
        for month, end, price in zip(contracts, maturity, prices, strict=True):
            rows.append((day, "cu" + str(month)[2:4] + str(month)[5:7], end, price))
    return ShfeCuPanel(
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        np.ones(len(rows)),
        np.ones(len(rows)),
        dates,
        provenance={"synthetic": True},
    )


@pytest.fixture
def config():
    return EmpiricalConfig(
        windows=(3, 5),
        min_direct_count=2,
        bootstrap_block=2,
        bootstrap_replications=30,
        sensitivities=False,
    )


@pytest.fixture
def options():
    return JointRiskConfig(include_source=False)


def test_physical_transport_is_unchanged_by_diffusion_clock(panel, config, options):
    daily = prepare_daily_curves(panel, config)
    training = daily_training_prefix(daily, 20)
    chosen = select_contracts(training["sections"][-1], training["dates"][-1])
    calendar = joint_training_statistics(training, chosen, 5, clock="calendar")
    session = joint_training_statistics(training, chosen, 5, clock="session")
    np.testing.assert_array_equal(calendar["factor_changes"], session["factor_changes"])
    np.testing.assert_array_equal(calendar["residual_changes"], session["residual_changes"])
    np.testing.assert_array_equal(calendar["physical_durations"], session["physical_durations"])
    np.testing.assert_allclose(session["diffusion_durations"], 1 / 252)
    assert np.unique(calendar["diffusion_durations"]).size > 1
    assert calendar["exact_transport_identity_max_abs"] < 1e-9
    next_date = daily["dates"][21]
    one = issue_joint_risk_forecast(
        training,
        next_date=next_date,
        window=5,
        config=config,
        clock="calendar",
        joint_config=options,
    )
    two = issue_joint_risk_forecast(
        training,
        next_date=next_date,
        window=5,
        config=config,
        clock="session",
        joint_config=options,
    )
    assert one["physical_duration"] == two["physical_duration"]
    assert one["diffusion_duration"] != two["diffusion_duration"]


def test_joint_psd_and_cross_terms_are_retained_exactly(panel, config, options):
    daily = prepare_daily_curves(panel, config)
    training = daily_training_prefix(daily, 20)
    forecast = issue_joint_risk_forecast(
        training, next_date=daily["dates"][21], window=5, config=config, joint_config=options
    )
    covariance = forecast["joint_covariance"]
    assert covariance.shape == (6, 6)
    assert np.linalg.eigvalsh(covariance)[0] > -1e-6
    assert np.linalg.norm(covariance[:3, 3:]) > 0
    assert forecast["diagnostics"]["joint_psd"]["psd_within_roundoff"]
    for name, value in forecast["variances"].items():
        pieces = forecast["risk_components"][name]
        assert sum(pieces.values()) == pytest.approx(value, abs=1e-7)
    assert any(abs(r["cross"]) > 1 for r in forecast["risk_components"].values())
    with pytest.raises(ValueError):
        covariance.setflags(write=True)


def test_joint_direct_and_factor_use_identical_complete_cases(panel, config, options):
    daily = prepare_daily_curves(panel, config)
    training = daily_training_prefix(daily, 20)
    training = copy.deepcopy(training)
    training["coefficients"][18] = np.nan
    training["settlement_sections"][15] = {
        "ids": np.array([]),
        "prices": np.array([]),
        "maturity": np.array([], dtype="datetime64[D]"),
    }
    forecasts = [
        issue_joint_risk_forecast(
            training,
            next_date=daily["dates"][21],
            window=5,
            config=config,
            joint_config=options,
            method=method,
        )
        for method in ("factor", "joint", "direct")
    ]
    assert all(forecast is not None for forecast in forecasts)
    assert len({forecast["diagnostics"]["training_count"] for forecast in forecasts}) == 1
    assert forecasts[0]["diagnostics"]["training_count"] == 2
    assert forecasts[0]["diagnostics"]["unavailable_curves"] == 2
    assert forecasts[0]["diagnostics"]["unavailable_quotes"] == 1


def test_future_observations_cannot_change_forecast_or_issued_hedge(panel, config, options):
    daily = prepare_daily_curves(panel, config)
    origin = 30
    prefix = daily_training_prefix(daily, origin)
    issued = issue_joint_risk_forecast(
        prefix, next_date=daily["dates"][origin + 1], window=5, config=config, joint_config=options
    )
    altered = copy.deepcopy(daily)
    altered["coefficients"][origin + 1 :] = 1e15
    for section in altered["sections"][origin + 1 :] + altered["settlement_sections"][origin + 1 :]:
        section["prices"] = np.full(len(section["prices"]), 1e14)
    repeated = issue_joint_risk_forecast(
        daily_training_prefix(altered, origin),
        next_date=daily["dates"][origin + 1],
        window=5,
        config=config,
        joint_config=options,
    )
    np.testing.assert_array_equal(issued["hedge_weights"], repeated["hedge_weights"])
    np.testing.assert_array_equal(issued["covariance"], repeated["covariance"])
    assert issued["training_end"] == issued["origin"]
    first_score = score_joint_risk_forecast(
        issued, daily["settlement_sections"][origin + 1], config
    )
    changed_score = score_joint_risk_forecast(
        issued, altered["settlement_sections"][origin + 1], config
    )
    assert first_score != changed_score


def test_validation_selection_is_frozen_before_development_outcomes(panel, config, options):
    daily = prepare_daily_curves(panel, config)
    policies, audit = select_joint_policies(daily, config, joint_config=options)
    altered = copy.deepcopy(daily)
    mask = altered["dates"] > np.datetime64(config.validation_end)
    altered["coefficients"][mask] = 1e11
    for i in np.flatnonzero(mask):
        altered["sections"][i]["prices"] = altered["sections"][i]["prices"] * 17
        altered["settlement_sections"][i]["prices"] = (
            altered["settlement_sections"][i]["prices"] * 19
        )
    repeated, repeated_audit = select_joint_policies(altered, config, joint_config=options)
    assert policies == repeated
    assert audit == repeated_audit
    assert all(a["selection_last_outcome"] <= config.validation_end for a in audit.values())
    assert len(audit["joint"]["candidates"]) == len(config.windows) * 2 * 2


def test_missing_outcome_only_invalidates_required_exposure_legs(config):
    forecast = {
        "contract_ids": np.array(["a", "b", "c"]),
        "origin_prices": np.array([100.0, 110.0, 120.0]),
        "hedge_weights": np.array([1.0, -1.0, 0.0]),
    }
    scored = score_joint_risk_forecast(
        forecast, {"ids": np.array(["a", "b"]), "prices": np.array([102.0, 115.0])}, config
    )
    assert scored["squared_pnl"]["outright_3m"] == 100
    assert scored["squared_pnl"]["spread_3m_6m"] == 225
    assert scored["squared_pnl"]["butterfly_3m_6m_9m"] is None
    assert scored["hedge_pnl"] == -15


def test_hedge_has_known_solution_and_respects_bounds():
    matrix = np.array([[2.0, 1.0, 0.5], [1.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
    result = bounded_minimum_variance_hedge(matrix)
    np.testing.assert_allclose(result["hedge_positions"], [-1, -0.5])
    assert result["predicted_second_moment_price_units"] == pytest.approx(0.75)
    bounded = bounded_minimum_variance_hedge(matrix, bounds=(-0.2, 0.2))
    np.testing.assert_allclose(bounded["hedge_positions"], [-0.2, -0.2])
    assert bounded["bound_hits"] == 2
    zero = bounded_minimum_variance_hedge(np.zeros((3, 3)))
    np.testing.assert_array_equal(zero["hedge_positions"], [0, 0])
    with pytest.raises(ValueError, match="PSD"):
        bounded_minimum_variance_hedge(np.diag([1.0, -1.0, 1.0]))


def test_hedge_agrees_with_independent_convex_reference():
    cvxpy = pytest.importorskip("cvxpy")
    generator = np.random.default_rng(17)
    for rank in (1, 2, 3):
        features = generator.normal(size=(3, rank))
        covariance = features @ features.T
        computed = bounded_minimum_variance_hedge(covariance, bounds=(-0.7, 0.9))
        h = cvxpy.Variable(2)
        weights = cvxpy.hstack([1, h])
        objective = cvxpy.quad_form(weights, cvxpy.psd_wrap(covariance))
        problem = cvxpy.Problem(cvxpy.Minimize(objective), [h >= -0.7, h <= 0.9])
        problem.solve(solver="SCS", eps=1e-9, max_iters=20000)
        assert computed["predicted_second_moment_price_units"] == pytest.approx(
            problem.value, abs=1e-7
        )
        assert computed["objective_enumeration_gap"] <= 1e-10


def test_end_to_end_aggregates_reproducible_and_no_raw_data(panel, config, options):
    first = run_joint_risk_study(panel, config, joint_config=options)
    second = run_joint_risk_study(panel, config, joint_config=options)
    assert first == second
    assert first["risk_summary"] and first["hedge_summary"] and first["hedge_paired"]
    assert first["chronology"]["no_untouched_test_claim"]
    assert "joint_calendar" in first["evaluation_policies"]
    assert "joint_session" in first["evaluation_policies"]
    assert all(
        row["all_forecast_psd_within_roundoff"] for row in first["numerical_diagnostics"].values()
    )
    encoded = json.dumps(first)
    for forbidden in (
        "origin_prices",
        "contract_ids",
        "hedge_weights",
        "private_predictions",
        "cu2403",
    ):
        assert forbidden not in encoded
    # Only configuration/selection boundary dates, no individual outcome arrays.
    assert all("squared_pnl" not in row for row in first["risk_summary"])


def test_insufficient_history_is_unavailable_without_gap_bridging(panel, config, options):
    daily = prepare_daily_curves(panel, config)
    prefix = daily_training_prefix(daily, 20)
    changed = copy.deepcopy(prefix)
    changed["coefficients"][-4:-1] = np.nan
    result = issue_joint_risk_forecast(
        changed, next_date=daily["dates"][21], window=3, config=config, joint_config=options
    )
    assert result is None
    with pytest.raises(ValueError, match="next_date"):
        issue_joint_risk_forecast(
            prefix, next_date=prefix["dates"][-1], window=3, config=config, joint_config=options
        )


@pytest.mark.parametrize(
    "values",
    [
        {"clocks": ()},
        {"clocks": ("business",)},
        {"shrinkages": (1.2,)},
        {"session_year": 0},
        {"hedge_bounds": (2.0, 3.0)},
        {"rank_rtol": 0},
    ],
)
def test_invalid_joint_options_rejected(values):
    with pytest.raises(ValueError):
        JointRiskConfig(**values)


def test_diffusion_duration_rejects_invalid_dates_and_clock():
    assert diffusion_duration("2024-01-05", "2024-01-08", "calendar") == 3 / 365.25
    assert diffusion_duration("2024-01-05", "2024-01-08", "session") == 1 / 252
    with pytest.raises(ValueError):
        diffusion_duration("2024-01-08", "2024-01-05", "calendar")
    with pytest.raises(ValueError):
        diffusion_duration("2024-01-05", "2024-01-08", "other")


def test_joint_cross_cancellation_recovers_direct_in_constant_basis(config, options):
    dates = np.arange(np.datetime64("2024-01-03"), np.datetime64("2024-02-03"))
    maturity = np.array(["2024-04-15", "2024-07-15", "2024-10-15"], dtype="datetime64[D]")
    ids = np.array(["cu2404", "cu2407", "cu2410"])
    oscillation = np.sin(np.arange(len(dates)))
    coefficients = (60000 + 20 * oscillation)[:, None]
    sections = [
        {
            "ids": ids,
            "maturity": maturity,
            "prices": 60000 + oscillation[i] * np.array([1.0, 2.0, 3.0]),
        }
        for i in range(len(dates))
    ]
    daily = {
        "dates": dates,
        "coefficients": coefficients,
        "sections": sections,
        "settlement_sections": sections,
    }
    issued = issue_joint_risk_forecast(
        daily, next_date="2024-02-03", window=20, config=config, joint_config=options
    )
    direct = issue_joint_risk_forecast(
        daily,
        next_date="2024-02-03",
        window=20,
        config=config,
        joint_config=options,
        method="direct",
    )
    np.testing.assert_allclose(issued["covariance"], direct["covariance"], atol=1e-10)
    parts = issued["risk_components"]["outright_3m"]
    assert parts["cross"] < 0
    assert parts["factor"] + parts["residual"] > 100 * sum(parts.values())
    assert sum(parts.values()) == pytest.approx(issued["variances"]["outright_3m"])


def test_source_control_matches_existing_original_pipeline(panel, config):
    from spde_covariance.empirical_study import issue_risk_forecast

    daily = prepare_daily_curves(panel, config)
    origin = 20
    issued = issue_joint_risk_forecast(
        daily_training_prefix(daily, origin),
        next_date=daily["dates"][origin + 1],
        window=5,
        config=config,
        method="source",
    )
    original = issue_risk_forecast(daily, origin, 5, config, next_date=daily["dates"][origin + 1])
    for name in original["variances"]:
        assert issued["variances"][name] == pytest.approx(original["variances"][name], rel=1e-12)
    assert "full coefficient" in issued["diagnostics"]["sample_scope"]


def test_ties_prefer_longer_window_calendar_and_less_shrinkage(panel, config, options, monkeypatch):
    import spde_covariance.empirical_joint as joint

    daily = prepare_daily_curves(panel, config)
    original = joint.issue_joint_risk_forecast

    def tied(*args, **kwargs):
        issued = original(*args, **kwargs)
        if issued is None:
            return None
        return dict(issued, variances={name: 100.0 for name in issued["variances"]})

    monkeypatch.setattr(joint, "issue_joint_risk_forecast", tied)
    selected, _ = select_joint_policies(daily, config, joint_config=options)
    assert all(
        policy["window"] == max(config.windows)
        and policy["clock"] == "calendar"
        and policy["shrinkage"] == 0
        for policy in selected.values()
    )
