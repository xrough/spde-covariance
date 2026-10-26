"""Post hoc reconstruction identities, without covariance fitting or raw export."""

import json

import numpy as np
import pytest

from spde_covariance.empirical_diagnostics import reconstruction_risk_decomposition
from spde_covariance.empirical_study import EXPOSURES, EmpiricalConfig
from spde_covariance.laguerre import LaguerreTransport


def _daily(*, shocks=True, residuals=None, dimension=3, sessions=5):
    dates = np.array(
        ["2023-12-27", "2023-12-28", "2024-01-02", "2024-01-05", "2024-01-08"],
        dtype="datetime64[D]",
    )[:sessions]
    model = LaguerreTransport(dimension)
    coefficients = np.zeros((sessions, dimension))
    coefficients[0, : min(dimension, 3)] = [60000, 300, -100][:dimension]
    for i in range(1, sessions):
        dt = float((dates[i] - dates[i - 1]).astype(int)) / 365.25
        coefficients[i] = model.matrix(dt) @ coefficients[i - 1]
        if shocks:
            coefficients[i] += (i + 1) * np.arange(1, dimension + 1)
    maturity = dates[2] + np.array([90, 180, 270]).astype("timedelta64[D]")
    sections = []
    for i, day in enumerate(dates):
        prices = model.evaluate((maturity - day).astype(int) / 365.25) @ coefficients[i]
        if residuals is not None:
            prices += residuals[i]
        sections.append(
            {
                "ids": np.array(["private_a", "private_b", "private_c"]),
                "maturity": maturity.copy(),
                "prices": prices,
            }
        )
    daily = {
        "dates": dates,
        "times": (dates - dates[0]).astype(int) / 365.25,
        "coefficients": coefficients,
        "sections": sections,
        "settlement_sections": [{k: v.copy() for k, v in s.items()} for s in sections],
    }
    config = EmpiricalConfig(windows=(2,), sensitivities=False)
    return daily, config


def _rows(result):
    return {row["exposure"]: row for row in result["exposures"]}


@pytest.mark.parametrize("dimension", [1, 3, 4])
@pytest.mark.parametrize("shocks", [False, True])
def test_exact_basis_changes_have_zero_reconstruction_residual(dimension, shocks):
    daily, config = _daily(dimension=dimension, shocks=shocks)
    result = reconstruction_risk_decomposition(daily, config)
    assert result["scope"]["dimension"] == dimension
    assert result["scope"]["candidate_origin_count"] == 2
    assert result["scope"]["basis_identity_max_abs_error"] < 1e-14
    assert result["scope"]["reconstruction_change_max_abs_error_cny_per_tonne"] < 5e-10
    for row in result["exposures"]:
        assert row["count"] == 2
        assert row["residual_second_moment"] < 1e-17
        assert row["actual_second_moment"] == pytest.approx(row["modeled_second_moment"], abs=1e-7)
        assert row["max_identity_error"] < 1e-8
        if not shocks:
            assert row["actual_second_moment"] < 1e-17


def test_butterfly_idiosyncratic_risk_uses_uncentered_moments_and_cross_term():
    residuals = np.zeros((5, 3))
    residuals[3] = [1, -2, 1]
    residuals[4] = [3, -1, -1]
    daily, config = _daily(shocks=False, residuals=residuals)
    row = _rows(reconstruction_risk_decomposition(daily, config))["butterfly_3m_6m_9m"]
    # Cash changes are 30 and -10, whose second moment is 500, not variance 400.
    assert row["actual_second_moment"] == pytest.approx(500, abs=1e-7)
    assert row["residual_second_moment"] == pytest.approx(500, abs=1e-7)
    assert row["residual_share"] == pytest.approx(1, abs=1e-10)

    daily, config = _daily(shocks=True, residuals=residuals)
    result = reconstruction_risk_decomposition(daily, config)
    model = LaguerreTransport(3)
    for row in result["exposures"]:
        exposure = EXPOSURES[row["exposure"]]
        expected = []
        for i in (2, 3):
            dt = daily["times"][i + 1] - daily["times"][i]
            section = daily["sections"][i]
            evaluation = model.evaluate(
                (section["maturity"] - daily["dates"][i + 1]).astype(int) / 365.25
            )
            r = daily["coefficients"][i + 1] - model.matrix(dt) @ daily["coefficients"][i]
            s = config.multiplier * exposure @ evaluation @ r
            u = config.multiplier * exposure @ (residuals[i + 1] - residuals[i])
            expected.append([(s + u) ** 2, s**2, u**2, 2 * s * u])
        for key, value in zip(
            (
                "actual_second_moment",
                "modeled_second_moment",
                "residual_second_moment",
                "twice_cross_moment",
            ),
            np.mean(expected, axis=0),
            strict=True,
        ):
            assert row[key] == pytest.approx(value, abs=2e-7)
        assert row["modeled_share"] + row["residual_share"] + row["cross_share"] == pytest.approx(1)
        assert row["identity_relative_error"] < 1e-14


def test_same_id_lookup_and_missing_unheld_leg_do_not_require_future_liquidity():
    daily, config = _daily(sessions=4)
    expected = _rows(reconstruction_risk_decomposition(daily, config))
    # The future eligible section is irrelevant; retain outcome settlements in
    # a different order, then remove only the butterfly's third held contract.
    daily["sections"][3] = {key: value[:0] for key, value in daily["sections"][3].items()}
    future = daily["settlement_sections"][3]
    daily["settlement_sections"][3] = {key: value[[1, 0]] for key, value in future.items()}
    rows = _rows(reconstruction_risk_decomposition(daily, config))
    for name in ("outright_3m", "spread_3m_6m"):
        assert rows[name]["count"] == 1
        assert rows[name]["actual_second_moment"] == expected[name]["actual_second_moment"]
    butterfly = rows["butterfly_3m_6m_9m"]
    assert butterfly["count"] == 0
    assert butterfly["actual_second_moment"] is None
    assert butterfly["unavailable_reasons"] == {"held_contract_settlement_unavailable": 1}


def test_unavailable_states_are_not_refitted_and_output_is_aggregate_only(monkeypatch):
    import spde_covariance.empirical_study as study

    def forbidden(*args, **kwargs):
        raise AssertionError("post hoc diagnostic must not fit any model")

    monkeypatch.setattr(study, "fit_cross_section", forbidden)
    monkeypatch.setattr(study, "fit_source_covariance", forbidden)
    daily, config = _daily()
    daily["coefficients"][3] = np.nan
    result = reconstruction_risk_decomposition(daily, config)
    for row in result["exposures"]:
        assert row["count"] == 0
        assert row["unavailable_reasons"] == {"endpoint_reconstruction_unavailable": 2}
        assert row["max_identity_error"] is None
    exported = json.dumps(result, allow_nan=False)
    assert "private_a" not in exported
    assert "2024-01" not in exported
    assert "coefficients" not in exported


def test_zero_moment_shares_and_empty_schedule_are_explicit():
    daily, config = _daily(dimension=1, shocks=False)
    result = reconstruction_risk_decomposition(daily, config)
    for row in result["exposures"]:
        assert row["actual_second_moment"] == 0
        assert row["modeled_share"] is None
        assert row["identity_relative_error"] == 0
    daily["dates"] = daily["dates"] - np.timedelta64(365, "D")
    result = reconstruction_risk_decomposition(daily, config)
    assert result["scope"]["candidate_origin_count"] == 0
    assert all(row["count"] == 0 for row in result["exposures"])


def test_invalid_session_order_is_rejected():
    daily, config = _daily()
    daily["dates"][3] = daily["dates"][2]
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruction_risk_decomposition(daily, config)
