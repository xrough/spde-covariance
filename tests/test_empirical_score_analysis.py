"""Pairing, scoring, uncertainty and aggregate-publication guarantees."""

import copy
import json

import numpy as np
import pytest

from spde_covariance.empirical_score_analysis import analyze_forecast_scores
from spde_covariance.empirical_study import (
    EmpiricalConfig,
    paired_block_summary,
    qlike,
    quadratic_risk_score,
)


@pytest.fixture
def config():
    return {
        "pnl_variance_floor": 1.0,
        "seed": 20261026,
        "bootstrap_block": 5,
        "bootstrap_replications": 40,
        "private_metadata": {"contract_ids": ["secret_contract"]},
    }


def _records(count=120, methods=("direct", "spde_local", "unadjusted")):
    rows = []
    for index in range(count):
        origin = np.datetime64("2024-01-02") + np.timedelta64(index, "D")
        gap = (1, 2, 3, 5)[index % 4]
        actual = float(12 + index % 11)
        for method in methods:
            prediction = 7 + index % 17
            if method == "spde_local":
                prediction += 2 * np.sin(index / 7)
            elif method == "unadjusted":
                prediction += 3
            rows.append(
                {
                    "origin": str(origin),
                    "year": str(origin + np.timedelta64(gap, "D"))[:4],
                    "calendar_days": gap,
                    "exposure": "spread_3m_6m",
                    "method": method,
                    "variance": float(prediction),
                    "squared_pnl": actual,
                    "signed_pnl": -np.sqrt(actual),
                    "contract_ids": ["secret_contract"],
                    "raw_prices": [123456789.25],
                }
            )
    return rows


def _pair(result, *, loss="qlike", method="spde_local", stratum="pooled", group="all"):
    return next(
        row
        for row in result["paired_comparisons"]
        if row["loss"] == loss
        and row["method"] == method
        and row["stratum"] == stratum
        and row["group"] == group
    )


def test_exact_pairing_matches_existing_bootstrap_for_both_losses(config):
    rows = _records()
    result = analyze_forecast_scores(rows, config, blocks=(5, 10))
    left = [row for row in rows if row["method"] == "spde_local"]
    right = [row for row in rows if row["method"] == "direct"]
    for loss in ("qlike", "quadratic"):
        function = (
            (lambda v, z: qlike(v, z, floor=config["pnl_variance_floor"]))
            if loss == "qlike"
            else quadratic_risk_score
        )
        differences = np.array(
            [
                function(a["variance"], a["squared_pnl"])
                - function(b["variance"], b["squared_pnl"])
                for a, b in zip(left, right, strict=True)
            ]
        )
        expected = paired_block_summary(differences, block=5, replications=40, seed=20261026)
        actual = _pair(result, loss=loss)
        assert all(actual[key] == value for key, value in expected.items())
        assert actual["common_count"] == actual["method_count"] == actual["reference_count"] == 120
        assert actual["method_only_count"] == actual["reference_only_count"] == 0


def test_reordering_and_arbitrary_unused_metadata_cannot_change_results(config):
    rows = _records()
    original = copy.deepcopy(rows)
    result = analyze_forecast_scores(rows, config, blocks=(10, 5))
    np.random.default_rng(3).shuffle(rows)
    for row in rows:
        row["signed_pnl"] = 1e100
        row["raw_prices"] = [1e200]
    assert result == analyze_forecast_scores(rows, config, blocks=(5, 10))
    assert original[0]["signed_pnl"] != rows[0]["signed_pnl"]


def test_input_is_not_mutated(config):
    rows = _records(12)
    original, config_copy = copy.deepcopy(rows), copy.deepcopy(config)
    analyze_forecast_scores(rows, config)
    assert rows == original
    assert config == config_copy


def test_missing_reference_and_method_records_are_disclosed(config):
    rows = _records()
    rows = [
        row
        for row in rows
        if not (row["method"] == "direct" and row["origin"] == "2024-01-02")
        and not (row["method"] == "spde_local" and row["origin"] == "2024-01-03")
    ]
    result = analyze_forecast_scores(rows, config)
    pair = _pair(result)
    assert pair["method_count"] == pair["reference_count"] == 119
    assert pair["common_count"] == 118
    assert pair["union_count"] == 120
    assert pair["method_only_count"] == pair["reference_only_count"] == 1
    availability = next(row for row in result["availability"] if row["method"] == "spde_local")
    assert all(
        availability[key] == pair[key] for key in availability if key not in {"exposure", "method"}
    )


def test_no_direct_reference_is_explicitly_unavailable(config):
    result = analyze_forecast_scores(_records(methods=("spde_local",)), config)
    pair = _pair(result)
    assert pair["available"] is False
    assert pair["count"] == pair["common_count"] == pair["reference_count"] == 0
    assert pair["method_only_count"] == 120
    assert pair["mean"] is None
    assert pair["bootstrap_percentile_95"] is None


@pytest.mark.parametrize("change", ("duplicate", "outcome", "interval", "year"))
def test_inexact_or_duplicated_pairs_fail(config, change):
    rows = _records(12)
    if change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "outcome":
        rows[1]["squared_pnl"] = np.nextafter(rows[1]["squared_pnl"], np.inf)
    elif change == "interval":
        rows[1]["calendar_days"] += 1
    else:
        rows[1]["year"] = "2025"
    with pytest.raises(ValueError):
        analyze_forecast_scores(rows, config)


def test_floor_matches_frozen_study_and_does_not_floor_secondary_score(config):
    rows = _records(12, methods=("direct", "spde_local"))
    for row in rows:
        row["variance"] = 0 if row["method"] == "spde_local" else 0.5
    result = analyze_forecast_scores(rows, config)
    assert _pair(result)["mean"] == 0
    mean_outcome = np.mean([row["squared_pnl"] for row in rows])
    assert _pair(result, loss="quadratic")["mean"] == pytest.approx(mean_outcome - 0.25)
    for row in result["method_summaries"]:
        assert row["floor_hits"] == row["count"]


def test_calibration_quintiles_use_forecasts_and_preserve_totals(config):
    result = analyze_forecast_scores(_records(), config)
    rows = [row for row in result["calibration_bins"] if row["method"] == "spde_local"]
    total = next(
        row
        for row in result["method_summaries"]
        if row["method"] == "spde_local" and row["stratum"] == "pooled"
    )
    assert sum(row["count"] for row in rows) == 120
    assert all(row["available"] for row in rows)
    reconstructed = sum(row["count"] * row["mean_squared_pnl_cny2"] for row in rows) / 120
    assert reconstructed == pytest.approx(total["mean_squared_pnl_cny2"])
    predictions = [row["mean_predicted_variance_cny2"] for row in rows]
    assert predictions == sorted(predictions)


def test_constant_forecast_does_not_split_identical_values(config):
    rows = _records()
    for row in rows:
        row["variance"] = 10.0
    result = analyze_forecast_scores(rows, config)
    selected = [row for row in result["calibration_bins"] if row["method"] == "spde_local"]
    assert [row["count"] for row in selected] == [120, 0, 0, 0, 0]
    assert _pair(result)["mean"] == 0


def test_year_and_gap_summaries_account_for_all_records(config):
    result = analyze_forecast_scores(_records(420), config)
    for stratum in ("year", "calendar_gap"):
        pairs = [
            row
            for row in result["paired_comparisons"]
            if row["stratum"] == stratum
            and row["method"] == "spde_local"
            and row["loss"] == "qlike"
        ]
        assert sum(row["count"] for row in pairs) == 420
        weighted = sum(row["mean"] * row["count"] for row in pairs) / 420
        assert weighted == pytest.approx(_pair(result)["mean"])


def test_block_sensitivity_uses_same_point_estimate_and_effective_block(config):
    result = analyze_forecast_scores(_records(12), config, blocks=(5, 40), replications=25)
    assert result["protocol"]["bootstrap_replications"] == 25
    for loss in ("qlike", "quadratic"):
        rows = [
            row
            for row in result["block_sensitivity"]
            if row["method"] == "spde_local" and row["loss"] == loss
        ]
        assert {row["requested_block_length"] for row in rows} == {5, 40}
        assert {row["block_length"] for row in rows} == {5, 12}
        assert all(row["mean"] == _pair(result, loss=loss)["mean"] for row in rows)


def test_influence_reconstructs_full_mean_without_changing_headline(config):
    result = analyze_forecast_scores(_records(500), config, replications=5)
    for row in result["score_influence"]:
        assert row["available"]
        mean = (
            row["removed_mean_difference"] * row["removed_count"]
            + row["remaining_mean_difference"] * row["retained_count"]
        ) / row["count"]
        assert mean == pytest.approx(row["full_mean_difference"])
        assert (
            row["full_mean_difference"]
            == _pair(result, method=row["method"], loss=row["loss"])["mean"]
        )
        assert 0 <= row["removed_absolute_difference_share"] <= 1
    assert "post-hoc" in result["protocol"]["influence_scope"]


def test_ex_post_outcome_strata_share_definition_across_methods(config):
    result = analyze_forecast_scores(_records(), config)
    counts = {}
    for row in result["realized_outcome_strata"]:
        counts.setdefault(row["method"], []).append(row["count"])
    assert counts["direct"] == counts["spde_local"] == counts["unadjusted"]
    assert sum(counts["direct"]) == 120
    assert "ex-post" in result["protocol"]["outcome_strata_scope"]


def test_small_cells_and_single_observation_influence_are_suppressed(config):
    result = analyze_forecast_scores(_records(4), config)
    for row in result["method_summaries"]:
        assert row["available"] is False
        assert row["mean_squared_pnl_cny2"] is None
        assert row["mean_predicted_variance_cny2"] is None
    assert all(row["mean"] is None for row in result["paired_comparisons"])
    assert all(row["full_mean_difference"] is None for row in result["score_influence"])
    assert all(row["removed_count"] < 5 for row in result["score_influence"])


def test_allowlisted_metadata_and_no_individual_market_values(config):
    rows = _records()
    rows[0]["private_note"] = "unique_secret_text"
    result = analyze_forecast_scores(rows, config)
    public = json.dumps(result, allow_nan=False)
    assert "secret_contract" not in public
    assert "unique_secret_text" not in public
    assert "123456789.25" not in public
    assert "2024-01-02" not in public
    forbidden = {
        "origin",
        "contract_ids",
        "signed_pnl",
        "squared_pnl",
        "raw_prices",
        "variance",
        "private_metadata",
    }

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(result)


@pytest.mark.parametrize(
    "field,value",
    (
        ("method", "cu2403"),
        ("exposure", "secret_exposure"),
        ("variance", np.inf),
        ("squared_pnl", -1),
        ("calendar_days", True),
        ("origin", "2024-02-31"),
    ),
)
def test_invalid_or_unallowlisted_record_is_rejected(config, field, value):
    rows = _records(6)
    rows[0][field] = value
    with pytest.raises(ValueError):
        analyze_forecast_scores(rows, config)


@pytest.mark.parametrize("blocks", ((), (0,), (1, 1), (True,), (1.5,)))
def test_invalid_bootstrap_configuration(config, blocks):
    with pytest.raises(ValueError):
        analyze_forecast_scores(_records(6), config, blocks=blocks)


def test_real_dataclass_config_and_empty_inputs_are_supported():
    config = EmpiricalConfig(bootstrap_replications=5)
    result = analyze_forecast_scores([], config)
    assert result["availability"] == []
    assert result["paired_comparisons"] == []
    assert result["protocol"]["primary_loss"] == "qlike"
    json.dumps(result, allow_nan=False)


def test_overflow_cannot_create_nonfinite_public_aggregates(config):
    rows = _records(6)
    for row in rows:
        row["variance"] = 1e154
        row["squared_pnl"] = 0
    with pytest.raises(ValueError, match="overflow"):
        analyze_forecast_scores(rows, config)


def test_nonnumeric_or_unhashable_labels_fail_cleanly(config):
    rows = _records(6)
    rows[0]["method"] = ["private_contract"]
    with pytest.raises(ValueError, match="allowlisted"):
        analyze_forecast_scores(rows, config)


def test_imperfect_proxy_expectation_identities_without_gaussian_assumption():
    probabilities = np.array([0.2, 0.5, 0.3])
    proxy = np.array([0.0, 1.0, 4.0])
    target = probabilities @ proxy
    for prediction in (0.6, 2.3, 8.0):
        qlike_regret = probabilities @ (
            qlike(prediction, proxy, floor=0.1) - qlike(target, proxy, floor=0.1)
        )
        quadratic_regret = probabilities @ (
            quadratic_risk_score(prediction, proxy) - quadratic_risk_score(target, proxy)
        )
        assert qlike_regret == pytest.approx(np.log(prediction / target) + target / prediction - 1)
        assert quadratic_regret == pytest.approx((prediction - target) ** 2)
        assert qlike_regret >= 0
        assert quadratic_regret >= 0


def test_loss_ranking_can_reverse_without_redefining_primary_metric(config):
    rows = _records(12, methods=("direct", "spde_local"))
    for row in rows:
        row["variance"] = 10.0 if row["method"] == "direct" else 1.0
        row["squared_pnl"] = 4.0
    result = analyze_forecast_scores(rows, config)
    assert _pair(result)["mean"] > 0
    assert _pair(result, loss="quadratic")["mean"] < 0
    assert result["protocol"]["primary_loss"] == "qlike"


def test_quadratic_paired_identity_uses_same_actual_outcome(config):
    rows = _records()
    result = analyze_forecast_scores(rows, config)
    by_origin = {}
    for row in rows:
        by_origin.setdefault(row["origin"], {})[row["method"]] = row
    differences = []
    for methods in by_origin.values():
        left, right = methods["spde_local"], methods["direct"]
        a, b, proxy = left["variance"], right["variance"], left["squared_pnl"]
        differences.append((a - b) * (a + b - 2 * proxy))
    assert _pair(result, loss="quadratic")["mean"] == pytest.approx(np.mean(differences))
