"""Issued forecasts, future-only scores and joint-origin evidence aggregation."""

import hashlib
import json
import runpy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_short_history import (
    EVIDENCE_FILES,
    aggregate_short_history,
    export_short_history_evidence,
    issue_short_history_risk,
    issue_short_history_risks,
    load_short_history_evidence,
    paired_short_history,
    score_short_history_risk,
)
from spde_covariance.empirical_short_history_data import (
    ShortHistoryConfig,
    build_short_history_view,
    make_short_history_episodes,
)
from spde_covariance.empirical_short_history_models import BoundedModelCache


@pytest.fixture(scope="module")
def study():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples/short_history_pipeline.py"))
    panel = example["synthetic_panel"]()
    config = ShortHistoryConfig(
        training_end="2022-01-31",
        validation_end="2022-02-28",
        test_end="2022-03-31",
        window=20,
        min_training_intervals=16,
        origin_stride=10,
        bootstrap_replications=50,
        sensitivity_half_lives=(),
    )
    episode = make_short_history_episodes(panel, config, "evaluation")[-2]
    view = build_short_history_view(panel, episode, 0, config)
    issued = issue_short_history_risks(view, config)
    return panel, config, episode, view, issued


def _panel_prices(panel, prices):
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


def _future(panel, episode):
    keep = panel.dates == episode.next_date
    return dict(
        future_date=episode.next_date,
        future_ids=panel.contract_ids[keep],
        future_prices=panel.settlement[keep],
    )


def test_issuer_immutable_and_no_forbidden_prices_enter_or_change_forecasts(study):
    panel, config, episode, view, issued = study
    altered = panel.settlement.copy()
    forbidden = ((panel.contract_ids == episode.target_id) & (panel.dates < episode.origin)) | (
        panel.dates > episode.origin
    )
    altered[forbidden] *= 2.3
    other_view = build_short_history_view(_panel_prices(panel, altered), episode, 0, config)
    assert other_view.fingerprint == view.fingerprint
    other = issue_short_history_risks(other_view, config)
    for name, fit in issued.items():
        assert fit.available == other[name].available
        if fit.available:
            np.testing.assert_array_equal(fit.covariance, other[name].covariance)
            for component, value in fit.components.items():
                np.testing.assert_array_equal(value, other[name].components[component])
    chosen = issued["dfm_joint"]
    with pytest.raises(FrozenInstanceError):
        chosen.available = False
    with pytest.raises(ValueError):
        chosen.covariance.flags.writeable = True
    with pytest.raises(TypeError):
        chosen.exposures["new"] = np.ones(3)
    with pytest.raises(TypeError, match="future_prices"):
        issue_short_history_risk(view, config, future_prices=np.ones(3))
    with pytest.raises(ValueError, match="unknown"):
        issue_short_history_risk(view, config, method="lookahead")


def test_cached_and_uncached_issuance_and_history_order_agree(study):
    panel, config, episode, view, issued = study
    cache = BoundedModelCache(max_entries=3, max_bytes=64 * 1024**2)
    for budget in (0, 5, 20, None, 20, 5, 0):
        current = build_short_history_view(panel, episode, budget, config)
        predictions = issue_short_history_risks(current, config, cache=cache)
        if budget == 0:
            for method, fit in issued.items():
                assert predictions[method].available == fit.available
                if fit.available:
                    np.testing.assert_array_equal(predictions[method].covariance, fit.covariance)
    again = issue_short_history_risks(view, config, cache=cache)
    before = cache.hits
    singular = issue_short_history_risk(view, config, method="dfm_joint", cache=cache)
    assert cache.hits == before + 1
    np.testing.assert_array_equal(singular.covariance, again["dfm_joint"].covariance)
    assert cache.entries <= 3 and cache.bytes <= cache.max_bytes


def test_forecast_covariance_is_currency_scale_equivariant(study):
    panel, config, episode, _, issued = study
    factor = 10.0
    scaled = _panel_prices(panel, factor * panel.settlement)
    predictions = issue_short_history_risks(
        build_short_history_view(scaled, episode, 0, config), config
    )
    for method, original in issued.items():
        assert original.available == predictions[method].available
        if original.available:
            np.testing.assert_allclose(
                predictions[method].covariance,
                factor**2 * original.covariance,
                rtol=2e-7,
                atol=2e-6,
                err_msg=method,
            )


def test_scores_have_exact_units_quadratic_identity_and_bounded_hedge(study):
    _, _, _, _, issued = study
    matrix = np.array([[4.0, 1.0, 2.0], [1.0, 2.0, 0.3], [2.0, 0.3, 3.0]])
    fit = replace(
        issued["dfm_joint"],
        origin_prices=np.array([100.0, 105.0, 110.0]),
        covariance=matrix,
        components={"factor": 0.7 * matrix, "residual": 0.3 * matrix},
    )
    # Deliberately reverse the supplied ID order: score by actual contract ID.
    future = dict(
        future_date=fit.next_date,
        future_ids=fit.evaluation_ids[::-1],
        future_prices=np.array([113.0, 103.0, 101.0]),
    )
    result = score_short_history_risk(fit, **future)
    assert result["scored"]
    pnl, covariance = np.array([5.0, -10.0, 15.0]), 25 * matrix
    for name, exposure in fit.exposures.items():
        expected = float(exposure @ covariance @ exposure)
        squared = float(exposure @ pnl) ** 2
        values = result["metrics"][name]
        assert values["forecast"] == pytest.approx(expected)
        assert values["realized"] == pytest.approx(squared)
        assert values["quadratic"] == pytest.approx(expected**2 - 2 * expected * squared)
        assert values["qlike"] == pytest.approx(np.log(expected) + squared / expected)
        assert sum(values["components"].values()) == pytest.approx(expected)
    outer = np.outer(pnl, pnl)
    assert result["metrics"]["joint"]["quadratic"] == pytest.approx(
        np.linalg.norm(covariance - outer, "fro") ** 2 - np.linalg.norm(outer, "fro") ** 2
    )
    other_matrix = 1.3 * matrix
    other = score_short_history_risk(replace(fit, covariance=other_matrix), **future)
    assert (
        other["metrics"]["joint"]["quadratic"] - result["metrics"]["joint"]["quadratic"]
    ) == pytest.approx(
        np.linalg.norm(25 * other_matrix - outer, "fro") ** 2
        - np.linalg.norm(covariance - outer, "fro") ** 2
    )
    hedge = -np.linalg.solve(matrix[1:, 1:], matrix[1:, 0])
    positions = np.r_[1, hedge]
    np.testing.assert_allclose(result["private"]["hedge_positions"], positions, atol=1e-12)
    assert result["metrics"]["hedge"]["forecast"] == pytest.approx(
        positions @ covariance @ positions
    )
    assert result["metrics"]["hedge"]["realized"] == pytest.approx(float(positions @ pnl) ** 2)
    assert not result["metrics"]["hedge"]["bound_active"]
    bounded = score_short_history_risk(replace(fit, hedge_bounds=(-0.2, 2.0)), **future)
    assert np.all(bounded["private"]["hedge_positions"][1:] >= -0.2)
    assert bounded["metrics"]["hedge"]["bound_active"] == 1


def test_missing_outcomes_and_unavailable_forecasts_remain_explicit(study):
    panel, _, episode, _, issued = study
    future = _future(panel, episode)
    fit = issued["dfm_joint"]
    with pytest.raises(ValueError, match="next session"):
        score_short_history_risk(fit, **{**future, "future_date": episode.origin})
    with pytest.raises(ValueError, match="unique"):
        score_short_history_risk(
            fit,
            future_date=episode.next_date,
            future_ids=["cu2401", "cu2401"],
            future_prices=[1, 2],
        )
    missing = score_short_history_risk(
        fit, future_date=episode.next_date, future_ids=[], future_prices=[]
    )
    assert not missing["scored"]
    assert missing["reason"] == "missing_positive_same_contract_future_settlement"
    assert "private" not in missing
    unavailable = score_short_history_risk(issued["direct"], **future)
    assert not unavailable["available"] and not unavailable["scored"]
    assert unavailable["reason"] == "insufficient_complete_same_contract_changes"


def test_changing_future_outcome_changes_scores_but_not_issued_risk(study):
    panel, _, episode, _, issued = study
    fit = issued["dfm_joint"]
    covariance = fit.covariance.copy()
    future = _future(panel, episode)
    baseline = score_short_history_risk(fit, **future)
    altered = future["future_prices"].copy()
    altered[future["future_ids"] == fit.evaluation_ids[0]] += 1000
    changed = score_short_history_risk(fit, **{**future, "future_prices": altered})
    np.testing.assert_array_equal(fit.covariance, covariance)
    assert baseline["metrics"]["outright"]["forecast"] == changed["metrics"]["outright"]["forecast"]
    assert baseline["metrics"]["outright"]["realized"] != changed["metrics"]["outright"]["realized"]
    assert baseline["metrics"]["joint"]["quadratic"] != changed["metrics"]["joint"]["quadratic"]


def _row(origin, tenor, method, difference, *, budget="0", scored=True):
    reference = 10.0
    value = reference + difference if method == "dfm_joint" else reference
    metrics = {
        name: {"forecast": value + 1, "realized": value + 2, "qlike": value, "quadratic": value}
        for name in ("outright", "spread", "butterfly")
    }
    metrics.update(joint={"quadratic": value}, hedge={"forecast": value + 1, "realized": value})
    return dict(
        origin=origin,
        tenor=tenor,
        method=method,
        budget=budget,
        half_life=63,
        phase="evaluation",
        available=scored,
        scored=scored,
        reason=None if scored else "unavailable",
        metrics=metrics,
    )


def test_aggregate_targets_within_origins_before_pooled_average():
    config = ShortHistoryConfig(bootstrap_replications=50)
    rows = [
        _row("2024-01-02", 90, "dfm_joint", 2),
        _row("2024-01-09", 90, "dfm_joint", 0),
        _row("2024-01-09", 180, "dfm_joint", 20),
        _row("2024-01-16", 90, "dfm_joint", 4),
        _row("2024-01-16", 180, "dfm_joint", 6),
        _row("2024-01-23", 90, "dfm_joint", 999, scored=False),
    ]
    result = aggregate_short_history(rows, config)
    all_targets = next(r for r in result["rows"] if r["tenor"] == "all")
    assert all_targets["metrics"]["spread"]["qlike"] == pytest.approx(10 + 17 / 3)
    assert all_targets["origins"] == 3
    assert all_targets["episodes"] == 6
    assert all_targets["available"] == all_targets["scored"] == 5
    assert all_targets["failures"] == {"unavailable": 1}


def test_paired_bootstrap_uses_joint_calendar_draws_after_target_averaging():
    config = ShortHistoryConfig(bootstrap_replications=100)
    rows = []
    vector = np.array([1.0, 4.0, -2.0, 3.0, 8.0, -1.0])
    for budget, scale in (("0", 1), ("5", 3)):
        for index, value in enumerate(vector):
            origin = str(np.datetime64("2024-01-02") + np.timedelta64(7 * index, "D"))
            for tenor, offset in ((90, -0.4), (180, 0.4)):
                for method in ("dfm_joint", "spde_joint"):
                    rows.append(
                        _row(origin, tenor, method, scale * (value + offset), budget=budget)
                    )
    paired = paired_short_history(rows, config)
    primary = [
        r
        for r in paired["rows"]
        if r["method"] == "dfm_joint"
        and r["reference"] == "spde_joint"
        and r["tenor"] == "all"
        and r["exposure"] == "spread"
    ]
    assert len(primary) == 2
    for item in primary:
        scale = 1 if item["budget"] == "0" else 3
        assert item["paired_origins"] == len(vector)
        assert item["paired_episodes"] == 2 * len(vector)
        assert item["difference"] == pytest.approx(scale * np.mean(vector))
        for interval in item["bootstrap"]:
            block = interval["block_origins"]
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, block, 1]))
            starts = rng.integers(
                0,
                len(vector),
                size=(config.bootstrap_replications, int(np.ceil(len(vector) / block))),
            )
            indices = ((starts[..., None] + np.arange(block)) % len(vector)).reshape(
                config.bootstrap_replications, -1
            )[:, : len(vector)]
            estimates = scale * vector[indices].mean(axis=1)
            assert interval["bootstrap_se"] == pytest.approx(np.std(estimates, ddof=1))
            np.testing.assert_allclose(
                interval["percentile_95"], np.quantile(estimates, [0.025, 0.975])
            )


def test_exporter_excludes_local_records_and_verifies_all_checksums(tmp_path):
    local, public = tmp_path / "local", tmp_path / "public"
    local.mkdir()
    for name in EVIDENCE_FILES:
        (local / name).write_text(json.dumps({"aggregate": name}) + "\n")
    checksums = {
        name: hashlib.sha256((local / name).read_bytes()).hexdigest() for name in EVIDENCE_FILES
    }
    (local / "checksums.json").write_text(json.dumps(checksums))
    (local / "private_predictions.jsonl").write_text('"PRIVATE_RAW_SETTLEMENT_AND_PNL"\n')
    (local / "extra_prices.csv").write_text("PRIVATE_RAW_SETTLEMENT_AND_PNL\n")
    export_short_history_evidence(local, public)
    assert {p.name for p in public.iterdir()} == {*EVIDENCE_FILES, "checksums.json"}
    assert "PRIVATE_RAW_SETTLEMENT_AND_PNL" not in "".join(p.read_text() for p in public.iterdir())
    assert set(load_short_history_evidence(public)) == {Path(name).stem for name in EVIDENCE_FILES}
    (public / "summary.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_short_history_evidence(public)
    checksums["private_predictions.jsonl"] = "anything"
    (public / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="whitelist mismatch"):
        load_short_history_evidence(public)
