"""Information-safe issuance, paired calendar inference and public evidence."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_mechanisms import (
    MechanismConfig,
    aggregate_mechanisms,
    development_archive_paths,
    export_mechanism_evidence,
    issue_mechanism_risks,
    load_mechanism_evidence,
    paired_mechanisms,
    run_mechanism_benchmark,
)
from spde_covariance.empirical_short_history import score_short_history_risk
from spde_covariance.empirical_short_history_data import (
    ShortHistoryConfig,
    build_short_history_view,
    make_short_history_episodes,
)
from spde_covariance.laguerre import LaguerreTransport


@pytest.fixture(scope="module")
def panel_config():
    rng = np.random.default_rng(93)
    dates = np.busday_offset(np.datetime64("2023-09-01"), np.arange(50))
    months = np.arange(np.datetime64("2023-11", "M"), np.datetime64("2024-12", "M"))
    maturities = months.astype("datetime64[D]") + np.timedelta64(14, "D")
    model = LaguerreTransport(3, cache_max_bytes=0)
    state = np.array([65000.0, 1000.0, -400.0])
    rows = []
    for j, date in enumerate(dates):
        if j:
            delta = int((date - dates[j - 1]).astype(int)) / 365.25
            state = model.matrix(delta) @ state + rng.normal(size=3) * [60, 40, 20]
        prices = model.evaluate(np.maximum((maturities - date).astype(int), 0) / 365.25) @ state
        for month, maturity, price in zip(months, maturities, prices, strict=True):
            rows.append(
                (date, "cu" + str(month)[2:4] + str(month)[5:7], maturity, price, 100, 1000)
            )
    panel = ShfeCuPanel(
        *zip(*rows, strict=True), dates, provenance={"synthetic": True, "raw": [100.0]}
    )
    base = ShortHistoryConfig(
        training_end="2023-08-01",
        validation_end="2023-09-30",
        test_end="2023-12-31",
        window=20,
        min_training_intervals=15,
        history_budgets=(0, 5),
        origin_stride=15,
        sensitivity_half_lives=(),
        bootstrap_replications=12,
    )
    return panel, MechanismConfig(
        base, dimensions=(2,), interval_policies=("all", "one_day"), oracle_replications=2
    )


def _row(origin, tenor=90, method="all:d2:source_factor", budget="0", value=1.0, scored=True):
    exposure = {
        "forecast": 2 * value,
        "realized": 3 * value,
        "qlike": value,
        "quadratic": 4 * value,
        "components": {"factor": 2 * value},
    }
    return {
        "origin": origin,
        "next_date": origin,
        "phase": "evaluation",
        "budget": budget,
        "tenor": tenor,
        "method": method,
        "half_life": 63.0,
        "available": scored,
        "scored": scored,
        "reason": None if scored else "deliberate unavailable",
        "metrics": {
            "outright": dict(exposure),
            "spread": dict(exposure),
            "butterfly": dict(exposure),
            "joint": {"quadratic": 4 * value},
            "hedge": {"realized": 3 * value, "forecast": 2 * value},
        }
        if scored
        else {},
        "diagnostics": {},
    }


def test_aggregation_means_targets_within_origin_before_pooling(panel_config):
    _, config = panel_config
    records = [_row("2023-10-02", tenor, value=1) for tenor in (90, 180, 270)]
    records.append(_row("2023-10-09", value=5))
    result = aggregate_mechanisms(records, config)
    row = next(r for r in result["rows"] if r["year"] == "all" and r["target_tenor_days"] == "all")
    assert row["origins"] == 2
    assert row["episodes"] == 4
    assert row["context"] == "all" and row["dimension"] == 2 and row["method"] == "source_factor"
    assert row["history_budget"] == "0"
    assert row["metrics"]["spread"]["qlike"] == 3.0
    assert row["metrics"]["spread"]["calibration_ratio"] == pytest.approx(2 / 3)
    assert row["metrics"]["spread"]["components"]["factor"] == 6.0
    assert "evaluation_ids" not in repr(result)


def test_contrast_algebra_and_interactions_are_matched_episode_differences(panel_config):
    _, config = panel_config
    values = {
        "source_factor": 10,
        "empirical_factor": 7,
        "source_correlated": 4,
        "empirical_correlated": 6,
        "source_diagonal": 3,
        "empirical_diagonal": 5,
        "reconstructed_pool": 12,
    }
    records = [
        _row("2023-10-02", method="all:d2:" + name, value=value) for name, value in values.items()
    ]
    records.append(_row("2023-10-02", method="all:d0:pooled_shrink", value=9))
    result = paired_mechanisms(records, config)

    def find(kind, policy):
        return next(
            r
            for r in result["rows"]
            if r["contrast"] == kind
            and r["policy"] == policy
            and r["year"] == "all"
            and r["exposure"] == "spread"
        )

    assert find("source_policy", "factor")["difference"] == 3
    assert find("source_policy", "correlated")["difference"] == -2
    assert find("residual_correlated", "source")["difference"] == -6
    assert find("residual_correlated", "empirical")["difference"] == -1
    assert find("interaction_correlated", "source_minus_empirical")["difference"] == -5
    assert find("interaction_diagonal", "source_minus_empirical")["difference"] == 0
    assert find("reconstruction_pool", "fixed_selectors")["difference"] == 3
    assert result["joint_resampling"]
    assert all(
        b["percentile_95"] == [-5, -5]
        for b in find("interaction_correlated", "source_minus_empirical")["bootstrap"]
    )


def test_bootstrap_uses_common_pairs_and_shared_master_origin_draws(panel_config):
    _, config = panel_config
    origins = ["2023-10-02", "2023-10-09", "2023-10-16", "2023-10-23", "2023-10-30"]
    records = []
    for index, date in enumerate(origins):
        for budget, multiplier in (("0", 1), ("5", 2)):
            for tenor in (90, 180, 270):
                records.append(_row(date, tenor, "all:d2:empirical_factor", budget, value=0))
                records.append(
                    _row(
                        date, tenor, "all:d2:source_factor", budget, value=multiplier * (index + 1)
                    )
                )
                if index % 2:
                    records.append(
                        _row(date, tenor, "one_day:d2:empirical_factor", budget, value=0)
                    )
                    records.append(
                        _row(
                            date,
                            tenor,
                            "one_day:d2:source_factor",
                            budget,
                            value=multiplier * (index + 1),
                            scored=tenor != 180,
                        )
                    )
    result = paired_mechanisms(records, config)
    repeat = paired_mechanisms(list(reversed(records)), config)
    assert result == repeat
    assert result["scheduled_origins"] == 5
    assert result["one_day_missing_origins_retained"]

    def find(context, budget):
        return next(
            r
            for r in result["rows"]
            if r["contrast"] == "source_policy"
            and r["policy"] == "factor"
            and r["context"] == context
            and r["history_budget"] == budget
            and r["year"] == "all"
            and r["exposure"] == "spread"
        )

    first, doubled = find("all", "0"), find("all", "5")
    assert first["paired_episodes"] == 15 and first["paired_origins"] == 5
    assert first["difference"] == 3
    for a, b in zip(first["bootstrap"], doubled["bootstrap"], strict=True):
        assert b["bootstrap_se"] == pytest.approx(2 * a["bootstrap_se"])
        assert_allclose(b["percentile_95"], 2 * np.asarray(a["percentile_95"]))
    limited = find("one_day", "0")
    assert limited["paired_episodes"] == 4 and limited["paired_origins"] == 2
    assert limited["difference"] == 3


def test_issuer_is_frozen_and_independent_of_hidden_and_future_prices(panel_config):
    panel, config = panel_config
    episode = make_short_history_episodes(panel, config.base, "evaluation")[-3]
    view = build_short_history_view(panel, episode, 0, config.base)
    issued = issue_mechanism_risks(view, config)
    prices = panel.settlement.copy()
    hidden = ((panel.contract_ids == episode.target_id) & (panel.dates < episode.origin)) | (
        panel.dates > episode.origin
    )
    prices[hidden] += 15000
    changed_panel = ShfeCuPanel(
        panel.dates,
        panel.contract_ids,
        panel.contract_month_dates,
        prices,
        panel.volume,
        panel.open_interest,
        panel.session_dates,
    )
    changed_view = build_short_history_view(changed_panel, episode, 0, config.base)
    changed = issue_mechanism_risks(changed_view, config)
    assert issued.keys() == changed.keys()
    for key, fit in issued.items():
        assert fit.available == changed[key].available
        if fit.available:
            assert_allclose(fit.covariance, changed[key].covariance, atol=0)
            with pytest.raises(ValueError):
                fit.covariance[0, 0] = 0
    fit = issued["all:d2:source_factor"]
    future_mask = panel.dates == episode.next_date
    before = score_short_history_risk(
        fit,
        future_date=episode.next_date,
        future_ids=panel.contract_ids[future_mask],
        future_prices=panel.settlement[future_mask],
    )
    after = score_short_history_risk(
        fit,
        future_date=episode.next_date,
        future_ids=panel.contract_ids[future_mask],
        future_prices=prices[future_mask],
    )
    assert before["metrics"]["outright"]["qlike"] != after["metrics"]["outright"]["qlike"]
    assert fit.diagnostics["information_fingerprint"] == view.fingerprint
    with pytest.raises(TypeError):
        issue_mechanism_risks(view, config, future_prices=prices)


def test_2026_is_rejected_before_outputs_and_archive_allowlist_never_traverses_reserve(
    panel_config, tmp_path
):
    panel, config = panel_config
    extra_calendar = np.r_[panel.session_dates, np.datetime64("2026-01-05")]
    changed = ShfeCuPanel(
        panel.dates,
        panel.contract_ids,
        panel.contract_month_dates,
        panel.settlement,
        panel.volume,
        panel.open_interest,
        extra_calendar,
    )
    output = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="2026"):
        run_mechanism_benchmark(changed, config, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="2026"):
        MechanismConfig(replace(config.base, test_end="2026-01-01"))
    paths = development_archive_paths(tmp_path / "recent", tmp_path / "older")
    assert [p.name for p in paths] == [f"shfe-futures-{year}.zip" for year in range(2016, 2026)]
    assert all("2026" not in p.name for p in paths)


def test_synthetic_runner_reproducible_and_export_omits_private_stream(panel_config, tmp_path):
    panel, config = panel_config
    first = run_mechanism_benchmark(panel, config, tmp_path / "run-a")
    second = run_mechanism_benchmark(panel, config, tmp_path / "run-b", workers=2)
    assert first["summary"] == second["summary"]
    assert first["paired"] == second["paired"]
    assert first["protocol"] == second["protocol"]
    private = tmp_path / "run-a" / "private_predictions.jsonl"
    assert private.exists()
    assert '"pnl"' in private.read_text()
    exported = export_mechanism_evidence(tmp_path / "run-a", tmp_path / "evidence")
    evidence = load_mechanism_evidence(exported)
    assert not (exported / "private_predictions.jsonl").exists()
    assert evidence["manifest"]["episode_count"] > 0
    assert evidence["manifest"]["data_provenance"].get("raw") is None
    content = repr(evidence)
    for forbidden in (
        "evaluation_ids",
        "information_fingerprint",
        "origin_prices",
        "matched_raw_pool_covariance",
    ):
        assert forbidden not in content
    (exported / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_mechanism_evidence(exported)


def test_declared_cache_limits_and_unignored_output_rejected(panel_config):
    from spde_covariance.provenance import _source_root

    panel, config = panel_config
    with pytest.raises(ValueError, match="bounded"):
        MechanismConfig(replace(config.base, cache_entries=65))
    with pytest.raises(ValueError, match="bounded"):
        MechanismConfig(replace(config.base, cache_bytes=64 * 1024**2 + 1))
    root = _source_root()
    if root is None:
        pytest.skip("Git-ignored boundary applies inside a source checkout")
    output = root / "unignored-mechanism-test-output"
    assert not output.exists()
    with pytest.raises(ValueError, match="Git-ignored"):
        run_mechanism_benchmark(panel, config, output)
    assert not output.exists()
