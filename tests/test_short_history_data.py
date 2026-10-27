"""Masked-level information boundaries and same-contract transfer baselines."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_short_history_data import (
    ShortHistoryConfig,
    baseline_covariances,
    build_short_history_view,
    daily_sections,
    interval_records,
    load_short_history_config,
    make_short_history_episodes,
)


@pytest.fixture(scope="module")
def sample():
    dates = np.arange(np.datetime64("2022-01-03"), np.datetime64("2022-04-16"))
    dates = dates[np.is_busday(dates)]
    months = np.arange(np.datetime64("2022-02", "M"), np.datetime64("2023-05", "M"))
    rows = []
    for i, day in enumerate(dates):
        for j, month in enumerate(months):
            maturity = month.astype("datetime64[D]") + np.timedelta64(14, "D")
            if maturity < day:
                continue
            identifier = "cu" + str(month)[2:4] + str(month)[5:7]
            rows.append(
                (day, identifier, maturity, 60000 + j * 20 + i * (j + 1) + np.sin(i + j), 100, 1000)
            )
    panel = ShfeCuPanel(*zip(*rows, strict=True), dates, provenance={"unsafe_prices": [12345]})
    config = ShortHistoryConfig(
        training_end="2022-01-31",
        validation_end="2022-02-28",
        test_end="2022-04-15",
        window=40,
        min_training_intervals=10,
        bootstrap_replications=20,
    )
    return panel, config


def copy_panel(panel, *, retain=None, settlement=None, volume=None):
    retain = np.ones(len(panel.dates), dtype=bool) if retain is None else retain
    return ShfeCuPanel(
        panel.dates[retain],
        panel.contract_ids[retain],
        panel.contract_month_dates[retain],
        panel.settlement[retain] if settlement is None else settlement[retain],
        panel.volume[retain] if volume is None else volume[retain],
        panel.open_interest[retain],
        panel.session_dates,
        panel.provenance,
    )


def test_config_is_immutable_and_validated(tmp_path):
    config = ShortHistoryConfig()
    with pytest.raises(FrozenInstanceError):
        config.window = 2
    path = tmp_path / "config.toml"
    path.write_text('history_budgets=[0,5,20,"all"]\nsensitivity_half_lives=[21,126]\n')
    assert load_short_history_config(path).history_budgets == (0, 5, 20, None)
    for kwargs in (
        {"window": 100},
        {"history_budgets": (5, 5)},
        {"history_budgets": (-1,)},
        {"origin_stride": 0},
        {"half_life": 0},
        {"hedge_bounds": (1, 2)},
        {"rank_rtol": 1},
        {"target_tenors_days": (270, 180, 90)},
        {"bootstrap_sensitivity_blocks": (0,)},
        {"name": "../bad"},
        {"hedge_bounds": [-2, 2]},
        {"seed": -1},
        {"dimension": 4},
        {"factor_dimension": 2},
        {"min_tenor_days": 15},
        {"max_tenor_days": 366},
        {"reconstruction_ridge": 1e-4},
    ):
        with pytest.raises(ValueError):
            ShortHistoryConfig(**kwargs)


def test_origin_schedule_phase_anchor_and_target_order(sample):
    panel, config = sample
    validation = make_short_history_episodes(panel, config, "validation")
    test = make_short_history_episodes(panel, config, "evaluation")
    assert validation[0].origin == np.datetime64("2022-02-01")
    assert test[0].origin == np.datetime64("2022-03-01")
    for episodes in (validation, test):
        origins = [episodes[i].origin for i in range(0, len(episodes), 3)]
        indices = [np.flatnonzero(panel.session_dates == date)[0] for date in origins]
        np.testing.assert_array_equal(np.diff(indices), config.origin_stride)
        for episode in episodes:
            assert episode.evaluation_ids[0] == episode.target_id
            assert sorted(episode.maturity_order) == [0, 1, 2]
            view = build_short_history_view(panel, episode, 0, config)
            reordered = view.maturities[list(episode.maturity_order)]
            assert np.all(np.diff(reordered).astype(int) > 0)
            distances = np.abs((view.maturities[1:] - view.maturities[0]).astype(int))
            assert distances[0] <= distances[1]


@pytest.mark.parametrize("budget", [0, 5, 20, None])
def test_exact_target_history_levels_removed_and_calendar_preserved(sample, budget):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    view = build_short_history_view(panel, episode, budget, config)
    assert view.available
    retained_target = view.panel.dates[view.panel.contract_ids == episode.target_id]
    expected = config.window + 1 if budget is None else budget + 1
    assert len(retained_target) == expected
    assert len(view.panel.session_dates) == config.window + 1
    assert retained_target[-1] == episode.origin
    assert view.panel.session_dates[-1] == episode.origin
    assert np.all(view.panel.dates <= episode.origin)
    assert "unsafe_prices" not in view.panel.provenance
    with pytest.raises(ValueError):
        view.origin_prices.flags.writeable = True
    with pytest.raises(ValueError):
        view.panel.settlement.flags.writeable = True
    with pytest.raises(TypeError):
        daily_sections(view)[0]["prices"] = np.array([])


def test_hidden_and_future_prices_cannot_change_view_or_baselines(sample):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    view = build_short_history_view(panel, episode, 5, config)
    cutoff = view.panel.session_dates[-6]
    altered = panel.settlement.copy()
    hidden = ((panel.contract_ids == episode.target_id) & (panel.dates < cutoff)) | (
        panel.dates > episode.origin
    )
    altered[hidden] *= 17
    new_view = build_short_history_view(copy_panel(panel, settlement=altered), episode, 5, config)
    assert new_view.fingerprint == view.fingerprint
    np.testing.assert_array_equal(new_view.panel.settlement, view.panel.settlement)
    baseline1, baseline2 = (
        baseline_covariances(view, config),
        baseline_covariances(new_view, config),
    )
    for name in baseline1:
        assert baseline1[name]["available"] == baseline2[name]["available"]
        if baseline1[name]["available"]:
            np.testing.assert_array_equal(
                baseline1[name]["covariance"], baseline2[name]["covariance"]
            )


def test_missing_target_level_does_not_bridge_and_rejects_exact_budget(sample):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    full = build_short_history_view(panel, episode, None, config)
    removed_day = full.dates[-3]
    retain = ~((panel.contract_ids == episode.target_id) & (panel.dates == removed_day))
    panel_gap = copy_panel(panel, retain=retain)
    strict = build_short_history_view(panel_gap, episode, 5, config)
    assert not strict.available
    assert strict.reason == "requested_contiguous_target_history_unavailable"
    gap = build_short_history_view(panel_gap, episode, None, config)
    assert gap.available
    records = interval_records(gap, config)
    assert len(records) == config.window
    for record in records:
        if record["left_date"] == removed_day or record["right_date"] == removed_day:
            assert episode.target_id not in record["ids"]
    direct = baseline_covariances(gap, config)["direct"]
    assert direct["diagnostics"]["complete_changes"] == config.window - 2


@pytest.mark.parametrize(
    "budget, expected_available", [(0, False), (5, False), (20, True), (None, True)]
)
def test_cold_start_baseline_availability_and_psd(sample, budget, expected_available):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    view = build_short_history_view(panel, episode, budget, config)
    models = baseline_covariances(view, config)
    assert models["direct"]["available"] == expected_available
    assert models["pooled_shrink"]["available"]
    count = models["pooled_shrink"]["diagnostics"]["complete_changes"]
    assert count == (config.window if budget is None else budget)
    assert models["pooled_shrink"]["diagnostics"]["short_blend_weight"] == count / (count + 20)
    for result in models.values():
        if result["available"]:
            assert result["covariance"].shape == (3, 3)
            assert np.linalg.eigvalsh(result["covariance"])[0] >= -1e-8


def test_normalization_units_and_budget_order_invariance(sample):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    view = build_short_history_view(panel, episode, None, config)
    records = interval_records(view, config)
    rows = []
    for record in records:
        indices = [list(record["ids"]).index(x) for x in view.evaluation_ids]
        rows.append(record["response"][indices])
    values = np.asarray(rows)
    expected = values.T @ values / len(values)
    expected *= int((view.next_date - view.origin).astype(int)) / 365.25
    np.testing.assert_allclose(
        baseline_covariances(view, config, half_life=None)["direct"]["covariance"], expected
    )
    fingerprint = {}
    for budget in (None, 20, 5, 0, 5, None):
        current = build_short_history_view(panel, episode, budget, config)
        fingerprint.setdefault(budget, current.fingerprint)
        assert fingerprint[budget] == current.fingerprint
    volume = panel.volume.copy()
    missing = (panel.dates == view.dates[-3]) & (panel.contract_ids == episode.target_id)
    volume[missing] = 0
    filtered = build_short_history_view(copy_panel(panel, volume=volume), episode, None, config)
    assert len(daily_sections(filtered, eligible=False)[-3]["ids"]) > len(
        daily_sections(filtered)[-3]["ids"]
    )
    assert (
        baseline_covariances(filtered, config)["direct"]["diagnostics"]["complete_changes"]
        == config.window
    )
    assert episode.target_id not in interval_records(filtered)[-3]["ids"]
    assert episode.target_id in interval_records(filtered, eligible=False)[-3]["ids"]


def test_unavailable_origin_or_history_arguments_fail_explicitly(sample):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    with pytest.raises(ValueError, match="origin"):
        build_short_history_view(panel, replace(episode, origin="2022-03-12"), 0, config)
    with pytest.raises(ValueError, match="budget"):
        build_short_history_view(panel, episode, -1, config)
    with pytest.raises(ValueError, match="phase"):
        make_short_history_episodes(panel, config, "training")
    with pytest.raises(ValueError, match="half_life"):
        baseline_covariances(
            build_short_history_view(panel, episode, 0, config), config, half_life=0
        )


def test_ewma_weights_use_elapsed_sessions_not_complete_case_index(sample):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    full = build_short_history_view(panel, episode, None, config)
    gap_date = full.dates[-3]
    keep = ~((panel.contract_ids == episode.target_id) & (panel.dates == gap_date))
    view = build_short_history_view(copy_panel(panel, retain=keep), episode, None, config)
    records = interval_records(view)
    values, ages = [], []
    for i, record in enumerate(records):
        ids = list(record["ids"])
        if all(x in ids for x in episode.evaluation_ids):
            values.append(record["response"][[ids.index(x) for x in episode.evaluation_ids]])
            ages.append(len(records) - 1 - i)
    weights = np.exp2(-np.asarray(ages) / 3)
    weights /= weights.sum()
    values = np.asarray(values)
    expected = (values.T * weights) @ values
    expected *= int((view.next_date - view.origin).astype(int)) / 365.25
    result = baseline_covariances(view, config, half_life=3)["direct"]
    np.testing.assert_allclose(result["covariance"], expected)
    assert result["diagnostics"]["short_effective_count"] == pytest.approx(1 / (weights @ weights))


def test_roll_changes_cannot_create_returns_and_empty_sessions_do_not_bridge(sample):
    panel, config = sample
    episode = make_short_history_episodes(panel, config, "evaluation")[6]
    # Each contract has a different but perfectly constant price. Selecting new
    # maturity slots changes their price levels, but must create no returns.
    ids = list(np.unique(panel.contract_ids))
    prices = np.array([1000 * (1 + ids.index(x)) for x in panel.contract_ids], dtype=float)
    static = copy_panel(panel, settlement=prices)
    view = build_short_history_view(static, episode, None, config)
    for result in baseline_covariances(view, config).values():
        np.testing.assert_array_equal(result["covariance"], np.zeros((3, 3)))
    empty_date = view.dates[-3]
    gap = build_short_history_view(
        copy_panel(static, retain=static.dates != empty_date), episode, None, config
    )
    records = interval_records(gap)
    assert len(records) == config.window
    adjacent_empty = [
        record for record in records if empty_date in (record["left_date"], record["right_date"])
    ]
    assert len(adjacent_empty) == 2
    assert all(len(record["ids"]) == 0 for record in adjacent_empty)
    assert (
        baseline_covariances(gap, config)["direct"]["diagnostics"]["complete_changes"]
        == config.window - 2
    )
