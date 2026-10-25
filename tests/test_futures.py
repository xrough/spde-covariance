import csv
from dataclasses import FrozenInstanceError, replace
from datetime import date

import numpy as np
import pytest

from spde_covariance.futures import ContractSpec, SettlementChanges, SettlementPanel
from spde_covariance.workflow import fit_innovation_covariance, interval_diagnostics


def _spec(contract_id="cu2502", **options):
    return ContractSpec(
        contract_id,
        "SHFE",
        "cu",
        "CNY",
        "CNY/tonne",
        5,
        date(2024, 2, 19),
        "2025-02",
        date(2025, 1, 1),
        "https://example.invalid/synthetic-metadata",
        "a" * 64,
        **options,
    )


def _panel(prices=None, dates=None, specs=None, available=None):
    dates = dates or ["2025-01-01", "2025-01-02", "2025-01-03"]
    prices = (
        np.array([[100.0, 200.0], [102.0, 204.0], [105.0, 208.0]]) if prices is None else prices
    )
    specs = (_spec(), _spec("cu2503")) if specs is None else specs
    available = (
        np.tile(np.array(dates, dtype="datetime64[D]")[:, None], (1, len(specs)))
        if available is None
        else available
    )
    return SettlementPanel(tuple(dates), specs, prices, available)


def _changes(panel=None, **options):
    return (panel or _panel()).normalized_changes(
        asof="2025-01-03",
        integrated_drift="zero",
        price_assumption="reduced_form_settlements",
        **options,
    )


def _bridge(changes, **options):
    count = len(changes.raw_changes)
    parameters = dict(
        operators=[np.eye(len(changes.contracts))] * count,
        map_asof_dates=changes.valuation_dates[:-1],
        basis_id="declared-contract-basis",
        observation_map_assumption="identity on settlement-change coordinates",
        noise_assumption="none",
        curve_unit="CNY/tonne",
    )
    return changes.to_observed_batch(**(parameters | options))


def test_same_contract_changes_preserve_missing_calendar_rows_and_irregular_deltas():
    panel = _panel(
        prices=[[100, 200], [np.nan, 204], [105, 208]],
        dates=["2025-01-01", "2025-01-02", "2025-01-06"],
    )
    changes = panel.normalized_changes(
        asof="2025-01-06", integrated_drift="zero", price_assumption="reduced_form_settlements"
    )
    np.testing.assert_array_equal(changes.observed_mask, [[False, True], [False, True]])
    np.testing.assert_array_equal(changes.joint_counts, [[0, 0], [0, 2]])
    np.testing.assert_allclose(changes.interval_lengths, [1 / 365.25, 4 / 365.25])
    np.testing.assert_allclose(changes.values[:, 1], [4 * np.sqrt(365.25), 2 * np.sqrt(365.25)])
    observed = _bridge(changes)
    assert not interval_diagnostics(observed)["equal_intervals"]
    with pytest.raises(ValueError, match="irregular intervals"):
        fit_innovation_covariance(
            observed,
            regularization=0.1,
            trace_cap=1e5,
            covariance_assumption="common",
            numerical_error_budget=1e-4,
        )


def test_asof_selection_excludes_future_prices_publication_and_metadata():
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
    available = np.tile(np.array(dates, dtype="datetime64[D]")[:, None], (1, 3))
    available[1, 0] = np.datetime64("2025-01-04")
    specs = (_spec(), _spec("cu2503"), replace(_spec("cu2504"), metadata_asof=date(2025, 1, 4)))
    prices = np.array(
        [[100.0, 200.0, 999.0], [1e10, 204.0, -999.0], [105.0, 208.0, 1e20], [1e30, 2e30, 3e30]]
    )
    panel = _panel(prices, dates, specs, available)
    historical = panel.asof("2025-01-03")
    assert historical.contract_ids == ("cu2502", "cu2503")
    assert len(historical.valuation_dates) == 3
    assert np.isnan(historical.settlements[1, 0])
    one = _changes(panel)
    prices[3] = -1e35
    prices[:, 2] = -1e25
    prices[1, 0] = 0
    two = _changes(_panel(prices, dates, specs, available))
    np.testing.assert_equal(one.values, two.values)
    np.testing.assert_array_equal(one.observed_mask, [[False, True], [False, True]])


def test_metadata_and_maps_must_be_known_before_interval_start():
    spec = replace(_spec(), metadata_asof=date(2025, 1, 2))
    changes = _changes(_panel(specs=(spec, _spec("cu2503"))))
    np.testing.assert_array_equal(changes.observed_mask, [[False, True], [True, True]])
    observed = _bridge(changes)
    assert observed.observations[0].operator.shape == (1, 2)
    with pytest.raises(ValueError, match="interval start"):
        _bridge(changes, map_asof_dates=["2025-01-02", "2025-01-02"])


def test_drift_noise_and_delivery_observation_assumptions_are_explicit():
    spec = _spec(delivery_start=date(2025, 2, 17), delivery_end=date(2025, 2, 18))
    panel = _panel(specs=(spec, _spec("cu2503")))
    changes = panel.normalized_changes(
        asof="2025-01-03",
        integrated_drift=[[1, 2], [2, 3]],
        price_assumption="clean_settlement_levels",
    )
    np.testing.assert_allclose(changes.values, np.array([[1, 2], [1, 1]]) * np.sqrt(365.25))
    observed = _bridge(
        changes,
        noise_assumption="known_fresh_innovation_noise",
        noise_covariances=[np.diag([10, 20])] * 2,
    )
    np.testing.assert_array_equal(observed.observations[0].noise_covariance, np.diag([10, 20]))
    assert observed.metadata["physical_delivery_implies_average_observation"] is False
    assert "no H-valued trajectory" in observed.metadata["trajectory_claim"]
    with pytest.raises(TypeError, match="integrated_drift"):
        panel.normalized_changes(asof="2025-01-03", price_assumption="clean_settlement_levels")
    with pytest.raises(ValueError, match="separate error model"):
        panel.normalized_changes(
            asof="2025-01-03", integrated_drift="zero", price_assumption="noisy_price_levels"
        )
    with pytest.raises(ValueError, match="fresh innovation noise"):
        _bridge(changes, noise_assumption="difference_price_noise")
    with pytest.raises(ValueError, match="supply one known"):
        _bridge(changes, noise_assumption="known_fresh_innovation_noise")
    with pytest.raises(ValueError, match="unit conversion"):
        _bridge(changes, curve_unit="USD/barrel")


def test_realized_pnl_uses_raw_change_times_signed_contract_counts_and_multipliers():
    changes = _changes(_panel(specs=(_spec(), replace(_spec("cu2503"), multiplier=10))))
    # +2 copper contracts and -1 second contract: 2*5*2 - 1*10*4 = -20 CNY.
    assert changes.contract_pnl({"cu2502": 2, "cu2503": -1}, interval_index=0) == -20
    np.testing.assert_array_equal(
        changes.pnl_loading({"cu2502": 2, "cu2503": -1}, np.eye(2)), [10, -10]
    )
    drifted = _panel().normalized_changes(
        asof="2025-01-03",
        integrated_drift=np.ones((2, 2)),
        price_assumption="reduced_form_settlements",
    )
    assert drifted.contract_pnl({"cu2502": 1}, interval_index=0) == 10
    with pytest.raises(ValueError, match="known contract"):
        changes.contract_pnl({"continuous_front": 1}, interval_index=0)
    missing = _changes(_panel(prices=[[100, 200], [np.nan, 204], [105, 208]]))
    with pytest.raises(ValueError, match="held contract"):
        missing.contract_pnl({"cu2502": 1}, interval_index=0)


def test_arrays_and_metadata_are_detached_immutable_snapshots():
    prices = np.array([[100.0, 200.0], [102.0, 204.0], [105.0, 208.0]])
    panel = _panel(prices)
    changes = _changes(panel)
    prices[:] = -999
    assert panel.settlements[0, 0] == 100
    for array in (
        panel.settlements,
        panel.available_dates,
        changes.values,
        changes.raw_changes,
        changes.joint_counts,
        changes.observed_mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        panel.contracts[0].multiplier = 99
    with pytest.raises(TypeError):
        changes.metadata["asof"] = "future"


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_csv_provenance_exact_calendar_duplicates_and_no_presetlement_filling(tmp_path):
    contracts = tmp_path / "contracts.csv"
    prices = tmp_path / "prices.csv"
    spec = _spec()
    metadata = {name: getattr(spec, name) for name in spec.__dataclass_fields__}
    metadata["calendar_adjusted"] = "false"
    _write_csv(contracts, list(metadata), [metadata])
    rows = [
        dict(
            contract_id=spec.contract_id,
            valuation_date="2025-01-01",
            settlement="100",
            available_date="2025-01-01",
            source_url="synthetic:test",
            source_sha256="b" * 64,
            PRESETTLEMENTPRICE="99",
        ),
        dict(
            contract_id=spec.contract_id,
            valuation_date="2025-01-03",
            settlement="105",
            available_date="2025-01-03",
            source_url="synthetic:test",
            source_sha256="b" * 64,
            PRESETTLEMENTPRICE="102",
        ),
    ]
    _write_csv(prices, list(rows[0]), rows)
    panel = SettlementPanel.from_csv(
        prices, contracts, valuation_dates=["2025-01-01", "2025-01-02", "2025-01-03"]
    )
    assert np.isnan(panel.settlements[1, 0])
    assert not np.any(_changes(panel).observed_mask)
    assert len(panel.metadata["input_prices_sha256"]) == 64
    assert panel.metadata["price_sources"] == (("synthetic:test", "b" * 64),)
    with pytest.raises(ValueError, match="no jointly observed"):
        _bridge(_changes(panel))
    _write_csv(prices, list(rows[0]), rows + rows[:1])
    with pytest.raises(ValueError, match="duplicate contract-date"):
        SettlementPanel.from_csv(
            prices, contracts, valuation_dates=["2025-01-01", "2025-01-02", "2025-01-03"]
        )


@pytest.mark.parametrize(
    "change",
    [
        {"multiplier": 0},
        {"multiplier": True},
        {"source_sha256": "wrong"},
        {"delivery_start": date(2025, 2, 17)},
        {"delivery_month": "2025-2"},
    ],
)
def test_contract_metadata_validation(change):
    with pytest.raises(ValueError):
        replace(_spec(), **change)


def test_mixed_units_unavailable_quotes_and_prelisting_rows_are_rejected():
    with pytest.raises(ValueError, match="share currency"):
        _panel(specs=(_spec(), replace(_spec("cu2503"), currency="USD")))
    with pytest.raises(ValueError, match="requires available_dates"):
        _panel(available=np.full((3, 2), np.datetime64("NaT", "D")))
    with pytest.raises(ValueError, match="listing_date"):
        _panel(specs=(replace(_spec(), listing_date=date(2025, 1, 2)), _spec("cu2503")))


def test_bridge_equal_step_fit_is_past_only_and_keeps_units():
    rng = np.random.default_rng(20)
    prices = 100 + np.vstack([np.zeros(2), np.cumsum(rng.normal(0, 0.01, (12, 2)), axis=0)])
    dates = [date(2025, 1, day) for day in range(1, 14)]
    panel = _panel(prices, dates)
    changes = panel.normalized_changes(
        asof="2025-01-12", integrated_drift="zero", price_assumption="reduced_form_settlements"
    )
    observed = _bridge(changes)
    fit = fit_innovation_covariance(
        observed,
        regularization=0.01,
        trace_cap=2,
        covariance_assumption="common",
        numerical_error_budget=1e-5,
    )
    assert fit.metadata["asof"] == "2025-01-12"
    assert fit.metadata["covariance_unit"] == "(CNY/tonne)^2/year"
    assert fit.metadata["instantaneous_covariance_recovery"] is False
    assert fit.estimate.converged


@pytest.mark.parametrize("late_field", ["metadata_asof", "listing_date"])
def test_direct_changes_cannot_bypass_start_known_metadata_or_listing(late_field):
    spec = replace(_spec(), **{late_field: date(2025, 1, 2)})
    with pytest.raises(ValueError, match="interval start"):
        SettlementChanges(
            ("2025-01-01", "2025-01-02"),
            (spec,),
            [[2]],
            [[0]],
            [[True]],
            date(2025, 1, 3),
            "reduced_form_settlements",
        )


def test_auxiliary_only_cells_keep_availability_and_are_masked_asof():
    panel = SettlementPanel(
        ("2025-01-01", "2025-01-02", "2025-01-03"),
        (_spec(),),
        [[100], [np.nan], [102]],
        [["2025-01-01"], ["2025-01-10"], ["2025-01-03"]],
        volume=[[1], [999], [1]],
        open_interest=[[2], [888], [2]],
    )
    historical = panel.asof("2025-01-03")
    assert panel.available_dates[1, 0] == np.datetime64("2025-01-10")
    assert np.isnan(historical.volume[1, 0])
    assert np.isnan(historical.open_interest[1, 0])
    with pytest.raises(ValueError, match="available_dates"):
        SettlementPanel(("2025-01-01",), (_spec(),), [[np.nan]], [["NaT"]], volume=[[999]])


def _metadata_csv(tmp_path):
    path = tmp_path / "contracts.csv"
    spec = _spec()
    row = {name: getattr(spec, name) for name in spec.__dataclass_fields__}
    row["calendar_adjusted"] = "false"
    _write_csv(path, list(row), [row])
    return path


def test_csv_auxiliary_only_cells_require_and_preserve_source_availability(tmp_path):
    contracts = _metadata_csv(tmp_path)
    prices = tmp_path / "prices.csv"
    row = dict(
        contract_id="cu2502",
        valuation_date="2025-01-02",
        settlement="",
        available_date="2025-01-10",
        volume="999",
        open_interest="888",
        source_url="synthetic:late-liquidity",
        source_sha256="b" * 64,
    )
    _write_csv(prices, list(row), [row])
    panel = SettlementPanel.from_csv(
        prices, contracts, valuation_dates=["2025-01-01", "2025-01-02", "2025-01-03"]
    )
    assert panel.available_dates[1, 0] == np.datetime64("2025-01-10")
    assert np.isnan(panel.asof("2025-01-03").volume[1, 0])
    assert np.isnan(panel.asof("2025-01-03").open_interest[1, 0])
    assert panel.metadata["price_sources"] == (("synthetic:late-liquidity", "b" * 64),)
    row["available_date"] = ""
    _write_csv(prices, list(row), [row])
    with pytest.raises(ValueError, match="available_date"):
        SettlementPanel.from_csv(prices, contracts, valuation_dates=["2025-01-02"])


def test_csv_duplicate_headers_are_rejected(tmp_path):
    contracts = _metadata_csv(tmp_path)
    prices = tmp_path / "prices.csv"
    prices.write_text(
        "contract_id,valuation_date,settlement,settlement,available_date,source_url,source_sha256\n"
        "cu2502,2025-01-01,100,999,2025-01-01,synthetic:test," + "b" * 64 + "\n"
    )
    with pytest.raises(ValueError, match="duplicate.*columns"):
        SettlementPanel.from_csv(prices, contracts, valuation_dates=["2025-01-01"])


def test_pnl_rejects_finite_inputs_whose_cash_result_overflows():
    changes = _changes()
    with pytest.raises(ValueError, match="finite"):
        changes.contract_pnl({"cu2502": 1e308}, interval_index=0)
