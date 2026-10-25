"""Synthetic CSV -> past-only settlement covariance -> contract-count hedge.

Run from the repository root: python examples/futures_pipeline.py
No network requests, real data, execution assumptions, or empirical claims.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spde_covariance.futures import SettlementPanel
from spde_covariance.hedging import solve_plugin_hedge
from spde_covariance.workflow import fit_innovation_covariance


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(20)]
    cutoff = dates[15]
    ids = ("toy2503", "toy2504")
    rng = np.random.default_rng(710)
    covariance = np.array([[9.0, 6.0], [6.0, 16.0]])
    steps = rng.multivariate_normal(np.zeros(2), covariance / 365.25, size=len(dates) - 1)
    prices = [100.0, 102.0] + np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
    source_hash = hashlib.sha256(b"synthetic settlement fixture; no exchange data").hexdigest()
    specifications = [
        dict(
            contract_id=key,
            exchange="SYNTHETIC",
            product="toy",
            currency="CNY",
            price_unit="CNY/tonne",
            multiplier=5,
            listing_date="2024-12-01",
            delivery_month=month,
            metadata_asof="2024-12-01",
            expiry_date="",
            delivery_start="",
            delivery_end="",
            calendar_adjusted="false",
            source_url="synthetic:known-contract-specification",
            source_sha256=source_hash,
        )
        for key, month in zip(ids, ("2025-03", "2025-04"), strict=True)
    ]
    observations = [
        dict(
            contract_id=key,
            valuation_date=day.isoformat(),
            settlement=float(prices[i, j]),
            available_date=day.isoformat(),
            volume=100,
            open_interest=1000,
            source_url="synthetic:settlement-panel",
            source_sha256=source_hash,
        )
        for i, day in enumerate(dates)
        for j, key in enumerate(ids)
    ]
    with TemporaryDirectory(prefix="futures-pipeline-") as temporary:
        folder = Path(temporary)
        _csv(folder / "contracts.csv", specifications)
        _csv(folder / "prices.csv", observations)
        panel = SettlementPanel.from_csv(
            folder / "prices.csv", folder / "contracts.csv", valuation_dates=dates
        )
        training = panel.normalized_changes(
            asof=cutoff, integrated_drift="zero", price_assumption="reduced_form_settlements"
        )
        operator = np.eye(2)  # Declared contract-coordinate toy basis, not a maturity fit.
        observed = training.to_observed_batch(
            operators=[operator] * len(training.raw_changes),
            map_asof_dates=training.valuation_dates[:-1],
            basis_id="toy-contract-coordinate-basis",
            observation_map_assumption="identity map in the two declared settlement coordinates",
            noise_assumption="none",
            curve_unit="CNY/tonne",
        )
        fit = fit_innovation_covariance(
            observed,
            regularization=0.01,
            trace_cap=50,
            covariance_assumption="common",
            numerical_error_budget=1e-5,
        )
        target = training.pnl_loading({ids[0]: 1}, operator)
        instrument = training.pnl_loading({ids[1]: 1}, operator)[:, None]
        hedge = solve_plugin_hedge(fit.covariance, target, instrument)
        if not hedge.available:
            raise RuntimeError("synthetic point-covariance hedge failed")
        # Solver exposure convention is target - instrument @ hedge.positions.
        held_positions = {ids[0]: 1.0, ids[1]: -float(hedge.positions[0])}
        full_changes = panel.normalized_changes(
            asof=dates[-1], integrated_drift="zero", price_assumption="reduced_form_settlements"
        )
        heldout_index = 15  # Jan16 -> Jan17; Jan17 price was absent from every fit input.
        pnl = full_changes.contract_pnl(held_positions, interval_index=heldout_index)
        residual = training.pnl_loading(held_positions, operator)
        variance = float(
            full_changes.interval_lengths[heldout_index] * residual @ fit.covariance @ residual
        )
        print(
            json.dumps(
                {
                    "interpretation": "synthetic plumbing example; no hedge-performance evidence",
                    "fit_asof": cutoff.isoformat(),
                    "fit_intervals": observed.n_intervals,
                    "heldout_interval": [day.isoformat() for day in dates[15:17]],
                    "signed_contract_counts": held_positions,
                    "realized_pnl_CNY": pnl,
                    "declared_one_day_variance_CNY_squared": variance,
                    "pnl_formula": "sum(contracts * 5 tonnes/contract * raw price change in CNY/tonne)",
                    "price_covariance_unit": fit.metadata["covariance_unit"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
