"""Offline masked-history risk transfer with correlated maturity residuals.

This is a software smoke example, not empirical performance evidence. It uses
only simulated copper-style contract IDs and dates; no SHFE files are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_short_history_data import ShortHistoryConfig
from spde_covariance.empirical_short_history_models import residual_maturity_features
from spde_covariance.laguerre import LaguerreTransport


def synthetic_panel(seed=20261027):
    """A common transport component plus correlated maturity and ID residuals."""
    dates = np.arange(np.datetime64("2022-01-03"), np.datetime64("2022-04-01"))
    dates = dates[np.is_busday(dates)]
    model = LaguerreTransport(3)
    generator = np.random.default_rng(seed)
    state = np.array([65000.0, 450.0, -150.0])
    residual = np.zeros(4)
    contract_residual = {}
    rows = []
    cross_loading = np.array([[0.7, 0.2, 0], [0.3, 0.5, 0], [0.1, 0.3, 0.5], [0, 0.2, 0.8]])
    for i, day in enumerate(dates):
        if i:
            delta = int((day - dates[i - 1]).astype(int)) / 365.25
            shock = generator.normal(size=3)
            state = (
                model.matrix(delta) @ state + np.sqrt(delta) * np.array([1600, 700, 450]) * shock
            )
            residual = 0.94 * residual + 18 * (
                cross_loading @ shock + 0.6 * generator.normal(size=4)
            )
        months = day.astype("datetime64[M]") + np.arange(1, 14).astype("timedelta64[M]")
        maturity = months.astype("datetime64[D]") + np.timedelta64(14, "D")
        tenor_days = (maturity - day).astype(int)
        prices = (
            model.evaluate(tenor_days / 365.25) @ state
            + residual_maturity_features(tenor_days) @ residual
        )
        for month, end, price in zip(months, maturity, prices, strict=True):
            identifier = "cu" + str(month)[2:4] + str(month)[5:7]
            contract_residual[identifier] = 0.8 * contract_residual.get(
                identifier, 0.0
            ) + generator.normal(scale=3)
            rows.append(
                (day, identifier, end, float(price + contract_residual[identifier]), 100, 1000)
            )
    return ShfeCuPanel(
        *zip(*rows, strict=True),
        dates,
        provenance={"type": "synthetic correlated residual smoke", "seed": seed},
    )


def run(output, export=None):
    from spde_covariance.empirical_short_history import (
        export_short_history_evidence,
        run_short_history_benchmark,
    )

    config = ShortHistoryConfig(
        name="short-history-synthetic-smoke",
        training_end="2022-01-31",
        validation_end="2022-02-28",
        test_end="2022-03-31",
        window=20,
        min_training_intervals=16,
        origin_stride=10,
        bootstrap_replications=50,
        sensitivity_half_lives=(),
    )
    result = run_short_history_benchmark(synthetic_panel(config.seed), config, output)
    if export is not None:
        export_short_history_evidence(output, export)
    print("Offline short-history workflow completed:", output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new local run directory")
    parser.add_argument("--export", type=Path, help="new aggregate-only evidence directory")
    arguments = parser.parse_args()
    if arguments.output is None:
        with TemporaryDirectory(prefix="spde-short-history-") as temporary:
            run(Path(temporary) / "run", arguments.export)
    else:
        run(arguments.output, arguments.export)


if __name__ == "__main__":
    main()
