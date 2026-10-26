"""Offline synthetic smoke for the entire empirical calibration/risk pipeline.

This contains no market data. Use empirical-shfe with local archives for the
separate historical study; this example verifies installation and chronology.
"""

from tempfile import TemporaryDirectory

import numpy as np

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_study import EmpiricalConfig, run_empirical_study
from spde_covariance.laguerre import LaguerreTransport


def synthetic_panel():
    dates = np.arange(np.datetime64("2022-01-03"), np.datetime64("2022-04-01"))
    dates = dates[np.is_busday(dates)]
    model = LaguerreTransport(3)
    generator = np.random.default_rng(1729)
    state = np.array([70000.0, 1000.0, 200.0])
    source = np.diag([4e6, 5e5, 2e5])
    months = np.arange(np.datetime64("2022-02", "M"), np.datetime64("2023-05", "M"))
    ids = np.array(["cu" + str(month)[2:4] + str(month)[5:7] for month in months])
    maturities = months.astype("datetime64[D]") + np.timedelta64(14, "D")
    rows = []
    for i, day in enumerate(dates):
        if i:
            delta = (day - dates[i - 1]).astype(int) / 365.25
            covariance = delta * model.innovation_covariance(source, delta)
            state = model.matrix(delta) @ state + generator.multivariate_normal(
                np.zeros(3), covariance
            )
        for key, maturity in zip(ids, maturities, strict=True):
            if maturity > day:
                value = float(model.evaluate([(maturity - day).astype(int) / 365.25])[0] @ state)
                rows.append((day, key, maturity, value, 100.0, 1000.0))
    return ShfeCuPanel(
        *zip(*rows, strict=True),
        dates,
        provenance={"type": "synthetic offline control", "seed": 1729},
    )


def main():
    config = EmpiricalConfig(
        training_end="2022-01-31",
        validation_end="2022-02-28",
        test_end="2022-03-31",
        windows=(8, 16),
        min_direct_count=3,
        bootstrap_replications=100,
        sensitivities=False,
    )
    with TemporaryDirectory() as directory:
        summary = run_empirical_study(synthetic_panel(), config, directory + "/run")
    assert summary["numerical_diagnostics"]["all_converged"]
    assert summary["selection"]["selection_last_outcome"] <= config.validation_end
    assert summary["test_origin_count"] > 10
    print(
        "Offline synthetic empirical workflow passed:", summary["test_origin_count"], "test origins"
    )


if __name__ == "__main__":
    main()
