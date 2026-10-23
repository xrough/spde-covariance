"""Fixed selections, target labels and complete-case checks for the foundation report."""

from copy import deepcopy
from itertools import product

import matplotlib.image as mpimg
import numpy as np
import pytest

from spde_covariance import infinite_reporting
from spde_covariance.infinite_reporting import write_report


@pytest.fixture
def report_payload():
    config = {
        "name": "infinite-report-test",
        "seed": 1729,
        "replications": 2,
        "horizon": 1.0,
        "forcing_dimensions": (4, 8),
        "projection_dimensions": (2, 4),
        "deltas": (0.25, 0.125),
        "source_exponent": 3,
        "source_trace": 1,
    }
    results = {"population": [], "summary": [], "risk_summary": []}
    for level, dimension, delta in product(
        config["forcing_dimensions"], config["projection_dimensions"], config["deltas"]
    ):
        common = {"M": level, "J": dimension, "delta": delta}
        results["population"].append(
            {
                **common,
                "intervals": round(1 / delta),
                "source_trace": 0.99,
                "innovation_trace": 1.01,
                "state_trace": 1.2,
                "full_source_tail_bound": 0.01 / level,
                "full_innovation_tail_bound": 0.011 / level,
                "full_state_mse_tail_bound": 0.014 / level,
                "projected_innovation_tail_bound": 0.001 * dimension / level,
                "operational_covariance_tail_bound_max": 0.002 * dimension / level,
                "reference_M": 8,
                "forcing_innovation_hs_error": (8 - level) * 1e-4 / dimension,
                "forcing_state_hs_error": (8 - level) * 2e-4 / dimension,
                "projection_innovation_hs_error": 0.003 / dimension,
                "source_to_innovation_hs_error": delta * 0.1,
                "leakage_mean_hs": delta * 0.002 / dimension,
                "leakage_terminal_hs": delta * 0.003 / dimension,
                "leakage_mean_trace": delta * 0.004 / dimension,
                "tail_resolved": False,
            }
        )
        row = {**common, "replications": 2, "oracle_gaussian_mse": 0.003 * delta}
        for metric in (
            "oracle_covariance_mse",
            "operational_covariance_mse",
            "operational_centered_covariance_mse",
            "state_terminal_coupling_error_squared",
            "projected_state_terminal_error_squared",
            "projection_terminal_error_squared",
            "leakage_sample_mse",
        ):
            row[metric + "_mean"] = 0.003 * delta / dimension
            row[metric + "_mcse"] = 0.0002 * delta / dimension
        row["operational_minus_oracle_covariance_mse_mean"] = -0.001
        row["operational_minus_oracle_covariance_mse_mcse"] = 0.0001
        results["summary"].append(row)
        for exposure in ("near_spread", "butterfly"):
            results["risk_summary"].append(
                {
                    **common,
                    "exposure": exposure,
                    "replications": 2,
                    "oracle_risk_mse_mean": 0.004 / dimension,
                    "oracle_risk_mse_mcse": 0.0003 / dimension,
                    "operational_risk_mse_mean": 0.005 / dimension,
                    "operational_risk_mse_mcse": 0.0004 / dimension,
                    "projected_risk_truth": 0.2,
                    "reference_risk_truth": 0.21,
                    "projection_bias": -0.01,
                    "forcing_bias": -0.001,
                    "leakage_bias": 0.0001,
                    "oracle_gaussian_risk_mse": 0.004 / dimension,
                    "operational_minus_oracle_risk_mse_mean": -0.002,
                    "operational_minus_oracle_risk_mse_mcse": 0.0002,
                }
            )
    grid = np.linspace(0, 2, 11)
    kernel = np.exp(-np.abs(grid[:, None] - grid[None, :]))
    results.update(
        kernels={
            "maturities": grid.tolist(),
            "source": kernel.tolist(),
            "innovation": (kernel * 1.01).tolist(),
            "state": (kernel * 1.2).tolist(),
            "reference_M": 8,
            "delta": 0.25,
            "horizon": 1,
        },
        acceptance={"numerical_passed": True, "reference_resolution_passed": False},
        numerical_checks={
            "covariance_discrepancy": 1e-14,
            "numerical_error_certificate": False,
        },
        fit_summary=[{"J": 2, "max_covariance_error_bound": 1e-5, "converged": True}],
        paired_differences=[],
        model={"basis": "Laguerre", "source": "power law"},
        protocol={"coupling": "shared forcing"},
    )
    return config, results


def test_report_keeps_targets_bounds_and_all_configured_cases_distinct(tmp_path, report_payload):
    config, results = report_payload
    path = write_report(tmp_path, config, results)
    report = path.read_text()
    for phrase in (
        "foundation benchmark",
        "not a new estimator theorem",
        "weights are never renormalized",
        "not J-projected kernels",
        "Full source trace tail",
        "Full state MSE tail bound",
        "not full-state trace bounds",
        "Operational vs Rbar",
        "V projection error",
        "past-dependent conditional mean",
        "without renormalization",
        "Signed projection bias",
        "not statistical error",
        "reference_resolution_passed | False",
        "numerical_error_certificate | False",
        "Paired error differences at reference forcing",
        "not a winner acceptance gate",
        "-0.001 ± 0.0001",
        "-0.002 ± 0.0002",
    ):
        assert phrase in report
    for level, dimension, delta in product(
        config["forcing_dimensions"], config["projection_dimensions"], config["deltas"]
    ):
        assert f"| {level} | {dimension} | {delta} |" in report
    for exposure in ("near_spread", "butterfly"):
        assert exposure in report
    for name in ("covariance_kernels.png", "resolution_errors.png", "leakage_risk.png"):
        image = mpimg.imread(tmp_path / name)
        assert image.shape[0] >= 500
        assert image.shape[1] >= 1000
        assert np.isfinite(image).all()
        assert image.min() < 0.5


@pytest.mark.parametrize(
    "corruption",
    ("case", "missing", "duplicate", "risk", "nonfinite", "count", "kernel", "negative"),
)
def test_invalid_results_fail_before_writing_any_artifact(tmp_path, report_payload, corruption):
    config, original = report_payload
    results = deepcopy(original)
    if corruption == "case":
        results["population"][0]["J"] = 9
    elif corruption == "missing":
        results["summary"].pop()
    elif corruption == "duplicate":
        results["population"].append(results["population"][0].copy())
    elif corruption == "risk":
        results["risk_summary"].pop()
    elif corruption == "nonfinite":
        results["summary"][0]["oracle_covariance_mse_mean"] = float("nan")
    elif corruption == "count":
        results["summary"][0]["replications"] = 1
    elif corruption == "kernel":
        results["kernels"]["delta"] = min(config["deltas"])
    elif corruption == "negative":
        results["summary"][0]["oracle_covariance_mse_mcse"] = -1
    with pytest.raises(ValueError):
        write_report(tmp_path / "invalid", config, results)
    assert not (tmp_path / "invalid").exists()


def test_plot_selections_include_all_exposures_and_never_choose_by_error(
    tmp_path, report_payload, monkeypatch
):
    config, results = report_payload
    checked = {}

    def inspect(figure, output, name):
        figure.canvas.draw()
        bounds = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0 >= 0
        assert bounds.x1 <= figure.bbox.width
        assert bounds.y1 <= figure.bbox.height
        checked[name] = figure

    monkeypatch.setattr(infinite_reporting, "_save", inspect)
    write_report(tmp_path, config, results)
    resolution = checked["resolution_errors.png"]
    assert len(resolution.axes[0].lines) == 4  # Every J × Δ at all forcing levels.
    assert all(tuple(line.get_xdata()) == (4, 8) for line in resolution.axes[0].lines)
    leakage = checked["leakage_risk.png"]
    labels = {line.get_label() for axis in leakage.axes for line in axis.lines}
    # errorbar labels are carried on containers, not their internal plot lines.
    labels.update(label for axis in leakage.axes for label in axis.get_legend_handles_labels()[1])
    for exposure in ("near spread", "butterfly"):
        assert f"{exposure}: oracle innovation" in labels
        assert f"{exposure}: projected-level residual" in labels
    for figure in checked.values():
        infinite_reporting.plt.close(figure)


def test_configurations_with_projections_above_small_forcing_levels_are_supported(
    tmp_path, report_payload, monkeypatch
):
    config, results = report_payload
    config["forcing_dimensions"] = (2, 4)
    for name in ("population", "summary", "risk_summary"):
        for row in results[name]:
            row["M"] //= 2
            if "reference_M" in row:
                row["reference_M"] = 4
        results[name] = [row for row in results[name] if row["J"] <= row["M"]]
    results["kernels"]["reference_M"] = 4
    monkeypatch.setattr(
        infinite_reporting, "_save", lambda figure, *_: infinite_reporting.plt.close(figure)
    )
    report = write_report(tmp_path, config, results).read_text()
    assert "| 2 | 2 | 0.125 |" in report
    assert "| 4 | 4 | 0.125 |" in report
    assert "| 2 | 4 | 0.125 |" not in report
