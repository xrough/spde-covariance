"""Checks for the RQ3 observed-score report and its fixed figure selections."""

from copy import deepcopy

import matplotlib.image as mpimg
import numpy as np
import pytest

from spde_covariance import rq3_reporting
from spde_covariance.rq3_reporting import write_report


@pytest.fixture
def report_payload():
    config = {
        "name": "rq3-report-test",
        "seed": 1729,
        "replications": 2,
        "intervals": 512,
        "delta": 1 / 52,
        "windows": (16, 32, 64, 128),
        "validation_origins": (128, 192),
        "test_origins": (256, 272, 304, 448),
        "horizon": 16,
        "static_window": 128,
        "fixed_window": 128,
        # Intentionally reversed: plots still use the declared canonical order.
        "scenarios": ("abrupt", "constant"),
        "designs": ("full", "dense", "split", "joint8", "joint64"),
    }
    results = {
        name: []
        for name in (
            "summary",
            "risk_summary",
            "truth_summary",
            "paired_differences",
            "selections",
            "horizon_summary",
            "origin_summary",
            "geometry",
        )
    }
    targets = ("near_spread", "cross_spread", "butterfly")
    for scenario in config["scenarios"]:
        for design_index, design in enumerate(config["designs"]):
            for rule, difference in (("static", 0.2), ("fixed", 0), ("selected", -0.1)):
                common = {"scenario": scenario, "design": design, "rule": rule, "replications": 2}
                score = -3 + design_index / 10 + difference
                results["summary"].append(
                    {**common, "observed_score_mean": score, "observed_score_mcse": 0.03}
                )
                for origin in config["test_origins"]:
                    results["origin_summary"].append(
                        {
                            **common,
                            "origin": origin,
                            "observed_score_mean": score,
                            "observed_score_mcse": 0.03,
                        }
                    )
                for horizon in range(1, config["horizon"] + 1):
                    results["horizon_summary"].append(
                        {
                            **common,
                            "horizon": horizon,
                            "observed_score_mean": score + horizon / 100,
                            "observed_score_mcse": 0.03,
                        }
                    )
                truth = common.copy()
                for metric in (
                    "operator_hs_error",
                    "operator_hs_relative_error",
                    "observed_covariance_error",
                    "kernel_grid_rmse",
                    "exposure_risk_rmse",
                    "nullspace_hs_error",
                    "weak_direction_hs_error",
                ):
                    truth[metric + "_mean"], truth[metric + "_mcse"] = 0.12, 0.02
                results["truth_summary"].append(truth)
                if rule != "fixed":
                    results["paired_differences"].append(
                        {
                            "scenario": scenario,
                            "design": design,
                            "comparison_rule": rule,
                            "baseline_rule": "fixed",
                            "metric": "observed_score",
                            "target": None,
                            "replications": 2,
                            "mean_difference": difference,
                            "mcse": 0.01,
                        }
                    )
                for target in targets:
                    available = 128
                    if target != "near_spread":
                        available = {"split": 0, "joint8": 16, "joint64": 2}.get(design, 128)
                    results["risk_summary"].append(
                        {
                            **common,
                            "target": target,
                            "available_scores": available,
                            "available_replications": 2 if available else 0,
                            "risk_score_mean": -0.04 + difference / 10 if available else None,
                            "risk_score_mcse": 0.004 if available else None,
                        }
                    )
                    if rule != "fixed":
                        results["paired_differences"].append(
                            {
                                "scenario": scenario,
                                "design": design,
                                "comparison_rule": rule,
                                "baseline_rule": "fixed",
                                "metric": "risk_score",
                                "target": target,
                                "replications": 2 if available else 0,
                                "mean_difference": difference / 10 if available else None,
                                "mcse": 0.002 if available else None,
                            }
                        )
            for replication, window in enumerate((16, 128)):
                results["selections"].append(
                    {
                        "scenario": scenario,
                        "design": design,
                        "replication": replication,
                        "selected_window": window,
                        "selection_complete_interval": 208,
                        "validation_origins": [128, 192],
                        "candidates": [
                            {
                                "window": candidate,
                                "validation_score": -1.0,
                                "validation_scores": [-1.0] * 32,
                            }
                            for candidate in config["windows"]
                        ],
                    }
                )
    for design in config["designs"]:
        for origin in (*config["validation_origins"], *config["test_origins"]):
            for window in config["windows"]:
                results["geometry"].append(
                    {
                        "design": design,
                        "origin": origin,
                        "window": window,
                        "training_first_interval": origin - window + 1,
                        "training_last_interval": origin,
                        "rank": 6 if design == "split" else 10,
                        "eigenvalues": [1.0] * 10,
                        "targets": {},
                        "regularization": 0.001,
                    }
                )
    results["acceptance"] = {
        "passed": True,
        "failed_fits": 0,
        "failures": [],
        "feasibility_passed": True,
        "reference_passed": True,
        "reference_checks_requested": False,
        "reference_cases": 0,
        "reference_cases_expected": 0,
        "maximum_covariance_error_bound": 2e-8,
        "numerical_error_budget": 1e-4,
        "statistical_performance_gate": None,
    }
    results["model"] = {"basis_id": "test-orthonormal-basis", "exposures": {}}
    results["protocol"] = {}
    return config, results


def test_report_writes_finite_white_figures_and_honest_complete_tables(tmp_path, report_payload):
    config, results = report_payload
    output = tmp_path / "experiments" / "results" / "rq3"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "research-question-3.md").write_text("Protocol\n")
    (docs / "methods.md").write_text("Methods\n")
    write_report(output, config, results)
    for name in ("forecast_scores.png", "horizon_sensitivity.png", "risk_evaluation.png"):
        path = output / name
        assert path.stat().st_size > 10_000
        pixels = mpimg.imread(path)
        assert np.isfinite(pixels).all()
        np.testing.assert_allclose(pixels[0, 0, :3], 1)
        assert np.ptp(pixels[..., :3]) > 0.8
    report = (output / "report.md").read_text()
    assert "-3.1 ± 0.03" in report  # Negative reduced scores are retained.
    assert "| Constant covariance | Identity | 1 | 0 | 0 | 1 | 2 |" in report
    assert "| Joint every 64 | Cross-pair spread | 1 | 2 / 128 | 2 |" in report
    assert "| Separate pairs | Cross-pair spread | 0 | 0 / 128 | 0 | unavailable" in report
    assert "full rank while most of their future cross-spread proxies remain unavailable" in report
    assert "Fit intervals 1 through 128 once" in report
    assert "refit that chosen trailing window at each test origin" in report
    assert "Selection is frozen before the first test origin 256" in report
    assert "| 448 | 449 through 464 |" in report
    assert "Simulation-only latent diagnostics" in report
    assert "no statistical superiority gate" in report
    assert "replication-cluster" in report
    assert "not normalized by the number of future measurements" in report
    assert "```math" in report
    assert "\\operatorname" not in report
    for name in ("research-question-3.md", "methods.md"):
        relative = f"../../../docs/{name}"
        assert f"]({relative})" in report
        assert (output / relative).resolve() == (docs / name).resolve()


def test_latent_errors_cannot_select_figures_or_change_observed_comparisons(
    tmp_path, report_payload
):
    config, results = report_payload
    first, second = tmp_path / "first", tmp_path / "second"
    write_report(first, config, results)
    changed = deepcopy(results)
    for row in changed["truth_summary"]:
        for key in row:
            if key.endswith("_mean"):
                row[key] = 987654.0 if row["scenario"] == "constant" else 0.0
    write_report(second, config, changed)
    for name in ("forecast_scores.png", "horizon_sensitivity.png", "risk_evaluation.png"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    first_report, second_report = (
        (first / "report.md").read_text(),
        (second / "report.md").read_text(),
    )
    before_truth = "## Simulation-only latent diagnostics"
    assert first_report.split(before_truth)[0] == second_report.split(before_truth)[0]
    assert "987654" not in first_report
    assert "9.8765e+05" in second_report
    assert first_report.index("Constant covariance") < first_report.index("Abrupt change (stress)")


@pytest.mark.parametrize(
    "problem", ("window", "candidate", "chronology", "nonfinite", "geometry", "unavailable")
)
def test_report_rejects_inconsistent_or_undeclared_payload_before_writing(
    tmp_path, report_payload, problem
):
    config, results = report_payload
    if problem == "window":
        results["selections"][0]["selected_window"] = 8
    elif problem == "candidate":
        results["selections"][0]["candidates"][0]["window"] = 8
    elif problem == "chronology":
        results["selections"][0]["selection_complete_interval"] = 257
    elif problem == "nonfinite":
        results["summary"][0]["observed_score_mean"] = np.nan
    elif problem == "geometry":
        results["geometry"][0]["training_last_interval"] += 1
    else:
        row = next(row for row in results["risk_summary"] if row["available_scores"] == 0)
        row["risk_score_mean"] = 0.0
    with pytest.raises(ValueError):
        write_report(tmp_path / "invalid", config, results)
    assert not (tmp_path / "invalid").exists()


def test_report_does_not_silently_select_unconfigured_scenarios(tmp_path, report_payload):
    config, results = report_payload
    config["scenarios"] = ("constant",)
    with pytest.raises(ValueError, match="outside the declared experiment"):
        write_report(tmp_path, config, results)


def test_figure_captions_fit_the_reduced_ci_configuration(tmp_path, report_payload, monkeypatch):
    config, results = report_payload
    config["scenarios"], config["designs"] = ("constant",), ("full", "split")
    for key, rows in results.items():
        if isinstance(rows, list):
            results[key] = [
                row
                for row in rows
                if row.get("scenario", "constant") == "constant"
                and row.get("design", "full") in config["designs"]
            ]
    checked = []

    def inspect_caption(figure, output, name):
        figure.canvas.draw()
        bounds = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0 >= 0
        assert bounds.x1 <= figure.bbox.width
        assert bounds.y1 <= figure.bbox.height
        checked.append(name)
        rq3_reporting.plt.close(figure)

    monkeypatch.setattr(rq3_reporting, "_save", inspect_caption)
    write_report(tmp_path, config, results)
    assert checked == ["forecast_scores.png", "horizon_sensitivity.png", "risk_evaluation.png"]
