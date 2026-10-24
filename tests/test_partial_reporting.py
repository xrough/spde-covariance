"""Coverage, target separation and unavailable-score checks for partial reports."""

from copy import deepcopy
from itertools import product

import matplotlib.image as mpimg
import numpy as np
import pytest

from spde_covariance import partial_reporting
from spde_covariance.partial_reporting import write_report


@pytest.fixture
def report_payload():
    config = {
        "name": "partial-report-test",
        "seed": 1729,
        "replications": 2,
        "scenarios": ("powerlaw", "rotated"),
        "designs": ("fixed_points", "fixed_deliveries"),
        "training_counts": (8, 16),
        "test_count": 8,
        "validation_count": 8,
        "candidate_dimensions": (4, 8),
        "geometry_dimensions": (4, 8, 16),
        "relative_regularizations": (0.001, 0.01),
    }
    results = {
        name: []
        for name in (
            "geometry",
            "observation_approximation",
            "selected_summary",
            "risk_summary",
            "paired_differences",
            "selections",
            "candidate_summary",
            "full_h_geometry",
        )
    }
    targets = ("point_spread", "delivery_spread", "offgrid_spread")
    for design, target in product(config["designs"], targets):
        available = (design == "fixed_points" and target == "point_spread") or (
            design == "fixed_deliveries" and target == "delivery_spread"
        )
        row = {
            "design": design,
            "target": target,
            "contract_indices": (0, 1) if design == "fixed_points" else (2, 3),
            "available": available,
            "relative_residual": 0 if available else 0.2,
        }
        results["full_h_geometry"].append({**row, "kind": "same_date"})
        for size in config["training_counts"]:
            results["full_h_geometry"].append(
                {**row, "kind": "training_union_functional_span", "N": size}
            )
    for design, size, dimension in product(
        config["designs"], config["training_counts"], config["geometry_dimensions"]
    ):
        results["geometry"].append(
            {
                "design": design,
                "N": size,
                "J": dimension,
                "rank": 3,
                "dimension": dimension * (dimension + 1) // 2,
                "smallest_positive_eigenvalue": 0.001 / dimension,
                "targets": {
                    target: {"nullspace_relative_residual": index / 3}
                    for index, target in enumerate(targets)
                },
            }
        )
    for scenario, design, size in product(
        config["scenarios"], config["designs"], config["training_counts"]
    ):
        common = {"scenario": scenario, "design": design, "N": size}
        for dimension, penalty in product(
            config["candidate_dimensions"], config["relative_regularizations"]
        ):
            results["candidate_summary"].append(
                {**common, "J": dimension, "relative_regularization": penalty, "replications": 2}
            )
        for dimension in config["geometry_dimensions"]:
            results["observation_approximation"].append(
                {**common, "J": dimension, "relative_hs_error": 0.01 / dimension}
            )
        for replication in range(config["replications"]):
            results["selections"].append(
                {
                    **common,
                    "replication": replication,
                    "selected_J": 4 if replication == 0 else 8,
                    "selected_relative_regularization": 0.01,
                }
            )
        for rule in ("selected", "fixed"):
            results["selected_summary"].append(
                {
                    **common,
                    "rule": rule,
                    "replications": 2,
                    "test_observed_score_mean": -2.4,
                    "test_observed_score_mcse": 0.1,
                    "operator_hs_error_mean": 0.2,
                    "operator_hs_error_mcse": 0.02,
                }
            )
        for target in targets:
            available = int(
                (design == "fixed_points" and target == "point_spread")
                or (design == "fixed_deliveries" and target == "delivery_spread")
            )
            for rule in ("selected", "fixed", "direct"):
                results["risk_summary"].append(
                    {
                        **common,
                        "rule": rule,
                        "target": target,
                        "replications": 2,
                        "available_test_dates": 8 * available,
                        "risk_score_mean": -0.4 if available else None,
                        "risk_score_mcse": 0.02 if available else None,
                        "risk_mse_mean": 0.01 if rule != "direct" or available else None,
                        "risk_mse_mcse": 0.001 if rule != "direct" or available else None,
                    }
                )
            results["paired_differences"].append(
                {
                    **common,
                    "target": target,
                    "metric": "risk_score",
                    "comparison_rule": "selected",
                    "baseline_rule": "fixed",
                    "available_replications": 2 * available,
                    "mean_difference": -0.02 if available else None,
                    "mcse": 0.01 if available else None,
                }
            )
        results["paired_differences"].append(
            {
                **common,
                "target": None,
                "metric": "observed_score",
                "comparison_rule": "selected",
                "baseline_rule": "fixed",
                "replications": 2,
                "mean_difference": 0.02,
                "mcse": 0.01,
            }
        )
    results.update(
        acceptance={"passed": True, "statistical_performance_gate": None},
        reference_checks=[{"converged": True}],
        model={
            "reference_dimension": 256,
            "tail_bounds": {"powerlaw": {"full_innovation_trace_bound": 6.4e-6}},
        },
        protocol={"validation_refit": False},
    )
    return config, results


def test_report_preserves_targets_complete_cases_and_unavailable_scores(tmp_path, report_payload):
    config, results = report_payload
    report = write_report(tmp_path, config, results).read_text()
    for phrase in (
        "fixed interval",
        "does not estimate source C",
        "not finite-coordinate coincidence",
        "retained-omitted cross terms",
        "not independent omitted-coordinate noise",
        "not a pooled covariance nonidentification proof",
        "Selected latent HS error",
        "relative penalty 0.001",
        "same candidate fits",
        "No winner",
        "unavailable",
    ):
        assert phrase in report
    assert "J=8" in report  # Smoke baseline follows its candidate grid.
    for scenario, design, size in product(
        config["scenarios"], config["designs"], config["training_counts"]
    ):
        assert (
            f"| {partial_reporting._label(scenario)} | {partial_reporting._label(design)} | {size} |"
            in report
        )
    for target in partial_reporting._TARGETS:
        assert partial_reporting._label(target) in report
    for name in (
        "geometry_observability.png",
        "observation_approximation.png",
        "selected_performance.png",
    ):
        image = mpimg.imread(tmp_path / name)
        assert image.shape[0] >= 500
        assert image.shape[1] >= 1000
        assert np.isfinite(image).all()


@pytest.mark.parametrize(
    "corruption",
    (
        "case",
        "geometry",
        "approximation",
        "risk",
        "selection",
        "duplicate",
        "paired",
        "unavailable",
        "nonfinite",
    ),
)
def test_incomplete_or_contradictory_results_fail_before_output(
    tmp_path, report_payload, corruption
):
    config, original = report_payload
    results = deepcopy(original)
    if corruption == "case":
        results["selected_summary"][0]["design"] = "undeclared"
    elif corruption == "geometry":
        results["geometry"].pop()
    elif corruption == "approximation":
        results["observation_approximation"].pop()
    elif corruption == "risk":
        results["risk_summary"].pop()
    elif corruption == "selection":
        results["selections"][0]["selected_J"] = 3
    elif corruption == "duplicate":
        results["selections"].append(results["selections"][0].copy())
    elif corruption == "paired":
        results["paired_differences"].pop()
    elif corruption == "unavailable":
        row = next(row for row in results["risk_summary"] if row["available_test_dates"] == 0)
        row["risk_score_mean"] = 0
    elif corruption == "nonfinite":
        results["geometry"][0]["smallest_positive_eigenvalue"] = float("inf")
    with pytest.raises(ValueError):
        write_report(tmp_path / "invalid", config, results)
    assert not (tmp_path / "invalid").exists()


def test_fixed_panels_and_captions_survive_a_reduced_ci_case(tmp_path, report_payload, monkeypatch):
    config, results = report_payload
    config["scenarios"] = ("powerlaw",)
    for name, rows in results.items():
        if isinstance(rows, list):
            results[name] = [row for row in rows if row.get("scenario", "powerlaw") == "powerlaw"]
    checked = []

    def inspect(figure, output, name):
        figure.canvas.draw()
        bounds = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.width
        assert bounds.y1 <= figure.bbox.height
        if name == "selected_performance.png":
            assert len(figure.axes) == 4
            assert any(text.get_text() == "unavailable" for text in figure.axes[-1].texts)
        checked.append(name)
        partial_reporting.plt.close(figure)

    monkeypatch.setattr(partial_reporting, "_save", inspect)
    write_report(tmp_path, config, results)
    assert checked == [
        "geometry_observability.png",
        "observation_approximation.png",
        "selected_performance.png",
    ]
