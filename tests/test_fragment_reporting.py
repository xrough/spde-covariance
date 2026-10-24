import copy
import json
from types import SimpleNamespace

import matplotlib.image as mpimg
import numpy as np
import pytest

from spde_covariance.fragment_reporting import (
    _estimate,
    _risk_display,
    _selected_example,
    write_fragment_report,
)


def _fixture():
    config = SimpleNamespace(
        profile="spde",
        seed=7,
        replications=2,
        models=("powerlaw",),
        grids=(11, 21, 41),
        training_counts=(16, 32, 64),
        covers=(2,),
        noise_variances=(0.001,),
    )
    summary, risks, paired = [], [], []
    metrics = (
        "observed_weighted_error",
        "missing_weighted_error",
        "integrated_squared_error",
        "canonical_weighted_error",
        "observed_integrated_squared_error",
        "missing_integrated_squared_error",
    )
    for q in config.grids:
        for n in config.training_counts:
            for variant in ("raw", "psd"):
                common = dict(
                    model="powerlaw",
                    noise_variance=0.001,
                    base_grid=q,
                    cover=2,
                    N=n,
                    method="canonical",
                    policy="rank2",
                    variant=variant,
                )
                row = common | dict(
                    replications=2, available_replications=2, unavailable_replications=0
                )
                for metric in metrics:
                    for suffix, value in (
                        ("mean", 0.1),
                        ("mcse", 0.01),
                        ("median", 0.09),
                        ("mad", 0.005),
                        ("mean_absolute_deviation_from_median", 0.008),
                    ):
                        row[metric + "_" + suffix] = value
                summary.append(row)
                for target in ("point_spread", "long_spread"):
                    risks.append(
                        common
                        | dict(
                            target=target,
                            replications=2,
                            available_test_dates_mean=0 if target == "long_spread" else 16,
                            risk_squared_error_mean=0.02,
                            risk_squared_error_mcse=0.001,
                            risk_squared_error_median=0.018,
                            risk_squared_error_mean_absolute_deviation_from_median=0.003,
                            risk_score_mean=None if target == "long_spread" else -0.1,
                            risk_score_mcse=None if target == "long_spread" else 0.01,
                        )
                    )
                paired.append(
                    common
                    | dict(
                        paired_replications=2,
                        score_difference_mean=0.0 if variant == "raw" else 0.001,
                        score_difference_mcse=0.0001,
                    )
                )
    nodes = np.linspace(0, 3, 21)
    truth = np.exp(-((nodes[:, None] - nodes[None, :]) ** 2))
    mask = np.abs(nodes[:, None] - nodes[None, :]) < 2
    examples = [
        dict(
            model="powerlaw",
            noise_variance=0.001,
            base_grid=21,
            cover=2,
            N=32,
            method="canonical",
            policy="rank2",
            variant=variant,
            replication=0,
            nodes=nodes,
            truth=truth,
            canonical=truth * 0.98,
            kernel=truth * 0.99,
            mask=mask,
        )
        for variant in ("raw", "psd")
    ]
    results = dict(
        summary=summary,
        risk_summary=risks,
        paired_differences=paired,
        acceptance={"passed": True, "raw_indefiniteness_is_allowed": True},
        available_fits=36,
        unavailable_fits=0,
    )
    population = dict(
        controls={
            "paper": [],
            "noncanonical": {"risk_range": {"lower": 0.0, "upper": 4.0}},
            "truncation": {"rank1_min_eigenvalue": 1 - np.sqrt(2)},
        },
        sensitivity=[],
    )
    return config, results, examples, population


def test_report_writes_plain_finite_figures_all_cells_and_scope_disclosures(tmp_path):
    config, results, examples, population = _fixture()
    results["summary"][0]["runtime_seconds_mean"] = 0.125
    (tmp_path / "completion_audit.json").write_text(
        json.dumps(
            [
                {
                    "policy": "fve95",
                    "diagnostics": {"available": False, "reason": "indefinite overlap"},
                },
                {
                    "policy": "rank2",
                    "diagnostics": {
                        "available": True,
                        "joins": [
                            {
                                "selected_rank": 2,
                                "retained_condition_number": 4.0,
                                "inverse_norm": 8.0,
                                "inverse_perturbation_multiplier": 16.0,
                            }
                        ],
                    },
                },
                {
                    "projection": {
                        "weighted_hs_change": 0.02,
                        "observed_weighted_hs_change": 0.01,
                        "observed_maximum_change": 0.03,
                        "weighted_trace_change": 0.07,
                        "kernel_diagonal_sum_change": 0.08,
                    }
                },
            ]
        )
    )
    path = write_fragment_report(config, tmp_path, results, examples, population)
    text = path.read_text()
    for name in ("fragment_errors.png", "fragment_kernels.png", "fragment_risks.png"):
        image = mpimg.imread(tmp_path / name)
        assert image.shape[0] > 300 and image.shape[1] > 500
        assert np.isfinite(image).all() and image.std() > 0.01
    assert "11" in text and "41" in text and "indefinite overlap" in text
    assert "not a completion-error certificate" in text
    assert "cannot tune those ranks" in text and "Available/unavailable reps" in text
    assert "finite point measurements" in text and "manifest.json" in text
    assert "0.03" in text
    assert "weighted_trace_change" in text and "inverse_perturbation_multiplier" in text
    assert "0.125" in text and "hardware-independent" in text


def test_representative_selection_is_predeclared_and_does_not_choose_an_available_winner():
    config, _, examples, _ = _fixture()
    assert _selected_example(config, examples, "powerlaw", "raw") is examples[0]
    changed = copy.deepcopy(examples)
    changed[0]["N"] = 64
    assert _selected_example(config, changed, "powerlaw", "raw") is None


def test_reference_absolute_deviation_center_is_explicit():
    row = {
        "error_median": 5.0,
        "error_mad": 1.0,
        "error_mean_absolute_deviation_from_median": 2.0,
        "error_mean_absolute_deviation_from_mean": 3.0,
    }
    assert _estimate(row, "error", reference=True) == "5 ± 2"


def test_risk_display_is_fixed_by_model_rank_and_keeps_ridge():
    row = dict(model="powerlaw", method="canonical", policy="rank2", variant="raw")
    assert _risk_display(row)
    assert not _risk_display(row | {"policy": "rank4", "risk_squared_error_mean": 0.0})
    assert _risk_display(row | {"method": "ridge", "policy": "validation_selected"})
    assert _risk_display(row | {"model": "finite_rank", "policy": "rank4"})


@pytest.mark.parametrize("change", ["nan", "undeclared_grid"])
def test_invalid_measurements_or_unconfigured_cells_are_rejected(tmp_path, change):
    config, results, examples, population = _fixture()
    if change == "nan":
        results["summary"][0]["missing_weighted_error_mean"] = np.nan
    else:
        results["summary"][0]["base_grid"] = 999
    with pytest.raises(ValueError):
        write_fragment_report(config, tmp_path, results, examples, population)


def test_unavailable_cells_and_missing_representative_remain_visible(tmp_path):
    config, results, _, population = _fixture()
    for row in results["summary"]:
        row.update(available_replications=0, unavailable_replications=2)
        for key in list(row):
            if any(key.endswith("_" + suffix) for suffix in ("mean", "mcse", "median", "mad")):
                row[key] = None
    results.update(risk_summary=[], paired_differences=[], available_fits=0, unavailable_fits=36)
    path = write_fragment_report(config, tmp_path, results, [], population)
    assert "0/2" in path.read_text() and "unavailable" in path.read_text()
