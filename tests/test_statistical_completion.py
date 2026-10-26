"""Fixed-data chronology, equal-law controls and separate coverage accounting."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from spde_covariance.observation_statistics import partial_kernel_from_arrays
from spde_covariance.statistical_completion_benchmark import (
    METHODS,
    NATIVE_CLIQUES,
    WORLDS,
    StatisticalCompletionConfig,
    _native_values,
    _paired,
    _summary,
    coverage_experiment,
    fit_statistical_candidates,
    load_statistical_completion_config,
    paired_completion_parents,
    population_completion_controls,
    run_statistical_completion,
    training_reference,
)


@pytest.fixture(scope="module")
def small_config():
    return StatisticalCompletionConfig(
        replications=2,
        training_counts=(16, 32),
        validation_count=16,
        test_count=32,
        source_dimension=8,
        reference_dimension=16,
        coverage_replications=3,
        coverage_training_count=32,
        coverage_precision_required=False,
    )


@pytest.fixture(scope="module")
def population(small_config):
    return population_completion_controls(small_config)


def test_population_controls_are_true_point_equivalent_but_not_same_missing_truth(population):
    minus, plus, canonical = (population["kernels"][world] for world in WORLDS)
    mask = population["observed_mask"]
    np.testing.assert_allclose(minus[mask], plus[mask], atol=1e-12, rtol=0)
    np.testing.assert_allclose(minus[mask], canonical[mask], atol=1e-12, rtol=0)
    assert np.linalg.norm((minus - plus)[~mask]) > 1e-3
    assert population["canonical_structural_error"]["canonical_control"] == 0
    assert population["canonical_bridge_gaussian_kl"]["canonical_control"] < 1e-10
    assert population["canonical_bridge_gaussian_kl"]["source_minus"] > 0.01
    assert population["source_pair"]["main_projected_benchmark_reinterpreted"] is False
    for row in population["identification_ranges"]:
        assert row["available"]
        for value in row["world_values"].values():
            assert row["lower"] - 1e-10 <= value <= row["upper"] + 1e-10
    across = next(
        row for row in population["identification_ranges"] if row["target"] == "across_fragments"
    )
    assert across["upper"] - across["lower"] > 0.01


def test_conditional_coupling_preserves_each_marginal_and_exact_native_data():
    first = np.array([[1.0, 0.4, 0.16], [0.4, 1, 0.4], [0.16, 0.4, 1]])
    second = first.copy()
    second[0, 2] = second[2, 0] = 0.65
    groups = ((0, 1), (1, 2))
    parents = paired_completion_parents(
        (first, second), groups, count=16000, noise_variance=0.1, seed=41
    )
    for i, group in enumerate(groups):
        np.testing.assert_array_equal(parents[0][i::2, group], parents[1][i::2, group])
    for parent, truth in zip(parents, (first, second), strict=True):
        target = truth + 0.1 * np.eye(3)
        observed = parent.T @ parent / len(parent)
        standard_error = np.sqrt(
            (np.outer(np.diag(target), np.diag(target)) + target**2) / len(parent)
        )
        assert np.max(np.abs(observed - target) / standard_error) < 5


def test_coupling_rejects_different_native_laws():
    first, second = np.eye(3), np.diag([1, 2, 1])
    with pytest.raises(ValueError, match="same native"):
        paired_completion_parents(
            (first, second), ((0, 1), (1, 2)), count=4, noise_variance=0, seed=1
        )


def test_training_reference_uses_only_corrected_training_diagonal():
    rows = (np.array([0, 1]), np.array([1, 2]))
    statistics = partial_kernel_from_arrays(
        (np.array([0.1, 2]), np.array([2, 3])), rows, noise_variance=0.2, size=3
    )
    reference, metadata = training_reference(statistics, relative_floor=1e-4)
    assert reference[0, 0] == metadata["variance_floor"]
    assert reference[1, 1] == pytest.approx(3.8)
    assert reference[2, 2] == pytest.approx(8.8)
    np.testing.assert_array_equal(reference, np.diag(np.diag(reference)))
    assert all(
        "truth" not in name and "world" not in name
        for name in inspect.signature(fit_statistical_candidates).parameters
    )


def test_observed_only_fits_have_same_inputs_and_no_future_dependency(small_config, population):
    kernels = tuple(population["kernels"][world] for world in WORLDS)
    parents = paired_completion_parents(
        kernels,
        NATIVE_CLIQUES,
        count=small_config.total_count,
        noise_variance=small_config.noise_variance,
        seed=42,
    )
    training, rows = _native_values(parents[0], 0, 16)
    validation, validation_rows = _native_values(parents[0], 32, 16)
    fits, candidates, selections = fit_statistical_candidates(
        training,
        rows,
        validation,
        validation_rows,
        config=small_config,
        weights=population["weights"],
    )
    assert all(fits[m] is not None for m in METHODS)
    assert len(candidates) == len(METHODS) * len(small_config.prior_counts)
    for method in METHODS:
        assert candidates[selections[method]]["available"]
        assert np.linalg.eigvalsh(fits[method])[0] >= -1e-10
    for parent in parents:
        other_training, _ = _native_values(parent, 0, 16)
        for a, b in zip(training, other_training, strict=True):
            np.testing.assert_array_equal(a, b)
    # Future and never-observed values can be arbitrary without changing fit inputs.
    altered = parents[0].copy()
    altered[48:] = 1e9
    for i in range(48):
        missing = sorted(set(range(7)) - set(NATIVE_CLIQUES[i % 2]))
        altered[i, missing] = -1e9
    repeated = fit_statistical_candidates(
        *_native_values(altered, 0, 16),
        *_native_values(altered, 32, 16),
        config=small_config,
        weights=population["weights"],
    )
    for method in METHODS:
        np.testing.assert_array_equal(fits[method], repeated[0][method])


def test_coverage_experiment_separates_mc_precision_and_unidentified_risk(small_config):
    pytest.importorskip("cvxpy")
    coverage = coverage_experiment(small_config)
    assert coverage["precision_goal_met"] is False
    assert coverage["nominal_coverage_mcse"] == pytest.approx(np.sqrt(0.95 * 0.05 / 3))
    assert coverage["partial_coverage"]["count"] == 3
    assert coverage["truth_risks"] == pytest.approx({"canonical": 0.84, "noncanonical": 0.35})
    bounds = coverage["population_identification_range"]
    assert bounds["upper"] - bounds["lower"] > 1
    for name in coverage["truth_risks"]:
        assert coverage["risk_coverage_all"][name]["count"] == 3
        assert (
            coverage["risk_coverage_available"][name]["count"]
            == coverage["interval_available_count"]
        )
    for row in coverage["records"]:
        if row["interval_available"]:
            assert row["interval_diagnostics"]["outer_bounds"]
            assert row["lower"] <= row["inner_lower"] + 1e-5
            assert row["upper"] >= row["inner_upper"] - 1e-5


@pytest.fixture(scope="module")
def small_run(tmp_path_factory, small_config):
    pytest.importorskip("cvxpy")
    output = tmp_path_factory.mktemp("statistical-completion") / "run"
    result = run_statistical_completion(small_config, output)
    return output, result


def test_end_to_end_audits_chronology_failure_accounting_and_equal_native_outputs(
    small_run, small_config
):
    output, result = small_run
    assert result["acceptance"]["passed"]
    assert result["native_inputs_identical"]
    assert result["unavailable_selected_count"] == 0
    assert len(result["rows"]) == small_config.replications * len(
        small_config.training_counts
    ) * len(METHODS) * len(WORLDS)
    for issued in result["issued"]:
        assert issued["training_stop"] <= issued["validation_first"]
        assert issued["validation_stop"] == issued["test_first"]
        selected = [
            r
            for r in result["rows"]
            if (r["replication"], r["training_count"], r["method"])
            == (issued["replication"], issued["training_count"], issued["method"])
        ]
        assert len({r["native_nll"] for r in selected}) == 1
        assert len({r["native_test_sha256"] for r in selected}) == 1
        assert len({r["kernel_sha256"] for r in selected}) == 1
    manifest_hash = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    for rep in range(small_config.replications):
        checkpoint = json.loads((output / f"replication-{rep:04d}.json").read_text())
        assert checkpoint["parent_manifest_sha256"] == manifest_hash
    assert result["summary"] == _summary(result["rows"])
    assert result["paired"] == _paired(result["rows"])
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        run_statistical_completion(small_config, output)


def test_plain_report_and_all_predeclared_figures_are_written(small_run):
    output, _ = small_run
    report = (output / "report.md").read_text()
    assert "not a global optimization certificate" in report
    assert "not native tuning data" not in report or "audit" in report
    assert "unavailable counted as misses" in report
    assert "no SPDE-source membership" in report
    for world in ("SPDE source −", "SPDE source +", "Canonical Gaussian control"):
        assert world in report
    for name in ("errors", "scores", "uncertainty"):
        path = output / f"statistical_completion_{name}.png"
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10000
    # JSON encoding rejects NaN and therefore keeps missing metrics explicit.
    decoded = json.loads((output / "results.json").read_text())
    assert decoded["acceptance"]["statistical_superiority_gate"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"training_counts": (16, 16)},
        {"training_counts": (15,)},
        {"replications": True},
        {"noise_variance": -1},
        {"validation_count": 3},
        {"prior_counts": (True,)},
        {"prior_counts": (2, 1)},
        {"source_dimension": 7},
        {"source_dimension": 32, "reference_dimension": 16},
        {"confidence_level": 1},
        {"interval_solver": "unknown"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        StatisticalCompletionConfig(**kwargs)


def test_committed_profiles_lock_full_and_smoke_precision():
    root = Path(__file__).resolve().parents[1]
    full = load_statistical_completion_config(
        root / "experiments/configs/statistical-completion.toml"
    )
    smoke = load_statistical_completion_config(
        root / "experiments/configs/statistical-completion-smoke.toml"
    )
    assert full.coverage_replications == 500
    assert full.coverage_precision_required
    assert full.training_counts == (128, 512, 2048)
    assert smoke.replications == 2 and smoke.coverage_replications == 20
    assert not smoke.coverage_precision_required


def test_failed_candidates_are_retained_and_unavailable_selection_is_explicit(
    monkeypatch, small_config, population
):
    import spde_covariance.gaussian_completion as gaussian

    def failure(*args, **kwargs):
        raise ValueError("declared numerical failure")

    monkeypatch.setattr(gaussian, "fit_gaussian_completion", failure)
    parent = np.random.default_rng(4).normal(size=(48, 7))
    fits, candidates, selections = fit_statistical_candidates(
        *_native_values(parent, 0, 32),
        *_native_values(parent, 32, 16),
        config=small_config,
        weights=population["weights"],
    )
    failed = [r for r in candidates if r["method"] == "gaussian_selected"]
    assert len(failed) == len(small_config.prior_counts)
    assert all(not r["available"] and r["error"] == "declared numerical failure" for r in failed)
    assert fits["gaussian_selected"] is None and selections["gaussian_selected"] is None


def test_validation_only_selects_from_issued_training_candidates(small_config, population):
    parent = np.random.default_rng(6).normal(size=(64, 7))
    args = (*_native_values(parent, 0, 32), *_native_values(parent, 32, 32))
    _, candidates, selections = fit_statistical_candidates(
        *args, config=small_config, weights=population["weights"]
    )
    for method, index in selections.items():
        available = [
            i for i, r in enumerate(candidates) if r["method"] == method and r["available"]
        ]
        assert index == min(
            available,
            key=lambda i: (candidates[i]["validation_nll"], -candidates[i]["prior_count"]),
        )
        selected = candidates[index]
        assert selected["relative_regularization"] == selected["prior_count"] / 32


def test_coverage_resolution_uses_paired_parents_and_keeps_stratum_uncertainty_separate():
    from spde_covariance.statistical_completion_benchmark import coverage_resolution_study

    config = StatisticalCompletionConfig(
        coverage_replications=2,
        coverage_training_count=16,
        coverage_training_counts=(8, 16),
        coverage_precision_required=False,
    )
    result = coverage_resolution_study(config)
    assert result["paired_nested_sample_sizes"]
    assert result["paired_complete_observation_views"]
    assert result["partial_coverage"]["count"] == 2
    assert len(result["resolution_study"]) == 4
    for count in (8, 16):
        native, complete = (
            next(
                r
                for r in result["resolution_study"]
                if r["training_count"] == count and r["design"] == design
            )
            for design in ("native", "complete")
        )
        assert "noncanonical" in native["risk_coverage_all"]
        assert "noncanonical" not in complete["risk_coverage_all"]
        assert (
            complete["population_identification_range"]["lower"]
            == complete["population_identification_range"]["upper"]
        )
        for a, b in zip(native["records"], complete["records"], strict=True):
            # The separator node is seen every date in both views. Its raw
            # variance-probe sum verifies common parent draws, not just seeds.
            def separator_sum(record):
                return next(
                    p["sum_squares"]
                    for p in record["region_diagnostics"]["scalar_group_intervals"]
                    if p["probe"] == "node" and tuple(p["nodes"]) == (1, 1)
                )

            assert separator_sum(a) == separator_sum(b)


def test_cli_reports_output_directory_instead_of_printing_numerical_results(
    monkeypatch, capsys, tmp_path
):
    import spde_covariance.statistical_completion_benchmark as benchmark
    from spde_covariance.experiments import main

    target = tmp_path / "new-run"
    received = []
    monkeypatch.setattr(
        benchmark, "load_statistical_completion_config", lambda _: StatisticalCompletionConfig()
    )
    monkeypatch.setattr(
        benchmark,
        "run_statistical_completion",
        lambda c, p: received.append((c, p)) or {"acceptance": {"passed": True}},
    )
    main(["benchmark-statistical-completion", "input.toml", "--output", str(target)])
    assert received[0][1] == target
    assert capsys.readouterr().out.strip() == str(target.resolve())
