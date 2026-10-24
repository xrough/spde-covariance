"""Independent statistical contracts of the dense-fragment study."""

import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import spde_covariance.fragment_benchmark as benchmark
import spde_covariance.fragment_reporting as reporting
from spde_covariance.completion import KernelCompletion
from spde_covariance.fragment_config import FragmentBenchmarkConfig
from spde_covariance.fragment_reference import paper_cover, paper_kernel
from spde_covariance.observation_statistics import partial_kernel_from_arrays


def _small_config(**changes):
    config = FragmentBenchmarkConfig(
        replications=1,
        training_counts=(4, 8),
        validation_count=4,
        test_count=4,
        grids=(21,),
        covers=(3,),
        models=("powerlaw",),
        noise_variances=(0,),
        reference_dimension=8,
        candidate_dimensions=(4,),
        relative_regularizations=(0.001, 0.01),
        ranks=(1, 2),
        ridge_comparison=False,
        population_checks=False,
    )
    return replace(config, **changes)


def test_balanced_assignments_are_reproducible_nested_prefixes():
    for groups in (3, 9):
        short = benchmark.balanced_assignments(128, groups, 817)
        long = benchmark.balanced_assignments(512, groups, 817)
        np.testing.assert_array_equal(short, long[:128])
        assert np.ptp(np.bincount(long, minlength=groups)) <= 1
        for start in range(0, len(long) - groups + 1, groups):
            np.testing.assert_array_equal(np.sort(long[start : start + groups]), np.arange(groups))


def test_observed_score_matches_datewise_noise_corrected_moment_identity():
    values = np.array([[1.2, -0.3, 8], [9, 0.7, -1.1], [-0.4, 0.2, 7]], float)
    cliques = (np.array([0, 1]), np.array([1, 2]))
    assigned = np.array([0, 1, 0])
    kernel = np.array([[0.8, 0.1, -4], [0.1, 0.5, -0.2], [-4, -0.2, 0.9]])
    noise = 0.17
    expected = 0
    for value, group in zip(values, assigned, strict=True):
        ids = cliques[group]
        prediction = kernel[np.ix_(ids, ids)]
        corrected = np.outer(value[ids], value[ids]) - noise * np.eye(len(ids))
        expected += np.sum(prediction**2) - 2 * np.sum(prediction * corrected)
    actual = benchmark.array_score(kernel, values, assigned, cliques, noise)
    assert actual == pytest.approx(expected / len(values))
    changed = values.copy()
    changed[0, 2], changed[1, 0], changed[2, 2] = 1e8, -1e8, 2e8
    assert benchmark.array_score(kernel, changed, assigned, cliques, noise) == actual


def test_exact_gaussian_second_moments_recover_rank_four_from_noisy_fragments():
    # Symmetric sigma points integrate centered Gaussian second moments exactly.
    # This isolates the estimator/noise correction from random sampling variation.
    nodes = np.linspace(0, 1, 21)
    truth = paper_kernel("finite_rank", nodes)
    variance = 0.2
    eigenvalues, eigenvectors = np.linalg.eigh(truth + variance * np.eye(len(nodes)))
    root = eigenvectors * np.sqrt(eigenvalues)
    sigma = np.sqrt(len(nodes)) * np.concatenate([root.T, -root.T])
    cliques = benchmark.clique_indices(nodes, paper_cover(3))
    parent = np.tile(sigma, (len(cliques), 1))
    assigned = np.repeat(np.arange(len(cliques)), len(sigma))
    rows = tuple(cliques[group] for group in assigned)
    stats = partial_kernel_from_arrays(
        tuple(value[ids] for value, ids in zip(parent, rows, strict=True)),
        rows,
        noise_variance=variance,
        size=len(nodes),
    )
    np.testing.assert_allclose(
        stats.partial_kernel[stats.observed_mask], truth[stats.observed_mask], atol=1e-13
    )
    result = benchmark.fit_fragment_kernel(
        stats, cliques=cliques, weights=np.ones(len(nodes)), rank=4
    )
    assert result.available
    np.testing.assert_allclose(result.kernel, truth, atol=1e-10)
    other = truth + 0.1 * np.eye(len(nodes))
    score_difference = benchmark.array_score(
        other, parent, assigned, cliques, variance
    ) - benchmark.array_score(truth, parent, assigned, cliques, variance)
    expected = np.mean([np.sum((other[np.ix_(c, c)] - truth[np.ix_(c, c)]) ** 2) for c in cliques])
    assert score_difference == pytest.approx(expected, abs=1e-12)


def test_weighted_kernel_errors_are_product_measure_norms_not_h_operator_norms():
    truth = np.array([[2.0, 0.5], [0.5, 1.0]])
    estimate = truth + np.array([[0.1, 0.2], [0.2, -0.3]])
    canonical = truth + 0.05
    weights = np.array([0.2, 0.8])
    mask = np.eye(2, dtype=bool)
    result = benchmark._metrics(estimate, truth, canonical, weights, mask)
    product = np.outer(weights, weights)
    squared = product * (estimate - truth) ** 2
    assert result["observed_weighted_error"] ** 2 == pytest.approx(squared[mask].sum())
    assert result["missing_weighted_error"] ** 2 == pytest.approx(squared[~mask].sum())
    assert result["integrated_squared_error"] == pytest.approx(squared.sum())
    assert result["canonical_generating_discrepancy"] ** 2 == pytest.approx(
        np.sum(product * (canonical - truth) ** 2)
    )


def test_full_parent_is_shared_across_nested_grids_and_truth_stays_outside_fit():
    config = _small_config(profile="reference", models=("finite_rank",), grids=(21, 41))
    parent_nodes, models, geometry = benchmark._prepare(config)
    values = (
        np.random.default_rng(77).normal(size=(8, len(parent_nodes)))
        @ models["finite_rank"]["sample_map"].T
    )
    small, large = geometry[("finite_rank", 21, 3)], geometry[("finite_rank", 41, 3)]
    large_ids = np.searchsorted(large["nodes"], small["nodes"])
    np.testing.assert_array_equal(
        values[:, small["indices"]], values[:, large["indices"]][:, large_ids]
    )


def test_ridge_fit_only_receives_training_prefix_and_selection_ignores_test(monkeypatch):
    config = _small_config()
    nodes = np.linspace(0, 3, 7)
    cliques = (np.arange(len(nodes)),)
    assigned = np.zeros(config.total_count, dtype=int)
    values = np.random.default_rng(44).normal(scale=0.1, size=(config.total_count, len(nodes)))
    values[8:12] = 0
    calls = []

    def fake_fit(observations, settings, **options):
        calls.append((tuple(observations), settings, options))
        covariance = np.eye(options["dimension"]) * options["relative_regularization"]
        estimate = SimpleNamespace(
            covariance_error_bound=1e-6,
            objective_gap_bound=1e-9,
            converged=True,
            iterations=1,
            trace=float(np.trace(covariance)),
        )
        return SimpleNamespace(covariance=covariance, estimate=estimate)

    monkeypatch.setattr(benchmark, "fit_candidate", fake_fit)
    safe = dict(nodes=nodes, cliques=cliques, noise_variance=0, base_grid=21)
    first, records = benchmark._ridge(values, assigned, safe, config, 4, {})
    changed = values.copy()
    changed[4:8] = -1e8
    changed[12:] = 1e8
    second, again = benchmark._ridge(changed, assigned, safe, config, 4, {})
    np.testing.assert_array_equal(first[0], second[0])
    assert [row["validation_score"] for row in records] == [
        row["validation_score"] for row in again
    ]
    for observations, settings, options in calls:
        assert len(observations) == 4
        np.testing.assert_array_equal(np.array([item.y for item in observations]), values[:4])
        assert not hasattr(settings, "models")
        assert "truth" not in options
        assert observations[-1].time == 4 * config.delta
    assert all(row["validation_first_interval"] == 9 for row in records)
    assert all(row["validation_last_interval"] == 12 for row in records)
    assert all(not row["refit_on_validation"] for row in records)
    validation_changed = values.copy()
    validation_changed[8:12] = 1e4
    validation_fit, _ = benchmark._ridge(validation_changed, assigned, safe, config, 4, {})
    assert first[1]["relative_regularization"] == 0.001
    assert validation_fit[1]["relative_regularization"] == 0.01


def test_ridge_exact_score_ties_use_smaller_dimension_then_larger_penalty():
    rows = [
        dict(J=8, relative_regularization=0.1, validation_score=-2),
        dict(J=4, relative_regularization=0.001, validation_score=-2),
        dict(J=4, relative_regularization=0.01, validation_score=-2),
    ]
    chosen = benchmark.choose_candidate(rows)
    assert (chosen["J"], chosen["relative_regularization"]) == (4, 0.01)


def test_risk_score_only_uses_dates_observing_the_actual_grid_exposure():
    values = np.array([[2, 3, 100], [100, 4, 5], [6, 7, 100]], float)
    cliques = (np.array([0, 1]), np.array([1, 2]))
    assigned = np.array([0, 1, 0])
    exposure = np.array([1, -1, 0])
    result = benchmark.exposure_score(2, exposure, values, assigned, cliques, 0.3)
    expected_proxy = (1 + 1) / 2 - 0.3 * 2
    assert result["available_test_dates"] == 2
    assert result["risk_score"] == pytest.approx(4 - 4 * expected_proxy)
    unavailable = benchmark.exposure_score(2, np.array([1, 0, -1]), values, assigned, cliques, 0.3)
    assert unavailable["risk_score"] is None
    assert unavailable["available_test_dates"] == 0


def test_replication_summary_distinguishes_availability_mcse_and_both_mad_definitions():
    rows = [dict(case="x", available=True, error=value) for value in (0, 0, 100)]
    rows.append(dict(case="x", available=False))
    result = benchmark._summary(rows, ("error",), ("case",))[0]
    assert result["replications"] == 4
    assert result["available_replications"] == result["error_available"] == 3
    assert result["unavailable_replications"] == 1
    assert result["error_mean"] == pytest.approx(100 / 3)
    assert result["error_mcse"] == pytest.approx(100 / 3)
    assert result["error_median"] == result["error_mad"] == 0
    assert result["error_mean_absolute_deviation_from_median"] == pytest.approx(100 / 3)
    assert result["error_mean_absolute_deviation_from_mean"] == pytest.approx(400 / 9)


def test_small_reference_runner_covers_every_declared_cell_and_pairs_replications(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(reporting, "write_fragment_report", lambda *args: None)
    config = _small_config(
        profile="reference",
        models=("finite_rank",),
        replications=2,
        training_counts=(24, 48),
        covers=(3, 9),
        include_fve=False,
    )
    output = benchmark.run_fragment_benchmark(config, tmp_path / "reference")
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["passed"]
    assert results["acceptance"]["controlled_population_recovery"]
    assert results["available_fits"] == 16
    assert results["unavailable_fits"] == 0
    assert len(results["summary"]) == 8
    assert {(r["cover"], r["N"], r["variant"]) for r in results["summary"]} == {
        (m, n, variant) for m in (3, 9) for n in (24, 48) for variant in ("raw", "psd")
    }
    assert all(row["replications"] == 2 for row in results["summary"])
    rows = list(csv.DictReader((output / "estimates.csv").open()))
    chosen = [r for r in rows if r["cover"] == "3" and r["N"] == "48"]
    by_variant = {
        variant: {
            int(r["replication"]): float(r["test_observed_score"])
            for r in chosen
            if r["variant"] == variant
        }
        for variant in ("raw", "psd")
    }
    differences = np.array([by_variant["psd"][rep] - by_variant["raw"][rep] for rep in (0, 1)])
    paired = next(
        r
        for r in results["paired_differences"]
        if r["cover"] == 3 and r["N"] == 48 and r["variant"] == "psd"
    )
    assert paired["paired_replications"] == 2
    assert paired["score_difference_mean"] == pytest.approx(differences.mean())
    assert paired["score_difference_mcse"] == pytest.approx(differences.std(ddof=1) / np.sqrt(2))
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        benchmark.run_fragment_benchmark(config, output)


def test_all_unavailable_fits_cannot_pass_numerical_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "write_fragment_report", lambda *args: None)
    monkeypatch.setattr(
        benchmark,
        "fit_fragment_kernel",
        lambda *args, **kwargs: KernelCompletion(
            None, {"available": False, "reason": "controlled unavailable inverse"}
        ),
    )
    config = _small_config(
        profile="reference", models=("finite_rank",), training_counts=(24,), include_fve=False
    )
    output = tmp_path / "unavailable"
    with pytest.raises(RuntimeError, match="acceptance"):
        benchmark.run_fragment_benchmark(config, output)
    results = json.loads((output / "results.json").read_text())
    assert not results["acceptance"]["passed"]
    assert results["available_fits"] == 0
    assert results["unavailable_fits"] == 2
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_grid_delivery_identification_never_becomes_continuous_delivery_identification(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(reporting, "write_fragment_report", lambda *args: None)
    config = _small_config(training_counts=(24,), covers=(9,), include_fve=False)
    output = benchmark.run_fragment_benchmark(config, tmp_path / "spde")
    risks = list(csv.DictReader((output / "risks.csv").open()))
    grid = [row for row in risks if row["target"] == "grid_delivery_spread"]
    continuous = [row for row in risks if row["target"] == "continuous_delivery_spread"]
    assert grid and continuous
    assert all(row["pair_coverage_identified"] == "True" for row in grid)
    assert all(row["pair_coverage_identified"] == "False" for row in continuous)
    assert all(row["grid_pair_coverage_identified"] == "True" for row in continuous)
    assert all(row["scoring_available"] == "False" for row in continuous)
    assert all(row["available_test_dates"] == "0" for row in continuous)
