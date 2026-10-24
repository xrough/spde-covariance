"""Truth-free strategy composition, observed-score blindness and guarded tuning."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from spde_covariance.completion import observed_mask
from spde_covariance.completion_strategy import (
    CompletionStabilityGuard,
    CompletionStrategySettings,
    balanced_fragment_split,
    fit_completion_strategy,
    select_completion_strategy,
    split_completion_stability,
)
from spde_covariance.observation_statistics import (
    kernel_quadratic_score_from_statistics,
    partial_kernel_from_arrays,
)


def exact_data(kernel, cliques):
    """Actual centered finite samples whose corrected clique moments are exact."""
    values, rows = [], []
    noise = max(0.0, -min(np.linalg.eigvalsh(kernel[np.ix_(g, g)])[0] for g in cliques)) + 0.2
    for group in cliques:
        block = kernel[np.ix_(group, group)] + noise * np.eye(len(group))
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
        samples = np.sqrt(len(group)) * np.concatenate([root, -root], axis=0)
        values.extend(samples)
        rows.extend([np.asarray(group)] * len(samples))
    return tuple(values), tuple(rows), noise


def exact_statistics(kernel, cliques):
    values, rows, noise = exact_data(kernel, cliques)
    return partial_kernel_from_arrays(values, rows, noise, size=len(kernel))


def case():
    x = np.linspace(0, 1, 7)
    kernel = np.exp(-2 * np.abs(x[:, None] - x)) + 0.3 * np.eye(7)
    cliques = (np.arange(5), np.arange(2, 7))
    return kernel, cliques, np.linspace(0.1, 0.2, 7)


def test_pairwise_and_entrywise_ridge_preserve_distinct_upstream_choices():
    kernel, cliques, weights = case()
    statistics = exact_statistics(kernel, cliques)
    pairwise = fit_completion_strategy(
        statistics, cliques=cliques, weights=weights, settings=CompletionStrategySettings()
    )
    ridge = fit_completion_strategy(
        statistics,
        cliques=cliques,
        weights=weights,
        settings=CompletionStrategySettings(partial_method="ridge", relative_regularization=0.2),
    )
    mask = observed_mask(cliques, len(kernel))
    assert pairwise.available and ridge.available
    np.testing.assert_allclose(pairwise.partial_kernel[mask], kernel[mask], atol=2e-15)
    penalty = 0.2 * statistics.pair_weight_sums.max()
    expected = (
        statistics.pair_weight_sums[mask]
        / (statistics.pair_weight_sums[mask] + penalty)
        * kernel[mask]
    )
    np.testing.assert_allclose(ridge.partial_kernel[mask], expected, atol=2e-15)
    assert pairwise.diagnostics["observed_modification"]["weighted_hs"] == 0
    assert ridge.diagnostics["observed_modification"]["weighted_hs"] > 0
    assert not pairwise.diagnostics["completion"]["population"]
    assert ridge.diagnostics["objective_gap_bound"] == 0
    assert np.isnan(ridge.partial_kernel[~mask]).all()


def test_coherent_floor_certificate_psd_and_weighted_units():
    kernel, cliques, weights = case()
    statistics = exact_statistics(kernel, cliques)
    settings = CompletionStrategySettings(
        partial_method="coherent",
        relative_regularization=0.01,
        floor_policy="resolution",
        numerical_error_budget=1e-6,
    )
    fit = fit_completion_strategy(statistics, cliques=cliques, weights=weights, settings=settings)
    assert fit.available and fit.converged
    assert fit.diagnostics["partial_error_bound"] <= 1e-6
    assert fit.diagnostics["objective_gap_bound"] >= 0
    assert fit.diagnostics["completion"]["population"]
    assert fit.diagnostics["completion"]["output_psd"]
    assert fit.diagnostics["maximum_inverse_norm"] > 0
    assert fit.diagnostics["maximum_retained_condition_number"] >= 1
    for group in cliques:
        root = np.sqrt(weights[group])
        block = root[:, None] * fit.partial_kernel[np.ix_(group, group)] * root[None, :]
        assert np.linalg.eigvalsh(block)[0] >= settings.eigenvalue_floor(len(kernel)) - 1e-13
    assert not fit.diagnostics["continuum_convergence_claim"]


def test_resolution_floor_budget_decreases_and_legacy_anchor_is_explicit():
    resolution = CompletionStrategySettings(partial_method="coherent", floor_policy="resolution")
    constant = replace(resolution, floor_policy="constant")
    zero = replace(resolution, floor_policy="zero")
    assert resolution.eigenvalue_floor(46) == pytest.approx(1e-5)
    assert constant.eigenvalue_floor(92) == 1e-5
    assert zero.eigenvalue_floor(46) == 0
    assert 92 * resolution.eigenvalue_floor(92) == pytest.approx(
        46 * resolution.eigenvalue_floor(46) / 2
    )
    assert resolution.eigenvalue_floor(92) == pytest.approx(resolution.eigenvalue_floor(46) / 4)
    assert 92 * constant.eigenvalue_floor(92) > 46 * constant.eigenvalue_floor(46)


def test_rank2_ablation_can_be_indefinite_and_validation_cannot_see_changed_cross_block():
    # A and B copy the third separator mode. Rank2 discards that mode.
    features = np.array([[0, 0, 1], [np.sqrt(3), 0, 0], [0, np.sqrt(2), 0], [0, 0, 1], [0, 0, 1]])
    kernel = features @ features.T
    cliques = (np.arange(4), np.arange(1, 5))
    statistics = exact_statistics(kernel, cliques)
    full = fit_completion_strategy(
        statistics, cliques=cliques, weights=np.ones(5), settings=CompletionStrategySettings()
    )
    rank2 = fit_completion_strategy(
        statistics,
        cliques=cliques,
        weights=np.ones(5),
        settings=CompletionStrategySettings(completion_rule="rank2"),
    )
    assert full.available and rank2.available
    assert full.kernel[0, 4] == pytest.approx(1)
    assert rank2.kernel[0, 4] == pytest.approx(0, abs=1e-14)
    assert not rank2.diagnostics["completion"]["output_psd"]
    assert np.linalg.eigvalsh(rank2.kernel)[0] == pytest.approx(1 - np.sqrt(2))
    mask = observed_mask(cliques, 5)
    np.testing.assert_array_equal(full.kernel[mask], rank2.kernel[mask])
    assert kernel_quadratic_score_from_statistics(
        full.kernel, statistics
    ) == kernel_quadratic_score_from_statistics(rank2.kernel, statistics)
    selection = select_completion_strategy(
        {"rank2_first": rank2, "support_second": full}, statistics
    )
    assert selection.selected_name == "rank2_first"
    assert "exactly equal" in selection.diagnostics["tie_rule"]


def test_coherent_rank2_is_an_explicit_non_population_completion():
    kernel, cliques, weights = case()
    fit = fit_completion_strategy(
        exact_statistics(kernel, cliques),
        cliques=cliques,
        weights=weights,
        settings=CompletionStrategySettings(
            partial_method="coherent", completion_rule="rank2", numerical_error_budget=1e-6
        ),
    )
    assert fit.available
    assert fit.diagnostics["completion"]["rank_rule"] == 2
    assert not fit.diagnostics["completion"]["population"]


def test_numerical_certificate_failure_has_no_completed_kernel_or_silent_fallback():
    kernel = np.array([[1.0, 2, 0], [2, 1, 0], [0, 0, 4]])
    cliques = (np.arange(2), np.arange(1, 3))
    settings = CompletionStrategySettings(
        partial_method="coherent", max_iterations=1, numerical_error_budget=1e-10
    )
    fit = fit_completion_strategy(
        exact_statistics(kernel, cliques), cliques=cliques, weights=np.ones(3), settings=settings
    )
    assert not fit.available and not fit.converged
    assert fit.kernel is None
    assert fit.diagnostics["completion"] is None
    assert fit.diagnostics["partial_error_bound"] > settings.numerical_error_budget
    assert "certificate" in fit.reason
    assert fit.diagnostics["partial_fit"]["feasibility_diagonal_shift"] > 0
    selection = select_completion_strategy({"failed": fit}, exact_statistics(kernel, cliques))
    assert not selection.available
    assert not selection.diagnostics["candidates"][0]["admissible"]


def test_missing_training_coverage_and_unavailable_rank_remain_in_audit():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    statistics = partial_kernel_from_arrays([[1, 0], [-1, 0]], ([0, 1], [0, 1]), size=3)
    missing = fit_completion_strategy(
        statistics, cliques=cliques, weights=np.ones(3), settings=CompletionStrategySettings()
    )
    assert not missing.available
    assert missing.diagnostics["missing_training_pair_count"] == 3
    assert "cover" in missing.reason
    complete_stats = exact_statistics(np.eye(3), cliques)
    rank = fit_completion_strategy(
        complete_stats,
        cliques=cliques,
        weights=np.ones(3),
        settings=CompletionStrategySettings(completion_rule="rank2"),
    )
    assert not rank.available
    assert "rank" in rank.reason
    selection = select_completion_strategy({"missing": missing, "rank": rank}, complete_stats)
    assert not selection.available
    assert len(selection.diagnostics["candidates"]) == 2
    assert all(row["rejection_reasons"] for row in selection.diagnostics["candidates"])


def test_split_is_seeded_complementary_balanced_by_support_and_ignores_row_order():
    rows = ([0, 1], [1, 0], [0, 1], [1, 2], [2, 1], [1, 2], [1, 2], [0, 2], [2, 0])
    split = balanced_fragment_split(rows, size=3, seed=123)
    again = balanced_fragment_split(rows, size=3, seed=123)
    assert split.available
    np.testing.assert_array_equal(split.first_indices, again.first_indices)
    assert not set(split.first_indices).intersection(split.second_indices)
    assert set(split.first_indices).union(split.second_indices) == set(range(len(rows)))
    assert abs(len(split.first_indices) - len(split.second_indices)) <= 1
    for group in split.diagnostics["groups"]:
        assert abs(group["first_count"] - group["second_count"]) <= 1
    unavailable = balanced_fragment_split(([0, 1], [1, 2], [1, 2]), size=3)
    assert not unavailable.available
    assert "fewer than twice" in unavailable.reason


def test_split_stability_is_training_only_reproducible_and_keeps_failed_halves():
    kernel, cliques, weights = case()
    values, rows, noise = exact_data(kernel, cliques)
    settings = CompletionStrategySettings()
    split = split_completion_stability(
        values,
        rows,
        size=7,
        cliques=cliques,
        weights=weights,
        settings=settings,
        noise_variance=noise,
        seed=9,
    )
    again = split_completion_stability(
        tuple(np.array(value, copy=True) for value in values),
        rows,
        size=7,
        cliques=cliques,
        weights=weights,
        settings=settings,
        noise_variance=noise,
        seed=9,
    )
    assert split.available and again.available
    np.testing.assert_array_equal(split.first_fit.kernel, again.first_fit.kernel)
    assert split.diagnostics["relative_difference"] >= 0
    assert split.diagnostics["missing_relative_difference"] >= 0
    assert "not a sampling" in split.diagnostics["scope"]
    failed = split_completion_stability(
        values,
        rows,
        size=7,
        cliques=cliques,
        weights=weights,
        settings=replace(settings, completion_rule="rank2"),
        noise_variance=noise,
        seed=9,
    )
    # Sparse half-moments can have fewer than two positive separator modes.
    assert failed.first_fit is not None and failed.second_fit is not None
    assert failed.available == (failed.first_fit.available and failed.second_fit.available)


def test_observed_only_selection_explicit_guard_and_unavailable_stability():
    kernel, cliques, weights = case()
    statistics = exact_statistics(kernel, cliques)
    fit = fit_completion_strategy(
        statistics, cliques=cliques, weights=weights, settings=CompletionStrategySettings()
    )
    values, rows, noise = exact_data(kernel, cliques)
    split = split_completion_stability(
        values,
        rows,
        size=7,
        cliques=cliques,
        weights=weights,
        settings=CompletionStrategySettings(),
        noise_variance=noise,
    )
    assert split.available
    unchecked = select_completion_strategy({"a": fit}, statistics)
    assert unchecked.available
    assert unchecked.diagnostics["candidates"][0]["stability_available"] is None
    unavailable = select_completion_strategy(
        {"a": fit}, statistics, guard=CompletionStabilityGuard()
    )
    assert not unavailable.available
    guarded = select_completion_strategy(
        {"a": fit},
        statistics,
        stability={"a": split},
        guard=CompletionStabilityGuard(max_relative_difference=0),
    )
    assert not guarded.available
    assert (
        "exceeds declared threshold" in guarded.diagnostics["candidates"][0]["rejection_reasons"][0]
    )
    condition = select_completion_strategy(
        {"a": fit},
        statistics,
        guard=CompletionStabilityGuard(max_inverse_norm=0, require_available=False),
    )
    assert not condition.available
    accepted = select_completion_strategy(
        {"a": fit},
        statistics,
        stability={"a": split},
        guard=CompletionStabilityGuard(max_relative_difference=3),
    )
    assert accepted.available


def test_selection_cannot_use_latent_truth_test_values_or_unseen_origins():
    kernel, cliques, weights = case()
    training = exact_statistics(kernel, cliques)
    fit = fit_completion_strategy(
        training, cliques=cliques, weights=weights, settings=CompletionStrategySettings()
    )
    validation = exact_statistics(kernel * 1.1, cliques)
    ridge = fit_completion_strategy(
        training,
        cliques=cliques,
        weights=weights,
        settings=CompletionStrategySettings(partial_method="ridge", relative_regularization=1),
    )
    candidates = {"unshrunk": fit, "ridge": ridge}
    first = select_completion_strategy(candidates, validation)
    # Unobserved truth, future paths and scenario labels have no interface path.
    hidden_truth, future_values = kernel * 100, np.zeros((100, 7))
    hidden_truth[:] = -999
    future_values[:] = 1e10
    second = select_completion_strategy(candidates, validation)
    assert first.selected_name == second.selected_name
    assert first.diagnostics["selected_score"] == second.diagnostics["selected_score"]
    # Genuine observed validation changes the decision; hidden values above do not.
    lower_validation = exact_statistics(kernel * 0.1, cliques)
    assert first.selected_name == "unshrunk"
    assert select_completion_strategy(candidates, lower_validation).selected_name == "ridge"
    with pytest.raises(TypeError):
        select_completion_strategy({"fixed": fit}, validation, truth=hidden_truth)
    with pytest.raises(TypeError):
        fit_completion_strategy(
            training,
            cliques=cliques,
            weights=weights,
            settings=CompletionStrategySettings(),
            future=future_values,
        )


def test_no_condition_number_is_invented_for_an_empty_separator_spectrum():
    cliques = (np.array([0, 1]), np.array([1, 2]))
    samples = ([1, 0], [-1, 0], [0, 1], [0, -1]) * 2
    rows = ([0, 1],) * 4 + ([1, 2],) * 4
    statistics = partial_kernel_from_arrays(samples, rows, noise_variance=0.5, size=3)
    fit = fit_completion_strategy(
        statistics,
        cliques=cliques,
        weights=np.ones(3),
        settings=CompletionStrategySettings(),
    )
    assert fit.available
    assert not fit.diagnostics["conditioning_available"]
    assert fit.diagnostics["maximum_inverse_norm"] == 0
    assert fit.diagnostics["maximum_retained_condition_number"] is None
    selection = select_completion_strategy(
        {"zero": fit},
        statistics,
        guard=CompletionStabilityGuard(max_condition_number=100, require_available=False),
    )
    assert not selection.available
    assert (
        "condition number unavailable"
        in selection.diagnostics["candidates"][0]["rejection_reasons"][0]
    )


def test_nonconvergence_flag_cannot_override_an_excessive_certificate(monkeypatch):
    from spde_covariance import completion_strategy
    from spde_covariance.coherent import CoherentPartialFit

    kernel, cliques, weights = case()
    statistics = exact_statistics(kernel, cliques)
    mask = observed_mask(cliques, len(kernel))
    false_success = CoherentPartialFit(
        np.where(mask, statistics.partial_kernel, np.nan),
        {"converged": True, "objective_gap_bound": 1.0, "partial_error_bound": 2.0},
    )
    monkeypatch.setattr(
        completion_strategy, "fit_coherent_partial", lambda *args, **kwargs: false_success
    )
    result = fit_completion_strategy(
        statistics,
        cliques=cliques,
        weights=weights,
        settings=CompletionStrategySettings(partial_method="coherent"),
    )
    assert not result.available and not result.converged
    assert result.kernel is None and result.diagnostics["completion"] is None


def test_weighted_dates_and_zero_weight_half_unavailability_are_explicit():
    kernel, cliques, weights = case()
    values, rows, noise = exact_data(kernel, cliques)
    settings = CompletionStrategySettings()
    split = balanced_fragment_split(rows, size=7, seed=123)
    date_weights = np.zeros(len(values))
    date_weights[split.first_indices] = 1
    result = split_completion_stability(
        values,
        rows,
        size=7,
        cliques=cliques,
        weights=weights,
        settings=settings,
        noise_variance=noise,
        seed=123,
        date_weights=date_weights,
    )
    assert not result.available
    assert "no positive date weight" in result.reason


def test_results_settings_and_nested_diagnostics_are_immutable_snapshots():
    kernel, cliques, weights = case()
    settings = CompletionStrategySettings()
    fit = fit_completion_strategy(
        exact_statistics(kernel, cliques), cliques=cliques, weights=weights, settings=settings
    )
    with pytest.raises(FrozenInstanceError):
        settings.floor_gamma = 0
    with pytest.raises(ValueError):
        fit.kernel[0, 0] = 1
    with pytest.raises(TypeError):
        fit.diagnostics["settings"]["partial_method"] = "coherent"
    values, rows, noise = exact_data(kernel, cliques)
    split = split_completion_stability(
        values,
        rows,
        size=7,
        cliques=cliques,
        weights=weights,
        settings=settings,
        noise_variance=noise,
    )
    with pytest.raises(ValueError):
        split.split.first_indices[0] = 99


@pytest.mark.parametrize(
    "kwargs",
    [
        {"partial_method": "other"},
        {"completion_rule": "auto"},
        {"floor_policy": "auto"},
        {"relative_regularization": -1},
        {"relative_regularization": 1},
        {"floor_policy": "resolution"},
        {"floor_gamma": np.inf},
        {"numerical_error_budget": 0},
        {"rho": False},
        {"max_iterations": 1.5},
        {"support_rtol": 1},
    ],
)
def test_invalid_settings_are_rejected_instead_of_ignored(kwargs):
    with pytest.raises(ValueError):
        CompletionStrategySettings(**kwargs)


def test_validation_noise_correction_score_matches_explicit_formula():
    kernel, cliques, weights = case()
    values, rows, noise = exact_data(kernel, cliques)
    statistics = partial_kernel_from_arrays(values, rows, noise, size=7)
    fit = fit_completion_strategy(
        statistics, cliques=cliques, weights=weights, settings=CompletionStrategySettings()
    )
    score = select_completion_strategy({"a": fit}, statistics).diagnostics["selected_score"]
    explicit = np.mean(
        [
            np.sum(fit.kernel[np.ix_(row, row)] ** 2)
            - 2
            * np.sum(
                fit.kernel[np.ix_(row, row)] * (np.outer(value, value) - noise * np.eye(len(row)))
            )
            for value, row in zip(values, rows, strict=True)
        ]
    )
    assert score == pytest.approx(explicit, abs=2e-14)
