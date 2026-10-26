"""Training-only fitting and directly measured hedge baseline controls."""

import inspect

import numpy as np
import pytest

import spde_covariance.hedging_workflow as workflow
from spde_covariance.hedging import solve_target_block_hedge


def sample_case(*, complete=False, seed=20):
    rng = np.random.default_rng(seed)
    kernel = np.array([[0.1, 0.04, 0.01], [0.04, 0.1, 0.04], [0.01, 0.04, 0.1]])
    values = rng.multivariate_normal(np.zeros(3), kernel + 0.001 * np.eye(3), size=300)
    if complete:
        cliques = ((0, 1, 2),)
    else:
        values[:150, 2], values[150:, 0] = np.nan, np.nan
        cliques = ((0, 1), (1, 2))
    options = {
        "evaluation_matrix": np.eye(3),
        "source_factors": (np.eye(3),),
        "source_offset": np.zeros((3, 3)),
        "cliques": cliques,
        "target": [1, 0, 0],
        "instrument": [0, 0, 1],
    }
    return values, options


def require_cvxpy():
    pytest.importorskip("cvxpy")


def test_partial_training_fits_and_joint_baseline_unavailability():
    require_cvxpy()
    values, options = sample_case()
    records = workflow.fit_hedging_rules(values, **options)
    assert set(records) == {
        "canonical_coherent",
        "ridge",
        "robust_psd",
        "robust_source",
        "direct_target_block",
        "empirical_shrinkage",
    }
    for method in ("canonical_coherent", "ridge", "robust_psd", "robust_source"):
        record = records[method]
        assert record["available"], record["reason"]
        assert record["positions"].shape == (1,)
        assert abs(record["positions"][0]) <= 8
        assert np.isfinite(record["risk"])
        assert not record["fit_diagnostics"]["uses_latent_truth"]
        assert not record["fit_diagnostics"]["uses_future_outcomes"]
    for method in ("direct_target_block", "empirical_shrinkage"):
        assert not records[method]["available"]
        assert records[method]["positions"] is None
        assert "jointly observe" in records[method]["reason"]
    assert records["canonical_coherent"]["fit_diagnostics"]["full_numerical_support"]
    assert records["canonical_coherent"]["fit_diagnostics"]["hard_rank_constraint"] is None
    assert records["ridge"]["fit_diagnostics"]["converged"]
    assert records["ridge"]["fit_diagnostics"]["covariance_error_bound"] <= 1e-4
    for method in ("canonical_coherent", "ridge"):
        assert records[method]["fitted_kernel"].shape == (3, 3)
        assert records[method]["fitted_kernel_scope"] == "grid_covariance"
    assert records["robust_source"]["fitted_kernel"] is None
    assert records["robust_psd"]["fitted_kernel"] is None


def test_both_robust_classes_use_identical_bands_and_derived_outer_trace_cap(monkeypatch):
    require_cvxpy()
    values, options = sample_case()
    options["source_factors"] = (0.8 * np.eye(3), 0.4 * np.eye(3))
    options["source_offset"] = 0.02 * np.eye(3)
    options["source_trace_cap"] = 1.5
    models = []
    original = workflow.solve_robust_hedge

    def capture(model, target, instruments, **settings):
        models.append((model, target, instruments))
        return original(model, target, instruments, **settings)

    monkeypatch.setattr(workflow, "solve_robust_hedge", capture)
    records = workflow.fit_hedging_rules(values, **options)
    assert len(models) == 2
    free, source = models[0][0], models[1][0]
    assert free.calibrated_bands and source.calibrated_bands
    np.testing.assert_array_equal(free.entry_pairs, source.entry_pairs)
    np.testing.assert_array_equal(free.lower, source.lower)
    np.testing.assert_array_equal(free.upper, source.upper)
    assert free.trace_cap == pytest.approx(1.5 * (0.8**2 + 0.4**2) + 3 * 0.02)
    assert source.trace_cap == 1.5
    np.testing.assert_array_equal(free.offset_covariance, np.zeros((3, 3)))
    np.testing.assert_allclose(source.offset_covariance, options["source_offset"])
    # The outer cap contains every allowed source image; check a saturated rank-one source.
    source_candidate = np.diag([1.5, 0, 0])
    assert np.trace(source.covariance(source_candidate)) <= free.trace_cap + 1e-12
    assert (
        records["robust_psd"]["training_observation_bounds"]
        is records["robust_source"]["training_observation_bounds"]
    )
    assert "grid covariance K" in records["robust_psd"]["fit_diagnostics"]["trace_cap_scope"]
    assert "source Q" in records["robust_source"]["fit_diagnostics"]["trace_cap_scope"]


def test_default_spread_contrasts_read_only_fully_observed_supports():
    require_cvxpy()
    rng = np.random.default_rng(212)
    complete = rng.normal(scale=0.2, size=(200, 7))
    values = complete.copy()
    values[:100, 4:], values[100:, :3] = np.nan, np.nan
    records = workflow.fit_hedging_rules(
        values,
        evaluation_matrix=np.eye(7),
        source_factors=(np.eye(7),),
        source_offset=None,
        cliques=((0, 1, 2, 3), (3, 4, 5, 6)),
    )
    bands = records["robust_psd"]["training_observation_bounds"]
    contrast = np.eye(7)
    contrast[0] = [1, -1, 0, 0, 0, 0, 0]
    contrast[5] = [0, 0, 0, 0, 0, 1, -1]
    np.testing.assert_array_equal(bands["contrast_matrix"], contrast)
    np.testing.assert_allclose(bands["noise_covariance"], 0.001 * contrast @ contrast.T)
    target_sample = complete[:100, 0] - complete[:100, 1]
    hedge_sample = complete[100:, 5] - complete[100:, 6]
    assert bands["corrected_covariance"][0, 0] == pytest.approx(np.mean(target_sample**2) - 0.002)
    assert bands["corrected_covariance"][5, 5] == pytest.approx(np.mean(hedge_sample**2) - 0.002)
    assert bands["pair_counts"][0, 0] == bands["pair_counts"][5, 5] == 100
    assert bands["pair_counts"][0, 5] == 0
    assert not np.any(np.all(bands["entry_pairs"] == [0, 5], axis=1))
    assert not records["direct_target_block"]["available"]


def test_complete_joint_baseline_matches_exact_corrected_joint_measurements():
    require_cvxpy()
    values, options = sample_case(complete=True)
    records = workflow.fit_hedging_rules(values, **options)
    covariance = values[:, [0, 2]].T @ values[:, [0, 2]] / len(values) - 0.001 * np.eye(2)
    expected = solve_target_block_hedge(
        covariance[0, 0], [covariance[0, 1]], covariance[1:, 1:], position_bounds=([-8], [8])
    )
    direct = records["direct_target_block"]
    assert direct["available"] and expected.available
    np.testing.assert_allclose(direct["positions"], expected.positions)
    np.testing.assert_allclose(direct["fitted_target_block"], covariance)
    assert direct["fit_diagnostics"]["joint_training_rows"] == len(values)
    assert direct["risk"] == pytest.approx(expected.risk)
    shrinkage = records["empirical_shrinkage"]
    assert shrinkage["available"]
    expected_shrink = 0.9 * covariance + 0.1 * np.diag(np.diag(covariance))
    np.testing.assert_allclose(shrinkage["fitted_target_block"], expected_shrink)
    assert shrinkage["fit_diagnostics"]["shrinkage_alpha"] == 0.1
    assert shrinkage["fitted_kernel"] is None


def test_direct_block_ignores_nonjoint_dates_and_unseen_latent_fillers():
    require_cvxpy()
    values, options = sample_case()
    complete, _ = sample_case(complete=True, seed=30)
    values[:30] = complete[:30]
    options["cliques"] = ((0, 1, 2),)
    first = workflow.fit_hedging_rules(values, **options)
    changed = values.copy()
    changed[30:, np.arange(3)] *= 25
    second = workflow.fit_hedging_rules(changed, **options)
    assert first["direct_target_block"]["available"]
    np.testing.assert_allclose(
        first["direct_target_block"]["positions"], second["direct_target_block"]["positions"]
    )
    np.testing.assert_allclose(
        first["direct_target_block"]["fitted_target_block"],
        second["direct_target_block"]["fitted_target_block"],
    )
    # Hidden parent-array values cannot enter a NaN-masked training argument.
    mask = np.isfinite(values)
    latent_parent = np.nan_to_num(values, nan=1e8)
    latent_parent[~mask] = -1e12
    remasked = np.where(mask, latent_parent, np.nan)
    third = workflow.fit_hedging_rules(remasked, **options)
    for method in first:
        assert first[method]["available"] == third[method]["available"]
        if first[method]["positions"] is not None:
            np.testing.assert_array_equal(first[method]["positions"], third[method]["positions"])
        np.testing.assert_array_equal(
            first[method]["training_observation_bounds"]["lower"],
            third[method]["training_observation_bounds"]["lower"],
        )
    assert "future" not in inspect.signature(workflow.fit_hedging_rules).parameters
    assert "truth" not in inspect.signature(workflow.fit_hedging_rules).parameters


def test_one_estimator_failure_does_not_replace_it_or_discard_other_rules(monkeypatch):
    require_cvxpy()
    values, options = sample_case()

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate ridge failure")

    monkeypatch.setattr(workflow, "fit_covariance", fail)
    records = workflow.fit_hedging_rules(values, **options)
    assert not records["ridge"]["available"]
    assert records["ridge"]["positions"] is None
    assert "deliberate ridge failure" in records["ridge"]["reason"]
    assert records["canonical_coherent"]["available"]
    assert records["robust_psd"]["available"]
    assert records["robust_source"]["available"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"source_trace_cap": -1}, "trace_cap"),
        ({"noise_variance": -1}, "noise_variance"),
        ({"relative_regularization": 0}, "relative_regularization"),
        ({"confidence_level": 1}, "confidence_level"),
        ({"contrast_matrix": np.ones((3, 3))}, "invertible"),
        ({"shrinkage_alpha": 1.1}, "shrinkage_alpha"),
        ({"target": [1, 0]}, "exposure"),
        ({"evaluation_matrix": np.eye(2)}, "evaluation_matrix"),
    ],
)
def test_invalid_workflow_inputs_are_rejected_before_fitting(change, match):
    values, options = sample_case()
    options.update(change)
    with pytest.raises(ValueError, match=match):
        workflow.fit_hedging_rules(values, **options)


def test_infinite_or_all_missing_training_values_are_rejected():
    values, options = sample_case()
    invalid = values.copy()
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        workflow.fit_hedging_rules(invalid, **options)
    with pytest.raises(ValueError, match="training observation"):
        workflow.fit_hedging_rules(np.full(values.shape, np.nan), **options)
