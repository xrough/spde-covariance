import numpy as np
import pytest

from spde_covariance.hedging_controls import fixed_power_hedge_control
from spde_covariance.hedging_pilot import (
    HedgingPilotConfig,
    _paired,
    evaluate_issued_hedge,
    mask_training_prefix,
    paired_observation_parents,
)


def test_training_prefix_never_reads_future_and_design_views_are_nested():
    parent = np.arange(140.0).reshape(20, 7)
    native = mask_training_prefix(parent, 12, "fragments")
    bridge = mask_training_prefix(parent, 12, "bridge")
    complete = mask_training_prefix(parent, 12, "complete")
    assert np.all(np.isfinite(bridge)[np.isfinite(native)])
    np.testing.assert_array_equal(native[np.isfinite(native)], bridge[np.isfinite(native)])
    np.testing.assert_array_equal(complete[np.isfinite(native)], native[np.isfinite(native)])
    parent[12:] = np.nan
    np.testing.assert_array_equal(native, mask_training_prefix(parent, 12, "fragments"))


def test_correct_conditional_parent_laws_and_exact_shared_fragment_data():
    control = fixed_power_hedge_control(source_dimension=8, reference_dimension=64)
    parents = paired_observation_parents(control, count=20_000, noise_variance=0.001, seed=42)
    native = [mask_training_prefix(p, 20_000, "fragments") for p in parents]
    np.testing.assert_array_equal(native[0], native[1])
    for parent, kernel in zip(parents, control["kernels"], strict=True):
        covariance = kernel + 0.001 * np.eye(7)
        root = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(root, parent.T).T
        np.testing.assert_allclose(whitened.T @ whitened / len(parent), np.eye(7), atol=0.04)


def test_noise_corrected_future_score_and_native_visibility_exactly():
    a, b = np.eye(7)[0] - np.eye(7)[1], np.eye(7)[5] - np.eye(7)[6]
    audit = np.arange(35.0).reshape(5, 7)
    for design, count in (("fragments", 0), ("bridge", 0), ("complete", 5)):
        row = evaluate_issued_hedge(
            [1], audit, np.eye(7), target=a, instrument=b, noise_variance=0.2, native_design=design
        )
        residual = a - b
        assert row["complete_audit_score"] == np.mean(
            (audit @ residual) ** 2 - 0.2 * (residual @ residual)
        )
        assert row["native_score_count"] == count
    zero = evaluate_issued_hedge([0], audit, np.eye(7), target=a, instrument=b, noise_variance=0.2)
    assert zero["native_score_count"] == 3  # first clique is measured on rows 0,2,4
    tiny = evaluate_issued_hedge(
        [1e-20], audit, np.eye(7), target=a, instrument=b, noise_variance=0.2
    )
    assert tiny["native_score_count"] == 0  # no silent rounding of an issued position


def test_paired_mcse_uses_replication_differences_and_separate_observed_scores():
    rows = []
    for rep, value in enumerate([1.0, 3.0]):
        for method, offset in (("canonical_coherent", 0), ("ridge", value)):
            rows.append(
                dict(
                    replication=rep,
                    world=0,
                    design="complete",
                    training_count=128,
                    method=method,
                    available=True,
                    true_residual_variance=10 + offset,
                    complete_audit_score=20 + 2 * offset,
                )
            )
    paired = {r["metric"]: r for r in _paired(rows) if r["contrast"].startswith("ridge")}
    assert paired["true_residual_variance"]["mean"] == 2
    assert paired["true_residual_variance"]["mcse"] == 1
    assert paired["complete_audit_score"]["mean"] == 4
    assert paired["complete_audit_score"]["mcse"] == 2


@pytest.mark.parametrize(
    "options",
    [
        {"replications": 0},
        {"training_counts": [32, 31]},
        {"source_trace_cap": 0.5},
        {"confidence_level": 1},
    ],
)
def test_invalid_pilot_config(options):
    with pytest.raises(ValueError):
        HedgingPilotConfig(**options)
