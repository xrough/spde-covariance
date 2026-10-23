import json

import numpy as np

from spde_covariance.benchmarks import BenchmarkConfig, benchmark_data, evaluation_metrics


def test_benchmark_observations_are_nested_paired_and_reproducible():
    config = BenchmarkConfig(train=16, validation=4, evaluation=4, window_candidates=(8, 16))
    _, truth, designs, ablations = benchmark_data(config, 0)
    _, other_truth, other, _ = benchmark_data(config, 0)
    np.testing.assert_array_equal(truth, other_truth)
    for i, (dense, sparse) in enumerate(zip(designs["dense"], designs["sparse"], strict=True)):
        subset = [i % 8, (i + 3) % 8]
        np.testing.assert_array_equal(sparse.y, dense.y[subset])
        np.testing.assert_array_equal(sparse.operator, dense.operator[subset])
        np.testing.assert_array_equal(sparse.y, other["sparse"][i].y)
        np.testing.assert_array_equal(sparse.operator, ablations["sparse"][i].operator)
        np.testing.assert_array_equal(
            sparse.noise_covariance, ablations["sparse"][i].noise_covariance
        )


def test_error_decomposition_separates_missing_covariance_directions():
    truth = np.array([[1, 0.4], [0.4, 1]])
    estimate = np.eye(2)
    operators = [np.array([[1.0, 0]]), np.array([[0.0, 1]])]
    result = evaluation_metrics(estimate, truth, operators, np.eye(2), np.eye(2))
    assert result["numerical_rank"] == 2
    assert result["observed_covariance_rmse"] == 0
    assert result["resolved_direction_error"] == 0
    assert result["weak_direction_error"] > 0
    np.testing.assert_allclose(
        result["operator_hs_error"] ** 2,
        result["resolved_direction_error"] ** 2 + result["weak_direction_error"] ** 2,
    )


def test_evaluation_truth_cannot_change_tuning_or_fits(tmp_path, monkeypatch):
    import spde_covariance.benchmarks as benchmarks

    config = BenchmarkConfig(
        train=16,
        validation=4,
        evaluation=4,
        replications=2,
        window_candidates=(8, 16),
        relative_candidates=(0.01,),
        shrinkage_candidates=(0, 0.5),
        reference_checks=False,
        stress_smoke=True,
    )
    monkeypatch.setattr(benchmarks, "run_stress_suite", lambda **kwargs: {})
    original = benchmarks.benchmark_data

    def run(output):
        try:
            benchmarks.run_benchmark(config, output)
        except RuntimeError as error:
            # The tiny sample is for a leakage check, not the recovery gate.
            assert "benchmark acceptance failed" in str(error)

    run(tmp_path / "first")

    def changed_truth(config, replication):
        model, truth, designs, ablations = original(config, replication)
        return model, 7 * truth, designs, ablations

    monkeypatch.setattr(benchmarks, "benchmark_data", changed_truth)
    run(tmp_path / "changed")
    first = json.loads((tmp_path / "first" / "tuning.json").read_text())
    changed = json.loads((tmp_path / "changed" / "tuning.json").read_text())
    assert first == changed
    # Only evaluation metrics change, including the recovery gate.
    one = json.loads((tmp_path / "first" / "results.json").read_text())["summary"][0]
    two = json.loads((tmp_path / "changed" / "results.json").read_text())["summary"][0]
    assert one["operator_hs_relative_error_mean"] != two["operator_hs_relative_error_mean"]
