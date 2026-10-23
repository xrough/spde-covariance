"""Separate forcing, projection and time resolution of one fixed source model."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from spde_covariance import infinite_benchmark as benchmark
from spde_covariance.infinite_benchmark import (
    _prepare_bands,
    _prepare_population,
    coarsen_time_grid,
    fixed_exposures,
    population_audit,
    run_infinite_transport_benchmark,
    simulate_coupled_paths,
    state_covariance_sum,
)
from spde_covariance.infinite_config import InfiniteTransportConfig
from spde_covariance.laguerre import LaguerreTransport


def tiny_config(**changes):
    values = dict(
        replications=2,
        forcing_dimensions=(4, 8),
        projection_dimensions=(2, 4),
        deltas=(0.25, 0.125),
        fit_dimensions=(2,),
        reference_checks=False,
    )
    values.update(changes)
    return InfiniteTransportConfig(**values)


def test_default_config_and_invalid_values():
    config = InfiniteTransportConfig()
    assert config.forcing_dimensions == (64, 128, 256)
    assert config.projection_dimensions == (8, 16, 32, 64)
    assert config.source_exponent == 3
    assert config.source_trace == 1
    assert config.fine_intervals == 208
    assert config.fine_delta == 1 / 208
    for changes in (
        {"replications": 0},
        {"source_exponent": 2},
        {"source_trace": 0},
        {"forcing_dimensions": ()},
        {"forcing_dimensions": (8, 4)},
        {"projection_dimensions": (2, 16)},
        {"projection_dimensions": (2, 2)},
        {"fit_dimensions": (3,)},
        {"deltas": (0.3, 0.125)},
        {"deltas": (0.2500000001, 0.125)},
        {"deltas": (0.25, 0.25)},
        {"deltas": (0.25, float("nan"))},
        {"seed": True},
        {"numerical_error_budget": 0},
        {"quadrature_order": 0},
        {"reference_checks": 1},
    ):
        with pytest.raises(ValueError):
            tiny_config(**changes)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 16, 52, 208])
def test_binary_covariance_sum_matches_naive_noncommuting_recursion(count):
    shift = np.array([[0.8, 0.2], [0.0, 0.5]])
    step = np.array([[0.1, 0.03], [0.03, 0.07]])
    expected, expected_sum = np.zeros((2, 2)), np.zeros((2, 2))
    for _ in range(count):
        expected_sum += expected
        expected = shift @ expected @ shift.T + step
    result, result_sum = state_covariance_sum(shift, step, count)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result_sum, expected_sum, rtol=0, atol=2e-11)
    with pytest.raises(ValueError):
        state_covariance_sum(shift, step, -1)


def test_exposures_use_one_infinite_h_normalization_across_prefixes():
    short, long = fixed_exposures(LaguerreTransport(4)), fixed_exposures(LaguerreTransport(64))
    for name in short:
        np.testing.assert_allclose(short[name], long[name][:4], rtol=0, atol=1e-14)
        assert short[name][0] == pytest.approx(0.0, abs=1e-14)
        assert np.linalg.norm(short[name]) < np.linalg.norm(long[name]) <= 1 + 1e-12
        assert not np.isclose(np.linalg.norm(short[name]), 1)
        assert not short[name].flags.writeable
    point = long["point_spread"]
    rows = LaguerreTransport(64).evaluate([0.5, 1.5])
    np.testing.assert_allclose(point, (rows[0] - rows[1]) / np.sqrt(np.exp(-0.5) - np.exp(-1.5)))


def test_full_population_matches_independent_constants_and_resolves_reference():
    config = InfiniteTransportConfig(fit_dimensions=(), reference_checks=False)
    audit = population_audit(config)
    assert len(audit["population"]) == 36
    assert all(row["passed"] for row in audit["numerical_checks"])
    weekly = {
        row["J"]: row for row in audit["population"] if row["M"] == 256 and row["delta"] == 1 / 52
    }
    # Independently evaluated reference values recorded before runner construction.
    assert weekly[8]["source_to_innovation_hs_error"] == pytest.approx(0.0025788233643, abs=2e-13)
    assert weekly[8]["projection_innovation_hs_error"] == pytest.approx(0.00172012552, abs=5e-12)
    assert weekly[64]["projection_innovation_hs_error"] == pytest.approx(1.09713759e-5, abs=5e-13)
    assert weekly[8]["full_source_tail_bound"] == pytest.approx(6.3222052103e-6, rel=1e-10)
    assert weekly[8]["full_state_mse_tail_bound"] == pytest.approx(8.6480145300e-6, rel=1e-10)
    assert all(row["tail_resolved"] for row in audit["population"] if row["M"] == 256)
    assert audit["kernels"]["reference_M"] == 256
    assert audit["kernels"]["source"].shape == (61, 61)


def test_population_skips_only_invalid_j_and_retains_unresolved_tails():
    config = tiny_config(projection_dimensions=(2, 6), fit_dimensions=())
    audit = population_audit(config)
    expected = {
        (m, j, delta)
        for m in config.forcing_dimensions
        for j in config.projection_dimensions
        for delta in config.deltas
        if j <= m
    }
    assert {(row["M"], row["J"], row["delta"]) for row in audit["population"]} == expected
    assert all(row["passed"] for row in audit["numerical_checks"])
    assert not any(row["tail_resolved"] for row in audit["population"])


def test_band_marginals_and_cross_level_covariances_are_correct():
    config = tiny_config()
    prepared = _prepare_population(config)
    bands = _prepare_bands(config, prepared)
    for m in config.forcing_dimensions:
        covariance = np.zeros((m, m))
        for band in bands:
            if band["upper"] <= m:
                covariance[: band["upper"], : band["upper"]] += band["covariance"]
        expected = prepared["models"][m].cumulative_covariance(
            prepared["source_covariances"][m],
            config.fine_delta,
        )
        np.testing.assert_allclose(covariance, expected, rtol=1e-12, atol=1e-15)
    # Cross-level covariance is the embedded covariance of the shared low forcing,
    # not a cross-product of roots applied to the same unrelated normal vector.
    low, high = config.forcing_dimensions
    low_covariance = bands[0]["root"] @ bands[0]["root"].T
    joint_root_low = np.column_stack((bands[0]["root"], np.zeros((low, high))))
    joint_root_high = np.column_stack(
        (np.pad(bands[0]["root"], ((0, high - low), (0, 0))), bands[1]["root"])
    )
    np.testing.assert_allclose(
        joint_root_low @ joint_root_high.T,
        np.pad(low_covariance, ((0, 0), (0, high - low))),
        atol=1e-15,
    )


def test_seed_reproducibility_and_pathwise_omitted_band_coupling():
    config = tiny_config()
    prepared = _prepare_population(config)
    bands = _prepare_bands(config, prepared)
    first = simulate_coupled_paths(config, 0, prepared=prepared, bands=bands)
    second = simulate_coupled_paths(config, 0, prepared=prepared, bands=bands)
    independent = simulate_coupled_paths(config, 1, prepared=prepared, bands=bands)
    for m in config.forcing_dimensions:
        np.testing.assert_array_equal(first.paths[m], second.paths[m])
        assert not np.array_equal(first.paths[m], independent.paths[m])
        assert not first.paths[m].flags.writeable
    low, high = config.forcing_dimensions
    omitted = first.band_increments[1]["values"]
    np.testing.assert_allclose(
        first.fine_increments[high] - np.pad(first.fine_increments[low], ((0, 0), (0, high - low))),
        omitted,
        atol=1e-16,
    )
    omitted_path = np.zeros_like(first.paths[high])
    shift = prepared["models"][high].matrix(config.fine_delta)
    for index, increment in enumerate(omitted):
        omitted_path[index + 1] = shift @ omitted_path[index] + increment
    np.testing.assert_allclose(
        first.paths[high] - np.pad(first.paths[low], ((0, 0), (0, high - low))),
        omitted_path,
        atol=1e-15,
    )
    with pytest.raises(TypeError):
        first.paths[low] = np.zeros_like(first.paths[low])
    with pytest.raises(ValueError):
        simulate_coupled_paths(config, 2, prepared=prepared, bands=bands)


def test_time_coarsening_and_operational_hidden_term_use_the_same_full_path():
    config = tiny_config()
    prepared = _prepare_population(config)
    simulation = simulate_coupled_paths(config, 0, prepared=prepared)
    m, j, delta = 8, 2, 0.25
    coarse = coarsen_time_grid(
        simulation.paths[m],
        simulation.fine_increments[m],
        prepared["models"][m],
        fine_delta=config.fine_delta,
        delta=delta,
    )
    assert coarse["composition_error"] < 1e-13
    assert coarse["stride"] == 2
    states = coarse["states"]
    operational = (
        states[1:, :j] - states[:-1, :j] @ LaguerreTransport(j).matrix(delta).T
    ) / np.sqrt(delta)
    predicted_leakage = (
        states[:-1] @ prepared["cases"][(m, j, delta)]["leakage_map"].T / np.sqrt(delta)
    )
    np.testing.assert_allclose(
        operational - coarse["innovations"][:, :j], predicted_leakage, atol=1e-14
    )
    assert np.linalg.norm(predicted_leakage) > 1e-8
    # Full-M extraction removes transport exactly; autonomous low-J extraction
    # is deliberately not substituted for the projected full-model innovation.
    direct = (states[1:] - states[:-1] @ prepared["models"][m].matrix(delta).T) / np.sqrt(delta)
    np.testing.assert_array_equal(coarse["innovations"], direct)
    with pytest.raises(ValueError):
        coarsen_time_grid(
            simulation.paths[m],
            simulation.fine_increments[m],
            prepared["models"][m],
            fine_delta=config.fine_delta,
            delta=0.3,
        )


def test_gaussian_oracle_mse_control_with_independent_replications():
    config = tiny_config(
        replications=300,
        forcing_dimensions=(1,),
        projection_dimensions=(1,),
        fit_dimensions=(),
        deltas=(0.125,),
    )
    prepared = _prepare_population(config)
    bands = _prepare_bands(config, prepared)
    variance = prepared["innovations"][(1, 0.125)][0, 0]
    errors = []
    for replication in range(config.replications):
        simulation = simulate_coupled_paths(config, replication, prepared=prepared, bands=bands)
        values = simulation.fine_increments[1][:, 0] / np.sqrt(config.fine_delta)
        errors.append((np.mean(values**2) - variance) ** 2)
    expected = 2 * variance**2 / config.fine_intervals
    assert prepared["population"][0]["oracle_gaussian_mse"] == pytest.approx(expected)
    mcse = np.std(errors, ddof=1) / np.sqrt(config.replications)
    assert abs(np.mean(errors) - expected) <= 4 * mcse


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    config = tiny_config()
    output = tmp_path_factory.mktemp("infinite") / "run"
    run_infinite_transport_benchmark(config, output)
    return config, output


def test_end_to_end_outputs_and_resolution_flags(tiny_run):
    config, output = tiny_run
    expected = {
        "config.json",
        "manifest.json",
        "status.json",
        "results.json",
        "population.json",
        "numerical_checks.json",
        "population_covariances.npz",
        "replications.csv",
        "risks.csv",
        "fit_results.json",
        "report.md",
        "covariance_kernels.png",
        "resolution_errors.png",
        "leakage_risk.png",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    results = json.loads((output / "results.json").read_text())
    assert results["acceptance"]["passed"]
    assert not results["acceptance"]["reference_resolution_passed"]
    assert results["acceptance"]["maximum_time_composition_error"] < config.transport_tolerance
    assert results["acceptance"]["maximum_leakage_identity_error"] < config.transport_tolerance
    assert results["population"] == json.loads((output / "population.json").read_text())
    for path in output.glob("*.json"):
        json.dumps(json.loads(path.read_text()), allow_nan=False)
    with np.load(output / "population_covariances.npz", allow_pickle=False) as matrices:
        for row in results["population"]:
            matrix = matrices[row["matrix_key_prefix"] + "_innovation"]
            assert matrix.shape == (row["J"], row["J"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert "independent centered Gaussian innovations" not in manifest["experiment"]
    assert "dependent" in manifest["operational_residuals"]
    fits = json.loads((output / "fit_results.json").read_text())
    assert all(row["J"] <= 16 and row["M"] == max(config.forcing_dimensions) for row in fits)
    assert all(row["gram_operator_norm"] == 1 and row["regularization"] == 0.001 for row in fits)
    assert all(row["basis_id"] and row["basis_family_id"] for row in fits)
    assert all("dependent" in row["target"] for row in fits if row["input_kind"] == "operational")


def test_saved_risk_biases_and_paired_mcse_match_raw_replications(tiny_run):
    config, output = tiny_run
    results = json.loads((output / "results.json").read_text())
    with (output / "risks.csv").open() as stream:
        risks = list(csv.DictReader(stream))
    with (output / "replications.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for summary in results["risk_summary"]:
        selected = [
            row
            for row in risks
            if int(row["M"]) == summary["M"]
            and int(row["J"]) == summary["J"]
            and float(row["delta"]) == summary["delta"]
            and row["exposure"] == summary["exposure"]
        ]
        assert len(selected) == config.replications
        errors = np.array([float(row["oracle_risk_mse"]) for row in selected])
        assert summary["oracle_risk_mse_mean"] == pytest.approx(errors.mean())
        assert summary["oracle_risk_mse_mcse"] == pytest.approx(errors.std(ddof=1) / np.sqrt(2))
        paired_risk = np.array(
            [float(row["operational_risk_mse"]) - float(row["oracle_risk_mse"]) for row in selected]
        )
        assert summary["operational_minus_oracle_risk_mse_mean"] == pytest.approx(
            paired_risk.mean()
        )
        assert summary["operational_minus_oracle_risk_mse_mcse"] == pytest.approx(
            paired_risk.std(ddof=1) / np.sqrt(2)
        )
        assert summary["projection_bias"] + summary["forcing_bias"] == pytest.approx(
            summary["projected_risk_truth"] - summary["reference_risk_truth"]
        )
    for summary in results["summary"]:
        selected = [
            row
            for row in rows
            if int(row["M"]) == summary["M"]
            and int(row["J"]) == summary["J"]
            and float(row["delta"]) == summary["delta"]
        ]
        differences = np.array(
            [
                float(row["operational_covariance_mse"]) - float(row["oracle_covariance_mse"])
                for row in selected
            ]
        )
        assert summary["operational_minus_oracle_covariance_mse_mean"] == pytest.approx(
            differences.mean()
        )
        assert summary["operational_minus_oracle_covariance_mse_mcse"] == pytest.approx(
            differences.std(ddof=1) / np.sqrt(2)
        )
    for summary in results["paired_differences"]:
        selected = [
            row
            for row in rows
            if int(row["M"]) == summary["M"]
            and int(row["J"]) == summary["J"]
            and float(row["delta"]) == summary["delta"]
        ]
        reference = {
            int(row["replication"]): row
            for row in rows
            if int(row["M"]) == summary["reference_M"]
            and int(row["J"]) == summary["J"]
            and float(row["delta"]) == summary["delta"]
        }
        differences = [
            float(row[summary["metric"]])
            - float(reference[int(row["replication"])][summary["metric"]])
            for row in selected
        ]
        assert summary["mean_difference"] == pytest.approx(np.mean(differences))
        assert summary["mcse"] == pytest.approx(np.std(differences, ddof=1) / np.sqrt(2))


def test_small_reference_checks_are_nonempty_and_certified(tmp_path):
    pytest.importorskip("cvxpy")
    config = tiny_config(replications=1, reference_checks=True)
    output = tmp_path / "reference"
    run_infinite_transport_benchmark(config, output)
    result = json.loads((output / "results.json").read_text())
    assert result["acceptance"]["reference_cases"] == 2
    assert result["acceptance"]["reference_passed"]
    assert {row["input_kind"] for row in result["reference_checks"]} == {"oracle", "operational"}
    assert all(
        row["J"] <= 8 and row["covariance_error_bound"] <= config.numerical_error_budget
        for row in result["reference_checks"]
    )


def test_failed_fit_preserves_evidence_and_rejects_run(tmp_path, monkeypatch):
    from spde_covariance import infinite_reporting

    original = benchmark.fit_covariance

    def fail(*args, **kwargs):
        return replace(original(*args, **kwargs), converged=False, status="maximum_iterations")

    monkeypatch.setattr(benchmark, "fit_covariance", fail)
    monkeypatch.setattr(
        infinite_reporting,
        "write_report",
        lambda output, *_: (output / "report.md").write_text("Numerical acceptance failed.\n"),
    )
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="numerical checks failed"):
        run_infinite_transport_benchmark(tiny_config(replications=1), output)
    result = json.loads((output / "results.json").read_text())
    assert result["acceptance"]["failed_fits"] > 0
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_existing_outputs_and_invalid_inputs_are_preserved(tiny_run, tmp_path):
    config, output = tiny_run
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        run_infinite_transport_benchmark(config, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(TypeError):
        run_infinite_transport_benchmark({}, tmp_path / "invalid")
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("source_exponent = 1\n")
    with pytest.raises(ValueError):
        run_infinite_transport_benchmark(config_path, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()
