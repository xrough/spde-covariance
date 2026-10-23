"""Scientific controls and observation schedules for the RQ2 experiment."""

from dataclasses import fields

import numpy as np
import pytest

from spde_covariance.geometry import diagnose_design
from spde_covariance.observations import Observation
from spde_covariance.rq2_design import (
    RQ2Config,
    build_geometry,
    covariance_paths,
    generate_data,
)


def test_covariance_paths_separate_level_shape_rotation_and_jump_controls():
    config = RQ2Config()
    paths = covariance_paths(config)
    base = np.diag([0.4, 0.3, 0.2, 0.1])
    for path in paths.values():
        assert path.shape == (512, 4, 4)
        np.testing.assert_allclose(path, path.swapaxes(1, 2), atol=1e-14)
        assert np.linalg.eigvalsh(path).min() > 0
        assert np.trace(path, axis1=1, axis2=2).max() <= config.trace_cap
    np.testing.assert_array_equal(paths["constant"], np.broadcast_to(base, (512, 4, 4)))
    traces = np.trace(paths["scalar"], axis1=1, axis2=2)
    np.testing.assert_allclose(traces[[0, -1]], [0.75, 1.25])
    np.testing.assert_allclose(
        paths["scalar"] / traces[:, None, None], np.broadcast_to(base, (512, 4, 4))
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(paths["rotating"]),
        np.broadcast_to([0.1, 0.2, 0.3, 0.4], (512, 4)),
        atol=1e-14,
    )
    np.testing.assert_allclose(paths["rotating"][0], base, atol=1e-14)
    np.testing.assert_allclose(paths["rotating"][-1], np.diag([0.1, 0.3, 0.2, 0.4]), atol=1e-14)
    assert abs(paths["rotating"][256, 0, 3]) > 0.1
    np.testing.assert_array_equal(paths["abrupt"][:256], paths["constant"][:256])
    np.testing.assert_array_equal(
        paths["abrupt"][256:], np.broadcast_to(np.diag([0.1, 0.2, 0.3, 0.4]), (256, 4, 4))
    )


def test_exposures_use_raw_maturity_rows_and_unit_hilbert_norm():
    model, point_map, exposures = build_geometry()
    maturities = np.array([1 / 12, 0.5, 1.5, 3.0])
    np.testing.assert_allclose(point_map, model.evaluate(maturities))
    assert np.linalg.matrix_rank(point_map) == 4
    weights = {
        "near_spread": np.array([1, -1, 0, 0]),
        "cross_spread": np.array([0, 1, -1, 0]),
        "butterfly": np.array([12, -17, 5, 0]),
    }
    assert weights["butterfly"].sum() == 0
    assert weights["butterfly"] @ maturities == pytest.approx(0.0, abs=1e-14)
    for name, value in weights.items():
        expected = value @ point_map
        np.testing.assert_allclose(exposures[name], expected / np.linalg.norm(expected))
        assert np.linalg.norm(exposures[name]) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def paired_data():
    config = RQ2Config(
        intervals=80,
        windows=(16, 32, 64),
        endpoints=(80,),
        replications=2,
    )
    return config, generate_data(config, 0)


def test_coverage_designs_are_exact_slices_with_shared_noise(paired_data):
    config, data = paired_data
    assert {field.name for field in fields(Observation)} == {
        "time",
        "y",
        "operator",
        "noise_covariance",
    }
    np.testing.assert_allclose(data.times, np.arange(1, 81) / 52)
    for scenario, designs in data.designs.items():
        for index, dense in enumerate(designs["dense"]):
            assert dense.time == data.times[index]
            np.testing.assert_array_equal(dense.operator, data.point_map)
            np.testing.assert_array_equal(dense.noise_covariance, config.noise_variance * np.eye(4))
            full = designs["full"][index]
            np.testing.assert_array_equal(full.operator, np.eye(4))
            np.testing.assert_array_equal(full.y, data.latent[scenario][index])
            np.testing.assert_array_equal(full.noise_covariance, np.zeros((4, 4)))
            for name, joint_every in (("split", None), ("joint8", 8), ("joint64", 64)):
                item = designs[name][index]
                rows = (
                    np.arange(4)
                    if joint_every and (index + 1) % joint_every == 0
                    else (np.array([0, 1]) if index % 2 == 0 else np.array([2, 3]))
                )
                assert item.time == dense.time
                np.testing.assert_array_equal(item.y, dense.y[rows])
                np.testing.assert_array_equal(item.operator, dense.operator[rows])
                np.testing.assert_array_equal(
                    item.noise_covariance, dense.noise_covariance[np.ix_(rows, rows)]
                )
        # Dense samples are generated once, and every covariance scenario uses
        # the same measurement noise draws for paired Monte Carlo comparisons.
        noise = np.array([item.y for item in designs["dense"]]) - (
            data.latent[scenario] @ data.point_map.T
        )
        baseline_noise = np.array([item.y for item in data.designs["constant"]["dense"]]) - (
            data.latent["constant"] @ data.point_map.T
        )
        np.testing.assert_allclose(noise, baseline_noise, atol=1e-15)


def test_reproducibility_replication_independence_and_subset_stability(paired_data):
    config, first = paired_data
    second = generate_data(config, 0)
    independent = generate_data(config, 1)
    subset = generate_data(
        RQ2Config(
            intervals=80,
            windows=(16, 32, 64),
            endpoints=(80,),
            replications=2,
            scenarios=("rotating", "constant"),
        ),
        0,
    )
    for scenario in first.truths:
        np.testing.assert_array_equal(first.latent[scenario], second.latent[scenario])
        assert not np.array_equal(first.latent[scenario], independent.latent[scenario])
    for scenario in subset.truths:
        np.testing.assert_array_equal(first.latent[scenario], subset.latent[scenario])
        for original, selected in zip(
            first.designs[scenario]["dense"], subset.designs[scenario]["dense"], strict=True
        ):
            np.testing.assert_array_equal(original.y, selected.y)


def test_pair_coverage_controls_rank_and_target_identification():
    config = RQ2Config(scenarios=("constant",))
    data = generate_data(config, 0)
    targets = {name: np.outer(value, value) for name, value in data.exposures.items()}
    split = diagnose_design(
        [item.operator for item in data.designs["constant"]["split"][-64:]],
        targets=targets,
        rank_rtol=config.rank_rtol,
    )
    assert split["rank"] == 6
    assert split["dimension"] == 10
    assert split["targets"]["near_spread"]["identified"]
    assert not split["targets"]["cross_spread"]["identified"]
    assert not split["targets"]["butterfly"]["identified"]
    # Exact one-based trailing slices distinguish recent and stale joint dates.
    for endpoint, window, expected_rank in (
        (272, 16, 6),
        (272, 32, 10),
        (304, 32, 6),
        (304, 64, 10),
    ):
        observations = data.designs["constant"]["joint64"][endpoint - window : endpoint]
        assert len(observations) == window
        assert observations[0].time == pytest.approx((endpoint - window + 1) / 52)
        assert observations[-1].time == pytest.approx(endpoint / 52)
        geometry = diagnose_design(
            [item.operator for item in observations], targets=targets, rank_rtol=config.rank_rtol
        )
        assert geometry["rank"] == expected_rank
        assert geometry["targets"]["cross_spread"]["identified"] == (expected_rank == 10)
        assert geometry["targets"]["butterfly"]["identified"] == (expected_rank == 10)


@pytest.mark.parametrize(
    "parameters",
    [
        {"seed": -1},
        {"seed": True},
        {"replications": 0},
        {"intervals": 0},
        {"delta": 0},
        {"delta": np.inf},
        {"noise_variance": -1},
        {"relative_regularization": np.nan},
        {"numerical_error_budget": 0},
        {"trace_cap": 1.0},
        {"rank_rtol": 1},
        {"rank_rtol": 0},
        {"windows": ()},
        {"windows": (16, 16)},
        {"windows": (False,)},
        {"windows": (257,)},
        {"endpoints": (513,)},
        {"endpoints": (256.0,)},
        {"sensitivity_window": 513},
        {"sensitivity_relatives": (0.0,)},
        {"sensitivity_relatives": (np.inf,)},
        {"sensitivity_relatives": (0.01, 0.01)},
        {"scenarios": ()},
        {"scenarios": ("unknown",)},
        {"scenarios": ("constant", "constant")},
        {"reference_checks": 1},
        {"name": " "},
    ],
)
def test_invalid_configuration_is_rejected_before_simulation(parameters):
    with pytest.raises(ValueError):
        RQ2Config(**parameters)


@pytest.mark.parametrize("replication", [-1, True, 20, 0.5])
def test_invalid_replication_is_rejected(replication):
    with pytest.raises(ValueError):
        generate_data(RQ2Config(), replication)
