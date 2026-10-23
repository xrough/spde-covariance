import numpy as np
import pytest

from spde_covariance.observations import Observation, delivery_average_map, point_observation_map
from spde_covariance.spaces import SplineSpace


def test_noise_corrected_measurement_can_be_indefinite_and_is_not_clipped():
    y = np.array([1.0, 0.0])
    observation = Observation(0.1, y, np.eye(2), np.eye(2))
    np.testing.assert_array_equal(observation.measurement, np.diag([0.0, -1.0]))
    y[0] = 9
    assert observation.y[0] == 1
    assert observation.dimension == observation.n_contracts == 2


@pytest.mark.parametrize(
    "y, operator, noise",
    [
        ([1.0], np.eye(2), np.eye(1)),
        ([1.0], [[1.0]], [[-1.0]]),
        ([np.nan], [[1.0]], [[0.0]]),
        ([1.0], [[np.inf]], [[0.0]]),
        ([1.0], [[1.0]], np.eye(2)),
    ],
)
def test_observation_validates_dimensions_and_noise(y, operator, noise):
    with pytest.raises(ValueError):
        Observation(0, np.asarray(y), np.asarray(operator), np.asarray(noise))


def test_delivery_average_integrates_linear_function_and_constant_tail():
    space = SplineSpace(n_basis=5, length=2)
    coefficients = space.project(lambda x: x, lambda x: np.ones_like(x))
    operator = delivery_average_map(space, [[0.2, 1.4], [1.0, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(operator @ coefficients, [0.8, 1.75, 2.0], atol=1e-12)
    np.testing.assert_allclose(point_observation_map(space, [0.5, 1.5]), space.evaluate([0.5, 1.5]))
    with pytest.raises(ValueError):
        delivery_average_map(space, [[2.0, 1.0]])
