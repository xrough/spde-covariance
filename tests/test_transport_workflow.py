"""Independent deterministic moment control for the complete transport workflow."""

import numpy as np

from spde_covariance.estimation import fit_covariance
from spde_covariance.trajectories import CurveTrajectory, extract_innovations
from spde_covariance.transport import ExponentialTransport


def test_exact_transport_adjustment_and_estimation_match_closed_form():
    model = ExponentialTransport()
    delta = 1 / 12
    diffusion = np.diag([0.4, 0.3, 0.2, 0.1])
    covariance = model.innovation_covariance(diffusion, delta)
    dimension = model.dimension
    factor = np.sqrt(dimension) * np.linalg.cholesky(covariance)
    values = np.concatenate((factor.T, -factor.T))
    states = [np.array([0.5, -0.2, 0.1, 0.3])]
    for innovation in values:
        states.append(model.matrix(delta) @ states[-1] + np.sqrt(delta) * innovation)
    trajectory = CurveTrajectory(
        times=np.arange(len(states)) * delta,
        coefficients=np.asarray(states),
        basis_id=model.basis_id,
        propagation_method=model.kind,
    )
    batch = extract_innovations(trajectory, model)
    np.testing.assert_allclose(batch.values, values, atol=1e-14)
    regularization = 0.01
    estimate = fit_covariance(
        batch.observe(),
        regularization=regularization,
        trace_cap=2,
        numerical_error_budget=1e-7,
    )
    assert estimate.converged
    # O=I makes the Gram identity: with inactive trace constraint the unique
    # ridge minimizer has this independent, closed-form expression.
    np.testing.assert_allclose(estimate.covariance, covariance / (1 + regularization), atol=1e-12)
