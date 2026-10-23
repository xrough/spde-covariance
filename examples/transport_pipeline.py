"""Simulate transport, extract innovations, and recover their covariance.

Run from the checkout: uv run python examples/transport_pipeline.py
No data files or network are required. The analytical covariance is used only
for evaluation; it is not passed to the estimator.
"""

import json

import numpy as np

from spde_covariance.trajectories import extract_innovations, simulate_transport
from spde_covariance.transport import ExponentialTransport
from spde_covariance.workflow import fit_innovation_covariance


def main():
    model = ExponentialTransport()
    delta = 1 / 52
    diffusion = np.diag([0.4, 0.3, 0.2, 0.1])
    simulation = simulate_transport(model, np.arange(1025) * delta, diffusion, seed=1729)
    innovations = extract_innovations(simulation.trajectory, model)
    fitted = fit_innovation_covariance(
        innovations,
        regularization=1e-3,
        trace_cap=2.0,
        covariance_assumption="common",
        numerical_error_budget=1e-5,
    )
    estimate = fitted.estimate
    truth = model.innovation_covariance(diffusion, delta)
    if not estimate.converged:
        raise RuntimeError(f"covariance solver did not converge: {estimate.status}")
    print(
        json.dumps(
            {
                "target": "common normalized innovation covariance V_delta",
                "propagation": model.kind,
                "intervals": innovations.n_intervals,
                "delta_years": delta,
                "relative_hilbert_schmidt_error": float(
                    np.linalg.norm(estimate.covariance - truth) / np.linalg.norm(truth)
                ),
                "innovation_vs_diffusion_difference": float(np.linalg.norm(truth - diffusion)),
                "certified_numerical_error": estimate.covariance_error_bound,
                "min_eigenvalue": estimate.min_eigenvalue,
                "trace": estimate.trace,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
