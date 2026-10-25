"""Fit a Gaussian completion model and keep structural uncertainty separate.

This uses a centered finite-grid Gaussian control, not empirical price data.
The separate transport example validates trajectory-to-innovation adjustment.
Run: uv run python examples/statistical_completion.py
"""

import importlib.util

import numpy as np

from spde_covariance import (
    FragmentLayout,
    ObservedInnovationBatch,
    completion_risk_interval,
    condition_gaussian_kernel,
    gaussian_completion_region,
    select_gaussian_completion,
)
from spde_covariance.observations import Observation


def main():
    rng = np.random.default_rng(20261030)
    # This control obeys X0 independent X2 conditional on X1.
    truth = np.array([[1.0, 0.4, 0.16], [0.4, 1.0, 0.4], [0.16, 0.4, 1.0]])
    noisy = rng.multivariate_normal(np.zeros(3), truth + 0.02 * np.eye(3), size=192)

    def observed(start, stop):
        times = np.arange(start, stop + 1) / 52
        rows = tuple(np.array([i % 2, i % 2 + 1]) for i in range(start, stop))
        observations = tuple(
            Observation(times[j + 1], noisy[i, row], np.eye(3)[row], 0.02 * np.eye(2))
            for j, (i, row) in enumerate(zip(range(start, stop), rows, strict=True))
        )
        data = ObservedInnovationBatch(times, observations, "grid-control", "Gaussian control")
        layout = FragmentLayout(
            np.arange(3),
            np.ones(3),
            ((0.0, 1.1), (0.9, 2.0)),
            rows,
            np.eye(3),
            data.basis_id,
        )
        return data, layout

    training, train_layout = observed(0, 128)
    validation, valid_layout = observed(128, 192)
    selected = select_gaussian_completion(
        training,
        validation,
        training_layout=train_layout,
        validation_layout=valid_layout,
        reference_kernel=np.eye(3),  # external, predeclared unit-scale shrinkage reference
        regularizations=[0.5 / 128, 2 / 128, 8 / 128],
        covariance_assumption="common",
    )
    if not selected.available:
        raise RuntimeError("all Gaussian completion candidates failed; inspect the selection audit")
    kernel = selected.fit.kernel
    assert np.linalg.eigvalsh(kernel)[0] > 0
    print("Selected regularization:", selected.selected_regularization)
    print("Signal covariance kernel:\n", kernel)
    context = validation.observations[0]
    conditional = condition_gaussian_kernel(
        kernel,
        context.y,
        valid_layout.row_to_nodes[0],
        noise_covariance=context.noise_covariance,
    )
    print("Unobserved-node conditional mean:", conditional.mean)
    print("Plug-in conditional covariance:\n", conditional.covariance)
    region = gaussian_completion_region(
        tuple(item.y for item in training.observations),
        train_layout.row_to_nodes,
        noise_covariances=tuple(item.noise_covariance for item in training.observations),
        size=3,
        confidence_level=0.95,
    )
    if not region.available:
        raise RuntimeError("partial covariance confidence region is unavailable")
    if importlib.util.find_spec("cvxpy") is not None:
        risk = completion_risk_interval(region, np.array([1.0, 0.0, -1.0]) / np.sqrt(2))
        if not risk.available:
            raise RuntimeError("risk interval solve failed; inspect its diagnostics")
        print("Risk range over all PSD completions:", (risk.lower, risk.upper))
    else:
        print("Partial confidence bands are available; risk endpoints use the reference extra.")
    print("Native observations do not validate the missing cross-covariance convention.")


if __name__ == "__main__":
    main()
