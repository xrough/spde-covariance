"""Transport trajectory -> fragments -> two explicit covariance strategies.

Run: uv run python examples/canonical_pipeline.py
No unobserved innovation values are supplied to either fitter.
"""

import numpy as np

import spde_covariance as spde
from spde_covariance.fragment_benchmark import quadrature_weights

model = spde.ExponentialTransport()
times = np.arange(513) / 52
simulation = spde.simulate_transport(model, times, np.diag([0.4, 0.3, 0.2, 0.1]), seed=1729)
innovations = spde.extract_innovations(simulation.trajectory, model)
nodes = np.linspace(0, 3, 41)
E = model.evaluate(nodes)
intervals = ((0.0, 1.8), (0.6, 2.4), (1.2, 3.0))
cliques = tuple(np.flatnonzero((nodes >= a - 1e-12) & (nodes <= b + 1e-12)) for a, b in intervals)
rows = tuple(cliques[i % 3] for i in range(innovations.n_intervals))
observed = spde.observe_innovations(
    innovations,
    [E[ids] for ids in rows],
    noise_covariances=[1e-6 * np.eye(len(ids)) for ids in rows],
    seed=1730,
)
layout = spde.FragmentLayout(nodes, quadrature_weights(nodes), intervals, rows, E, model.basis_id)
canonical = spde.fit_covariance_kernel(
    observed,
    layout=layout,
    method="canonical",
    method_options={"rank": 2, "psd_projection": True},
    covariance_assumption="common",
)
ridge = spde.fit_covariance_kernel(
    observed,
    layout=layout,
    method="ridge",
    method_options={"regularization": 0.001, "trace_cap": 2.0, "numerical_error_budget": 1e-4},
    covariance_assumption="common",
)
assert canonical.operator_fit is None
assert ridge.operator_fit.estimate.covariance_error_bound <= 1e-4
assert np.linalg.eigvalsh(canonical.projected_kernel)[0] >= -1e-10
print("Canonical estimand:", canonical.estimand)
print("Raw weighted PSD:", canonical.diagnostics["completion"]["output_psd"])
print(
    "Ridge certificate (H covariance distance):", ridge.operator_fit.estimate.covariance_error_bound
)
print("Grid kernel shape:", canonical.kernel.shape)
print(
    "Completion selects unseen entries by a structural convention; observation alone need not determine them."
)
