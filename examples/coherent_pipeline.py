"""Known transport -> fragments -> compatible PSD estimate -> finite source test.

Run with: uv run python examples/coherent_pipeline.py
"""

import numpy as np

import spde_covariance as spde
from spde_covariance.fragment_benchmark import quadrature_weights

model = spde.LaguerreTransport(4)
simulation = spde.simulate_transport(
    model, np.arange(513) / 52, np.diag([0.4, 0.3, 0.2, 0.1]), seed=1729
)
innovations = spde.extract_innovations(simulation.trajectory, model)
nodes = np.linspace(0, 3, 41)
evaluation = model.evaluate(nodes)
intervals = ((0.0, 1.8), (0.6, 2.4), (1.2, 3.0))
cliques = tuple(np.flatnonzero((nodes >= a) & (nodes <= b)) for a, b in intervals)
rows = tuple(cliques[i % len(cliques)] for i in range(innovations.n_intervals))
observed = spde.observe_innovations(
    innovations,
    [evaluation[ids] for ids in rows],
    noise_covariances=[0.001 * np.eye(len(ids)) for ids in rows],
    seed=1730,
)
layout = spde.FragmentLayout(
    nodes, quadrature_weights(nodes), intervals, rows, evaluation, model.basis_id
)
fit = spde.fit_covariance_kernel(
    observed,
    layout=layout,
    method="coherent",
    covariance_assumption="common",
    method_options={
        "regularization": 0.001,
        "eigenvalue_floor": 1e-5,
        "numerical_error_budget": 1e-4,
        "rho": 0.2,
    },
)
assert fit.diagnostics["completion"]["output_psd"]
assert fit.operator_fit is None
print(
    "Observed-partial optimizer distance:",
    fit.diagnostics["coherent_partial"]["partial_error_bound"],
)
diagnostic = spde.diagnose_constant_source_admissibility(
    fit.kernel,
    evaluation,
    weights=layout.weights,
    propagator=model,
    delta=1 / 52,
    trace_cap=2.0,
    objective_gap_tolerance=1e-8,
)
print("Distance to the declared finite constant-source class:", diagnostic.fit_distance_bounds)
print("Diagnostic:", diagnostic.admissibility_status, "converged:", diagnostic.converged)
print(
    "The source candidate is a witness; missing pairs remain selected by the canonical assumption."
)
