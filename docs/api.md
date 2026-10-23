# Stage 1 API

The primary target is normalized transport-innovation covariance `V_delta`,
in a declared orthonormal Hilbert basis. It differs from instantaneous diffusion
covariance `C`, observed covariance `O @ V_delta @ O.T`, and an exposure variance.
Matching dimensions alone does not establish matching bases. Interfaces are evolving.

## End-to-end example

```python
import numpy as np

from spde_covariance import (
    ExponentialTransport,
    extract_innovations,
    fit_innovation_covariance,
    observe_innovations,
    simulate_transport,
)

model = ExponentialTransport()
times = np.arange(257) / 52
diffusion = np.diag([0.4, 0.3, 0.2, 0.1])
simulation = simulate_transport(model, times, diffusion, seed=1729)
innovations = extract_innovations(simulation.trajectory, model)
observed = observe_innovations(innovations)  # O=I; no observation noise
fitted = fit_innovation_covariance(
    observed,
    regularization=1e-3,
    trace_cap=2.0,
    covariance_assumption="common",
    numerical_error_budget=1e-4,
    expected_basis_id=model.basis_id,
)
print(fitted.target, fitted.covariance)
```

Simulation truth remains on `simulation`; fitting does not receive it. See the
runnable [transport example](../examples/transport_pipeline.py) and
[workflow conventions](workflow.md).

## Transport and curve coordinates

The runtime-checkable `transport.TransportPropagator` protocol requires
`dimension`, `basis_id`, `kind`, and these methods:

```python
matrix(dt)
innovation_covariance(source_covariance, dt, quadrature_order=24)
```

`matrix` accepts finite `dt >= 0`. The covariance method requires `dt > 0` and
a real PSD `J x J` source covariance in the same orthonormal coordinates. It
returns `V_delta = integral S(r) C S(r).T dr / delta`, not the unnormalized
transition covariance or `C` itself.

`ExponentialTransport(rates=(0.6, 1.0, 2.0, 4.0), tau=1.0)` implements an exact
shift-invariant exponential span on the nonnegative maturity axis. Rates must be
distinct and greater than `1/(2*tau)`. Numerically ill-conditioned Gram matrices
are rejected. Its `kind` is `"exact_transport"`.

The model also supplies `evaluate(x)`, `derivative(x)`, `raw_gram`, `transform`,
and `breakpoints` for observation maps. Evaluation returns one row per maturity;
a scalar produces one row. The metric is
`f(0)g(0) + integral exp(x/tau) f'(x)g'(x) dx`. The transport matrix is generally
nonnormal after orthonormalization.

`spaces.SplineSpace(n_basis=8, length=6.0, tau=1.0)` supplies cubic splines with
a constant tail beyond `length`. In addition to evaluation/coordinate helpers:

```python
space.project(function, derivative=None, *, quadrature_order=16, breakpoints=None)
space.quadrature(*, order=16, breakpoints=None, weighted=False)
space.shift_matrix(dt, *, quadrature_order=16)
```

`project` uses an analytic derivative when supplied, otherwise numerical
differentiation. Extra breakpoints specify derivative knots or shifted boundaries.
`quadrature` returns nodes and weights; `weighted=True` includes the Hilbert
derivative weight. `ProjectedSplineTransport(space)` adapts this space to the
protocol. Its `kind` is `"projected_mild_recursion"`: `P S(dt) P` generally fails
the exact semigroup identity. Rebuild the adapter after changing the underlying
space. Spatial quadrature, repeated projection, and boundary truncation remain
separate approximations.

## Trajectories, innovations, and observations

```python
simulate_transport(
    propagator, times, source_covariance, *, initial=None,
    integrated_drift=None, seed=0, curve_unit="arbitrary curve unit",
)
extract_innovations(trajectory, propagator, *, integrated_drift=None)
observe_innovations(batch, operators=None, *, noise_covariances=None, seed=0)
interval_diagnostics(data)
```

`simulate_transport` returns `trajectories.TransportSimulation`, containing a
`CurveTrajectory` and separate source/innovation covariance truths. `times` has
`N+1` strictly increasing endpoints in years; coefficients have shape `(N+1,J)`.
Source covariance is `(J,J)` or `(N,J,J)`, deterministic and constant within
each interval. Exponential trajectories use the exact finite-span transport
model; spline trajectories use the declared numerical recursion. No measurement
noise is added to coefficient trajectories.

Construct a `CurveTrajectory` from `times`, `coefficients`, `basis_id`, and
`propagation_method`, with optional `time_unit="year"`, `curve_unit`, and metadata.
Extraction checks basis identity and computes

```text
X[i] = (f[i+1] - S(delta[i]) @ f[i] - integrated_drift[i]) / sqrt(delta[i]).
```

Supplied `integrated_drift` has shape `(N,J)` and is the already integrated,
transported drift contribution. Omitting it declares zero mild drift. The
`InnovationBatch` preserves endpoints, interval lengths, basis, propagation
method, and units. Arrays and metadata are snapshots. Irregular intervals remain
interval-specific; normalization does not make their covariances equal.

`observe_innovations` returns an `ObservedInnovationBatch`, preserving provenance
without latent coefficient values. A common operator or one per interval is
accepted; row counts may vary. Identity is the default. Optional noise is fresh
independent Gaussian innovation noise with known covariance, added after
extraction. It does not model differencing noisy price levels. The lower-level
`InnovationBatch.observe` returns an `Observation` tuple without the batch wrapper.

`observations.Observation(time, y, operator, noise_covariance)` validates one
measurement. Its `.measurement` is `outer(y,y) - noise_covariance` and can be
indefinite. `point_observation_map(space, maturities)` and
`delivery_average_map(space, intervals, *, quadrature_order=16)` return
orthonormal-coordinate observation rows. Delivery intervals have shape `(m,2)`
and use maturity years, without exchange-calendar parsing.

## Covariance fitting and diagnostics

```python
fit_innovation_covariance(
    data, *, regularization, trace_cap, covariance_assumption,
    numerical_error_budget, weights=None, trailing_window=None, targets=None,
    rank_rtol=1e-10, expected_basis_id=None, max_iterations=20_000,
    absolute_tolerance=1e-9, relative_tolerance=1e-8, initial=None,
)
```

This high-level workflow accepts `InnovationBatch` or `ObservedInnovationBatch`.
It requires equal intervals, positive ridge, a trace cap, a positive numerical
budget, and explicit `covariance_assumption="common"` or `"local"`. These are
caller-declared assumptions. A local fit is a weighted common-covariance
approximation over its window. Irregular pooling is rejected; inspect
`interval_diagnostics` instead.

`weights` and `trailing_window` are mutually exclusive. Here `trailing_window`
is a calendar width in years. `InnovationCovarianceFit` exposes `covariance`,
the low-level `estimate`, design `diagnostics`, target/assumption labels, basis,
units, and metadata. Failure raises `ConvergenceError`; its `.estimate` retains
the feasible iterate and certificate.

The low-level `estimation` interfaces accept observation sequences under
separately justified target assumptions:

```python
fit_covariance(
    observations, *, regularization, trace_cap, weights=None,
    max_iterations=20_000, absolute_tolerance=1e-9, relative_tolerance=1e-8,
    numerical_error_budget=None, initial=None,
)
fit_covariance_reference(
    observations, *, regularization, trace_cap, weights=None,
    solver_tolerance=1e-8, max_iterations=100_000,
)
estimate_risk(observations, target, *, weights=None, relative_threshold=1e-10)
```

`fit_covariance` minimizes full PSD, trace-capped ridge by accelerated projected
gradient with restart. `CovarianceEstimate` contains covariance, objective/gap,
covariance-error bound, feasibility, iterations, weights, and convergence status.
A maximum-iteration exit returns `converged=False` without raising automatically.
`fit_covariance_reference` uses optional CVXPY/SCS and evaluates the same certificate.

Certificates bound numerical distance to the constrained optimum, not statistical
error to truth. They are floating-point evaluations of convex bounds. A finite
sieve does not impose a fixed true factor rank. `estimate_risk` requires a
numerically observable target under the constant-covariance deterministic-design
model; its scalar estimate can be negative, and clipping alters its unbiasedness.

In `geometry`, `svec`/`smat` preserve the Frobenius metric with sqrt(2)-weighted
off-diagonal entries:

```python
gram_operator(operators, weights=None, *, cache_max_bytes=32 * 1024 * 1024)
diagnose_design(
    operators, weights=None, targets=None, regularization=1e-3, rank_rtol=1e-10,
)
```

Geometry preserves supplied weights; default weights sum to one. Estimators
normalize weights across dates. The Gram cache is a local byte-bounded LRU of
normal contributions; zero disables it. The budget counts array/key payloads,
excluding Python container overhead and temporary arrays. Diagnostics report
spectrum, numerical rank threshold, target nullspace residuals, and regularized
observability. Numerical rank is not an exact-rank or infinite-dimensional theorem.

## Baselines, selection, and refitting

The `baselines` module exposes:

```python
empirical_covariance(observations, weights=None)
linear_shrinkage_covariance(observations, shrinkage, weights=None)
projected_ridge_covariance(observations, *, regularization, trace_cap, weights=None)
validation_score(covariance, observations)
select_covariance(
    observations, validation, *, method, relative_candidates, window_candidates,
    trace_cap, numerical_error_budget=1e-4, solver_options=None,
)
refit_selection(
    observations, selection, *, trace_cap, numerical_error_budget=1e-4,
    solver_options=None,
)
```

Empirical and linear shrinkage require `O=I` at every training date. Shrinkage
lies in `[0,1]` and targets the empirical matrix's scaled identity. Both keep
native matrices, including possible indefiniteness after noise correction;
neither is trace capped. Postprojected ridge solves unconstrained symmetric
ridge then Euclidean-projects to PSD/trace, generally differing from the
constrained observation-space optimizer.

`validation_score` averages `||O B O.T||_F^2 - 2 <Z,O B O.T>`. The omitted
outer-product norm is candidate independent. Selection methods are
`"constrained_ridge"`, `"projected_ridge"`, `"empirical"`, and `"linear_shrinkage"`.
The caller supplies intended order and independent observed validation data;
there is no truth/model argument.

`window_candidates` are trailing observation **counts**, unlike the workflow's
calendar-width `trailing_window`. Ridge uses `lambda = relative * ||Q||op`,
recalculated per training window. For shrinkage, `relative_candidates` supplies
alphas instead; empirical selection has windows only and accepts
`relative_candidates=None`.

`Selection` contains `covariance`, `method`, `selected_parameters`,
`validation_score`, all `candidates`, and `estimate`. Only constrained ridge
has a `CovarianceEstimate` in `.estimate`; closed forms use `None`. Parameters
record `window` and either `shrinkage` or `relative_regularization`,
`gram_operator_norm`, and actual `regularization`. Failed candidates are audited
and skipped; all-failed selection raises `RuntimeError`. Ties retain the first
successful candidate in input order.

Refitting retains the chosen count/relative strength but recalculates the Gram
norm and actual ridge. The returned selection keeps the original validation
score/candidate audit; these do not score its refitted covariance. Solver options
are `max_iterations`, `absolute_tolerance`, `relative_tolerance`, and `initial`.

## Configured benchmarks

`benchmarks.run_benchmark(config, output=None)` accepts `BenchmarkConfig` or a
TOML path and rejects an existing output directory. The public CLI runs the
paired covariance comparison:

```sh
spde-covariance benchmark experiments/configs/stage1-smoke.toml --output NEW_DIRECTORY
```

Stage 1 outputs include configuration, source/environment provenance, replicate
metrics, tuning audit, comparisons, and stress diagnostics. See the
[checked-in report](../experiments/results/stage1/report.md). The
`simulate_innovations` and `integrate_innovation_covariance` support independent
innovation simulation and the deterministic-source SPDE bridge. These interfaces
do not provide real-market curve calibration or prediction.
