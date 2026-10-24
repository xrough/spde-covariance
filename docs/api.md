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

## Fixed infinite-rank transport control

`LaguerreTransport(dimension, *, cache_max_bytes=16 * 1024**2)` implements the
first `dimension` elements of one orthonormal basis in the weighted Sobolev
space with `tau=1`. Its finite span is shift invariant, so its transport is
exact within that span. Orthogonal projection of a larger curve can still
introduce a hidden-mode term when extracting innovations. The basis family is
fixed across dimensions; `basis_id` also identifies the particular dimension.

It implements the same transport protocol, evaluation and observation helpers
as the other backends. Additional methods are:

```python
model = LaguerreTransport(32)
source = PowerLawSource(exponent=3.0, trace=1.0)
C_M = source.finite_matrix(model.dimension)  # no trace renormalization
V_M = model.innovation_covariance(C_M, 1 / 52)
K_M = model.cumulative_covariance(C_M, 1.0)
check = model.covariance_diagnostics(C_M, 1 / 52)
```

`PowerLawSource.eigenvalues(M)` retains the first `M` weights of the fixed
infinite sequence. `tail_trace(M)`, `innovation_tail_bound(M, delta)` and
`state_mse_tail_bound(M, t)` bound omitted forcing. Finite-projection bounds
are supplied by `projected_tail_bound(J, M, delta)` and
`operational_tail_bound(J, M, delta, t)`. The latter includes the omitted
past-state contribution to residuals extracted from projected curves. A bound
on covariance error and a bound on path mean-square error are different units;
the benchmark keeps them separate.

The [benchmark protocol](infinite-transport.md) specifies the basis, spectrum,
coupling, reference integrations and error budgets. These numerical reference
calculations do not assert statistical recovery of an infinite-rank operator.

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
    solver_tolerance=1e-8, max_iterations=100_000, numerical_error_budget=None,
)
estimate_risk(observations, target, *, weights=None, relative_threshold=1e-10)
```

`fit_covariance` minimizes full PSD, trace-capped ridge by accelerated projected
gradient with restart. `CovarianceEstimate` contains covariance, objective/gap,
covariance-error bound, feasibility, iterations, weights, and convergence status.
A maximum-iteration exit returns `converged=False` without raising automatically.
`fit_covariance_reference` uses optional CVXPY/SCS and evaluates the same
certificate after PSD/trace projection. With `numerical_error_budget=None`, it
retains the single-solve objective-gap criterion. A supplied positive budget
also requires the certified covariance-distance bound to meet that budget.
If necessary, the same independent reference problem is solved up to twice
more, warm-starting SCS and reducing its tolerance by a factor of 100 each time.
The feasible iterate with the smallest evaluated gap is retained; failure to
meet either criterion leaves `converged=False` after the bounded refinements.

`CovarianceEstimate.reference_refinements` records additional reference-solver
attempts, `reference_solver_tolerance` records the tolerance of the retained
iterate, and `iterations` includes work across the attempts. Primary fits keep
the reference fields at their defaults, zero and `None`. RQ2/RQ3 reference
records retain the tolerance, refinement count and evaluated error bound.

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

`rq2.run_rq2_benchmark(config, output=None)` accepts an `rq2_design.RQ2Config`
or a TOML path. It writes only to a fresh or empty output directory:

```sh
spde-covariance benchmark-rq2 experiments/configs/rq2-smoke.toml --output NEW_DIRECTORY
```

The RQ2 suite estimates terminal innovation covariance and exposure risks using
predeclared trailing windows and relative ridge strengths. It performs no
parameter selection. Outputs retain fitted matrices, per-replication metrics,
paired risk-error differences, direct-risk analytic bias/variance, local Gram
diagnostics, numerical certificates and independent references. A failed
numerical check raises `RuntimeError` after saving diagnostics; unexpected
errors and interruptions also save an explicit status. See the
[mathematical problem and experiment protocol](research-question-2.md).

`rq3.run_rq3_benchmark(config, output=None)` accepts an experiment-local
`rq3.RQ3Config` or a TOML path. It also requires a fresh or empty output directory:

```sh
spde-covariance benchmark-rq3 experiments/configs/rq3-smoke.toml --output NEW_DIRECTORY
```

The RQ3 suite reuses the RQ2 innovation simulations. It compares a static
first-128 estimate, a refitted trailing-128 estimate, and a trailing window
selected using strictly earlier validation blocks. Each issue origin produces
one covariance matrix held fixed for the next 16 weekly innovation forecasts.
It does not forecast the covariance of an accumulated 16-week return.

The primary score is the observed, noise-corrected quadratic covariance score.
Exposure-risk scores are available only when the future observation map measures
the exposure jointly. Latent covariance and risk errors are separate simulation
diagnostics and never enter fitting or selection. Monte Carlo uncertainty uses
independent replication-level paired averages, rather than treating forecast
dates as independent replications.

Outputs include issued forecasts and their training boundaries, validation
candidate scores and selected windows, future scores by origin and horizon,
exposure availability, observability diagnostics, numerical certificates,
references, summaries, three figures and source/environment provenance. Failed
or interrupted runs retain explicit status diagnostics. See the
[RQ3 protocol](research-question-3.md). These experiment interfaces do not change
the core covariance estimator API.

`infinite_benchmark.run_infinite_transport_benchmark(config, output=None)`
accepts an `infinite_config.InfiniteTransportConfig` or a TOML path:

```sh
spde-covariance benchmark-infinite-transport experiments/configs/infinite-transport-smoke.toml --output NEW_DIRECTORY
```

It compares population covariances and coupled trajectories across forcing
dimension, projection dimension and observation interval. The small smoke run
tests numerical identities with an intentionally coarse forcing truncation;
its tail-resolution flags must be read separately from software acceptance.
Full runs use the predeclared configuration, preserve independent-path Monte
Carlo uncertainty and record their source/environment manifest. Projected true
innovations and residuals extracted from projected levels are separate targets.

`partial_benchmark.run_partial_benchmark(config, output=None)` accepts an
experiment-local `partial_config.PartialBenchmarkConfig` or a TOML path:

```sh
spde-covariance benchmark-partial experiments/configs/partial-observations-smoke.toml --output NEW_DIRECTORY
```

This study fixes the weekly innovation-covariance target and applies point or
delivery maps to the full forcing-reference innovations before reducing the
estimation dimension. The same observed values and known fresh-noise corrections
are used for every candidate dimension. Unresolved curve contributions and
their covariance cross terms are recorded as model approximation, rather than
subtracted as if they were independent measurement noise.

Candidates use the existing constrained solver. Joint dimension/ridge selection
receives only training and validation observations. Exact score ties prefer
smaller dimensions, then larger relative ridge. The fixed comparator uses the
largest configured candidate dimension and smallest relative ridge. Both fits
remain frozen for a disjoint future test block, without validation refitting.
Relative ridge is scaled by each training design's lifted Gram norm.

Finite-sieve target-range diagnostics, full-H functional-span diagnostics and
availability of a same-date direct risk proxy are separate outputs. Failure of
the per-date proxy check does not by itself establish failure of pooled
identification. Observed validation loss also cannot validate an unobserved
operator direction. See the [protocol](partial-observations.md) for fixed
exposures, source families, chronology, approximation bounds and limitations.

## Kernel-facing covariance workflow

`fit_innovation_covariance` continues to return a certified PSD/trace-constrained
H-coordinate estimate. The additional `fit_covariance_kernel(data, *, layout,
method, method_options, covariance_assumption)` returns `KernelCovarianceFit`:
its `.kernel` is a grid of covariance values, and **has no `.covariance` alias**.
`.nodes`, `.weights`, `.grid_id`, `.estimand` and `.metadata` retain grid measure,
units, innovation intervals and basis provenance. Arrays are immutable snapshots.

`FragmentLayout(nodes, weights, intervals, row_to_nodes, evaluation_matrix,
basis_id)` validates that every observed row is the stated point evaluation on
one dense fragment. It never reinterprets delivery averages as point values.
The original ridge estimator still supports delivery-average observations.

For `method="ridge"`, options pass unchanged to `fit_innovation_covariance`;
`.kernel` is the evaluation of that H-coordinate estimate and `.operator_fit`
retains its numerical certificate. For `method="canonical"`, options are
`rank` or `fve`, `support_rtol`, optional date `weights`, `minimum_pair_count`,
and `psd_projection`. Common covariance and equal innovation intervals are
required. A rank beyond numerical positive support, or FVE on a materially
indefinite separator, is explicitly unavailable. The high-level adapter raises
an informative error; `complete_kernel` returns a result with `.kernel=None`
and diagnostics for experiment reporting. FVE must lie strictly between zero
and one and uses the paper's strict cumulative-trace threshold.

The raw canonical estimate remains `.kernel`, including possible indefiniteness.
An explicitly requested `.projected_kernel` is a separate weighted PSD estimate
with recorded observed-entry and trace changes. Neither is an H-coordinate
matrix or has an H-optimization certificate. See the [completion methods note](canonical-completion.md)
and [runnable SPDE example](../examples/canonical_pipeline.py).

### Compatible PSD partial kernels

The same adapter accepts `method="coherent"`. It jointly fits one shared
observed partial kernel whose declared clique matrices are PSD, then applies
supported canonical completion without hard rank truncation. It requires
`covariance_assumption="common"`, equal innovation intervals, positive
`regularization`, and an explicit `numerical_error_budget`. Accepted options
are:

```python
method_options = {
    "regularization": 1e-3,
    "eigenvalue_floor": 1e-5,
    "numerical_error_budget": 1e-4,
    "rho": 0.2,
    "max_iterations": 20_000,
    "support_rtol": 1e-12,
    # Optional: weights (across dates), minimum_pair_count, psd_projection.
}
```

The shared optimization variable is `A = sqrt(W) @ K @ sqrt(W)` on the
observed pair mask, where `W` contains the layout's quadrature weights.
Pair coverage weights the noise-corrected data fit; ridge and the clique
eigenvalue floor also act in these weighted grid coordinates. The floor is
not a lower bound on eigenvalues of an H-coordinate covariance operator.
The observed partial kernel can change in this fit; it is not an unchanged
empirical partial followed by a PSD repair of the completed matrix.

`fit.diagnostics["coherent_partial"]` retains the feasible primal/dual
objectives, gap, `partial_error_bound`, clique feasibility restoration and
ADMM residuals. The error bound measures weighted Frobenius distance **on
observed pairs to the declared partial optimizer**. It excludes completion
amplification, statistical error, H-operator error and source-covariance error.
`fit.operator_fit` is `None`; no H-coordinate certificate is attached.
Missing entries remain chosen by the canonical conditional-independence
assumption. A PSD completed kernel does not establish that those entries
match the generating kernel.

The adapter raises `CoherentConvergenceError` when the partial fit misses its
requested numerical budget. Its `.fit` retains the feasible partial and
diagnostics. The lower-level interface returns that result directly:

```python
fit_coherent_partial(
    partial_kernel, *, pair_weights, cliques, weights=None,
    regularization=0.0, eigenvalue_floor=0.0, numerical_error_budget=1e-5,
    rho=1.0, max_iterations=20_000, absolute_tolerance=1e-8,
    relative_tolerance=1e-7, check_every=10, adaptive_rho=True,
)
```

`CoherentPartialFit.partial_kernel` has `NaN` on unobserved pairs. Check
`.converged` and `.partial_error_bound` before using it as a successful fit;
small ADMM residuals alone do not meet the optimizer-distance requirement.
`coherent.fit_coherent_partial_reference` supplies an optional CVXPY/SCS
formulation of the same finite problem and checks the same certificate.
See the [method and scope note](coherent-completion.md) and
[runnable coherent example](../examples/coherent_pipeline.py).

### Finite constant-source admissibility diagnostic

This optional diagnostic takes an already fitted grid kernel and a declared
transport model. It leaves the innovation-kernel fitting target unchanged:

```python
diagnose_constant_source_admissibility(
    kernel, evaluation_matrix, *, delta, trace_cap, propagator=None,
    integration_matrix=None, weights=None, quadrature_order=24,
    objective_gap_tolerance=1e-10, max_iterations=20_000,
    initial_source=None, admissibility_tolerance=0.0,
    grid_error_bound=0.0, source_tail_bound=0.0,
)
```

Exactly one of `propagator` or `integration_matrix` is required. The latter
is the linear map for **normalized** covariance integration in the
Frobenius-isometric `svec` coordinates of the declared source basis. Use
`admissibility.covariance_integration_matrix(propagator, delta,
quadrature_order=24)` to prepare and reuse this map. The evaluation matrix
has one row per grid node and one column per **output** coordinate J.
The propagator-only path uses the matched source/output dimension. For a larger
forcing support M, `source_classes.projected_covariance_integration_matrix(M,J,delta)`
returns the rectangular `dJ x dM` map, where `dL=L*(L+1)//2`.
The source witness is M-by-M while the projected innovation is J-by-J.
See [output versus forcing support](source-classes.md) for numerical-map and
fixed-generating-source tail allowances; neither excludes every infinite source.
The kernel may be indefinite for diagnosis. Positive grid weights retain
their supplied scale; the default is unweighted discrete Frobenius distance.

The solver minimizes half the squared weighted distance from the input
kernel to `E @ T_delta(C) @ E.T`, subject to `C >= 0` and
`trace(C) <= trace_cap`, for a source constant over the interval. Its
`AdmissibilityDiagnostic` exposes immutable `source_candidate`,
`projected_innovation`, `projected_innovation_kernel`, source feasibility,
objective/gap, `fit_distance_bounds`, `converged`, `status` and metadata.
The source candidate is a feasible optimization witness, not a claim of
recovering the physical source. The projected kernel is a diagnostic output
and does not replace the fitted input kernel.

At a feasible candidate with objective `f` and global convex gap `g`, the
minimum distance lies in
`[sqrt(2 * max(0, f-g)), sqrt(2*f)]`. This interval remains available at an
iteration limit, when `converged=False`; report both endpoints and the
status. `admissibility_status` compares the distance interval with the
caller-declared tolerance and can be compatible, separated from the
declared finite class, or unresolved. A positive candidate residual alone
does not establish separation; its certified lower endpoint must exceed
the tolerance.

Optional grid/tail bounds must already be in the same weighted kernel norm.
They widen `reference_distance_bounds` for a perturbation of the input
kernel against the **same finite source class**. A fixed generating-source
tail bound does not compare that class with all infinite-dimensional
sources. The convex certificate concerns the finite numerical integration
map in floating point; quadrature/map error and interval-arithmetic rigor
are not supplied automatically. A failed finite cutoff, trace cap or
constant-source restriction therefore cannot exclude a larger SPDE source
class. Invalid inputs or nonfinite calculations raise `ValueError`.

`coherent_benchmark.run_coherent_benchmark(config, output=None)` runs the
configured paired follow-up through the existing partial-benchmark CLI:

```sh
spde-covariance benchmark-partial experiments/configs/coherent-smoke.toml --output NEW_DIRECTORY
```

The study retains fixed and observed-validation-selected partial ridge,
the historical rank-two comparison, completion diagnostics and finite-source
distance intervals. Its certificates and held-out observed scores do not
establish correctness of the completed missing entries. See the
[coherent-completion protocol](coherent-completion.md) for the declared
models, evaluation targets and source-cutoff limitations.

The experimental composition layer in `completion_strategy` provides immutable
`CompletionStrategySettings`, `fit_completion_strategy`, balanced training
splits and explicit guarded validation selection. It does not change the
kernel-facing or H-coordinate estimator APIs. See [strategy conventions](completion-strategies.md).
`CompletionResearchConfig` and `CompletionForecastConfig` are experiment-local
interfaces selected by the TOML `study` marker through the existing
`benchmark-partial` and `benchmark-rq3` commands. They retain unavailable fits
and method-specific diagnostics; they do not convert completed kernels to
H-coordinate covariance matrices.
