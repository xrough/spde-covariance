# Output coordinates and forcing support

The output covariance dimension and the forcing dimension are separate model
choices. Write `J` for the retained orthonormal Laguerre coordinates and `M`
for a finite forcing prefix, with `M >= J`. For a source covariance constant
over one interval, the normalized projected innovation map is

```math
\Phi_{\Delta,J\leftarrow M}(C_M)
=\frac{1}{\Delta}\int_0^\Delta
P_{J\leftarrow M}S_M(r)C_MS_M(r)^T P_{J\leftarrow M}^T\,dr.
```

Here `P` selects the first `J` coordinates after transport has acted in
dimension `M`. Transporting only the leading `J x J` source block generally
gives a different answer: omitted forcing can feed retained coordinates.
The target remains innovation covariance `V_delta`; a candidate source is
an optimization witness rather than an identified physical noise loading.

For grid evaluation `E_J`, positive quadrature matrix `W`, and source cap `R`,
the finite-class distance is

```math
d_{J,M,R}(K)=\min_{C_M\succeq0,\ \mathrm{tr}(C_M)\le R}
\|W^{1/2}(E_J\Phi_{\Delta,J\leftarrow M}(C_M)E_J^T-K)W^{1/2}\|_F.
```

This is a weighted grid-kernel norm. It is distinct from the Frobenius norm
of the orthonormal-coordinate source or innovation covariance. A general
grid kernel may also have a representation residual outside the evaluation
span; that residual is not solely a transport-source obstruction.

## Numerical API

The source-class helpers live in `spde_covariance.source_classes`:

```python
projected_covariance_integration_matrix(
    source_dimension, output_dimension, delta, *, quadrature_order=24,
)
projected_integration_diagnostics(
    source_dimension, output_dimension, delta, *, source_covariance=None,
    quadrature_order=24, evaluation_matrix=None, weights=None,
)
embed_source_covariance(source_covariance, dimension)
evaluate_source_witness(
    source_covariance, integration_matrix, evaluation_matrix, *,
    target_kernel=None, weights=None, trace_cap=None,
)
```

The integration matrix has shape `dJ x dM`, where `dL=L*(L+1)//2`, and
acts on the Frobenius-isometric `svec` coordinates. It uses a direct positive
Gauss rule applied to the covariance lift of `P_J @ S_M(r)`, without
allocating all `M x M` symmetric basis matrices. It returns an immutable
snapshot. The default interval weights sum to one: the result is normalized
`V_delta`, not the unnormalized transition covariance.

Pass the rectangular matrix to the existing diagnostic:

```python
diagnose_constant_source_admissibility(
    K,
    E_J,
    delta=delta,
    trace_cap=R,
    integration_matrix=B_JM,
    weights=grid_weights,
)
```

The source dimension is inferred from the matrix's triangular column count.
`source_candidate` is `M x M`, while `projected_innovation` is `J x J` and
`projected_innovation_kernel` is a grid kernel. The result and metadata retain
both dimensions. The existing propagator-only call keeps its matched-dimension
behavior. For `M > J`, normal equations factor through the smaller output
space, avoiding a dense `dM x dM` Hessian.

The objective-gap certificate has the same meaning for square and
rectangular maps. At a feasible source, if `f` is half the squared residual
and `g` is the global convex gap, the exact optimum of that finite numerical
map has distance in

```math
\left[\sqrt{2\max(0,f-g)},\ \sqrt{2f}\right].
```

An iteration-limit result retains the interval and `converged=False`.
Its upper endpoint is the residual of a feasible candidate, not a claim
that the minimum distance was accurately resolved. Numerical quadrature and
floating-point error remain outside this convex optimization certificate.

## Nested witnesses and monotonicity

For fixed output coordinates, interval, grid and trace cap, the exact finite
source classes are nested as `M` increases. Laguerre prefixes are invariant
under the full finite transport. Consequently zero-padding a source gives

```math
\Phi_{\Delta,J\leftarrow M}(C_m\oplus0)
=\Phi_{\Delta,J\leftarrow m}(C_m),\qquad J\le m\le M.
```

Embedding preserves PSD and trace. Exact minimum distances therefore cannot
increase with forcing support. Independently stopped solver witnesses need
not have monotone residuals. `embed_source_covariance` and
`evaluate_source_witness` let a study re-evaluate an earlier witness under the
larger numerical map and retain a feasible upper envelope. A requested trace
cap is checked before returning a witness distance. Differences between the
particular numerical maps must be recorded if they use different integration
settings.

A concrete control separates the matched and larger forcing classes. Set
`J=2`, `delta=1`, and use a three-dimensional trace-one source

```math
C_3=vv^T,\qquad v=(-1,-1,1)^T/\sqrt{3}.
```

Its projected innovation covariance is approximately

```math
\Phi_{1,2\leftarrow3}(C_3)
=\begin{pmatrix}
0.40823787&0.32752101\\
0.32752101&0.27005344
\end{pmatrix}.
```

The inverse matched two-mode covariance bridge has minimum source eigenvalue
approximately `-0.07891`. With source trace cap two, the matched-class distance
is approximately `0.03931458`, while the declared three-mode source and its
four-mode embedding attain the target to rounding. An independent small
CVXPY formulation verifies this control. It demonstrates that matched finite
source failure can disappear when unresolved forcing is retained, even with
the same output dimension and trace cap.

## Integration comparisons and fixed-source tails

`projected_integration_diagnostics` compares the whole quadrature map at
orders `q` and `2q`, then compares their output against the independent
small-step block-exponential covariance integration for one supplied PSD
source. If no source is supplied, the block check uses the trace-one identity
source. A fixed-source block agreement does not certify all source directions.
The direct positive rule preserves PSD, but does not itself prove accuracy of
the integral at an arbitrary dimension or interval.

For two supplied numerical maps, the helper

```python
integration_map_comparison_allowance(
    first_map, second_map, *, trace_cap, evaluation_matrix=None, weights=None,
)
```

provides a uniform comparison over their common finite PSD trace-capped
source domain. Since `||C||_F <= R`, the weighted image discrepancy is bounded
by `R` times the spectral norm of the weighted map difference. This also
bounds how far their finite image sets can differ. It compares the two
computed maps; agreement of two approximations is not a certified error bound
against the exact integral.

For the fixed `PowerLawSource`, use

```python
fixed_source_tail_allowance(
    source, source_dimension, output_dimension, delta, *,
    evaluation_matrix=None, weights=None, rotation_angle=0.0,
)
```

This wraps the analytic Laguerre tail bounds in the requested grid metric.
With `a=(1-exp(-delta))/delta` and the first omitted source eigenvalue `q_M`,

```math
0\preceq P_J V_{\Delta,\mathrm{tail}}P_J^*
\preceq q_M\,\mathrm{diag}(2-a,a,\ldots,a).
```

The helper evaluates that PSD upper matrix through `sqrt(W) @ E_J` and returns
a weighted kernel-norm bound, separately from the full-operator trace-tail
bound. Adjacent-pair source rotations preserve this allowance at even forcing
cutoffs; an odd rotated cutoff is rejected. These bounds concern one fixed
source spectrum and declared rotation, evaluated in floating point.

A fixed generating-source tail allowance can widen distance intervals for a
perturbation of the target against the same finite source class. It does not
uniformly approximate every trace-capped infinite source: such a class may
place its forcing mass beyond any finite cutoff. Finite source-cone failure,
even with the fixed-source tail reported, does not establish exclusion of an
infinite-dimensional SPDE or a time-varying source. No sampling-calibrated
rejection test or completed-source recovery guarantee is supplied.

## Population source-support study

`completion_source_study.run_source_class_study(config)` returns population
rows and a controls mapping with `passed` and `rows` fields. It writes no
files and simulates no observations. The output grid is fixed as
`grid_nodes(41, "spde")`; each generating target evaluates the first `J`
coordinates of one configured forcing-reference covariance. For each
configured three- or nine-fragment cover, the canonical target completes
that same generating partial. Unavailable population completions remain
explicit rows with a reason rather than silently disappearing.

At fixed `J`, candidates use `M=J` times the predeclared source multipliers.
The first source initialization is zero; subsequent initializations embed
only earlier fitted witnesses. Generating source matrices never initialize
these fits. Rows retain the original convex distance interval, a nested
feasible upper witness, representation diagnostics, map-refinement comparison,
fixed-source tail allowances and both source/output basis identities. The
generating-source tail is reported at both candidate and reference cutoffs;
it is not automatically applied to a canonical target.

An empty `source_dimensions` configuration skips the large population study
while retaining the deterministic matched-versus-larger-support control.
Population fits are numerical model diagnostics rather than Monte Carlo
evidence, and forcing-support nesting comparisons require fixed output
coordinates and a fixed target kernel.
