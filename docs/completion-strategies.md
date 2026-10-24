# Shared completion strategies

`spde_covariance.completion_strategy` composes the existing observed-moment,
coherent partial-fitting and canonical-completion routines. It is shared by the
completion ablations and forecast experiments. It accepts no latent covariance,
source label, future path or test score. Existing estimation APIs are unchanged.

The target is a kernel on a declared maturity grid. A completed matrix is not an
orthonormal Sobolev-coordinate covariance, an instantaneous SPDE source
covariance, or evidence that an SPDE can generate that kernel.

## Explicit choices

`CompletionStrategySettings` declares three choices independently:

| Partial estimator | Change to observed corrected moments |
| --- | --- |
| `pairwise` | Retain pairwise empirical means; no penalty or PSD constraint. |
| `ridge` | Replace each observed entry by `c / (c + lambda)` times its mean; no PSD constraint. |
| `coherent` | Solve the shared-overlap, clique-PSD weighted quadratic problem with the existing certified consensus solver. |

Here `c` is the normalized date-weight coverage of that pair, not its raw sample
count. The actual penalty is `relative_regularization * max(c)`. Both orientations
of every off-diagonal pair are counted, consistently with the Frobenius objective.
Missing pairs are not fitted to zero. A declared clique pair with no positive
training coverage makes the strategy unavailable.

The completion choice is `support` or `rank2`. Coherent `support` completion uses
the existing `population=True` numerical PSD/support compatibility checks. The
flag describes the completion checks; it does not assert that empirical input is
a population covariance. Pairwise and ridge support completions use
`population=False`. Every `rank2` completion uses an explicit rank of two at each
nontrivial separator and `population=False`; insufficient positive numerical
support is unavailable, rather than silently capped. Rank truncation can produce
an indefinite completed matrix even after a coherent partial fit. No strategy
silently applies a global PSD projection.

`CompletionStrategyResult.available` means the declared calculation completed and
its required numerical certificate passed. For raw or rank-truncated strategies
it does **not** imply PSD. The nested completion diagnostics retain that separate
property, support compatibility, separator spectra, retained conditions and
inverse amplification. A failed coherent certificate retains the feasible
partial iterate but returns `kernel=None`, without attempting completion.

## Grid floor and its limits

The floor is imposed on each clique in weighted grid coordinates
`A = sqrt(W) K sqrt(W)`. It has kernel-covariance units times the supplied maturity
quadrature units; it is not an eigenvalue floor in orthonormal H coordinates.
Quadrature weights retain their physical scale.

The declared policies are:

| Policy | Weighted-coordinate eigenvalue floor |
| --- | --- |
| `zero` | zero |
| `constant` | `constant_floor`, default `1e-5` |
| `resolution` | `floor_gamma / Q**2`, default `floor_gamma = 1e-5 * 46**2` |

Thus the resolution policy matches the legacy floor at `Q=46`. Its diagonal
baseline has weighted trace `Q * floor = floor_gamma / Q`, which tends to zero
as the number of grid nodes increases. This is the baseline implied by the floor,
not the fitted covariance trace or the actual change in trace. The constant-floor
baseline grows with `Q` instead. Floors apply only to coherent fitting; selecting
a floor policy for the other estimators raises an error instead of ignoring it.

Vanishing floor trace alone proves no continuum convergence. Separator inversion
can amplify errors as grids are refined, and neither a uniform inverse bound nor
a continuum completion-error theorem is supplied by this implementation. The
zero, constant and resolution policies remain explicit sensitivity ablations.

## Certificates and diagnostics

The coherent solver's primal-dual gap and implied weighted observed-partial
distance bound are retained unchanged. A strategy is unavailable unless the
solver passes its declared `numerical_error_budget`; the default is `1e-4`.
The certificate does not bound sampling error, missing entries after completion,
H-operator error, or SPDE-source error. Pairwise/ridge partial fitting has an
entrywise closed-form optimizer and reports zero optimization gap, with ordinary
floating-point arithmetic rather than an interval-arithmetic certificate.

Top-level diagnostics include the actual and relative penalty, floor and its
diagonal trace baseline, weighted and unweighted changes to observed entries,
the coherent gap/distance bound, maximum retained separator inverse norm,
condition number and inverse-perturbation multiplier. Full underlying solver and
completion diagnostics remain available. Empty retained separator spectra have
no condition number; this is reported as unavailable, not replaced by a favorable
number. A cover requiring no separator inversion reports inverse norm zero and
condition one by convention.

All result arrays and nested diagnostics are read-only snapshots.

## Training-only stability and observed validation

`balanced_fragment_split` partitions only the supplied training positions. It
groups dates by their set of observed nodes, including different row permutations
of the same fragment, shuffles within each group using a fixed seed, and divides
each group into complementary halves. Group sizes and total half sizes differ
by at most one. A group represented fewer than twice makes the split unavailable.
No value, latent model or validation/test date determines the partition.

`split_completion_stability` fits the same declared strategy on both halves.
Known fresh independent point-noise variance is corrected separately in each
half; date weights and relative penalties are recomputed using that half's
coverage. It reports weighted Frobenius differences on the observed pairs,
missing pairs and full completed grid. Each relative difference divides by the
larger of the two corresponding weighted-kernel norms; two zero norms give zero
difference. Failed half-fits and their reasons remain visible. This measures
training sensitivity. It is not a sampling confidence bound, a proof of
identification, or a test of the underlying stochastic process.

`select_completion_strategy` accepts a named ordered mapping of already trained
fits and observed validation sufficient statistics. Its score is the existing
noise-corrected observed quadratic score, omitting the common measurement-only
constant. Exactly equal scores retain the first candidate in the supplied
mapping order. The full audit includes unavailable, rejected and eligible
candidates. The caller must declare candidate order and restrict fitting and
splitting to training dates before calling this helper; the helper cannot infer
chronology from sufficient statistics.

Observed validation cannot distinguish completions that preserve the same
observed entries. In particular, changing a separator rank can change missing
cross-blocks, PSD status and exposure risks while leaving this score identical.
The default selection therefore makes no claim to learn a separator rank from
unobserved pairs.

An optional `CompletionStabilityGuard` makes declared training diagnostics an
admissibility condition before score minimization. It can bound the full-grid
relative split difference, separator inverse norm and retained condition number.
With its default `require_available=True`, missing or failed split diagnostics
reject that candidate. A requested threshold cannot pass an unavailable metric.
With `guard=None`, observed score alone selects among numerically available fits,
and absent/unavailable stability remains explicit in the audit. Thresholds are
not tuned against latent truth or held-out test scores. Inverse thresholds depend
on the physical quadrature convention and must be declared accordingly; they are
not scale-free guarantees of completed-kernel accuracy.

## Minimal use

```python
from spde_covariance.completion_strategy import (
    CompletionStrategySettings,
    fit_completion_strategy,
)
from spde_covariance.observation_statistics import partial_kernel_from_arrays

# values and rows contain training observations only; cliques and grid_weights
# describe the fixed maturity grid and its physical quadrature.
statistics = partial_kernel_from_arrays(
    values, rows, noise_variance=known_noise_variance, size=len(grid_weights)
)
fit = fit_completion_strategy(
    statistics,
    cliques=cliques,
    weights=grid_weights,
    settings=CompletionStrategySettings(
        partial_method="coherent",
        relative_regularization=0.001,
        completion_rule="support",
        floor_policy="resolution",
    ),
)
if not fit.available:
    raise RuntimeError(fit.reason)
```

The low-level helper returns an unavailable result for an expected numerical or
coverage failure. Experiment runners should retain that audit and decide whether
their predeclared protocol must stop. Malformed inputs raise an exception.
