# Canonical covariance completion from dense fragments

This implementation studies the established completion construction of
[Waghmare and Panaretos, *The Completion of Covariance Kernels* (2022)](https://arxiv.org/html/2107.07350v3).
It separates the canonical target, its finite-grid approximation and sampling
error. Reproducing an algorithm or a numerical table does not establish a new
estimator, a convergence rate or recovery of every unobserved covariance.

## Target before estimation

Let independent centered curves have covariance evaluation kernel
$K(s,t)$ on an interval $I$. A fragment supported on $I_j$ supplies information
about $K$ only on $I_j^2$. The observed domain is the serrated set

```math
\Omega=\bigcup_{j=1}^{m}(I_j\times I_j).
```

For two overlapping intervals, write $B=I_1\cap I_2$, with exclusive regions
$A=I_1\setminus B$ and $D=I_2\setminus B$. Canonical completion fills the
missing cross covariance by the overlap RKHS inner product:

```math
K^*(s,t)=\langle K(s,\cdot),K(\cdot,t)\rangle_{\mathcal H(K_B)},
\qquad s\in A,\quad t\in D.
```

On a compatible finite PSD grid the corresponding formula is

```math
K^*_{AD}=K_{AB}K_{BB}^{\dagger}K_{BD}.
```

The pseudoinverse acts on overlap support. Repeating the two-region operation
gives the finite-grid canonical completion of an interval cover. The exact
population construction has a Gaussian conditional-independence
interpretation: separated regions are conditionally independent given an
overlap separator. It selects a completion even when the observed covariance
has many valid completions. Unique completion is a stronger condition.
Numerical support thresholds and statistical truncation are separate choices.
The statistical integral-operator construction requires overlaps of positive
measure; a touching-only cover is not a substitute for those assumptions.

For the transport study the input kernel is

```math
K(s,t)=\mathrm{Cov}(\xi(s),\xi(t)),
\qquad \mathrm{Cov}(\xi)=V_\Delta
=\frac{1}{\Delta}\int_0^\Delta S_rCS_r^*dr.
```

Thus this method completes the evaluation kernel of fixed-interval
innovations. It does not estimate instantaneous source $C$, identify a unique
loading $L$ in $C=LL^*$, or reconstruct innovations from noisy sparse curve
levels. An SPDE model does not itself imply the conditional-independence
assumption for a chosen fragment cover.

For any generating kernel, the useful error decomposition is

```math
\widehat K-K=(\widehat K-K^*)+(K^*-K).
```

The first term includes estimation and regularization error; the second is
completion/model bias. A finite-grid $K^*$ need not equal the restriction of
the continuum canonical kernel. In particular, continuum analytic uniqueness
does not turn an ordinary finite collection of matrix entries into a uniquely
completable matrix. Grid refinement is an empirical resolution check here.

## Empirical partial covariance and regularization

With known zero mean and known independent measurement-noise covariance,
the partial covariance averages noise-corrected outer products over dates
that observe each pair. Pair counts are retained. Missing entries are never
encoded as observed zeros. Different pair counts can produce locally
indefinite or mutually incompatible empirical blocks even when the population
kernel is PSD.

At each join, the estimator replaces the overlap inverse by its leading
positive eigenspace. In unweighted matrix notation,

```math
\widehat K_{AD}
=\widehat K_{AB}U_r\Lambda_r^{-1}U_r^T\widehat K_{BD}.
```

Quadrature-weighted eigensystems are used when approximating an integral
operator. With $W_B$ the positive overlap quadrature weights, the overlap
matrix is $W_B^{1/2}\widehat K_{BB}W_B^{1/2}$; both cross factors receive the
matching weights. Uniform unit weights reproduce the ordinary matrix
recurrence used in the author's R code. Uniform rescaling of all weights
cancels in an explicit-rank inverse; nonuniform weights can change its
truncated eigenspace.

Manual rank and FVE are alternative truncation rules. FVE chooses the
smallest retained positive rank whose cumulative eigenvalues exceed the
declared fraction of total positive variance, following the paper's strict
threshold below 100%. This implementation records
rank, overlap spectrum, numerical support and unavailable joins. Unsupported
rank requests fail explicitly. Data-driven selection must use observed
validation data; the generating full kernel is reserved for evaluation.

Raw truncated completion is not guaranteed PSD. Incompatible empirical
blocks can preclude a PSD completion. Discarding a needed overlap direction
can also make the algorithm's output indefinite despite a valid original
population completion. A separately requested weighted PSD projection is reported
as postprocessing; it may change observed entries. Exact population
order-invariance does not justify order-invariance of finite truncated
recursions. The default algorithm follows the declared left-to-right cover.

## Reference kernels and controls

`paper_kernel` implements the three Section 10 kernels on $[0,1]$:

```math
K_1(s,t)=\sum_{j=1}^4 2^{1-j}\phi_j(s)\phi_j(t),
\qquad K_2(s,t)=\min(s,t).
```

```math
K_3(s,t)=10st\exp[-10(s-t)^2].
```

The functions $\phi_1,\ldots,\phi_4$ are the normalized shifted Legendre
polynomials of degrees zero through three. The first kernel has rank four;
the others have infinite rank. `paper_cover` supplies the paper's 2, 3, 5, 9
and 17 interval covers. Every interval has length 0.6 on $[0,1]$, with starts
equally spaced from 0 to 0.4. Covers can be affinely rescaled independently
of kernel evaluation.

The deterministic population controls include every combination of the
three kernels and covers with 3 and 9 intervals. They use the paper's manual
rank four for the finite-rank kernel and rank two for the others. They record
observed-entry preservation, missing-region integrated squared error and raw
minimum eigenvalues. These controls test the numerical recurrence rather
than reproduce the paper's complete Monte Carlo tables. The original tables
also involve sample-size choices, sparse-fragment smoothing and 100
replications; that sparse smoothing pipeline is outside these controls.

Two additional controls make the interpretation testable:

1. A family with common factor 1 and private factors
   $l(t)=\max(1-t/0.4,0)$ and $r(t)=\max((t-0.6)/0.4,0)$ uses
   $K_\rho=1+ll^T+rr^T+\rho(lr^T+rl^T)$, for $|\rho|\le1$.
   Every member has identical covariance on the two observed squares.
   Canonical completion selects $\rho=0$; endpoint-spread variance ranges
   from zero to four across the compatible family. Unobserved risk is not
   identified by these observations alone.
2. In the four-variable PSD control, the overlap contains independent
   directions with variances two and one. Both exterior variables equal the
   second overlap direction. Rank-two inversion recovers their covariance
   one. Rank-one inversion fills zero and produces minimum eigenvalue
   $1-\sqrt{2}$. PSD failure is exposed rather than silently repaired.

Integrated errors use an explicit product quadrature measure. These are
$L^2$ evaluation-kernel errors, not Hilbert-coordinate Sobolev operator errors
from the earlier ridge studies. A grid refinement, a source-forcing cutoff,
a basis dimension and a statistical inverse rank represent different
approximations and must retain separate labels.

## Pinned author-code comparison

The independent cross-language check sources only
[the author's R file at revision 0d332d99583315b60e17b6fd83b6c2e0b74e8da0](https://github.com/kgwstat/covcomp/blob/0d332d99583315b60e17b6fd83b6c2e0b74e8da0/R/covcomp.R).
The source-file SHA256 is
`a38acc2ed2149a0b727f8d8bf1a760e1f7819d7bc0eddccd3dc5117af5596ebe`.
No author implementation is vendored. The exported fixtures contain six
paper kernel/cover cases and both ranks of the PSD control. Python and base
R receive identical observed matrices, masks and explicit truncation ranks.
Finite-rank overlap inversion is ill-conditioned, so the check uses a
declared floating-point comparison tolerance rather than exact equality.

```bash
uv run python scripts/export_completion_reference.py experiments/runs/author-reference
Rscript scripts/check_author_completion.R experiments/runs/author-reference
```

The dedicated Ubuntu workflow installs R and runs this comparison. A Python
fixture export alone is not evidence that the R comparison passed; its
workflow result and recorded maximum differences provide that evidence.

The pinned `ComplnFVE` function computes `trace(M)/nrow(M)` but compares it
with cumulative unscaled eigenvalues of `M`. The paper's FVE expression uses
the trace on the same scale as those eigenvalues. These thresholds differ
by overlap dimension. The implementation follows the consistent weighted
positive-eigenvalue fraction and tests explicit-rank parity only. It does
not claim parity with this legacy FVE behavior. Negative empirical
eigenvalues also require an explicit policy absent from the original code.

## Source audit and limits

The [author's publication page](https://people.math.ethz.ch/~kwaghmare/publications)
links an errata PDF. The linked
[errata URL](https://people.math.ethz.ch/~kwaghmare/assets/paper-1-errata.pdf)
returned HTTP 404 during this implementation audit. Its content has not
been reviewed. The v3 HTML contains differing rate expressions in Theorems
7.1 and 12.14, so this project makes no reproduced-rate claim. The Brownian
population check is verified independently by finite matrix completion;
no uncertain worked RKHS expression is used as numerical code.

The benchmark does not establish that a transported innovation kernel is
canonical, that a finite-grid inverse resolves an infinite-dimensional
inverse, or that observed validation loss controls unseen risk. Results
must show canonical bias separately from estimation error, raw PSD failures
separately from optional repair, and all predeclared cases rather than a
favorable subset. A completed covariance can support subsequent prediction
only under the corresponding modeling and observation assumptions.
