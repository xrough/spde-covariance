# PSD-consistent partial covariance and SPDE admissibility

This increment addresses two different questions exposed by the
[dense-fragment study](canonical-completion.md): whether empirical partial
covariances admit a PSD completion, and whether a PSD innovation covariance
belongs to a declared SPDE source model. Neither question is answered by a
small observed prediction error alone. The methods below use established
finite convex optimization and covariance-completion algebra; no new
statistical estimator theorem or infinite-dimensional recovery guarantee is
claimed.

## One shared partial covariance

For independent centered innovations, pairwise averages of noise-corrected
outer products estimate the kernel on the observed domain $\Omega$. Unequal
pair counts can make these estimates indefinite even on an observed clique.
Projecting each clique separately creates inconsistent overlap entries;
averaging those entries afterward can destroy PSD again.

Let $T$ be the estimated partial kernel, $W$ the positive maturity-quadrature
weights and $c_{ab}$ the declared normalized pair-coverage weights. Work in
weighted grid coordinates:

```math
\widetilde T=W^{1/2}TW^{1/2}.
```

Estimate one symmetric variable $A$ on $\Omega$, shared by every clique:

```math
\min_A\ \frac12\sum_{(a,b)\in\Omega}
c_{ab}(A_{ab}-\widetilde T_{ab})^2
+\frac{\lambda}{2}\sum_{(a,b)\in\Omega}A_{ab}^2.
```

```math
A_{I_j,I_j}\succeq\tau I\quad\text{for every observed clique }I_j.
```

The sums use the symmetric Frobenius convention: each off-diagonal entry
appears twice. Coverage and quadrature are distinct weights. Coverage records
how often a pair was observed; quadrature defines the evaluation-kernel
$L^2$ metric. Neither is the weighted-Sobolev coordinate metric used by the
transport model. The fitted partial kernel is $W^{-1/2}AW^{-1/2}$ on
$\Omega$; unobserved entries remain missing until completion.

The constraints fit the observed values rather than preserve incompatible
empirical entries exactly. All overlapping cliques refer to the same
variable, so overlap equality is structural. A nonzero $\lambda$ shrinks
the partial kernel. A positive $\tau$ imposes a lower eigenvalue bound on
each weighted clique and changes the estimation problem. It must be declared
and reported; a fixed floor need not have the same physical meaning across
different grids.

For an ordered interval cover satisfying the running-intersection property,
compatible PSD clique blocks have a PSD matrix completion. The next operation
is the canonical completion using every supported overlap direction. An
additional hard rank truncation would define a different estimator and can
destroy PSD even when the fitted partial kernel is perfectly coherent.
Numerical support thresholds still require explicit conditioning and output
checks. Positive physical overlap remains necessary for the intended
integral-operator interpretation.

## Consensus optimization and a feasible returned result

The formulas above and below use the historical zero shrinkage reference.
The optional `reference_kernel` instead supplies a finite-grid covariance
target R. In weighted coordinates, replace the ridge term by
$\lambda\sum_\Omega(A-\widetilde R)^2/2$. The shared-update right-hand side
becomes $c\widetilde T+\lambda\widetilde R$; the dual constant also gains
$\lambda\sum_\Omega\widetilde R^2/2$. The same primal-dual certificate and
strong-convexity argument apply. The default is unchanged. The
[statistical completion study](statistical-completion.md) uses a common
training-only reference for its moment and likelihood fits, with explicitly
different Frobenius and KL penalties.

The solver separates the shared variable from local PSD clique copies and
uses consensus ADMM, a standard splitting method described by
[Boyd, Parikh, Chu, Peleato and Eckstein (2011)](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf).
The shared update is entrywise quadratic; local updates project clique
eigenvalues onto the declared floor. A small consensus residual alone does
not make the returned shared matrix PSD on every clique.

Before evaluating a final result, the solver checks the actual shared clique
eigenvalues. A reported uniform diagonal loading makes the shared variable
feasible without independently editing overlaps:

```math
\delta=\max\{0,\tau-\min_j\lambda_{\min}(A_{I_j,I_j})\}.
```

The implementation allows a rounding margin and records the actual loading.
Adding the same $\delta$ to every global observed diagonal preserves
consensus; each clique receives $\delta I$. The objective and gap are
evaluated on the returned feasible partial covariance, including this change.
This operation is not a global PSD projection of a completed kernel.

The returned fit uses the feasible primal--dual optimizer-distance budget
as its convergence criterion. Primal and dual residuals and their tolerance
flags remain separate diagnostics. To state the certificate, write
$H_{ab}=c_{ab}+\lambda>0$ on $\Omega$, let $P_j$ restrict to a clique and
let $Q_j\succeq0$ be its dual matrix. With $q=\sum_jP_j^*Q_j$, a dual
lower bound is

```math
D(Q)=\frac12\sum_\Omega c_{ab}\widetilde T_{ab}^2
-\frac12\sum_\Omega
\frac{(c_{ab}\widetilde T_{ab}+q_{ab})^2}{H_{ab}}
+\tau\sum_j\mathrm{tr}(Q_j).
```

The sign convention uses the Lagrangian
$F(A)-\sum_j\langle Q_j,P_jA-\tau I\rangle$.
The implementation evaluates an equivalent nonnegative square-plus-
complementarity expression to reduce cancellation. If $g=F(A)-D(Q)$ and
$\mu=\min_\Omega H_{ab}$, strong convexity gives

```math
\|A-A_{\mathrm{opt}}\|_{F,\Omega}
\le\sqrt{\frac{2g}{\mu}}.
```

This is the weighted observed-partial optimizer distance, evaluated in
floating-point arithmetic. It is not an interval-arithmetic proof, a sampling
confidence bound, a completed-kernel error bound or a source-covariance
identification result. Completion may amplify a small partial perturbation
through an ill-conditioned overlap inverse. A separately declared eigenvalue
floor can control this mechanism only within its stated finite-grid model.

## Controlled failures that the implementation must distinguish

For three grid variables, observe cliques $(1,2)$ and $(2,3)$ with target

```math
T_\Omega=
\begin{pmatrix}1&2&?\\2&1&0\\?&0&4\end{pmatrix}.
```

With unit quadrature and coverage, zero ridge and zero floor, the coherent
optimum has $T_{11}=T_{12}=T_{22}=1.5$, $T_{23}=0$, and $T_{33}=4$.
Its objective is 0.5 and canonical completion sets $T_{13}=0$. Independent
clique PSD projections instead propose shared diagonal values 1.5 and 1;
averaging them to 1.25 leaves the first clique indefinite. This is why an
actual consensus constraint is needed.

The earlier four-variable truncation control supplies a second check. Its
partial covariance is already PSD and compatible. The two overlap
directions have variances two and one; both exterior variables equal the
second direction. Exact-support completion recovers their covariance one.
Keeping only the first overlap direction fills zero and produces minimum
eigenvalue $1-\sqrt{2}$. Coherent partial estimation does not make arbitrary
subsequent rank truncation safe.

## PSD innovation covariance is not the same as a PSD source

For a known finite transport matrix $M_r$ and a source covariance constant
over an interval, the normalized innovation map is

```math
\mathcal T_\Delta(C)
=\frac{1}{\Delta}\int_0^\Delta M_rCM_r^Tdr,
\qquad C\succeq0.
```

Its image is a cone inside the PSD innovation matrices. A generic PSD
estimate of $V_\Delta$ is not automatically in that cone. Kernel PSD
completion addresses covariance feasibility; source-cone feasibility adds a
specific generator, coordinates, interval length and constant-source model.

The numerical diagnostic also declares a source trace cap $R$. Given an
evaluation map $E$ in the same orthonormal source coordinates, it minimizes

```math
d_R(K)=\min_{C\succeq0,\ \mathrm{tr}(C)\le R}
\|W^{1/2}(E\mathcal T_\Delta(C)E^T-K)W^{1/2}\|_F.
```

The solver minimizes one half of the squared displayed residual, denoted
$f(C)$. At a feasible candidate, its gradient gives the global convex objective gap

```math
g_C=\langle\nabla f(C),C\rangle
-R\min\{\lambda_{\min}(\nabla f(C)),0\}.
```

Hence the minimum distance lies between
$\sqrt{2\max(0,f(C)-g_C)}$ and $\sqrt{2f(C)}$.
These bounds can remain informative at an iteration limit; failure to meet
the requested objective-gap tolerance remains explicit. A candidate source
is an optimization witness, not an estimate proven to recover the physical
source. The kernel fit is not replaced by this diagnostic projection.
For an empirical input, separation describes the fitted kernel. Sampling and
representation error can produce a positive distance even when its generating
process belongs to the declared class; this is not a statistical rejection test.

An exact two-dimensional transport example uses the orthonormal first
integrated-Laguerre functions $e_0(x)=1$ and $e_1(x)=1-e^{-x}$. Their shift is

```math
M_r=\begin{pmatrix}1&1-e^{-r}\\0&e^{-r}\end{pmatrix}.
```

Put $a=(1-e^{-\Delta})/\Delta$ and
$b=(1-e^{-2\Delta})/(2\Delta)$. For the PSD innovation target
$V=\mathrm{diag}(0,1)$, the inverse symmetric source matrix is

```math
C=\begin{pmatrix}
1+b^{-1}-2a^{-1}&a^{-1}-b^{-1}\\
a^{-1}-b^{-1}&b^{-1}
\end{pmatrix}.
```

At $\Delta=1$, its minimum eigenvalue is approximately $-0.074744$.
Even the positive-definite target $V+0.01I$ has an indefinite inverse source,
with minimum eigenvalue approximately $-0.066351$. Thus this distinction is
not solely a boundary-rank issue. The zero-variance direction also gives a
direct explanation: a rank-one innovation must contain all shifted source
directions in one line, whereas the line spanned by $e_1$ is not shift
invariant.

This is a diagnostic of the declared constant-source finite model. With an
arbitrary deterministic time-varying source and invertible $M_r$, choosing
$C_r=M_r^{-1}VM_r^{-T}$ realizes any PSD $V$ in that finite model. The
constant-source restriction is therefore essential to the counterexample.
Even when $C$ is identified, a loading $L$ satisfying $C=LL^T$ remains
nonunique.

## Truncation and interpretation limits

A grid kernel needs an explicit basis evaluation map and metric-aware
coordinate convention before an SPDE-source diagnostic can be applied. A
quadrature-weighted matrix is not an orthonormal Hilbert covariance matrix.
If a fitted grid kernel is not exactly represented by the declared basis,
the representation residual must be reported separately from source-cone
distance.

Let $J$ denote the output projection and $M$ the forcing cutoff. In general,

```math
P_JS_rC S_r^*P_J
\ne P_JS_rP_JC P_JS_r^*P_J.
```

Unresolved forcing can feed retained coordinates through
$P_JS_r(I-P_J)$. Failure of a matched finite-$J$ source cone consequently
does not exclude an infinite-dimensional SPDE source. A larger forcing cone
and a verified tail allowance define different feasibility questions. A
small source tail does not automatically control a reconstructed source or
a canonical completion after an ill-conditioned inverse.

If a supplied grid/tail error bounds a perturbation of the input kernel by
$\varepsilon$ in the same weighted grid norm, distance to the same declared
finite class changes by at most $\varepsilon$. The diagnostic widens its
distance interval accordingly. This does not establish closeness between a
finite source class and an infinite source class: that would require a
separate uniform class-approximation bound. Integration error and floating
point error are not automatically certified by caller-supplied tail values.

Any experimental comparison must keep the observed training/validation/test
data fixed, report both observed-partial modification and completed-kernel
error, retain raw incompatibility and conditioning diagnostics, and separate
finite numerical certificates from statistical or infinite-model claims.
The completion target remains the conditional-independence completion of
[Waghmare and Panaretos (2022)](https://arxiv.org/html/2107.07350v3), unless
additional source-model restrictions are imposed explicitly.

## Configured follow-up

The focused paired study uses base grid 41, 20 replications, source cutoff 512,
training counts 128/512/2048, three/nine fragments and noise variances zero and
0.001. The larger historical grid and paper-control studies remain separate.
The full run allows 100,000 ADMM iterations while retaining a $10^{-4}$
observed-partial distance budget. An independent retry of a difficult noiseless
nine-fragment case required 25,820 iterations; ordinary residual tolerances
alone would have accepted it too early under the smaller iteration allowance.

Four spawned workers each prepare one bounded model cache. All random streams
depend on replication indices, and aggregation sorts those indices rather than
worker completion order. Per-replication checkpoints retain the parent manifest
hash and are not silently reused across revisions. The smoke study uses one
worker and a $10^{-3}$ partial budget against its much larger sampling error.
