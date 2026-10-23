# Fixed infinite-rank transport: foundation benchmark

This benchmark introduces one fixed infinite-rank transport model and studies
its finite numerical representations. It is a foundation and reproduction
exercise using established stochastic-convolution and covariance mathematics,
not a new estimator theorem or a claim that one large finite simulation equals
an infinite-dimensional path.

The source covariance, finite-interval innovation covariance, and state
covariance are different objects. Forcing rank, curve projection dimension and
sampling interval are varied separately. Population calculations precede
Monte Carlo estimation, so numerical representation errors are not confused
with statistical errors.

## One fixed model

Time and remaining maturity are in years. The state space has inner product

```math
\langle f,g\rangle_H
=f(0)g(0)+\int_0^\infty e^x f'(x)g'(x)dx,
\qquad
(S_t f)(x)=f(x+t).
```

Use the ordinary Laguerre polynomials $L_k$ and the orthonormal basis

```math
\begin{aligned}
e_0(x)&=1,\\
e_j(x)&=\int_0^x e^{-u}L_{j-1}(u)du,\qquad j\geq1.
\end{aligned}
```

The anchor and derivative coordinates give the orthonormality: the classical
Laguerre inner product is $\int_0^\infty e^{-u}L_k(u)L_l(u)du$.
The model uses zero initial state, zero additional drift and constant source
covariance

```math
\begin{aligned}
C&=\sum_{k=0}^\infty q_k(e_k\otimes e_k),\\
q_k&=\frac{(k+1)^{-3}}{\zeta(3)},\\
df_t&=(\partial_x f_t)dt+C^{1/2}dW_t.
\end{aligned}
```

Here $W$ is cylindrical Brownian noise in the declared basis, with infinitely
many independent channels. Every $q_k$ is positive and their sum is one:
$C$ is positive, trace-class and infinite rank. The weights are source
eigenvalues because the basis is orthonormal. No additional spatial mollifier
is introduced. In this basis, exponent greater than one gives a trace-class
source; exponent greater than two also makes
$\sum_k q_k\|\partial_x e_k\|_H^2$ finite. The declared exponent is three.

Keeping modes 0 through $M-1$ defines $C_M$. Its weights retain their original
values: the truncated covariance is **not renormalized**. Otherwise each
resolution would modify the already retained forcing.

## Three covariance targets

For the infinite model, define

```math
\begin{aligned}
K_t&=\mathrm{Cov}(f_t)
=\int_0^t S_rCS_r^*dr,\\
r_i&=f_{t_i}-S_\Delta f_{t_{i-1}},\\
\xi_i&=\frac{r_i}{\sqrt{\Delta}},\\
V_\Delta&=\mathrm{Cov}(\xi_i)
=\frac{1}{\Delta}\int_0^\Delta S_rCS_r^*dr.
\end{aligned}
```

$C$ is instantaneous forcing covariance. $V_\Delta$ is normalized
finite-interval shock covariance. $K_t$ is covariance of the state started
from zero. In this particular constant-source, zero-initial-state model,
$K_\Delta=\Delta V_\Delta$, but $K_t$ is not generally $tC$.

The benchmark computes their finite-forcing counterparts using $C_M$.
Exact finite-dimensional transport in an invariant span removes Euler
time-stepping error; it does not remove forcing truncation or finite-interval
transport effects. A displayed evaluated kernel is the covariance of curve
evaluations, rather than automatically a Lebesgue integral kernel for an
operator defined with the Sobolev inner product.

The population covariance-projection diagnostic at fixed forcing is
$\|V_{\Delta,M}-P_JV_{\Delta,M}P_J\|_{\mathrm{HS}}$, with the projected
matrix extended by zero to the $M$-dimensional span. It is different from
terminal path projection error, forcing truncation error and leakage.

## Independent resolutions and temporal coupling

$M$ is the number of retained forcing modes. $J$ is the number of coordinates
kept by the orthogonal projection $P_J$. They are not the same control.
$\Delta$ is the observation interval. The state horizon and exposure
functionals are also declared before computing errors.
Configured Monte Carlo cases retain $J\leq M$. At a fixed forcing reference,
every configured projection is compared; changing $J$ does not change the
already simulated forcing model.

The Laguerre spans including the constant anchor are shift invariant.
Consequently their finite transport matrices and integrated Gaussian
innovation covariances define an exact finite-forcing recursion. A simulation
on the finest configured time grid is sampled at coarser endpoints. Coarse
innovations aggregate the same fine Gaussian shocks through the semigroup;
the covariance aggregation identity is checked independently.

This couples temporal resolutions within a declared finite-forcing reference.
A common random seed alone would not establish a shared Brownian-forcing
coupling between arbitrary different-dimensional simulations.

Use a fixed underlying exposure when comparing $J$. For example, the unit
$H$ spread between maturities $x<y$ has coefficient sequence

```math
a_k=\frac{e_k(x)-e_k(y)}{\sqrt{e^{-x}-e^{-y}}}.
```

Its first $J$ coefficients are retained without a new $J$-dependent
normalization. Projected-exposure errors and errors for the full infinite
exposure are distinct diagnostics.

## Projection leakage: levels versus innovations

Projecting a true adjusted innovation gives $P_Jr_i$. Reconstructing an
adjusted residual from projected states instead gives

```math
\begin{aligned}
r_{i,J}
&=P_Jf_{t_i}-P_JS_\Delta P_Jf_{t_{i-1}}\\
&=P_Jr_i+B_Jf_{t_{i-1}},\\
B_J&=P_JS_\Delta(I-P_J).
\end{aligned}
```

The second term is unresolved-to-resolved leakage. Invariance of the
resolved span does not imply that it vanishes: invariance restricts transport
out of the resolved span, whereas this term concerns transport into it.

With deterministic additive forcing, current innovations are independent of
the previous state. Since the model is centered, the population residual
covariance is

```math
R_J(t,\Delta)
=P_JV_\Delta P_J
+\frac{1}{\Delta}B_JK_tB_J^*.
```

Conditionally on the previous state, the reconstructed residual has a
generally nonzero mean. Its unconditional covariance can also change with
$t$, despite the constant source covariance. A pooled sample of reconstructed
residuals must therefore be compared with its declared time-average population
second moment, rather than silently labelled an independent common-$V$
sample. The benchmark keeps projected true innovations, reconstructed
residuals and their population targets separate.

For a predeclared exposure, leakage contributes the quadratic form of the
second term above. Risk error against an appropriate population target is
reported separately from error against source risk or a full-exposure target.

## Remainder bounds and their scope

The first omitted source eigenvalue and total omitted trace are

```math
q_M=\frac{(M+1)^{-3}}{\zeta(3)},
\qquad
\rho_M=\frac{\zeta(3,M+1)}{\zeta(3)}.
```

The second zeta expression is the Hurwitz zeta function. Positivity gives

```math
0\preceq C-C_M\preceq q_M(I-P_M),
\qquad
\mathrm{tr}(C-C_M)=\rho_M.
```

These quantities support two different types of control.

The full-state mean-square forcing remainder admits the bound

```math
\mathbb E\|f_t-f_t^{(M)}\|_H^2
\leq \rho_M(2t-1+e^{-t}).
```

It uses the same retained Brownian channels and independent omitted channels.
Its square root is a state RMS bound, not an operator covariance error in the
same units. The corresponding full normalized-innovation trace remainder is
bounded by

```math
\mathrm{tr}(V_\Delta-V_{\Delta,M})
\leq \rho_M\frac{2\Delta-1+e^{-\Delta}}{\Delta}.
```

Sharper **finite-$J$ operational bounds** use the first omitted eigenvalue and
the finite-dimensional transport response to omitted forcing. Write
$S_J=P_JS_\Delta|_{P_JH}$ and define the positive diagonal matrix

```math
\begin{aligned}
D_J(s)&=P_JS_sS_s^*P_J
=\mathrm{diag}(2-e^{-s},e^{-s},\ldots,e^{-s}),\\
a_\Delta&=\frac{1-e^{-\Delta}}{\Delta},\\
\overline D_J(\Delta)
&=\mathrm{diag}(2-a_\Delta,a_\Delta,\ldots,a_\Delta).
\end{aligned}
```

The omitted-source contribution to projected innovation covariance is
bounded in positive-semidefinite order by

```math
P_J(V_\Delta-V_{\Delta,M})P_J
\preceq q_M\overline D_J(\Delta).
```

For the operational residual whose preceding state is at time $t$, an
upper-bound matrix is

```math
q_M\left[
\overline D_J(\Delta)
+\frac{1-e^{-t}}{\Delta}
\left(D_J(\Delta)-S_JS_J^*\right)
\right].
```

This includes both current omitted innovations and past-state leakage.
The sharper time coefficient uses $B_Je_0=0$; it does not assume leakage
vanishes as $\Delta$ decreases. The report records the HS norms of these
positive upper-bound matrices. They control the declared finite-$J$
covariance, and must not be presented as the above full-state or full-operator
remainder. They are conservative inequalities, not exact tail errors.
The operational maximum in the report uses the latest observed preceding
state, $t=T-\Delta$, rather than a state after the horizon.

The reported matrices and bounds are evaluated numerically. Quadrature and an
independent scaled block exponential are compared at the declared tolerance;
their agreement is not a rigorous numerical error certificate. Small floating
errors and feasibility checks remain separate from the analytic truncation
inequalities. Agreement with a finite reference is a reference discrepancy;
only an accompanying stated remainder bound connects that reference to the
infinite target.

## Population checks before Monte Carlo

The configured study proceeds in three parts:

1. Check orthonormal coordinates, finite transport and covariance aggregation,
   source tails, and the separate $C_M$, $V_{\Delta,M}$ and $K_{t,M}$ targets.
2. Vary forcing rank at fixed projection and interval, then projection at fixed
   reference forcing, and then the interval. Retain all configured cases and
   label the remaining reference tail explicitly.
3. Run independent path replications only after population checks. Compare
   projected innovations and projected-level residuals against their matching
   population covariances and risk targets.

For constant source and an exactly divisible horizon, the expectation of
semigroup-adjusted realised covariation is $TV_\Delta$, not $K_T$.
Its integrated-source target is $TC$ as the sampling interval decreases.
Finite interval, representation, leakage and Monte Carlo effects should not
be collapsed into one estimation-error number.

## Declared study and statistical diagnostics

The [full configuration](../experiments/configs/infinite-transport.toml) fixes
$M=64,128,256$, $J=8,16,32,64$, $\Delta=1/52,1/104,1/208$ years,
$T=1$ year, seed 20261024 and 20 independent path replications. The source
exponent and infinite trace stay at three and one. Gaussian covariance
integrals use order-24 quadrature, independently checked by block exponentials.
No maturity-domain cutoff or fitted source spectrum is used. The smaller
[smoke configuration](../experiments/configs/infinite-transport-smoke.toml)
checks the execution path; its forcing remainder may be unresolved at the
full-study budget and is reported that way.

The sample covariance is a mean of uncentered outer products: the oracle
innovations have known mean zero. For $N=T/\Delta$ independent Gaussian
projected innovations with covariance $V=V_{\Delta,M,J}$,

```math
\mathbb E\|\widehat V-V\|_{\mathrm{HS}}^2
=\frac{\|V\|_{\mathrm{HS}}^2+(\mathrm{tr}V)^2}{N}.
```

The report compares Monte Carlo error with this finite-dimensional identity.
For the projected-level residuals it separately reports error against $V$
and against $\overline R$, the average of $R_J(t_{i-1},\Delta)$ over the
actual preceding-state times. The latter sample remains temporally dependent;
the oracle iid formula is not reused for its uncertainty. The word "centered"
in the machine-readable `operational_centered_covariance_mse` identifies the
matching population-covariance target, not subtraction of an estimated mean.

The two predeclared exposures are a point spread between maturities 0.5 and
1.5 years and a difference of delivery averages over [0.5,1] and [1.5,2]
years. Each is normalized once by its full-$H$ dual norm. With coefficient
restriction $a_J$, the projected risk is $a_J^*V_{\Delta,M,J}a_J$. The risk
error target is instead the full-exposure risk at the largest forcing
reference, keeping signed projection and forcing biases explicit. The scalar
Gaussian reference MSE therefore includes both sampling variance and the
squared total bias. Operational risk adds the time-average leakage quadratic
form; it is not assigned an iid Gaussian MSE formula.

All error summaries average within a path replication first. MCSE is the
sample standard deviation of independent replication averages divided by the
square root of their count. Shared dates, forcing bands and coarse/fine time
levels are coupling devices, not extra independent samples. These MCSEs are
Monte Carlo precision measures, not confidence intervals for an empirical
market parameter.
Operational-minus-oracle covariance and risk MSE differences are paired
within each path before replication MCSE is calculated. This separates the
measured leakage effect from between-path variation without imposing a
favorable sign or a superiority acceptance gate.

The existing PSD trace-capped ridge solver is also exercised at $J=8,16$;
larger projections use population and direct sample diagnostics. The solver's
finite objective certificate is kept separate from forcing, projection and
sampling errors. This benchmark neither estimates a unique diffusion factor
nor solves an inverse map from $V_\Delta$ to $C$.

Reproduce a smoke run with the reference-solver extra installed:

```bash
uv sync --locked --extra reference
uv run spde-covariance benchmark-infinite-transport experiments/configs/infinite-transport-smoke.toml --output experiments/runs/infinite-transport-smoke
```

For the full study, substitute `infinite-transport.toml` and choose a fresh
output directory. A published numerical report should come from committed
source, with its configuration, manifest and raw replication results retained.

## Reports, acceptance and limitations

The report records configuration, basis/source conventions, source and
environment provenance, population identities, truncation bounds, full
resolution ladders and independent-replication summaries. The three standard
Matplotlib figures show covariance kernels; separate forcing/projection/interval
effects; and projection leakage with exposure-risk errors. Figure selections
are fixed by configuration rather than by the smallest realized error. Kernels
use the largest configured $M$ and $\Delta$ and the full finite forcing model,
not a selected $J$ projection. Forcing comparisons retain all valid $J,\Delta$
pairs; projection, interval, leakage and risk panels use reference forcing and
all configured projections, frequencies and exposures. A separate interval
panel shows $\|V_{\Delta,M}-C_M\|_{\mathrm{HS}}$, independently of sample MSE.

Acceptance concerns the declared numerical identities, tolerances, bound
calculations and reproducibility. It does not require an estimator to beat a
baseline or a finite model to equal the infinite process. A population
truncation bound is not a Monte Carlo confidence interval; an optimizer
certificate is not statistical error against truth.

Numerical identity acceptance and reference-tail resolution are separate
checks. For the full study, the largest forcing reference must satisfy the
declared tail budget for full source trace, full innovation trace, full state
mean square, and the finite-$J$ covariance bounds at every configured
projection and frequency. These criteria have distinct units; the configured
numerical budget is a declared audit threshold rather than one common error
norm. Smaller forcing levels intentionally remain a resolution ladder; they
need not all resolve that same budget. The report displays both the full
trace/MSE remainders and the sharper operational covariance bounds.

This foundation uses known transport, constant deterministic trace-class
forcing and clean projected states. It does not establish new statistical
rates, feasible confidence intervals, unknown-generator recovery,
price-level measurement-noise correction or empirical futures forecasting.
[The three-question roadmap](research-roadmap.md) describes the later
identification, estimation and forecast questions.

The semigroup adjustment follows the established realised-covariation
programme; [Benth, Schroers and Veraart (2024)](https://arxiv.org/abs/2205.03927)
develop high-frequency asymptotic inference for a broader stochastic-volatility
setting. This fixed deterministic-source foundation does not reproduce their
CLT or claim a new infinite-dimensional statistical result.

## Recorded first full study

The [full report](../experiments/results/infinite-transport/report.md) records
20 paired replications from clean numerical source
`3f573c746d720aceda8861a4655cb1f7d27ed2ef`, using the committed full configuration.
The [manifest](../experiments/results/infinite-transport/manifest.json) records
the environment and numerical-source hashes; the artifact index preserves
checksums for both raw and published measurements.

At reference forcing $M=256$, the finite-interval effect and empirical sampling
control are:

| Interval (years) | HS norm of finite-reference V minus C | Oracle covariance MSE, J=8 (mean ± MCSE) | Exact Gaussian MSE |
| --- | ---: | ---: | ---: |
| 1/52 | 0.00257882 | 0.0245323 ± 0.00678248 | 0.0324263 |
| 1/104 | 0.00130334 | 0.0143521 ± 0.00459749 | 0.0162437 |
| 1/208 | 0.000655229 | 0.00940489 ± 0.00294011 | 0.00812968 |

The first column of errors is a population HS norm; MSE is a squared HS error.
The near-halving of the finite-interval effect is a numerical observation for
this model and ladder, not an estimated general convergence rate. Increasing
sampling frequency also increases the number of observations over the fixed
one-year horizon. The J=8 empirical means differ from exact Gaussian expectations
by at most 1.2 reported MCSE; these cases are not additional independent studies.

At weekly sampling, increasing $J$ from 8 to 64 reduces the population covariance
projection error from 0.00172013 to 0.0000109714, measured against full
finite-reference $V$. The HS norm of time-average hidden-state leakage falls
from 0.0000988364 to 0.00000295820. These are different errors. In the paired
$J=8$ experiment, operational-minus-oracle covariance MSE is
0.0000355182 ± 0.0000666618 MCSE. The nonzero population leakage does not establish
a universally larger realized risk or covariance MSE in this small study.

The remaining source trace is bounded by 6.32221e-6; the largest full innovation
trace bound is 6.38261e-6 and the terminal state mean-square bound is 8.64801e-6.
The maximum configured finite-J operational covariance tail bound is 1.78454e-6.
Each passes its declared 1e-5 audit threshold, with units and targets kept
separate. This does not require the intentionally varied J-projection errors
to fall below that forcing-reference threshold.

All 240 constrained fits are feasible and converged; the largest numerical
covariance-distance certificate is 5.65966e-8 against a 1e-4 budget. Two small
independent convex-reference comparisons have matrix discrepancies below
1.8e-15. Independent-method integration and path-identity checks remain
floating-point validation, not interval-arithmetic proofs. These results close
the foundation benchmark; partial spatial observations and statistical
operator recovery remain the next research questions.
