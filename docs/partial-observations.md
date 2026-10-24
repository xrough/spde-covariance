# Partial observations of fixed-interval innovations

This study asks which covariance quantities can be learned from a limited
set of point and delivery observations of the same infinite-model innovation.
It studies a fixed weekly interval and established finite covariance methods.
It does not claim a new estimator, a uniform recovery theorem or a solution
to sparse/noisy curve-level reconstruction.

## Target and fixed model

Use the half-line Hilbert space, orthonormal integrated-Laguerre basis and
known shift semigroup of the [infinite-transport foundation](infinite-transport.md).
For a fixed interval $\Delta=1/52$ years, the target is

```math
V_\Delta=\frac{1}{\Delta}\int_0^\Delta S_rCS_r^*dr.
```

It is the covariance of a normalized transport-adjusted innovation, not
instantaneous source covariance $C$, state covariance $K_t$ or a unique
Brownian loading. Independent Gaussian innovations with this covariance are
sampled directly. They are not extracted from observed sparse curve levels.

Two fixed infinite-rank source cases are used: the trace-one power-law source
with eigenvalues $q_k=(k+1)^{-3}/\zeta(3)$ and a fixed orthogonal rotation of
adjacent basis pairs through angle $\pi/6$. The rotation changes eigenvectors,
not eigenvalues or total trace. The finite reference retains $M=256$ forcing
coordinates without renormalizing eigenvalues. Rotation blocks and cutoffs
are aligned, so omitted-source trace and eigenvalue bounds retain their scope.
The source case defines simulation truth; it is not supplied as a prior
covariance to the numerical fitter.

## One full observation before choosing a representation

Let $O_i$ contain the actual point/delivery functionals observed on date $i$.
The conceptual observation model is

```math
y_i=O_i\xi_i+\varepsilon_i,
\qquad
\xi_i\sim N(0,V_\Delta),
\qquad
\varepsilon_i\sim N(0,N_i).
```

Innovation and fresh measurement noise are independent, and noise
covariances are known. The numerical generator approximates this full
observation with the declared forcing reference and reports its remaining
tail. Each contract has fresh independent noise variance 0.001. A single full
16-contract observation bank is generated before any
candidate dimension is fitted. Designs select contracts from this bank;
changing $J$ never regenerates or projects the observed values.

The five declared designs are two fixed points at maturities 0.5 and 1.5,
two fixed delivery averages over [0.5,1] and [1.5,2], nested sets of 8 and
16 contracts, and a sliding cyclic window of 4 contracts with period 16.
The ordered pool begins with the two matching points and the two matching
deliveries, followed by points at 0, 1, 2, 3, 0.25, 1.25, 2.5, 3.5, 4, 4.5,
5 and 6 years. Nested designs take the first 8 or all 16 rows. Rotating dates
advance the consecutive four-row window by one, wrapping at the end. Exact
maps are also retained in machine-readable metadata. A delivery observation
is an average, not an integral with an unreported interval-length scale.

A $J$-coordinate fit uses $O_{i,J}=O_iP_J$ and a PSD covariance matrix $B$.
Its observation covariance prediction is $O_{i,J}BO_{i,J}^*$. The population
approximation incurred even by the true projected covariance is

```math
E_{i,J}=O_iV_\Delta O_i^*
-O_iP_JV_\Delta P_JO_i^*.
```

Writing $Q_J=I-P_J$, this difference includes

```math
\begin{aligned}
E_{i,J}
={}&O_iP_JV_\Delta Q_JO_i^*\\
&+O_iQ_JV_\Delta P_JO_i^*\\
&+O_iQ_JV_\Delta Q_JO_i^*.
\end{aligned}
```

The first two terms are cross covariance. Even a diagonal source generally
produces correlated retained and omitted innovation coordinates after
transport. Thus $E_{i,J}$ need not be positive semidefinite or monotone in
$J$. It cannot silently be treated as independent additional measurement
noise. Numerical source-tail bounds and this representation error are
separate diagnostics.
For time-varying designs, reported HS errors are RMS norms across the
declared date maps; relative error divides by the RMS norm of full
observation covariance on those same dates. Cross-term and omitted-omitted
norms are separate components, not additive error magnitudes.
For an algebraic two-coordinate example, take

```math
V=\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix},
\qquad O=(1,-1),\qquad J=1.
```

The observed variance is 0.4, whereas
the sieve variance is 1. The difference is $-0.6$: omitted-coordinate
variance contributes $1$ but the two cross terms contribute $-1.6$.

## Three different observation questions

The normalized finite-coordinate Gram operator is

```math
\mathcal G_J(D)
=\sum_i w_i O_{i,J}^*O_{i,J}D O_{i,J}^*O_{i,J},
\qquad
\sum_iw_i=1.
```

Its rank and spectrum describe the finite symmetric-matrix inverse problem.
For a restricted exposure $a_J$, the lifted target is $a_Ja_J^*$. A numerical
range residual for this target distinguishes missing finite-coordinate
information from weak information at the declared threshold. Regularization
can select one matrix even when this target or other directions are absent.
These diagnostics concern the unrestricted finite PSD covariance class used
by the fitter. A generic covariance nullspace witness is not automatically
the image of a positive source covariance under the SPDE transport bridge;
no such source-constrained identification theorem is claimed.

This differs from membership of a full-H exposure in the functional span
of a contract set. A finite truncation can make functionals look dependent
or make a restricted target appear observable while the full functional
remains outside that span. Full functional spans use the actual half-line
Sobolev dual metric, rather than a conveniently small coefficient truncation.

A third question is whether a same-date unbiased scalar risk proxy is
available. If $a=O_i^*c_i$ as a full-H functional, then

```math
z_{a,i}=(c_i^*y_i)^2-c_i^*N_ic_i,
\qquad
\mathbb E z_{a,i}=\langle a,V_\Delta a\rangle_H.
```

Only those dates receive a direct risk score. Unavailable dates are not
filled with zero or simulated latent truth. Failure of a same-date span test
alone does not prove that pooled covariance information cannot identify the
risk. Conversely, membership in the union of contracts does not guarantee
the necessary joint covariance coverage on individual dates.

The fixed exposures are the matching point spread, the matching delivery
spread, and the off-grid point spread between 0.75 and 2.25 years. The last
is deliberately outside the contract pool. Each exposure is normalized once
in full $H$ and then restricted without a dimension-dependent rescaling.

## Fitting, selection and chronology

The full protocol fixes seed 20261025 and 20 independent replications.
Training sizes are 128 and 512 innovations. Candidates have
$J=4,8,16$ and relative ridge penalties 0.001, 0.01 and 0.1. Numerical geometry
and population approximation are also checked at $J=32$. The trace cap is
two and the finite-objective covariance error budget is $10^{-4}$.

Every candidate fits only its declared training observations. The 128-date
validation block follows the largest training prefix; the disjoint 128-date
test block follows validation. Both training-size experiments use those same
validation/test observations. Validation selects a candidate, whose fitted
matrix remains frozen: there is no refit through validation and no test-data
selection. Unused observations between the shorter prefix and validation
are not additional training data.
Exact validation ties prefer smaller $J$, then larger relative penalty.
The predeclared fixed baseline is the already fitted candidate $J=16$,
relative penalty 0.001. It also remains frozen, with no validation refit.

With corrected outer product $Z_i=y_iy_i^*-N_i$, candidates minimize

```math
\min_{B\succeq0,\,\mathrm{tr}(B)\leq2}
\frac12\sum_iw_i
\|Z_i-O_{i,J}BO_{i,J}^*\|_F^2
+\frac{\lambda}{2}\|B\|_F^2.
```

Weights sum to one and $\lambda=\lambda_{\mathrm{rel}}\lVert\mathcal G_J\rVert$,
so an unnormalized sample count does not silently change the penalty scale.
The solver certificate concerns this finite constrained objective in
floating point. It does not bound truncation error or statistical recovery of
the infinite covariance.

Selection and test evaluation use the reduced observed score

```math
\ell_i(B)=\|O_{i,J}BO_{i,J}^*\|_F^2
-2\langle Z_i,O_{i,J}BO_{i,J}^*\rangle_F.
```

Its expectation differs from squared error against the full observation
covariance by a candidate-independent constant. Scores can be negative.
Candidates are compared on exactly the same observed values, maps and noise
correction. Score magnitudes across different designs need not be comparable
because the measurements and dimensions differ. Observed validation selects
observed-space prediction performance; it cannot validate unseen covariance
directions or guarantee recovery of an unobserved risk.

Observed-covariance forcing tails use the full-H contract Gram matrix $G$.
With $a_\Delta=(1-e^{-\Delta})/\Delta$ and each contract having unit total
mass, a positive upper-bound matrix for the omitted-source observation
covariance is

```math
q_M\left[a_\Delta G+(2-2a_\Delta)\boldsymbol 1\boldsymbol 1^*\right].
```

This is a finite-observation analytic inequality evaluated in floating point,
not a bound on numerical quadrature error or a guarantee of covariance
recovery. It is separate from the full innovation trace-tail bound and from
the candidate-$J$ observation approximation error.

For an available exposure, the analogous score is
$\widehat\theta_a^2-2\widehat\theta_a z_{a,i}$, where
$\widehat\theta_a=a_J^*Ba_J$. Latent operator and full-exposure risk errors are
evaluation-only diagnostics. They never choose $J$, a penalty or a panel.
The direct comparison estimates risk by averaging available training proxies.
It can be negative because an unbiased noise-corrected scalar estimate is not
constrained to be a variance; no clipping is introduced for the comparison.

## Evidence and limitations

The [full report](../experiments/results/partial-observations/report.md) records
20 paired replications from clean numerical source
`6d2be636fe23bb5db9ad39e2296ae731abddda1a`. All 3,600 candidate fits meet the
$10^{-4}$ finite-objective covariance budget; the largest certificate is
$9.9997\times10^{-5}$. Both independent convex comparisons pass, with maximum
matrix discrepancy $1.52\times10^{-9}$. These are numerical checks, not
statistical recovery bounds.

At training size 512, observed-validation-selected fits have the following
relative operator HS errors against the common finite reference. Entries are
means ± replication MCSE; all five designs and both sources are retained.

| Design | Power-law source | Rotated source |
| --- | ---: | ---: |
| Two points | 0.6750 ± 0.0032 | 0.4145 ± 0.0016 |
| Two deliveries | 0.7209 ± 0.0045 | 0.4092 ± 0.0022 |
| Nested 8 | 0.3359 ± 0.0442 | 0.2294 ± 0.0194 |
| Nested 16 | 0.3929 ± 0.0456 | 0.1784 ± 0.0186 |
| Rotating 4 | 0.5370 ± 0.0384 | 0.4982 ± 0.0841 |

Observed-space tuning does not optimize latent operator error. The paired
test-score comparisons do not show a uniform advantage over the fixed
$J=16$, relative-ridge-0.001 baseline. A larger observation set or candidate
dimension does not guarantee a better selected estimate in this experiment.
The 16-contract numerical Gram rank is 10/10 at $J=4$, 44/136 at $J=16$ and
69/528 at $J=32$, at relative threshold $10^{-10}$. Discarded numerical
directions can include very weak positive eigenvalues; these counts are not
proofs of exact nullspace dimensions.

The off-grid exposure has no same-date full-H proxy in any design, so all
its operational risk scores are unavailable. Matching point and delivery
risks have 24/128 future proxy dates in the rotating design, and 128/128 when
their contracts are jointly observed throughout. Their direct estimates do
not require recovering the entire covariance matrix.
For the power-law source, two-point design and 512 training observations,
direct point-spread risk MSE is $(6.6755\pm2.4399)\times10^{-6}$, versus
$(4.0173\pm0.5300)\times10^{-4}$ for the observed-score-selected matrix's
plug-in risk. Both uncertainties are replication MCSE. This is a measured
case, not a universal ordering of estimators; the full report retains all
available and unavailable exposure comparisons.

At $M=256$, the full innovation trace-tail bound is $6.38\times10^{-6}$, the
16-contract observed HS tail bound is $1.27\times10^{-6}$, and each declared
unit H-norm spread-risk tail bound is $4.86\times10^{-8}$; all meet the declared
$10^{-5}$ tail budget. Some $J=32$ finite-reference approximation errors are
smaller than their observed tail bound. They are not resolved measurements of
the corresponding exact infinite-model errors. The report preserves these
reference and representation diagnostics separately, alongside the numerical
quadrature/block-exponential cross-checks.

Reproduce the smaller execution check with

```bash
uv sync --locked --extra reference
uv run spde-covariance benchmark-partial experiments/configs/partial-observations-smoke.toml --output experiments/runs/partial-observations-smoke
```

The [full configuration](../experiments/configs/partial-observations.toml)
fixes the protocol above. The
[smoke configuration](../experiments/configs/partial-observations-smoke.toml)
uses fewer replications, smaller source/candidate dimensions and shorter
blocks; it does not substitute for the full numerical report. Use a fresh
output directory for each run. Reference-tail resolution is separate from
numerical run acceptance; a smaller smoke reference can remain unresolved at
the same declared tail budget.

The report retains every configured source/design/training-size case and
every exposure, including unavailable risk scores. Population geometry,
observation approximation and selected performance appear in three plain
Matplotlib figures. Numerical checks and infinite-source remainder bounds
are reported separately. Configuration, source/environment manifests,
candidate selections and raw replication outcomes accompany the report.

Monte Carlo uncertainty is computed across independent replication averages;
dates within a replication are not extra independent experiments. Paired
comparisons use within-replication differences before their MCSE is computed.
Acceptance checks implementation, chronology, coverage, numerical tolerances
and declared reference resolution. It does not require a selected candidate
to beat another method or make an unavailable exposure observable.

This is an observation-geometry and estimation experiment for cleanly
available innovations with fresh observation noise. It does not reconstruct
innovations from sparse/noisy price levels. Differencing noisy levels would
introduce noise shared across dates, amplified by normalization; that requires
a separate observation/filtering model. It also does not estimate source
covariance, introduce stochastic volatility, establish a spatially uniform
statistical rate or forecast market curves.
