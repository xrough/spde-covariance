# Computational method and supported assumptions

## Estimated target

The fitted output is the normalized innovation covariance V_delta. In curve
coordinates, transport adjustment is

```text
xi[i] = (a[i+1] - M(delta) @ a[i] - integrated_drift[i]) / sqrt(delta)
V_delta = Cov(xi[i])
```

The loading matrix L generates Brownian forcing. Its instantaneous covariance
C=L L.T is supplied to simulation. For constant deterministic C, the exact
finite-dimensional transport model has

```text
V_delta = integral_0^delta M(r) @ C @ M(r).T dr / delta.
```

The current estimator fits V_delta. Fitting C or individual loadings L is an
additional problem. L is not uniquely determined by a covariance factorization.
The default analytical control uses four independent Brownian sources in an
exact shift-invariant exponential span. No specific spatial mollifier is used.

## Observation inputs

Each observed innovation contains a date, vector y, linear map O, and known
fresh innovation-noise covariance N. The model is y=O xi+epsilon. Fitting uses
the noise-corrected outer product Z=y y.T-N, including negative eigenvalues in Z.
Observation maps can change over dates and represent maturity evaluations or
delivery averages. Noise is added after extracting innovations from clean
coefficient trajectories. Sparse/noisy price-level reconstruction is outside
this workflow.

The high-level fit requires equal intervals, correct zero/supplied integrated
drift, and an explicit common or local covariance assumption. A local fit is an
approximation over its selected window; the software does not verify stationarity.
Irregular extraction preserves interval-specific targets and rejects automatic
common-target pooling.

## Estimation and numerical diagnostics

The solver minimizes

```text
sum_i w[i] * ||Z[i] - O[i] B O[i].T||_F^2 / 2 + lambda * ||B||_F^2 / 2
subject to B positive semidefinite and trace(B) <= R.
```

Date weights are normalized, ridge lambda is positive, and the trace cap R is
explicit. Accelerated projected gradient supplies the fitted covariance;
CVXPY/SCS independently checks small problems. The reported objective-gap and
covariance-distance bounds quantify numerical error relative to the constrained
optimum. They are evaluated in floating point and do not bound statistical error
against truth. Failure to meet the requested budget raises `ConvergenceError`.

Gram spectra and target residuals describe which finite-coordinate covariance
directions the observation maps constrain. Numerical rank uses a reported
threshold; weakly observed directions can remain inaccurate even at full rank.
Direct risk estimation is available when the target passes its range check.
PSD constraints and a unique optimizer do not establish data identification.

## Validation and approximation

The benchmark selects window counts and relative ridge strengths using observed
validation outcomes. Penalties are scaled by the window's Gram norm and rescaled
on refit. Latent truth is reserved for evaluation. Complete-data empirical and
shrinkage baselines receive identity-map data; partial-data methods receive the
same partial observations within each comparison.

The weighted Sobolev basis metric is anchored at maturity zero with derivative
weight exp(x/tau), tau=1 year. Exponential transport is exact within its chosen
finite span. Spline transport is a projected mild recursion with a padded domain
and constant tail. Projection, boundary, quadrature and transport errors are
reported separately. Computation uses dense matrices for moderate dimensions.

See the [API](api.md), [workflow](workflow.md), and
[benchmark report](../experiments/results/stage1/report.md) for runnable details.
