# Transport and innovation workflow

`CurveTrajectory` stores clean coefficients at interval endpoints, in an
identified orthonormal Hilbert basis. Times are numeric years. `curve_unit`
records the units of the curves; extracted innovations have units
`curve_unit/sqrt(year)` and their covariance has units `curve_unit²/year`.

For interval length Δ, extraction computes

```text
X[i] = (f[i+1] - S(Δ[i]) @ f[i] - integrated_drift[i]) / sqrt(Δ[i])
```

Omitting `integrated_drift` assumes zero mild drift. A supplied drift is the
already integrated, transported drift contribution in the same coefficient
basis, rather than an instantaneous drift rate. Centered Gaussian innovation
assumptions require the drift convention to be correct.

`ExponentialTransport` uses an exact shift-invariant span of exponentials with
rates `(0.6, 1, 2, 4)` per year by default. Its Gram matrix and normalized
innovation covariance are analytic. `ProjectedSplineTransport` uses `P S(Δ) P`;
its trajectories follow a projected mild recursion. Repeated projection does
not form an exact semigroup and introduces a separate approximation error.

`InnovationBatch` preserves every interval. Its `observe` method applies point,
delivery-average or other explicit linear maps after extraction. Noise supplied
here means fresh independent innovation observation noise with known covariance.
It does not represent differencing noisy price levels. Latent covariance truth
is stored separately on `TransportSimulation` and never enters fitting.

The high-level `fit_innovation_covariance` requires an explicit `common` or
`local` covariance assumption, equal interval lengths, positive ridge, a trace
cap and a numerical covariance-error budget. A local fit is a weighted common
covariance approximation over its window; the code records this assumption and
does not establish local stationarity. Trailing windows use endpoint dates in
years. A failed convergence check raises an exception containing the feasible
iterate and its certificate.

For irregular dates, extraction still uses the correct interval normalization.
The target is interval-specific

```text
V[i] = integral_0^Δ[i] S(r) C[i] S(r)* dr / Δ[i].
```

The common-covariance fitting workflow rejects such pooling. Inspect interval
diagnostics and the SPDE bridge instead. The low-level `fit_covariance` remains
available for explicitly constructed observation sequences under separately
justified target assumptions.

All arrays are in orthonormal coordinates. Matching dimensions alone does not
establish matching bases; basis identifiers must agree. An objective-gap bound
certifies numerical optimization error relative to the constrained optimum,
not statistical covariance estimation error or identification of weakly
observed directions. Certificates are evaluated in floating point and are not
interval-arithmetic proofs.
