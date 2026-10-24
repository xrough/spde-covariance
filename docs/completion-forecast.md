# Covariance-completion forecasts

This study asks whether local covariance completion gives useful forecasts of
future **weekly normalized innovation covariance**, and whether observed
validation can choose a useful window. It combines existing strategies and the
[RQ3 chronology](research-question-3.md); it introduces no new forecast model or
estimator theorem. The joint-resolution and partial-estimation gates precede
interpretation of its results. A selected window need not outperform a static
or fixed window.

## Source model and prediction target

Use the same half-line Hilbert space and integrated Laguerre basis as the
[infinite-source foundation](infinite-transport.md). The base source eigenvalues are
fixed, infinite and unrenormalized:

```math
q_k=\frac{(k+1)^{-3}}{\zeta(3)},\qquad k=0,1,\ldots.
```

Let a deterministic source covariance be constant within each weekly interval.
Scale it by a positive number and rotate only the first two Laguerre modes:

```math
C_i=s_i R_i\,\mathrm{diag}(q_k)R_i^*.
```

The remaining modes are unchanged by the rotation. Deterministic diffusion
and disjoint Brownian intervals give independent centered Gaussian normalized
innovations, with target

```math
\xi_i=\frac{f(t_i)-S_{\Delta}f(t_{i-1})}{\sqrt{\Delta}},
\qquad V_i=\frac{1}{\Delta}\int_0^{\Delta}S_r C_i S_r^*\,dr.
```

Drift is zero in this controlled experiment. The scenarios are constant source;
scale increasing linearly from 0.75 to 1.25; rotation increasing linearly from
zero to pi/2; and an abrupt first-two-mode swap immediately after interval 256.
These changes are specified at the source C, rather than applying an arbitrary
path to V and calling it an SPDE. Four fixed finite-prefix integrations evaluate
the paths by linearity; quadrature/block agreement is a numerical cross-check,
not interval arithmetic.

The full configuration uses forcing cutoff M=256 and base grid 21 on maturities
0–3 years, augmented by the fixed exposure points. The fixed infinite source is
approximated by its first M modes, with remaining source and normalized-innovation
trace bounds saved in `model.json`. The maximum scale 1.25 is included in those
bounds. Finite-reference prediction error does not certify prediction of the
exact infinite process. This is a fixed-grid forecast experiment; it does not
establish convergence as M, grid size and sample size increase.

## Observations and fitting

One parent grid-valued Gaussian innovation and point-noise bank are generated
per replication/scenario before any method or fragment view. Each observation
contains all grid points in one declared fragment, with independent known noise
variance 0.001. Three- and nine-fragment catalogues use the established
overlapping covers. One seed-shuffled catalogue repeats cyclically, ensuring
every training window of at least nine dates sees every fragment. This design
choice differs from independently reshuffling each catalogue cycle.

These are **clean innovations observed through noisy point fragments**. The
study does not recover innovations from sparse/noisy price levels or forecast
the curve level. It does not infer source covariance or physical loadings.

The three methods are empirical pairwise partial covariance followed by
rank-two canonical completion; that same estimate followed by the explicitly
separate weighted PSD projection; and coherent shared-clique PSD fitting followed
by full numerical-support completion. See [coherent fitting](coherent-completion.md)
for the objective and certificate scope.

Within each fitting window, one constant covariance is a working approximation
to the changing source path; it is not asserted to be the common population
covariance of every training observation. The design can give different
calendar weights to different observed pairs.

For coherent fitting, relative penalty eta=0.001 is predeclared and multiplied by
the maximum normalized training pair coverage. The weighted-coordinate floor
is gamma/Q² with gamma=0.02116 and actual augmented grid size Q. This declining
grid floor is a numerical restriction; it is not a continuum white-noise
component or a theorem about joint limits. The observed-partial optimizer-error
budget is 1e-4, maximum ADMM iterations 100,000 and support cutoff 1e-12.

## Frozen forecasts and selection

There are 512 weekly intervals and 20 independent paired replications.

| Rule | Training observations | Update policy |
|---|---|---|
| Static | First 128 | Fit once |
| Fixed | Last 128 before each issue | Fit at issue, then freeze |
| Selected | Last 16, 32, 64 or 128 | Select one window per method using validation, then fit at each test issue |

Validation issues are 128 and 192, each scored on the following 16 observations.
Every method selects its own window using those identical observations; exact
ties choose the longest window. Selection is completed at interval 208. No
candidate window is silently discarded for numerical failure. If any declared
candidate has an unavailable validation forecast, that method's selected rule
is unavailable, while its static/fixed forecasts remain separately recorded.

Test issues are 256, 272, 304 and 448. Each forecast is frozen for horizons 1–16;
intervening test outcomes do not update that issue's forecast. Horizon h refers
to the covariance of the h-th future weekly innovation. It does not mean the
covariance of an accumulated h-week return. All fit APIs receive only the exact
past fragment prefix and scenario-free numerical settings.

## Scores, risks and simulation diagnostics

If O selects a future fragment and its noise covariance is N, put Z=yy*−N. The
reduced observed score is

```math
\ell(B;y,O)=\lVert OBO^*\rVert_F^2-2\langle Z,OBO^*\rangle_F.
```

Its expectation equals squared observed prediction error minus a
forecast-independent constant. It may be negative. Only paired scores on the
same observations are compared; raw scores across three- and nine-fragment
designs have different scales. This moment identity extends the scalar proxy
logic discussed by [Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf);
that paper does not establish this study's completion or infinite-dimensional
claims.

For a point-functional exposure a, an exact future proxy exists when its entire
support is contained in that one observed fragment. Then

```math
\widehat r=(a^*y)^2-a^*Na,
\qquad \ell_r(\theta)=\theta^2-2\theta\widehat r.
```

Here a is restricted to the actual observed row order. Point and off-grid
spreads have scores only on qualifying future dates. The long spread between
0.25 and 2.75 years is never jointly observed by these catalogues: its operational
score stays unavailable. Pooled historical coverage does not create that proxy.
All exposures are fixed point functionals; no continuous-delivery approximation
is presented as exact.

Weighted full/missing kernel error and risk MSE use the generating finite-grid
V_i **after issue**, as separately labelled simulation diagnostics. They are not
selection inputs. Canonical completion can differ from the generating kernel;
an observed score cannot establish recovery of never-observed covariance.
The raw rank-two estimate may be indefinite and produce negative variance
predictions; those values are retained rather than silently projected.

For each reported contrast, first average dates/origins within replication,
then compute the mean and MCSE across independent replications. Failed fits
cannot improve a replication average by disappearing from it. Risk scores
average only the identical qualifying future dates for every method. MCSE is
Monte Carlo uncertainty of this experiment, not a statistical confidence
interval for market data.

## Reproduction and evidence

```bash
uv run spde-covariance benchmark-rq3 experiments/configs/completion-forecast-smoke.toml --output experiments/runs/completion-forecast-smoke
uv run spde-covariance benchmark-rq3 experiments/configs/completion-forecast.toml --output experiments/runs/completion-forecast
```

The smoke configuration uses two replications and constant/abrupt scenarios,
retaining the same validation/test chronology. Its smaller forcing/grid levels
are development checks, not substitutes for the full evidence.

The run stores source/configuration manifests, independent replication
checkpoints bound to the parent manifest SHA256, all fits, validation scores,
selections, future scores, risk availability and paired summaries. Up to four
spawned workers execute deterministic replication seeds; aggregation is in
replication order. Numerical acceptance requires complete declared coverage,
available coherent fits, coherent certificate budgets and PSD coherent forecasts.
Unavailable historical baselines remain descriptive outcomes with explicit
counts. Summaries using fewer than all replications are labelled conditional
on available complete pairs. No
advantage over another strategy is required. The
[accepted full report](../experiments/results/completion-forecast/report.md)
records 20 paired replications from clean numerical source
`c451aa7a55596a16ac726e7a2507ecc308897847`, with 6,772 available fits and
2,264 certified coherent fits. The [findings](completion-continuation-findings.md)
describe the measured comparisons and limits; window selection does not
demonstrate an advantage over the fixed 128-observation rule.
