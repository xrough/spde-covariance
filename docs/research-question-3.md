# Research question 3: forecast evaluation

**Do current covariance estimates predict subsequent observed covariance and
exposure risk, when fitting and tuning use only earlier observations?**

This study closes the first, finite-dimensional pass through the three research
questions. RQ1 diagnoses information in observation maps; [RQ2](research-question-2.md)
estimates current risk. RQ3 evaluates estimates on later data. It is a controlled
forecast-evaluation exercise using established estimators, not a new forecasting
model or a market prediction claim.

**Status:** completed with 20 paired replications. The [full report](../experiments/results/rq3/report.md)
retains the predeclared comparisons, three figures, raw measurements and source
provenance. All 5,708 unique fits and 4,800 issued forecast records satisfy the
numerical checks; four convex references have maximum matrix discrepancy
3.84e-8. Numerical accuracy is separate from forecasting accuracy.

The measured outcomes are mixed in the intended way. For dense observations
under rotating covariance, static minus fixed-local observed score is
0.01691 ± 0.00166 MCSE, favouring refitting. Selected minus fixed-local is
0.000583 ± 0.000473, with no demonstrated benefit from validation selection.
Selected-minus-fixed mean scores are positive in all 20 scenario/design cells;
many are small relative to their paired MCSE. These finite comparisons do not
establish that a fixed window is universally optimal.

Future exposure availability matters independently: the joint-every-64 design
can score cross-pair risk on only one of the 64 test dates per replication.
The report distinguishes those operational scores from latent simulation errors.

## Target and observation model

Reuse the independent Gaussian innovation paths, four-dimensional basis,
covariance paths, exposures and deterministic observation designs from RQ2:

```math
\begin{aligned}
X_j&\sim N(0,V_j),\\
y_j&=O_jX_j+\varepsilon_j,\\
\varepsilon_j&\sim N(0,N_j),\\
Z_j&=y_jy_j^\top-N_j.
\end{aligned}
```

Innovations and fresh noise are independent across dates, and noise is
independent of the innovations. Covariance paths, observation maps and noise
covariances are deterministic in the simulation. Fitters receive observed
values, maps and noise corrections; they do not receive latent $X_j$ or $V_j$.

At origin $n$, each policy fits one PSD covariance matrix $B_n$ using only
observations through $n$. It is frozen for the next 16 intervals. Its targets
are the individual future innovation covariances $V_{n+1},\ldots,V_{n+16}$,
and exposure variances $a^\top V_j a$ for exposures declared before simulation.
This is not the covariance of a cumulative 16-week curve move.

As in RQ2, $V_j$ is the covariance of a normalized finite-interval innovation.
These paths are sampled directly to isolate estimation and evaluation. The
study does not simulate an SPDE coefficient trajectory, infer a source
covariance $C$, identify a loading $L$, or establish that every prescribed
$V_j$ arises from a PSD source through a transport bridge. The separate Stage 1
transport experiment validates that bridge and innovation extraction.

## Three forecast policies

All three policies use the same PSD/trace-constrained ridge estimator. A
relative ridge strength is converted to an actual penalty using the fitted
window's lifted observation Gram norm:

```math
\lambda=10^{-3}\|Q_{\mathrm{window}}\|_2,
\qquad
B\succeq0,\qquad \mathrm{tr}(B)\leq2.
```

The numerical covariance-distance budget is $10^{-4}$ and the relative
rank threshold is $10^{-10}$. These are fixed controls, not parameters
selected using latent forecast errors.

| Policy | Information used at an origin |
| --- | --- |
| Static training fit | Fit observations 1–128 once; reuse the same matrix at every later origin |
| Fixed local fit | Refit on the last 128 observations through that origin |
| Selected local fit | Refit on the last selected count, using a count chosen earlier by observed validation |

Selected-window candidates are 16, 32, 64 and 128 observations. Selection is
performed separately for each replication, covariance path and observation
design. For each candidate, fit through validation origins 128 and 192 and
freeze the resulting matrices on their following 16-date blocks. Average the
observed covariance scores across those two blocks and choose the smallest.
Exact score ties choose the longest count. All candidate scores and the selected
count are retained.

The selected count is then fixed before test evaluation. The selected policy
refits through each test origin using that count, without using a future test
observation to fit its own forecast. Earlier test observations can enter a
later origin's historical window, because they are then available. There is no
test-score-based reselection.

## Future observed covariance score

For a matrix forecast $B_n$ issued before date $j$, define
$P_{n,j}=O_jB_nO_j^\top$. Score its prediction using

```math
\ell_{n,j}
=\|P_{n,j}\|_F^2-2\langle Z_j,P_{n,j}\rangle_F.
```

The common term $\|Z_j\|_F^2$ is omitted from squared prediction loss.
Scores can consequently be negative. Lower scores are better, and methods
must be compared on the same observations, dates and design. Negative scores
are not clipped.

The reason for this choice is the conditional second-moment identity. Under
the specified model, for a forecast based on information through $n<j$,

```math
\begin{aligned}
\mathbb E[\ell_{n,j}\mid\mathcal F_n]
&=\|P_{n,j}-O_jV_jO_j^\top\|_F^2\\
&\quad-\|O_jV_jO_j^\top\|_F^2.
\end{aligned}
```

Thus conditional expected score differences compare squared errors in the
future observed covariance. They do not automatically compare errors in
unobserved covariance directions. This algebra uses the known noise correction
and centered independent future innovations; an arbitrary volatility proxy
does not automatically justify an arbitrary loss.

[Patton (2011), *Volatility forecast comparison using imperfect volatility
proxies*](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)
provides the scalar foundation for evaluating forecasts with conditionally
unbiased volatility proxies and explains why some losses can distort rankings.
The matrix score here is justified by the displayed moment identity, rather
than a claim to reproduce every setting or estimator in that paper. Proxy-noise
ranking robustness does not mean robustness to outliers or misspecification.

## Exposure-risk scores and observable dates

For a predeclared coefficient exposure $a$, use the plug-in forecast
$\widehat\theta_n=a^\top B_n a$. A future observed scalar proxy is available
only when $a$ lies in the numerical range of $O_j^\top$. On such a date,
solve $O_j^\top c_j=a$ and form

```math
\begin{aligned}
z_{j,a}&=(c_j^\top y_j)^2-c_j^\top N_jc_j,\\
\ell_{n,j,a}&=\widehat\theta_n^2-2\widehat\theta_n z_{j,a}.
\end{aligned}
```

The proxy has expectation $a^\top V_j a$, so the same omitted-constant score
argument applies. It can be negative, and so can its reduced score. Numerical
range residuals and eligible date counts are reported. A target's
identification from a pooled historical Gram matrix does not imply that its
realized future exposure is observed on a particular date.

Methods use identical eligible dates for each comparison. Unavailable scores
remain unavailable; they are not filled with a regularized covariance estimate.
Latent exposure errors can still be measured in simulation, but are reported
separately from observable scores. These are covariance plug-in forecasts;
the direct unbiased risk estimator from RQ2 is not an additional forecast policy.

## Predeclared simulation and chronology

Reuse RQ2's four covariance paths: constant, changing scalar scale, rotating
eigendirections, and a separately labelled abrupt-change stress. The five
observation designs are full noiseless coefficients, dense maturity values,
split maturity pairs, and split pairs supplemented by joint observations every
8 or 64 dates. See the [RQ2 protocol](research-question-2.md) for their exact
construction and normalized exposures.

| Control | Full study |
| --- | --- |
| Dimension and interval | Four coordinates; one week, Δ=1/52 year |
| Path length | 512 intervals |
| Independent replications | 20, with base seed 20261023 |
| Static fit | First 128 observations |
| Candidate local counts | 16, 32, 64, 128 |
| Validation origins | 128 and 192, each followed by a 16-date block |
| Test origins | 256, 272, 304, 448, each followed by a 16-date block |
| Forecast update rule | Fit at the origin; hold the matrix fixed throughout its block |
| Primary penalty | Relative ridge 0.001, recalibrated to each fitted window |
| Feasibility and numerical budget | PSD, trace cap 2, covariance-distance budget 0.0001 |
| Numerical range threshold | Relative spectral threshold 0.0000000001 |

The validation blocks are dates 129–144 and 193–208. The test blocks are
257–272, 273–288, 305–320 and 449–464. These are predeclared selections;
unscored dates can still become historical information for a later origin.
No change-point date or covariance-path parameters are supplied to a forecaster.

Generate each replication's innovation path and dense noise once, then reuse
the corresponding observations across policies and nested partial designs.
The noiseless full-coordinate design is an information control, separate from
equal-information partial-data comparisons.

## Measurements and acceptance

The report retains origin/date scores, validation candidate scores, selected
counts, actual penalties, numerical certificates, eligible exposure dates,
latent covariance and exposure errors, and configuration/source provenance.
Observable scores determine window selection; latent errors are evaluation-only
diagnostics.

For uncertainty summaries, first average the scored dates and origins within
each independent replication, separately for each covariance path, design,
policy and target. Compute Monte Carlo means and standard errors across those
20 replication averages. Method differences use paired replication averages.
Treating fitted origins or forecast dates as independent replications would
understate dependence from shared fitting data and covariance paths.

Acceptance checks chronology and held-fixed forecasts, observed-only tuning,
the longest-count tie rule, correct moment/noise identities, the exposure
range condition, common eligible dates, PSD/trace feasibility and requested
solver certificates. It also checks seeded reproducibility and retained
provenance. It does not require a particular policy to win, statistically
significant gains, or improvement at every origin.

No Gaussian likelihood score, confidence interval, new dynamics, curve-level
forecast or market evaluation is included. Observed scores describe the future
linear observations that were actually available; latent diagnostics do not
turn missing information into identified risk.

## Reproduce and continue

The full and smoke configurations are `experiments/configs/rq3.toml` and
`experiments/configs/rq3-smoke.toml`:

```sh
uv sync --locked --extra reference
uv run spde-covariance benchmark-rq3 experiments/configs/rq3-smoke.toml --output experiments/runs/rq3-smoke
uv run spde-covariance benchmark-rq3 experiments/configs/rq3.toml --output experiments/runs/rq3
```

Use a fresh output directory for each run and retain its report, figures,
replication measurements and manifest together. The [checked-in report](../experiments/results/rq3/report.md)
includes gzip-compressed row-level artifacts and bundle checksums.
[The roadmap](research-roadmap.md) describes how these
finite experiments precede later work on coherent spatial refinement, richer
volatility dynamics and empirical futures curves.
