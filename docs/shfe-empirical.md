# First empirical study: SHFE copper covariance and risk

This study connects the numerical laboratory to historical futures settlements.
It asks whether a small transport model, fitted using past curves, gives useful
next-session outright and calendar-spread risk estimates compared with direct
same-contract covariance estimates. It is a calibration and risk-evaluation
experiment. It does not establish profitable trading, physical source recovery,
or validity of an infinite-dimensional statistical model.

The [executed notebook](../notebooks/shfe_empirical.ipynb) explains the results.
The [aggregate report](../experiments/results/shfe-cu/report.md) retains the
numbers, configuration and source manifest. Public execution of the notebook
uses aggregate evidence; rerunning market calibration requires local archives.

## Data and coordinates

Use the official SHFE annual futures archives for 2022–2025, obtained through
the [historical-download catalogue](https://www.shfe.com.cn/reports/tradedata/datadownload/download.json).
The files contain monthly XLS/XLSX workbooks. The parser selects actual copper
contract IDs (`cuYYMM`), checks headers and dates, fails duplicate date/contract
keys, and preserves missing fields. It never constructs a rolled continuous
contract, fills missing settlements or substitutes close for settlement.
SHA-256 hashes identify the downloaded versions, including archive members.

The [2019 SHFE contract manual](https://tsite.shfe.com.cn/upload/20190919/1568892145195.pdf)
specifies CNY per tonne and five tonnes per contract. Model maturity is the
15th of the contract month, a deterministic coordinate obtainable from its ID.
It is **not** a claim about holiday-adjusted expiry or a physical delivery-average
pricing rule. Contract-month midpoint sensitivity and alternative delivery maps
remain future work. The observed exchange-session calendar is treated as known.
Historical archives are retrospective versions; original publication vintages
and subsequent revisions have not been reconstructed. Forecast origins are
after the day's settlements have become available.

At each origin, curve reconstruction uses positive settlements with positive
volume and open interest and 30–365 calendar days of model maturity. These
filters use current information. Portfolio selection uses that same origin
filter. Evaluation then looks up those exact IDs in the next settlement report,
without filtering on next-session liquidity. Missing outcomes remain unavailable.

## A declared finite transport model

Use the first three orthonormal integrated Laguerre functions on the half-line,
including the constant function. The metric is

```math
\langle f,g\rangle_H=f(0)g(0)+\int_0^\infty e^x f'(x)g'(x)\,dx.
```

This finite subspace is exactly shift invariant. With evaluation row $E(x)$
and transport matrix $M_\Delta$, $E(x)M_\Delta=E(x+\Delta)$.
Each day's coefficients minimize a cross-sectional squared-error objective
with a small shape penalty; the constant level is unpenalized. The penalty is
relative to the design Gram norm. Reconstruction errors, condition numbers and
the smoothing setting are recorded. Dimensions two and four and stronger
smoothing are diagnostic sensitivities, not parameters chosen on test results.

The local coefficient model is

```math
da_t=Aa_t\,dt+b\,dt+L\,dW_t,\qquad C=LL^\top.
```

For each actual ACT/365.25 interval,

```math
r_i=\widehat a_i-M_{\Delta_i}\widehat a_{i-1}
       -\left(\int_0^{\Delta_i}M_s\,ds\right)b,
\quad V_{\Delta_i}(C)=\frac1{\Delta_i}
                  \int_0^{\Delta_i}M_sCM_s^\top\,ds.
```

The source fitter compares $r_ir_i^\top/\Delta_i$ to its **own** interval map
$V_{\Delta_i}(C)$, under a common covariance within the trailing window. It
does not pool unequal intervals into a claimed common innovation covariance.
The full PSD source matrix is fitted with positive ridge and an explicit
training-derived trace cap. The solver records a convex objective-gap bound
and its implied error to the constrained optimizer. They concern numerical
optimization, not statistical accuracy or correct specification.

The main experiment sets $b=0$. The reusable calibration API also supports a
training-only constant drift. No clean latent coefficient observation or known
fresh measurement noise is asserted: the fitted source absorbs reconstruction,
settlement microstructure and model error. It is a **model-dependent effective
covariance**, not an identified market forcing operator. In particular, this
extends the clean-coefficient examples through an explicit approximation.

## Risk targets and comparisons

At each origin choose distinct contracts nearest 90, 180 and 270 days.
Portfolios contain one near contract; a near-minus-middle spread; and a
near-minus-two-middle-plus-far butterfly. Positions remain fixed until the next
exchange session. Their raw P&L uses settlement differences and the five-tonne
multiplier; there are no roll returns or transaction-cost adjustments.

For the origin's fixed contract dates, set $x_{\rm next}=T-t_{\rm next}$.

```math
\widehat\Sigma_{\rm next}
 = E(x_{\rm next})\left(\int_0^{\Delta_{\rm next}}
                 M_s\widehat C M_s^\top ds\right)E(x_{\rm next})^\top,
\qquad \widehat v=25\,q^\top\widehat\Sigma_{\rm next}q.
```

This is the next-interval covariance, without a second annualization. Exact
transport cancels the deterministic roll-down for the same contract. A mean
forecast is anchored to its observed origin settlement, avoiding artificial
mean reversion from cross-sectional reconstruction residuals.

Compare local SPDE, fixed-126-interval SPDE, static-2022 SPDE, ordinary
unadjusted coefficient increments, direct same-contract empirical covariance,
and fixed 10% diagonal shrinkage. The unadjusted ablation uses $A=0$ throughout.
Direct methods require at least 20 jointly observed adjacent intervals for
the particular exposure's contracts and select their own validation windows. They
use valid positive historical settlements without the cross-sectional tenor filter. Their effective
sample count may be smaller because the far contract was listed more recently.
Availability and paired common-origin comparisons are reported explicitly.

## Frozen chronology and scores

`experiments/configs/shfe-cu.toml` fixes the protocol before the full run:

| Phase | Data | Use |
| --- | --- | --- |
| Initial fit | 2022 | Static covariance and initial history |
| Validation | 2023, after sufficient history | Choose 63, 126 or 252 trailing intervals |
| Test | 2024–2025 | Refit with frozen settings; next-session risk evaluation |

Validation averages QLIKE over the same available origins and three portfolios.
SPDE, direct covariance and diagonal shrinkage independently select their own
window using 2023 observations. Exact ties prefer the longer window. Every refit uses observations through its
origin; later test fits may use earlier test outcomes. No window is reselected
using test performance. The entire test block is not an independent holdout
for each later rolling fit; this is a chronological rolling-origin evaluation.

For observed squared P&L $z$, the score is

```math
\ell(v,z)=\log v+z/v.
```

For positive $v$, its conditional expectation is minimized at
$v=\mathbb E[z\mid\mathcal F_t]$. With zero conditional mean this is the
variance target; otherwise it is a second moment. A fixed floor of 25 CNY²
prevents division by zero; floor hits are counted. QLIKE is used as a proxy
loss, **not** a Gaussian-density claim or tail-risk guarantee. The quadratic
proxy score $v^2-2vz$ is also retained in aggregate output.
See [Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)
for evaluation with imperfect volatility proxies.

Report realized/predicted second-moment ratios, RMS risk in CNY, yearly and
pooled scores, and paired QLIKE differences on common origins. Circular
moving-block bootstrap summaries preserve 20-origin blocks, with 1,000 draws.
They describe sensitivity to temporal dependence; two years of one market
do not supply independent Monte Carlo replications or structural confidence
intervals. Standardized P&L histograms test the usefulness of Gaussian intuition
without claiming VaR coverage. More horizon, block-length and regime studies
can follow this first experiment.

## Local reproduction and public evidence

```sh
uv sync --locked --extra empirical --extra reference
uv run python scripts/download_shfe.py --years 2022 2023 2024 2025 \
  --cache-dir data/local/shfe
uv run spde-covariance empirical-shfe experiments/configs/shfe-cu.toml \
  --data data/local/shfe --output data/local/shfe/runs/first-study
uv run python scripts/check_notebook.py notebooks/shfe_empirical.ipynb
```

Raw archives, cleaned panels, fitted daily curves, individual P&Ls and forecast
audits stay under `/data/local/`, which Git ignores. The public evidence export
copies only aggregate summaries, hashes, report and aggregate Matplotlib figures.
It does not copy price or return panels. The repository's MIT license covers
our code; it does not relicense SHFE market data. Public website access is not
a general raw-data redistribution license.

Next improvements are settlement revision/availability auditing, reconstruction
and maturity-coordinate sensitivity, intraday or business-time clocks, richer
conditional covariance dynamics and multi-market replication. Sparse noisy-price
innovation recovery and infinite-dimensional guarantees remain separate work.
