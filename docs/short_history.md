# Short-history copper risk transfer

**Question:** given a currently quoted copper contract and only a few permitted
own price changes, can observations of other contracts supply useful next-session
risk information? Compare learned maturity-dependent factors with the existing
transport-SPDE source model, using the same residual-transfer adapter.

[Executed notebook 07](../notebooks/07_short_history_risk_transfer.ipynb) ·
[Aggregate results](../experiments/results/short-history/report.md) ·
[Research roadmap](research-roadmap.md)

This is an exploratory information-restriction study on the inspected daily SHFE
copper sample. It does not simulate actual listing decisions, forecast price
direction, price derivatives or establish an executable hedge backtest. The
existing first-study and research-suite evidence remain unchanged. Covariance
completion and intraday data are separate studies.

## Information restriction and chronology

- Initial history: 2022. Development validation: 2023. Exploratory evaluation:
  2024–2025. Every setting and sensitivity is fixed; evaluation selects no winner.
- Schedule origins every five trading sessions, anchored independently at the
  first eligible validation/evaluation session. Only origins whose next outcome
  is inside the corresponding phase are scored.
- Select three current eligible actual contracts nearest 90, 180 and 270 days.
  Each is the target in turn. The other two are hedge legs, ordered by maturity
  distance from the target, breaking equal-distance ties toward shorter maturity.
- Retain at most 252 intervals. History budgets are 0, 5, 20 and all available
  target changes. The current target settlement and metadata are always visible.
  A finite budget requires its contiguous suffix of h+1 target price levels;
  absent required levels make the episode unavailable.
- Remove all older target rows before reconstruction, factor fitting, residual
  estimation, scaling or cache construction. Forecast issuers receive only this
  masked panel, current metadata and the known next-session calendar date.
- Retain the original session calendar. Returns use adjacent sessions and the
  same actual contract ID. No filling, rolling-series differencing or gap bridging
  is permitted. Future settlements enter a separate scoring function only.

Training donors require positive settlement, volume and open interest and
midmonth maturity between 30 and 365 days at both endpoints. The direct
same-contract baseline retains all positive observed settlements regardless of
historical donor liquidity/range, matching the earlier direct baseline. These
eligibility differences are explicit; both respect the same target-history mask.
The contract-month 15th is a maturity coordinate, not actual expiry or delivery.

## Learned dynamic factors and shared residual risk

For allowed same-contract changes, use ACT/365.25 normalization:

```math
x_{i,s}=\frac{F_{i,s}-F_{i,s-1}}{\sqrt{\Delta_s}}
       =b(\tau_{i,s})^\top z_s+u_{i,s}.
```

Six cubic B-splines on 30–365 days, with interior knots at 120 and 240 days,
are orthonormal under the uniform probability measure on that interval.
Cross-sectional ridge is 10⁻⁶ times the design Gram norm. The leading three
eigenvectors of the weighted **uncentered** coefficient second moment define
the learned loading functions. This is a return-factor model with evolving
second moments, not a latent price-level state model or GARCH specification.
Loadings are refitted from the masked prefix; conditioning, effective counts
and retained/discarded eigenvalue gaps are reported.

The transport comparator reconstructs three Laguerre price coefficients and
forms exact finite transport-adjusted innovations. Source covariance C is fit
through its interval-specific covariance integration maps. Uniform weighting
preserves the original estimator; EWMA interval weights apply consistently to
drift fitting, objectives, Gram matrices, scaling, trace caps and certificates.
Positive relative regularization is 10⁻³, trace cap is 100 times training scale,
and the source covariance numerical-error budget is 10⁻⁵.

Both model families fit remaining cross-sectional changes using four
piecewise-linear maturity features at 30, 140, 250 and 365 days. Let their joint
weighted second moment with the main factors be

```math
M=\begin{pmatrix}H&G\\G^\top&R\end{pmatrix},\qquad
A=G^\top H^\dagger,\qquad \Omega=R-G^\top H^\dagger G.
```

For main and residual evaluation matrices B and Ψ, issue

```math
\widehat\Sigma_t=\Delta_{t+1}
\left[(B+\Psi A)H_{\mathrm{forecast}}(B+\Psi A)^\top
+\Psi\Omega\Psi^\top+D_t\right].
```

DFM uses its factor EWMA moment. SPDE uses integrated source covariance divided
by the future interval length. A 10⁻¹⁰ relative pseudoinverse cutoff is explicit;
material support incompatibility or indefiniteness fails rather than silently
projecting the forecast. Roundoff corrections and component sums are recorded.
The cross correction is signed and is not an independent positive risk source.

D is a nonnegative maturity-dependent diagonal correction from leave-one-donor-out
predictive errors. The donor is excluded from that interval's factor extraction
and residual reconstruction, with loading shapes frozen. This correction can
include estimation error; it is not identified fresh observation noise or physical
forcing. Four correlated residual features plus a diagonal remainder remain a
restriction. Both principal models include no-diagonal-correction sensitivities.
Known next-session maturities falling below 30 days use flat endpoint extension
for the learned loadings and residual features; Laguerre transport remains exact
on its half-line domain.

## Policies, scoring and uncertainty

| Policy | Role |
| --- | --- |
| Learned factors + shared residual | Proposed risk-transfer model |
| EWMA transport + shared residual | Principal structured comparator |
| Original uniform source C; EWMA source C | Original policy and recency ablation |
| Uniform learned factors + residual | Dynamic-weighting ablation |
| Learned factors only | Residual-risk ablation |
| Same-contract direct covariance | Unavailable below 20 complete changes |
| Maturity pool blended with short history | Simple transfer baseline |

The pooled baseline selects visible historical maturity slots, forms differences
within actual IDs, and reorders the slots into the current target coordinates.
Blend its moment with the short-history triad moment using n/(n+20), where n is
the available complete count. At n=0 the pooled estimate supplies the forecast.
The primary half-life is 63 sessions; 21 and 126 are prespecified sensitivities.

Forecasts target uncentered price-change second moments. A variance interpretation
requires negligible conditional means. Convert to CNY P&L second moments using
the five-tonne multiplier squared. Score target outright, target/nearest spread
and original near/middle/far butterfly under QLIKE and quadratic risk loss;
retain the existing CNY² floor 25 for QLIKE. Score the full triad using the
quadratic matrix moment criterion. Choose hedges from past forecasts, with one
target contract and the two hedge positions independently bounded to [−2,2].

Average matched target differences within origin before pooling dates. Repeated
contracts, origins and budgets are dependent. The circular block bootstrap uses
one joint set of origin resamples across methods, targets and budgets: primary
block four scheduled origins (about 20 sessions), sensitivities 1, 2 and 8,
1,000 draws, seed 20261027. These are descriptive ranges on inspected data,
not IID Monte Carlo uncertainty or untouched confirmation. A favorable observed
score is not identification of an unobserved historical path. Compare principal
models as complete policies; this does not identify a causal transport effect.

## Reproduction, evidence and acceptance

No market files are needed for the offline control:

```sh
uv run python examples/short_history_pipeline.py
```

Run the full local study, exporting only aggregate evidence:

```sh
uv run spde-covariance benchmark-short-history experiments/configs/short-history.toml \
  --data data/local/shfe --output data/local/shfe/runs/short-history-v1 \
  --export experiments/results/short-history
uv run python scripts/render_short_history.py experiments/results/short-history
uv run python scripts/check_notebook.py notebooks/07_short_history_risk_transfer.ipynb
```

Output directories must be new. Commit numerical source before a full run;
retain revision, source hashes, archive identities, settings, lockfile,
environment/thread settings and runtime. Saved public files are checksummed
summary, paired comparisons, diagnostics, protocol and manifest only. The
renderer creates a separate presentation manifest, report and three plain
Matplotlib figures. It never refits or changes numerical evidence.

Raw panels, individual contract IDs in issued forecasts, realized P&Ls and hedge
positions stay local in the ignored run. Public notebook 07 redraws its figures
from verified aggregates and requires no archive access or Jupyter installation.

Tests cover hidden/future-price invariance, exact history budgets, original
session adjacency, same-ID differencing, units, uniform-source parity,
independent convex references, Schur support/PSD checks, factor rotation/sign
invariance, cutoff/gap diagnostics, leave-one-donor-out correction and bounded
cache/order invariance. Synthetic execution and saved-evidence notebook execution
run on Linux/macOS. No model is required to outperform another.
