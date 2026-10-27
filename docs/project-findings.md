# Current findings: from covariance observations to futures risk

**Main question:** when do partial observations and a smooth transport-SPDE
representation provide enough information to estimate the risks of interest?
The project investigates identification, estimation and subsequent evaluation
as separate questions. Its strongest finding is that a good curve fit or a PSD
covariance matrix can still omit important spread risk.

Start with the executed [project overview notebook](../notebooks/00_project_findings.ipynb).
It connects the controlled numerical experiments to the empirical findings below.

![Empirical curve representation and selected hedge outcomes](../assets/current-findings.png)

## RQ1: which risks do the observations identify?

**Synthetic result: more observations of the same fragments do not identify
never-observed covariance.** A three-node Gaussian example observes pairs
{0,1} and {1,2}, but never {0,2}. Two generating worlds have identical observed
laws and different missing covariance. Even with exact native covariance, the
normalized endpoint-spread variance can take any value in **[0, 1.68]** across
compatible PSD completions. This is an identification range, not a confidence
interval. Canonical completion selects a structural convention within that range.

The larger completion study also constructs SPDE worlds with identical native
observations. Fitting and validation therefore agree across worlds, although
their missing covariance differs. Native validation cannot establish the
completion assumption; additional bridge observations can test it.
See the [synthetic findings notebook](../notebooks/current_findings.ipynb) and
[statistical completion findings](statistical-completion-findings.md).

**Empirical result: fitting price levels well does not preserve spread risk.**
For the three-factor SHFE copper representation, fitted changes account for
**99.1% of outright second moment, 38.3% of spread and 11.4% of butterfly**.
Residual changes and signed cross terms complete the decomposition; the parts
are not independent sources of risk. Increasing dimension from three to four
lowers median level RMSE from 48.58 to 44.22 CNY/tonne while increasing median
design conditioning from 70 to 1,611.

The nominal 252-session window contains only **49–77 jointly observed adjacent
changes** for the currently held three contracts, with median 62. Contract life
and maturity coverage matter as much as the nominal window length.
See notebooks [01: data](../notebooks/01_data_and_risk_targets.ipynb) and
[02: representation](../notebooks/02_representation_and_identification.ipynb).

## RQ2: what improves estimation, and under which assumptions?

**Completion estimation improves with sample size under the declared model.**
In the correctly specified canonical Gaussian control, the likelihood policy's
mean weighted kernel error falls from **0.07484 ± 0.01127** at N=128 to
**0.02197 ± 0.00422** at N=2048. These are means ± MCSE over 20 paired replications.
At N=2048 it improves complete-kernel error over the moment policy, but the
missing-region error difference, −0.00160 ± 0.00203, does not resolve an advantage.
Different penalties make this a comparison of fitting policies, not likelihood
alone. Sampling uncertainty and permanent completion ambiguity remain separate.

**Joint residual modeling repairs much of the empirical factor-model failure.**
The revised estimator retains factor/residual cross-covariances and produces a
PSD forecast. Validation selects 63 intervals, calendar time and zero diagonal
shrinkage before evaluation. On common historical support, its curvature-risk
scores approach the direct covariance baseline:

| Exposure | Factor-only QLIKE | Joint QLIKE | Direct QLIKE |
| --- | ---: | ---: | ---: |
| Outright | 17.2728 | 17.2776 | 17.2774 |
| Spread | 15.1725 | 13.7103 | 13.6371 |
| Butterfly | 23.7601 | 15.0690 | 15.0665 |

Lower is better within each exposure. All three use the same future observations;
their common historical support differs from the frozen first study's
exposure-specific direct baseline. Matched adjusted/unadjusted comparisons
establish **no forecasting advantage attributable to transport** in this sample.
Clock sensitivities improve some exposures and worsen others.
See notebooks [03: attribution](../notebooks/03_performance_attribution.ipynb) and
[05: residual risk](../notebooks/05_joint_residual_risk_and_clocks.ipynb).

## RQ3: do the estimates help subsequent risk decisions?

**Direct covariance remains the stronger empirical hedge baseline.** For one
near contract hedged with two positions constrained to [−2, 2], realized
same-contract settlement-P&L RMS is:

| Rule | RMS, CNY |
| --- | ---: |
| Unhedged | 3,487 |
| Direct covariance | 501 |
| Joint factor/residual | 533 |
| Original source-C | 624 |
| Factor-only | 685 |

Joint minus direct squared P&L averages +32,657 CNY², with descriptive
block-bootstrap percentile range [17,545, 47,609]. Large reductions relative
to unhedged exposure mainly reflect correlated contracts, not trading alpha.
No fills, roll costs, margin, capacity or profitability backtest is implemented.

**Score choice changes the answer.** In the frozen first study, local-SPDE minus
direct spread loss is +0.953 under primary QLIKE, range [0.188, 1.839], but
−0.391 × 10¹² CNY⁴ under secondary quadratic loss, range [−0.665, −0.177] × 10¹².
The same future P&Ls receive different error weightings. The primary criterion
has not changed. See notebooks [04: forecast evaluation](../notebooks/04_forecast_scores_and_model_risk.ipynb)
and [06: hedging](../notebooks/06_hedging_decisions.ipynb).

## What the infinite-rank benchmark establishes

One fixed trace-class infinite-rank transport model underlies the resolution
ladder. It distinguishes source covariance C, normalized finite-interval
innovation covariance V, state covariance, projection leakage and sampling error.
At forcing reference M=256, refining weekly to quarter-week sampling reduces
the population HS norm of V−C from **0.002579 to 0.000655**. At weekly sampling,
increasing projection dimension J=8 to 64 reduces covariance projection error
from **0.001720 to 0.00001097**, against the fixed finite reference.

Analytic tail bounds connect that reference to the declared infinite model;
independent numerical checks audit transport, integration and optimization.
These are implementation and approximation results, not an operator recovery
theorem or identification of physical market forcing. See the
[benchmark and numerical scope](infinite-transport.md).

## Reading path, evidence and limits

The [six-notebook roadmap](../README.md#research-roadmap) develops the current
empirical suite. Its [findings](empirical-research-findings.md),
[protocol](empirical-research-protocol.md) and [generated report](../experiments/results/shfe-research/report.md)
retain chronology, aggregate evidence and provenance. The older
[first-study notebook](../notebooks/shfe_empirical.ipynb) is a frozen benchmark,
not the latest executive summary; its unfavorable results remain intact.

The empirical data contain 969 SHFE copper sessions and 11,628 settlements from
2022–2025, with 484 evaluation origins. Raw data and individual P&Ls stay local;
public notebooks execute from checksummed aggregates. The inspected 2024–2025
period has informed model development, so these findings are exploratory even
though issuance uses past data only. A frozen revised policy needs an uninspected
confirmation period. Established estimators are implemented; methodological
novelty, SPDE superiority and infinite-dimensional statistical guarantees remain
unestablished. The [research roadmap](research-roadmap.md) separates those later
questions from the numerical and empirical evidence already available.
