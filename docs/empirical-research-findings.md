# What the expanded futures-risk study establishes

The expanded analysis explains the first baseline's failure more precisely:
**smooth curve factors omit important spread risk; adding joint residual risk
repairs much of that failure, but does not outperform direct covariance in this
sample**. The software now carries the investigation through constrained hedge
decisions. These are exploratory development findings on already inspected years.

Read the [six notebook roadmap](../README.md#research-roadmap),
[protocol](empirical-research-protocol.md),
[generated report](../experiments/results/shfe-research/report.md) and
[source manifest](../experiments/results/shfe-research/manifest.json).
The original [first-study findings](shfe-findings.md) remain intact.

## RQ1: targets, observations and representation

The data contain 969 sessions, 11,628 settlements and 60 actual copper contracts.
Fifteen observations have negative midmonth-coordinate tenors:
the coordinate is not expiry, and these rows remain in the audit. Origin curve
eligibility still uses the declared 30–365-day range. The data are retrospective
annual archives, with no reconstructed publication-vintage history.

The three currently held contracts have only **49–77 jointly observed adjacent
historical changes** even within a nominal 252-session window; the median is 62.
A long nominal window is not a long sample for a short-lived far contract.
Covariance comparisons must report actual pair counts, not only window labels.

Increasing dimension from three to four lowers median level RMSE from 48.58 to
44.22 CNY/tonne but increases median design conditioning from 70 to 1,611. The
three-mode representation accounts for 99.1% of outright second moment but
38.3% of spread and 11.4% of butterfly second moment. Residual shares and signed
cross terms complete the identity. These are sample decompositions, not
independent sources of risk or latent operator confidence intervals.

## RQ2: mechanism attribution and residual-risk repair

The matched-window experiment tests 16, 32, 63, 126 and 252 intervals and three nested
maturity masks. At 16 intervals the unchanged 20-pair direct minimum makes raw
baselines unavailable; factor comparisons are still reported. Information
stress results and matched transport comparisons remain separate.

At 252 intervals and all nodes, adjusted minus unadjusted QLIKE is −0.000142
for outright, +0.00530 for spread and +0.0323 for butterfly. The outright/spread
bootstrap ranges include zero; the butterfly difference favors unadjusted.
This supplies no forecasting advantage attributable to transport. Some masked
factor forecasts improve under the fixed estimator, but that does not make
less information intrinsically better or establish an SPDE advantage.

The revised joint model estimates PSD factor/residual second moments including
cross-covariances. Validation selects 63 intervals, calendar time and zero diagonal
shrinkage. The original source-C rule still selects 252 intervals. All choices
are frozen before the inspected evaluation period.

| Exposure | Factor-only QLIKE | Joint QLIKE | Direct QLIKE |
| --- | ---: | ---: | ---: |
| Outright |17.2728|17.2776|17.2774|
| Spread |15.1725|13.7103|13.6371|
| Butterfly |23.7601|15.0690|15.0665|

These factor/joint/direct figures use the common three-contract historical
support; direct outright/spread figures therefore differ slightly from the
original exposure-specific baseline. The original source-C control reproduces
the frozen forecasts. All 11 evaluated policies score 484 common outcomes and
produce PSD forecasts within roundoff.

Session normalization improves joint spread and butterfly QLIKE to 13.6280 and
14.9685 but worsens outright QLIKE to 17.4457. This is a fixed-policy sensitivity,
not a retrospectively selected winner. Physical calendar maturity transport
never changes. Pooling joint normalized innovations across unequal intervals
is a local empirical approximation, distinct from exact source-C integration.

## RQ3: scores, model risk and decisions

The frozen local SPDE-minus-direct spread comparison is +0.953 under primary
QLIKE, with block-bootstrap percentile range [0.188,1.839]. Under secondary
quadratic loss it is −0.391×10¹² CNY⁴, range [−0.665,−0.177]×10¹². Both differences
use identical future P&Ls. Their opposing rankings reflect different error
weighting; the primary criterion has not changed. Outright and butterfly
quadratic ranges include zero. Notebook04 retains block lengths 5/10/20/40,
annual/gap groups, calibration bins and post hoc influence diagnostics.

For one near contract and two hedge positions constrained to [−2,2], realized
settlement-P&L RMS is:

| Rule | RMS CNY |
| --- | ---: |
| Unhedged |3,487|
| Direct covariance |501|
| Joint factor/residual |533|
| Original source-C |624|
| Factor-only |685|

The joint hedge's squared-P&L difference versus direct averages+32,657 CNY²,
with descriptive block-bootstrap range [17,545,47,609]. Direct hedging has lower
realized risk on this sample. Large reductions relative to unhedged exposure
mainly reflect correlated contracts and are not trading alpha. Position-change
statistics omit executable turnover, roll costs, fills, margin and capacity.

## Verification and next evidence gate

Numerical source was committed before the full run. The manifest records a clean
source revision, source/configuration/lock hashes, archive identities, actual
thread settings and environment. The runner checks every frozen scored proxy
against the supplied panel. Public notebooks use only checksummed aggregates;
raw files, daily curves and individual outcomes remain local.

The complete suite took 228.6 seconds excluding archive parsing in the recorded
single-thread environment. This is a measured run, not a portable speed claim.
Unit and integration checks cover chronology, future-price isolation, loss
identities, missing outcomes, cross terms, PSD feasibility, independent convex
hedge references and aggregate exports.

The next scientific gate is to freeze a revised policy and evaluate an
uninspected period. Later work can pool residual risks across contract ages,
audit point-in-time security metadata and introduce an execution-aware protocol.
Do not select a model from these inspected outcomes and describe the same
period as a fresh confirmation backtest. Infinite-dimensional statistical
guarantees and physical forcing identification remain separate research tasks.
