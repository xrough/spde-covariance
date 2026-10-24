# Completion: what the continuation establishes

These are numerical conclusions for the declared fixed transport/source family,
observation designs and finite references. The [ablation](../experiments/results/completion-ablation/report.md),
[refinement](../experiments/results/completion-refinement/report.md) and
[forecast](../experiments/results/completion-forecast/report.md) reports retain
configurations, source hashes, solver audits, all cases and independent
replication checkpoints. They extend the [earlier findings](completion-findings.md).
No new estimator, infinite-dimensional statistical theorem or market validation
is claimed.

## What causes the improvement?

The ablation uses 20 paired replications, N=512, actual grid Q=46, known noise
variance 0.001 and the two fixed source models. Each row below changes PSD
coherence alone, holding relative penalty zero, weighted floor zero and rank-two
completion fixed. Values are differences in complete weighted kernel error:
coherent minus pairwise, with replication-paired MCSE.

| Source | Fragments | Change in error ± MCSE |
|---|---:|---:|
| Power law | 3 | +0.00057 ± 0.00148 |
| Power law | 9 | −0.03587 ± 0.00582 |
| Rotated | 3 | +0.00072 ± 0.00246 |
| Rotated | 9 | −0.04570 ± 0.00768 |

The nine-fragment cases show gains; the three-fragment cases do not show a
resolved gain. The [matched-factor table](../experiments/results/completion-factor-contrasts/paired.csv)
also isolates shrinkage, positive floors, hard truncation and posthoc projection.
Reproduce it from the committed postprocessor:

```sh
uv run python scripts/completion_contrasts.py experiments/results/completion-ablation/estimates.csv.gz --output /tmp/completion-contrasts-new
```

Raw full-support completion is unstable: its four mean complete errors are
2.094, 40.717, 0.780 and 314.799 (large MCSE/outliers). Corresponding zero-penalty,
zero-floor coherent full-support errors are 0.2272, 0.2795, 0.3076 and 0.3691.
Coherence changes the estimated partial kernel; this is not free identification
of its missing entries.

Every rank-two policy remains indefinite in all 80 cases, including after
coherent partial fitting. Full-support coherent completion passes the numerical PSD check. Hard
truncation therefore needs a separate feasibility check; the upstream PSD
constraint alone does not protect it. Observed scores are exactly identical
for support/rank-two pairs with the same fitted partial, even when missing-block
errors differ dramatically.

At this anchored grid, the constant and resolution floors are exactly equal.
Positive-versus-zero floor changes in complete error are at most about
`2e-5`. Relative shrinkage 0.001 shows no resolved incremental complete-error
gain. The predeclared stability guard rejects no case: it selects exactly the
same candidates as observed validation in all 80 cases. These findings do not
establish an optimal floor, penalty or guard. The tiny floor contrasts are also
below the `1e-4` observed-partial optimizer-distance budget, before completion
amplification; a small paired MCSE does not numerically certify their benefit.

All 1,520 ablation estimates are available. All 320 distinct main coherent
partial solves meet the declared optimizer-error budget. Split/candidate
diagnostics are retained separately. Posthoc PSD projection can improve global
error while worsening observed scores or particular risks; these criteria are
not interchangeable. Long-spread operational scores remain unavailable.

## What numerical refinement means

The empirical refinement keeps the 512-mode source and parent observations
fixed while changing sample size and grid. The deterministic checks separately
vary forcing, output representation and the physical completion grid.

For resolution-floor coherent completion at base grid 81, complete weighted
kernel error falls as training size increases (mean ± MCSE, 20 replications):

| Source | Fragments | N=128 | N=2048 |
|---|---:|---:|---:|
| Power law | 3 | 0.3890 ± 0.0355 | 0.1130 ± 0.0112 |
| Power law | 9 | 0.5951 ± 0.0366 | 0.1391 ± 0.0079 |
| Rotated | 3 | 0.5970 ± 0.0499 | 0.1598 ± 0.0139 |
| Rotated | 9 | 0.8171 ± 0.0498 | 0.2063 ± 0.0098 |

Increasing the grid alone does not uniformly improve estimation. For example,
at N=2048 with nine fragments, power-law error rises from 0.1295 at base grid 21
to 0.1391 at 81; rotated error rises from 0.1910 to 0.2063. Reported MCSE and
paired tables accompany these means. All 3,040 estimates are available, all
1,200 distinct main coherent fits meet the budget and the maximum iteration
count is 21,890. The 720 raw rank-two outputs are indefinite; coherent and
separately PSD-projected outputs pass their numerical PSD checks.

The two floor policies nearly overlap. Their measured error contrasts are
smaller than the observed-partial optimizer budget; no numerically certified
floor-policy accuracy advantage follows, even when paired MCSE is small.

On common anchor nodes, the forcing error at 64 modes is approximately
`8.1e-9`/`8.3e-9` for the two models and falls to `4.7e-11` at 256 modes,
relative to the 512-mode reference. These small **kernel** errors do not state
that the full H-operator tail is equally small. Fixed-source tail allowances
are saved separately, and the reference itself still has an omitted tail.
Canonical grid discrepancies relative to the 321-point reference are below
`1.2e-6` in these comparisons. Agreement is not a proof of continuum convergence.

The protected-population first-join perturbation bound is available in all
40 declared cases. Bounds range from 0.704 to 93,890 while measured changes range
from `5.7e-12` to `1.6e-7`; the bounds are numerically uninformative here.
The measured change, separator gap and bound are all saved;
the bound does not cover arbitrary recursive completion, sampling error or
changing numerical support. Structured joint-limit theory remains unfinished.

## Does a larger forcing class change compatibility?

Yes in the independent control: with output J=2 and interval one, a target
generated by a three-mode PSD source has distance about 0.03931458 from the
matched two-mode source class at trace cap two. A declared three-mode witness
and its four-mode embedding attain it to rounding. An independent CVXPY check
confirms the separation. Thus output dimension and forcing support must differ
in a meaningful source-class diagnostic.

For the main population targets, all 72 reported finite-class lower bounds are
zero at the declared numerical tolerance; four solves remain unconverged and
retain their distance intervals. The maximum order-24 versus order-48 numerical
map allowance is below `1.5e-15`. This compares two numerical maps, not either
map to an exact integral. These cases provide no positive finite-source
exclusion. They do not identify the generating source, establish infinite-class
membership, or yield a sampling-calibrated statistical test.

Nested source witnesses, finite-reference integration checks and generating-source
tail bounds have separate meanings; [the source note](source-classes.md) states
those scopes explicitly.

## Do historical estimates help on subsequent observations?

The forecast study completes 20 paired replications with 6,772 unique fits and
92,160 future scoring rows. Every fit is available. All 2,264 coherent fits meet
the numerical budget; the maximum is 36,940 iterations and a partial-distance
bound below `1e-4`. Coherent completed forecasts pass the numerical PSD check.

These forecasts predict each future weekly normalized innovation covariance,
not an accumulated multiweek curve return. Their generating covariances come
from deterministic time-varying SPDE sources. The fitted completion constraints
are grid PSD/overlap constraints, not membership in a transport-source class;
the forecasts do not identify physical diffusion covariance.

For coherent completion, the primary observed-score differences are below.
Each entry is a replication-paired mean ± MCSE, with lower values preferable.
Compare rules **within** each scenario and cover, not raw score magnitudes
between different fragment designs.

| Source path | Fragments | Fixed 128 minus static | Selected window minus fixed 128 |
|---|---:|---:|---:|
| Constant | 3 | −4.41 ± 3.56 | +15.51 ± 8.22 |
| Constant | 9 | −5.32 ± 2.47 | +14.08 ± 4.48 |
| Scale | 3 | −8.66 ± 4.39 | +10.32 ± 6.15 |
| Scale | 9 | −10.31 ± 3.98 | +12.88 ± 5.37 |
| Rotating | 3 | −21.02 ± 9.01 | +10.18 ± 6.86 |
| Rotating | 9 | −19.39 ± 6.26 | +8.46 ± 3.55 |
| Abrupt | 3 | −28.29 ± 7.08 | +2.98 ± 3.58 |
| Abrupt | 9 | −31.44 ± 5.25 | +6.56 ± 3.85 |

Updating a fixed historical window helps relative to the static rule in these
changing-source examples. Selected-window point estimates are worse than the
fixed rule in all eight cells; these runs establish no selection advantage or
universal forecasting superiority. Some differences remain noisy. The report
also separates latent weighted kernel error, risk MSE and operational risk scores.

Selected-versus-fixed risk MSE is higher in all 24 scenario/design/exposure
cells; this is a separate criterion from the kernel norm. Raw rank-two
forecasts produce negative variance predictions, which are retained in the
audit. Across the 160 coherent scenario/design selections, windows
16/32/64/128 are chosen 19/22/45/74 times. These are selection counts across
cells, not 160 independent Monte Carlo replications.

Constant and abrupt scenarios select exactly the same windows within each
method/replication because their validation information is identical and
entirely before the change. Improvements after the change come from subsequent
historical observations entering refits, not anticipation of the change point.
The long-spread score is unavailable throughout. Latent errors there can be
measured in simulation but cannot supply an observed risk proxy. MCSE uses
20 independent replications, not the 92,160 dependent scoring rows.

## What remains a research question

The numerical workflow now tests regularization mechanisms, fixed-model
refinement, larger forcing supports and strictly subsequent risk prediction.
The next substantive steps concern uniform stability/regularity conditions,
functional identification with representation error and informative source-class
bounds as output and forcing resolutions increase together. Sparse noisy
price-level reconstruction and empirical futures applications remain separate
extensions requiring observation models.
