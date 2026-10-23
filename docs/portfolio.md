# Portfolio and CV wording

## Recommended CV bullet

Developed and benchmarked a transport-SPDE covariance estimation toolkit with
PSD-constrained optimization, observability diagnostics, and reproducible
numerical validation.

This describes the implemented work without claiming a new statistical theorem
or demonstrated market forecasting gains.

## Supporting evidence

The [Stage 1 report](../experiments/results/stage1/report.md) records 20 paired
synthetic replications in a four-dimensional exact transport model, with
256/128/128 training/validation/evaluation intervals. Tuning uses observed
validation outer products; latent covariance is used only for evaluation.
Entries below are Monte Carlo means and standard errors, not confidence intervals.

| Evidence | Measured result | Interpretation |
| --- | --- | --- |
| Full-coordinate recovery control, constrained ridge | Relative HS error `0.1201 ± 0.0097` | Recovery under well-observed synthetic conditions. |
| Full-coordinate empirical baseline | Relative HS error `0.1189 ± 0.0098` | The constrained estimator does not outperform every baseline. |
| Dense / sparse observations, constrained ridge | Relative HS errors `0.3133 ± 0.0411` / `0.8068 ± 0.1228` | Observation geometry matters in these cases. |
| Missing-pair observation design | Numerical Gram rank `6/10` | Detects absent contemporaneous covariance directions. |
| Selected/refitted covariance fits and transport ablations | `80` covariance estimates + `80` unadjusted-increment ablations meet the `1e-4` numerical budget; maximum bound `2.6923e-5` | Distance to the constrained optimum, not statistical error to truth. The ablations are labelled transport controls. |
| Independent CVXPY/SCS checks | Maximum matrix discrepancy `2.4637e-9` across four checked cases | Independent verification of the same constrained objective. |
| Automated verification of the code-focused tree | `159` passing tests | Transport, extraction, fitting, baselines, diagnostics, and reproducibility. |

The [manifest](../experiments/results/stage1/manifest.json) records clean public
source revision `bf64c1d01ba5ea34394df2e33c9ef67d43bf864d`, package version
`0.3.0`, dependency versions, source hashes, and solver settings used for the
checked-in run. Numerical results must be read with that provenance.

## Optional quantified wording

Validated 80 selected/refitted PSD-constrained covariance estimates and
80 transport-adjustment ablations against a `1e-4` numerical-error budget,
with independent convex-reference checks and 159 automated software checks.

Use this with the report link and its synthetic scope. The numerical budget
must not be presented as a statistical estimation-error bound. Full, dense,
and sparse observations are separate information settings. These results do
not establish uniform superiority of an estimator.

## Technical discussion points

- Exact shift-invariant exponential transport provides an analytic control;
  spline propagation exposes projection and boundary errors.
- The pipeline distinguishes diffusion covariance from semigroup-adjusted
  innovation covariance, preserving basis and interval provenance.
- Positive covariance constraints have numerical certificates; observability
  diagnostics disclose unsupported covariance directions.
- Comparisons retain observed-only tuning, paired replicates, and outcomes
  where empirical estimation performs better.

The project demonstrates quantitative research and numerical/software reliability.
The evidence concerns synthetic covariance estimation. Futures-curve calibration,
prediction, market evaluation, and outlier robustness are not validated outcomes.
Timing measurements apply to the environment in the report, not a
hardware-independent speed claim.
