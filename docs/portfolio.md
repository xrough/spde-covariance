# Portfolio evidence and CV wording

## What has been delivered

The project is a finite-dimensional introductory research laboratory. It
combines established covariance estimators with transport adjustment,
observation diagnostics and reproducible synthetic experiments. Its value is
in mathematical clarity, numerical implementation, fair evaluation and
research-software practice; a new estimator or theorem is not a delivered
result.

A supported CV bullet is:

> Developed and benchmarked a finite-dimensional transport-covariance research
> toolkit with PSD-constrained optimization, observation-geometry diagnostics,
> analytical controls and reproducible synthetic validation.

The four-mode setting should be explained in a technical discussion. It is a
controlled starting point for later infinite-dimensional curve dynamics,
rather than empirical evidence about a futures market.

## Supporting Stage 1 evidence

The [Stage 1 report](../experiments/results/stage1/report.md) records 20 paired
synthetic replications in a four-dimensional exact transport model, with
256/128/128 training/validation/evaluation intervals. Tuning uses observed
validation outer products; latent covariance is used only for evaluation.
Entries are Monte Carlo means and standard errors, not confidence intervals.

| Evidence | Measured result | Interpretation |
| --- | --- | --- |
| Full-coordinate constrained ridge | Relative HS error 0.1201 ± 0.0097 | A well-observed synthetic recovery control |
| Full-coordinate empirical baseline | Relative HS error 0.1189 ± 0.0098 | Ridge does not outperform every baseline |
| Dense / sparse constrained ridge | Relative HS errors 0.3133 ± 0.0411 / 0.8068 ± 0.1228 | Full numerical rank does not ensure precise recovery |
| Missing-pair design | Numerical Gram rank 6/10 | Some contemporaneous covariance directions are absent |
| Selected/refitted covariance fits and transport ablations | 80 covariance fits + 80 ablations meet a 0.0001 numerical budget | The budget concerns distance to the optimizer, not statistical truth |
| Independent CVXPY/SCS checks | Maximum matrix discrepancy 2.4637e-9 across four cases | Numerical comparison for the same objective |

The [manifest](../experiments/results/stage1/manifest.json) records the actual
source revision, environment, source hashes and solver settings for these
measurements. Current automated-check details are maintained in
[verification](verification.md); a historical run's test count should not be
presented as the current suite count.

Optional quantified wording, with the report link and synthetic scope:

> Validated 80 covariance fits and 80 transport-adjustment ablations against a
> 0.0001 numerical-error budget, with independent convex-reference checks.

Full, dense and sparse observations have different information. The transport
ablations are labelled controls, not additional valid innovation-covariance
estimators. Neither a small numerical certificate nor a small error at one
truth establishes identification or estimator superiority.

## RQ2 and RQ3 evidence

The [RQ2 protocol](research-question-2.md) preserves the local-estimation study:
changing covariance, local pair coverage, constrained/projected fitting and
direct risk estimates. It separates terminal estimation from prediction.

The [RQ3 protocol](research-question-3.md) adds chronological forecast
evaluation with earlier observed validation, frozen-origin forecasts and
future observed scores. Its full report is pending computation. Until the
measured report is available, describe this as a predeclared study; do not
claim improved forecasting or a winning window policy.

Even after a report is available, a valid description must identify its
synthetic model, information set, comparison baseline and replication
uncertainty. Results need not favor adaptation, and latent-error diagnostics
must remain distinct from observable forecast scores.

## Technical discussion points and limits

- Exact exponential transport provides an analytical control; projected
  splines expose representation, boundary and semigroup defects.
- Source covariance C, finite-interval innovation covariance V and nonunique
  loading L have separate meanings; C and L are not estimated here.
- Optimization reliability and observation identification are different
  questions. Weak information can dominate statistical error.
- Proxy-based future scores require the stated second-moment/noise identity,
  valid chronology and comparable observations.
- The [three-question roadmap](research-roadmap.md) explains how identification,
  estimation and forecast evaluation continue toward richer curve dynamics.

The delivered evidence concerns synthetic covariance computations and
numerical/software reliability. It does not validate empirical futures
forecasting, curve calibration, confidence intervals, outlier robustness,
or infinite-dimensional statistical guarantees. Timing results apply to the
reported environment, not to all hardware.
