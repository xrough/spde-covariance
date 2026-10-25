# Statistical completion: findings and limits

The [full report](../experiments/results/statistical-completion/report.md)
comes from clean numerical source `f24286a`, without post-run tuning. It
contains 20 paired point-study replications and 500 independent uncertainty
replications in each of six paired design/sample-size strata. The implemented
[model and protocol](statistical-completion.md) use established Gaussian
graphical modelling, EM and simultaneous moment regions.

## What changed in the prototype

The new Gaussian method fits the likelihood of each actually observed vector,
including its known fresh noise covariance. It does not treat corrected
outer products as Gaussian observations or individual covariance pairs as
independent samples. The precision graph declares a canonical
conditional-independence model and remains fixed when bridge data are scored.

The workflow now supplies chronological observed-only penalty selection,
conditional latent means/covariances, local convergence and initialization
audits, and a separate partial-covariance confidence region. Risk bounds from
that region allow all compatible PSD completions, rather than conditioning
on the fitted graph. The [runnable example](../examples/statistical_completion.py)
also works from a core-only wheel; numerical risk endpoints use the optional
reference solvers.

## Fitting works, but likelihood and missing-entry accuracy differ

All 360 candidate fits were available. There are 120 distinct issued fits,
reused across three equal-native-law worlds, producing 360 evaluation rows.
The 540 Gaussian initialization attempts all reached the declared local
criterion; the largest selected relative score residual was `8.53e-7` and
the maximum selected iteration count was 34. No global observed-likelihood
optimum or completed-kernel statistical error certificate is asserted.

Under the correctly specified canonical Gaussian control, weighted kernel
error decreases with N. Entries below are mean ± replication MCSE:

| N | Gaussian likelihood fit | Coherent moment fit |
|---:|---:|---:|
| 128 | 0.07484 ± 0.01127 | 0.10354 ± 0.01329 |
| 512 | 0.04893 ± 0.00814 | 0.06669 ± 0.00684 |
| 2048 | 0.02197 ± 0.00422 | 0.03325 ± 0.00413 |

At N=2048, Gaussian minus moments has native test NLL difference
`-0.6051 ± 0.1700` and full-panel difference `-1.2187 ± 0.3227`, using
20 replication-paired contrasts. Lower is better. Weighted complete-kernel
error changes by `-0.01128 ± 0.00328`, but missing-entry error changes by
only `-0.00160 ± 0.00203`: no resolved missing-region advantage follows.

Both methods use the same training-only reference and observed validation
data. Their KL and Frobenius penalties are different, so these differences
evaluate the complete fitting policies, not likelihood alone. No universal
superiority or market-performance claim follows from these synthetic cases.

## Native validation cannot validate the completion convention

The two true-point SPDE source worlds and the canonical Gaussian control
have exactly shared native observations. Fits and selections therefore
coincide across worlds. The canonical control is a grid Gaussian model;
membership in an SPDE source class is not asserted.

The population canonical completion has positive full-panel Gaussian KL
discrepancy `0.2450` and `0.2905` against the two SPDE worlds, despite the
same native covariance. It has zero discrepancy against the canonical
control, to rounding. This is an explicit model-misspecification example.
More native samples estimate the shared observed covariance more accurately;
they cannot determine which missing block generated the data. Full-panel
test measurements are a disclosed bridge audit, not information available
to native fitting or selection.

## Uncertainty retains unidentified directions

The separate three-node Gaussian experiment has native cliques {0,1} and
{1,2}. Its two worlds share those marginals but have missing covariance
0.16 versus 0.65. For the normalized spread $(X_0-X_2)/\sqrt{2}$, their
variances are 0.84 versus 0.35. Even exact native covariance leaves the
all-PSD variance range [0, 1.68] for this exposure.

All 3,000 interval solves were available. Mean interval widths and their
replication MCSE are:

| N | Native fragments | Complete observations |
|---:|---:|---:|
| 128 | 3.38399 ± 0.01773 | 1.46593 ± 0.00546 |
| 512 | 2.50934 ± 0.00680 | 0.70124 ± 0.00135 |
| 2048 | 2.10707 ± 0.00290 | 0.34750 ± 0.00035 |

The complete views and nested prefixes use the same parents. MCSE remains
within each 500-replication stratum; designs, sample sizes and worlds are
not pooled as additional independent replications. Complete observations
have different laws across the two worlds; complete-design coverage is
assessed against its own canonical generating law only.

Simultaneous observed-entry coverage ranges from 97.4% to 98.6% across
strata, versus a nominal lower guarantee of 95% under the declared Gaussian
assumptions. Risk intervals cover the relevant truth in 500/500 cases per
stratum. This indicates conservative intervals, not a true coverage
probability of 100%: the exact binomial 95% interval for 500/500 is
[99.265%, 100%]. Conditional model predictions are not substituted for this
identification-inclusive uncertainty.

The numerical intervals use repaired feasible dual outer bounds, with primal
witnesses and gaps retained. Chi-square quantiles, covariance feasibility and
dual checks use floating point; these are not interval-arithmetic proofs.
The simulation does not calibrate inference for dependent, heavy-tailed,
unknown-mean or unknown-noise market returns.

## What is ready for later empirical work

The package now has an explicit fit/selection/prediction/uncertainty boundary,
common-step target and unit checks, immutable results and informative failure
states. A region's ambiguity model can feed the existing downstream hedge
interface, but this increment makes no new hedging-performance claim.

An empirical stage still needs contract-to-curve observation maps, appropriate
drift and price-noise assumptions, treatment of time dependence, and legally
usable histories. The likelihood model is finite-grid and does not impose an
H-operator rank constraint or transport-source cone. Singular algebraic
completions remain a separate supported method. No operator recovery theorem,
unknown-tail inference or uniform infinite-dimensional coverage is established.
