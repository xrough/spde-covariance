# What the integrated completion study establishes

The [reference study](../experiments/results/canonical-reference/report.md)
uses 100 independent replications of each paper control. The
[SPDE study](../experiments/results/canonical-spde/report.md) uses 20 paired
replications of the existing fixed infinite-rank sources. Both were run from
clean numerical source [`476d389`](https://github.com/xrough/spde-covariance/commit/476d389df9b005df5a679763b0086992ba7e70bf).
Their manifests, full measurements and checksums accompany the reports.
These are controlled numerical results for established estimators.

## RQ1: observations and a completion convention answer different questions

The two-block control has identical observed covariance for every residual
correlation between minus one and one, yet endpoint-spread variance spans
**0 to 4**. These attainable identification ranges concern all compatible PSD
grid completions, not confidence intervals or the class of SPDE-admissible
sources. Canonical completion chooses residual correlation zero. It does not
make the generating missing covariance identifiable.

Exact population controls check compatible singular separators, recursive
order invariance, and CVXPY maximum-logdet references. The
[actual pinned author-R comparison](../experiments/results/author-completion-reference/README.md)
passes eight explicit-rank cases with maximum absolute difference 8.33e-13.
This does not test the author's differently normalized FVE routine.

## RQ2: more data helps, but does not remove a fixed inverse approximation

For three fragments, the finite-rank control's median missing-region ISE falls
from 0.02485 at N=100 to 0.009490 at N=500. Brownian's falls from 0.0005874 to
0.0001305. With nine fragments the respective pairs are 0.1173→0.04326 and
0.0009215→0.0002396. Holding total sample size fixed changes per-fragment
sample sizes, recursive steps and observed domains; this is not evidence that
more coverage generally harms estimation.

The analytic control illustrates a remaining bias. At three fragments its
median observed ISE falls 0.3039→0.05602, whereas missing ISE changes only
0.02017→0.01595 with the fixed rank-two policy. At nine fragments, missing
ISE changes 0.02261→0.02157. This concerns the finite-grid, fixed-rank
algorithm; it does not contradict uniqueness results under a continuum
analyticity assumption.

For a predeclared central SPDE setting—base grid 81 (86 augmented nodes),
three fragments, noise variance 0.001—the primary rank-two policy gives:

| Source | N | Raw complete weighted L2 error | Separate PSD error |
| --- | ---: | ---: | ---: |
| Power-law | 128 | 0.3943 ± 0.0424 | 0.3846 ± 0.0420 |
| Power-law | 512 | 0.2224 ± 0.0192 | 0.2179 ± 0.0192 |
| Power-law | 2048 | 0.1064 ± 0.00962 | 0.1043 ± 0.00950 |
| Rotated | 128 | 0.5806 ± 0.0574 | 0.5656 ± 0.0562 |
| Rotated | 512 | 0.3100 ± 0.0337 | 0.3044 ± 0.0335 |
| Rotated | 2048 | 0.1469 ± 0.0123 | 0.1441 ± 0.0121 |

Entries are means ± one MCSE across 20 independent replications; all primary
fits are available. These are maturity-weighted evaluation-kernel norms, not
H-coordinate covariance errors. The generating/canonical-reference discrepancy
at this setting is 2.74e-6 for power-law and 3.10e-6 for rotated. A small discrepancy
in these particular smooth, low-effective-rank sources does not establish
canonicity for general transport-SPDE kernels.

The existing ridge alternative is selected only from observed validation data.
At base grid 81, N=512 and noise variance 0.001, its paired test-score differences
against raw rank-two completion are 3.19 ± 3.64 and 3.72 ± 6.63 for power-law
(three/nine fragments), and 8.02 ± 10.77 and −11.12 ± 9.35 for rotated. Lower is
better; signs vary and the Monte Carlo uncertainty is substantial. This study
does not establish a uniform predictive advantage. All 1440 ridge candidate
fits converge; maximum certified H-distance to the optimizer is 1.71e-5,
below the declared 1e-4 budget. Kernel completion has no such optimizer certificate.

A more direct warning comes from the same grid/noise setting with nine fragments and N=128:

| Inverse rank | Complete weighted L2 error (mean ± MCSE) | Mean held-out observed score |
| --- | ---: | ---: |
| rank1 | 0.5211 ± 0.0359 | -2079.830247 |
| rank2 | 0.5643 ± 0.0366 | -2079.830247 |
| rank4 | 2.383 ± 1.35 | -2079.830247 |
| rank8 | 3.913 ± 2.11 | -2079.830247 |

The raw ranks have exactly the same observed entries and datewise scores. Increasing inverse rank can amplify missing-block errors while the ordinary validation score remains blind to that change. This is why ranks are predeclared rather than selected from this score.

## Feasibility and numerical sensitivity remain part of the result

Every available raw completion in these runs is materially indefinite:
1215 in the reference study and 5760 in the SPDE study. Pairwise empirical
moments can be incompatible, and truncation can itself break PSD even on a
compatible population input. Raw arrays are retained as estimator diagnostics;
they are not silently advertised as valid covariance matrices.

Under the declared policy that FVE is unavailable on an indefinite separator,
1185/1200 reference attempts and 1440/1440 SPDE attempts are unavailable.
These counts describe this guardrail and this empirical partial estimator;
they do not show failure of every possible PSD-preprocessed FVE variant.
All explicit manual-rank attempts in the full studies remain available.
Unavailable results are never replaced by an undeclared fallback rank.

Weighted PSD projection restores feasibility and can reduce weighted L2
error to PSD truth, but changes observed entries, traces and individual risks.
It does not uniformly improve exposure-risk error. The archive retains the
changes and treats projection as a separate estimator. Availability-conditioned
summaries and intersection-only paired comparisons must be read with their
replication counts.

All 144 population forcing/grid/overlap/support checks are available. Their
largest relative difference on common physical targets is 5.34e-5, below the
predeclared 1e-4 instability threshold. Retained separator condition numbers
reach 9.77e13. Passing this comparison is numerical evidence, not a continuum
convergence certificate or a bound on inverse-amplified truncation error.

## RQ3 and the next research target

The new held-out scores compare fixed covariance fits on subsequent independent
innovations. The existing RQ3 forecasting workflow is preserved; this increment
adds no new dynamic forecasting model or empirical futures evidence.

The SPDE exposures preserve the existing full-H-normalized point spread
(0.5 versus 1.5 years) and off-grid spread (0.75 versus 2.25); a further unnormalized
spread compares 0.25 versus 2.75. Continuous delivery-spread risk and its
finite-grid quadrature approximation remain distinct. A continuous integral
cannot receive an exact operational score from finitely many point values.

The next substantive target is to control how separator inversion amplifies
spatial approximation and sampling errors for one fixed infinite-dimensional
transport model, and to test when its canonical restriction is defensible.
Only then should later forecast studies interpret completed functional risks
as conditional uncertainty under that structure. Sparse fragments, noisy price
reconstruction and new forecast models remain subsequent extensions.
