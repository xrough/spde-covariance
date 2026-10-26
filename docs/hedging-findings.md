# What the identification-to-hedging pilot establishes

The identification question is consequential in the declared transport model.
The current conservative hedge rule does **not** demonstrate a consistent
finite-sample performance advantage. Both findings are retained in the
[full 20-replication report](../experiments/results/identified-hedging/report.md),
with endpoint witnesses, failures and a clean-source manifest.

The 20-replication study observes **evaluations of the J=8 projected curve**,
not evaluations of the full curve. The separate
[true-point control](../experiments/results/hedging-true-points/report.md)
below checks whether the identification issue survives that distinction.

## Covariance ambiguity can change the decision

The two fixed infinite-rank sources have a common power-law tail. A
finite-supported PSD perturbation leaves all 19 native observed covariance
pairs unchanged to about `2e-16`, but changes the optimal spread hedge from
1.67708 to 5.27663. Swapping those hedges adds residual variance 0.00284185,
or 14.70% of unhedged variance. This is a numerical observational-equivalence
control, not an empirical market estimate.

Under the baseline population observations, an unknown 16-mode PSD source
block, known block trace and fixed common tail, the optimal hedge ranges from
0.94001 to 7.32806. Adding maturity three years to the first fragment narrows
it to [3.21409, 3.69739], a 92.43% width reduction. Unconstrained scalar
minimax regret falls from about 0.00223760 to 0.00001281. This nested calculation
uses the same baseline covariance and source family at every design.

![Population hedge ranges](../experiments/results/identified-hedging/identification.png)

These identification ranges condition on transport, known trace, finite
unknown support and fixed tail. They are not confidence intervals, exact SDP
certificates or bounds over every infinite-dimensional source. Trace and
source restriction act together; the comparison does not isolate transport
alone. Minimax variance and minimax regret are different objectives. The
generating sources are PSD, but numerical endpoint witnesses have tiny negative
eigenvalues within the declared tolerance; exact PSD is not claimed for them.

## The ambiguity survives true point observations

For the full curve's point evaluations, a separately constructed source pair
has optimal hedges **0.81678 and 6.07917**. Using the other world's hedge adds
0.00613651 variance, or **31.73%** of unhedged variance. Its 19 observed pairs
agree within about `2e-15` in the finite numerical map. Both sources are PSD,
trace one and share the same infinite power-law tail. Quadrature refinement
and an independent semigroup block calculation agree within `4.5e-15`;
the omitted fixed tail changes each optimal hedge by less than `9e-6`
under the stated analytic bound evaluated in floating point.

This is a deterministic control, separate from the sample comparison and
population added-maturity ranges above. The original J=8 perturbation is
**not** observationally equivalent under true point observations: its largest
observed-entry discrepancy is about `9.55e-5`. The new control demonstrates
the phenomenon under a different observation map; it does not validate
reusing the original pair as spatial resolution changes. See the
[reproducible annex](../experiments/results/hedging-true-points/report.md)
for the construction, numerical residuals and fixed-tail scope.

## Sampling does not establish useful robust hedging yet

The full run contains 20 independent paired replications, training prefixes
128/512, two source worlds and three nested designs. Each issued hedge is
scored on 2048 complete future audit observations. The four main methods are
available in all scheduled cases (240/240 per method). Of 1440 records, 1099
are available and 341 unavailable: 320 lack joint training observations for
direct/shrinkage comparisons; 21 have indefinite noise-corrected complete-panel
blocks. Those baselines are not silently replaced by PSD projections.

Both robust methods choose essentially zero; the largest absolute
source-robust position is about `3.2e-6`. Source and free-PSD robustness perform
almost identically. The sets combine simultaneous Gaussian entry bands, a
trace cap and, for the source rule, a fixed known tail. Their conservatism
cannot be attributed to one component without another ablation. Sampling uses
source cap two, whereas population ranges use known block trace: their prior
strengths differ.

At N=512 these are paired **future audit score** differences, source robust
minus coherent canonical, mean ± replication MCSE; lower is better:

| World | Design | Difference ± MCSE | Paired replications |
|---|---|---:|---:|
| Minus | Fragments | +0.000127 ± 0.000040 | 20 |
| Plus | Fragments | +0.000663 ± 0.000173 | 20 |
| Plus | Complete | +0.005793 ± 0.000479 | 20 |

For the complete plus world, source robust minus direct target-block is
+0.006293 ± 0.000692 on 18 common available replications. In the minus world
that comparison is noisy. There is no universal ranking; added-observation
ridge improves on canonical in the plus world, while other cells favour
canonical. Kernel error, hedge choice and the risk of that choice are separate
evaluation targets.

After deduplicating shared native inputs, 197/200 distinct band sets contain
the simulated entries. These are correlated outcomes within 20 independent
replications, not 200 independent coverage trials or a run-wide 95% event.
Coverage requires centered independent common Gaussian rows, deterministic
masks and known fresh noise; it does not cover dependent market returns.

Complete future audit scores evaluate **controlled masking**. Native
cross-fragment risk scores are unavailable for the issued fragment/bridge
hedges. Latent risk and regret do not manufacture operational observations.

## Empirical bridge and next step

The [contract-price interface](futures-data.md) reads auditable CSV panels,
preserves identity and publication/metadata availability, retains irregular
intervals, and maps signed contract positions into currency P&L. It performs
no implicit rolling or automatic reinterpretation of differenced noisy levels
as fresh innovation noise. The [offline example](../examples/futures_pipeline.py)
fits past data and evaluates subsequent synthetic P&L. A curve/SPDE
interpretation still requires caller-supplied observation maps.

The [six-date SHFE audit](../experiments/results/futures-data-audit/report.md)
establishes parser feasibility and lifecycle missingness, without estimating
covariance, execution costs or market hedging performance.

Next compare matched trace assumptions, separate sampling width from permanent
ambiguity, and investigate more informative decision-relevant uncertainty
sets. A real-data stage must establish observation/noise and temporal-dependence
assumptions before inference. Generic minimax optimization and completion are
established methods; this pilot proves no methodological novelty or uniform
infinite-dimensional statistical guarantee.
