# Identification-aware curve hedging: pilot protocol

This study asks whether partially observed innovation covariance determines a
finite-instrument hedge. It extends RQ1 (decision identification), RQ2
(finite-sample decision stability) and RQ3 (past-only future risk evaluation).
It does not introduce a new generic robust optimizer or establish market
hedging performance. Numerical source is committed before the full run.

## Mathematical experiment

In orthonormal coordinates a residual exposure is $a-Bh$, with variance

$$R_V(h)=(a-Bh)^\top V(a-Bh),\qquad
V=T_\Delta(C)=\Delta^{-1}\int_0^\Delta S_r C S_r^*\,dr.$$

The grid kernel is $K=EVE^\top$. Numerical grid exposures $a_g,B_g$ use
$R_K(h)$, equivalently H-coordinate exposures $E^\top a_g,E^\top B_g$ with V.
Grid kernels are never treated as H-coordinate covariance matrices.

Use the existing weighted-Sobolev Laguerre transport family, weekly intervals,
output projection J=8, unknown source block M=16, and seven fixed maturity
nodes 0, 0.5, ..., 3 years. The underlying baseline source has eigenvalues
$q_k=(k+1)^{-3}/\zeta(3)$. No truncation is renormalized. A positive common tail
above mode M is fixed; finite numerical evaluation retains 512 modes and
records the omitted-tail allowance separately. Thus the generating pair is
infinite rank, while optimization is over a declared **finite unknown source
block with a fixed common tail**. This is not optimization over every
infinite-dimensional source.

The native fragments are nodes {0,1,2,3} and {3,4,5,6}, alternated across dates.
The target is the spread at nodes 0 and 1; the sole hedge is the spread at nodes
5 and 6. The exposure cannot be cancelled using its identical instruments.
The additional-observation design adds maturity 3 years (node 6) to the first
fragment. A complete-panel control observes all nodes.

An explicit trace-preserving perturbation in the nullspace of the observed
source-to-kernel map constructs two PSD sources. Their observed covariance
entries coincide while their optimal hedges differ. The perturbation is
supported on the first M modes; the common infinite tail cancels in their
observational difference. Checks include source eigenvalues, original-entry
residuals, quadrature refinement and an independent block integration.

Population ranges condition on a **known trace** of the unknown block. They
compare the same baseline observations under nested added measurements. Free
PSD completion is an outer comparison, not an SPDE-source claim. The ordinary
canonical completion is a point selection whose source compatibility is not
assumed. Minimax variance and minimax regret are reported as distinct decisions.

## Finite-sample study

The full study uses 20 paired replications, seed 20261029, nested training
prefixes of 128 and 512, and 2048 complete audit observations after the largest
prefix. All innovations are independent, centered Gaussian with common
covariance and known fresh noise variance 0.001. No tuning uses held-out data.
These conditions isolate identification and sampling; changing or heavy-tailed
market covariances are outside the band guarantee.

Within each replication the two worlds have **identical native-fragment
training observations**. Their unobserved coordinates are conditionally
simulated under each world's correct Gaussian law. Bridge and complete designs
are nested views of these same parent observations. Fitting receives only the
masked training prefix and fixed model metadata, never the world label, latent
covariance, source perturbation or future observations.

Compare coherent full-support canonical completion, certified H-coordinate
ridge, free-PSD robust hedging, source-class robust hedging, and direct and
fixed-shrinkage target-block controls when the target/hedge are jointly seen.
Ridge uses relative regularization 0.001 scaled by its observation Gram norm
and an explicit V trace cap 2. Source robust fitting uses a separate C-block
trace cap 2; the free-PSD kernel cap contains that image class. These are
different coordinate-specific constraints, not an assertion of equivalent priors.
Positions are bounded by 8 in magnitude in the sample comparison.

Robust methods use the same observed-entry bands. An invertible within-fragment
contrast exposes the target and hedge spreads before band construction. Each
contrast is computed only where all required quoted nodes are observed.
Chi-square intervals for scalar Gaussian variances, polarization for cross
covariances and a Bonferroni correction supply simultaneous coverage of at least
95% under the declared experiment. Known noise is subtracted explicitly;
indefinite corrected moments are retained. The entry-band guarantee does not
transfer to dependent, non-Gaussian market returns, unknown means/noise or
informative masks. Source-class inclusion additionally requires the declared
fixed known tail; entry-band validity alone does not require a particular source tail.

The robust SDP imports optional CVXPY/SCS. Original-entry, PSD and trace
residuals, dual/support consistency and solver status are checked. Inaccurate
or failed results remain unavailable. These checks are not exact numerical
certificates or a coverage theorem for the completed kernel. Existing ridge
and coherent optimization certificates retain their original scope.
The declared retry policy tightens SCS tolerance tenfold with a 300,000-iteration
budget, then tries CLARABEL when necessary; every attempt is retained. It never
changes observation bands or the estimator to obtain a successful result.

## Evaluation and gates

Complete future audit observations score every issued hedge using the same
noise-corrected squared residual. This is a **controlled masking evaluation**,
not evidence that a native fragment panel observes a long-spread P&L. Native
scoring availability is recorded separately. Latent risks/regret are simulation
diagnostics; observed scores and their Monte Carlo errors are kept separate.
MCSE uses replication-level paired contrasts, not dates, worlds or optimization
iterations. Missing direct-block comparisons remain unavailable.

The positive gate is a numerically verified observationally equivalent pair
with material hedge differences, plus narrowing under a nested added
measurement. No estimator must win. A joint-observation control checks whether
a direct small covariance block suffices; full covariance completion is not
automatically useful for the decision.

Run into a new directory:

```sh
uv sync --locked --extra reference
uv run spde-covariance benchmark-hedging experiments/configs/identified-hedging-smoke.toml --output /tmp/hedging-smoke-new
uv run spde-covariance benchmark-hedging experiments/configs/identified-hedging.toml --output experiments/runs/identified-hedging-new
```

The [futures interface](futures-data.md) retains contract identity, availability,
units and irregular intervals for a subsequent empirical study. Its six-date
SHFE feasibility audit is not an estimation or hedging-performance study.

## Prior work and novelty boundary

The study builds on [semigroup-adjusted covariation](https://arxiv.org/abs/2205.03927),
[canonical completion](https://arxiv.org/abs/2107.07350),
[dynamics-consistent completion](https://arxiv.org/abs/1412.3399),
[infinite-rank curve hedging](https://arxiv.org/abs/2606.28891), and
[uncertainty-aware hedging](https://arxiv.org/abs/2604.02126).
Generic completion ambiguity, risk bounds and minimax portfolio optimization
are established. A future contribution would require informative
transport/observation-specific decision conditions, sampling and representation
guarantees, or new empirical evidence. This pilot makes no first-of-its-kind claim.
