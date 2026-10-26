# Futures risk research suite: protocol and boundaries

The financial question is **when a curve representation helps forecast contract
risk, and when that representation removes risk that matters for spreads or
hedges**. The suite implements six complementary investigations, rather than
treating a single average covariance error as a sufficient result.

The [first benchmark](shfe-empirical.md) and its selected rules remain frozen.
All new investigations on 2024–2025 are **exploratory development**: their
outcomes have already influenced the research agenda. Issuance remains past-only
and new policies select settings on 2023 validation observations. This prevents
mechanical future leakage; it does not restore an untouched research holdout.
No final confirmation period has been evaluated. Acquire an uninspected period
only after the revised protocol and acceptance criteria are frozen.

## Six investigations

| Notebook | Question | Evidence and decision |
| --- | --- | --- |
| [01 Data and targets](../notebooks/01_data_and_risk_targets.ipynb) | What economic quantity is measured, and is its history available? | Contract identity, quality, tenor/liquidity coverage, effective same-contract history |
| [02 Representation](../notebooks/02_representation_and_identification.ipynb) | Does a good curve-level fit preserve exposure risk? | Dimension/smoothing/conditioning and fitted/residual/cross second moments |
| [03 Attribution](../notebooks/03_performance_attribution.ipynb) | Which mechanism changes scores? | Matched windows, transport on/off, raw baselines, nested maturity masking and short histories |
| [04 Forecast evaluation](../notebooks/04_forecast_scores_and_model_risk.ipynb) | How robust is the ranking to loss, dependence and influential outcomes? | Both proper scores, paired block uncertainty, annual/gap groups, calibration and influence |
| [05 Residual risk](../notebooks/05_joint_residual_risk_and_clocks.ipynb) | Can a joint model preserve omitted risk, and does the diffusion clock matter? | PSD joint factor/residual second moments with cross terms, calendar/session clocks |
| [06 Hedging](../notebooks/06_hedging_decisions.ipynb) | Do covariance forecasts support useful risk decisions? | Past-only constrained minimum-risk hedges, realized squared P&L and position-change proxies |

The [executive notebook](../notebooks/shfe_empirical.ipynb) remains available at
its existing path. The [synthetic findings](../notebooks/current_findings.ipynb)
provide identification controls without market-data ambiguity.

## Targets, conventions and chronology

Use official SHFE copper settlement files, 2022–2025, with the existing immutable
long panel. Keep actual contract IDs and do not splice continuous contracts.
Select origin contracts nearest 90, 180 and 270 calendar days and hold those
identities until the next session. Multipliers are five tonnes per contract.
The 15th of contract month is a modeling coordinate, not an audited expiry or
physical delivery-average observation. The archives are retrospective; original
publication/revision vintages and executable prices are not reconstructed.

For holdings $a$ and multiplier $m$, the measured outcome is

```math
Y_{i+1}=m a^\top(p_{i+1}^{\mathrm{same\ IDs}}-p_i),\qquad
v_i\approx\mathbb E[Y_{i+1}^2\mid\mathcal F_i].
```

These are conditional second moments. A variance interpretation additionally
requires a negligible or zero conditional mean. No Gaussian-tail calibration
or profitable trading claim follows from a squared-P&L score.

Reconstruct the existing three-mode Laguerre curve independently each session.
With physical elapsed time $\Delta_i$ in ACT/365.25 years and exact finite
transport $M_{\Delta_i}$, define

```math
r_i=a_i-M_{\Delta_i}a_{i-1},\qquad
\Delta p_i=E_i r_i+u_i.
```

The residual $u_i$ is the change of same-contract reconstruction errors. It is
not assumed fresh independent measurement noise. The original source fit uses
the interval maps $V_\Delta(C)=\Delta^{-1}\int_0^\Delta M_sCM_s^\top ds$.
The new joint risk model instead estimates the second moment of stacked factor
and residual innovations. It does not identify the physical market forcing $C$.

Training is 2022, policy selection is 2023, and the inspected evaluation period
is 2024–2025. Later origins may use earlier realized evaluation observations;
no issuer receives its own future settlement. Settings are frozen before
evaluation. The original independently selected windows remain in the baseline;
matched attribution fixes the same nominal window for each compared method.
Actual usable contract counts are reported separately because contract life
can make a 252-session window much shorter for a currently held contract.

## Comparisons and diagnostics

The primary loss remains $\log(v)+z/v$, with the original positive floor and
$z=Y^2$. The secondary loss is $v^2-2vz$. Conditional score differences replace
$z$ by $\mathbb E[z\mid\mathcal F_i]$ in expectation because both differences
are affine in $z$. Their rankings can differ under misspecification: quadratic
loss weights absolute cash-risk errors, while QLIKE strongly penalizes severe
relative underprediction. Neither score validates invisible covariance directions.
This interpretation follows the imperfect-proxy evaluation framework in
[Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf).

Report paired differences, common and unmatched counts, and circular
moving-block bootstrap summaries for 5, 10, 20 and 40 scored sessions. Dates
are dependent observations, not independent Monte Carlo replications. Bootstrap
percentile ranges are descriptive, with no multiplicity adjustment or market
regime stability guarantee. Calibration bins, realized outcome strata and
influence removals are post hoc diagnostics; they do not replace the main score
or select a revised model on its evaluation outcomes.

The attribution experiment uses deterministic nested maturity masks. They affect
factor reconstruction; raw direct baselines retain their original contract
history. Therefore masked-factor versus raw-direct comparisons are information
stress tests. Within-mask adjusted versus unadjusted factors isolate transport
more closely. Insufficient history and numerical failures remain explicit.

For the joint model, calendar transport always uses the physical elapsed time.
Only stochastic normalization changes between calendar and session diffusion
clocks. Joint PSD second moments retain factor/residual cross-covariances;
diagonal shrinkage is an explicit regularizer, not an independence conclusion.
Compare factor-only and raw direct covariance on the same joint historical
support to distinguish missing-risk effects from sample changes.

Hedges retain one near contract and choose bounded positions in the middle/far
contracts to minimize predicted raw squared risk. Selection and position choice
receive historical information only. Report unhedged and hedged realized RMS,
predicted risk, common availability and paired uncertainty. Position changes in
maturity-selected hedge slots are an instability/turnover proxy, not a fully
specified self-financing execution backtest. No transaction-cost or alpha claim
is made.

## Reproduction and data boundary

```sh
uv sync --locked --extra empirical --extra reference
uv run python examples/empirical_research_pipeline.py  # synthetic, no exchange files
uv run python scripts/download_shfe.py --years 2022 2023 2024 2025 --cache-dir data/local/shfe
uv run spde-covariance empirical-shfe experiments/configs/shfe-cu.toml --data data/local/shfe --output data/local/shfe/runs/baseline-new
uv run spde-covariance empirical-research experiments/configs/shfe-cu-research.toml --data data/local/shfe --baseline-run data/local/shfe/runs/baseline-new --output data/local/shfe/runs/research-new
```

Use one BLAS thread for reproducible numerical runs. Commit numerical source
before full historical runs. Each run records source/configuration hashes,
archive provenance, environment, selected settings, numerical diagnostics and
runtime. Export only declared aggregate products with checksums. Raw exchange
files, reconstructed daily curves, contract-level forecasts and individual P&Ls
remain local and Git-ignored. Public notebooks execute from the aggregate bundle
without market files or Jupyter-specific magics.

The SPDE foundations, including semigroup-adjusted covariation, are separate
from this empirical finite-model approximation. See
[Benth–Schroers–Veraart](https://arxiv.org/abs/2205.03927); their asymptotic
guarantees are not asserted for reconstructed settlement curves here.
