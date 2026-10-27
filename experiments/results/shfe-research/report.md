# SHFE copper: exploratory research suite

This report is generated from checksummed aggregate evidence. It supplements the frozen first benchmark; all new analyses on inspected 2024–2025 data are exploratory development. No untouched confirmation period has been evaluated.

For the research narrative and figures, start with the [current project overview](../../../notebooks/00_project_findings.ipynb) or [short findings report](../../../docs/project-findings.md). The tables below are the aggregate audit appendix; full precision and all sensitivities remain in the linked JSON files.

Numerical source: `a2f99054886b3154a311665cee7cacd8e6af483c`; clean at execution: `True`.

## Data and effective history

969 sessions, 11628 rows, 60 actual contracts. Midmonth coordinates are not actual expiry dates; the negative-coordinate bin is retained. Archive identities and every frozen benchmark scored proxy were checked against the supplied panel.

| Window (intervals) | Count | Joint count: min / Q1 / median / Q3 / max | Origins meeting minimum |
| --- | --- | --- | --- |
| 63 | 484 | [49, 57, 62, 63, 63] | 484 |
| 126 | 484 | [49, 57, 62, 67, 77] | 484 |
| 252 | 484 | [49, 57, 62, 67, 77] | 484 |

## Frozen benchmark: both losses

Negative paired differences favor local SPDE versus direct. QLIKE remains primary. Uncertainty is a circular block bootstrap of scored sessions, not IID Monte Carlo precision. Quadratic values have units CNY⁴.

| Exposure | Loss | Count | Mean difference | Block-bootstrap SE | 95% block-bootstrap percentile range |
| --- | --- | --- | --- | --- | --- |
| Butterfly | QLIKE | 484 | 5.882 | 1.874 | [2.623, 9.859] |
| Butterfly | quadratic | 484 | -1.647e+12 | 9.286e+11 | [-3.478e+12, 1.887e+11] |
| Outright | QLIKE | 484 | 0.1303 | 0.136 | [-0.154, 0.3938] |
| Outright | quadratic | 484 | -7.444e+12 | 5.232e+13 | [-1.119e+14, 8.761e+13] |
| Spread | QLIKE | 484 | 0.953 | 0.4261 | [0.1878, 1.839] |
| Spread | quadratic | 484 | -3.907e+11 | 1.22e+11 | [-6.654e+11, -1.766e+11] |

## Matched transport ablation

All-node reconstruction, fixed nominal windows. The statistic is adjusted minus unadjusted (A=0) QLIKE on identical paired observations; negative differences favor transport adjustment. This does not compare separately tuned pipelines.

| Window (intervals) | Exposure | Count | Mean difference | 95% block-bootstrap percentile range |
| --- | --- | --- | --- | --- |
| 16 | Outright | 484 | -0.001049 | [-0.002984, 0.0003767] |
| 16 | Spread | 484 | -0.01955 | [-0.06475, 0.02092] |
| 16 | Butterfly | 484 | 0.04517 | [0.02148, 0.07369] |
| 32 | Outright | 484 | -0.0007912 | [-0.001969, 0.0002482] |
| 32 | Spread | 484 | -0.001322 | [-0.02113, 0.01858] |
| 32 | Butterfly | 484 | 0.04137 | [0.0282, 0.05569] |
| 63 | Outright | 484 | 0.0004053 | [-0.0005017, 0.001304] |
| 63 | Spread | 484 | 0.021 | [0.0002723, 0.05619] |
| 63 | Butterfly | 484 | 0.05008 | [0.0249, 0.08558] |
| 126 | Outright | 484 | -3.225e-05 | [-0.0005608, 0.0005016] |
| 126 | Spread | 484 | 0.008182 | [-0.001966, 0.02052] |
| 126 | Butterfly | 484 | 0.0376 | [0.02092, 0.05593] |
| 252 | Outright | 484 | -0.0001419 | [-0.0003686, 5.539e-05] |
| 252 | Spread | 484 | 0.005305 | [-0.001739, 0.01365] |
| 252 | Butterfly | 484 | 0.03226 | [0.019, 0.0469] |

Full mask/short-history, available-method and numerical diagnostics are in [attribution.json](attribution.json). At 16 intervals the unchanged 20-pair direct minimum makes raw baselines unavailable; factor comparisons remain available.

## Revised joint-risk policies

mean validation QLIKE across outright, spread and butterfly on identical validation origins within each rule; exact ties longer window then calendar clock then smaller shrinkage

| Policy | Window (intervals) | Diffusion clock | Diagonal shrinkage | Validation origins | Last validation outcome |
| --- | --- | --- | --- | --- | --- |
| Factor only | 63 | calendar | 0 | 231 | 2023-12-29 |
| Joint factor + residual | 63 | calendar | 0 | 231 | 2023-12-29 |
| Direct covariance | 63 | calendar | 0 | 231 | 2023-12-29 |
| Source-C model | 252 | calendar | 0 | 231 | 2023-12-29 |

Joint factor/residual second moments retain cross terms and PSD feasibility. Pooling normalized innovations across unequal calendar intervals is an empirical local approximation, distinct from the original interval-integrated source-C model. Direct/factor/joint share historical complete cases at a fixed window. Source uses its own full coefficient window.

| Exposure | Policy | Common outcomes | QLIKE | Realized / predicted second moment |
| --- | --- | --- | --- | --- |
| Outright | Direct covariance | 484 | 17.28 | 1.09 |
| Outright | Factor only | 484 | 17.27 | 1.103 |
| Outright | Joint factor + residual | 484 | 17.28 | 1.076 |
| Outright | Source-C model | 484 | 17.4 | 1.186 |
| Spread | Direct covariance | 484 | 13.64 | 0.4563 |
| Spread | Factor only | 484 | 15.17 | 2.218 |
| Spread | Joint factor + residual | 484 | 13.71 | 0.4068 |
| Spread | Source-C model | 484 | 14.56 | 2.279 |
| Butterfly | Direct covariance | 484 | 15.07 | 0.6582 |
| Butterfly | Factor only | 484 | 23.76 | 7.563 |
| Butterfly | Joint factor + residual | 484 | 15.07 | 0.6254 |
| Butterfly | Source-C model | 484 | 20.95 | 6.933 |

## Bounded hedge decisions

One near contract and two hedge positions in [−2, 2], chosen from past information. Common evaluation origins, actual fixed-contract next-session settlement P&L. Position changes are maturity-slot instability proxies, excluding contract-switch, roll and execution costs.

| Policy | Common outcomes | Realized RMS (CNY) | Second-moment reduction vs unhedged | Mean position-change proxy |
| --- | --- | --- | --- | --- |
| Unhedged | 484 | 3487 | 0 | Unavailable |
| Direct covariance | 484 | 501.4 | 0.9793 | 0.0541 |
| Factor only | 484 | 684.9 | 0.9614 | 0.07205 |
| Joint factor + residual | 484 | 533 | 0.9766 | 0.06438 |
| Source-C model | 484 | 624.1 | 0.968 | 0.03937 |

Paired mean squared-P&L differences (CNY²), listed policy minus direct covariance; negative differences favor the listed policy. These are differences in second moments, not differences between the RMS values above.

| Policy | Reference | Count | Mean difference | 95% block-bootstrap percentile range |
| --- | --- | --- | --- | --- |
| Factor only | Direct covariance | 484 | 2.177e+05 | [8.331e+04, 3.985e+05] |
| Joint factor + residual | Direct covariance | 484 | 3.266e+04 | [1.754e+04, 4.761e+04] |
| Source-C model | Direct covariance | 484 | 1.381e+05 | [2.35e+04, 2.982e+05] |

## Interpretation and limitations

Residual risk explains a major weakness of smooth-factor forecasts. Repairing it does not establish superiority over direct risk estimation. Calendar/session clocks have different effects across exposures; selection remains fixed on validation data. Price reconstruction, numerical convergence and economic risk adequacy are separate gates.

See the six executed notebooks and docs/empirical-research-protocol.md for mathematical targets, coverage, calibration, influence, clock-gap and rank diagnostics. The original benchmark is unchanged. No physical forcing identification, infinite-dimensional statistical guarantee, trading profitability or execution-aware backtest is claimed.

Measured suite runtime excluding archive parsing: 228.59 seconds on macOS-15.0.1-arm64-arm-64bit-Mach-O. This is a single environment measurement, not a hardware-independent performance claim.
