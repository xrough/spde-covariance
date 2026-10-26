# SHFE copper: exploratory research suite

This report is generated from checksummed aggregate evidence. It supplements the frozen first benchmark; all new analyses on inspected 2024–2025 data are exploratory development. No untouched confirmation period has been evaluated.

Numerical source: `a2f99054886b3154a311665cee7cacd8e6af483c`; clean at execution: `True`.

## Data and effective history

969 sessions, 11628 rows, 60 actual contracts. Midmonth coordinates are not actual expiry dates; the negative-coordinate bin is retained. Archive identities and every frozen benchmark scored proxy were checked against the supplied panel.

| window | count | joint_count_quantiles | sufficient_count |
| --- | --- | --- | --- |
| 63 | 484 | [49.0, 57.0, 62.0, 63.0, 63.0] | 484 |
| 126 | 484 | [49.0, 57.0, 62.0, 67.0, 77.0] | 484 |
| 252 | 484 | [49.0, 57.0, 62.0, 67.0, 77.0] | 484 |

## Frozen benchmark: both losses

Negative paired differences favor local SPDE versus direct. QLIKE remains primary. Uncertainty is a circular block bootstrap of scored sessions, not IID Monte Carlo precision. Quadratic values have units CNY⁴.

| exposure | loss | count | mean | bootstrap_standard_error | bootstrap_percentile_95 |
| --- | --- | --- | --- | --- | --- |
| butterfly_3m_6m_9m | qlike | 484 | 5.88185 | 1.87372 | [2.6234802059417115, 9.858522943011737] |
| butterfly_3m_6m_9m | quadratic | 484 | -1.64749e+12 | 9.28587e+11 | [-3477748199215.237, 188690188480.62622] |
| outright_3m | qlike | 484 | 0.130267 | 0.136023 | [-0.15404206947818722, 0.39375463904022096] |
| outright_3m | quadratic | 484 | -7.44415e+12 | 5.23154e+13 | [-111890118567187.53, 87610635626521.31] |
| spread_3m_6m | qlike | 484 | 0.952961 | 0.426141 | [0.18776715756133833, 1.839482306926703] |
| spread_3m_6m | quadratic | 484 | -3.90677e+11 | 1.21984e+11 | [-665443057264.4683, -176561001815.96085] |

## Matched transport ablation

All-node reconstruction, fixed nominal windows. The table isolates adjusted versus A=0 factor forecasts on identical paired observations; it does not compare separately tuned pipelines.

| window | exposure | count | mean | bootstrap_percentile_95 |
| --- | --- | --- | --- | --- |
| 16 | outright_3m | 484 | -0.00104938 | [-0.0029836866053074805, 0.0003766834094093128] |
| 16 | spread_3m_6m | 484 | -0.019554 | [-0.06475028931454324, 0.020923454933053046] |
| 16 | butterfly_3m_6m_9m | 484 | 0.0451675 | [0.02147830222976421, 0.07369414680773471] |
| 32 | outright_3m | 484 | -0.000791151 | [-0.0019691422218801277, 0.000248209583105393] |
| 32 | spread_3m_6m | 484 | -0.00132208 | [-0.021132574644136683, 0.018583582886036242] |
| 32 | butterfly_3m_6m_9m | 484 | 0.0413657 | [0.028204405688102795, 0.05568670125465712] |
| 63 | outright_3m | 484 | 0.000405297 | [-0.000501732124109962, 0.0013042136912765141] |
| 63 | spread_3m_6m | 484 | 0.0209979 | [0.0002722891943210505, 0.056185653909442454] |
| 63 | butterfly_3m_6m_9m | 484 | 0.0500802 | [0.0249004209474559, 0.08558435130369599] |
| 126 | outright_3m | 484 | -3.22483e-05 | [-0.0005607613380700567, 0.00050161842967755] |
| 126 | spread_3m_6m | 484 | 0.00818178 | [-0.0019658420760519293, 0.020516385799957626] |
| 126 | butterfly_3m_6m_9m | 484 | 0.0375993 | [0.020924673591640307, 0.0559339349944153] |
| 252 | outright_3m | 484 | -0.000141869 | [-0.00036864749200331146, 5.53862771191218e-05] |
| 252 | spread_3m_6m | 484 | 0.00530471 | [-0.001738784855601223, 0.013652889172699703] |
| 252 | butterfly_3m_6m_9m | 484 | 0.0322563 | [0.019004774275689435, 0.046904885206208585] |

Full mask/short-history, available-method and numerical diagnostics are in attribution.json. At16 intervals the unchanged20-pair direct minimum makes raw baselines unavailable; factor comparisons remain available.

## Revised joint-risk policies

mean validation QLIKE across outright, spread and butterfly on identical validation origins within each rule; exact ties longer window then calendar clock then smaller shrinkage

| method | window | clock | shrinkage | validation_origins | last_validation_outcome |
| --- | --- | --- | --- | --- | --- |
| factor | 63 | calendar | 0 | 231 | 2023-12-29 |
| joint | 63 | calendar | 0 | 231 | 2023-12-29 |
| direct | 63 | calendar | 0 | 231 | 2023-12-29 |
| source | 252 | calendar | 0 | 231 | 2023-12-29 |

Joint factor/residual second moments retain cross terms and PSD feasibility. Pooling normalized innovations across unequal calendar intervals is an empirical local approximation, distinct from the original interval-integrated source-C model. Direct/factor/joint share historical complete cases at a fixed window. Source uses its own full coefficient window.

| exposure | method | common_count | qlike | realized_over_predicted_second_moment |
| --- | --- | --- | --- | --- |
| outright_3m | direct | 484 | 17.2774 | 1.09 |
| outright_3m | factor | 484 | 17.2728 | 1.10279 |
| outright_3m | joint | 484 | 17.2776 | 1.07648 |
| outright_3m | source | 484 | 17.3961 | 1.1857 |
| spread_3m_6m | direct | 484 | 13.6371 | 0.456263 |
| spread_3m_6m | factor | 484 | 15.1725 | 2.21831 |
| spread_3m_6m | joint | 484 | 13.7103 | 0.406791 |
| spread_3m_6m | source | 484 | 14.5615 | 2.27944 |
| butterfly_3m_6m_9m | direct | 484 | 15.0665 | 0.658215 |
| butterfly_3m_6m_9m | factor | 484 | 23.7601 | 7.56348 |
| butterfly_3m_6m_9m | joint | 484 | 15.069 | 0.625367 |
| butterfly_3m_6m_9m | source | 484 | 20.9483 | 6.93341 |

## Bounded hedge decisions

One near contract and two hedge positions in[−2,2], chosen from past information. Common evaluation origins, actual fixed-contract next-session settlement P&L. Position changes are maturity-slot instability proxies, excluding contract-switch, roll and execution costs.

| method | common_count | realized_rms_cny | reduction_vs_unhedged | mean_position_change_proxy |
| --- | --- | --- | --- | --- |
| unhedged | 484 | 3487.1 | 0 | None |
| direct | 484 | 501.431 | 0.979323 | 0.0541029 |
| factor | 484 | 684.946 | 0.961418 | 0.0720462 |
| joint | 484 | 533 | 0.976637 | 0.0643819 |
| source | 484 | 624.091 | 0.967969 | 0.0393728 |

| method | reference | count | mean | bootstrap_percentile_95 |
| --- | --- | --- | --- | --- |
| factor | direct | 484 | 217718 | [83309.11816469523, 398474.22926803364] |
| joint | direct | 484 | 32656.5 | [17544.871922151367, 47609.35404987494] |
| source | direct | 484 | 138057 | [23501.104908564943, 298162.9913140412] |

## Interpretation and limitations

Residual risk explains a major weakness of smooth-factor forecasts. Repairing it does not establish superiority over direct risk estimation. Calendar/session clocks have different effects across exposures; selection remains fixed on validation data. Price reconstruction, numerical convergence and economic risk adequacy are separate gates.

See the six executed notebooks and docs/empirical-research-protocol.md for mathematical targets, coverage, calibration, influence, clock-gap and rank diagnostics. The original benchmark is unchanged. No physical forcing identification, infinite-dimensional statistical guarantee, trading profitability or execution-aware backtest is claimed.

Measured suite runtime excluding archive parsing: 228.59seconds on macOS-15.0.1-arm64-arm-64bit-Mach-O. This is a single environment measurement, not a hardware-independent performance claim.
