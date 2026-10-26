# What the first SHFE experiment found

The small transport model is a useful **empirical baseline, but its local
three-mode calibration does not adequately forecast calendar-spread and
butterfly risk in this sample**. Direct same-contract covariance has lower
out-of-sample QLIKE. The experiment identifies concrete model weaknesses;
it does not establish an SPDE forecasting advantage.

See the [executed notebook](../notebooks/shfe_empirical.ipynb),
[generated report](../experiments/results/shfe-cu/report.md) and
[protocol](shfe-empirical.md). Results use official copper settlements from
2022–2025: 969 sessions, 11,628 contract observations and 60 distinct IDs.
The 484 test origins occur in 2024–2025. No raw quotes or individual return
series are included in public evidence.

## Main measured result

The SPDE method selected 252 trailing intervals on 2023 validation data;
direct covariance and 10% diagonal shrinkage independently selected 63.
All listed methods score the same 484 subsequent observations per exposure.

| Portfolio | Local SPDE: realized / predicted variance | Direct: realized / predicted variance | Paired QLIKE, SPDE minus direct | Block-bootstrap SE |
| --- | ---: | ---: | ---: | ---: |
| One near contract | 1.186 | 1.096 | 0.130 | 0.136 |
| Near–middle spread | 2.279 | 0.455 | 0.953 | 0.426 |
| Near–2 middle+far butterfly | 6.933 | 0.658 | 5.882 | 1.874 |

Ratios above one indicate aggregate underprediction of squared P&L. Direct
covariance is conservative for the spread and butterfly in this sample,
and also has the lower QLIKE. A ratio closer to one does not alone establish
better conditional forecasts. QLIKE comparisons use paired observations;
bootstrap uncertainty concerns time-series score differences, not a latent
operator or future coverage guarantee.

The local SPDE forecast's RMS risks are approximately CNY 3,202, 334 and 402,
against realized RMS P&Ls of CNY 3,487, 504 and 1,058, respectively. These are
one-session risks for fixed actual contracts, using five-tonne multipliers.
They are neither annual P&Ls nor rolled continuous-contract returns.

## What the checks establish

All source fits satisfy the declared numerical budget, with no trace-cap
violations. The largest propagated numerical variance bound is below 0.65%
of a forecast's variance/floor scale, much smaller than the measured risk
misspecification. Optimizer convergence does not validate the financial model.

The median curve reconstruction RMSE is CNY 48.58 per tonne. A small error
relative to the large copper price level can still matter greatly for a
spread or butterfly, where common level movements cancel. The reconstruction
diagnostic decomposes actual P&L into fitted-curve and residual changes and
retains their cross term. It is a subsequent model diagnostic, not a forecast
input or a known measurement-noise correction.

Increasing the basis from three to four modes slightly lowers median level-fit
RMSE (48.58 to 44.22), while raising median design condition number from about
70 to 1,611. Butterfly risk remains substantially underpredicted. Two modes
are particularly poor for butterfly risk: their realized/predicted ratio is
about 225. Stronger smoothing also worsens butterfly risk. More coordinates
or a good fit to price levels are insufficient acceptance criteria.

Ordinary unadjusted coefficient increments and the adjusted local model have
very similar daily risk scores. This study therefore provides no measured
forecasting benefit from transport adjustment at this horizon. The exact
transport identity remains the correct way to align fixed contracts.

Calendar-time scaling also deserves attention. For outright risk, the
realized/predicted ratio is 1.45 over one-day intervals and 0.63 over two- to
three-day intervals. Eleven longer holiday intervals give a ratio of 1.99.
These small groups do not by themselves establish a preferred clock.

Fixed 10% shrinkage toward a diagonal covariance is conservative for spread
risk: it weakens the high positive cross-contract correlations that cancel
under spread positions. A generic covariance regularizer can therefore have
an undesirable effect on the financial risk target.

## Next implementation targets

1. **Represent residual contract risk explicitly.** Keep the transport-factor
   covariance and add a separately calibrated residual covariance, rather than
   treating the smooth reconstruction as clean observed truth. Select all
   settings on validation data and compare with direct risk forecasts.
2. **Separate clocks.** Keep physical time for maturity transport and compare
   calendar-, business- and event-time diffusion scales. Reuse the same fixed
   contracts, origins and scores; freeze choices before the test period.
3. **Use risk-aware regularization and adaptation.** Compare factor-plus-residual
   shrinkage and local volatility dynamics. Evaluate spread and butterfly
   targets directly, not only a matrix norm or level-fit error.
4. **Broaden the evidence.** Audit release vintages and maturity conventions,
   add block-length and period sensitivity, then another SHFE product. Retain
   the current baseline and test chronology rather than replacing a difficult
   comparison after observing its results.

Real-data calibration and next-session risk evaluation are now implemented.
Execution, trading costs, profitable-strategy backtesting, sparse noisy-level
innovation recovery and infinite-dimensional statistical guarantees remain
outside this first empirical experiment.
