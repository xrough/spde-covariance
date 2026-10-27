# Copper pipeline diagnostics: reconstruction, source estimation and residual risk

**Question:** which stages of the implemented transport risk pipeline change
next-session copper risk forecasts, and in which direction?

[Notebook 08](../notebooks/08_copper_mechanism_attribution.ipynb) ·
[Numerical report](../experiments/results/copper-mechanisms/report.md) ·
[Earlier short-history study](short_history.md)

This extends the earlier masked-history experiment to 2016–2025. The real-data
comparisons identify effects of specified **modeling policies**. Latent curve
coefficients, forcing covariance and residual covariance are not observed, so
the experiment cannot assign a unique percentage of statistical error to each
stage. Separate synthetic controls have known truth and illustrate that
distinction.

## Data boundary and counting conventions

2016 supplies initial history. Origins are scheduled every five trading sessions
over 2017–2025, anchored once at the start of that evaluation calendar. There is
no outcome-based tuning or newly untouched validation claim: this is a rolling,
descriptive diagnostic study that includes previously inspected years. **2026
is held out:** its archive is not parsed, incorporated into reconstruction or
used for scores or model selection.

The archive importer distinguishes data rows from exchange footer material;
unexpected nonblank rows are not silently accepted as observations. The source
archives and import policy are recorded in the run manifest. Raw archived
integer volume and open-interest counts remain unchanged. An explicit
`single_sided_counts()` view divides counts by two only where the workbook
declares double-sided counting; the view allows half-counts. No year or filename
heuristic fills in missing declarations. Workbook notes identify double-sided
counts in 2016–2019 and single-sided counts through October 2024; November and
December 2024 and the 2025 XLSX files omit the note, so their counting convention
is recorded as unknown. Requesting a normalized view containing unknown rows
raises an explicit unavailable error; it never returns guessed counts.
Settlement prices, the five-tonne multiplier and price changes are never
rescaled by this view.

This experiment only uses positivity of volume and open interest for donor
eligibility. Multiplication by a positive convention factor leaves that rule
unchanged; it would matter for later studies using liquidity magnitudes or
weights. The notebook reports calendar coverage and the convention metadata.

## Fixed information and risk targets

The current contracts nearest 90, 180 and 270 days are selected at each origin.
Each is treated as the target in turn. Its permitted history is 0, 5, 20 or all
available changes inside a maximum 252-interval window. The current settlement
and contemporaneous metadata are visible at every budget. Older target prices
are removed before reconstruction, scaling and every fit. Issuers receive only
the masked prefix; future same-ID prices enter a separate scorer.

Differences use adjacent original sessions and the same actual contract at both
endpoints. Missing observations are never bridged. The maturity coordinate is
the contract-month 15th, rather than an exact expiry or delivery timestamp.
ACT/365.25 normalization and a 63-session EWMA half-life are fixed. The model
dimension is three in the primary comparison, with nested two- and four-mode
Laguerre representations as prespecified sensitivities.

The target is the uncentered next-session price-change second moment. A variance
interpretation requires negligible conditional means. CNY P&L second moments
multiply price-change moments by $5^2$. Report target outright, target/nearest
spread, original near/middle/far butterfly, full triad quadratic moment score
and bounded hedging with two hedge positions in $[-2,2]$. QLIKE retains the
earlier CNY² floor of 25. Settlement outcomes omit execution costs, rolls,
capacity, fills and margin.

## Reconstruction control: change only the price representation

The raw maturity-pooling policy selects historical donor contracts nearest the
same maturity slots and blends its second moment with that of the current triad
using $n/(n+20)$. The reconstructed version preserves exactly:

- The historical selected actual contract IDs and adjacent intervals.
- Donor eligibility, calendar ages, EWMA weights and complete-case counts.
- The own-history blend count and weight.

It replaces each observed same-ID price change with the difference between the
two fitted endpoint prices. In the invariant Laguerre family this equals

```math
\widehat F_i(T)-\widehat F_{i-1}(T)
=E(T-t_i)\bigl(\widehat a_i-M_{\Delta_i}\widehat a_{i-1}\bigr).
```

Thus the matched comparison tests the effect of passing observed movements
through the chosen reconstruction. It does not simultaneously change maturity
mapping or add transport drift. Report fitted, residual and **signed cross**
second moments; they add to the raw second moment. In-sample quote error,
leverage and donor-excluded diagnostics describe the reconstruction but do not
reveal the true continuum approximation error.

## Source control: hold reconstructed innovations fixed

For the same fitted coefficients, form

```math
z_i=\frac{\widehat a_i-M_{\Delta_i}\widehat a_{i-1}}{\sqrt{\Delta_i}}.
```

Compare direct EWMA innovation second moments with the source-C estimator:

```math
\widehat H_{\rm direct}=\sum_i w_i z_i z_i^\top,
\qquad
\widehat H_{\rm source}(\delta)
=\frac1\delta\int_0^\delta M_u\widehat C M_u^\top\,du.
```

The source fit retains relative ridge $10^{-3}$, training-scale trace cap 100
and relative numerical covariance-error budget $10^{-5}$. Its comparison with
direct H includes integration assumptions **and** source regularization and
constraints. It is a source-estimation policy contrast, not a pure causal test
of transport or an assertion that optimizer error is statistical error.

With unequal intervals, direct H is a historical mixture of interval-specific
normalized moments. The full-calendar comparison is therefore accompanied by a
matched one-calendar-day control: both rows train only on one-day intervals and
are compared only for one-day future intervals. Excluded intervals retain their
original calendar positions and receive zero weight. Keeping the full state
chronology prevents spurious returns across excluded weekends or holidays.

## Residual control: six cells with shared fitted statistics

For each reconstruction and interval context, fit the joint weighted moment of
the main innovations and four piecewise-linear residual maturity coefficients:

```math
M=\begin{pmatrix}H&G\\G^\top&R\end{pmatrix},\qquad
A=G^\top H^\dagger,\qquad \Omega=R-G^\top H^\dagger G.
```

| Main covariance policy | Factor only | Correlated residuals and cross terms | Also diagonal correction |
| --- | --- | --- | --- |
| Source-C integration | $B H_f B^\top$ | $(B+\Psi A)H_f(B+\Psi A)^\top+\Psi\Omega\Psi^\top$ | Previous cell $+D$ |
| Direct innovation EWMA | Same form | Same form | Same form |

Each covariance is multiplied by the future ACT/365.25 interval. For a row
comparison, only $H_f$ changes: reconstruction, training joint moment, Schur
blocks and donor-excluded diagonal correction remain frozen. Column contrasts
add the named correction. Comparing those effects across rows reveals
interactions rather than forcing an additive error attribution.

The diagonal D is fitted from leave-one-donor-out predictive errors. The donor
is absent from both endpoint curve fits and the interval residual fit. Its
predictive error can contain estimation error and representation mismatch;
**D is not identified latent noise or physical SPDE forcing**. Support cutoff
$10^{-10}$, PSD checks, component-sum residuals, conditioning, source certificates
and unavailable results are retained. Four residual features plus a diagonal
remainder remain an explicit restriction.

## Paired reporting and synthetic truth

Average target comparisons within each origin before pooling dates. Repeat
appearances of contracts and budgets are dependent. Calendar-origin blocks are
resampled jointly across policies, budgets, dimensions and contexts: four
scheduled origins are the primary block (approximately twenty sessions), with
1, 2 and 8 as sensitivities. Unavailable one-day outcomes remain absent within
the master calendar rather than compressing time. The 95% percentile ranges are
descriptive dependence checks on this sample, not independent Monte Carlo
confidence statements. Report availability before rankings, and retain annual
and maturity strata alongside pooled summaries.

Two analytic controls use no market files. An exact three-mode transport model
provides known coefficients and source C; independently reconstructed spatial
snapshots allow direct measurement of reconstruction and source errors. A
separate Gaussian increment control has correlated factor/residual blocks,
known Schur components and a known diagonal remainder. It verifies the effect
of retained or omitted cross terms without claiming a level-consistent residual
SPDE. Synthetic Monte Carlo standard errors use independent replications;
calendar-origin bootstrap ranges in copper have a different interpretation.

## Reproduction and evidence

Run the analytical controls without SHFE files:

```sh
uv run python examples/mechanism_oracle.py --smoke --output /tmp/mechanism-oracle.json
```

The empirical command and frozen full/smoke settings are listed in the generated
report. Numerical source is committed before the full run. Public evidence
contains checksummed aggregates, protocol, environment, archive identities and
source hashes. Raw settlements, contract-level forecasts, individual P&Ls and
hedge positions stay in the ignored local run. Notebook 08 reads verified
aggregates and never opens an exchange archive. Historical evidence bundles
remain unchanged.

This study can determine which implemented changes improve or worsen measured
risk performance on 2016–2025. It cannot establish general SPDE superiority,
identify physical forcing from settlements or replace a future frozen
confirmation on the held-out 2026 sample.
