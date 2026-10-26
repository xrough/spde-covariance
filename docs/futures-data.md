# Contract settlement adapter and public-data feasibility audit

`spde_covariance.futures` supplies an input boundary for a later futures-price
study. It preserves actual contract identities, source provenance, publication
availability, units, and the caller's valuation calendar. It does not calibrate
an SPDE, recover instantaneous covariance from daily settlements, or establish
empirical hedging gains. The [offline example](../examples/futures_pipeline.py)
uses synthetic CSVs throughout.

The separate [first SHFE empirical study](shfe-empirical.md) now implements
archive ingestion, finite-model curve reconstruction, interval-aware source
calibration and next-session risk evaluation. Its [findings](shfe-findings.md)
retain retrospective-vintage and maturity-coordinate limitations. The generic
CSV adapter below continues to enforce its original input boundary.

## CSV boundary

`SettlementPanel.from_csv(prices_path, contracts_path, valuation_dates=...)`
requires an explicit, increasing calendar. A declared date with no row stays
missing. Rows outside the declared calendar and duplicate contract-date rows
are errors. There is no interpolation, forward filling, continuous-front series,
or artificial roll. Identifiers are case-sensitive and must join exactly.

The long price CSV requires these columns:

```csv
contract_id,valuation_date,settlement,available_date,source_url,source_sha256
```

`volume` and `open_interest` are optional numeric columns; blank values remain
unknown. Availability applies to any supplied settlement, volume, or open-interest
field, including rows whose settlement is missing. When fields were published
at different times, supply the latest availability date for that cell; this
conservative initial schema does not infer separate field publication times.
Rows with any market field require an availability date and source report
URL/hash. A blank settlement is missing, including when a source provides a
nonempty previous-settlement reference. `PRESETTLEMENTPRICE` and any other extra
columns are ignored. Source digests use lowercase SHA-256. Duplicate header
columns are rejected instead of silently selecting the last duplicate value.

The metadata CSV requires:

```csv
contract_id,exchange,product,currency,price_unit,multiplier,listing_date,delivery_month,metadata_asof,source_url,source_sha256,calendar_adjusted
```

Optional `expiry_date`, `delivery_start`, and `delivery_end` may be blank.
Dates use `YYYY-MM-DD`, delivery months use `YYYY-MM`, and `calendar_adjusted`
is explicitly `true` or `false`. Both physical delivery endpoints must either
be supplied or be missing. `multiplier` is the physical quantity per contract
when prices are quoted in currency per physical unit; for example, five
tonnes/contract with `price_unit=CNY/tonne`. All columns in a panel must share
currency and price unit. Conversions must be performed explicitly upstream.

A `ContractSpec` is one immutable metadata snapshot, with `metadata_asof`
declaring when that snapshot became available. This initial adapter does not
reconstruct a versioned security master. A late snapshot cannot be presented
as known throughout an earlier backtest. Expiry and delivery dates remain
metadata: they never select a point or delivery-average observation map.
In particular, physical delivery logistics do not establish that the quoted
futures settlement is a uniform average of a forward curve over those dates.

Arrays are detached, immutable snapshots. Input CSV hashes and supplied report
hashes are retained as provenance. Hashes detect changes; they do not authenticate
the source or grant permission to redistribute its contents.

## Past-only changes and the workflow bridge

```python
from spde_covariance.futures import SettlementPanel

panel = SettlementPanel.from_csv(prices_csv, contracts_csv, valuation_dates=calendar)
training = panel.normalized_changes(
    asof="2025-01-16",
    integrated_drift="zero",
    price_assumption="reduced_form_settlements",
)
```

`panel.asof(cutoff)` removes future calendar dates and unavailable metadata and
masks any quote published after the cutoff. `normalized_changes` applies this
restriction before extraction, so the resulting input can be passed to selection
or refitting without exposing later prices. Availability is recorded at date
resolution: a same-date quote means available by that day's end, not before the
close or before a particular intraday decision. An intraday study needs a finer
availability boundary upstream.

For each contract and consecutive pair of declared dates, extraction uses

```text
dt = actual calendar days / 365.25
y = (settlement[end] - settlement[start] - integrated_drift) / sqrt(dt)
```

Both prices must be observed for that same contract. Contract metadata must be
available and the contract listed by the interval start. Direct construction
of `SettlementChanges` enforces these boundaries as well. The mask is retained;
`joint_counts` counts
jointly available changes, including zero counts. If a declared intermediate
date has no quote, neither adjacent change for that contract becomes a longer
filled change. If the caller's calendar itself skips dates, the actual longer
duration remains. For example, Friday to Monday retains three calendar days.
The existing high-level `fit_innovation_covariance` rejects unequal durations;
this adapter does not rescale them into a common-step model.

`integrated_drift="zero"` explicitly declares zero drift. Alternatively, supply
one integrated drift in price units per interval and retained contract. No drift
is estimated automatically. `clean_settlement_levels` is a caller assumption;
`reduced_form_settlements` leaves the target as covariance of normalized
settlement changes. Neither option asserts that the prices are coefficients of
a clean H-valued trajectory or removes transport from them.

`training.to_observed_batch(...)` requires all of:

- Caller-supplied `operators`, one full contract-row map per interval, in
  `training.contract_ids` order before masking.
- `map_asof_dates`, no later than each interval's start; a `basis_id`; a
  nonempty `observation_map_assumption`; and an explicit `curve_unit` matching
  the settlement price unit.
- `noise_assumption="none"`, or `"known_fresh_innovation_noise"` together with
  one supplied PSD noise covariance per interval, in normalized-change units
  squared and full contract-row order.

Maps are masked with the observed contract rows. An entirely empty interval
raises an error; it is not silently dropped or bridged. The result is the existing
`ObservedInnovationBatch`, retaining ACT/365.25 intervals, provenance, and the
declared assumptions. The adapter records that physical delivery does not imply
a delivery-average observation. It performs no automatic maturity-to-basis map.

Noise covariance here is **fresh normalized innovation noise**. Differencing
noisy settlement levels generally creates serially dependent errors and does
not justify this correction. The adapter never estimates such a covariance
from volume or spread proxies. Daily exchange settlements can also average
transactions over time; their relation to an instantaneous latent curve requires
a separate measurement model.

## Contract-count risk and P&L

`changes.pnl_loading(positions, operator)` multiplies signed contract counts
by their contract multipliers and the supplied observation map. For normalized
price-change covariance `V`, currency variance over duration `dt` is
`dt * loading @ V @ loading`. Positions and operators must refer to the same
contract order. There is no second annualization.

`changes.contract_pnl(positions, interval_index=i)` instead uses the **raw**
same-contract settlement differences, without subtracting drift. A missing
held-contract change is an error. For two contracts quoted in CNY/tonne, with
multipliers five and ten tonnes/contract, positions `+2` and `-1`, and raw changes
`+2` and `+4`, P&L is `2*5*2 - 1*10*4 = -20 CNY`. Currency conversions, fees,
margin/funding cash flows, trading constraints, and execution costs are outside
this adapter. The synthetic example demonstrates past-only fitting, the
point-covariance hedge interface, and exact multiplier units, without claiming
that fractional toy positions describe an executable strategy.
Nonfinite currency results after multiplier conversion raise an error.

## Actual SHFE audit: six dates, one product

On 2026-10-25, a bounded read-only audit retrieved Shanghai Futures Exchange
copper outright futures reports for **2025-10-14, 15, 16, 20, 21, and 22**, plus
contract metadata snapshots for October 14 and 16. Raw reports were kept in
temporary local storage and are not bundled with this project. This is a tiny
data-feasibility sample, not a statistical study or hedge-performance result.

The official [daily report page](https://www.shfe.com.cn/reports/tradedata/dailyandweeklydata/)
uses the endpoint
`https://www.shfe.com.cn/data/tradedata/future/dailydata/kxYYYYMMDD.dat`.
The official [contract parameter page](https://www.shfe.com.cn/reports/businessdata/prmsummary/)
uses `/data/busiparamdata/future/ContractBaseInfoYYYYMMDD.dat`.
The older `/data/dailydata/kx/kxYYYYMMDD.dat` route returned 404 during this audit.
Public reports provide settlement, volume, and open interest for each actual
delivery month; subtotal rows must be excluded.

| Window | Contracts/date | Union / common contracts | Observed settlement cells | Native lifecycle absence |
|---|---:|---:|---:|---:|
| October 14–16 | 12 | 13 / 11 | 36 / 39 | 3 / 39 |
| October 20–22 | 12 | 12 / 12 | 36 / 36 | 0 / 36 |

Within reported copper contract rows, settlement, volume, and open interest were
all present, and all volumes were positive on these six dates. In the first
window `cu2510` disappeared after October 15 and `cu2610` first appeared on
October 16. Across the 78 unordered pairs in that window's 13-contract union,
joint settlement-date counts had histogram `{0: 1, 1: 11, 2: 11, 3: 55}`.
Joint same-contract increment counts had histogram `{0: 12, 1: 11, 2: 55}`.
The pair `cu2510`/`cu2610` was never observed together. In the second window,
all 66 pairs had three joint settlement dates and two joint increments.
This is a maturity-lifecycle mask, not independent random thinning. Positive
semidefiniteness cannot create observations for the absent pair.

October 17 was deliberately not requested. The two windows are reported
separately; October 16→20 must not be relabeled as a one-day change. Nor does a
new contract's `PRESETTLEMENTPRICE` establish an observed previous settlement:
the October 16 `cu2610` reference was nonempty although no October 15 `cu2610`
settlement row existed.

The metadata joins yielded exact IDs, listing dates, scheduled expiry dates,
and physical delivery windows. The October 14 snapshot gave `cu2510` expiry
October 15 and delivery October 16–17. The October 16 snapshot gave `cu2610`
listing October 16, expiry October 15, 2026 and delivery October 16–19, 2026.
The official parameter page warns that statutory-holiday adjustments are not
included. Preserve these as dated scheduled metadata and verify the applicable
exchange calendar before calculating maturity; do not assume every supplied
future expiry has already been holiday-adjusted.

The [sample-period copper rules](https://www.shfe.com.cn/regulation/exchangerules/historicalversion/202508/t20250807_828528.html)
specify CNY/tonne quotation, five tonnes/contract, and a CNY10/tonne tick.
The [exchange trading-hours page](https://www.shfe.com.cn/services/calenderandholidays/tradinghours/)
identifies the preceding working day's night session as part of the trading
date. Preserve exchange trading dates instead of assigning all observations
from a midnight calendar boundary. Exact intraday observation timestamps,
bid/ask spreads, execution costs, and independently observed measurement noise
were not supplied by these daily reports.

SHA-256 digests of the downloaded raw report bytes:

| Report | SHA-256 |
|---|---|
| kx20251014.dat | `90fbb3f5c60e6d99756701da485df83f81a11080bc8f958fb67468103303a484` |
| kx20251015.dat | `fb05c15f71cf88acf7a96444e61747f3a4347f1acfc2601d1f4936f1c660d05f` |
| kx20251016.dat | `58fd106508f80ab2f9e4ebf0579aab11a60058c248b9572606a5789381f17e09` |
| kx20251020.dat | `11afd297668d723e36cb49b61f672fd4cb1b5dbbc463d47a118317e8718b878c` |
| kx20251021.dat | `f3761a96034643c8ad8f5b1cb62bd7582fcdda6f6af07896d0f44e84ce02e3ea` |
| kx20251022.dat | `4cba498321412624ae97b916dfdccb77c41434c66776e3ee199df19743452bcf` |
| ContractBaseInfo20251014.dat | `a4f8eeba4c385974de6cd158c1951f917a2faf189e0107327fa8d4fb9facddc7` |
| ContractBaseInfo20251016.dat | `c81450d0fd05237ec10df0b07f4e451375f067d002805c1ce406b474fdd5d0a6` |

`scripts/audit_shfe.py` accepts only existing local raw report paths and emits
aggregate identifiers, source hashes, missingness, overlap histograms, and actual
calendar gaps. It performs no downloads and emits no raw settlement prices:

```sh
python scripts/audit_shfe.py /private/tmp/kx20251014.dat /private/tmp/kx20251015.dat /private/tmp/kx20251016.dat
```

The [SHFE copyright statement](https://www.shfe.com.cn/disclaimer/) permits
noncommercial browsing and downloading subject to its conditions and reserves
rights over market information. It is not an open-data redistribution license.
This project provides code, sources, hashes, and aggregate feasibility findings;
do not infer permission to publish a raw exchange-price dataset from public
accessibility or from the project's software license.
