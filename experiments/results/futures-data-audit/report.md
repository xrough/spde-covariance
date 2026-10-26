# SHFE copper: a bounded futures-data feasibility audit

Six official settlement reports and two metadata snapshots establish parser
feasibility and native coverage only. No covariance or hedging performance is
estimated. Raw exchange files are not redistributed.

| Window | Contracts/date | Union/common | Settlement cells | Joint increment-pair counts |
|---|---:|---:|---:|---|
| 2025-10-14 to 2025-10-16 | 12 | 13 / 11 | 36 / 39 | 0:12, 1:11, 2:55 |
| 2025-10-20 to 2025-10-22 | 12 | 12 / 12 | 36 / 36 | 2:66 |

The first window spans expiry of `cu2510` and listing of `cu2610`, which never
overlap in these three dates. All reported rows have settlement, volume and
open interest, with positive volume. Absence is contract lifecycle, not random
quote thinning. The second window is dense. Do not call October 16-to-20 a
one-day increment: October 17 was not requested.

This sample does not prove permanent covariance nonidentification in a longer
history or under changing maturity observation maps. Native absence and
controlled masking are different studies; two increments cannot establish
statistically reliable covariance.

Contract snapshots provide listing, scheduled expiry and physical delivery
dates. The official parameter page warns that statutory-holiday adjustments
are omitted. Keep these as scheduled, as-of metadata until calendar checks.
A physical delivery period is not automatically an average-price observation.
`PRESETTLEMENTPRICE` on a new contract is a reference, not a prior observed
settlement.

Sample-period copper units are CNY/tonne with five tonnes per contract.
Signed counts times multiplier times raw price changes produce CNY P&L.
Reports supply no bid-ask spreads, execution costs, independent noise estimate
or intraday trading availability. The adapter declares date-resolution availability.

## Sources and reproduction

- [Daily reports](https://www.shfe.com.cn/reports/tradedata/dailyandweeklydata/).
- [Contract parameters](https://www.shfe.com.cn/reports/businessdata/prmsummary/?query_date=20251014&query_options=1&query_params=currinstrumentprop_f&query_product_code=cu_f).
- [Sample-period rules](https://www.shfe.com.cn/regulation/exchangerules/historicalversion/202508/t20250807_828528.html).
- [Trading dates/hours](https://www.shfe.com.cn/services/calenderandholidays/tradinghours/).
- [Use terms](https://www.shfe.com.cn/disclaimer/): access is not an open redistribution license.

[Expiry aggregate](expiry-window.json), [dense aggregate](dense-window.json)
and [manifest](manifest.json) retain source URLs/hashes, identifiers, missingness,
calendar gaps, pair counts and the audit-code revision. Reproduce with lawfully
obtained local reports:

```sh
python scripts/audit_shfe.py /local/kx20251014.dat /local/kx20251015.dat /local/kx20251016.dat --output /tmp/expiry-audit.json
python scripts/audit_shfe.py /local/kx20251020.dat /local/kx20251021.dat /local/kx20251022.dat --output /tmp/dense-audit.json
```

See [CSV/modeling conventions](../../../docs/futures-data.md).
