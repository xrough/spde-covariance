"""Audit local, lawfully obtained SHFE daily JSON reports without downloading data.

Only source hashes, identifiers, missingness and overlap counts are emitted.
Raw settlements are not copied into the output. The provided report dates are
the audit calendar; calendar gaps are reported explicitly, never called one day.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def _present(value):
    try:
        return value is not None and value != "" and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def audit_reports(paths, *, product="cu_f"):
    """Return an aggregate audit of local reports, validating date and uniqueness."""
    reports, panels, universe = [], {}, set()
    for supplied in paths:
        path = Path(supplied)
        raw = path.read_bytes()
        data = json.loads(raw)
        day = str(data["report_date"])
        if len(day) != 8 or not day.isascii() or not day.isdigit():
            raise ValueError("report_date must use canonical YYYYMMDD")
        try:
            parsed = datetime.strptime(day, "%Y%m%d")
        except ValueError as error:
            raise ValueError("report_date must use canonical YYYYMMDD") from error
        if parsed.strftime("%Y%m%d") != day:
            raise ValueError("report_date must use canonical YYYYMMDD")
        if day in panels:
            raise ValueError("duplicate report_date")
        rows = [
            row
            for row in data["o_curinstrument"]
            if str(row.get("PRODUCTID", "")).strip() == product
            and str(row.get("DELIVERYMONTH", "")).isdigit()
            and len(str(row["DELIVERYMONTH"])) == 4
        ]
        contracts = {str(row["DELIVERYMONTH"]): row for row in rows}
        if len(contracts) != len(rows):
            raise ValueError("duplicate delivery month within product/date")
        universe.update(contracts)
        panels[day] = {
            key for key, row in contracts.items() if _present(row.get("SETTLEMENTPRICE"))
        }
        reports.append(
            {
                "valuation_date": day,
                "source_url": "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx"
                + day
                + ".dat",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "exchange_update_date": data.get("update_date"),
                "contract_count": len(contracts),
                "delivery_month_codes": sorted(contracts),
                "missing_fields": {
                    field: sum(not _present(row.get(field)) for row in rows)
                    for field in ("SETTLEMENTPRICE", "VOLUME", "OPENINTEREST")
                },
                "zero_volume_contracts": sum(
                    _present(row.get("VOLUME")) and float(row["VOLUME"]) == 0 for row in rows
                ),
            }
        )
    if not reports:
        raise ValueError("at least one report is required")
    dates = sorted(panels)
    union = sorted(universe)
    pairs = list(itertools.combinations(union, 2))
    increments = [panels[a] & panels[b] for a, b in zip(dates[:-1], dates[1:], strict=True)]
    level_counts = Counter(sum(a in panels[d] and b in panels[d] for d in dates) for a, b in pairs)
    increment_counts = Counter(
        sum(a in present and b in present for present in increments) for a, b in pairs
    )
    return {
        "product": product,
        "reports": sorted(reports, key=lambda item: item["valuation_date"]),
        "audit_calendar": dates,
        "union_contract_count": len(union),
        "intersection_contract_count": len(set.intersection(*(panels[day] for day in dates))),
        "observed_settlement_cells": sum(len(panels[day]) for day in dates),
        "possible_union_cells": len(union) * len(dates),
        "pair_level_count_histogram": dict(sorted(level_counts.items())),
        "pair_increment_count_histogram": dict(sorted(increment_counts.items())),
        "interval_calendar_days": [
            (datetime.strptime(b, "%Y%m%d") - datetime.strptime(a, "%Y%m%d")).days
            for a, b in zip(dates[:-1], dates[1:], strict=True)
        ],
        "interpretation": "tiny feasibility audit of supplied report dates; no hedge performance, "
        "statistical identification, full-history coverage, or license conclusion",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--product", default="cu_f")
    parser.add_argument("--output", type=Path, help="aggregate output; default is stdout")
    args = parser.parse_args()
    result = json.dumps(audit_reports(args.reports, product=args.product), indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
