import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "audit_shfe", Path(__file__).resolve().parents[1] / "scripts" / "audit_shfe.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_reports = _MODULE.audit_reports


def _report(path, day, months):
    rows = [
        dict(PRODUCTID="cu_f", DELIVERYMONTH=month, SETTLEMENTPRICE=price, VOLUME=1, OPENINTEREST=2)
        for month, price in months
    ]
    rows.append(dict(PRODUCTID="cu_f", DELIVERYMONTH="小计", SETTLEMENTPRICE=""))
    path.write_text(json.dumps(dict(report_date=day, o_curinstrument=rows)))
    return path


def test_audit_native_entry_exit_does_not_emit_raw_prices_and_retains_gaps(tmp_path):
    paths = [
        _report(tmp_path / "one.dat", "20251014", [("2510", 100), ("2511", 200)]),
        _report(tmp_path / "two.dat", "20251015", [("2510", 101), ("2511", 201)]),
        _report(tmp_path / "three.dat", "20251020", [("2511", 202), ("2610", 203)]),
    ]
    result = audit_reports(paths[::-1])
    assert result["observed_settlement_cells"] == 6
    assert result["possible_union_cells"] == 9
    assert result["pair_level_count_histogram"] == {0: 1, 1: 1, 2: 1}
    assert result["pair_increment_count_histogram"] == {0: 2, 1: 1}
    assert result["interval_calendar_days"] == [1, 5]
    assert all(len(report["sha256"]) == 64 for report in result["reports"])
    assert "SETTLEMENTPRICE" not in json.dumps(result).replace('"SETTLEMENTPRICE": 0', "")
    assert result["reports"][0]["contract_count"] == 2


def test_audit_missing_settlement_is_not_counted_as_joint_observation(tmp_path):
    one = _report(tmp_path / "one.dat", "20251014", [("2510", ""), ("2511", 200)])
    result = audit_reports([one])
    assert result["reports"][0]["missing_fields"]["SETTLEMENTPRICE"] == 1
    assert result["observed_settlement_cells"] == 1
    with pytest.raises(ValueError, match="duplicate report_date"):
        audit_reports([one, one])


def test_audit_rejects_noncanonical_report_date_before_calendar_comparison(tmp_path):
    one = _report(tmp_path / "one.dat", "2025101", [("2510", 100)])
    two = _report(tmp_path / "two.dat", "20251001", [("2510", 101)])
    with pytest.raises(ValueError, match="YYYYMMDD"):
        audit_reports([one, two])
