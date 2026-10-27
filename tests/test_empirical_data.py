"""Synthetic-only SHFE adapter fixtures: no exchange prices are redistributed."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pytest

import spde_covariance.empirical_data as data

HEADER = ["合约", "日期", "结算价", "成交量", "持仓量"]
ENGLISH = ["Contract", "Date", "Settle", "Volume", "OI"]


def _rows():
    return [
        ["所内合约行情报表", "", "", "", ""],
        HEADER,
        ENGLISH,
        ["cu2403", "20240102", 100, 10, 20],
        ["cu2402", "20240102", 101, 11, 21],
        ["ag2402", "20240103", 30, 20, 40],
        ["cu2402", "20240104", 102, 12, 22],
    ]


def _xlsx(rows, *, merges=()):
    """Minimal self-created OOXML workbook, read by the independent Excel engine."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    xml_rows = []
    for i, row in enumerate(rows, 1):
        cells = []
        for j, value in enumerate(row):
            coordinate = f"{chr(65 + j)}{i}"
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{coordinate}"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{coordinate}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                )
        xml_rows.append(f'<row r="{i}">{"".join(cells)}</row>')
    merge_xml = "".join(f'<mergeCell ref="{value}"/>' for value in merges)
    sheet = (
        f'<worksheet xmlns="{ns}"><sheetData>{"".join(xml_rows)}</sheetData>'
        f'<mergeCells count="{len(merges)}">{merge_xml}</mergeCells></worksheet>'
    )
    content = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as workbook:
        workbook.writestr("[Content_Types].xml", content)
        workbook.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" '
            f'Type="{rel_type}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        workbook.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{rel_type}"><sheets>'
            '<sheet name="Synthetic" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" '
            f'Type="{rel_type}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        workbook.writestr("xl/worksheets/sheet1.xml", sheet)
    return payload.getvalue()


def _archive(tmp_path, rows=None, *, name="shfe-futures-2024.zip", members=None, merges=()):
    pytest.importorskip("python_calamine")
    path = tmp_path / name
    if members is None:
        members = {"SHFE/month.xlsx": _xlsx(_rows() if rows is None else rows, merges=merges)}
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)
    return path


def test_rows_are_sorted_but_missing_copper_sessions_are_not_filled():
    panel = data.parse_shfe_rows(_rows())
    assert panel.contract_ids.tolist() == ["cu2402", "cu2403", "cu2402"]
    assert panel.dates.dtype == np.dtype("datetime64[D]")
    assert panel.contract_ids.dtype.kind == "U"
    assert panel.session_dates.astype(str).tolist() == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert "2024-01-03" not in panel.dates.astype(str)
    assert panel.contract_month_dates.astype(str).tolist() == [
        "2024-02-15",
        "2024-03-15",
        "2024-02-15",
    ]
    assert panel.currency == "CNY" and panel.price_unit == "CNY/tonne" and panel.multiplier == 5


def test_missing_and_zero_values_are_retained_and_audited_without_price_repair():
    panel = data.parse_shfe_rows(
        [
            HEADER,
            ["cu2402", "20240102", "--", 0, ""],
            ["cu2402", "20240103", 0, 5, 10],
        ]
    )
    assert np.isnan(panel.settlement[0]) and panel.settlement[1] == 0
    assert panel.volume.tolist() == [0, 5] and np.isnan(panel.open_interest[0])
    counts = panel.provenance["tables"][0]["counts"]
    assert counts["missing_settlement"] == 1 and counts["zero_settlement"] == 1


def test_only_declared_merged_contract_labels_are_expanded():
    rows = [HEADER, ["cu2402", "20240102", 100, 10, 20], ["", "20240103", 101, 11, 21]]
    panel = data.parse_shfe_rows(rows, merged_cell_ranges=[((1, 0), (2, 0))])
    assert panel.contract_ids.tolist() == ["cu2402", "cu2402"]
    assert panel.settlement.tolist() == [100, 101]
    assert panel.provenance["tables"][0]["counts"]["expanded_merged_contract_labels"] == 1
    with pytest.raises(ValueError, match="outside a merge"):
        data.parse_shfe_rows(rows)


def test_missing_contract_anchor_and_inconsistent_merge_fail():
    with pytest.raises(ValueError, match="no anchor"):
        data.parse_shfe_rows(
            [HEADER, ["", "20240102", 1, 1, 1], ["", "20240103", 2, 2, 2]],
            merged_cell_ranges=[((1, 0), (2, 0))],
        )
    with pytest.raises(ValueError, match="inconsistent merged"):
        data.parse_shfe_rows(
            [HEADER, ["cu2402", "20240102", 1, 1, 1], ["cu2403", "20240103", 2, 2, 2]],
            merged_cell_ranges=[((1, 0), (2, 0))],
        )


def test_unknown_schema_cannot_silently_select_wrong_columns():
    with pytest.raises(ValueError, match="no recognized"):
        data.parse_shfe_rows([["product", "day", "price"]])
    with pytest.raises(ValueError, match="exactly one settlement"):
        data.parse_shfe_rows([["合约", "日期", "收盘价", "成交量", "持仓量"]])
    with pytest.raises(ValueError, match="exactly one volume"):
        data.parse_shfe_rows([HEADER + ["Volume"]])


@pytest.mark.parametrize("value", ["20240230", "2024-1-02", "tomorrow", 20240102.5, True])
def test_bad_trade_dates_fail(value):
    with pytest.raises(ValueError, match="trade date"):
        data.parse_shfe_rows([HEADER, ["cu2402", value, 100, 10, 20]])


@pytest.mark.parametrize("contract", ["cu2400", "cu2413", "cu240", "cu202402"])
def test_malformed_copper_contracts_fail(contract):
    with pytest.raises(ValueError, match="copper|delivery month"):
        data.parse_shfe_rows([HEADER, [contract, "20240102", 100, 10, 20]])


@pytest.mark.parametrize("volume,oi", [(-1, 1), (1.5, 2), (1, -1), (1, 2.5)])
def test_counts_cannot_be_negative_or_fractional(volume, oi):
    with pytest.raises(ValueError, match="counts"):
        data.parse_shfe_rows([HEADER, ["cu2402", "20240102", 100, volume, oi]])


def test_options_are_not_mistaken_for_futures_and_numeric_dates_are_explicit():
    panel = data.parse_shfe_rows(
        [
            ENGLISH,
            ["cu2402C70000", "20240102", 1, 1, 1],
            [" CU2402 ", 20240103, 100, "1,000", 2000],
        ]
    )
    assert panel.contract_ids.tolist() == ["cu2402"]
    assert panel.volume.tolist() == [1000]
    assert len(panel.session_dates) == 2


def test_duplicate_keys_and_unlabelled_rows_fail():
    rows = [HEADER, ["cu2402", "20240102", 1, 1, 1]]
    with pytest.raises(ValueError, match="sorted and unique"):
        data.parse_shfe_rows(rows + rows[1:])
    with pytest.raises(ValueError, match="no trade date"):
        data.parse_shfe_rows([HEADER, ["cu2402", "", 1, 1, 1]])


def test_panel_vectors_and_provenance_are_immutable_snapshots():
    panel = data.parse_shfe_rows(_rows())
    for field in (
        "dates",
        "contract_ids",
        "contract_month_dates",
        "settlement",
        "volume",
        "open_interest",
        "session_dates",
    ):
        with pytest.raises(ValueError):
            getattr(panel, field).setflags(write=True)
    original = panel.provenance
    original["units"]["tonnes_per_contract"] = 100
    assert panel.provenance["units"]["tonnes_per_contract"] == 5
    assert json.loads(json.dumps(panel.provenance)) == panel.provenance


def test_midpoint_is_not_an_expiry_filter():
    panel = data.parse_shfe_rows([HEADER, ["cu2401", date(2024, 1, 16), 100, 1, 1]])
    assert panel.contract_month_dates[0] < panel.dates[0]


def test_real_xlsx_reader_merge_metadata_and_archive_provenance(tmp_path):
    path = _archive(
        tmp_path,
        [HEADER, ["cu2402", "20240102", 100, 1, 1], ["", "20240103", 101, 2, 2]],
        merges=["A2:A3"],
    )
    panel = data.load_shfe_cu_archives([path])
    assert panel.contract_ids.tolist() == ["cu2402", "cu2402"]
    assert (
        panel.provenance["archives"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert panel.provenance["tables"][0]["counts"]["expanded_merged_contract_labels"] == 1
    assert "point-in-time not verified" in panel.provenance["data_vintage"]


def test_archive_bounds_preserve_other_product_calendar_and_skip_ine(tmp_path):
    path = _archive(
        tmp_path,
        members={
            "SHFE/one.xlsx": _xlsx(_rows()),
            "INE/one.xlsx": _xlsx(_rows()),
        },
    )
    panel = data.load_shfe_cu_archives([path], start_date="2024-01-03", end_date="2024-01-04")
    assert panel.dates.astype(str).tolist() == ["2024-01-04"]
    assert panel.session_dates.astype(str).tolist() == ["2024-01-03", "2024-01-04"]
    assert len(panel.provenance["tables"]) == 1


def _counting_footer(convention="双边"):
    return [
        ["注：", "1、 合约单位：铜为5吨/手", "", "", ""],
        ["", "2、 报价单位：铜为元/吨", "", "", ""],
        [
            "",
            f"3、 成交量、持仓量、持仓变化单位为手，{convention}计算；成交金额单位为万元，双边计算",
            "",
            "",
            "",
        ],
        ["", "4、 涨跌1=收盘价-前结算； 涨跌2=结算价-前结算", "", "", ""],
    ]


def test_old_metadata_in_date_column_is_recognized_without_truncating_later_data():
    rows = [
        HEADER,
        ENGLISH,
        ["cu1702", "20161229", 100, 11, 21],
        *_counting_footer(),
        HEADER,
        ["cu1702", "20161230", 101, 12, 22],
    ]
    panel = data.parse_shfe_rows(rows)
    assert panel.dates.astype(str).tolist() == ["2016-12-29", "2016-12-30"]
    assert panel.volume.tolist() == [11, 12]
    counting = panel.provenance["tables"][0]["counting"]
    assert counting["volume_convention"] == counting["open_interest_convention"] == "double-sided"
    assert counting["declaration_count"] == 1
    assert panel.provenance["tables"][0]["counts"]["recognized_metadata_rows"] == 4
    normalized = panel.single_sided_counts()
    assert normalized.volume.tolist() == [5.5, 6]
    assert normalized.open_interest.tolist() == [10.5, 11]
    assert normalized.provenance["price_and_multiplier_transformation"] == "none"
    assert panel.settlement.tolist() == [100, 101] and panel.multiplier == 5
    with pytest.raises(ValueError):
        normalized.volume.setflags(write=True)
    snapshot = normalized.provenance
    snapshot["declarations"].clear()
    assert normalized.provenance["declarations"]


def test_short_recognized_metadata_rows_and_single_sided_notes_preserve_raw_counts():
    rows = [
        HEADER,
        ["cu2102", "20210104", 100, 11, 21],
        ["1、 合约单位：铜为5吨/手"],
        ["3、 成交量、持仓量单位为手，单边计算；成交金额双边计算"],
    ]
    panel = data.parse_shfe_rows(rows)
    normalized = panel.single_sided_counts()
    np.testing.assert_array_equal(normalized.volume, panel.volume)
    np.testing.assert_array_equal(normalized.open_interest, panel.open_interest)


@pytest.mark.parametrize(
    "bad_row,match",
    [
        (["cu1702", "3、 成交量、持仓量，双边计算", "", "", ""], "trade date"),
        (["cu1702", "", "", "", ""], "no trade date"),
        (["cu1702", "20161230", "bad", 10, 20], "settlement"),
        (["注：", "1、 合约单位：铜为5吨/手", 123, "", ""], "trade date"),
        (["", "unrecognized footer", "", "", ""], "trade date"),
    ],
)
def test_metadata_recognition_does_not_hide_malformed_quote_rows(bad_row, match):
    with pytest.raises(ValueError, match=match):
        data.parse_shfe_rows(
            [HEADER, ["cu1702", "20161229", 100, 10, 20], *_counting_footer(), bad_row]
        )


def test_counting_conflicts_fail_and_unknown_conventions_are_not_guessed_from_year():
    with pytest.raises(ValueError, match="conflicting"):
        data.parse_shfe_rows(
            [
                HEADER,
                ["cu1702", "20161229", 100, 10, 20],
                *_counting_footer("双边"),
                *_counting_footer("单边"),
            ]
        )
    with pytest.raises(ValueError, match="conflicting"):
        data.parse_shfe_rows(
            [HEADER, ["cu1702", "20161229", 100, 10, 20], ["3、 成交量、持仓量，单边及双边计算"]]
        )
    with pytest.raises(ValueError, match="conflicting"):
        data.parse_shfe_rows(
            [
                HEADER,
                ["cu1702", "20161229", 100, 10, 20],
                ["3、 成交量、持仓量，单边计算；成交量，双边计算"],
            ]
        )
    unknown = data.parse_shfe_rows(_rows(), source="shfe-futures-2024.zip")
    assert unknown.provenance["tables"][0]["counting"]["volume_convention"] == "unknown"
    with pytest.raises(ValueError, match="unknown declaration"):
        unknown.single_sided_counts()


def test_multi_archive_counts_normalize_per_declared_date_scope(tmp_path):
    old = _archive(
        tmp_path,
        [HEADER, ["cu2002", "20191231", 100, 11, 21], *_counting_footer("双边")],
        name="shfe-futures-2019.zip",
    )
    new = _archive(
        tmp_path,
        [HEADER, ["cu2002", "20200102", 101, 11, 21], *_counting_footer("单边")],
        name="shfe-futures-2020.zip",
    )
    panel = data.load_shfe_cu_archives([old, new])
    np.testing.assert_array_equal(panel.volume, [11, 11])
    np.testing.assert_array_equal(panel.single_sided_counts().volume, [5.5, 11])
    bounded = data.load_shfe_cu_archives([old, new], start_date="2020-01-01")
    np.testing.assert_array_equal(bounded.single_sided_counts().volume, [11])


def test_counting_declarations_conflicting_across_same_date_scope_fail(tmp_path):
    first = _archive(
        tmp_path,
        [HEADER, ["cu2002", "20200102", 100, 10, 20], *_counting_footer("双边")],
        name="first.zip",
    )
    second = _archive(
        tmp_path,
        [HEADER, ["cu2003", "20200102", 100, 10, 20], *_counting_footer("单边")],
        name="second.zip",
    )
    panel = data.load_shfe_cu_archives([first, second])
    with pytest.raises(ValueError, match="conflicting volume"):
        panel.single_sided_counts()


def test_counting_public_metadata_is_aggregate_and_fail_closed():
    panel = data.parse_shfe_rows([HEADER, ["cu1702", "20161229", 100, 10, 20], *_counting_footer()])
    provenance = panel.provenance
    provenance["tables"][0]["counting"]["raw_prices"] = [100, 101]
    public = data.public_data_provenance(provenance)
    assert "raw_prices" not in json.dumps(public)
    assert public["tables"][0]["counting"]["volume_convention"] == "double-sided"
    provenance["tables"][0]["counting"]["volume_convention"] = [100, 101]
    with pytest.raises(ValueError, match="text fields"):
        data.public_data_provenance(provenance)


def test_duplicate_archives_rows_and_manifest_hash_mismatch_fail(tmp_path):
    first = _archive(tmp_path)
    with pytest.raises(ValueError, match="paths must be unique"):
        data.load_shfe_cu_archives([first, first])
    second = _archive(tmp_path, name="another.zip")
    with pytest.raises(ValueError, match="sorted and unique"):
        data.load_shfe_cu_archives([first, second])
    (tmp_path / "acquisition_manifest.json").write_text(
        json.dumps({"archives": [{"path": str(first), "sha256": "0" * 64}]})
    )
    with pytest.raises(ValueError, match="SHA-256 disagrees"):
        data.load_shfe_cu_archives([first])


def test_archive_path_safety_and_size_guard(tmp_path, monkeypatch):
    path = _archive(tmp_path, members={"../not-extracted.xlsx": _xlsx(_rows())})
    with pytest.raises(ValueError, match="unsafe"):
        data.load_shfe_cu_archives([path])
    assert not (tmp_path.parent / "not-extracted.xlsx").exists()
    safe = _archive(tmp_path, name="safe.zip")
    monkeypatch.setattr(data, "_MAX_MEMBER_BYTES", 1)
    with pytest.raises(ValueError, match="oversized"):
        data.load_shfe_cu_archives([safe])


def _daily_sample(tmp_path, *, value=100, internal_day="02", duplicate=False):
    row = {
        "PRODUCTID": "cu_f   ",
        "DELIVERYMONTH": "2402",
        "SETTLEMENTPRICE": value,
        "VOLUME": "10",
        "OPENINTEREST": 20,
    }
    content = {
        "o_year": "2024",
        "o_month": "01",
        "o_day": internal_day,
        "o_curinstrument": [row, {"PRODUCTID": "cu_f", "DELIVERYMONTH": "小计"}],
    }
    if duplicate:
        content["o_curinstrument"].append(row)
    path = tmp_path / "kx20240102.dat"
    path.write_text(json.dumps(content))
    return path


def test_daily_reconciliation_is_exact_aggregate_only_with_source_hash(tmp_path):
    panel = data.parse_shfe_rows([HEADER, ["cu2402", "20240102", 100, 10, 20]])
    path = _daily_sample(tmp_path)
    audit = data.audit_shfe_daily_samples(panel, [path], source_urls={path.name: "official-source"})
    assert audit["all_sampled_values_equal"]
    assert audit["compared_field_cells"] == audit["exact_equal_field_cells"] == 3
    sample = audit["samples"][0]
    assert sample["source_url"] == "official-source"
    assert sample["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sample["fields"]["settlement"]["max_absolute_error"] == 0
    assert "SETTLEMENTPRICE" not in json.dumps(audit)


def test_daily_reconciliation_detects_revisions_and_missing_contracts(tmp_path):
    panel = data.parse_shfe_rows(
        [HEADER, ["cu2402", "20240102", 100, 10, 20], ["cu2403", "20240102", 101, 11, 21]]
    )
    audit = data.audit_shfe_daily_samples(panel, [_daily_sample(tmp_path, value=101)])
    assert not audit["all_sampled_values_equal"]
    assert audit["samples"][0]["missing_from_daily"] == ["cu2403"]
    assert audit["samples"][0]["fields"]["settlement"]["max_absolute_error"] == 1
    assert audit["exact_equal_field_cells"] == 2


def test_daily_reconciliation_rejects_wrong_date_and_duplicate_contracts(tmp_path):
    panel = data.parse_shfe_rows([HEADER, ["cu2402", "20240102", 100, 10, 20]])
    with pytest.raises(ValueError, match="internal date"):
        data.audit_shfe_daily_samples(panel, [_daily_sample(tmp_path, internal_day="03")])
    with pytest.raises(ValueError, match="duplicate copper"):
        data.audit_shfe_daily_samples(panel, [_daily_sample(tmp_path, duplicate=True)])


def test_public_provenance_drops_root_and_nested_raw_values():
    private = {
        "raw_prices": [100, 101],
        "archives": [
            {
                "filename": "shfe-futures-2024.zip",
                "sha256": "a" * 64,
                "bytes": 10,
                "year": 2024,
                "raw_prices": [100, 101],
            }
        ],
        "tables": [
            {
                "source": "archive:member:sheet",
                "member_sha256": "b" * 64,
                "raw_prices": [100, 101],
                "counts": {"copper_rows": 2, "raw_prices": [100, 101]},
            }
        ],
        "units": {"currency": "CNY", "tonnes_per_contract": 5.0, "raw_prices": [100, 101]},
        "date_range": {
            "start_inclusive": "2024-01-01",
            "end_inclusive": None,
            "raw_prices": [100, 101],
        },
    }
    public = data.public_data_provenance(private)
    assert "raw_prices" not in json.dumps(public)
    assert public["archives"][0]["sha256"] == "a" * 64
    assert public["tables"][0]["counts"] == {"copper_rows": 2}
    assert private["raw_prices"] == [100, 101]
    public["units"]["currency"] = "changed"
    assert private["units"]["currency"] == "CNY"


def test_public_reconciliation_drops_nested_raw_values_and_replaces_ids_with_counts():
    private = {
        "sample_count": 1,
        "raw_prices": [100, 101],
        "annual_archive_sha256": {"shfe-futures-2024.zip": "a" * 64, "raw_prices": [100, 101]},
        "samples": [
            {
                "date": "2024-01-02",
                "filename": "kx20240102.dat",
                "sha256": "b" * 64,
                "raw_prices": [100, 101],
                "missing_from_annual": ["cu2402"],
                "missing_from_daily": [],
                "fields": {
                    "raw_prices": [100, 101],
                    "settlement": {
                        "compared_cells": 2,
                        "exact_equal_cells": 2,
                        "max_absolute_error": 0.0,
                        "raw_prices": [100, 101],
                    },
                },
            }
        ],
    }
    public = data.public_data_validation(private)
    assert "raw_prices" not in json.dumps(public) and "cu2402" not in json.dumps(public)
    sample = public["samples"][0]
    assert sample["missing_from_annual_count"] == 1
    assert sample["missing_from_daily_count"] == 0
    assert sample["fields"]["settlement"]["max_absolute_error"] == 0
    assert data.public_data_validation(None) is None


def test_public_metadata_known_scalar_fields_fail_closed_for_raw_arrays():
    with pytest.raises(ValueError, match="counts"):
        data.public_data_provenance({"tables": [{"counts": {"copper_rows": [100, 101]}}]})
    with pytest.raises(ValueError, match="numeric fields"):
        data.public_data_validation(
            {"samples": [{"fields": {"settlement": {"max_absolute_error": [100, 101]}}}]}
        )


def _downloader():
    path = Path(__file__).parents[1] / "scripts/download_shfe.py"
    spec = importlib.util.spec_from_file_location("shfe_download_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downloader_refuses_unignored_or_outside_cache(tmp_path):
    download = _downloader()
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    with pytest.raises(ValueError, match="subdirectory"):
        download._check_cache(tmp_path / "outside", repository)
    with pytest.raises(ValueError, match="not ignored"):
        download._check_cache(repository / "data/local/shfe", repository)
    (repository / ".gitignore").write_text("/data/local/\n")
    assert download._check_cache(repository / "data/local/shfe", repository).exists()


def test_downloader_cached_hash_verification_is_offline_and_detects_corruption(
    tmp_path, monkeypatch
):
    download = _downloader()
    raw = b"synthetic archive bytes"
    pin = {
        "url": "https://example.invalid/synthetic",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    monkeypatch.setitem(download.PINNED_ARCHIVES, 2024, pin)
    (tmp_path / "shfe-futures-2024.zip").write_bytes(raw)
    (tmp_path / "download_catalogue.json").write_text("{}")
    monkeypatch.setattr(download, "_check_cache", lambda cache, repository: cache)
    monkeypatch.setattr(
        download, "_request", lambda *a, **k: pytest.fail("unexpected network call")
    )
    manifest = download.acquire_archives([2024], tmp_path, repository=tmp_path)
    assert not manifest["archives"][0]["downloaded_this_run"]
    (tmp_path / "shfe-futures-2024.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cached archive"):
        download.acquire_archives([2024], tmp_path, repository=tmp_path)


def test_downloader_resumes_only_exact_pinned_range_and_verifies_final_hash(tmp_path, monkeypatch):
    download = _downloader()
    raw = b"a synthetic complete archive"
    pin = {
        "url": "https://example.invalid/synthetic",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    monkeypatch.setitem(download.PINNED_ARCHIVES, 2024, pin)
    part = tmp_path / "shfe-futures-2024.zip.part"
    part.write_bytes(raw[:5])
    (tmp_path / "shfe-futures-2024.zip.part.json").write_text(json.dumps({"year": 2024, **pin}))

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 5-{len(raw) - 1}/{len(raw)}"}

    def request(url, *, start):
        assert url == pin["url"] and start == 5
        return Response(raw[5:])

    monkeypatch.setattr(download, "_request", request)
    result = download._download_archive(2024, tmp_path)
    assert result["sha256"] == pin["sha256"] and result["downloaded_this_run"]
    assert not part.exists()


def test_downloader_rejected_resume_does_not_append_or_promote(tmp_path, monkeypatch):
    download = _downloader()
    raw = b"synthetic"
    pin = {
        "url": "https://example.invalid/synthetic",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    monkeypatch.setitem(download.PINNED_ARCHIVES, 2024, pin)
    part = tmp_path / "shfe-futures-2024.zip.part"
    part.write_bytes(raw[:2])
    (tmp_path / "shfe-futures-2024.zip.part.json").write_text(json.dumps({"year": 2024, **pin}))

    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr(download, "_request", lambda *a, **k: Response(raw))
    with pytest.raises(ValueError, match="resume range"):
        download._download_archive(2024, tmp_path)
    assert part.read_bytes() == raw[:2]
    assert not (tmp_path / "shfe-futures-2024.zip").exists()
