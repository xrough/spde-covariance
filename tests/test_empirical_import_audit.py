"""Aggregate audit and loader-boundary tests using synthetic data only."""

import hashlib
import json

import numpy as np
import pytest

import spde_covariance.empirical_import_audit as audit
from spde_covariance.empirical_data import ShfeCuPanel, parse_shfe_rows


def _part(day, convention=None, *, volume=11):
    year = str(day)[:4]
    rows = [
        ["合约", "日期", "结算价", "成交量", "持仓量"],
        [f"cu{int(year) % 100:02d}12", day, 100, volume, 21],
    ]
    if convention is not None:
        rows.append([f"3、 成交量、持仓量单位为手，{convention}计算"])
    return parse_shfe_rows(rows, source=f"shfe-futures-{year}.zip:synthetic.xlsx:sheet")


def _combine(*parts):
    fields = (
        "dates",
        "contract_ids",
        "contract_month_dates",
        "settlement",
        "volume",
        "open_interest",
        "session_dates",
    )
    return ShfeCuPanel(
        *(np.concatenate([getattr(part, key) for part in parts]) for key in fields),
        provenance={"tables": [t for part in parts for t in part.provenance["tables"]]},
    )


def test_audit_counts_declared_normalization_unknown_and_exact_legacy_parity():
    recent = _part("2024-01-02", "单边")
    panel = _combine(_part("2019-01-02", "双边"), _part("2020-01-02", "单边"), recent)
    report = audit.audit_copper_import(panel, legacy_panel=recent)
    assert (report["rows"], report["sessions"], report["distinct_contracts"]) == (3, 3, 3)
    assert all(report["price_and_raw_count_parity_2022_2025"].values())
    assert report["by_year"][0]["volume_counting_conventions"] == ["double-sided"]
    assert report["counting_policy"]["normalized_known_scope_rows"] == 3
    assert report["counting_policy"]["normalized_volume_half_integer_count"] == 1
    assert report["counting_policy"]["normalized_open_interest_half_integer_count"] == 1
    assert panel.volume.tolist() == [11, 11, 11]


def test_unknown_notes_remain_unknown_and_no_legacy_is_explicit():
    panel = _combine(_part("2024-01-02", "单边"), _part("2025-01-02"))
    result = audit.audit_copper_import(panel)
    assert result["legacy_parity_available"] is False
    assert result["price_and_raw_count_parity_2022_2025"] is None
    assert result["by_year"][1]["normalization_unavailable_rows"] == 1
    assert result["by_year"][1]["volume_counting_conventions"] == ["unknown"]
    assert result["counting_policy"]["normalized_known_scope_rows"] == 1
    assert result["held_out_2026"] is None


def test_audit_rejects_held_out_prices_or_sessions():
    with pytest.raises(ValueError, match="held out"):
        audit.audit_copper_import(_part("2026-01-02", "单边"))
    known = _part("2024-01-02", "单边")
    panel = ShfeCuPanel(
        known.dates,
        known.contract_ids,
        known.contract_month_dates,
        known.settlement,
        known.volume,
        known.open_interest,
        np.array(["2024-01-02", "2026-01-02"], dtype="datetime64[D]"),
        provenance=known.provenance,
    )
    with pytest.raises(ValueError, match="held out"):
        audit.audit_copper_import(panel)


def test_public_audit_recursively_drops_private_metadata_and_canonicalizes_text():
    payload = {
        "rows": 3,
        "raw_prices": [100, 101],
        "data_provenance": {"path": "/private/panel"},
        "price_and_raw_count_parity_2022_2025": {"volume": True, "raw_prices": [100, 101]},
        "by_year": [
            {
                "year": 2024,
                "rows": 2,
                "raw_prices": [100, 101],
                "volume_counting_conventions": ["single-sided"],
            }
        ],
        "counting_policy": {"raw": "/private/panel with values 100,101", "raw_prices": [100, 101]},
        "held_out_2026": {
            "year": 2026,
            "bytes": 100,
            "sha256": "a" * 64,
            "source": "/private/reserve",
            "raw_prices": [100, 101],
            "url": "https://www.shfe.com.cn/reports/tradedata/datadownload/annualdata/reserve.zip",
        },
    }
    clean = audit.public_import_audit(payload)
    text = json.dumps(clean)
    assert "raw_prices" not in text and "/private" not in text and "data_provenance" not in clean
    assert clean["price_and_raw_count_parity_2022_2025"] == {"volume": True}
    assert clean["held_out_2026"]["sha256"] == "a" * 64
    assert payload["counting_policy"]["raw"].startswith("/private")


@pytest.mark.parametrize(
    "payload",
    [
        {"rows": [100, 101]},
        {"by_year": [{"rows": [100, 101]}]},
        {"by_year": [{"volume_counting_conventions": ["raw price 100"]}]},
        {"price_and_raw_count_parity_2022_2025": {"volume": [100, 101]}},
        {"counting_policy": {"raw": [100, 101]}},
        {"held_out_2026": {"sha256": [100, 101]}},
        {"held_out_2026": {"source": [100, 101]}},
        {"held_out_2026": {"url": "/private/reserve"}},
        {"held_out_2026": {"year": 2025}},
        {"by_year": [{"year": 2026}]},
    ],
)
def test_public_known_scalar_fields_fail_closed(payload):
    with pytest.raises(ValueError):
        audit.public_import_audit(payload)


def _loader_fixture(tmp_path, monkeypatch):
    older, recent = tmp_path / "extension" / "development", tmp_path / "recent"
    older.mkdir(parents=True)
    recent.mkdir()
    archives, parts = [], []
    for year in range(2016, 2026):
        path = (older if year < 2022 else recent) / f"shfe-futures-{year}.zip"
        content = f"synthetic {year}".encode()
        path.write_bytes(content)
        archives.append(
            {
                "filename": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
        parts.append(_part(f"{year}-01-04", "单边"))
    panel = _combine(*parts)
    provenance = panel.provenance | {"archives": archives}
    panel = ShfeCuPanel(
        *(getattr(panel, key) for key in (*audit._PARITY_FIELDS, "session_dates")),
        provenance=provenance,
    )
    calls = []

    def loader(paths, **kwargs):
        calls.append((paths, kwargs))
        return panel

    monkeypatch.setattr(audit, "load_shfe_cu_archives", loader)
    return older, recent, panel, calls


def test_loader_uses_exact_development_allowlist_and_enriches_hashmatched_metadata(
    tmp_path, monkeypatch
):
    older, recent, panel, calls = _loader_fixture(tmp_path, monkeypatch)
    source = panel.provenance["archives"][0]
    entry = {
        "path": str(older / source["filename"]),
        "sha256": source["sha256"],
        "bytes": source["bytes"],
        "year": 2016,
        "url": "https://www.shfe.com.cn/official.zip",
        "retrieved_utc": "2026-10-27",
    }
    (older.parent / "extension_acquisition_manifest.json").write_text(
        json.dumps(
            {"archives": [entry, {"path": "/reserve/shfe-futures-2026.zip", "raw_prices": [100]}]}
        )
    )
    loaded = audit.load_development_copper(recent, older)
    assert loaded.provenance["archives"][0]["url"] == entry["url"]
    assert len(calls) == 1
    paths, kwargs = calls[0]
    assert [p.name for p in paths] == [f"shfe-futures-{year}.zip" for year in range(2016, 2026)]
    assert kwargs == {"start_date": "2016-01-01", "end_date": "2025-12-31"}
    assert "2026" not in ",".join(p.name for p in paths)


def test_loader_rejects_missing_archive_before_loading(tmp_path, monkeypatch):
    older, recent, _, calls = _loader_fixture(tmp_path, monkeypatch)
    (older / "shfe-futures-2016.zip").unlink()
    with pytest.raises(FileNotFoundError, match="2016"):
        audit.load_development_copper(recent, older)
    assert not calls


def test_loader_rejects_bad_manifest_and_date_year_mismatch(tmp_path, monkeypatch):
    older, recent, panel, _ = _loader_fixture(tmp_path, monkeypatch)
    source = panel.provenance["archives"][0]
    manifest = older.parent / "extension_acquisition_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archives": [
                    {
                        "path": str(older / source["filename"]),
                        "sha256": "0" * 64,
                        "bytes": source["bytes"],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="manifest"):
        audit.load_development_copper(recent, older)
    manifest.unlink()
    provenance = panel.provenance
    provenance["tables"][0]["counting"]["first_copper_session"] = "2017-01-01"
    bad = ShfeCuPanel(
        *(getattr(panel, key) for key in (*audit._PARITY_FIELDS, "session_dates")),
        provenance=provenance,
    )
    monkeypatch.setattr(audit, "load_shfe_cu_archives", lambda *a, **kw: bad)
    with pytest.raises(ValueError, match="filename year"):
        audit.load_development_copper(recent, older)
