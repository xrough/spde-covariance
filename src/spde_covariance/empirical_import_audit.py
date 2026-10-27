"""Public aggregate audits for the fixed 2016--2025 copper development panel.

No function in this module searches for, opens, or parses the reserved 2026
archive. Its optional provenance is supplied as metadata by the caller.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .empirical_data import ShfeCuPanel, load_shfe_cu_archives

_YEARS = tuple(range(2016, 2026))
_COUNTING = {"single-sided", "double-sided", "unknown"}
_PARITY_FIELDS = (
    "dates",
    "contract_ids",
    "contract_month_dates",
    "settlement",
    "volume",
    "open_interest",
)
_POLICY = {
    "raw": "archived integer volume/open_interest preserved exactly",
    "view": "explicit opt-in single-sided equivalents from per-workbook notes only; no year fallback",
    "known_double_sided": "only where workbook notes explicitly declare double-sided counts",
    "known_single_sided": "only where workbook notes explicitly declare single-sided counts",
    "unavailable": "missing declarations remain unknown; no year inference",
    "prices_and_multiplier": "unchanged; CNY/tonne and 5 tonnes/contract",
    "positive_liquidity_eligibility": "unchanged under positive rescaling",
}
_YEAR_COUNTS = {
    "year",
    "rows",
    "sessions",
    "normalization_unavailable_rows",
    "normalization_unavailable_sessions",
    "settlement_nonfinite",
    "settlement_nonpositive",
    "volume_zero",
    "open_interest_zero",
}
_POLICY_COUNTS = {
    "normalized_known_scope_rows",
    "normalized_volume_half_integer_count",
    "normalized_open_interest_half_integer_count",
}


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("import audit counts must be nonnegative integers")
    return value


def _flag(value):
    if not isinstance(value, bool):
        raise ValueError("import audit parity values must be boolean")
    return value


def public_import_audit(importer_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist aggregate audit fields; do not copy arbitrary caller metadata.

    Policy/source descriptions are canonical constants, rather than user text.
    Known scalar fields fail closed on arrays. Unexpected keys are dropped at
    every level, including nested provenance, raw observations and local paths.
    """
    original = _mapping(importer_audit, "importer audit")
    result = {"scope": "2016--2025 copper only; reserved 2026 archive not opened by import audit"}
    for key in ("rows", "sessions", "distinct_contracts"):
        if key in original:
            result[key] = _count(original[key])
    if "legacy_parity_available" in original:
        result["legacy_parity_available"] = _flag(original["legacy_parity_available"])
    if "price_and_raw_count_parity_2022_2025" in original:
        values = original["price_and_raw_count_parity_2022_2025"]
        result["price_and_raw_count_parity_2022_2025"] = (
            None
            if values is None
            else {
                key: _flag(value)
                for key, value in _mapping(values, "legacy parity").items()
                if key in (*_PARITY_FIELDS, "session_dates")
            }
        )
    if "by_year" in original:
        rows = original["by_year"]
        if not isinstance(rows, (list, tuple)):
            raise ValueError("import audit by_year must be a list")
        clean = []
        for supplied in rows:
            supplied = _mapping(supplied, "annual import audit")
            row = {key: _count(value) for key, value in supplied.items() if key in _YEAR_COUNTS}
            if "year" in row and row["year"] not in _YEARS:
                raise ValueError("import audit years must be within 2016--2025")
            for key in ("volume_counting_conventions", "open_interest_counting_conventions"):
                if key in supplied:
                    conventions = supplied[key]
                    if not isinstance(conventions, (list, tuple)) or any(
                        not isinstance(value, str) or value not in _COUNTING
                        for value in conventions
                    ):
                        raise ValueError("invalid import audit counting conventions")
                    row[key] = sorted(set(conventions))
            clean.append(row)
        result["by_year"] = clean
    if "counting_policy" in original:
        policy = _mapping(original["counting_policy"], "counting policy")
        clean = {}
        for key, canonical in _POLICY.items():
            if key in policy:
                if not isinstance(policy[key], str):
                    raise ValueError("import audit policy descriptions must be strings")
                clean[key] = canonical
        for key in _POLICY_COUNTS:
            if key in policy:
                clean[key] = _count(policy[key])
        if "double_sided_divisor" in policy:
            if _count(policy["double_sided_divisor"]) != 2:
                raise ValueError("double-sided count divisor must equal two")
            clean["double_sided_divisor"] = 2
        result["counting_policy"] = clean
    if "held_out_2026" in original:
        reserve = original["held_out_2026"]
        if reserve is None:
            result["held_out_2026"] = None
        else:
            reserve = _mapping(reserve, "held-out metadata")
            clean = {}
            if "year" in reserve:
                if _count(reserve["year"]) != 2026:
                    raise ValueError("held-out metadata year must equal 2026")
                clean["year"] = 2026
            if "bytes" in reserve:
                clean["bytes"] = _count(reserve["bytes"])
            if "sha256" in reserve:
                digest = reserve["sha256"]
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError("held-out metadata requires a lowercase SHA-256 digest")
                clean["sha256"] = digest
            if "url" in reserve:
                url = reserve["url"]
                if (
                    not isinstance(url, str)
                    or re.fullmatch(
                        r"https://www\.shfe\.com\.cn/reports/tradedata/datadownload/"
                        r"[A-Za-z0-9_./-]+\.zip",
                        url,
                    )
                    is None
                    or ".." in url
                ):
                    raise ValueError("held-out metadata URL must identify an official SHFE ZIP")
                clean["url"] = url
            if "source" in reserve and not isinstance(reserve["source"], str):
                raise ValueError("held-out metadata source must be text")
            clean["source"] = (
                "provided acquisition metadata only; reserved archive not opened by audit"
            )
            result["held_out_2026"] = clean
    return result


def _development_bounds(panel: ShfeCuPanel) -> None:
    if not isinstance(panel, ShfeCuPanel):
        raise TypeError("supply a ShfeCuPanel")
    lower, upper = np.datetime64("2016-01-01"), np.datetime64("2025-12-31")
    if any(
        np.any((values < lower) | (values > upper)) for values in (panel.dates, panel.session_dates)
    ):
        raise ValueError(
            "development copper must be restricted to 2016--2025; 2026 remains held out"
        )


def audit_copper_import(
    panel: ShfeCuPanel,
    *,
    legacy_panel: ShfeCuPanel | None = None,
    held_out_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit coverage, missingness, declared count units and optional raw parity.

    This deterministic aggregate contains no settlements, individual contracts,
    normalized volume values, arbitrary provenance or filesystem paths.
    """
    _development_bounds(panel)
    n = panel.dates.size
    factors = {name: np.full(n, np.nan) for name in ("volume", "open_interest")}
    unknown = np.zeros(n, dtype=bool)
    tables = panel.provenance.get("tables", [])
    if not isinstance(tables, (list, tuple)):
        raise ValueError("import audit counting tables must be a list")
    for table in tables:
        table = _mapping(table, "counting table")
        declaration = _mapping(table.get("counting", {}), "counting declaration")
        first, last = (
            declaration.get("first_copper_session"),
            declaration.get("last_copper_session"),
        )
        if first is None:
            continue
        try:
            first, last = np.datetime64(first, "D"), np.datetime64(last, "D")
        except (TypeError, ValueError) as error:
            raise ValueError("counting declaration has invalid session bounds") from error
        if np.isnat(first) or np.isnat(last) or first > last:
            raise ValueError("counting declaration has invalid session bounds")
        selected = (panel.dates >= first) & (panel.dates <= last)
        for name, values in factors.items():
            convention = declaration.get(name + "_convention", "unknown")
            if not isinstance(convention, str) or convention not in _COUNTING:
                raise ValueError("unrecognized counting declaration")
            if convention == "unknown":
                unknown |= selected
                continue
            factor = 1.0 if convention == "single-sided" else 0.5
            if np.any(selected & np.isfinite(values) & (values != factor)):
                raise ValueError("conflicting counting declarations")
            values[selected] = factor
    known = ~unknown & np.isfinite(factors["volume"]) & np.isfinite(factors["open_interest"])
    half_counts = {name: 0 for name in factors}
    if np.any(known):
        fields = (
            "dates",
            "contract_ids",
            "contract_month_dates",
            "settlement",
            "volume",
            "open_interest",
        )
        subset = ShfeCuPanel(
            *(getattr(panel, key)[known] for key in fields),
            panel.session_dates,
            provenance=panel.provenance,
        )
        normalized = subset.single_sided_counts()
        half_counts = {
            name: int(
                np.sum(
                    np.isfinite(getattr(normalized, name)) & (getattr(normalized, name) % 1 != 0)
                )
            )
            for name in factors
        }
    annual = []
    for year in sorted(set(panel.dates.astype("datetime64[Y]").astype(int) + 1970)):
        selected = panel.dates.astype("datetime64[Y]").astype(int) + 1970 == year
        row = {
            "year": int(year),
            "rows": int(selected.sum()),
            "sessions": int(np.unique(panel.dates[selected]).size),
            "normalization_unavailable_rows": int(np.sum(selected & ~known)),
            "normalization_unavailable_sessions": int(
                np.unique(panel.dates[selected & ~known]).size
            ),
            "settlement_nonfinite": int(np.sum(~np.isfinite(panel.settlement[selected]))),
            "settlement_nonpositive": int(np.sum(panel.settlement[selected] <= 0)),
            "volume_zero": int(np.sum(panel.volume[selected] == 0)),
            "open_interest_zero": int(np.sum(panel.open_interest[selected] == 0)),
        }
        for name, values in factors.items():
            conventions = set()
            if np.any(selected & known & (values == 0.5)):
                conventions.add("double-sided")
            if np.any(selected & known & (values == 1)):
                conventions.add("single-sided")
            if np.any(selected & ~known):
                conventions.add("unknown")
            row[name + "_counting_conventions"] = sorted(conventions)
        annual.append(row)
    parity = None
    if legacy_panel is not None:
        _development_bounds(legacy_panel)
        selected = panel.dates >= np.datetime64("2022-01-01")
        parity = {
            key: bool(
                np.array_equal(
                    getattr(panel, key)[selected],
                    getattr(legacy_panel, key),
                    equal_nan=key in ("settlement", "volume", "open_interest"),
                )
            )
            for key in _PARITY_FIELDS
        }
    return public_import_audit(
        {
            "rows": int(n),
            "sessions": int(panel.session_dates.size),
            "distinct_contracts": int(np.unique(panel.contract_ids).size),
            "legacy_parity_available": legacy_panel is not None,
            "price_and_raw_count_parity_2022_2025": parity,
            "by_year": annual,
            "counting_policy": _POLICY
            | {
                "double_sided_divisor": 2,
                "normalized_known_scope_rows": int(known.sum()),
                "normalized_volume_half_integer_count": half_counts["volume"],
                "normalized_open_interest_half_integer_count": half_counts["open_interest"],
            },
            "held_out_2026": held_out_metadata,
        }
    )


def load_development_copper(recent_directory, older_directory) -> ShfeCuPanel:
    """Load exactly 2016--2025 archives; never glob or inspect the 2026 reserve.

    Extension acquisition metadata are added only when its archive name, byte
    size and SHA-256 exactly match the bytes already verified by the loader.
    Missing inputs or a workbook's copper dates outside its named year fail.
    """
    recent, older = Path(recent_directory), Path(older_directory)
    paths = [(older if year < 2022 else recent) / f"shfe-futures-{year}.zip" for year in _YEARS]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing development archives: " + ", ".join(missing))
    panel = load_shfe_cu_archives(paths, start_date="2016-01-01", end_date="2025-12-31")
    provenance = panel.provenance
    for table in provenance.get("tables", []):
        match = re.match(r"shfe-futures-(\d{4})\.zip:", table.get("source", ""))
        if match is None:
            raise ValueError("development workbook lacks its expected archive year")
        expected = match[1]
        declaration = table.get("counting", {})
        for key in ("first_copper_session", "last_copper_session"):
            if declaration.get(key) is not None and declaration[key][:4] != expected:
                raise ValueError("development archive copper dates disagree with its filename year")
    manifest_path = older.parent / "extension_acquisition_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        entries = {}
        for entry in manifest.get("archives", []):
            name = Path(entry.get("path", "")).name
            if name not in {path.name for path in paths}:
                continue
            if name in entries:
                raise ValueError("duplicate development acquisition metadata")
            entries[name] = entry
        for archive in provenance.get("archives", []):
            entry = entries.get(archive["filename"])
            if entry is None:
                continue
            if entry.get("sha256") != archive["sha256"] or entry.get("bytes") != archive["bytes"]:
                raise ValueError(
                    "development archive disagrees with extension acquisition manifest"
                )
            expected_year = int(re.fullmatch(r"shfe-futures-(\d{4})\.zip", archive["filename"])[1])
            if "year" in entry and entry["year"] != expected_year:
                raise ValueError("development acquisition year disagrees with archive filename")
            for key in ("year", "url", "retrieved_utc"):
                if key in entry:
                    archive[key] = entry[key]
    _development_bounds(panel)
    return ShfeCuPanel(
        *(getattr(panel, key) for key in (*_PARITY_FIELDS, "session_dates")), provenance=provenance
    )
