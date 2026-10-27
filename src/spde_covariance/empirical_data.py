"""Audited, unfilled SHFE copper settlement panels from official annual archives.

The archive is a retrospective exchange publication, not a point-in-time feed.
Contract-month midpoints are coordinates only: they are neither inferred expiry
dates nor delivery-average observation maps. Excel support is an optional extra.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

SHFE_DOWNLOAD_PAGE = "https://www.shfe.com.cn/reports/tradedata/datadownload/"
SHFE_CU_MANUAL = "https://tsite.shfe.com.cn/upload/20190919/1568892145195.pdf"

_ALIASES = {
    "contract": {"合约", "合约代码", "contract", "contractcode"},
    "date": {"日期", "交易日期", "date", "tradedate"},
    "settlement": {"结算价", "结算价格", "settle", "settlement", "settlementprice"},
    "volume": {"成交量", "volume"},
    "open_interest": {"持仓量", "oi", "openinterest"},
}
_MISSING = {"", "-", "--", "—", "NA", "N/A", "NAN", "NULL", "NONE"}
_MAX_MEMBER_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
_COUNTING_FACTORS = {"single-sided": 1.0, "double-sided": 0.5}


def _header_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _header(row: Sequence[Any]) -> dict[str, int] | None:
    names = [_header_text(value) for value in row]
    if not any(name in _ALIASES["contract"] for name in names) or not any(
        name in _ALIASES["date"] for name in names
    ):
        return None
    result = {}
    for field, aliases in _ALIASES.items():
        indices = [index for index, name in enumerate(names) if name in aliases]
        if len(indices) != 1:
            raise ValueError(f"SHFE header requires exactly one {field} column")
        result[field] = indices[0]
    return result


def _day(value: Any, label: str) -> np.datetime64:
    if isinstance(value, np.datetime64):
        if np.isnat(value) or value != value.astype("datetime64[D]"):
            raise ValueError(f"{label} must be a calendar date")
        return value.astype("datetime64[D]")
    if isinstance(value, datetime):
        if value.time() != datetime.min.time() or value.tzinfo is not None:
            raise ValueError(f"{label} must be a calendar date without a timezone")
        value = value.date()
    if isinstance(value, date):
        return np.datetime64(value, "D")
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        if not np.isfinite(value) or value != int(value):
            raise ValueError(f"{label} must be YYYYMMDD or YYYY-MM-DD")
        value = str(int(value))
    if not isinstance(value, str):
        raise ValueError(f"{label} must be YYYYMMDD or YYYY-MM-DD")
    text = value.strip()
    if re.fullmatch(r"\d{8}", text):
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"{label} must be YYYYMMDD or YYYY-MM-DD")
    try:
        return np.datetime64(date.fromisoformat(text), "D")
    except ValueError as error:
        raise ValueError(f"{label} is not a valid calendar date") from error


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a real number or missing")
    if isinstance(value, str):
        text = value.strip()
        if text.upper() in _MISSING:
            return float("nan")
        value = text.replace(",", "")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a real number or missing") from error
    if np.isinf(number):
        raise ValueError(f"{label} must be finite or missing")
    return number


def _readonly(value: Any, *, dtype: Any) -> NDArray:
    copied = np.array(value, dtype=dtype, copy=True)
    return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)


def _month_midpoint(contract: str) -> np.datetime64:
    if not re.fullmatch(r"cu\d{4}", contract):
        raise ValueError("copper contract IDs must have the form cuYYMM")
    # This parser deliberately covers the 2000--2099 convention used by the
    # historical study. It does not guess a century from future observations.
    year, month = 2000 + int(contract[2:4]), int(contract[4:6])
    try:
        return np.datetime64(date(year, month, 15), "D")
    except ValueError as error:
        raise ValueError(f"invalid delivery month in copper contract {contract}") from error


@dataclass(frozen=True, slots=True, init=False)
class ShfeCuPanel:
    """Immutable long panel; no rolling, filling, interpolation or price repair.

    ``session_dates`` includes every date observed in the selected SHFE files,
    before selecting copper. It is an observed session calendar, not a claim
    that an omitted exchange session was a holiday. ``provenance`` returns a
    fresh JSON-compatible dictionary; mutating it cannot alter this panel.
    """

    dates: NDArray[np.datetime64]
    contract_ids: NDArray[np.str_]
    contract_month_dates: NDArray[np.datetime64]
    settlement: NDArray[np.float64]
    volume: NDArray[np.float64]
    open_interest: NDArray[np.float64]
    session_dates: NDArray[np.datetime64]
    _provenance_json: str

    currency: ClassVar[str] = "CNY"
    price_unit: ClassVar[str] = "CNY/tonne"
    quantity_unit: ClassVar[str] = "tonne"
    multiplier: ClassVar[float] = 5.0
    tenor_convention: ClassVar[str] = "15th of contract month; coordinate only, not expiry"

    def __init__(
        self,
        dates,
        contract_ids,
        contract_month_dates,
        settlement,
        volume,
        open_interest,
        session_dates,
        provenance: Mapping[str, Any] | None = None,
    ):
        d = _readonly([_day(value, "dates") for value in dates], dtype="datetime64[D]")
        ids = np.asarray(contract_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "US":
            raise ValueError("contract_ids must be a vector of cuYYMM strings")
        ids = _readonly(ids, dtype="U6")
        if any(len(str(value)) != 6 for value in np.asarray(contract_ids)):
            raise ValueError("contract_ids must have exactly six characters")
        m = _readonly(
            [_day(value, "contract_month_dates") for value in contract_month_dates],
            dtype="datetime64[D]",
        )
        n = d.size
        if d.ndim != 1 or n == 0 or ids.shape != (n,) or m.shape != (n,):
            raise ValueError("panel columns must be nonempty vectors of the same length")
        if any(_month_midpoint(str(value)) != midpoint for value, midpoint in zip(ids, m)):
            raise ValueError("contract_month_dates must match the cuYYMM midpoint convention")
        keys = list(zip(d.astype(str), ids.tolist(), strict=True))
        if keys != sorted(keys) or len(set(keys)) != n:
            raise ValueError("date/contract rows must be sorted and unique")
        sessions = _readonly(
            [_day(value, "session_dates") for value in session_dates], dtype="datetime64[D]"
        )
        if (
            sessions.ndim != 1
            or sessions.size == 0
            or np.any(sessions[1:] <= sessions[:-1])
            or not np.all(np.isin(d, sessions))
        ):
            raise ValueError("session_dates must be sorted, unique, and contain every row date")
        for name, value in (("dates", d), ("contract_ids", ids), ("contract_month_dates", m)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "session_dates", sessions)
        for name, value in (
            ("settlement", settlement),
            ("volume", volume),
            ("open_interest", open_interest),
        ):
            raw = np.asarray(value)
            if raw.shape != (n,) or raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
                raise ValueError(f"{name} must be a real vector with one value per row")
            array = _readonly(raw, dtype=float)
            if np.any(np.isinf(array)) or (name != "settlement" and np.any(array < 0)):
                raise ValueError(f"{name} must be finite or missing and counts nonnegative")
            if name != "settlement" and np.any(array[np.isfinite(array)] % 1 != 0):
                raise ValueError(f"{name} must contain integer contract counts")
            object.__setattr__(self, name, array)
        if provenance is not None and not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a JSON-compatible mapping")
        try:
            snapshot = json.dumps(dict(provenance or {}), allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must be JSON-compatible and finite") from error
        object.__setattr__(self, "_provenance_json", snapshot)

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._provenance_json)

    def single_sided_counts(self) -> ShfeSingleSidedCounts:
        """Return an explicit normalized view; leave archived integer counts unchanged.

        Normalization requires an unambiguous declaration in every table covering
        a row date. No convention is guessed from the filename or calendar year.
        Half-contract values are valid here when an archived odd double-sided
        count is halved; the raw panel still requires integer archived counts.
        """
        factors = {name: np.full(self.dates.size, np.nan) for name in ("volume", "open_interest")}
        unknown = {name: np.zeros(self.dates.size, dtype=bool) for name in factors}
        declarations = []
        tables = self.provenance.get("tables", [])
        if not isinstance(tables, (list, tuple)):
            raise ValueError("counting tables must be a sequence of mappings")
        for table in tables:
            if not isinstance(table, Mapping):
                raise ValueError("counting tables must contain mappings")
            counting = table.get("counting", {})
            if not isinstance(counting, Mapping):
                raise ValueError("counting declarations must be mappings")
            if counting.get("first_copper_session") is None:
                continue
            if counting.get("last_copper_session") is None:
                raise ValueError("counting declaration requires both session bounds")
            first = _day(counting["first_copper_session"], "counting first session")
            last = _day(counting["last_copper_session"], "counting last session")
            if first > last:
                raise ValueError("counting declaration has an invalid date scope")
            selected = (self.dates >= first) & (self.dates <= last)
            declarations.append(
                {
                    key: counting[key]
                    for key in (
                        "first_copper_session",
                        "last_copper_session",
                        "volume_convention",
                        "open_interest_convention",
                        "declaration_source",
                    )
                    if key in counting
                }
            )
            for name, values in factors.items():
                convention = counting.get(name + "_convention", "unknown")
                if not isinstance(convention, str):
                    raise ValueError("counting conventions must be strings")
                factor = _COUNTING_FACTORS.get(convention)
                if factor is None:
                    unknown[name] |= selected
                    continue
                conflict = selected & np.isfinite(values) & (values != factor)
                if np.any(conflict):
                    raise ValueError(f"conflicting {name} counting declarations on the same dates")
                values[selected] = factor
        for name, values in factors.items():
            if np.any(~np.isfinite(values) | unknown[name]):
                raise ValueError(
                    f"single-sided {name} normalization unavailable: unknown declaration"
                )
        return ShfeSingleSidedCounts(
            self.volume * factors["volume"],
            self.open_interest * factors["open_interest"],
            {
                "raw_scope": "archived integer contract counts; retained without modification",
                "normalized_scope": "single-sided equivalent contract counts; floating point",
                "source": "explicit per-workbook counting notes; no year fallback",
                "price_and_multiplier_transformation": "none",
                "declarations": declarations,
            },
        )


@dataclass(frozen=True, slots=True, init=False)
class ShfeSingleSidedCounts:
    """Read-only single-sided equivalent counts, aligned with a panel's rows."""

    volume: NDArray[np.float64]
    open_interest: NDArray[np.float64]
    _provenance_json: str

    def __init__(self, volume, open_interest, provenance):
        object.__setattr__(self, "volume", _readonly(volume, dtype=float))
        object.__setattr__(self, "open_interest", _readonly(open_interest, dtype=float))
        object.__setattr__(
            self, "_provenance_json", json.dumps(provenance, allow_nan=False, sort_keys=True)
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._provenance_json)


def _metadata_note(row: Sequence[Any], columns: Mapping[str, int]) -> str | None:
    """Recognize documented unit/counting footers without terminating parsing.

    Restrict recognition to text-only rows in the contract/date cells. A dated
    row or any copper contract label must pass through ordinary data validation.
    """
    contract = row[columns["contract"]] if len(row) > columns["contract"] else ""
    raw_day = row[columns["date"]] if len(row) > columns["date"] else ""
    # Almost all archive rows are dated quotes (including millions of option
    # rows). Leave even a syntactically dated but invalid date to the ordinary
    # validator, instead of scanning every numeric cell for a possible footer.
    if re.match(r"cu\d", str(contract).strip().lower()) or re.fullmatch(
        r"\d{8}|\d{4}-\d{2}-\d{2}", str(raw_day).strip()
    ):
        return None
    nonempty = [(i, str(value).strip()) for i, value in enumerate(row) if str(value).strip()]
    if any(i not in {columns["contract"], columns["date"]} for i, _ in nonempty):
        return None
    if not nonempty or any(re.match(r"cu\d", value.lower()) for _, value in nonempty):
        return None
    if str(raw_day).strip():
        try:
            _day(raw_day, "trade date")
        except ValueError:
            pass
        else:
            return None
    text = " ".join(value for _, value in nonempty)
    body = re.sub(r"^注\s*[:：]\s*", "", text)
    body = re.sub(r"^\d+\s*[、.．]\s*", "", body)
    if not re.match(r"(?:合约单位\s*[:：]|报价单位\s*[:：]|成交量[、，,]|涨跌1\s*=)", body):
        return None
    return body


def _counting_note(text: str, *, source: str, row_number: int) -> dict[str, str]:
    """Read only volume/OI declarations, excluding the separate turnover units."""
    result = {}
    for clause in re.split(r"[；;。]", text):
        fields = [
            name
            for name, chinese in (("volume", "成交量"), ("open_interest", "持仓量"))
            if chinese in clause
        ]
        if not fields:
            continue
        conventions = {
            name
            for word, name in (("单边", "single-sided"), ("双边", "double-sided"))
            if word in clause
        }
        if len(conventions) > 1:
            raise ValueError(f"{source}, row {row_number}: conflicting counting declaration")
        if conventions:
            convention = conventions.pop()
            for field in fields:
                if field in result and result[field] != convention:
                    raise ValueError(
                        f"{source}, row {row_number}: conflicting counting declaration"
                    )
                result[field] = convention
    return result


def _row_records(
    rows: Iterable[Sequence[Any]],
    *,
    source: str,
    merged_cell_ranges: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (),
) -> tuple[list[tuple], set[np.datetime64], dict[str, Any]]:
    columns = None
    records, sessions = [], set()
    counts: Counter[str] = Counter()
    conventions: dict[str, str] = {}
    declaration_count = 0
    vertical: dict[tuple[int, int], int] = {}
    last_end: dict[int, int] = {}
    for (start_row, start_col), (end_row, end_col) in sorted(merged_cell_ranges):
        if min(start_row, start_col) < 0 or end_row < start_row or end_col < start_col:
            raise ValueError(f"{source}: invalid Excel merged-cell range")
        if start_col != end_col or start_row == end_row:
            continue
        if start_row <= last_end.get(start_col, -1):
            raise ValueError(f"{source}: overlapping Excel merged-cell ranges")
        vertical[(start_row, start_col)] = end_row
        last_end[start_col] = end_row
    active_merge: tuple[int, str] | None = None
    for number, row in enumerate(rows, 1):
        if not any(str(value).strip() for value in row):
            counts["empty_rows"] += 1
            continue
        try:
            header = _header(row)
        except ValueError as error:
            raise ValueError(f"{source}, row {number}: {error}") from error
        if header is not None:
            columns = header
            active_merge = None
            counts["header_rows"] += 1
            continue
        if columns is None:
            counts["preamble_rows"] += 1
            continue
        note = _metadata_note(row, columns)
        if note is not None:
            counts["footer_rows"] += 1
            counts["recognized_metadata_rows"] += 1
            declaration = _counting_note(note, source=source, row_number=number)
            if declaration:
                declaration_count += 1
            for field, convention in declaration.items():
                if field in conventions and conventions[field] != convention:
                    raise ValueError(f"{source}, row {number}: conflicting {field} counting notes")
                conventions[field] = convention
            active_merge = None
            continue
        if len(row) <= max(columns.values()):
            raise ValueError(f"{source}, row {number}: truncated SHFE data row")
        contract = str(row[columns["contract"]]).strip().lower()
        end = vertical.get((number - 1, columns["contract"]))
        if end is not None:
            if not contract:
                raise ValueError(f"{source}, row {number}: merged contract label has no anchor")
            active_merge = (end, contract)
        if active_merge is not None:
            if number - 1 > active_merge[0]:
                active_merge = None
            elif not contract:
                contract = active_merge[1]
                counts["expanded_merged_contract_labels"] += 1
            elif contract != active_merge[1]:
                raise ValueError(f"{source}, row {number}: inconsistent merged contract label")
        raw_day = row[columns["date"]]
        if not str(raw_day).strip():
            if contract.startswith("cu") and re.fullmatch(r"cu\d+", contract):
                raise ValueError(f"{source}, row {number}: copper row has no trade date")
            counts["footer_rows"] += 1
            continue
        try:
            day = _day(raw_day, "trade date")
        except ValueError as error:
            raise ValueError(f"{source}, row {number}: {error}") from error
        sessions.add(day)
        counts["dated_rows"] += 1
        if not contract:
            raise ValueError(f"{source}, row {number}: unlabeled dated row outside a merge")
        if not re.fullmatch(r"cu\d{4}", contract):
            if re.fullmatch(r"cu\d+", contract):
                raise ValueError(f"{source}, row {number}: invalid copper contract ID")
            counts["non_copper_rows"] += 1
            continue
        try:
            midpoint = _month_midpoint(contract)
            values = [
                _number(row[columns[field]], field)
                for field in ("settlement", "volume", "open_interest")
            ]
        except ValueError as error:
            raise ValueError(f"{source}, row {number}: {error}") from error
        for field, value in zip(("settlement", "volume", "open_interest"), values, strict=True):
            if np.isnan(value):
                counts[f"missing_{field}"] += 1
            elif value == 0:
                counts[f"zero_{field}"] += 1
            elif value < 0:
                counts[f"negative_{field}"] += 1
        records.append((day, contract, midpoint, *values))
        counts["copper_rows"] += 1
    if columns is None:
        raise ValueError(f"{source}: no recognized SHFE header")
    copper_dates = [record[0] for record in records]
    counting = {
        "volume_convention": conventions.get("volume", "unknown"),
        "open_interest_convention": conventions.get("open_interest", "unknown"),
        "declaration_source": "explicit workbook notes" if conventions else "unavailable",
        "declaration_count": declaration_count,
        "first_copper_session": str(min(copper_dates)) if copper_dates else None,
        "last_copper_session": str(max(copper_dates)) if copper_dates else None,
        "raw_counts": "archived integer contract counts; no normalization applied",
        "single_sided_view": "explicit opt-in; double-sided counts divided by two",
    }
    return (
        records,
        sessions,
        {"source": source, "counts": dict(sorted(counts.items())), "counting": counting},
    )


def _panel(
    records: list[tuple], sessions: set[np.datetime64], provenance: dict[str, Any]
) -> ShfeCuPanel:
    if not records:
        raise ValueError("no copper futures rows in the requested archive/date range")
    records.sort(key=lambda row: (row[0], row[1]))
    columns = list(zip(*records, strict=True))
    return ShfeCuPanel(*columns, sorted(sessions), provenance=provenance)


def parse_shfe_rows(
    rows: Iterable[Sequence[Any]],
    *,
    source: str = "supplied SHFE rows",
    merged_cell_ranges: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (),
) -> ShfeCuPanel:
    """Parse a supplied table with explicit, zero-based Excel merge metadata.

    Only contract labels are expanded, and only inside declared vertical merges.
    Blank labels in dated rows outside these ranges raise instead of being filled.
    """
    records, sessions, audit = _row_records(
        rows, source=source, merged_cell_ranges=merged_cell_ranges
    )
    return _panel(records, sessions, {"tables": [audit], "units": _units()})


def _units() -> dict[str, Any]:
    return {
        "currency": "CNY",
        "price_unit": "CNY/tonne",
        "volume_unit": "contracts",
        "open_interest_unit": "contracts",
        "counting_scope": "raw archived counts; see each table's counting declaration",
        "tonnes_per_contract": 5.0,
        "specification_source": SHFE_CU_MANUAL,
        "tenor_coordinate": ShfeCuPanel.tenor_convention,
    }


def load_shfe_cu_archives(
    paths: Sequence[str | Path], *, start_date=None, end_date=None
) -> ShfeCuPanel:
    """Load official mixed XLS/XLSX ZIPs without extracting their members.

    INE workbooks are excluded. Duplicate date/contract keys raise, including
    duplicates across input archives. Missing numbers remain NaN and zero
    settlements remain zero, with counts; no repair/filter is hidden here.
    Date bounds are inclusive and also bound the observed session calendar.
    Excel dependencies are imported only when this function is called.
    """
    try:
        from python_calamine import CalamineWorkbook
    except ImportError as error:
        raise ImportError("SHFE Excel ingestion requires the 'empirical' extra") from error
    if isinstance(paths, (str, Path)) or not paths:
        raise ValueError("paths must be a nonempty sequence of annual ZIP paths")
    lower = None if start_date is None else _day(start_date, "start_date")
    upper = None if end_date is None else _day(end_date, "end_date")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("start_date must not exceed end_date")
    records, sessions, tables, archives = [], set(), [], []
    seen_paths = set()
    for supplied in paths:
        path = Path(supplied)
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError("annual archive paths must be unique")
        seen_paths.add(resolved)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        archive_record: dict[str, Any] = {
            "filename": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        manifest_path = path.with_name("acquisition_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            matching = [
                entry
                for entry in manifest.get("archives", [])
                if Path(entry.get("path", "")).name == path.name
            ]
            if len(matching) > 1:
                raise ValueError("duplicate archive entries in acquisition manifest")
            if matching:
                entry = matching[0]
                if entry.get("sha256") != digest:
                    raise ValueError("archive SHA-256 disagrees with acquisition manifest")
                for field in ("url", "retrieved_utc", "year"):
                    if field in entry:
                        archive_record[field] = entry[field]
        archives.append(archive_record)
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len({info.filename for info in infos}) != len(infos):
                raise ValueError("duplicate ZIP member names")
            if sum(info.file_size for info in infos) > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError("annual archive exceeds the uncompressed size limit")
            used = 0
            for info in infos:
                filename = info.filename.replace("\\", "/")
                if not filename.lower().endswith((".xls", ".xlsx")) or "ine" in filename.lower():
                    continue
                if (
                    filename.startswith("/")
                    or ".." in filename.split("/")
                    or info.file_size > _MAX_MEMBER_BYTES
                    or info.flag_bits & 1
                ):
                    raise ValueError("unsafe or oversized Excel ZIP member")
                payload = archive.read(info)
                member_hash = hashlib.sha256(payload).hexdigest()
                with CalamineWorkbook.from_filelike(io.BytesIO(payload)) as workbook:
                    for name in workbook.sheet_names:
                        source = f"{path.name}:{filename}:{name}"
                        sheet = workbook.get_sheet_by_name(name)
                        part, calendar, audit = _row_records(
                            sheet.iter_rows(),
                            source=source,
                            merged_cell_ranges=sheet.merged_cell_ranges or (),
                        )
                        audit.update(
                            {"member": filename, "sheet": name, "member_sha256": member_hash}
                        )
                        records.extend(part)
                        sessions.update(calendar)
                        tables.append(audit)
                used += 1
            if used == 0:
                raise ValueError("annual archive contains no SHFE Excel workbooks")
    if lower is not None:
        records = [row for row in records if row[0] >= lower]
        sessions = {day for day in sessions if day >= lower}
    if upper is not None:
        records = [row for row in records if row[0] <= upper]
        sessions = {day for day in sessions if day <= upper}
    return _panel(
        records,
        sessions,
        {
            "archives": archives,
            "tables": tables,
            "units": _units(),
            "date_range": {
                "start_inclusive": None if lower is None else str(lower),
                "end_inclusive": None if upper is None else str(upper),
            },
            "data_vintage": "retrospective official annual archives; point-in-time not verified",
            "calendar": "dates observed in SHFE workbooks; unfilled, not an inferred holiday list",
            "transformations": (
                "declared merged contract labels expanded; cuYYMM selected and sorted; "
                "no roll, date filling or price repair"
            ),
            "expiry_metadata": "not supplied; contract-month midpoint is a coordinate only",
        },
    )


def audit_shfe_daily_samples(
    panel: ShfeCuPanel,
    paths: Sequence[str | Path],
    *,
    source_urls: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare supplied daily JSON samples with the annual panel, without networking.

    Returns aggregate equality counts, errors and file hashes, never prices.
    Passing this bounded check does not certify other dates or point-in-time
    availability. Daily total rows and non-copper products are excluded.
    """
    if not isinstance(panel, ShfeCuPanel) or isinstance(paths, (str, Path)) or not paths:
        raise ValueError("supply a ShfeCuPanel and a nonempty sequence of daily JSON paths")
    lookup = {
        (str(day), str(contract)): index
        for index, (day, contract) in enumerate(zip(panel.dates, panel.contract_ids, strict=True))
    }
    fields = {
        "settlement": "SETTLEMENTPRICE",
        "volume": "VOLUME",
        "open_interest": "OPENINTEREST",
    }
    samples = []
    seen_dates = set()
    for supplied in paths:
        path = Path(supplied)
        match = re.fullmatch(r"kx(\d{8})\.dat", path.name)
        if match is None:
            raise ValueError("daily sample filenames must be kxYYYYMMDD.dat")
        day = _day(match[1], "daily sample filename")
        if day in seen_dates:
            raise ValueError("daily sample dates must be unique")
        seen_dates.add(day)
        raw = path.read_bytes()
        report = json.loads(raw)
        if not isinstance(report.get("o_curinstrument"), list):
            raise ValueError("daily sample lacks its o_curinstrument records")
        if all(key in report for key in ("o_year", "o_month", "o_day")):
            internal = (
                f"{int(report['o_year']):04d}{int(report['o_month']):02d}{int(report['o_day']):02d}"
            )
            if _day(internal, "daily report date") != day:
                raise ValueError("daily sample filename disagrees with its internal date")
        daily = {}
        for row in report["o_curinstrument"]:
            if str(row.get("PRODUCTID", "")).strip().lower() != "cu_f":
                continue
            delivery = str(row.get("DELIVERYMONTH", "")).strip()
            if not re.fullmatch(r"\d{4}", delivery):
                continue  # Product totals are not a dated futures contract.
            contract = "cu" + delivery
            _month_midpoint(contract)
            if contract in daily:
                raise ValueError("duplicate copper contract in a daily sample")
            try:
                daily[contract] = {
                    field: _number(row[source], source) for field, source in fields.items()
                }
            except KeyError as error:
                raise ValueError(
                    f"daily sample lacks the required {error.args[0]} field"
                ) from error
        if not daily:
            raise ValueError("daily sample contains no copper futures")
        annual_ids = set(panel.contract_ids[panel.dates == day].tolist())
        common = sorted(annual_ids & set(daily))
        comparisons = {}
        for field in fields:
            annual_values = np.array(
                [getattr(panel, field)[lookup[(str(day), contract)]] for contract in common]
            )
            daily_values = np.array([daily[contract][field] for contract in common])
            equal = (annual_values == daily_values) | (
                np.isnan(annual_values) & np.isnan(daily_values)
            )
            finite = np.isfinite(annual_values) & np.isfinite(daily_values)
            comparisons[field] = {
                "compared_cells": len(common),
                "exact_equal_cells": int(equal.sum()),
                "mismatched_missingness": int(
                    np.sum(np.isnan(annual_values) != np.isnan(daily_values))
                ),
                "max_absolute_error": (
                    float(np.max(np.abs(annual_values[finite] - daily_values[finite])))
                    if np.any(finite)
                    else None
                ),
            }
        samples.append(
            {
                "date": str(day),
                "filename": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_url": (source_urls or {}).get(path.name),
                "annual_contract_count": len(annual_ids),
                "daily_contract_count": len(daily),
                "common_contract_count": len(common),
                "missing_from_annual": sorted(set(daily) - annual_ids),
                "missing_from_daily": sorted(annual_ids - set(daily)),
                "fields": comparisons,
            }
        )
    compared = sum(
        field["compared_cells"] for sample in samples for field in sample["fields"].values()
    )
    equal = sum(
        field["exact_equal_cells"] for sample in samples for field in sample["fields"].values()
    )
    return {
        "sample_count": len(samples),
        "compared_field_cells": compared,
        "exact_equal_field_cells": equal,
        "all_sampled_values_equal": compared == equal
        and all(
            not row["missing_from_annual"] and not row["missing_from_daily"] for row in samples
        ),
        "samples": sorted(samples, key=lambda item: item["date"]),
        "limitation": "bounded date reconciliation; neither a whole-history nor availability audit",
    }


def _public_text(value):
    if not isinstance(value, str):
        raise ValueError("public metadata text fields must be strings")
    return value


def _public_hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("public metadata hashes must be lowercase SHA-256 digests")
    return value


def _public_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("public metadata counts must be nonnegative integers")
    return value


def _public_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("public metadata numeric fields must be finite scalar numbers")
    return value


def _public_bool(value):
    if not isinstance(value, bool):
        raise ValueError("public metadata flags must be boolean")
    return value


def _public_nullable(validator):
    return lambda value: None if value is None else validator(value)


def _public_mapping(value, schema):
    if not isinstance(value, Mapping):
        raise ValueError("public metadata sections must be mappings")
    return {key: validate(value[key]) for key, validate in schema.items() if key in value}


def _public_records(value, schema):
    if not isinstance(value, (list, tuple)):
        raise ValueError("public metadata records must be a list")
    return [_public_mapping(row, schema) for row in value]


def public_data_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Select typed source/aggregate fields; never publish arbitrary caller metadata.

    Unknown keys are dropped at every supported nesting level. Known fields with
    arrays or other invalid types fail closed rather than becoming a raw-data
    escape route. This function returns a separate JSON-compatible dictionary.
    """
    count_names = {
        "empty_rows",
        "header_rows",
        "preamble_rows",
        "footer_rows",
        "dated_rows",
        "non_copper_rows",
        "copper_rows",
        "expanded_merged_contract_labels",
        "recognized_metadata_rows",
        *(
            f"{kind}_{field}"
            for kind in ("missing", "zero", "negative")
            for field in ("settlement", "volume", "open_interest")
        ),
    }
    archive_schema = {
        "filename": _public_text,
        "sha256": _public_hash,
        "bytes": _public_count,
        "url": _public_text,
        "retrieved_utc": _public_text,
        "year": _public_count,
    }
    table_schema = {
        "source": _public_text,
        "member": _public_text,
        "sheet": _public_text,
        "member_sha256": _public_hash,
        "counts": lambda value: _public_mapping(
            value, {key: _public_count for key in sorted(count_names)}
        ),
        "counting": lambda value: _public_mapping(
            value,
            {
                key: _public_text
                for key in (
                    "volume_convention",
                    "open_interest_convention",
                    "declaration_source",
                    "raw_counts",
                    "single_sided_view",
                )
            }
            | {
                "declaration_count": _public_count,
                "first_copper_session": _public_nullable(_public_text),
                "last_copper_session": _public_nullable(_public_text),
            },
        ),
    }
    units_schema = {
        key: _public_text
        for key in (
            "currency",
            "price_unit",
            "volume_unit",
            "open_interest_unit",
            "specification_source",
            "tenor_coordinate",
            "counting_scope",
        )
    } | {"tonnes_per_contract": _public_number}
    schema = {
        "archives": lambda value: _public_records(value, archive_schema),
        "tables": lambda value: _public_records(value, table_schema),
        "units": lambda value: _public_mapping(value, units_schema),
        "date_range": lambda value: _public_mapping(
            value,
            {
                "start_inclusive": _public_nullable(_public_text),
                "end_inclusive": _public_nullable(_public_text),
            },
        ),
    } | {
        key: _public_text
        for key in (
            "data_vintage",
            "calendar",
            "transformations",
            "expiry_metadata",
        )
    }
    return _public_mapping(provenance, schema)


def public_data_validation(audit: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Select reconciliation counts, errors and source hashes, excluding raw rows.

    Missing contract lists become counts, so individual IDs need not be public.
    A missing audit remains explicitly unavailable rather than being fabricated.
    """
    if audit is None:
        return None
    field_schema = {
        key: _public_count
        for key in (
            "compared_cells",
            "exact_equal_cells",
            "mismatched_missingness",
        )
    } | {"max_absolute_error": _public_nullable(_public_number)}
    sample_schema = {
        "date": _public_text,
        "filename": _public_text,
        "sha256": _public_hash,
        "source_url": _public_nullable(_public_text),
        "annual_contract_count": _public_count,
        "daily_contract_count": _public_count,
        "common_contract_count": _public_count,
        "fields": lambda value: _public_mapping(
            value,
            {
                key: lambda field: _public_mapping(field, field_schema)
                for key in ("settlement", "volume", "open_interest")
            },
        ),
    }

    def samples(values):
        result = _public_records(values, sample_schema)
        for original, clean in zip(values, result, strict=True):
            for name in ("missing_from_annual", "missing_from_daily"):
                if name in original:
                    if not isinstance(original[name], (list, tuple)):
                        raise ValueError("missing-contract audit entries must be lists")
                    clean[name + "_count"] = len(original[name])
        return result

    def hashes(values):
        if not isinstance(values, Mapping):
            raise ValueError("annual archive hash audit must be a mapping")
        return {
            key: _public_hash(value)
            for key, value in values.items()
            if isinstance(key, str) and re.fullmatch(r"shfe-futures-\d{4}\.zip", key)
        }

    return _public_mapping(
        audit,
        {
            "sample_count": _public_count,
            "compared_field_cells": _public_count,
            "exact_equal_field_cells": _public_count,
            "all_sampled_values_equal": _public_bool,
            "samples": samples,
            "limitation": _public_text,
            "annual_archive_sha256": hashes,
        },
    )
