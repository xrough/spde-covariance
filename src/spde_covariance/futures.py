"""Immutable contract panels and explicitly declared settlement-change observations.

This adapter does not roll contracts, infer delivery-average maps, fit drifts,
or turn differenced price-level noise into fresh innovation noise.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import validate_psd
from .observations import Observation
from .trajectories import _label, _metadata_snapshot
from .workflow import ObservedInnovationBatch


def _date(value, name: str) -> date:
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"{name} must be an ISO calendar date (YYYY-MM-DD)")


def _optional_date(value, name: str) -> date | None:
    return None if value is None or value == "" else _date(value, name)


def _array(value, *, name: str, shape=None, allow_nan=False, dtype=float):
    raw = np.asarray(value)
    if dtype is bool and raw.dtype.kind != "b":
        raise ValueError(f"{name} must be boolean")
    if dtype is not bool and (np.iscomplexobj(raw) or raw.dtype.kind not in "iuf"):
        raise ValueError(f"{name} must contain real numbers")
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if dtype is not bool and (
        np.any(np.isinf(result)) or (not allow_nan and np.any(np.isnan(result)))
    ):
        raise ValueError(f"{name} must be finite" + (" or missing (NaN)" if allow_nan else ""))
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def _calendar(values) -> tuple[date, ...]:
    dates = tuple(_date(value, "valuation_date") for value in values)
    if not dates or any(right <= left for left, right in zip(dates[:-1], dates[1:], strict=True)):
        raise ValueError("valuation_dates must be nonempty, unique, and strictly increasing")
    return dates


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """One supplied metadata snapshot; delivery dates are logistical metadata.

    ``metadata_asof`` means the snapshot was available by that date. Expiry and
    physical delivery dates may be unknown or provisional. They never generate
    an observation operator automatically.
    """

    contract_id: str
    exchange: str
    product: str
    currency: str
    price_unit: str
    multiplier: float
    listing_date: date
    delivery_month: str
    metadata_asof: date
    source_url: str
    source_sha256: str
    expiry_date: date | None = None
    delivery_start: date | None = None
    delivery_end: date | None = None
    calendar_adjusted: bool = False

    def __post_init__(self):
        for name in ("contract_id", "exchange", "product", "currency", "price_unit", "source_url"):
            object.__setattr__(self, name, _label(getattr(self, name), name))
        if (
            isinstance(self.multiplier, (bool, np.bool_))
            or not np.isscalar(self.multiplier)
            or not np.isreal(self.multiplier)
            or not np.isfinite(self.multiplier)
            or self.multiplier <= 0
        ):
            raise ValueError("multiplier must be a finite positive real number")
        object.__setattr__(self, "multiplier", float(self.multiplier))
        for name in ("listing_date", "metadata_asof"):
            object.__setattr__(self, name, _date(getattr(self, name), name))
        for name in ("expiry_date", "delivery_start", "delivery_end"):
            object.__setattr__(self, name, _optional_date(getattr(self, name), name))
        if len(self.delivery_month) != 7:
            raise ValueError("delivery_month must be YYYY-MM")
        _date(self.delivery_month + "-01", "delivery_month")
        if self.expiry_date is not None and self.expiry_date < self.listing_date:
            raise ValueError("expiry_date precedes listing_date")
        if (self.delivery_start is None) != (self.delivery_end is None):
            raise ValueError(
                "delivery_start and delivery_end must both be supplied or both missing"
            )
        if self.delivery_start is not None and self.delivery_end < self.delivery_start:
            raise ValueError("delivery_end precedes delivery_start")
        if not isinstance(self.calendar_adjusted, bool):
            raise ValueError("calendar_adjusted must be a boolean")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class SettlementPanel:
    """Contract columns on the caller's complete, unfilled valuation calendar.

    Any present settlement, volume, or OI requires a publication-availability
    date for that cell. Missing fields use NaN and missing availability uses NaT.
    Fields published at different dates must use their latest availability date.
    All contracts must
    share a currency and price unit; no currency or quotation conversion occurs.
    """

    valuation_dates: tuple[date, ...]
    contracts: tuple[ContractSpec, ...]
    settlements: NDArray[np.float64]
    available_dates: NDArray[np.datetime64]
    volume: NDArray[np.float64] | None = None
    open_interest: NDArray[np.float64] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        dates = _calendar(self.valuation_dates)
        specs = tuple(self.contracts)
        if not specs or not all(isinstance(spec, ContractSpec) for spec in specs):
            raise ValueError("contracts must contain ContractSpec objects")
        if len({spec.contract_id for spec in specs}) != len(specs):
            raise ValueError("contract_id values must be unique")
        if len({(spec.currency, spec.price_unit) for spec in specs}) != 1:
            raise ValueError(
                "contracts must share currency and price_unit; convert explicitly first"
            )
        shape = (len(dates), len(specs))
        prices = _array(self.settlements, name="settlements", shape=shape, allow_nan=True)
        present = np.isfinite(prices)
        for name in ("volume", "open_interest"):
            value = getattr(self, name)
            if value is not None:
                array = _array(value, name=name, shape=shape, allow_nan=True)
                if np.any(array < 0):
                    raise ValueError(f"{name} cannot be negative")
                present |= np.isfinite(array)
                object.__setattr__(self, name, array)
        try:
            available = np.array(self.available_dates, dtype="datetime64[D]", copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("available_dates must contain dates or NaT") from error
        if available.shape != shape:
            raise ValueError(f"available_dates must have shape {shape}")
        if np.any(present & np.isnat(available)):
            raise ValueError("every present settlement, volume, or OI requires available_dates")
        calendar = np.asarray(dates, dtype="datetime64[D]")[:, None]
        if np.any(present & (available < calendar)):
            raise ValueError("market data cannot be available before its valuation_date")
        listed = np.asarray([spec.listing_date for spec in specs], dtype="datetime64[D]")[None, :]
        if np.any(present & (calendar < listed)):
            raise ValueError("market data cannot precede the contract listing_date")
        available[~present] = np.datetime64("NaT", "D")
        available = np.frombuffer(available.tobytes(), dtype=available.dtype).reshape(
            available.shape
        )
        object.__setattr__(self, "valuation_dates", dates)
        object.__setattr__(self, "contracts", specs)
        object.__setattr__(self, "settlements", prices)
        object.__setattr__(self, "available_dates", available)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _metadata_snapshot(self.metadata))

    @property
    def contract_ids(self) -> tuple[str, ...]:
        return tuple(spec.contract_id for spec in self.contracts)

    @classmethod
    def from_csv(cls, prices_path, contracts_path, *, valuation_dates: Sequence[date | str]):
        """Load the documented long CSV schema, retaining absent calendar rows.

        ``available_date`` is source availability, not the local retrieval date.
        Each input CSV is hashed. Price rows also carry source report URL/hash;
        hashes preserve provenance but do not authenticate an exchange report.
        """
        dates = _calendar(valuation_dates)
        with Path(contracts_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "contract_id",
                "exchange",
                "product",
                "currency",
                "price_unit",
                "multiplier",
                "listing_date",
                "delivery_month",
                "metadata_asof",
                "source_url",
                "source_sha256",
                "calendar_adjusted",
            }
            _required_columns(reader, required, "contracts")
            specs = []
            for row in reader:
                if row["calendar_adjusted"] not in ("true", "false"):
                    raise ValueError("calendar_adjusted must be 'true' or 'false'")
                specs.append(
                    ContractSpec(
                        **{
                            name: row[name]
                            for name in required - {"multiplier", "calendar_adjusted"}
                        },
                        multiplier=float(row["multiplier"]),
                        calendar_adjusted=row["calendar_adjusted"] == "true",
                        **{
                            name: row.get(name) or None
                            for name in ("expiry_date", "delivery_start", "delivery_end")
                        },
                    )
                )
        if len({spec.contract_id for spec in specs}) != len(specs):
            raise ValueError("contract_id values must be unique")
        columns = {spec.contract_id: index for index, spec in enumerate(specs)}
        calendar = {value: index for index, value in enumerate(dates)}
        shape = (len(dates), len(specs))
        prices = np.full(shape, np.nan)
        available = np.full(shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
        volume, interest = np.full(shape, np.nan), np.full(shape, np.nan)
        seen, sources = set(), set()
        with Path(prices_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _required_columns(
                reader,
                {
                    "contract_id",
                    "valuation_date",
                    "settlement",
                    "available_date",
                    "source_url",
                    "source_sha256",
                },
                "prices",
            )
            for row in reader:
                key = (_date(row["valuation_date"], "valuation_date"), row["contract_id"])
                if key[0] not in calendar or key[1] not in columns:
                    raise ValueError("price row refers to an undeclared calendar date or contract")
                if key in seen:
                    raise ValueError("duplicate contract-date price row")
                seen.add(key)
                i, j = calendar[key[0]], columns[key[1]]
                if row["settlement"]:
                    prices[i, j] = float(row["settlement"])
                for name, target in (("volume", volume), ("open_interest", interest)):
                    if row.get(name):
                        target[i, j] = float(row[name])
                if any(row.get(name) for name in ("settlement", "volume", "open_interest")):
                    available[i, j] = np.datetime64(_date(row["available_date"], "available_date"))
                    if not row["source_url"] or not _valid_hash(row["source_sha256"]):
                        raise ValueError(
                            "present market fields require source_url and source_sha256"
                        )
                    sources.add((row["source_url"], row["source_sha256"]))
        return cls(
            dates,
            tuple(specs),
            prices,
            available,
            volume,
            interest,
            metadata={
                "input_prices_sha256": _file_hash(prices_path),
                "input_contracts_sha256": _file_hash(contracts_path),
                "price_sources": tuple(sorted(sources)),
                "calendar_source": "explicit caller-supplied valuation_dates; never filled or resampled",
            },
        )

    def asof(self, cutoff: date | str) -> SettlementPanel:
        """Restrict dates, metadata, and quote availability before downstream selection."""
        cutoff = _date(cutoff, "cutoff")
        dates = tuple(value for value in self.valuation_dates if value <= cutoff)
        columns = [
            j
            for j, spec in enumerate(self.contracts)
            if spec.metadata_asof <= cutoff and spec.listing_date <= cutoff
        ]
        if not dates or not columns:
            raise ValueError("no calendar dates or contract metadata available as of cutoff")
        selection = np.ix_(np.arange(len(dates)), columns)
        prices = self.settlements[selection].copy()
        available = self.available_dates[selection].copy()
        unavailable = available > np.datetime64(cutoff)
        prices[unavailable] = np.nan
        available[unavailable] = np.datetime64("NaT", "D")
        auxiliaries = []
        for original in (self.volume, self.open_interest):
            value = None if original is None else original[selection].copy()
            if value is not None:
                value[unavailable] = np.nan
            auxiliaries.append(value)
        return SettlementPanel(
            dates,
            tuple(self.contracts[j] for j in columns),
            prices,
            available,
            *auxiliaries,
            metadata=dict(self.metadata) | {"asof": cutoff.isoformat()},
        )

    def normalized_changes(
        self,
        *,
        asof: date | str,
        integrated_drift: ArrayLike | str,
        price_assumption: str,
    ) -> SettlementChanges:
        """Compute (same-contract difference - supplied drift)/sqrt(ACT/365.25).

        Explicit ``integrated_drift='zero'`` is a modeling assumption, not an
        estimated drift. Otherwise supply drift increments in price units on
        the as-of panel shape. Noisy price levels require a separate error model;
        no fresh-noise correction is inferred here.
        """
        panel = self.asof(asof)
        if len(panel.valuation_dates) < 2:
            raise ValueError("at least two valuation dates are needed")
        if price_assumption not in ("clean_settlement_levels", "reduced_form_settlements"):
            raise ValueError(
                "price_assumption must explicitly be clean_settlement_levels or "
                "reduced_form_settlements; noisy levels need a separate error model"
            )
        raw = np.diff(panel.settlements, axis=0)
        mask = np.isfinite(panel.settlements[:-1]) & np.isfinite(panel.settlements[1:])
        # A contract chosen for an interval must already have metadata at its start.
        for j, spec in enumerate(panel.contracts):
            mask[:, j] &= np.array(
                [spec.metadata_asof <= start for start in panel.valuation_dates[:-1]]
            )
        raw[~mask] = np.nan
        if isinstance(integrated_drift, str):
            if integrated_drift != "zero":
                raise ValueError("integrated_drift string must be 'zero'")
            drift = np.zeros(raw.shape)
            drift_label = "zero integrated drift explicitly assumed"
        else:
            drift = _array(integrated_drift, name="integrated_drift", shape=raw.shape)
            drift_label = "integrated drift supplied by caller in settlement price units"
        return SettlementChanges(
            panel.valuation_dates,
            panel.contracts,
            raw,
            drift,
            mask,
            _date(asof, "asof"),
            price_assumption,
            dict(panel.metadata) | {"drift_convention": drift_label},
        )


@dataclass(frozen=True, slots=True)
class SettlementChanges:
    """Reduced-form contract changes, with calendar and joint masks retained."""

    valuation_dates: tuple[date, ...]
    contracts: tuple[ContractSpec, ...]
    raw_changes: NDArray[np.float64]
    integrated_drift: NDArray[np.float64]
    observed_mask: NDArray[np.bool_]
    asof_date: date
    price_assumption: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        dates = _calendar(self.valuation_dates)
        specs = tuple(self.contracts)
        if len(dates) < 2 or not specs or not all(isinstance(spec, ContractSpec) for spec in specs):
            raise ValueError("changes need at least two dates and one contract")
        if len({spec.contract_id for spec in specs}) != len(specs):
            raise ValueError("contract_id values must be unique")
        if len({(spec.currency, spec.price_unit) for spec in specs}) != 1:
            raise ValueError("contracts must share currency and price_unit")
        if self.price_assumption not in ("clean_settlement_levels", "reduced_form_settlements"):
            raise ValueError(
                "price_assumption must explicitly declare clean or reduced-form levels"
            )
        shape = (len(dates) - 1, len(specs))
        raw = _array(self.raw_changes, name="raw_changes", shape=shape, allow_nan=True)
        mask = _array(self.observed_mask, name="observed_mask", shape=shape, dtype=bool)
        if not np.array_equal(np.isfinite(raw), mask):
            raise ValueError("observed_mask must match finite raw_changes")
        for j, spec in enumerate(specs):
            eligible = np.array(
                [spec.metadata_asof <= start and spec.listing_date <= start for start in dates[:-1]]
            )
            if np.any(mask[:, j] & ~eligible):
                raise ValueError(
                    "observed changes require metadata and listing by the interval start"
                )
        object.__setattr__(self, "valuation_dates", dates)
        object.__setattr__(self, "contracts", specs)
        object.__setattr__(self, "raw_changes", raw)
        object.__setattr__(self, "observed_mask", mask)
        object.__setattr__(
            self,
            "integrated_drift",
            _array(self.integrated_drift, name="integrated_drift", shape=shape),
        )
        cutoff = _date(self.asof_date, "asof_date")
        if dates[-1] > cutoff:
            raise ValueError("changes cannot contain dates after asof_date")
        if any(spec.metadata_asof > cutoff for spec in specs):
            raise ValueError("contract metadata cannot be after asof_date")
        object.__setattr__(self, "asof_date", cutoff)
        object.__setattr__(self, "metadata", _metadata_snapshot(self.metadata))

    @property
    def contract_ids(self) -> tuple[str, ...]:
        return tuple(spec.contract_id for spec in self.contracts)

    @property
    def interval_lengths(self) -> NDArray[np.float64]:
        return _array(
            [
                (b - a).days / 365.25
                for a, b in zip(self.valuation_dates[:-1], self.valuation_dates[1:], strict=True)
            ],
            name="interval_lengths",
        )

    @property
    def values(self) -> NDArray[np.float64]:
        return _array(
            (self.raw_changes - self.integrated_drift) / np.sqrt(self.interval_lengths[:, None]),
            name="values",
            allow_nan=True,
        )

    @property
    def joint_counts(self) -> NDArray[np.int64]:
        mask = self.observed_mask.astype(np.int64)
        result = mask.T @ mask
        return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)

    def to_observed_batch(
        self,
        *,
        operators: Sequence[ArrayLike],
        map_asof_dates: Sequence[date | str],
        basis_id: str,
        observation_map_assumption: str,
        noise_assumption: str,
        curve_unit: str,
        noise_covariances: Sequence[ArrayLike] | None = None,
    ) -> ObservedInnovationBatch:
        """Bridge caller-declared maps and normalized noise into the workflow.

        Maps have all contract rows in ``contract_ids`` order before masking.
        Their declared availability must precede the interval. Noise matrices
        are in normalized-change units squared and have the same full row order.
        Delivery metadata never selects the maps. This method does not assert
        that settlement prices are coefficients of a clean H-valued trajectory.
        """
        count, width = self.observed_mask.shape
        if len(operators) != count or len(map_asof_dates) != count:
            raise ValueError("supply one operator and map_asof_date per retained interval")
        assumption = _label(observation_map_assumption, "observation_map_assumption")
        if curve_unit != self.contracts[0].price_unit:
            raise ValueError(
                "curve_unit must match price_unit; perform any unit conversion explicitly"
            )
        if noise_assumption not in ("none", "known_fresh_innovation_noise"):
            raise ValueError(
                "noise_assumption must be none or known_fresh_innovation_noise; "
                "differenced noisy levels are not fresh innovation noise"
            )
        if noise_assumption == "none" and noise_covariances is not None:
            raise ValueError("noise_covariances require known_fresh_innovation_noise")
        if noise_assumption != "none" and (
            noise_covariances is None or len(noise_covariances) != count
        ):
            raise ValueError("supply one known normalized noise covariance per interval")
        times = np.array(
            [(value - self.valuation_dates[0]).days / 365.25 for value in self.valuation_dates]
        )
        observations = []
        normalized_values = self.values
        for i, (operator, known_by) in enumerate(zip(operators, map_asof_dates, strict=True)):
            if _date(known_by, "map_asof_date") > self.valuation_dates[i]:
                raise ValueError("observation maps must be available by the interval start")
            matrix = _array(operator, name="operator")
            if matrix.ndim != 2 or matrix.shape[0] != width or matrix.shape[1] == 0:
                raise ValueError(
                    "each operator must have (number of contracts, basis dimension) shape"
                )
            mask = self.observed_mask[i]
            if not np.any(mask):
                raise ValueError(
                    "an interval has no jointly observed contracts; choose an explicit "
                    "calendar window rather than dropping or bridging it"
                )
            noise = (
                np.zeros((width, width))
                if noise_covariances is None
                else _array(noise_covariances[i], name="noise_covariance", shape=(width, width))
            )
            noise = validate_psd(noise, name="noise_covariance")
            observations.append(
                Observation(
                    times[i + 1],
                    normalized_values[i, mask],
                    matrix[mask],
                    noise[np.ix_(mask, mask)],
                )
            )
        metadata = dict(self.metadata) | {
            "asof": self.asof_date.isoformat(),
            "contract_ids": self.contract_ids,
            "valuation_dates": tuple(value.isoformat() for value in self.valuation_dates),
            "day_count": "ACT/365.25",
            "price_assumption": self.price_assumption,
            "observation_map_assumption": assumption,
            "observation_noise": noise_assumption,
            "noise_assumption_status": "declared by caller; not inferred from settlement levels",
            "trajectory_claim": "reduced-form settlement changes; no H-valued trajectory assertion",
            "physical_delivery_implies_average_observation": False,
        }
        return ObservedInnovationBatch(
            times,
            tuple(observations),
            basis_id,
            "same-contract settlement differences; no transport extraction",
            curve_unit=curve_unit,
            metadata=metadata,
        )

    def contract_pnl(self, positions: Mapping[str, float], *, interval_index: int) -> float:
        """Realized currency P&L from raw price changes, without drift subtraction.

        Positions are signed numbers of contracts. Missing held-contract prices
        raise an error instead of silently shrinking the portfolio.
        """
        quantities = self._positions(positions)
        if not isinstance(interval_index, (int, np.integer)) or not 0 <= interval_index < len(
            self.raw_changes
        ):
            raise ValueError("interval_index is out of range")
        held = quantities != 0
        if np.any(held & ~self.observed_mask[interval_index]):
            raise ValueError("a held contract lacks an observed same-contract price change")
        multipliers = np.array([spec.multiplier for spec in self.contracts])
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(
                (quantities[held] * multipliers[held]) @ self.raw_changes[interval_index, held]
            )
        if not np.isfinite(result):
            raise ValueError("P&L must remain finite after multiplier and price-change conversion")
        return result

    def pnl_loading(
        self, positions: Mapping[str, float], operator: ArrayLike
    ) -> NDArray[np.float64]:
        """Map contract counts to a currency-risk exposure in the supplied basis.

        For covariance of normalized changes, variance over dt is
        ``dt * loading @ covariance @ loading``. No annualization is applied twice.
        """
        matrix = _array(operator, name="operator")
        if matrix.ndim != 2 or matrix.shape[0] != len(self.contracts) or matrix.shape[1] == 0:
            raise ValueError("operator must have one row per contract")
        with np.errstate(over="ignore", invalid="ignore"):
            cash_weights = self._positions(positions) * [spec.multiplier for spec in self.contracts]
            return _array(cash_weights @ matrix, name="pnl_loading")

    def _positions(self, positions):
        if not isinstance(positions, Mapping) or set(positions) - set(self.contract_ids):
            raise ValueError("positions must map known contract IDs to signed contract counts")
        return _array([positions.get(key, 0.0) for key in self.contract_ids], name="positions")


def _required_columns(reader, required, name):
    fields = reader.fieldnames or ()
    if len(fields) != len(set(fields)):
        raise ValueError(f"{name} CSV has duplicate header columns")
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{name} CSV is missing columns: {', '.join(sorted(missing))}")


def _valid_hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
