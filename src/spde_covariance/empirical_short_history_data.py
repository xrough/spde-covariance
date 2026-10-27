"""Information-safe panels for masked-history copper risk-transfer episodes.

Only the masked panel is passed to forecasting code.  The original session
calendar is retained, so an absent price never creates a bridged return.
All price changes are within actual contract IDs, in CNY/tonne; annualization
uses ACT/365.25 and forecasts cover one known subsequent session interval.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .empirical_data import ShfeCuPanel
from .hedging import _snapshot


@dataclass(frozen=True, slots=True)
class ShortHistoryConfig:
    """Prespecified experiment settings; test outcomes never select them."""

    name: str = "shfe-cu-short-history"
    training_end: str = "2022-12-31"
    validation_end: str = "2023-12-31"
    test_end: str = "2025-12-31"
    window: int = 252
    origin_stride: int = 5
    history_budgets: tuple[int | None, ...] = (0, 5, 20, None)
    half_life: float = 63.0
    sensitivity_half_lives: tuple[float, ...] = (21.0, 126.0)
    min_training_intervals: int = 126
    dimension: int = 3
    factor_dimension: int = 3
    smoothing: float = 1e-6
    reconstruction_ridge: float = 1e-6
    relative_regularization: float = 1e-3
    numerical_error_budget: float = 1e-5
    trace_cap_multiplier: float = 100.0
    rank_rtol: float = 1e-10
    min_tenor_days: int = 30
    max_tenor_days: int = 365
    target_tenors_days: tuple[int, ...] = (90, 180, 270)
    multiplier: float = 5.0
    pnl_variance_floor: float = 25.0
    min_direct_count: int = 20
    hedge_bounds: tuple[float, float] = (-2.0, 2.0)
    bootstrap_block: int = 4
    bootstrap_sensitivity_blocks: tuple[int, ...] = (1, 2, 8)
    bootstrap_replications: int = 1000
    seed: int = 20261027
    cache_entries: int = 64
    cache_bytes: int = 64 * 1024**2

    def __post_init__(self):
        boundaries = tuple(
            np.datetime64(value, "D")
            for value in (self.training_end, self.validation_end, self.test_end)
        )
        if (
            any(np.isnat(day) for day in boundaries)
            or not boundaries[0] < boundaries[1] < boundaries[2]
        ):
            raise ValueError("training, validation and test dates must increase")
        for name in (
            "window",
            "origin_stride",
            "min_training_intervals",
            "dimension",
            "factor_dimension",
            "min_direct_count",
            "bootstrap_block",
            "bootstrap_replications",
            "cache_entries",
            "cache_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.window < self.min_training_intervals or self.bootstrap_replications < 2:
            raise ValueError("window must cover minimum training and bootstrap needs >=2 draws")
        if self.min_direct_count < 2:
            raise ValueError("direct minimum must be at least two")
        if self.dimension != 3 or self.factor_dimension != 3:
            raise ValueError("this protocol fixes source and learned factor dimensions at three")
        if (self.min_tenor_days, self.max_tenor_days) != (30, 365):
            raise ValueError("this protocol fixes the maturity domain at 30–365 days")
        if self.reconstruction_ridge != 1e-6:
            raise ValueError("this protocol fixes reconstruction_ridge at 1e-6")
        if (
            not isinstance(self.history_budgets, tuple)
            or not self.history_budgets
            or len(set(self.history_budgets)) != len(self.history_budgets)
            or any(
                h is not None
                and (
                    not isinstance(h, (int, np.integer))
                    or isinstance(h, bool)
                    or not 0 <= h <= self.window
                )
                for h in self.history_budgets
            )
        ):
            raise ValueError("history_budgets must be distinct counts in [0, window] or None")
        if (
            not isinstance(self.target_tenors_days, tuple)
            or len(self.target_tenors_days) != 3
            or tuple(sorted(set(self.target_tenors_days))) != self.target_tenors_days
            or self.min_tenor_days < 1
            or self.max_tenor_days <= self.min_tenor_days
            or not all(
                self.min_tenor_days <= t <= self.max_tenor_days for t in self.target_tenors_days
            )
        ):
            raise ValueError("three increasing target maturities within the tenor domain required")
        for name in (
            "half_life",
            "smoothing",
            "reconstruction_ridge",
            "relative_regularization",
            "numerical_error_budget",
            "trace_cap_multiplier",
            "rank_rtol",
            "multiplier",
            "pnl_variance_floor",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rank_rtol >= 1:
            raise ValueError("rank_rtol must be below one")
        if not isinstance(self.sensitivity_half_lives, tuple) or any(
            not np.isfinite(x) or x <= 0 for x in self.sensitivity_half_lives
        ):
            raise ValueError("sensitivity half lives must be a tuple of positive finite values")
        if not isinstance(self.bootstrap_sensitivity_blocks, tuple) or any(
            not isinstance(x, (int, np.integer)) or x < 1 for x in self.bootstrap_sensitivity_blocks
        ):
            raise ValueError("bootstrap block sensitivities must be positive integers")
        if (
            not isinstance(self.hedge_bounds, tuple)
            or len(self.hedge_bounds) != 2
            or not all(np.isfinite(self.hedge_bounds))
            or not self.hedge_bounds[0] <= 0 <= self.hedge_bounds[1]
            or self.hedge_bounds[0] >= self.hedge_bounds[1]
        ):
            raise ValueError("hedge bounds must be finite, increasing and contain zero")
        if (
            not isinstance(self.seed, (int, np.integer))
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if not self.name or any(c in self.name for c in "/\\"):
            raise ValueError("name must be a simple nonempty identifier")

    @property
    def maturity_min_days(self):
        return self.min_tenor_days

    @property
    def maturity_max_days(self):
        return self.max_tenor_days

    @property
    def source_numerical_budget(self):
        return self.numerical_error_budget


def load_short_history_config(path):
    """Read a flat TOML protocol; the string ``all`` means unmasked history."""
    import tomllib

    values = tomllib.loads(Path(path).read_text())
    for name in (
        "history_budgets",
        "sensitivity_half_lives",
        "target_tenors_days",
        "hedge_bounds",
        "bootstrap_sensitivity_blocks",
    ):
        if name in values:
            values[name] = tuple(None if item == "all" else item for item in values[name])
    return ShortHistoryConfig(**values)


@dataclass(frozen=True, slots=True)
class ShortHistoryEpisode:
    """Origin metadata only, with target-first evaluation coordinates."""

    origin: np.datetime64
    next_date: np.datetime64
    phase: str
    target_id: str
    target_tenor_days: int
    evaluation_ids: tuple[str, str, str]
    maturity_order: tuple[int, int, int]

    def __post_init__(self):
        object.__setattr__(self, "origin", np.datetime64(self.origin, "D"))
        object.__setattr__(self, "next_date", np.datetime64(self.next_date, "D"))
        if np.isnat(self.origin) or np.isnat(self.next_date) or self.next_date <= self.origin:
            raise ValueError("episode must have an increasing known next session date")
        if self.phase not in ("validation", "test", "evaluation"):
            raise ValueError("phase must be validation or test/evaluation")
        if (
            not isinstance(self.evaluation_ids, tuple)
            or len(set(self.evaluation_ids)) != 3
            or self.evaluation_ids[0] != self.target_id
        ):
            raise ValueError("evaluation IDs must be three distinct target-first contracts")
        if not isinstance(self.maturity_order, tuple) or sorted(self.maturity_order) != [0, 1, 2]:
            raise ValueError("maturity_order must index near/middle/far in evaluation coordinates")


@dataclass(frozen=True, slots=True, eq=False)
class ShortHistoryView:
    """Masked data boundary: no reference to the original/full panel survives."""

    panel: ShfeCuPanel
    episode: ShortHistoryEpisode
    history_budget: int | None
    maturities: np.ndarray
    origin_prices: np.ndarray
    fingerprint: str
    available: bool
    reason: str | None
    min_tenor_days: int
    max_tenor_days: int

    def __post_init__(self):
        object.__setattr__(
            self, "maturities", _snapshot(np.asarray(self.maturities, dtype="datetime64[D]"))
        )
        object.__setattr__(
            self, "origin_prices", _snapshot(np.asarray(self.origin_prices, dtype=float))
        )

    @property
    def origin(self):
        return self.episode.origin

    @property
    def next_date(self):
        return self.episode.next_date

    @property
    def evaluation_ids(self):
        return self.episode.evaluation_ids

    @property
    def target_id(self):
        return self.episode.target_id

    @property
    def target_tenor_days(self):
        return self.episode.target_tenor_days

    @property
    def dates(self):
        return self.panel.session_dates


def _section(panel, day, *, eligible, minimum, maximum):
    rows = np.flatnonzero(panel.dates == day)
    valid = np.isfinite(panel.settlement[rows]) & (panel.settlement[rows] > 0)
    if eligible:
        days = (panel.contract_month_dates[rows] - day).astype(int)
        valid &= (
            (days >= minimum)
            & (days <= maximum)
            & (panel.volume[rows] > 0)
            & (panel.open_interest[rows] > 0)
        )
    rows = rows[valid]
    rows = rows[np.argsort(panel.contract_month_dates[rows], kind="stable")]
    return MappingProxyType(
        {
            "ids": _snapshot(panel.contract_ids[rows]),
            "maturity": _snapshot(panel.contract_month_dates[rows]),
            "prices": _snapshot(panel.settlement[rows]),
            "volume": _snapshot(panel.volume[rows]),
            "open_interest": _snapshot(panel.open_interest[rows]),
        }
    )


def make_short_history_episodes(panel, config, phase="test"):
    """Select current contracts without consulting subsequent prices or eligibility."""
    if phase == "validation":
        lower, upper = config.training_end, config.validation_end
    elif phase in ("test", "evaluation"):
        lower, upper = config.validation_end, config.test_end
    else:
        raise ValueError("phase must be validation or test/evaluation")
    dates = panel.session_dates
    candidates = np.flatnonzero(
        (dates > np.datetime64(lower, "D")) & (dates <= np.datetime64(upper, "D"))
    )
    candidates = candidates[
        (candidates >= config.min_training_intervals) & (candidates < len(dates) - 1)
    ]
    candidates = candidates[dates[candidates + 1] <= np.datetime64(upper, "D")]
    selected = {}
    for index in candidates:
        section = _section(
            panel,
            dates[index],
            eligible=True,
            minimum=config.min_tenor_days,
            maximum=config.max_tenor_days,
        )
        if len(section["ids"]) < 3:
            continue
        tenor = (section["maturity"] - dates[index]).astype(int)
        chosen = tuple(int(np.argmin(np.abs(tenor - t))) for t in config.target_tenors_days)
        if len(set(chosen)) == 3:
            selected[int(index)] = tuple(str(section["ids"][j]) for j in chosen)
    if not selected:
        return ()
    anchor = min(selected)
    episodes = []
    for index in range(anchor, int(candidates[-1]) + 1, config.origin_stride):
        if index not in selected:
            continue
        triad = selected[index]
        for target_slot, target_id in enumerate(triad):
            # Remaining legs are sorted by distance to the target, then maturity.
            current = _section(
                panel,
                dates[index],
                eligible=True,
                minimum=config.min_tenor_days,
                maximum=config.max_tenor_days,
            )
            lookup = dict(zip(current["ids"], current["maturity"], strict=True))
            target_month = lookup[target_id]
            others = sorted(
                (x for x in triad if x != target_id),
                key=lambda x: (abs(int((lookup[x] - target_month).astype(int))), lookup[x]),
            )
            ordered = (target_id, *others)
            episodes.append(
                ShortHistoryEpisode(
                    dates[index],
                    dates[index + 1],
                    phase,
                    target_id,
                    config.target_tenors_days[target_slot],
                    ordered,
                    tuple(ordered.index(x) for x in triad),
                )
            )
    return tuple(episodes)


def build_short_history_view(panel_prefix, episode, history_budget, config):
    """Remove hidden target rows before any transform; current metadata is allowed."""
    if history_budget is not None and (
        not isinstance(history_budget, (int, np.integer))
        or isinstance(history_budget, bool)
        or not 0 <= history_budget <= config.window
    ):
        raise ValueError("history budget must be None or a count in [0, window]")
    dates = panel_prefix.session_dates[panel_prefix.session_dates <= episode.origin]
    if not len(dates) or dates[-1] != episode.origin:
        raise ValueError("origin must be present in the observed session calendar")
    dates = dates[-(config.window + 1) :]
    retain = np.isin(panel_prefix.dates, dates)
    reason = None
    if len(dates) - 1 < config.min_training_intervals:
        reason = "insufficient_training_intervals"
    if history_budget is not None:
        suffix = dates[-(history_budget + 1) :]
        target = panel_prefix.contract_ids == episode.target_id
        retain &= ~target | np.isin(panel_prefix.dates, suffix)
        target_dates = panel_prefix.dates[
            target
            & np.isin(panel_prefix.dates, suffix)
            & np.isfinite(panel_prefix.settlement)
            & (panel_prefix.settlement > 0)
        ]
        if len(suffix) != history_budget + 1 or not np.array_equal(target_dates, suffix):
            reason = "requested_contiguous_target_history_unavailable"
    # Do not copy full-panel provenance: arbitrary provenance may contain prices.
    masked = ShfeCuPanel(
        panel_prefix.dates[retain],
        panel_prefix.contract_ids[retain],
        panel_prefix.contract_month_dates[retain],
        panel_prefix.settlement[retain],
        panel_prefix.volume[retain],
        panel_prefix.open_interest[retain],
        dates,
        provenance={"masked_history": True},
    )
    current = _section(
        masked,
        episode.origin,
        eligible=True,
        minimum=config.min_tenor_days,
        maximum=config.max_tenor_days,
    )
    lookup = {str(x): i for i, x in enumerate(current["ids"])}
    if not all(x in lookup for x in episode.evaluation_ids):
        raise ValueError("episode evaluation contracts must be origin-eligible")
    indices = [lookup[x] for x in episode.evaluation_ids]
    maturities, prices = current["maturity"][indices], current["prices"][indices]
    digest = hashlib.sha256()
    for column in (
        masked.dates,
        masked.contract_ids,
        masked.contract_month_dates,
        masked.settlement,
        masked.volume,
        masked.open_interest,
        masked.session_dates,
    ):
        digest.update(column.dtype.str.encode())
        digest.update(column.tobytes())
    digest.update(
        json.dumps(
            {
                "origin": str(episode.origin),
                "next_date": str(episode.next_date),
                "ids": episode.evaluation_ids,
                "budget": history_budget,
                "tenor_domain": [config.min_tenor_days, config.max_tenor_days],
            },
            sort_keys=True,
        ).encode()
    )
    return ShortHistoryView(
        masked,
        episode,
        history_budget,
        maturities,
        prices,
        digest.hexdigest(),
        reason is None,
        reason,
        config.min_tenor_days,
        config.max_tenor_days,
    )


def daily_sections(view, eligible=True):
    """Return immutable sections aligned with every original training session."""
    return tuple(
        _section(
            view.panel,
            date,
            eligible=eligible,
            minimum=view.min_tenor_days,
            maximum=view.max_tenor_days,
        )
        for date in view.panel.session_dates
    )


def interval_records(view, config=None, *, eligible=True):
    """Adjacent same-ID changes, without crossing an absent session.

    The default applies the donor eligibility rule at both endpoints. Direct
    same-contract baselines use ``eligible=False`` to retain all positive
    observed settlements, matching the existing empirical benchmark.
    """
    sections, dates = daily_sections(view, eligible=eligible), view.panel.session_dates
    records = []
    for left, right, first, last in zip(
        sections[:-1], sections[1:], dates[:-1], dates[1:], strict=True
    ):
        ids, left_index, right_index = np.intersect1d(
            left["ids"], right["ids"], return_indices=True
        )
        order = np.argsort(right["maturity"][right_index], kind="stable")
        ids, left_index, right_index = ids[order], left_index[order], right_index[order]
        delta = int((last - first).astype(int)) / 365.25
        records.append(
            MappingProxyType(
                {
                    "left_date": first,
                    "right_date": last,
                    "delta": delta,
                    "ids": _snapshot(ids),
                    "response": _snapshot(
                        (right["prices"][right_index] - left["prices"][left_index]) / np.sqrt(delta)
                    ),
                    "tenors_days": _snapshot((right["maturity"][right_index] - last).astype(int)),
                    "maturity": _snapshot(right["maturity"][right_index]),
                    "left_section": left,
                    "right_section": right,
                }
            )
        )
    return tuple(records)


def _moment(values, ages, half_life):
    values, ages = np.asarray(values, dtype=float), np.asarray(ages, dtype=float)
    if len(values) == 0:
        return None, 0.0
    weights = np.ones(len(values)) if half_life is None else np.exp2(-ages / half_life)
    weights /= np.sum(weights)
    return (values.T * weights) @ values, float(1 / np.sum(weights**2))


def baseline_covariances(view, config, half_life=63):
    """Direct and maturity-pooled forecasts using only the masked level panel."""
    if half_life is not None and (not np.isfinite(half_life) or half_life <= 0):
        raise ValueError("half_life must be positive or None for uniform weights")
    delta_next = int((view.next_date - view.origin).astype(int)) / 365.25
    direct_values, direct_ages, pooled_values, pooled_ages = [], [], [], []
    records = interval_records(view, config)
    raw_records = interval_records(view, config, eligible=False)
    # Reorder chronological maturity slots into the current target-first frame.
    inverse_order = np.argsort(view.episode.maturity_order)
    for index, (record, raw) in enumerate(zip(records, raw_records, strict=True)):
        ids = record["ids"].tolist()
        raw_ids = raw["ids"].tolist()
        age = len(records) - 1 - index
        if all(x in raw_ids for x in view.evaluation_ids):
            direct_values.append(raw["response"][[raw_ids.index(x) for x in view.evaluation_ids]])
            direct_ages.append(age)
        if len(ids) < 3:
            continue
        slots = [
            int(np.argmin(np.abs(record["tenors_days"] - t))) for t in config.target_tenors_days
        ]
        if len(set(slots)) != 3:
            continue
        pooled_values.append(record["response"][np.asarray(slots)[inverse_order]])
        pooled_ages.append(age)
    short, short_ess = _moment(direct_values, direct_ages, half_life)
    pooled, pool_ess = _moment(pooled_values, pooled_ages, half_life)
    count = len(direct_values)
    diagnostics = {
        "complete_changes": count,
        "pooled_changes": len(pooled_values),
        "short_effective_count": short_ess,
        "pooled_effective_count": pool_ess,
        "half_life": half_life,
        "future_delta": delta_next,
        "target": "next_same_contract_price_change_second_moment",
        "price_unit": "CNY/tonne",
    }
    direct_available = view.available and short is not None and count >= config.min_direct_count
    pooled_available = view.available and pooled is not None
    blend = count / (count + 20.0)
    combined = (
        None
        if pooled is None
        else pooled
        if short is None
        else blend * short + (1 - blend) * pooled
    )
    return _snapshot(
        {
            "direct": {
                "available": direct_available,
                "covariance": delta_next * short if direct_available else None,
                "diagnostics": diagnostics,
                "reason": None
                if direct_available
                else view.reason or "insufficient_complete_same_contract_changes",
            },
            "pooled_shrink": {
                "available": pooled_available,
                "covariance": delta_next * combined if pooled_available else None,
                "diagnostics": {**diagnostics, "short_blend_weight": blend},
                "reason": None if pooled_available else view.reason or "no_complete_maturity_slots",
            },
        }
    )
