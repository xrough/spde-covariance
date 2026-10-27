"""Matched mechanism contrasts on masked copper histories.

Only masked data are sent to issuers/process workers. Outcomes are accepted by
the existing future-only scorer afterwards. Public evidence contains aggregates;
the contract-level forecast stream is restricted to an ignored local directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import tomllib
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .empirical_data import ShfeCuPanel, public_data_provenance
from .empirical_mechanism_models import mechanism_covariances
from .empirical_research import _local_output, _package_versions
from .empirical_short_history import (
    IssuedShortHistoryRisk,
    _hash,
    _masked_wire,
    _origin_mean,
    _write,
    aggregate_short_history,
    score_short_history_risk,
)
from .empirical_short_history_data import (
    ShortHistoryConfig,
    ShortHistoryEpisode,
    ShortHistoryView,
    baseline_covariances,
    build_short_history_view,
    make_short_history_episodes,
)
from .empirical_short_history_models import BoundedModelCache
from .provenance import _git, _jsonable, _source_root

EVIDENCE_FILES = (
    "summary.json",
    "paired.json",
    "diagnostics.json",
    "protocol.json",
    "manifest.json",
    "oracle.json",
)


@dataclass(frozen=True, slots=True)
class MechanismConfig:
    """Frozen diagnostic protocol; no model/parameter selection from outcomes."""

    base: ShortHistoryConfig = ShortHistoryConfig(
        name="copper-mechanisms-2016-2025",
        training_end="2015-12-31",
        validation_end="2016-12-31",
        sensitivity_half_lives=(),
    )
    dimensions: tuple[int, ...] = (2, 3, 4)
    interval_policies: tuple[str, ...] = ("all", "one_day")
    oracle_replications: int = 20

    def __post_init__(self):
        if not isinstance(self.base, ShortHistoryConfig):
            raise TypeError("base must be ShortHistoryConfig")
        if (
            not isinstance(self.dimensions, tuple)
            or not self.dimensions
            or any(type(d) is not int or d not in (2, 3, 4) for d in self.dimensions)
            or len(set(self.dimensions)) != len(self.dimensions)
        ):
            raise ValueError("dimensions must be distinct members of (2, 3, 4)")
        if (
            not isinstance(self.interval_policies, tuple)
            or not self.interval_policies
            or any(p not in ("all", "one_day") for p in self.interval_policies)
            or len(set(self.interval_policies)) != len(self.interval_policies)
        ):
            raise ValueError("interval_policies must contain distinct all/one_day values")
        if np.datetime64(self.base.test_end) >= np.datetime64("2026-01-01"):
            raise ValueError("2026 is held out")
        if self.base.cache_entries > 64 or self.base.cache_bytes > 64 * 1024**2:
            raise ValueError("mechanism caches are bounded at 64 entries and 64 MiB")
        if type(self.oracle_replications) is not int or self.oracle_replications < 2:
            raise ValueError("oracle_replications must be an integer >=2")


def load_mechanism_config(path):
    values = tomllib.loads(Path(path).read_text())
    base = values.pop("base", {})
    for key in (
        "history_budgets",
        "target_tenors_days",
        "sensitivity_half_lives",
        "bootstrap_sensitivity_blocks",
        "hedge_bounds",
    ):
        if key in base:
            base[key] = tuple(None if v == "all" else v for v in base[key])
    for key in ("dimensions", "interval_policies"):
        if key in values:
            values[key] = tuple(values[key])
    if base:
        defaults = asdict(MechanismConfig().base)
        defaults.update(base)
        values["base"] = ShortHistoryConfig(**defaults)
    return MechanismConfig(**values)


def development_archive_paths(recent_directory, older_directory):
    """Exact allowlist, without traversing or opening the 2026 reserve."""
    return tuple(
        Path(older_directory if year <= 2021 else recent_directory) / f"shfe-futures-{year}.zip"
        for year in range(2016, 2026)
    )


def _name(context, dimension, method):
    return f"{context}:d{dimension}:{method}"


def _decode(name):
    context, dimension, method = name.split(":")
    return {"context": context, "dimension": int(dimension[1:]), "method": method}


def issue_mechanism_risks(view, config, *, cache=None):
    """Issue frozen forecasts; accepts no original panel, truth or outcome."""
    base = config.base
    payloads = {
        _name("all", 0, name): payload
        for name, payload in baseline_covariances(view, base, base.half_life).items()
    }
    next_days = int((view.next_date - view.origin).astype(int))
    for dimension in config.dimensions:
        for context in config.interval_policies:
            if context == "one_day" and next_days != 1:
                continue
            fitted = mechanism_covariances(
                view, base, dimension=dimension, interval_policy=context, cache=cache
            )
            for method, payload in fitted.items():
                if context == "one_day" and method == "reconstructed_pool":
                    continue
                payloads[_name(context, dimension, method)] = payload
    butterfly = np.zeros(3)
    butterfly[np.asarray(view.episode.maturity_order)] = [1, -2, 1]
    exposures = {
        "outright": np.array([1.0, 0, 0]),
        "spread": np.array([1.0, -1, 0]),
        "butterfly": butterfly,
    }
    issued = {}
    for method, payload in payloads.items():
        covariance = payload["covariance"]
        if payload["available"]:
            eigenvalues = np.linalg.eigvalsh(covariance)
            if np.shape(covariance) != (3, 3) or not np.all(np.isfinite(covariance)):
                raise ValueError("invalid forecast matrix")
            if eigenvalues[0] < -1e-10 * max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny):
                raise ValueError("materially indefinite forecast")
        issued[method] = IssuedShortHistoryRisk(
            method,
            str(view.origin),
            str(view.next_date),
            "evaluation",
            "all" if view.history_budget is None else str(view.history_budget),
            view.target_tenor_days,
            view.evaluation_ids,
            view.origin_prices,
            exposures,
            covariance,
            payload.get("components", {}),
            payload["available"],
            payload["reason"],
            {
                **payload["diagnostics"],
                "information_fingerprint": view.fingerprint,
                "training_session_start": str(view.dates[0]),
                "training_session_end": str(view.dates[-1]),
                "training_calendar_intervals": len(view.dates) - 1,
            },
            base.half_life,
            base.multiplier,
            base.pnl_variance_floor,
            base.hedge_bounds,
        )
    return issued


def aggregate_mechanisms(records, config):
    """Equal targets per origin, then equal origins; years are descriptive strata."""
    expanded = []
    for row in records:
        expanded.extend(({**row, "phase": "all"}, {**row, "phase": row["origin"][:4]}))
    result = aggregate_short_history(expanded, config.base)
    for row in result["rows"]:
        row.update(_decode(row["method"]))
        row["year"] = row.pop("phase")
        row["history_budget"] = row.pop("budget")
        row["target_tenor_days"] = row.pop("tenor")
    return result


def _contrasts(config):
    """Fixed contrast vectors, including source/residual interactions."""
    output = []
    for context in config.interval_policies:
        for d in config.dimensions:

            def cell(name, c=context, dimension=d):
                return _name(c, dimension, name)

            for column in ("factor", "correlated", "diagonal"):
                output.append(
                    (
                        "source_policy",
                        context,
                        d,
                        column,
                        {cell("source_" + column): 1, cell("empirical_" + column): -1},
                    )
                )
            for source in ("source", "empirical"):
                for high, low in (("correlated", "factor"), ("diagonal", "correlated")):
                    output.append(
                        (
                            "residual_" + high,
                            context,
                            d,
                            source,
                            {cell(source + "_" + high): 1, cell(source + "_" + low): -1},
                        )
                    )
            for high, low in (("correlated", "factor"), ("diagonal", "correlated")):
                output.append(
                    (
                        "interaction_" + high,
                        context,
                        d,
                        "source_minus_empirical",
                        {
                            cell("source_" + high): 1,
                            cell("source_" + low): -1,
                            cell("empirical_" + high): -1,
                            cell("empirical_" + low): 1,
                        },
                    )
                )
            if context == "all":
                output.append(
                    (
                        "reconstruction_pool",
                        context,
                        d,
                        "fixed_selectors",
                        {cell("reconstructed_pool"): 1, _name("all", 0, "pooled_shrink"): -1},
                    )
                )
            if d != 3 and 3 in config.dimensions:
                for method in ("source_diagonal", "empirical_diagonal"):
                    output.append(
                        (
                            "dimension",
                            context,
                            d,
                            method,
                            {cell(method): 1, _name(context, 3, method): -1},
                        )
                    )
    return output


def paired_mechanisms(records, config):
    """Shared scheduled-origin circular block draws, with common-pair support."""
    base = config.base
    origins = sorted({r["origin"] for r in records})
    n = len(origins)
    if not n:
        return {"rows": [], "scheduled_origins": 0}
    position = {date: i for i, date in enumerate(origins)}
    cells = defaultdict(dict)
    for r in records:
        cells[r["budget"]][(r["origin"], r["tenor"], r["method"])] = r
    rows, vectors = [], []
    years = ("all", *sorted({d[:4] for d in origins}))
    for budget, lookup in sorted(cells.items()):
        episodes = sorted({(date, tenor) for date, tenor, _ in lookup})
        for kind, context, d, policy, contrast in _contrasts(config):
            for exposure, metric in (
                ("outright", "qlike"),
                ("spread", "qlike"),
                ("butterfly", "qlike"),
                ("joint", "quadratic"),
                ("hedge", "realized"),
            ):
                differences = []
                for date, tenor in episodes:
                    paired = [lookup.get((date, tenor, method)) for method in contrast]
                    if all(r is not None and r["scored"] for r in paired):
                        differences.append(
                            (
                                date,
                                tenor,
                                sum(
                                    weight * r["metrics"][exposure][metric]
                                    for weight, r in zip(contrast.values(), paired, strict=True)
                                ),
                            )
                        )
                for year in years:
                    selected = [
                        (date, value)
                        for date, _, value in differences
                        if year == "all" or date[:4] == year
                    ]
                    by_origin = _origin_mean(selected)
                    if not by_origin:
                        continue
                    vector = np.full(n, np.nan)
                    for date, value in by_origin.items():
                        vector[position[date]] = value
                    rows.append(
                        {
                            "contrast": kind,
                            "context": context,
                            "dimension": d,
                            "policy": policy,
                            "history_budget": budget,
                            "year": year,
                            "target_tenor_days": "all",
                            "exposure": exposure,
                            "metric": metric,
                            "terms": contrast,
                            "paired_episodes": len(selected),
                            "paired_origins": len(by_origin),
                            "difference": float(np.mean(list(by_origin.values()))),
                            "bootstrap": [],
                        }
                    )
                    vectors.append(vector)
    matrix = np.asarray(vectors).T
    finite = np.isfinite(matrix).astype(float)
    values = np.nan_to_num(matrix)
    for block in dict.fromkeys((base.bootstrap_block, *base.bootstrap_sensitivity_blocks)):
        rng = np.random.default_rng(np.random.SeedSequence([base.seed, block, 71]))
        starts = rng.integers(0, n, size=(base.bootstrap_replications, int(np.ceil(n / block))))
        indices = ((starts[..., None] + np.arange(block)) % n).reshape(
            base.bootstrap_replications, -1
        )[:, :n]
        counts = np.zeros((base.bootstrap_replications, n))
        for i, draw in enumerate(indices):
            counts[i] = np.bincount(draw, minlength=n)
        for start in range(0, len(rows), 128):
            denominator = counts @ finite[:, start : start + 128]
            numerator = counts @ values[:, start : start + 128]
            sampled = np.divide(
                numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
            )
            for j in range(sampled.shape[1]):
                v = sampled[:, j]
                v = v[np.isfinite(v)]
                rows[start + j]["bootstrap"].append(
                    {
                        "block_origins": block,
                        "valid_draws": len(v),
                        "bootstrap_se": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                        "percentile_95": np.quantile(v, [0.025, 0.975]).tolist() if len(v) else [],
                    }
                )
    return {
        "rows": rows,
        "scheduled_origins": n,
        "replications": base.bootstrap_replications,
        "sign": "weighted score contrast; negative favors positive-weight policy",
        "scope": "descriptive calendar-origin blocks; common paired episodes, targets averaged first",
        "joint_resampling": True,
        "one_day_missing_origins_retained": True,
    }


def _numeric_diagnostics(value, prefix=""):
    """Scalar aggregate audit only; excludes individual covariance/weight arrays."""
    output = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            output.update(_numeric_diagnostics(item, name + "."))
        elif isinstance(item, (float, int, np.floating, np.integer)) and np.isfinite(item):
            output[name] = float(item)
    return output


def diagnostic_mechanisms(records, timings, cache_info):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["budget"])].append(row)
    output = []
    for (method, budget), rows in sorted(grouped.items()):
        numeric = defaultdict(list)
        for r in rows:
            for k, v in r["diagnostics"].items():
                numeric[k].append(v)
        output.append(
            {
                **_decode(method),
                "history_budget": budget,
                "issued": len(rows),
                "scored": sum(r["scored"] for r in rows),
                "failures": dict(Counter(r["reason"] for r in rows if not r["scored"])),
                "numeric": {
                    k: {
                        "count": len(v),
                        "mean": float(np.mean(v)),
                        "min_median_max": np.quantile(v, [0, 0.5, 1]).tolist(),
                    }
                    for k, v in numeric.items()
                },
            }
        )
    return {
        "rows": output,
        "cache": cache_info,
        "masked_view_seconds_min_median_max": np.quantile(timings, [0, 0.5, 1]).tolist()
        if timings
        else [],
        "scope": "scalar diagnostics aggregated across masked episodes; not latent truth",
    }


_CACHE = None


def _issue_job(job):
    global _CACHE
    wire, config = job
    if _CACHE is None:
        _CACHE = BoundedModelCache(config.base.cache_entries, config.base.cache_bytes)
    view = ShortHistoryView(
        panel=ShfeCuPanel(**wire["panel"]),
        episode=ShortHistoryEpisode(**wire["episode"]),
        **{k: v for k, v in wire.items() if k not in ("panel", "episode")},
    )
    start = time.perf_counter()
    results = issue_mechanism_risks(view, config, cache=_CACHE)
    return (
        [_jsonable({f.name: getattr(r, f.name) for f in fields(r)}) for r in results.values()],
        time.perf_counter() - start,
        _CACHE.diagnostics,
    )


def _parallel(jobs, workers):
    import multiprocessing

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        iterator, queue = iter(jobs), deque()
        for _ in range(2 * workers):
            job = next(iterator, None)
            if job is not None:
                queue.append(pool.submit(_issue_job, job))
        while queue:
            yield queue.popleft().result()
            job = next(iterator, None)
            if job is not None:
                queue.append(pool.submit(_issue_job, job))


def run_mechanism_benchmark(panel, config, output, *, workers=1, importer_audit=None):
    """Run 2016warmup/2017–25 diagnostic evaluation, never a 2026 price row."""
    global _CACHE
    from .empirical_import_audit import public_import_audit
    from .mechanism_oracle import run_oracle_controls

    if isinstance(config, (str, Path)):
        config = load_mechanism_config(config)
    if not isinstance(config, MechanismConfig):
        raise TypeError("expected MechanismConfig")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer 1–8")
    if np.any(panel.dates >= np.datetime64("2026-01-01")) or np.any(
        panel.session_dates >= np.datetime64("2026-01-01")
    ):
        raise ValueError("2026 data must not enter the development panel")
    output = Path(output)
    _local_output(output)
    start = time.perf_counter()
    root = _source_root()
    sources = sorted((root / "src/spde_covariance").glob("*.py")) if root else []
    source_provenance = {
        "code_revision": _git(root, "rev-parse", "HEAD") if root else None,
        "source_dirty": bool(_git(root, "status", "--porcelain")) if root else None,
        "source_sha256": {p.name: _hash(p) for p in sources},
    }
    _CACHE = None
    base = config.base
    episodes = make_short_history_episodes(panel, base, "evaluation")
    if not episodes:
        raise ValueError("no eligible development episodes")
    total = len(episodes) * len(base.history_budgets)
    jobs = (
        (_masked_wire(build_short_history_view(panel, episode, budget, base)), config)
        for episode in episodes
        for budget in base.history_budgets
    )
    iterator = _parallel(jobs, workers) if workers > 1 else map(_issue_job, jobs)
    records, timings, caches = [], [], []
    with (output / "private_predictions.jsonl").open("w") as stream:
        for index, (forecasts, duration, cache) in enumerate(iterator):
            timings.append(duration)
            caches.append(cache)
            for wire in forecasts:
                issued = IssuedShortHistoryRisk(**wire)
                mask = panel.dates == np.datetime64(issued.next_date)
                row = score_short_history_risk(
                    issued,
                    future_date=issued.next_date,
                    future_ids=panel.contract_ids[mask],
                    future_prices=panel.settlement[mask],
                )
                stream.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
                # Keep only small aggregates in RAM; the private stream retains full audit.
                row.pop("private", None)
                row["diagnostics"] = _numeric_diagnostics(row["diagnostics"])
                records.append(row)
            if (index + 1) % 48 == 0 or index + 1 == total:
                print(
                    f"mechanisms: {index + 1}/{total} masked views, {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    cache_info = {
        "scope": "per-worker bounded cache",
        "workers": workers,
        "max_entries_observed": max(c["entries"] for c in caches),
        "max_bytes_observed": max(c["bytes"] for c in caches),
        "max_entries": base.cache_entries,
        "max_bytes": base.cache_bytes,
    }
    products = {
        "summary": aggregate_mechanisms(records, config),
        "paired": paired_mechanisms(records, config),
        "diagnostics": diagnostic_mechanisms(records, timings, cache_info),
        "protocol": mechanism_protocol(config),
        "oracle": run_oracle_controls(replications=config.oracle_replications, seed=base.seed + 1),
    }
    products["manifest"] = {
        "schema": "copper-mechanisms-v1",
        **source_provenance,
        "numerical_source_unchanged": source_provenance["source_sha256"]
        == {p.name: _hash(p) for p in sources},
        "uv_lock_sha256": _hash(root / "uv.lock") if root else None,
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "data_provenance": public_data_provenance(panel.provenance),
        "importer_audit": public_import_audit(importer_audit)
        if importer_audit is not None
        else None,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "threads": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "runtime_seconds": time.perf_counter() - start,
        "workers": workers,
        "episode_count": len(episodes),
        "issued_count": len(records),
        "scheduled_origins": len({str(e.origin) for e in episodes}),
        "first_origin": str(episodes[0].origin),
        "last_origin": str(episodes[-1].origin),
        "local_records_sha256": _hash(output / "private_predictions.jsonl"),
        "local_record_policy": "contract-level prices, forecasts and P&Ls are not exported",
    }
    for name, value in products.items():
        _write(output / f"{name}.json", value)
    _write(output / "checksums.json", {name: _hash(output / name) for name in EVIDENCE_FILES})
    return products


def mechanism_protocol(config):
    return {
        "question": "matched reconstruction, source-estimation policy and residual-correction contrasts",
        "config": asdict(config),
        "chronology": "2016 warmup; 2017–2025 descriptive rolling evaluation; 2026 sealed reserve",
        "selection": "fixed settings, no tuning, no winner selection; one calendar stride anchored once",
        "factorial": "same reconstructed innovations and joint residual blocks; source-C versus direct-H, crossed with factor/correlated/diagonal",
        "source_policy": "changes interval integration, ridge and trace constraints; not a pure covariance estimation error against latent truth",
        "reconstruction": "raw versus reconstructed maturity pool holds observed IDs, selectors, ages, own-history counts and blend fixed",
        "one_day": "training weights zero for non-one-day intervals, no calendar compression; score only next one-day intervals; master origin blocks preserved",
        "all_intervals": "direct H is an EWMA local mixture of normalized interval covariances, not a claimed terminal covariance",
        "units": "ACT/365.25 normalized training, next actual interval, five-tonne CNY P&L moments",
        "counting": "raw archived counts retained; explicit note-based single-sided view; unknown unavailable; no count magnitudes used in fitting",
        "information": "mask target history before every transformation; future only in scoring; 2026 never parsed",
        "interpretation": "policy contrasts with interactions, not unique additive percentages of latent error",
        "limitations": [
            "daily inspected data",
            "maturity proxy is contract-month15th",
            "no latent continuum or source truth",
            "four residual features plus diagonal is a restriction",
            "masked histories are not new listings",
            "not prices, derivative pricing or execution backtesting",
        ],
    }


def load_mechanism_evidence(path):
    path = Path(path)
    checksums = json.loads((path / "checksums.json").read_text())
    if set(checksums) != set(EVIDENCE_FILES):
        raise ValueError("mechanism evidence whitelist mismatch")
    for name, expected in checksums.items():
        if _hash(path / name) != expected:
            raise ValueError(f"mechanism checksum mismatch: {name}")
    return {Path(name).stem: json.loads((path / name).read_text()) for name in EVIDENCE_FILES}


def export_mechanism_evidence(local_run, destination):
    local_run, destination = Path(local_run), Path(destination)
    load_mechanism_evidence(local_run)
    destination.mkdir(parents=True, exist_ok=False)
    for name in (*EVIDENCE_FILES, "checksums.json"):
        (destination / name).write_bytes((local_run / name).read_bytes())
    return destination
