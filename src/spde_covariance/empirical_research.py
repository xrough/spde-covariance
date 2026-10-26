"""Orchestrate the exploratory futures-risk research suite.

The first historical benchmark remains frozen. New model development uses the
same already-inspected years and is explicitly exploratory, even when every
issued forecast is past-only. Raw inputs and forecast records remain local.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .empirical_data import public_data_provenance
from .empirical_diagnostics import reconstruction_risk_decomposition
from .empirical_study import EXPOSURES, EmpiricalConfig, prepare_daily_curves, select_contracts
from .provenance import _git, _jsonable, _source_root


@dataclass(frozen=True)
class ResearchSuiteConfig:
    name: str = "shfe-cu-research-suite"
    empirical: EmpiricalConfig = field(default_factory=EmpiricalConfig)
    score_blocks: tuple[int, ...] = (5, 10, 20, 40)
    representation_dimensions: tuple[int, ...] = (2, 3, 4)
    representation_smoothing: tuple[float, ...] = (1e-6, 1e-4)

    def __post_init__(self):
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("name must be a single directory name")
        if (
            not self.score_blocks
            or len(set(self.score_blocks)) != len(self.score_blocks)
            or any(
                isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in self.score_blocks
            )
        ):
            raise ValueError("score_blocks must be positive")
        if not self.representation_dimensions or any(
            isinstance(d, bool) or not isinstance(d, int) or d < 1
            for d in self.representation_dimensions
        ):
            raise ValueError("representation dimensions must be positive")
        if not self.representation_smoothing or any(
            not np.isfinite(s) or s <= 0 for s in self.representation_smoothing
        ):
            raise ValueError("representation smoothing must be finite and positive")


def load_research_config(path):
    import tomllib

    values = tomllib.loads(Path(path).read_text())
    empirical = values.pop("empirical", {})
    for key in ("windows", "target_tenors_days"):
        if key in empirical:
            empirical[key] = tuple(empirical[key])
    for key in ("score_blocks", "representation_dimensions", "representation_smoothing"):
        if key in values:
            values[key] = tuple(values[key])
    return ResearchSuiteConfig(empirical=EmpiricalConfig(**empirical), **values)


def data_and_target_diagnostics(panel, config):
    """Fixed-bin quality/coverage summaries; never export individual prices."""
    years, counts = np.unique(
        panel.session_dates.astype("datetime64[Y]").astype(str), return_counts=True
    )
    days = (panel.contract_month_dates - panel.dates).astype(int)
    # The modeling midpoint can precede the last trading session. It is not
    # the contract's actual expiry, so retain these rows in the data audit.
    bins = np.array([-10000, 0, 30, 90, 180, 270, 365, 450, 10000])
    tenor_rows = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (days >= lo) & (days < hi)
        tenor_rows.append(
            {
                "lower_days": int(lo),
                "upper_days": int(hi),
                "rows": int(mask.sum()),
                "positive_volume_rows": int(np.sum(mask & (panel.volume > 0))),
                "positive_open_interest_rows": int(np.sum(mask & (panel.open_interest > 0))),
            }
        )
    daily = prepare_daily_curves(panel, config)
    histories = {window: [] for window in config.windows}
    selected_count = 0
    for i, day in enumerate(daily["dates"][:-1]):
        if day <= np.datetime64(config.validation_end):
            continue
        chosen = select_contracts(daily["sections"][i], day, config.target_tenors_days)
        if chosen is None:
            continue
        selected_count += 1
        # Count actual adjacent observations for the origin's same contracts.
        for window in config.windows:
            complete = 0
            for j in range(max(1, i - window + 1), i + 1):
                before = set(daily["settlement_sections"][j - 1]["ids"].tolist())
                after = set(daily["settlement_sections"][j]["ids"].tolist())
                complete += set(chosen["ids"].tolist()) <= before & after
            histories[window].append(complete)
    return {
        "sessions": len(panel.session_dates),
        "rows": len(panel.dates),
        "distinct_contracts": len(np.unique(panel.contract_ids)),
        "sessions_by_year": dict(zip(years.tolist(), counts.tolist(), strict=True)),
        "quality": {
            field: {
                "missing": int(np.sum(~np.isfinite(getattr(panel, field)))),
                "nonpositive": int(np.sum(getattr(panel, field) <= 0)),
            }
            for field in ("settlement", "volume", "open_interest")
        },
        "tenor_coverage": tenor_rows,
        "outside_tenor_bins_count": int(np.sum((days < bins[0]) | (days >= bins[-1]))),
        "selected_origins": selected_count,
        "same_contract_history": [
            {
                "window": w,
                "count": len(v),
                "joint_count_quantiles": np.quantile(v, [0, 0.25, 0.5, 0.75, 1]).tolist()
                if v
                else [],
                "sufficient_count": int(np.sum(np.asarray(v) >= config.min_direct_count)),
            }
            for w, v in histories.items()
        ],
        "units": {"price": "CNY/tonne", "pnl": "CNY", "tonnes_per_contract": config.multiplier},
        "maturity_convention": "15th of contract month; coordinate, not expiry/delivery average",
        "vintage_scope": "retrospective annual archives; original release vintages not reconstructed",
        "risk_target": "next-session second moment of fixed actual-contract settlement P&L",
    }


def representation_diagnostics(panel, config):
    """Price-fit accuracy and financial representation error remain separate."""
    results = []
    for dimension in config.representation_dimensions:
        for smoothing in config.representation_smoothing:
            daily = prepare_daily_curves(
                panel, config.empirical, dimension=dimension, smoothing=smoothing
            )
            available = [d for d in daily["diagnostics"] if d["available"]]
            results.append(
                {
                    "dimension": dimension,
                    "smoothing": smoothing,
                    "available_sessions": len(available),
                    "sessions": len(daily["dates"]),
                    "rmse_quantiles": np.quantile(
                        [d["rmse_cny_per_tonne"] for d in available], [0, 0.25, 0.5, 0.75, 1]
                    ).tolist()
                    if available
                    else [],
                    "condition_quantiles": np.quantile(
                        [d["design_condition_number"] for d in available], [0, 0.5, 1]
                    ).tolist()
                    if available
                    else [],
                    "decomposition": reconstruction_risk_decomposition(daily, config.empirical),
                }
            )
    return {"scope": "post hoc representation diagnostic, not a forecast input", "fits": results}


def _write(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _package_versions():
    result = {}
    for name in ("numpy", "scipy", "matplotlib", "python-calamine"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _local_output(output):
    root = _source_root()
    if root is not None and output.resolve().is_relative_to(root):
        check = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "check-ignore",
                "-q",
                "--no-index",
                str(output.resolve() / "private_predictions.json"),
            ],
            check=False,
        )
        if check.returncode:
            raise ValueError("research outputs inside the repository must be Git-ignored")
    output.mkdir(parents=True, exist_ok=False)


def _validate_baseline_panel(panel, records, summary, manifest, config):
    """Verify historical archive identity and every scored same-contract proxy."""

    def archives(provenance):
        return sorted(
            (r.get("year"), r.get("sha256"), r.get("bytes")) for r in provenance.get("archives", [])
        )

    left, right = archives(panel.provenance), archives(manifest.get("data_provenance", {}))
    if left != right:
        raise ValueError("baseline data archive identities differ from the supplied panel")
    if not left and not (
        panel.provenance.get("synthetic") is True
        or str(panel.provenance.get("type", "")).startswith("synthetic")
    ):
        raise ValueError("historical research requires archive content provenance")
    overview = summary["data_overview"]
    if (overview["contract_rows"], overview["sessions"], overview["distinct_contracts"]) != (
        len(panel.dates),
        len(panel.session_dates),
        len(np.unique(panel.contract_ids)),
    ):
        raise ValueError("baseline data overview differs from supplied panel")
    dates = panel.session_dates
    indices = {str(day): i for i, day in enumerate(dates)}
    expected = {}
    for record in records:
        key = (record["origin"], record["exposure"])
        if key not in expected:
            i = indices.get(key[0])
            if i is None or i + 1 == len(dates) or key[1] not in EXPOSURES:
                raise ValueError("baseline scored origin/exposure is absent from supplied panel")
            day = dates[i]
            rows = np.flatnonzero(panel.dates == day)
            days = (panel.contract_month_dates[rows] - day).astype(int)
            valid = (
                (days >= config.min_tenor_days)
                & (days <= config.max_tenor_days)
                & (panel.settlement[rows] > 0)
                & (panel.volume[rows] > 0)
                & (panel.open_interest[rows] > 0)
            )
            rows = rows[valid]
            rows = rows[np.argsort(panel.contract_month_dates[rows])]
            chosen = select_contracts(
                {
                    "ids": panel.contract_ids[rows],
                    "maturity": panel.contract_month_dates[rows],
                    "prices": panel.settlement[rows],
                },
                day,
                config.target_tenors_days,
            )
            if chosen is None:
                raise ValueError("baseline contract target is unavailable in supplied panel")
            future_rows = np.flatnonzero(panel.dates == dates[i + 1])
            future = dict(
                zip(panel.contract_ids[future_rows], panel.settlement[future_rows], strict=True)
            )
            a = EXPOSURES[key[1]]
            support = a != 0
            change = (
                np.array([future.get(k, np.nan) for k in chosen["ids"][support]])
                - chosen["prices"][support]
            )
            expected[key] = float((config.multiplier * a[support] @ change) ** 2)
        if not np.isfinite(expected[key]) or not np.isclose(
            record["squared_pnl"], expected[key], rtol=1e-12, atol=1e-8
        ):
            raise ValueError("baseline scored outcomes differ from supplied panel")
    digest = hashlib.sha256()
    for name in (
        "dates",
        "contract_ids",
        "contract_month_dates",
        "settlement",
        "volume",
        "open_interest",
        "session_dates",
    ):
        array = np.asarray(getattr(panel, name))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return {
        "panel_content_sha256": digest.hexdigest(),
        "archive_identity_verified": bool(left),
        "same_contract_proxies_checked": len(records),
        "unique_scored_targets_checked": len(expected),
    }


def run_research_suite(panel, config, output, *, baseline_run):
    """Run every angle from committed numerical source and a frozen baseline.

    ``baseline_run`` is the local output of ``empirical-shfe``. It is not
    public data: only aggregate score diagnostics leave this directory.
    """
    from .empirical_attribution import run_attribution_study
    from .empirical_joint import run_joint_risk_study
    from .empirical_score_analysis import analyze_forecast_scores

    output, baseline = Path(output), Path(baseline_run)
    baseline_summary = json.loads((baseline / "summary.json").read_text())
    expected = _jsonable(asdict(config.empirical))
    actual = baseline_summary["config"]
    # Presentation/run names and the existing sensitivity toggle do not change
    # the shared baseline data chronology or estimation protocol.
    for key in expected.keys() - {"name", "sensitivities"}:
        if actual[key] != expected[key]:
            raise ValueError(f"baseline protocol mismatch: {key}")
    records = json.loads((baseline / "private_predictions.json").read_text())
    baseline_manifest = json.loads((baseline / "manifest.json").read_text())
    baseline_check = _validate_baseline_panel(
        panel, records, baseline_summary, baseline_manifest, config.empirical
    )
    _local_output(output)
    start = time.perf_counter()
    stages = {
        "data": lambda: data_and_target_diagnostics(panel, config.empirical),
        "representation": lambda: representation_diagnostics(panel, config),
        "scores": lambda: analyze_forecast_scores(
            records, config.empirical, blocks=config.score_blocks
        ),
        "attribution": lambda: run_attribution_study(
            panel,
            config.empirical,
            bootstrap_replications=config.empirical.bootstrap_replications,
        ),
        "joint": lambda: run_joint_risk_study(panel, config.empirical),
    }
    runtimes = {}
    for name, operation in stages.items():
        print(f"research stage: {name}", flush=True)
        stage_start = time.perf_counter()
        result = operation()
        runtimes[name] = time.perf_counter() - stage_start
        _write(output / f"{name}.json", result)
    _write(
        output / "protocol.json",
        {
            "schema": "shfe-research-suite-v1",
            "config": asdict(config),
            "status": "exploratory development on inspected 2024–2025 data",
            "selection": "2023 validation, then frozen choices; no future outcomes at issuance",
            "holdout": "no final untouched holdout evaluated; reserve an uninspected period for revised-model confirmation",
            "baseline_summary_sha256": hashlib.sha256(
                (baseline / "summary.json").read_bytes()
            ).hexdigest(),
            "baseline_predictions_sha256": hashlib.sha256(
                (baseline / "private_predictions.json").read_bytes()
            ).hexdigest(),
            "baseline_source_revision": json.loads((baseline / "manifest.json").read_text())[
                "git_revision"
            ],
            "baseline_data_verification": baseline_check,
        },
    )
    root = _source_root()
    _write(
        output / "manifest.json",
        {
            "git_revision": _git(root, "rev-parse", "HEAD"),
            "git_dirty": bool(_git(root, "status", "--porcelain")) if root else None,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
            "lockfile_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
            if root
            else None,
            "config_sha256": hashlib.sha256(
                json.dumps(_jsonable(asdict(config)), sort_keys=True).encode()
            ).hexdigest(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "data_provenance": public_data_provenance(panel.provenance),
            "runtime_seconds": time.perf_counter() - start,
            "stage_runtimes": runtimes,
            "solver": "existing source PSD/ridge numerical budget; joint second moments are PSD by construction",
            "threads": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
    )
    _write(
        output / "checksums.json",
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.json"))},
    )
    return output


def export_research_evidence(local_run, destination):
    """Publish only the generated aggregate products; no broad directory copy."""
    local_run, destination = Path(local_run), Path(destination)
    names = (
        "data.json",
        "representation.json",
        "scores.json",
        "attribution.json",
        "joint.json",
        "protocol.json",
        "manifest.json",
    )
    checks = json.loads((local_run / "checksums.json").read_text())
    for name in names:
        if hashlib.sha256((local_run / name).read_bytes()).hexdigest() != checks[name]:
            raise ValueError(f"evidence checksum mismatch: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        (destination / name).write_bytes((local_run / name).read_bytes())
    _write(destination / "checksums.json", {name: checks[name] for name in names})
    return destination


def load_research_evidence(directory):
    """Read and verify the aggregate evidence used by every research notebook."""
    directory = Path(directory)
    checksums = json.loads((directory / "checksums.json").read_text())
    expected = {
        "data.json",
        "representation.json",
        "scores.json",
        "attribution.json",
        "joint.json",
        "protocol.json",
        "manifest.json",
    }
    if set(checksums) != expected:
        raise ValueError("research evidence must contain exactly the declared aggregate products")
    result = {}
    for name, checksum in checksums.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"research evidence checksum mismatch: {name}")
        result[Path(name).stem] = json.loads(data)
    return result
