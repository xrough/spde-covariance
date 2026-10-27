"""Audit local records for the full published short-history profile.

Usage: python scripts/audit_short_history_local.py <local-run-directory>
Strict counts/availability checks apply to the fixed 2022–2025 full study,
not arbitrary smoke configurations. Only aggregate diagnostics are printed.
"""

import collections
import hashlib
import json
import pathlib
import sys

import numpy as np

from spde_covariance.empirical_short_history import load_short_history_evidence

path = pathlib.Path(sys.argv[1])
evidence = load_short_history_evidence(path)
manifest = evidence["manifest"]
source = pathlib.Path(__file__).resolve().parents[1] / "src/spde_covariance"
assert manifest["source_dirty"] is False
assert all(
    hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["source_sha256"].items()
)
assert (
    hashlib.sha256((path / "private_predictions.jsonl").read_bytes()).hexdigest()
    == manifest["local_records_sha256"]
)
counts = collections.Counter()
unavailable = collections.Counter()
n = 0
max_certificate = 0.0
max_trace_ratio = 0.0
min_relative_eigenvalue = 0.0
max_support = 0.0
max_component_error = 0.0
minimum_gap = float("inf")
with (path / "private_predictions.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        n += 1
        counts[(row["phase"], row["method"], row["budget"], row["half_life"])] += 1
        assert row["diagnostics"]["training_session_end"] == row["origin"]
        assert row["next_date"] > row["origin"]
        assert row["diagnostics"]["training_calendar_intervals"] <= 252
        if not row["scored"]:
            assert row["method"] == "direct" and row["budget"] in ("0", "5")
            assert row["reason"] == "insufficient_complete_same_contract_changes"
            unavailable[(row["method"], row["budget"])] += 1
            continue
        assert row["available"]
        matrix = np.asarray(row["private"]["forecast"])
        scale = max(np.linalg.norm(matrix, 2), np.finfo(float).tiny)
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert eigenvalues[0] >= -1e-10 * scale
        min_relative_eigenvalue = min(min_relative_eigenvalue, eigenvalues[0] / scale)
        positions = np.asarray(row["private"]["hedge_positions"])
        assert positions[0] == 1 and np.all(np.abs(positions[1:]) <= 2 + 1e-12)
        for name in ("outright", "spread", "butterfly"):
            scores = row["metrics"][name]
            if scores["components"]:
                discrepancy = abs(sum(scores["components"].values()) - scores["forecast"]) / max(
                    scores["forecast"], 1.0
                )
                max_component_error = max(max_component_error, discrepancy)
                assert discrepancy < 1e-8
            assert scores["forecast"] >= 0 and scores["realized"] >= 0
        diagnostics = row["diagnostics"]
        fit = diagnostics.get("source_fit")
        if fit:
            assert fit["converged"]
            certificate = fit["relative_covariance_error_bound"]
            assert certificate <= 1e-5 * (1 + 1e-10)
            max_certificate = max(max_certificate, certificate)
            ratio = fit["source_trace"] / max(fit["trace_cap"], np.finfo(float).tiny)
            max_trace_ratio = max(max_trace_ratio, ratio)
            assert ratio <= 1 + 1e-10
        if "relative_support_residual" in diagnostics:
            max_support = max(max_support, diagnostics["relative_support_residual"])
            assert diagnostics["relative_support_residual"] <= 1e-8
        if "third_fourth_relative_eigengap" in diagnostics:
            minimum_gap = min(minimum_gap, diagnostics["third_fourth_relative_eigengap"])
assert n == manifest["issued_count"]
assert n == manifest["episode_count"] * 4 * 22
print(
    json.dumps(
        {
            "rows": n,
            "episodes": manifest["episode_count"],
            "unavailable": {str(k): v for k, v in unavailable.items()},
            "max_source_relative_error_certificate": max_certificate,
            "max_source_trace_cap_ratio": max_trace_ratio,
            "min_forecast_relative_eigenvalue": min_relative_eigenvalue,
            "max_schur_support_residual": max_support,
            "max_exposure_component_relative_residual": max_component_error,
            "minimum_retained_boundary_eigengap": minimum_gap,
        },
        indent=2,
    )
)
