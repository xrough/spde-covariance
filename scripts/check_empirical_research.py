"""Audit the published empirical research bundle without exchange files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from spde_covariance.empirical_research import load_research_evidence


def check(directory, *, verify_current_source=False):
    directory = Path(directory)
    root = directory.resolve().parents[2]
    e = load_research_evidence(directory)
    baseline_path = root / "experiments/results/shfe-cu/summary.json"
    assert (
        hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        == e["protocol"]["baseline_summary_sha256"]
    )
    baseline = json.loads(baseline_path.read_text())
    assert e["manifest"]["git_dirty"] is False
    assert e["protocol"]["baseline_data_verification"]["same_contract_proxies_checked"] > 0
    assert (
        e["data"]["rows"]
        == sum(r["rows"] for r in e["data"]["tenor_coverage"])
        + e["data"]["outside_tenor_bins_count"]
    )
    for row in e["attribution"]["numerical_diagnostics"]:
        assert row["successful_factor_fits"] > 0
        assert row["relative_covariance_error_bound_max"] <= 1e-5 * (1 + 1e-10)
        assert row["trace_cap_violation_max"] <= 1e-8
    for row in e["joint"]["numerical_diagnostics"].values():
        assert row["available"]
        assert row["all_forecast_psd_within_roundoff"]
    for item in e["joint"]["selection"].values():
        assert item["selection_last_outcome"] <= e["joint"]["chronology"]["validation_end"]
    # An independently routed source-C control reproduces the frozen policy.
    for row in e["joint"]["risk_summary"]:
        assert row["common_count"] == baseline["test_origin_count"]
        if row["method"] == "source":
            original = next(
                r
                for r in baseline["risk_summary"]
                if r["method"] == "spde_local"
                and r["year"] == "all"
                and r["exposure"] == row["exposure"]
            )
            assert np.isclose(row["qlike"], original["qlike"], rtol=1e-12, atol=1e-12)
            assert np.isclose(
                row["realized_over_predicted_second_moment"],
                original["risk_ratio"],
                rtol=1e-12,
                atol=1e-12,
            )
    forbidden = {
        "contract_ids",
        "origin_prices",
        "signed_pnl",
        "squared_pnl",
        "raw_prices",
        "coefficients",
        "origin_index",
        "origin",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden & value.keys()
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(e)
    rendered = json.loads((directory / "report-manifest.json").read_text())
    assert (
        hashlib.sha256((directory / "report.md").read_bytes()).hexdigest()
        == rendered["report_sha256"]
    )
    assert (
        hashlib.sha256((directory / "checksums.json").read_bytes()).hexdigest()
        == rendered["numerical_checksums_sha256"]
    )
    assert (
        hashlib.sha256((root / "scripts/render_empirical_research.py").read_bytes()).hexdigest()
        == rendered["renderer_sha256"]
    )
    if verify_current_source:
        for name, digest in e["manifest"]["source_sha256"].items():
            assert (
                hashlib.sha256((root / "src/spde_covariance" / name).read_bytes()).hexdigest()
                == digest
            ), name
    return {
        "aggregate_checksums": "passed",
        "baseline_source_parity": "passed",
        "chronology": "passed",
        "numerical_feasibility": "passed",
        "raw_field_boundary": "passed",
        "current_source_hashes": verify_current_source,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "experiments/results/shfe-research",
    )
    parser.add_argument("--verify-current-source", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            check(arguments.directory, verify_current_source=arguments.verify_current_source),
            indent=2,
        )
    )
