"""Audit public short-history aggregates without exchange data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spde_covariance.empirical_short_history import load_short_history_evidence


def check(directory, *, verify_current_source=False):
    directory = Path(directory)
    evidence = load_short_history_evidence(directory)
    manifest = evidence["manifest"]
    assert manifest["schema"] == "short-history-v1"
    assert manifest["source_dirty"] is False
    config = evidence["protocol"]["config"]
    for row in evidence["summary"]["rows"]:
        assert 0 <= row["scored"] <= row["available"] <= row["episodes"]
        if row["method"] == "direct" and row["budget"] in ("0", "5"):
            assert row["available"] == 0
    for row in evidence["diagnostics"]["methods"]:
        numeric = row["numeric_min_median_max"]
        bound = numeric.get("source_relative_covariance_error_bound")
        if bound:
            assert bound[-1] <= config["numerical_error_budget"] * (1 + 1e-8)
        support = numeric.get("relative_support_residual")
        if support:
            assert support[-1] <= 1e-8
        component = numeric.get("component_sum_relative_residual")
        if component:
            assert component[-1] <= 1e-8
    cache = evidence["diagnostics"]["cache"]
    assert cache.get("max_bytes_observed", cache.get("bytes", 0)) <= config["cache_bytes"]
    assert cache.get("max_entries_observed", cache.get("entries", 0)) <= config["cache_entries"]
    assert evidence["paired"]["joint_resampling"] is True
    for row in evidence["paired"]["rows"]:
        assert row["paired_origins"] <= row["paired_episodes"]
        assert {r["block_origins"] for r in row["bootstrap"]} == {
            config["bootstrap_block"],
            *config["bootstrap_sensitivity_blocks"],
        }
    forbidden = {
        "contract_ids",
        "evaluation_ids",
        "origin_prices",
        "pnl",
        "signed_pnl",
        "private",
        "settlement",
        "origin",
        "next_date",
        "hedge_positions",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(evidence)
    if verify_current_source:
        root = Path(__file__).resolve().parents[1]
        for name, digest in manifest["source_sha256"].items():
            assert (
                hashlib.sha256((root / "src/spde_covariance" / name).read_bytes()).hexdigest()
                == digest
            ), name
        assert (
            hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
            == manifest["uv_lock_sha256"]
        )
    return {
        "checksums": "passed",
        "information_boundary": "aggregate fields only",
        "numerical_diagnostics": "passed",
        "issued_count": manifest["issued_count"],
        "code_revision": manifest["code_revision"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments/results/short-history",
    )
    parser.add_argument("--verify-current-source", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            check(args.directory, verify_current_source=args.verify_current_source), indent=2
        )
    )


if __name__ == "__main__":
    main()
