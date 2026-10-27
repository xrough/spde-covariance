"""Verify aggregate mechanism evidence and numerical/presentation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spde_covariance.empirical_mechanisms import load_mechanism_evidence


def check(directory, *, verify_current_source=False):
    directory = Path(directory)
    evidence = load_mechanism_evidence(directory)
    manifest = evidence["manifest"]
    assert manifest["schema"] == "copper-mechanisms-v1"
    assert manifest["numerical_source_unchanged"] is True
    config = evidence["protocol"]["config"]["base"]
    assert config["test_end"] < "2026-01-01"
    assert manifest["last_origin"] < "2026-01-01"
    for row in evidence["summary"]["rows"]:
        assert 0 <= row["scored"] <= row["available"] <= row["episodes"]
        if row["method"] == "direct" and row["history_budget"] in ("0", "5"):
            assert row["available"] == 0
    for row in evidence["diagnostics"]["rows"]:
        numeric = row["numeric"]
        for key, tolerance in (
            (
                "source_fit.relative_covariance_error_bound",
                config["numerical_error_budget"] * (1 + 1e-8),
            ),
            ("relative_support_residual", 1e-8),
            ("component_sum_relative_residual", 1e-8),
            ("reconstruction_identity_relative_residual", 1e-8),
        ):
            if key in numeric:
                assert numeric[key]["min_median_max"][-1] <= tolerance, (row["method"], key)
    cache = evidence["diagnostics"]["cache"]
    assert cache["max_entries_observed"] <= cache["max_entries"] <= 64
    assert cache["max_bytes_observed"] <= cache["max_bytes"] <= 64 * 1024**2
    assert evidence["paired"]["joint_resampling"]
    assert evidence["paired"]["one_day_missing_origins_retained"]
    for row in evidence["paired"]["rows"]:
        assert row["paired_origins"] <= row["paired_episodes"]
        assert {b["block_origins"] for b in row["bootstrap"]} == {
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
        "matched_raw_pool_covariance",
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
    root = Path(__file__).resolve().parents[1]
    if verify_current_source:
        for name, digest in manifest["source_sha256"].items():
            assert (
                hashlib.sha256((root / "src/spde_covariance" / name).read_bytes()).hexdigest()
                == digest
            )
    presentation = directory / "report-manifest.json"
    if presentation.exists():
        rendered = json.loads(presentation.read_text())
        assert (
            rendered["numerical_checksums_sha256"]
            == hashlib.sha256((directory / "checksums.json").read_bytes()).hexdigest()
        )
        assert (
            rendered["renderer_sha256"]
            == hashlib.sha256((root / "scripts/render_mechanisms.py").read_bytes()).hexdigest()
        )
        for name, digest in rendered["artifacts_sha256"].items():
            assert Path(name).name == name
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        notebook = json.loads(
            (root / "notebooks/08_copper_mechanism_attribution.ipynb").read_text()
        )
        source = [
            {"cell_type": c["cell_type"], "source": "".join(c["source"])} for c in notebook["cells"]
        ]
        assert (
            rendered["notebook_source_sha256"]
            == hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments/results/copper-mechanisms",
    )
    parser.add_argument("--verify-current-source", action="store_true")
    args = parser.parse_args()
    result = check(args.directory, verify_current_source=args.verify_current_source)
    print("Verified copper mechanism evidence:", result["manifest"]["issued_count"], "forecasts")


if __name__ == "__main__":
    main()
