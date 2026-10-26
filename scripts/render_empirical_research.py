"""Render an aggregate-only research report from verified numerical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spde_covariance.empirical_research import load_research_evidence


def _table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{row.get(c):.6g}" if isinstance(row.get(c), float) else str(row.get(c))
                for c in columns
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    e = load_research_evidence(args.directory)
    joint = e["joint"]
    lines = [
        "# SHFE copper: exploratory research suite",
        "",
        "This report is generated from checksummed aggregate evidence. It supplements the frozen first benchmark; all new analyses on inspected 2024–2025 data are exploratory development. No untouched confirmation period has been evaluated.",
        "",
        f"Numerical source: `{e['manifest']['git_revision']}`; clean at execution: `{not e['manifest']['git_dirty']}`.",
        "",
        "## Data and effective history",
        "",
        f"{e['data']['sessions']} sessions, {e['data']['rows']} rows, {e['data']['distinct_contracts']} actual contracts. Midmonth coordinates are not actual expiry dates; the negative-coordinate bin is retained. Archive identities and every frozen benchmark scored proxy were checked against the supplied panel.",
        "",
        _table(
            e["data"]["same_contract_history"],
            ["window", "count", "joint_count_quantiles", "sufficient_count"],
        ),
        "",
        "## Frozen benchmark: both losses",
        "",
        "Negative paired differences favor local SPDE versus direct. QLIKE remains primary. Uncertainty is a circular block bootstrap of scored sessions, not IID Monte Carlo precision. Quadratic values have units CNY⁴.",
        "",
        _table(
            [
                r
                for r in e["scores"]["paired_comparisons"]
                if r["method"] == "spde_local" and r["stratum"] == "pooled"
            ],
            [
                "exposure",
                "loss",
                "count",
                "mean",
                "bootstrap_standard_error",
                "bootstrap_percentile_95",
            ],
        ),
        "",
        "## Matched transport ablation",
        "",
        "All-node reconstruction, fixed nominal windows. The table isolates adjusted versus A=0 factor forecasts on identical paired observations; it does not compare separately tuned pipelines.",
        "",
        _table(
            [
                r
                for r in e["attribution"]["paired_comparisons"]
                if r["mask"] == "all"
                and r["reference"] == "factor_unadjusted"
                and r["score"] == "qlike"
                and r["year"] == "all"
            ],
            ["window", "exposure", "count", "mean", "bootstrap_percentile_95"],
        ),
        "",
        "Full mask/short-history, available-method and numerical diagnostics are in attribution.json. At16 intervals the unchanged20-pair direct minimum makes raw baselines unavailable; factor comparisons remain available.",
        "",
        "## Revised joint-risk policies",
        "",
        joint["selection_criterion"],
        "",
        _table(
            [
                {
                    "method": k,
                    **v["selected"],
                    "validation_origins": v["common_validation_origins"],
                    "last_validation_outcome": v["selection_last_outcome"],
                }
                for k, v in joint["selection"].items()
            ],
            [
                "method",
                "window",
                "clock",
                "shrinkage",
                "validation_origins",
                "last_validation_outcome",
            ],
        ),
        "",
        "Joint factor/residual second moments retain cross terms and PSD feasibility. Pooling normalized innovations across unequal calendar intervals is an empirical local approximation, distinct from the original interval-integrated source-C model. Direct/factor/joint share historical complete cases at a fixed window. Source uses its own full coefficient window.",
        "",
        _table(
            [
                r
                for r in joint["risk_summary"]
                if r["method"] in ("direct", "factor", "joint", "source")
            ],
            [
                "exposure",
                "method",
                "common_count",
                "qlike",
                "realized_over_predicted_second_moment",
            ],
        ),
        "",
        "## Bounded hedge decisions",
        "",
        "One near contract and two hedge positions in[−2,2], chosen from past information. Common evaluation origins, actual fixed-contract next-session settlement P&L. Position changes are maturity-slot instability proxies, excluding contract-switch, roll and execution costs.",
        "",
        _table(
            [
                r
                for r in joint["hedge_summary"]
                if r["method"] in ("unhedged", "direct", "factor", "joint", "source")
            ],
            [
                "method",
                "common_count",
                "realized_rms_cny",
                "reduction_vs_unhedged",
                "mean_position_change_proxy",
            ],
        ),
        "",
        _table(
            [
                r
                for r in joint["hedge_paired"]
                if r["method"] in ("factor", "joint", "source") and r["reference"] == "direct"
            ],
            ["method", "reference", "count", "mean", "bootstrap_percentile_95"],
        ),
        "",
        "## Interpretation and limitations",
        "",
        "Residual risk explains a major weakness of smooth-factor forecasts. Repairing it does not establish superiority over direct risk estimation. Calendar/session clocks have different effects across exposures; selection remains fixed on validation data. Price reconstruction, numerical convergence and economic risk adequacy are separate gates.",
        "",
        "See the six executed notebooks and docs/empirical-research-protocol.md for mathematical targets, coverage, calibration, influence, clock-gap and rank diagnostics. The original benchmark is unchanged. No physical forcing identification, infinite-dimensional statistical guarantee, trading profitability or execution-aware backtest is claimed.",
        "",
        f"Measured suite runtime excluding archive parsing: {e['manifest']['runtime_seconds']:.2f}seconds on {e['manifest']['platform']}. This is a single environment measurement, not a hardware-independent performance claim.",
        "",
    ]
    report = args.directory / "report.md"
    report.write_text("\n".join(lines))
    (args.directory / "report-manifest.json").write_text(
        json.dumps(
            {
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "numerical_checksums_sha256": hashlib.sha256(
                    (args.directory / "checksums.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
