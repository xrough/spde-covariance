"""Render an aggregate-only research report from verified numerical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spde_covariance.empirical_research import load_research_evidence

LABELS = {
    "window": "Window (intervals)",
    "count": "Count",
    "common_count": "Common outcomes",
    "joint_count_quantiles": "Joint count: min / Q1 / median / Q3 / max",
    "sufficient_count": "Origins meeting minimum",
    "exposure": "Exposure",
    "loss": "Loss",
    "mean": "Mean difference",
    "bootstrap_standard_error": "Block-bootstrap SE",
    "bootstrap_percentile_95": "95% block-bootstrap percentile range",
    "method": "Policy",
    "reference": "Reference",
    "clock": "Diffusion clock",
    "shrinkage": "Diagonal shrinkage",
    "validation_origins": "Validation origins",
    "last_validation_outcome": "Last validation outcome",
    "qlike": "QLIKE",
    "realized_over_predicted_second_moment": "Realized / predicted second moment",
    "realized_rms_cny": "Realized RMS (CNY)",
    "reduction_vs_unhedged": "Second-moment reduction vs unhedged",
    "mean_position_change_proxy": "Mean position-change proxy",
    "direct": "Direct covariance",
    "factor": "Factor only",
    "joint": "Joint factor + residual",
    "source": "Source-C model",
    "unhedged": "Unhedged",
    "outright_3m": "Outright",
    "spread_3m_6m": "Spread",
    "butterfly_3m_6m_9m": "Butterfly",
}


def _format(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(item) for item in value) + "]"
    return LABELS.get(str(value), str(value))


def _table(rows, columns):
    lines = [
        "| " + " | ".join(LABELS.get(c, c) for c in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(c)) for c in columns) + " |")
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
        "For the research narrative and figures, start with the [current project overview](../../../notebooks/00_project_findings.ipynb) or [short findings report](../../../docs/project-findings.md). The tables below are the aggregate audit appendix; full precision and all sensitivities remain in the linked JSON files.",
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
        "All-node reconstruction, fixed nominal windows. The statistic is adjusted minus unadjusted (A=0) QLIKE on identical paired observations; negative differences favor transport adjustment. This does not compare separately tuned pipelines.",
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
        "Full mask/short-history, available-method and numerical diagnostics are in [attribution.json](attribution.json). At 16 intervals the unchanged 20-pair direct minimum makes raw baselines unavailable; factor comparisons remain available.",
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
        "One near contract and two hedge positions in [−2, 2], chosen from past information. Common evaluation origins, actual fixed-contract next-session settlement P&L. Position changes are maturity-slot instability proxies, excluding contract-switch, roll and execution costs.",
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
        "Paired mean squared-P&L differences (CNY²), listed policy minus direct covariance; negative differences favor the listed policy. These are differences in second moments, not differences between the RMS values above.",
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
        f"Measured suite runtime excluding archive parsing: {e['manifest']['runtime_seconds']:.2f} seconds on {e['manifest']['platform']}. This is a single environment measurement, not a hardware-independent performance claim.",
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
