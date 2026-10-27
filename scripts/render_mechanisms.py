"""Render mechanism-attribution evidence without opening market archives."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spde_covariance.empirical_mechanisms import load_mechanism_evidence

ROOT = Path(__file__).resolve().parents[1]
METHOD_LABELS = {
    "source_factor": "Source C · factor",
    "source_correlated": "Source C · correlated",
    "source_diagonal": "Source C · diagonal",
    "empirical_factor": "Direct H · factor",
    "empirical_correlated": "Direct H · correlated",
    "empirical_diagonal": "Direct H · diagonal",
    "reconstructed_pool": "Reconstructed pool",
    "pooled_shrink": "Raw maturity pool",
    "direct": "Same-contract direct",
}
CELLS = (
    "source_factor",
    "source_correlated",
    "source_diagonal",
    "empirical_factor",
    "empirical_correlated",
    "empirical_diagonal",
)


def display(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(display(x) for x in value) + "]"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return METHOD_LABELS.get(str(value), str(value))


def table(rows, columns):
    return "\n".join(
        [
            "| " + " | ".join(label for _, label in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *(
                "| " + " | ".join(display(row.get(key)) for key, _ in columns) + " |"
                for row in rows
            ),
        ]
    )


def base_config(evidence):
    return evidence["protocol"]["config"]["base"]


def budgets(evidence):
    return [
        "all" if value is None else str(value) for value in base_config(evidence)["history_budgets"]
    ]


def select_summary(evidence, *, context="all", dimension=3, year="all", tenor="all"):
    return [
        row
        for row in evidence["summary"]["rows"]
        if row["context"] == context
        and row["dimension"] == dimension
        and row["year"] == str(year)
        and row["target_tenor_days"] == str(tenor)
    ]


def select_pairs(
    evidence, *, contrast, context="all", dimension=3, year="all", exposure="spread", policy=None
):
    return [
        row
        for row in evidence["paired"]["rows"]
        if row["contrast"] == contrast
        and row["context"] == context
        and row["dimension"] == dimension
        and row["year"] == str(year)
        and row["exposure"] == exposure
        and (policy is None or row["policy"] == policy)
    ]


def bounds(row, block):
    values = [item["percentile_95"] for item in row["bootstrap"] if item["block_origins"] == block]
    return values[0] if values and len(values[0]) == 2 else None


def _point(ax, x, row, block, *, color, label=None):
    interval = bounds(row, block)
    if interval is not None:
        ax.plot([x, x], interval, color=color, lw=1.2)
    ax.plot(x, row["difference"], "o", color=color, ms=4, label=label)


def plot_six_cells(evidence):
    history = budgets(evidence)
    shown = [history[0], history[-1]] if len(history) > 1 else history
    fig, axes = plt.subplots(2, len(shown), figsize=(6 * len(shown), 7), squeeze=False)
    rows = select_summary(evidence)
    colors = ("#d95f02", "#1f77b4")
    for column, budget in enumerate(shown):
        lookup = {row["method"]: row for row in rows if row["history_budget"] == budget}
        for row_index, exposure in enumerate(("spread", "butterfly")):
            ax = axes[row_index, column]
            for index, (prefix, color) in enumerate(
                zip(("source", "empirical"), colors, strict=True)
            ):
                values = [
                    lookup.get(prefix + "_" + correction, {})
                    .get("metrics", {})
                    .get(exposure, {})
                    .get("qlike", np.nan)
                    for correction in ("factor", "correlated", "diagonal")
                ]
                ax.plot(
                    range(3),
                    values,
                    "s--" if index == 0 else "o-",
                    color=color,
                    ms=6 if index == 0 else 4,
                    label="Source C" if index == 0 else "Direct H",
                )
            ax.set(
                title=f"{exposure.title()} · own history {budget}",
                xticks=range(3),
                xticklabels=["Factor", "+ correlated", "+ diagonal"],
                ylabel="QLIKE (lower is better)",
            )
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Three-mode representation: covariance policy crossed with residual corrections")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_stage_contrasts(evidence):
    history, block = budgets(evidence), base_config(evidence)["bootstrap_block"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    settings = (
        ("reconstruction_pool", "fixed_selectors", "Reconstructed − raw pool"),
        ("source_policy", "diagonal", "Source C − direct H; full residual"),
        ("residual_diagonal", "source", "Add diagonal to source + correlated"),
    )
    for ax, (contrast, policy, title) in zip(axes, settings, strict=True):
        ax.axhline(0, color=".6", lw=1)
        for j, (dimension, color) in enumerate(
            zip((2, 3, 4), ("#228833", "#1f77b4", "#d95f02"), strict=True)
        ):
            rows = select_pairs(evidence, contrast=contrast, policy=policy, dimension=dimension)
            for k, row in enumerate(rows):
                _point(
                    ax,
                    history.index(row["history_budget"]) + 0.16 * (j - 1),
                    row,
                    block,
                    color=color,
                    label=f"{dimension} modes" if k == 0 else None,
                )
        ax.set(
            title=title,
            xticks=range(len(history)),
            xticklabels=history,
            xlabel="Permitted own changes",
            ylabel="Paired spread QLIKE difference",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Matched stage contrasts: negative favors the added or positive-weight policy")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_calendar_context(evidence):
    years = sorted({row["year"] for row in evidence["paired"]["rows"] if row["year"] != "all"})
    block = base_config(evidence)["bootstrap_block"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, contrast, policy, title in (
        (axes[0], "source_policy", "diagonal", "Source C − direct H, full residual"),
        (axes[1], "residual_diagonal", "source", "Add diagonal to source + correlated"),
    ):
        ax.axhline(0, color=".6", lw=1)
        for j, (context, color, label) in enumerate(
            (
                ("all", "#1f77b4", "All intervals"),
                ("one_day", "#d95f02", "Matched one-day"),
            )
        ):
            for k, year in enumerate(years):
                candidates = [
                    row
                    for row in select_pairs(
                        evidence, contrast=contrast, policy=policy, context=context, year=year
                    )
                    if row["history_budget"] == "0"
                ]
                if candidates:
                    _point(
                        ax,
                        k + 0.09 * (2 * j - 1),
                        candidates[0],
                        block,
                        color=color,
                        label=label if k == 0 else None,
                    )
        ax.set(
            title=title,
            xticks=range(len(years)),
            xticklabels=years,
            ylabel="Paired spread QLIKE difference",
            xlabel="Calendar year; zero own history",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Annual and interval-context checks; fixed three-mode reconstruction")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_calibration(evidence):
    history = budgets(evidence)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    settings = (
        (0, "pooled_shrink", "Raw pool", "#555555"),
        (3, "reconstructed_pool", "Reconstructed pool", "#228833"),
        (3, "source_diagonal", "Source C + full residual", "#d95f02"),
        (3, "empirical_diagonal", "Direct H + full residual", "#1f77b4"),
    )
    for ax, exposure in zip(axes, ("outright", "spread", "butterfly"), strict=True):
        ax.axhline(1, color=".5", lw=1)
        for dimension, method, label, color in settings:
            rows = {
                row["history_budget"]: row
                for row in select_summary(evidence, dimension=dimension)
                if row["method"] == method
            }
            values = [
                rows.get(budget, {})
                .get("metrics", {})
                .get(exposure, {})
                .get("calibration_ratio", np.nan)
                for budget in history
            ]
            ax.plot(
                range(len(history)),
                values,
                "s--" if method == "source_diagonal" else "o-",
                color=color,
                ms=6 if method == "source_diagonal" else 4,
                label=label,
            )
        ax.set(
            title=exposure.title(),
            xticks=range(len(history)),
            xticklabels=history,
            xlabel="Permitted own changes",
            yscale="log",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Forecast / realized second moment (log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Calibration after each policy change; one denotes aggregate calibration")
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    return fig


def policy_table(evidence, *, context="all", dimension=3):
    output = []
    for row in select_summary(evidence, context=context, dimension=dimension):
        metrics = row["metrics"]
        output.append(
            {
                "budget": row["history_budget"],
                "method": row["method"],
                "scored": f"{row['scored']}/{row['episodes']}",
                "origins": row["origins"],
                "outright": metrics.get("outright", {}).get("qlike"),
                "spread": metrics.get("spread", {}).get("qlike"),
                "butterfly": metrics.get("butterfly", {}).get("qlike"),
                "calibration": metrics.get("spread", {}).get("calibration_ratio"),
                "rms": metrics.get("hedge", {}).get("realized_rms_cny"),
                "joint": metrics.get("joint", {}).get("quadratic") / 1e12
                if metrics.get("joint", {}).get("quadratic") is not None
                else None,
            }
        )
    methods = ("direct", "pooled_shrink", *CELLS, "reconstructed_pool")
    return sorted(
        output,
        key=lambda row: (budgets(evidence).index(row["budget"]), methods.index(row["method"])),
    )


def contrast_table(evidence, *, contrast, policy=None, context="all", exposure="spread"):
    output, block = [], base_config(evidence)["bootstrap_block"]
    for dimension in evidence["protocol"]["config"]["dimensions"]:
        for row in select_pairs(
            evidence,
            contrast=contrast,
            policy=policy,
            context=context,
            dimension=dimension,
            exposure=exposure,
        ):
            output.append(
                {
                    "dimension": dimension,
                    "budget": row["history_budget"],
                    "policy": row["policy"],
                    "difference": row["difference"],
                    "range": bounds(row, block),
                    "origins": row["paired_origins"],
                }
            )
    return sorted(
        output,
        key=lambda row: (
            evidence["protocol"]["config"]["dimensions"].index(row["dimension"]),
            budgets(evidence).index(row["budget"]),
            row["policy"],
        ),
    )


def findings_text(evidence):
    lines = [
        "The contrasts below measure changes in implemented policies on inspected settlements. They do not partition an unobserved true covariance error into causal percentages.",
        "",
    ]
    block = base_config(evidence)["bootstrap_block"]
    primary_source = [
        row
        for row in select_pairs(evidence, contrast="source_policy", policy="diagonal")
        if row["history_budget"] == "0"
    ]
    reconstruction = [
        row
        for row in select_pairs(evidence, contrast="reconstruction_pool", policy="fixed_selectors")
        if row["history_budget"] == "0"
    ]
    if (
        primary_source
        and reconstruction
        and abs(primary_source[0]["difference"]) < 0.01 * abs(reconstruction[0]["difference"])
    ):
        lines += [
            "**The practical bottleneck in these controls is the chosen curve representation and its residual-risk repair.** Replacing the source-C estimator with direct innovation H changes the primary spread score by much less than reconstructing raw movements or adding the residual corrections. A small source-policy gap neither proves that C is correctly estimated nor identifies the physical forcing operator.",
            "",
        ]
    for contrast, policy, label in (
        (
            "reconstruction_pool",
            "fixed_selectors",
            "Passing the matched maturity pool through three-mode reconstruction",
        ),
        (
            "source_policy",
            "diagonal",
            "Replacing direct innovation H with source-C fitting, with full residual repair fixed",
        ),
        (
            "residual_correlated",
            "source",
            "Adding correlated residuals and cross terms to source factors",
        ),
        ("residual_diagonal", "source", "Adding the donor-excluded diagonal correction"),
    ):
        rows = [
            row
            for row in select_pairs(evidence, contrast=contrast, policy=policy)
            if row["history_budget"] == "0"
        ]
        if rows:
            row, interval = rows[0], bounds(rows[0], block)
            lines.append(
                f"- **{label}:** zero-history spread QLIKE changes by {row['difference']:+.4f}, with descriptive 95% block range {display(interval)} across {row['paired_origins']} paired origins."
            )
    lines += [
        "",
        "Negative changes favor the positive-weight or added policy. Compare the full six cells and interaction contrasts before interpreting a single addition as a general improvement. Availability, annual strata and the one-day control remain part of the evidence.",
        "These matched interventions do not telescope into the entire SPDE-minus-raw-pool performance gap: historical maturity-slot mapping and own-history blending differ from evaluating the current actual contracts at their future maturities. They remain additional policy choices rather than errors uniquely assigned to one of these three stages.",
    ]
    one_day = [
        row
        for row in select_pairs(
            evidence, contrast="source_policy", policy="diagonal", context="one_day"
        )
        if row["history_budget"] == "0"
    ]
    if one_day:
        row = one_day[0]
        lines += [
            "",
            f"The matched one-day source-C minus direct-H spread difference is {row['difference']:+.6f}, with block range {display(bounds(row, block))}. This controls interval-length mixing while retaining the source estimator's regularization and constraints.",
        ]
    retention, _ = diagnostic_tables(evidence)
    retained = {
        row["exposure"]: row["retention"]
        for row in retention
        if row["dimension"] == 3 and row["budget"] == "0"
    }
    if all(retained.get(name) is not None for name in ("outright", "spread", "butterfly")):
        lines += [
            "",
            f"At zero own history, three-mode reconstruction retains {retained['outright']:.1%} of outright, {retained['spread']:.1%} of spread and {retained['butterfly']:.1%} of butterfly training second moments in the matched pooling control. These are ratios of mean window moments, rather than shares of true latent risk or out-of-sample calibration.",
        ]
    rows = {row["method"]: row for row in select_summary(evidence) if row["history_budget"] == "0"}
    raw = {
        row["method"]: row
        for row in select_summary(evidence, dimension=0)
        if row["history_budget"] == "0"
    }
    if "source_diagonal" in rows and "pooled_shrink" in raw:
        repaired, pool = rows["source_diagonal"]["metrics"], raw["pooled_shrink"]["metrics"]
        lines += [
            "",
            f"Full residual repair gives source-C spread QLIKE {repaired['spread']['qlike']:.3f}, versus raw pooling {pool['spread']['qlike']:.3f}; their hedge RMS levels are CNY {repaired['hedge']['realized_rms_cny']:.1f} and {pool['hedge']['realized_rms_cny']:.1f}. These descriptive levels illustrate that a better risk score need not imply a lower realized hedge RMS; they are not an additional paired significance test.",
        ]
    if "source_correlated" in rows and "source_factor" in rows:
        correlated, factor = rows["source_correlated"]["metrics"], rows["source_factor"]["metrics"]
        lines += [
            "",
            f"Correlated residuals alone are not a universal repair: at zero own history the source model's butterfly QLIKE changes from {factor['butterfly']['qlike']:.3f} to {correlated['butterfly']['qlike']:.3f}, and hedge RMS from CNY {factor['hedge']['realized_rms_cny']:.1f} to {correlated['hedge']['realized_rms_cny']:.1f}, despite the improved spread QLIKE. Evaluate all exposures and the diagonal addition separately.",
        ]
    stress_2022 = [
        row
        for row in select_pairs(
            evidence, contrast="residual_diagonal", policy="source", year="2022"
        )
        if row["history_budget"] == "0"
    ]
    if stress_2022:
        lines += [
            "",
            f"The 2022 diagonal-repair spread contrast is {stress_2022[0]['difference']:+.3f}, a particularly large annual effect. QLIKE means are sensitive to severe underprediction on stress observations; the large pooled effects and wide block ranges should not be read as equal improvement across regimes or as proof of erroneous source quotes.",
        ]
    lines += [
        "",
        "Annual results are essential: large correction effects and wide block ranges can be concentrated in particular years. The pooled mean is not a claim of equal improvement in every market regime.",
    ]
    return "\n".join(lines)


def oracle_tables(evidence):
    oracle = evidence["oracle"]
    transport = [
        {
            "representation": row["representation"],
            "policy": row["covariance_policy"],
            "days": row["future_calendar_days"],
            "error": row["relative_kernel_error"]["mean"],
            "mcse": row["relative_kernel_error"]["mcse"],
        }
        for row in oracle["transport"]["rows"]
    ]
    residual = [
        {
            "policy": row["policy"],
            "error": row["relative_kernel_error"],
            "spread_ratio": row["risk_ratio"]["spread"],
            "butterfly_ratio": row["risk_ratio"]["butterfly"],
        }
        for row in oracle["residual"]["rows"]
    ]
    return transport, residual


def diagnostic_tables(evidence):
    """Expose reconstruction retention and optimization, without private rows."""
    retention, numerical = [], []
    for row in evidence["diagnostics"]["rows"]:
        numeric = row["numeric"]

        def value(key, statistic="mean"):
            item = numeric.get(key, {})
            return item.get(statistic)

        def maximum(key):
            values = value(key, "min_median_max")
            return values[-1] if values else None

        if row["context"] == "all" and row["method"] == "reconstructed_pool":
            for exposure in ("outright", "spread", "butterfly"):
                prefix = f"reconstruction_retention.{exposure}."
                raw = value(prefix + "raw_second_moment")
                retention.append(
                    {
                        "dimension": row["dimension"],
                        "budget": row["history_budget"],
                        "exposure": exposure,
                        "count": numeric.get(prefix + "raw_second_moment", {}).get("count", 0),
                        "retention": value(prefix + "fitted_second_moment") / raw
                        if raw and value(prefix + "fitted_second_moment") is not None
                        else None,
                        "residual": value(prefix + "residual_second_moment") / raw
                        if raw and value(prefix + "residual_second_moment") is not None
                        else None,
                        "cross": value(prefix + "signed_cross_second_moment") / raw
                        if raw and value(prefix + "signed_cross_second_moment") is not None
                        else None,
                        "identity": maximum("reconstruction_identity_relative_residual"),
                    }
                )
        if row["method"] == "source_diagonal" and row["history_budget"] in ("0", "all"):
            numerical.append(
                {
                    "dimension": row["dimension"],
                    "context": row["context"],
                    "budget": row["history_budget"],
                    "attempted": row["issued"],
                    "scored": row["scored"],
                    "certificate": maximum("source_fit.relative_covariance_error_bound"),
                    "condition": maximum("reconstruction.curve_condition_max"),
                    "leverage": maximum("reconstruction.level_fit_leverage_max"),
                    "support": maximum("relative_support_residual"),
                    "component": maximum("component_sum_relative_residual"),
                    "loo": value("leave_one_donor_out_annual_mse"),
                }
            )
    retention.sort(
        key=lambda row: (
            row["dimension"],
            budgets(evidence).index(row["budget"]),
            ("outright", "spread", "butterfly").index(row["exposure"]),
        )
    )
    numerical.sort(
        key=lambda row: (row["dimension"], row["context"], budgets(evidence).index(row["budget"]))
    )
    return retention, numerical


def data_audit_table(evidence):
    audit = evidence["manifest"].get("importer_audit") or {}
    return [
        {
            "year": row["year"],
            "sessions": row["sessions"],
            "rows": row["rows"],
            "missing": row["settlement_nonfinite"],
            "nonpositive": row["settlement_nonpositive"],
            "volume_zero": row["volume_zero"],
            "oi_zero": row["open_interest_zero"],
            "convention": ", ".join(row.get("volume_counting_conventions", [])),
            "unknown_sessions": row["normalization_unavailable_sessions"],
        }
        for row in audit.get("by_year", [])
    ]


def history_availability_table(evidence):
    output = [
        {
            "budget": row["history_budget"],
            "windows": row["issued"],
            "own_count": row["numeric"].get("complete_changes", {}).get("min_median_max"),
            "pooled_count": row["numeric"].get("pooled_changes", {}).get("min_median_max"),
        }
        for row in evidence["diagnostics"]["rows"]
        if row["context"] == "all" and row["dimension"] == 0 and row["method"] == "pooled_shrink"
    ]
    return sorted(output, key=lambda row: budgets(evidence).index(row["budget"]))


def maturity_table(evidence):
    output = []
    for tenor in base_config(evidence)["target_tenors_days"]:
        for dimension, method in (
            (0, "pooled_shrink"),
            (3, "source_diagonal"),
            (3, "empirical_diagonal"),
        ):
            for row in select_summary(evidence, dimension=dimension, tenor=tenor):
                if row["method"] == method and row["history_budget"] in ("0", "all"):
                    metrics = row["metrics"]
                    output.append(
                        {
                            "tenor": tenor,
                            "budget": row["history_budget"],
                            "method": method,
                            "scored": f"{row['scored']}/{row['episodes']}",
                            "outright": metrics.get("outright", {}).get("qlike"),
                            "spread": metrics.get("spread", {}).get("qlike"),
                            "butterfly": metrics.get("butterfly", {}).get("qlike"),
                            "calibration": metrics.get("spread", {}).get("calibration_ratio"),
                            "rms": metrics.get("hedge", {}).get("realized_rms_cny"),
                        }
                    )
    return sorted(
        output,
        key=lambda row: (
            row["tenor"],
            budgets(evidence).index(row["budget"]),
            ("pooled_shrink", "source_diagonal", "empirical_diagonal").index(row["method"]),
        ),
    )


def report_text(evidence):
    config, manifest = base_config(evidence), evidence["manifest"]
    policies = policy_table(evidence)
    contrast_columns = [
        ("dimension", "Modes"),
        ("budget", "Own history"),
        ("policy", "Fixed policy"),
        ("difference", "QLIKE change"),
        ("range", "95% block range"),
        ("origins", "Paired origins"),
    ]
    transport, residual = oracle_tables(evidence)
    retention, numerical = diagnostic_tables(evidence)
    return "\n".join(
        [
            "# Copper pipeline mechanisms: 2016–2025",
            "",
            "**Question:** how much do curve reconstruction, source-covariance fitting and residual correction change the implemented short-history risk forecasts?",
            "",
            "2016 provides initial history; 2017–2025 is descriptive rolling evaluation. **2026 remains held out and is not parsed or scored.** Raw archived counts are preserved, counting conventions are note-based, and unknown declarations remain unknown. Prices and the five-tonne contract multiplier are unchanged.",
            "",
            "The holdout concerns observation dates and the 2026 archive, not delivery months. Prices observed in 2025 for contracts delivering in 2026 are contemporaneously known and remain eligible; removing those contracts would truncate the 2025 maturity curve.",
            "",
            f"Numerical source: `{manifest['code_revision']}`; clean source at execution: `{not manifest['source_dirty']}`. The run covers {manifest['scheduled_origins']:,} scheduled origins and {manifest['episode_count']:,} target episodes, from {manifest['first_origin']} to {manifest['last_origin']}. Repeated targets and contracts are dependent.",
            "",
            f"Fixed primary settings: {config['window']} adjacent training intervals, {config['half_life']:g}-session EWMA, own-history budgets {display(budgets(evidence))}, three Laguerre modes with two/four-mode sensitivities. Outcomes use the next actual session and the same contract IDs. Current target settlements remain visible; hidden older levels are removed before every transformation.",
            "",
            "## Imported data and counting audit",
            "",
            table(
                data_audit_table(evidence),
                [
                    ("year", "Year"),
                    ("sessions", "Sessions"),
                    ("rows", "Copper rows"),
                    ("missing", "Nonfinite settlements"),
                    ("nonpositive", "Nonpositive settlements"),
                    ("volume_zero", "Zero volume"),
                    ("oi_zero", "Zero OI"),
                    ("convention", "Declared volume convention"),
                    ("unknown_sessions", "Count normalization unavailable sessions"),
                ],
            ),
            "",
            "Counts are preserved as archived integers. An opt-in single-sided view divides only explicitly double-sided declarations by two. Missing declarations, including late-2024/2025 workbooks, remain unknown and a full normalized view is unavailable. This study uses count positivity only, so that normalization does not change donor eligibility or rescale prices. The importer audit is limited to 2016–2025.",
            "",
            "The ten-year panel adds market regimes and forecast origins, rather than extending the lifetime of any individual contract. All-available current-triad history remains limited by actual listings and overlap inside the 252-interval window. Expired contracts do not remain observable indefinitely.",
            "",
            table(
                history_availability_table(evidence),
                [
                    ("budget", "Permitted own history"),
                    ("windows", "Masked windows"),
                    ("own_count", "Complete actual-triad changes: min / median / max"),
                    ("pooled_count", "Pooled changes: min / median / max"),
                ],
            ),
            "",
            "## Findings",
            "",
            findings_text(evidence),
            "",
            "## Complete six-cell comparison",
            "",
            "Within a row comparison, the reconstructed innovations, residual joint moment and diagonal prediction-error estimate are identical. The source row fits C through interval integration, ridge and a trace constraint; the direct row estimates normalized innovation H. Columns add the named residual corrections. These are policy contrasts, not recovery errors against latent physical truth.",
            "",
            "![Covariance policy crossed with residual correction](six-cell-scores.png)",
            "",
            table(
                policies,
                [
                    ("budget", "Own history"),
                    ("method", "Policy"),
                    ("scored", "Scored / attempted"),
                    ("outright", "Outright QLIKE"),
                    ("spread", "Spread QLIKE"),
                    ("butterfly", "Butterfly QLIKE"),
                    ("calibration", "Spread forecast / realized"),
                    ("joint", "Joint quadratic / 10¹² CNY⁴"),
                    ("rms", "Hedge RMS CNY"),
                ],
            ),
            "",
            "QLIKE is lower-is-better. Calibration is a ratio of origin-averaged forecast and realized second moments, not conditional calibration. Hedge RMS is a settlement-risk outcome with positions bounded to [−2,2], without execution costs.",
            "",
            "### Raw-data baselines on the same history budgets",
            "",
            table(
                policy_table(evidence, dimension=0),
                [
                    ("budget", "Own history"),
                    ("method", "Raw-data policy"),
                    ("scored", "Scored / attempted"),
                    ("outright", "Outright QLIKE"),
                    ("spread", "Spread QLIKE"),
                    ("butterfly", "Butterfly QLIKE"),
                    ("calibration", "Spread forecast / realized"),
                    ("joint", "Joint quadratic / 10¹² CNY⁴"),
                    ("rms", "Hedge RMS CNY"),
                ],
            ),
            "",
            "The direct baseline requires at least twenty complete changes and is unavailable below that threshold. Joint quadratic scores omit a forecast-independent term and can be negative. Availability can differ across policies; use matched contrasts rather than unpaired level differences to establish a ranking.",
            "",
            "### Maturity strata: raw pooling and the two fully repaired policies",
            "",
            table(
                maturity_table(evidence),
                [
                    ("tenor", "Target days"),
                    ("budget", "Own history"),
                    ("method", "Policy"),
                    ("scored", "Scored / attempted"),
                    ("outright", "Outright QLIKE"),
                    ("spread", "Spread QLIKE"),
                    ("butterfly", "Butterfly QLIKE"),
                    ("calibration", "Spread forecast / realized"),
                    ("rms", "Hedge RMS CNY"),
                ],
            ),
            "",
            "## Matched reconstruction and source-policy contrasts",
            "",
            "![Stage contrasts by history and representation dimension](stage-contrasts.png)",
            "",
            "The reconstructed pooling control preserves raw donor selection, actual IDs, adjacent intervals, ages, complete counts and own-history blend weights. Only endpoint prices pass through reconstruction. This avoids attributing a change of maturity mapping to curve smoothing. Historical own-triad prices can have tenors outside the 30–365-day fitting domain; their fitted values include extrapolation, so this is an entire reconstruction-policy contrast rather than a pure interpolation-error estimate.",
            "",
            "### Reconstructed versus raw maturity pooling",
            "",
            table(contrast_table(evidence, contrast="reconstruction_pool"), contrast_columns),
            "",
            "### Source C versus direct H, full residual correction fixed",
            "",
            table(
                contrast_table(evidence, contrast="source_policy", policy="diagonal"),
                contrast_columns,
            ),
            "",
            "### Correlated residual and cross-term addition",
            "",
            table(
                contrast_table(evidence, contrast="residual_correlated", policy="source"),
                contrast_columns,
            ),
            "",
            "### Diagonal predictive-error addition",
            "",
            table(
                contrast_table(evidence, contrast="residual_diagonal", policy="source"),
                contrast_columns,
            ),
            "",
            "### Interaction: source-policy difference in diagonal-addition effects",
            "",
            table(contrast_table(evidence, contrast="interaction_diagonal"), contrast_columns),
            "",
            "Interactions prevent a unique additive allocation of performance gaps to stages. The diagonal correction may include leverage, reconstruction error and representation mismatch; it is not identified fresh noise or physical forcing.",
            "",
            "## Calendar, interval-length and calibration checks",
            "",
            "![Annual and matched-one-day source contrasts](calendar-context.png)",
            "",
            "The one-day context sets weights to zero for other training intervals, preserves the original state chronology and EWMA ages, and scores only one-day future intervals. Direct H over all intervals is otherwise a mixture of normalized interval moments. Contexts can have different future support; compare their source/direct differences within context, not unpaired score levels.",
            "",
            "![Forecast versus realized second moments](risk-calibration.png)",
            "",
            "Targets are averaged within each origin before dates. Circular origin-block draws are shared across contexts, methods, dimensions and budgets; excluded one-day origins remain absent in the full scheduled calendar. Four origins (about twenty sessions) is primary, with 1, 2 and 8-origin sensitivities. These descriptive percentile ranges are not IID Monte Carlo uncertainty or untouched confirmation.",
            "",
            "## Known-truth numerical controls",
            "",
            "These two synthetic controls are separate experiments. An exact finite transport trajectory has known latent coefficients and C; spatially perturbed twelve-quote snapshots are reconstructed independently. A Gaussian correlated factor/residual increment model verifies cross terms and Schur identities, without claiming its residuals come from a level-consistent SPDE.",
            "",
            table(
                transport,
                [
                    ("representation", "Coefficient input"),
                    ("policy", "Covariance policy"),
                    ("days", "Future days"),
                    ("error", "Mean relative kernel error"),
                    ("mcse", "Replication MCSE"),
                ],
            ),
            "",
            table(
                residual,
                [
                    ("policy", "Known-block control"),
                    ("error", "Relative kernel error"),
                    ("spread_ratio", "Spread forecast / truth"),
                    ("butterfly_ratio", "Butterfly forecast / truth"),
                ],
            ),
            "",
            f"The exact full-joint residual identity has relative error {evidence['oracle']['residual']['oracle_identity_relative_error']:.3g}. Synthetic MCSE uses {evidence['oracle']['replications']} independent replications; this differs from the copper calendar-block ranges. Full settings and sampled-joint errors are in [oracle.json](oracle.json).",
            "",
            "## Availability, numerical diagnostics and limitations",
            "",
            "### Reconstruction retention",
            "",
            "All three ratios use means of the corresponding training-window second moments across available masked views; each window uses its fixed EWMA and own-history blend. Fitted, residual and signed-cross parts therefore sum to one up to roundoff. The count reports available diagnostic windows, which overlap and are dependent. These are visible-change diagnostics rather than out-of-sample calibration or latent approximation errors.",
            "",
            table(
                retention,
                [
                    ("dimension", "Modes"),
                    ("budget", "Own history"),
                    ("exposure", "Exposure"),
                    ("count", "Available windows"),
                    ("retention", "Fitted / raw means"),
                    ("residual", "Residual / raw means"),
                    ("cross", "Signed cross / raw means"),
                    ("identity", "Max identity residual"),
                ],
            ),
            "",
            "### Source certificates and reconstruction leverage",
            "",
            "The numerical certificate bounds optimization error under the fitted objective, not statistical error. Leverage concerns the permitted level reconstruction; the donor-excluded predictive correction can contain that estimation uncertainty. Source support and component checks retain their relative scales.",
            "",
            table(
                numerical,
                [
                    ("dimension", "Modes"),
                    ("context", "Intervals"),
                    ("budget", "Own history"),
                    ("scored", "Scored"),
                    ("certificate", "Max relative optimizer bound"),
                    ("condition", "Max curve condition"),
                    ("leverage", "Max level leverage"),
                    ("support", "Max support residual"),
                    ("component", "Max component residual"),
                ],
            ),
            "",
            "No unavailable method receives a hidden fallback. Source numerical certificates, PSD/support checks, cross-term identities, reconstruction leverage, donor-excluded prediction errors and cache bounds are retained in [diagnostics.json](diagnostics.json). All history/maturity/year/context/dimension summaries are in [summary.json](summary.json); fixed contrasts, interaction effects and block sensitivities are in [paired.json](paired.json).",
            "",
            "The data have monthly maturity spacing and daily observations. Two/four-mode sensitivities change a finite representation; they do not measure continuum convergence. Four residual maturity features plus a diagonal remainder remain restrictive. No real-data latent source or curve truth is available. These are exploratory covariance-risk comparisons, not price forecasting, derivative pricing, new-listing performance or an executable hedge backtest.",
            "",
            f"Run time: {manifest['runtime_seconds']:.1f} seconds in the recorded environment; see [manifest.json](manifest.json) for archive hashes, convention audit and source provenance. Historical bundles remain unchanged. Raw panels, individual forecasts and P&Ls stay local.",
            "",
            "## Reproduction",
            "",
            "```sh",
            "uv run spde-covariance benchmark-mechanisms experiments/configs/copper-mechanisms.toml \\",
            "  --data data/local/shfe \\",
            "  --older-data data/local/shfe-extension/development-2016-2021 \\",
            "  --output data/local/shfe/runs/mechanisms-v1 --workers 8 \\",
            "  --export experiments/results/copper-mechanisms",
            "uv run python scripts/render_mechanisms.py experiments/results/copper-mechanisms --build-notebook",
            "uv run python scripts/check_notebook.py notebooks/08_copper_mechanism_attribution.ipynb --write-outputs",
            "```",
            "",
            "Output directories must be new. Commit numerical source before full runs. The renderer and aggregate-only notebook verify numerical checksums and do not refit models or read exchange files. Read the [protocol](../../../docs/mechanism-attribution.md) and [executed notebook](../../../notebooks/08_copper_mechanism_attribution.ipynb).",
            "",
        ]
    )


def build_notebook(path=ROOT / "notebooks/08_copper_mechanism_attribution.ipynb", *, evidence=None):
    def markdown(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}

    def code(source):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source.strip().splitlines(True),
        }

    helpers = "\n\n".join(
        inspect.getsource(function)
        for function in (
            display,
            table,
            base_config,
            budgets,
            select_summary,
            select_pairs,
            bounds,
            _point,
            plot_six_cells,
            plot_stage_contrasts,
            plot_calendar_context,
            plot_calibration,
            policy_table,
            contrast_table,
            findings_text,
            oracle_tables,
            diagnostic_tables,
            data_audit_table,
            history_availability_table,
            maturity_table,
        )
    )
    setup = f"""from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spde_covariance.empirical_mechanisms import load_mechanism_evidence

evidence = load_mechanism_evidence(Path("../experiments/results/copper-mechanisms"))
METHOD_LABELS = {METHOD_LABELS!r}
CELLS = {CELLS!r}
plt.rcParams.update({{"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False}})

{helpers}

print("Numerical source:", evidence["manifest"]["code_revision"])
print("Clean numerical source:", not evidence["manifest"]["source_dirty"])
print("Chronology:", evidence["protocol"]["chronology"])
print("Scheduled origins:", evidence["manifest"]["scheduled_origins"])
assert evidence["paired"]["joint_resampling"]
assert evidence["paired"]["one_day_missing_origins_retained"]
assert base_config(evidence)["test_end"] < "2026-01-01"
"""
    columns = [
        ("dimension", "Modes"),
        ("budget", "Own history"),
        ("policy", "Fixed policy"),
        ("difference", "Score change"),
        ("range", "95% block range"),
        ("origins", "Paired origins"),
    ]
    cells = [
        markdown("""# 08 · Reconstruction, source covariance and residual risk

**Question:** which stages of our implemented transport pipeline change the
quality of short-history copper risk forecasts?

2016 is initial history; 2017–2025 is descriptive rolling evaluation. **2026 stays
held out and is never parsed or scored.** The current settlement is permitted;
older target prices are masked before every transformation. Raw counts are
preserved, documented counting conventions have an explicit normalized view,
and unknown declarations are not guessed.
The 2026 holdout concerns observation dates, not delivery months: 2025 quotes
for 2026-delivery contracts are contemporaneously known and remain eligible.

More calendar years add regimes and origins, not longer lives for individual
contracts. Even all-available current-triad history is limited by listings and
overlap inside the 252-interval window.

This notebook reads checksummed aggregate evidence only. It estimates policy
effects, rather than causal percentages of unobserved true covariance error.
[Protocol](../docs/mechanism-attribution.md) · [Report](../experiments/results/copper-mechanisms/report.md)
"""),
        markdown(
            "## Findings before the implementation details\n\n" + findings_text(evidence)
            if evidence is not None
            else "## Findings\n\nBuild this notebook from the verified evidence bundle to populate the numerical findings."
        ),
        code(setup),
        code(
            """print(findings_text(evidence))
print(table(data_audit_table(evidence), [("year", "Year"), ("sessions", "Sessions"), ("rows", "Copper rows"), ("missing", "Nonfinite settlements"), ("nonpositive", "Nonpositive settlements"), ("volume_zero", "Zero volume"), ("oi_zero", "Zero OI"), ("convention", "Count convention"), ("unknown_sessions", "Unknown-count sessions")]))
print("Raw archived counts remain unchanged; unknown normalized views are unavailable, not guessed.")
print(table(history_availability_table(evidence), [("budget", "Own history"), ("windows", "Masked windows"), ("own_count", "Actual-triad complete changes: min/median/max"), ("pooled_count", "Pooled changes: min/median/max")]))
"""
        ),
        markdown(r"""## 1. A matched six-cell experiment

Hold fitted curves and innovations
$$z_i=(\widehat a_i-M_{\Delta_i}\widehat a_{i-1})/\sqrt{\Delta_i}$$
fixed. Compare source-C integration with direct EWMA innovation H. Cross both
with factor-only, correlated-residual/cross-term and added-diagonal forecasts.
The two main covariance forecasts are
$$\widehat H_D=\sum_iw_i z_i z_i^\top,\qquad
\widehat H_C(\delta)=\delta^{-1}\int_0^\delta M_u\widehat C M_u^\top\,du.$$
Row comparisons share exactly the same fitted residual statistics. Source-C
also imposes ridge and trace constraints, so its difference is a complete
source-estimation policy effect. A variance interpretation assumes negligible
conditional means. CNY risk uses the five-tonne multiplier squared.
"""),
        code("""plot_six_cells(evidence)
plt.show()
rows = policy_table(evidence)
print(table(rows, [("budget", "Own history"), ("method", "Policy"), ("scored", "Scored / attempted"), ("outright", "Outright QLIKE"), ("spread", "Spread QLIKE"), ("butterfly", "Butterfly QLIKE"), ("calibration", "Spread calibration"), ("joint", "Joint quadratic / 10¹² CNY⁴"), ("rms", "Hedge RMS CNY")]))
print("Raw-data baselines; direct unavailable below twenty complete changes:")
print(table(policy_table(evidence, dimension=0), [("budget", "Own history"), ("method", "Policy"), ("scored", "Scored / attempted"), ("outright", "Outright QLIKE"), ("spread", "Spread QLIKE"), ("butterfly", "Butterfly QLIKE"), ("calibration", "Spread calibration"), ("joint", "Joint quadratic / 10¹² CNY⁴"), ("rms", "Hedge RMS CNY")]))
print("Maturity strata at zero/all available history:")
print(table(maturity_table(evidence), [("tenor", "Target days"), ("budget", "History"), ("method", "Policy"), ("scored", "Scored / attempted"), ("outright", "Outright QLIKE"), ("spread", "Spread QLIKE"), ("butterfly", "Butterfly QLIKE"), ("calibration", "Spread calibration"), ("rms", "Hedge RMS CNY")]))
assert all(row["scored"] <= row["available"] <= row["episodes"] for row in evidence["summary"]["rows"])
"""),
        markdown("""## 2. Separate reconstruction from maturity selection

Raw and reconstructed pooling use identical historical actual IDs, intervals,
ages, own-triad counts and blend weights. Only price changes pass through the
curve fit. In this invariant finite basis, adjusted innovations evaluated at
end maturity equal fitted same-ID endpoint changes exactly. Retained spread and
butterfly movements, residuals and signed cross terms describe this filter.
Two- and four-mode sensitivity runs are fixed; they do not reveal continuum truth.
Historical own-triad fitted prices can extrapolate beyond the 30–365-day donor
fit domain. This is a reconstruction-policy effect, not pure interpolation error.
"""),
        code(f"""plot_stage_contrasts(evidence)
plt.show()
print(table(contrast_table(evidence, contrast="reconstruction_pool"), {columns!r}))
print("Source-C minus direct-H with full residual repair:")
print(table(contrast_table(evidence, contrast="source_policy", policy="diagonal"), {columns!r}))
"""),
        markdown(r"""## 3. Residual additions and their interactions

The correlated correction retains factor/residual cross terms through a Schur
decomposition. The diagonal correction uses truly donor-excluded predictive
errors and can include estimation and reconstruction error. It is not identified
latent noise. Compare column additions in both covariance rows and their
difference-in-differences; do not force an additive allocation of error.
These within-family contrasts do not telescope into the full SPDE/raw-pool gap:
historical maturity-slot mapping and own-history blending also differ from
evaluation at the current actual contracts' future maturities.

For the shared training blocks $(H,G,R)$, let $A=G^\top H^\dagger$ and
$\Omega=R-G^\top H^\dagger G$. The fully repaired price-change moment is
$$\widehat\Sigma=\delta\left[(B+\Psi A)H_f(B+\Psi A)^\top
+\Psi\Omega\Psi^\top+D\right].$$
The row chooses $H_f$; the other quantities stay fixed. The factor/residual
cross correction is signed, so it is not an independent positive risk source.
"""),
        code(f"""for contrast in ("residual_correlated", "residual_diagonal", "interaction_correlated", "interaction_diagonal"):
    print(contrast)
    print(table(contrast_table(evidence, contrast=contrast), {columns!r}))
"""),
        markdown("""## 4. Calendar and interval-length controls

All-interval direct H mixes normalized covariances from unequal intervals. The
one-day control trains both policies on one-calendar-day intervals and compares
only one-day future outcomes. Other intervals retain their positions with zero
weights; no state gap is bridged and no EWMA age is compressed. The bootstrap
retains the master scheduled-origin calendar across all dimensions and budgets.
Annual strata are exploratory; repeated contracts and origins are dependent.
"""),
        code("plot_calendar_context(evidence)\nplt.show()\nplot_calibration(evidence)\nplt.show()"),
        markdown("""## 5. Synthetic controls with known truth

The transport control has known coefficients and C and reconstructs twelve-quote
spatial snapshots under fixed perturbation. The separate Gaussian increment
control has known correlated factor/residual blocks and verifies exact cross
terms. Its residuals are not asserted to arise from a level-consistent SPDE.
Synthetic MCSE uses independent replications; copper ranges use calendar blocks.
"""),
        code("""transport, residual = oracle_tables(evidence)
print(table(transport, [("representation", "Coefficient input"), ("policy", "Covariance policy"), ("days", "Future days"), ("error", "Relative kernel error"), ("mcse", "Replication MCSE")]))
print(table(residual, [("policy", "Known-block control"), ("error", "Relative kernel error"), ("spread_ratio", "Spread / truth"), ("butterfly_ratio", "Butterfly / truth")]))
assert evidence["oracle"]["residual"]["oracle_identity_relative_error"] < 1e-10
assert evidence["oracle"]["transport"]["maximum_relative_covariance_certificate"] <= 1e-5
"""),
        markdown("""## 6. Availability, numerical evidence and boundaries

Inspect source certificates, PSD/support checks, reconstruction leverage and
remaining residual dependence before ranking scores. Unavailable policies get
no fallback. Four residual features plus a diagonal remainder remain restrictive.
Held-out 2026 has not been used for protocol choices or model comparison. No
price prediction, derivative pricing or execution-cost result is established.
"""),
        code("""retention, numerical = diagnostic_tables(evidence)
print("Training reconstruction retention: ratios of mean window second moments; windows overlap.")
print(table(retention, [("dimension", "Modes"), ("budget", "History"), ("exposure", "Exposure"), ("count", "Available windows"), ("retention", "Fitted/raw means"), ("residual", "Residual/raw means"), ("cross", "Signed cross/raw means"), ("identity", "Identity residual")]))
assert all(abs(row["retention"] + row["residual"] + row["cross"] - 1) < 1e-8 for row in retention if all(row[key] is not None for key in ("retention", "residual", "cross")))
print(table(numerical, [("dimension", "Modes"), ("context", "Intervals"), ("budget", "History"), ("certificate", "Max optimizer bound"), ("condition", "Curve condition"), ("leverage", "Level leverage"), ("support", "Support residual"), ("component", "Component residual")]))
availability = [{key: row[key] for key in ("context", "dimension", "method", "history_budget", "issued", "scored", "failures")} for row in evidence["diagnostics"]["rows"]]
print(table(availability, [("context", "Intervals"), ("dimension", "Modes"), ("method", "Policy"), ("history_budget", "History"), ("issued", "Attempted"), ("scored", "Scored"), ("failures", "Unavailable reasons")]))
print("Cache audit:", evidence["diagnostics"]["cache"])
print("Recorded runtime seconds:", evidence["manifest"]["runtime_seconds"])
print("Numerical diagnostics are available in the checksummed diagnostics.json aggregate.")
assert evidence["manifest"]["local_record_policy"] == "contract-level prices, forecasts and P&Ls are not exported"
"""),
        markdown("""## Reproduction

This notebook never opens market files. Run
`uv run python scripts/check_notebook.py notebooks/08_copper_mechanism_attribution.ipynb`.
The [report](../experiments/results/copper-mechanisms/report.md) gives local experiment
commands. Raw settlement panels, individual forecasts and P&Ls stay local;
public aggregates retain source, configuration, archive and environment hashes.
Historical evidence is preserved. The next confirmation requires frozen choices
before opening the held-out 2026 data.
"""),
    ]
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    # Notebook source is generated from functions and long literal metadata.
    # Format before render() records its source hash, rather than changing the
    # source after a presentation manifest has already been issued.
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(path)


def render(directory):
    directory = Path(directory)
    evidence = load_mechanism_evidence(directory)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    artifacts = {}
    for filename, function in (
        ("six-cell-scores.png", plot_six_cells),
        ("stage-contrasts.png", plot_stage_contrasts),
        ("calendar-context.png", plot_calendar_context),
        ("risk-calibration.png", plot_calibration),
    ):
        figure = function(evidence)
        figure.savefig(directory / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        artifacts[filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    report = directory / "report.md"
    report.write_text(report_text(evidence))
    artifacts["report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()
    notebook = ROOT / "notebooks/08_copper_mechanism_attribution.ipynb"
    notebook_source = [
        {"cell_type": cell["cell_type"], "source": "".join(cell["source"])}
        for cell in json.loads(notebook.read_text())["cells"]
    ]
    (directory / "report-manifest.json").write_text(
        json.dumps(
            {
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numerical_checksums_sha256": hashlib.sha256(
                    (directory / "checksums.json").read_bytes()
                ).hexdigest(),
                "artifacts_sha256": artifacts,
                "notebook_source_sha256": hashlib.sha256(
                    json.dumps(notebook_source, sort_keys=True).encode()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=ROOT / "experiments/results/copper-mechanisms"
    )
    parser.add_argument("--build-notebook", action="store_true")
    arguments = parser.parse_args()
    if arguments.build_notebook:
        print(build_notebook(evidence=load_mechanism_evidence(arguments.directory)))
    print(render(arguments.directory))


if __name__ == "__main__":
    main()
