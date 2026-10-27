"""Render mechanism-attribution evidence without opening market archives."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
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
                    "o-",
                    color=color,
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
            ax.plot(range(len(history)), values, "o-", color=color, ms=4, label=label)
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
                "spread": metrics.get("spread", {}).get("qlike"),
                "butterfly": metrics.get("butterfly", {}).get("qlike"),
                "calibration": metrics.get("spread", {}).get("calibration_ratio"),
                "rms": metrics.get("hedge", {}).get("realized_rms_cny"),
            }
        )
    return output


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
    return output


def findings_text(evidence):
    lines = [
        "The contrasts below measure changes in implemented policies on inspected settlements. They do not partition an unobserved true covariance error into causal percentages.",
        "",
    ]
    block = base_config(evidence)["bootstrap_block"]
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
    return "\n".join(
        [
            "# Copper pipeline mechanisms: 2016–2025",
            "",
            "**Question:** how much do curve reconstruction, source-covariance fitting and residual correction change the implemented short-history risk forecasts?",
            "",
            "2016 provides initial history; 2017–2025 is descriptive rolling evaluation. **2026 remains held out and is not parsed or scored.** Raw archived counts are preserved, counting conventions are note-based, and unknown declarations remain unknown. Prices and the five-tonne contract multiplier are unchanged.",
            "",
            f"Numerical source: `{manifest['code_revision']}`; clean source at execution: `{not manifest['source_dirty']}`. The run covers {manifest['scheduled_origins']:,} scheduled origins and {manifest['episode_count']:,} target episodes, from {manifest['first_origin']} to {manifest['last_origin']}. Repeated targets and contracts are dependent.",
            "",
            f"Fixed primary settings: {config['window']} adjacent training intervals, {config['half_life']:g}-session EWMA, own-history budgets {display(budgets(evidence))}, three Laguerre modes with two/four-mode sensitivities. Outcomes use the next actual session and the same contract IDs. Current target settlements remain visible; hidden older levels are removed before every transformation.",
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
                    ("spread", "Spread QLIKE"),
                    ("butterfly", "Butterfly QLIKE"),
                    ("calibration", "Spread forecast / realized"),
                    ("rms", "Hedge RMS CNY"),
                ],
            ),
            "",
            "QLIKE is lower-is-better. Calibration is a ratio of origin-averaged forecast and realized second moments, not conditional calibration. Hedge RMS is a settlement-risk outcome with positions bounded to [−2,2], without execution costs.",
            "",
            "## Matched reconstruction and source-policy contrasts",
            "",
            "![Stage contrasts by history and representation dimension](stage-contrasts.png)",
            "",
            "The reconstructed pooling control preserves raw donor selection, actual IDs, adjacent intervals, ages, complete counts and own-history blend weights. Only endpoint prices pass through reconstruction. This avoids attributing a change of maturity mapping to curve smoothing.",
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
            "  --data data/local/shfe --older-data data/local/shfe-extension \\",
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


def build_notebook(path=ROOT / "notebooks/08_copper_mechanism_attribution.ipynb"):
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

This notebook reads checksummed aggregate evidence only. It estimates policy
effects, rather than causal percentages of unobserved true covariance error.
[Protocol](../docs/mechanism-attribution.md) · [Report](../experiments/results/copper-mechanisms/report.md)
"""),
        code(setup),
        code(
            "print(findings_text(evidence))\nprint('Importer audit:', evidence['manifest'].get('importer_audit'))"
        ),
        markdown(r"""## 1. A matched six-cell experiment

Hold fitted curves and innovations
$$z_i=(\widehat a_i-M_{\Delta_i}\widehat a_{i-1})/\sqrt{\Delta_i}$$
fixed. Compare source-C integration with direct EWMA innovation H. Cross both
with factor-only, correlated-residual/cross-term and added-diagonal forecasts.
Row comparisons share exactly the same fitted residual statistics. Source-C
also imposes ridge and trace constraints, so its difference is a complete
source-estimation policy effect. A variance interpretation assumes negligible
conditional means. CNY risk uses the five-tonne multiplier squared.
"""),
        code("""plot_six_cells(evidence)
plt.show()
rows = policy_table(evidence)
print(table(rows, [("budget", "Own history"), ("method", "Policy"), ("scored", "Scored / attempted"), ("spread", "Spread QLIKE"), ("butterfly", "Butterfly QLIKE"), ("calibration", "Spread calibration"), ("rms", "Hedge RMS CNY")]))
assert all(row["scored"] <= row["available"] <= row["episodes"] for row in evidence["summary"]["rows"])
"""),
        markdown("""## 2. Separate reconstruction from maturity selection

Raw and reconstructed pooling use identical historical actual IDs, intervals,
ages, own-triad counts and blend weights. Only price changes pass through the
curve fit. In this invariant finite basis, adjusted innovations evaluated at
end maturity equal fitted same-ID endpoint changes exactly. Retained spread and
butterfly movements, residuals and signed cross terms describe this filter.
Two- and four-mode sensitivity runs are fixed; they do not reveal continuum truth.
"""),
        code(f"""plot_stage_contrasts(evidence)
plt.show()
print(table(contrast_table(evidence, contrast="reconstruction_pool"), {columns!r}))
print("Source-C minus direct-H with full residual repair:")
print(table(contrast_table(evidence, contrast="source_policy", policy="diagonal"), {columns!r}))
"""),
        markdown("""## 3. Residual additions and their interactions

The correlated correction retains factor/residual cross terms through a Schur
decomposition. The diagonal correction uses truly donor-excluded predictive
errors and can include estimation and reconstruction error. It is not identified
latent noise. Compare column additions in both covariance rows and their
difference-in-differences; do not force an additive allocation of error.
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
        code("""availability = [{key: row[key] for key in ("context", "dimension", "method", "history_budget", "issued", "scored", "failures")} for row in evidence["diagnostics"]["rows"]]
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
        print(build_notebook())
    print(render(arguments.directory))


if __name__ == "__main__":
    main()
