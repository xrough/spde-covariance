"""Render short-history reports and figures from checksummed aggregates only."""

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

from spde_covariance.empirical_short_history import load_short_history_evidence

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dfm_joint", "spde_joint", "pooled_shrink", "direct")
LABELS = {
    "dfm_joint": "Learned factors + residual",
    "spde_joint": "Transport + residual",
    "pooled_shrink": "Maturity pool + short history",
    "direct": "Same-contract direct",
    "spde_uniform": "Original uniform source C",
    "spde_ewma": "EWMA source C",
    "dfm_uniform_joint": "Uniform learned factors + residual",
    "dfm_factor": "Learned factors only",
    "spde_joint_no_nugget": "Transport, no diagonal correction",
    "dfm_joint_no_nugget": "Learned factors, no diagonal correction",
    "outright": "Outright",
    "spread": "Target / nearest spread",
    "butterfly": "Butterfly",
    "hedge": "Bounded hedge",
    "all": "All available",
}
COLORS = {
    "dfm_joint": "#1f77b4",
    "spde_joint": "#d95f02",
    "pooled_shrink": "#666666",
    "direct": "#228833",
}


def display(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(display(x) for x in value) + "]"
    return LABELS.get(str(value), str(value))


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


def phase(evidence):
    names = {r["phase"] for r in evidence["summary"]["rows"]}
    if "evaluation" in names:
        return "evaluation"
    if "test" in names:
        return "test"
    raise ValueError("no evaluation rows in evidence; validation is not a substitute")


def budgets(evidence):
    return [
        "all" if h is None else str(h) for h in evidence["protocol"]["config"]["history_budgets"]
    ]


def primary(evidence, *, tenor="all"):
    config = evidence["protocol"]["config"]
    return [
        r
        for r in evidence["summary"]["rows"]
        if r["phase"] == phase(evidence)
        and r["half_life"] == config["half_life"]
        and r["tenor"] == str(tenor)
    ]


def paired(evidence, *, exposure, tenor="all", half_life=None):
    config = evidence["protocol"]["config"]
    half_life = config["half_life"] if half_life is None else half_life
    return [
        r
        for r in evidence["paired"]["rows"]
        if r["phase"] == phase(evidence)
        and r["half_life"] == half_life
        and r["tenor"] == str(tenor)
        and r["method"] == "dfm_joint"
        and r["reference"] == "spde_joint"
        and r["exposure"] == exposure
    ]


def interval(row, block):
    values = [x for x in row["bootstrap"] if x["block_origins"] == block]
    if not values:
        return None
    return values[0]["percentile_95"]


def plot_scores(evidence):
    """Paired QLIKE differences by history budget and target maturity."""
    config, history = evidence["protocol"]["config"], budgets(evidence)
    tenors = config["target_tenors_days"]
    fig, axes = plt.subplots(1, len(tenors), figsize=(12.3, 4.0), squeeze=False)
    colors = ("#1f77b4", "#d95f02", "#555555")
    for ax, tenor in zip(axes[0], tenors, strict=True):
        ax.axhline(0, color="0.6", lw=1)
        for j, (exposure, color) in enumerate(
            zip(("outright", "spread", "butterfly"), colors, strict=True)
        ):
            rows = paired(evidence, exposure=exposure, tenor=tenor)
            offset = (j - 1) * 0.17
            for k, row in enumerate(rows):
                bounds = interval(row, config["bootstrap_block"])
                x, mean = history.index(row["budget"]) + offset, row["difference"]
                if bounds is not None:
                    ax.plot([x, x], bounds, color=color, lw=1.3)
                ax.plot(x, mean, "o", color=color, ms=4, label=LABELS[exposure] if k == 0 else None)
        ax.set(
            title=f"Target near {tenor} days",
            xticks=range(len(history)),
            xticklabels=["All" if h == "all" else h for h in history],
            xlabel="Permitted own changes",
        )
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].set_ylabel("QLIKE: learned factors − transport\n(shared residual adapter)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Matched risk forecasts: negative differences favor learned factors", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    return fig


def plot_calibration(evidence):
    """Ratios of origin-averaged forecast and realized second moments."""
    history = budgets(evidence)
    rows = primary(evidence)
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8))
    for ax, exposure in zip(axes, ("outright", "spread", "butterfly"), strict=True):
        ax.axhline(1, color="0.5", lw=1)
        for method in METHODS:
            lookup = {r["budget"]: r for r in rows if r["method"] == method}
            values = [
                lookup[h]["metrics"].get(exposure, {}).get("calibration_ratio")
                if h in lookup
                else None
                for h in history
            ]
            values = [np.nan if value is None or value <= 0 else value for value in values]
            ax.plot(
                range(len(history)), values, "o-", color=COLORS[method], ms=4, label=LABELS[method]
            )
        ax.set(
            title=LABELS[exposure],
            xticks=range(len(history)),
            xticklabels=["All" if h == "all" else h for h in history],
            xlabel="Permitted own changes",
            yscale="log",
        )
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Forecast / realized second moment\n(log scale; one is calibrated)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Risk calibration across history budgets; all target maturities", fontsize=12)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    return fig


def plot_hedges(evidence):
    """Paired squared hedge P&L, with half-life and block sensitivities."""
    config, history = evidence["protocol"]["config"], budgets(evidence)
    half_lives = [config["half_life"], *config["sensitivity_half_lives"]]
    blocks = list(
        dict.fromkeys([config["bootstrap_block"], *config["bootstrap_sensitivity_blocks"]])
    )
    fig, axes = plt.subplots(
        1, len(half_lives), figsize=(4.1 * len(half_lives), 4.1), squeeze=False
    )
    colors = plt.cm.Greys(np.linspace(0.85, 0.35, len(blocks)))
    for ax, half_life in zip(axes[0], half_lives, strict=True):
        ax.axhline(0, color="0.6", lw=1)
        rows = paired(evidence, exposure="hedge", half_life=half_life)
        for j, block in enumerate(blocks):
            for k, row in enumerate(rows):
                bounds = interval(row, block)
                x = history.index(row["budget"]) + (j - (len(blocks) - 1) / 2) * 0.13
                if bounds is not None:
                    ax.plot([x, x], np.asarray(bounds) / 1000, color=colors[j], lw=1.3)
                ax.plot(
                    x,
                    row["difference"] / 1000,
                    "o",
                    color=colors[j],
                    ms=3.5,
                    label=f"Block {block} origins" if k == 0 else None,
                )
        ax.set(
            title=f"Half-life {half_life:g} sessions",
            xticks=range(len(history)),
            xticklabels=["All" if h == "all" else h for h in history],
            xlabel="Permitted own changes",
        )
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].set_ylabel("Squared hedge P&L: learned − transport\n(thousand CNY²)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(blocks)), frameon=False)
    fig.suptitle("Bounded hedge outcomes: fixed sensitivities, no selected winner", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    return fig


def report_text(evidence):
    config, manifest = evidence["protocol"]["config"], evidence["manifest"]
    rows = primary(evidence)
    policies = []
    for budget in budgets(evidence):
        for method in METHODS:
            matches = [r for r in rows if r["budget"] == budget and r["method"] == method]
            if not matches:
                continue
            row = matches[0]
            policies.append(
                {
                    "budget": budget,
                    "method": method,
                    "scored": f"{row['scored']}/{row['episodes']}",
                    "origins": row["origins"],
                    "spread": row["metrics"]["spread"].get("qlike"),
                    "butterfly": row["metrics"]["butterfly"].get("qlike"),
                    "rms": row["metrics"]["hedge"].get("realized_rms_cny"),
                }
            )
    comparisons = []
    for exposure in ("outright", "spread", "butterfly", "hedge"):
        for row in paired(evidence, exposure=exposure):
            scale = 1000 if exposure == "hedge" else 1
            bounds = interval(row, config["bootstrap_block"])
            comparisons.append(
                {
                    "budget": row["budget"],
                    "exposure": exposure,
                    "origins": row["paired_origins"],
                    "difference": row["difference"] / scale,
                    "range": None if bounds is None else [x / scale for x in bounds],
                }
            )
    failures = [
        {
            "method": r["method"],
            "issued": r["issued"],
            "available": r["available"],
            "scored": r["scored"],
            "failures": "; ".join(f"{reason}: {count}" for reason, count in r["failures"].items())
            or "None",
        }
        for r in evidence["diagnostics"]["methods"]
    ]
    return "\n".join(
        [
            "# Copper short-history risk transfer",
            "",
            "**Question:** can other copper contracts supply useful next-session risk information when a currently quoted contract has little permitted own history?",
            "",
            "This is an exploratory masked-history experiment on inspected daily SHFE settlements. It is not an actual new-listing study, price forecast, pricing model or execution backtest. All reported comparisons concern conditional second moments; a variance interpretation requires negligible conditional means.",
            "",
            f"Numerical source: `{manifest['code_revision']}`; clean at execution: `{not manifest['source_dirty']}`. The run issued {manifest['issued_count']:,} method forecasts over {manifest['episode_count']:,} target episodes, including development validation and fixed half-life sensitivities. Repeated contracts, targets and dates are not independent replications.",
            "",
            f"Primary settings: {config['window']} historical intervals, {config['half_life']:g}-session EWMA half-life, history budgets {display(budgets(evidence))}, five-tonne contract multiplier. Outcomes are the next observed session of the same actual contracts. The current settlement is always visible. Target comparisons are averaged within origin before pooling.",
            "",
            "## Principal comparison",
            "",
            "Learned maturity-dependent return factors and the transport source-C model use the same residual-transfer adapter. Differences therefore compare complete policies, rather than isolating a causal transport effect. Each paired difference uses identical permitted information and future observations. Negative values favor learned factors.",
            "",
            "![Paired forecast scores by history and maturity](scores-by-budget.png)",
            "",
            f"Lines show descriptive 95% circular moving-block percentile ranges with {config['bootstrap_block']} scheduled origins per block (approximately {config['bootstrap_block'] * config['origin_stride']} trading sessions). They are dependence sensitivities on an inspected sample, not independent Monte Carlo uncertainty.",
            "",
            table(
                comparisons,
                [
                    ("budget", "Own-history budget"),
                    ("exposure", "Risk target"),
                    ("origins", "Paired origins"),
                    ("difference", "Learned − transport"),
                    ("range", "95% block range"),
                ],
            ),
            "",
            "QLIKE differences are dimensionless. Hedge differences above are mean squared P&L in **thousand CNY²**, not differences of RMS. A range spanning zero does not resolve a ranking.",
            "",
            "## Calibration and benchmark levels",
            "",
            "![Risk calibration by own-history budget](risk-calibration.png)",
            "",
            "A forecast/realized second-moment ratio of one is aggregate calibration, not a guarantee of conditional calibration. Below one indicates underprediction. Curves are descriptive policy summaries; availability can differ, so use the paired table for comparisons.",
            "",
            table(
                policies,
                [
                    ("budget", "Own-history budget"),
                    ("method", "Policy"),
                    ("scored", "Scored / attempted episodes"),
                    ("origins", "Origins"),
                    ("spread", "Spread QLIKE"),
                    ("butterfly", "Butterfly QLIKE"),
                    ("rms", "Hedge RMS, CNY"),
                ],
            ),
            "",
            "Direct same-contract covariance is unavailable below the declared complete-change minimum. Maturity pooling remains available at zero own history when donor maturity slots are observed. The pooled estimate blends with the short-history moment using n/(n+20), with n the available complete count.",
            "",
            "## Hedge and fixed sensitivity checks",
            "",
            "![Paired hedge outcomes across half-lives and block lengths](hedge-sensitivity.png)",
            "",
            "One target contract is hedged with the two selected other contracts, with each hedge position bounded to [−2,2]. Half-life and bootstrap-block sensitivities were fixed, not selected by evaluation performance. Repeated budgets are resampled jointly over calendar origins. No fills, roll mechanics, transaction costs, margin or capacity are modeled.",
            "",
            "## Availability and numerical scope",
            "",
            table(
                failures,
                [
                    ("method", "Policy"),
                    ("issued", "Attempted"),
                    ("available", "Issued"),
                    ("scored", "Scored"),
                    ("failures", "Unavailable reason counts"),
                ],
            ),
            "",
            "The full numerical diagnostics include source optimizer certificates, factor-boundary gaps, effective counts, Schur support/correction checks and bounded-cache measurements in [diagnostics.json](diagnostics.json). All fixed policies, maturity strata, component contributions and sensitivities remain in [summary.json](summary.json); paired ranges are in [paired.json](paired.json). No unavailable result is replaced by a fallback.",
            "",
            f"Run time: {manifest['runtime_seconds']:.1f} seconds, including masked preparation and model fitting, excluding archive acquisition. This is a single environment measurement. Environment and archive provenance: [manifest.json](manifest.json); frozen settings: [protocol.json](protocol.json).",
            "",
            "## Interpretation and reproduction",
            "",
            "The central test is risk transfer, not reconstruction of an already observed price. Success at a masked-history budget would support pooling across contract lifecycles under the tested conditions; it would not establish SPDE superiority, genuine new-listing performance or continuum identification. Four residual maturity features plus a diagonal prediction-error correction remain a restriction. The diagonal term can contain estimation error and is not identified latent noise.",
            "",
            "Read the [executed notebook](../../../notebooks/07_short_history_risk_transfer.ipynb) and [protocol](../../../docs/short_history.md). Raw exchange panels, contract-level forecasts and individual P&Ls remain local. This renderer verifies aggregate checksums and does not refit models or change numerical evidence.",
            "",
        ]
    )


def render(directory):
    directory = Path(directory)
    evidence = load_short_history_evidence(directory)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    artifacts = {}
    for filename, function in (
        ("scores-by-budget.png", plot_scores),
        ("risk-calibration.png", plot_calibration),
        ("hedge-sensitivity.png", plot_hedges),
    ):
        figure = function(evidence)
        figure.savefig(directory / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        artifacts[filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    report = directory / "report.md"
    report.write_text(report_text(evidence))
    artifacts["report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()
    notebook = ROOT / "notebooks/07_short_history_risk_transfer.ipynb"
    document = json.loads(notebook.read_text())
    notebook_source = [
        {"cell_type": cell["cell_type"], "source": "".join(cell["source"])}
        for cell in document["cells"]
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
                    json.dumps(notebook_source, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return report


def build_notebook(path=ROOT / "notebooks/07_short_history_risk_transfer.ipynb"):
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
            phase,
            budgets,
            primary,
            paired,
            interval,
            plot_scores,
            plot_calibration,
            plot_hedges,
        )
    )
    setup = f"""from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from spde_covariance.empirical_short_history import load_short_history_evidence

evidence = load_short_history_evidence(Path("../experiments/results/short-history"))
config = evidence["protocol"]["config"]
print("Numerical source:", evidence["manifest"]["code_revision"])
print("Clean numerical source:", not evidence["manifest"]["source_dirty"])
print("Target:", evidence["protocol"]["forecast_target"])
print("Primary comparison:", evidence["protocol"]["primary_comparison"])
plt.rcParams.update({{"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False}})
METHODS = {METHODS!r}
LABELS = {LABELS!r}
COLORS = {COLORS!r}

{helpers}
assert evidence["paired"]["joint_resampling"]
assert config["history_budgets"] == [0, 5, 20, None]
"""
    cells = [
        markdown("""# 07 · Short-history risk transfer

**Question:** given a currently quoted copper contract and only a few permitted
own price changes, can other contracts supply useful next-session risk information?

This compares learned maturity-dependent return factors with transport-SPDE source
estimation, both with the same residual-transfer adapter. It is an exploratory
information-restriction experiment on inspected daily settlements, not a genuine
new-listing experiment. The current settlement remains visible. Historical target
rows are removed before reconstruction, loading fitting, scaling and estimation.

[Protocol](../docs/short_history.md) · [Generated report](../experiments/results/short-history/report.md)
· [Project findings](00_project_findings.ipynb)
"""),
        code(setup),
        markdown(r"""## Information and financial targets

The 2022 initial period, 2023 development validation and inspected 2024–2025
evaluation are retained. Origins occur every five sessions, with three targets
near 90, 180 and 270 days. Own-history budgets are 0, 5, 20 and all available
changes in a 252-interval window. Missing observations never create bridged returns.

Both methods forecast the uncentered price-change second moment
$$B_t=\mathbb E_t[\Delta p_{t+1}\Delta p_{t+1}^{\top}].$$
For a five-tonne contract position a, the P&L second moment is
$$25a^{\top}B_ta.$$
A variance interpretation requires negligible conditional means.
The target/nearest spread and original near/middle/far butterfly preserve actual
contract identities. The principal comparison is between complete policies,
not a causal estimate of the benefit of transport alone.
"""),
        code("""rows = primary(evidence)
availability = [{"budget": r["budget"], "policy": LABELS[r["method"]], "scored": f"{r['scored']}/{r['episodes']}", "origins": r["origins"]} for r in rows if r["method"] in METHODS]
print(table(availability, [("budget", "Own history"), ("policy", "Policy"), ("scored", "Scored / attempted"), ("origins", "Origins")]))
assert all(r["scored"] <= r["available"] <= r["episodes"] for r in rows)
"""),
        markdown("""## Paired forecast scores by history and maturity

Negative QLIKE differences favor learned factors. Each difference uses the same
future observation and permitted history. Vertical lines show descriptive 95%
calendar-origin block-bootstrap percentile ranges. Targets are averaged within
origin before pooling; contracts and history budgets are not independent draws.
The primary block has four scheduled origins, approximately 20 trading sessions.
"""),
        code("""plot_scores(evidence)
plt.show()
comparison_rows = []
for exposure in ("outright", "spread", "butterfly"):
    for row in paired(evidence, exposure=exposure):
        comparison_rows.append({"budget": row["budget"], "exposure": exposure, "difference": row["difference"], "range": interval(row, config["bootstrap_block"]), "origins": row["paired_origins"]})
print(table(comparison_rows, [("budget", "Own history"), ("exposure", "Target"), ("difference", "Learned − transport"), ("range", "95% block range"), ("origins", "Paired origins")]))
"""),
        markdown("""## Calibration and residual contributions

The ratio below is forecast divided by realized second moment: values below one
indicate underprediction. Aggregate calibration does not establish conditional
calibration. The joint moment retains factor/residual cross terms. Four correlated
residual maturity features and a diagonal remainder remain a modeling restriction.
The diagonal correction is fitted from leave-one-donor-out predictive errors,
which can contain estimation error; it is not identified latent market noise.
"""),
        code("""plot_calibration(evidence)
plt.show()
components = []
for row in rows:
    if row["method"] in ("dfm_joint", "spde_joint"):
        values = row["metrics"]["spread"]
        components.append({"budget": row["budget"], "method": row["method"], "ratio": values.get("calibration_ratio"), "forecast": values.get("forecast"), "components": "; ".join(f"{name}: {value:.4g}" for name, value in values.get("components", {}).items())})
print(table(components, [("budget", "Own history"), ("method", "Policy"), ("ratio", "Forecast / realized"), ("forecast", "Spread CNY²"), ("components", "Component CNY²")]))
"""),
        markdown("""## Bounded hedges and prespecified sensitivities

One target contract is hedged with two positions bounded to [−2,2]. Negative
paired mean squared-P&L differences favor learned factors. Values are in
thousand CNY², not differences between RMS values. Half-lives 21, 63 and 126
sessions and block lengths 1, 2, 4 and 8 origins are fixed sensitivities.
No winner is selected from evaluation performance. Execution, rolls, costs,
margin and capacity are absent.
"""),
        code("""plot_hedges(evidence)
plt.show()
hedges = [{"budget": row["budget"], "method": row["method"], "rms": row["metrics"]["hedge"].get("realized_rms_cny"), "origins": row["origins"]} for row in rows if row["method"] in METHODS]
print(table(hedges, [("budget", "Own history"), ("method", "Policy"), ("rms", "Hedge RMS CNY"), ("origins", "Origins")]))
"""),
        markdown("""## Availability, numerical checks and interpretation

Convergence certificates concern numerical optimization, not statistical validity.
Unavailable methods receive no fallback. Small factor-boundary gaps indicate
unstable individual loading directions; predictive covariance is invariant under
rotations within a retained factor subspace. Native settlement observations do
not identify the missing historical target path.
"""),
        code("""diagnostics = evidence["diagnostics"]
print(table(diagnostics["methods"], [("method", "Policy"), ("issued", "Attempted"), ("available", "Available"), ("scored", "Scored"), ("failures", "Failure counts")]))
print("Cache:", diagnostics["cache"])
print("Runtime seconds:", evidence["manifest"]["runtime_seconds"])
print("Numerical diagnostics:")
for method in diagnostics["methods"]:
    print(LABELS.get(method["method"], method["method"]), method["numeric_min_median_max"])
assert evidence["manifest"]["local_record_policy"] == "contract-level prices, forecasts and P&Ls are not exported"
"""),
        markdown("""## Evidence boundary and reproduction

This notebook executes from checksummed public aggregates; raw settlements,
issued contract-level forecasts and individual P&Ls stay local. Every model
receives a masked data view and no future outcome. The inspected evaluation period
supports exploratory comparisons; it is not an uninspected confirmation sample.
Masked-history success would support risk transfer under these tested conditions,
not genuine new-listing performance, SPDE superiority or a pricing theorem.

Run `uv run python scripts/check_notebook.py notebooks/07_short_history_risk_transfer.ipynb`.
The [protocol](../docs/short_history.md) gives local-data and synthetic reproduction
commands, assumptions, fixed sensitivities and source provenance.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path, default=ROOT / "experiments/results/short-history"
    )
    parser.add_argument(
        "--build-notebook",
        action="store_true",
        help="rewrite the unexecuted aggregate-only notebook",
    )
    args = parser.parse_args()
    if not (args.directory / "checksums.json").exists():
        parser.error(
            "verified aggregate evidence is required; run/export benchmark-short-history first"
        )
    if args.build_notebook:
        print(build_notebook())
    print(render(args.directory))


if __name__ == "__main__":
    main()
