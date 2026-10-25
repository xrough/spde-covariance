"""Plain Matplotlib and complete paired summaries for the hedge pilot."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .provenance import _jsonable


def _number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def write_hedging_report(output, config, control, population, results):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = []
    for index, (name, row) in enumerate(population["designs"].items()):
        labels.append(name.replace("_", " "))
        source = row["source_interval"]
        if source["available"]:
            interval = source["hedge_interval"]
            axes[0].plot(
                interval,
                [index, index],
                "o-",
                color="C0",
                label="source class" if index == 0 else None,
            )
            axes[1].bar(index, source["scalar_decisions"]["minimax_regret"], color="C0")
        free = row["free_psd_interval"]
        if "hedge_interval" in free:
            interval = free["hedge_interval"]
            axes[0].plot(
                interval,
                [index + 0.15, index + 0.15],
                "o-",
                color="0.6",
                label="PSD completion" if index == 0 else None,
            )
    axes[0].set(
        yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Optimal scalar hedge position"
    )
    axes[0].legend()
    axes[1].set(
        xticks=np.arange(len(labels)),
        xticklabels=["Fragments", "+3y left", "+0y right", "Complete"],
        ylabel="Population minimax regret",
    )
    figure.savefig(output / "identification.png", dpi=160)
    plt.close(figure)

    summaries = results["summary"]
    methods = sorted({row["method"] for row in summaries})
    for metric, filename, ylabel in (
        ("complete_audit_score", "future-risk.png", "Future complete-audit quadratic score"),
        ("position", "hedge-positions.png", "Issued hedge position"),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")
        for world in (0, 1):
            for column, design in enumerate(("fragments", "bridge", "complete")):
                axis = axes[world, column]
                for index, method in enumerate(methods):
                    cells = sorted(
                        (
                            row
                            for row in summaries
                            if row["world"] == world
                            and row["design"] == design
                            and row["method"] == method
                            and row[metric]["mean"] is not None
                        ),
                        key=lambda row: row["training_count"],
                    )
                    if not cells:
                        continue
                    x = np.array([row["training_count"] for row in cells], dtype=float)
                    y = [row[metric]["mean"] for row in cells]
                    error = [row[metric]["mcse"] or 0.0 for row in cells]
                    axis.errorbar(
                        x * (1 + (index - 2.5) * 0.015),
                        y,
                        yerr=error,
                        marker="o",
                        capsize=2,
                        label=method.replace("_", " "),
                    )
                axis.set_title(f"World {'−' if world == 0 else '+'}; {design}")
                axis.set_xscale("log", base=2)
                axis.set_xticks(config.training_counts, [str(n) for n in config.training_counts])
                if column == 0:
                    axis.set_ylabel(ylabel)
                if world == 1:
                    axis.set_xlabel("Training observations")
        handles, legend = axes[0, 2].get_legend_handles_labels()
        figure.legend(handles, legend, loc="outside lower center", ncols=3, fontsize=9)
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)

    pair = control["paired_decisions"]
    lines = [
        "# Identification-to-hedging pilot",
        "",
        "The fixed-source population control establishes material decision ambiguity. "
        "The sampling study compares predeclared hedge rules; it does not require any rule to win. "
        "There is no empirical futures hedging-performance claim.",
        "",
        f"Run: {config.replications} paired replications; training prefixes {config.training_counts}; "
        f"{config.heldout_count} complete future audit observations; seed {config.seed}.",
        "",
        "## Population identification",
        "",
        "Both generating sources are PSD, trace one and infinite rank. The perturbation is finite-supported; "
        "the common power-law tail is fixed. Optimization ranges condition on known trace of the uncertain "
        "source block, the fixed tail and the declared finite numerical map. They are identification ranges, "
        "not confidence intervals or uniform infinite-dimensional guarantees.",
        "",
        f"The explicit pair has optimal hedges {_number(pair['optimal_hedges'][0])} and "
        f"{_number(pair['optimal_hedges'][1])}. Using the other world's hedge adds variance "
        f"{_number(pair['wrong_hedge_excess'][0])}, or "
        f"{100 * pair['wrong_hedge_excess'][0] / pair['target_variances'][0]:.3f}% of unhedged target variance.",
        "",
        "| Observations | Source-class hedge interval | Minimax variance hedge | Minimax regret |",
        "|---|---:|---:|---:|",
    ]
    for name, row in population["designs"].items():
        interval = row["source_interval"]
        if interval["available"]:
            lo, hi = interval["hedge_interval"]
            decisions = interval["scalar_decisions"]
            lines.append(
                f"| {name} | [{lo:.6f}, {hi:.6f}] | {decisions['minimax_variance_hedge']:.6f} | {decisions['minimax_regret']:.8f} |"
            )
        else:
            lines.append(f"| {name} | unavailable | unavailable | unavailable |")
    lines.extend(
        [
            "",
            "The minimax-variance hedge and minimax-regret hedge solve different objectives. "
            "The latter is the interval midpoint in this unconstrained one-instrument population control. "
            "The sample study instead imposes a position limit. Canonical completion is a structural selection; "
            "its compatibility with the source class is not presumed.",
            "",
            "![Population hedge ranges](identification.png)",
            "",
            "## Past-only sampling study",
            "",
            "Native-fragment training observations are identical between worlds. Added observations and "
            "complete-panel controls are nested views of the same parent data. Each method sees only its "
            "training prefix. Held-out observations start after the largest prefix and are never used for "
            "selection. Gaussian bands are constructed without truth; their coverage requires centered "
            "independent common Gaussian rows, deterministic masks and fixed known fresh noise. "
            "The conditional fixed source-tail assumption is stronger than unrestricted covariance completion.",
            "",
            "The primary score is noise-corrected squared residual on a **complete future audit panel**. "
            "This evaluates controlled masking; it is not an operational score for an unobserved native "
            "cross-fragment exposure. Native score availability/counts are retained in estimates.json. "
            "Latent residual variance and regret are separate simulation diagnostics.",
            "",
            "Mean ± MCSE below uses independent replications. Worlds, dates and methods are paired, "
            "not additional independent replications. Means condition on available fits, and paired "
            "contrasts use their common available replications. Unavailable methods retain reasons in the fit audit.",
            "",
            "| World | Design | N | Method | Available | Future audit score ± MCSE | Latent residual variance |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        score, risk = row["complete_audit_score"], row["true_residual_variance"]
        lines.append(
            f"| {row['world']} | {row['design']} | {row['training_count']} | {row['method']} | {row['available']}/{row['replications']} | {_number(score['mean'])} ± {_number(score['mcse'])} | {_number(risk['mean'])} |"
        )
    lines.extend(
        [
            "",
            "![Future risk scores](future-risk.png)",
            "",
            "![Issued positions](hedge-positions.png)",
            "",
            "Paired differences relative to coherent canonical completion are in [paired.csv](paired.csv), "
            "separately for observed future scores and latent risk. Point-kernel observed/missing/complete "
            "errors and exposure errors remain in [estimates.json](estimates.json). A robust hedge is a "
            "decision, not a full kernel estimator. No off-grid extension is assumed for a grid completion.",
            "",
            "## Acceptance and limitations",
            "",
            "```json",
            json.dumps(_jsonable(results["acceptance"]), indent=2),
            "```",
            "",
            "A successful optimizer check is not an exact numerical certificate. Failed or inaccurate "
            "SDP solves remain unavailable. The small SHFE coverage audit establishes data/metadata "
            "accessibility only; no market model, execution cost or trading advantage has been calibrated. "
            "Simultaneous Gaussian bands do not yet cover dependent real futures-price changes. "
            "See the repository's docs/identified-hedging.md and docs/futures-data.md for scope.",
            "",
            "## Artifacts and provenance",
            "",
            "[Configuration](config.json), [manifest](manifest.json), [control and numerical checks](control.json), "
            "[population endpoint witnesses](population.json), [all estimates](estimates.json), "
            "[fit and band diagnostics](fit_audit.json), [results](results.json), "
            "[summary](summary.csv), [paired differences](paired.csv), [status](status.json).",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
