"""Figures and an honest operational report for completion forecasts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("rank2", "rank2_psd", "coherent")
LABELS = {"rank2": "Rank 2", "rank2_psd": "Rank 2 + PSD", "coherent": "Coherent"}
RULES = ("static", "fixed", "selected")


def _value(row, field):
    mean, se = row.get(field + "_mean"), row.get(field + "_mcse")
    if mean is None:
        return "unavailable"
    return f"{mean:.6g} ± {se:.3g}" if se is not None else f"{mean:.6g} (MCSE unavailable)"


def _style(ax):
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, output, name):
    fig.savefig(output / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(output, config, results):
    """Render every declared scenario/cover, without favorable-case selection."""
    output = Path(output)
    expected = {
        (s, c, m, r)
        for s in config.scenarios
        for c in config.covers
        for m in METHODS
        for r in RULES
    }
    actual = {(r["scenario"], r["cover"], r["method"], r["rule"]) for r in results["summary"]}
    if actual != expected:
        raise ValueError("forecast report requires complete declared case coverage")
    fig, axes = plt.subplots(
        len(config.scenarios),
        len(config.covers),
        squeeze=False,
        figsize=(6 * len(config.covers), 3.3 * len(config.scenarios)),
        layout="constrained",
    )
    for i, scenario in enumerate(config.scenarios):
        for j, cover in enumerate(config.covers):
            ax = axes[i, j]
            for position, method in enumerate(METHODS):
                row = next(
                    x
                    for x in results["paired_differences"]
                    if x["scenario"] == scenario
                    and x["cover"] == cover
                    and x["method"] == method
                    and x["rule"] == "selected"
                    and x["comparison"] == "static"
                )
                if row["observed_score_mean"] is None:
                    ax.text(position, 0, "unavailable", rotation=90, va="bottom", ha="center")
                else:
                    ax.errorbar(
                        position,
                        row["observed_score_mean"],
                        yerr=row["observed_score_mcse"],
                        marker="o",
                        capsize=3,
                    )
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(3), [LABELS[m] for m in METHODS])
            ax.set_title(f"{scenario}; {cover} fragments")
            ax.set_ylabel("Selected minus static observed score")
            _style(ax)
    fig.suptitle(
        "Paired replication means ± MCSE; lower favors the selected window\nEach method selects its own window from the same observed validation blocks"
    )
    _save(fig, output, "forecast-comparison.png")
    fig, axes = plt.subplots(
        len(config.scenarios),
        len(config.covers),
        squeeze=False,
        figsize=(6 * len(config.covers), 3.3 * len(config.scenarios)),
        layout="constrained",
    )
    for i, scenario in enumerate(config.scenarios):
        for j, cover in enumerate(config.covers):
            ax = axes[i, j]
            for rule in RULES:
                rows = sorted(
                    (
                        r
                        for r in results["horizon_summary"]
                        if r["scenario"] == scenario
                        and r["cover"] == cover
                        and r["method"] == "coherent"
                        and r["rule"] == rule
                    ),
                    key=lambda r: r["horizon"],
                )
                xs = [r["horizon"] for r in rows]
                ys = [
                    np.nan if r["kernel_error_mean"] is None else r["kernel_error_mean"]
                    for r in rows
                ]
                se = [0 if r["kernel_error_mcse"] is None else r["kernel_error_mcse"] for r in rows]
                ax.errorbar(xs, ys, yerr=se, label=rule, capsize=2)
            ax.set_title(f"{scenario}; {cover} fragments")
            ax.set_xlabel("Future weekly interval after frozen issue")
            ax.set_ylabel("Latent weighted kernel error")
            ax.set_ylim(bottom=0)
            if i == 0 and j == 0:
                ax.legend()
            _style(ax)
    fig.suptitle(
        "Coherent forecasts: generating-kernel diagnostics, mean ± MCSE\nEach point averages the declared forecast origins within replication first"
    )
    _save(fig, output, "forecast-horizons.png")
    targets = sorted({r["target"] for r in results["risk_summary"]})
    fig, axes = plt.subplots(
        len(config.scenarios),
        len(config.covers),
        squeeze=False,
        figsize=(6 * len(config.covers), 3.3 * len(config.scenarios)),
        layout="constrained",
    )
    for i, scenario in enumerate(config.scenarios):
        for j, cover in enumerate(config.covers):
            ax = axes[i, j]
            for offset, rule in enumerate(RULES):
                means, errors = [], []
                for target in targets:
                    row = next(
                        r
                        for r in results["risk_summary"]
                        if r["scenario"] == scenario
                        and r["cover"] == cover
                        and r["method"] == "coherent"
                        and r["rule"] == rule
                        and r["target"] == target
                    )
                    means.append(
                        np.nan
                        if row["risk_squared_error_mean"] is None
                        else row["risk_squared_error_mean"]
                    )
                    errors.append(
                        0
                        if row["risk_squared_error_mcse"] is None
                        else row["risk_squared_error_mcse"]
                    )
                ax.errorbar(
                    np.arange(len(targets)) + (offset - 1) * 0.15,
                    means,
                    yerr=errors,
                    marker="o",
                    ls="none",
                    capsize=2,
                    label=rule,
                )
            ax.set_xticks(range(len(targets)), [t.replace("_", "\n") for t in targets])
            ax.set_title(f"{scenario}; {cover} fragments")
            ax.set_ylabel("Latent risk MSE")
            ax.set_ylim(bottom=0)
            if i == 0 and j == 0:
                ax.legend()
            _style(ax)
    fig.suptitle(
        "Coherent point-functional risks: simulation MSE, mean ± MCSE\nLong spread is never jointly observed: its operational score is unavailable"
    )
    _save(fig, output, "forecast-risks.png")
    lines = [
        "# Covariance-completion forecasts on time-varying transport sources",
        "",
        "This fixed-grid follow-up combines existing completion strategies with the predeclared RQ3 frozen-block protocol. It tests whole strategies; no forecast improvement, identification result or new estimator theorem is an acceptance condition.",
        "",
        f"Seed `{config.seed}`; {config.replications} paired replications; {config.intervals} weekly intervals; forcing cutoff {config.reference_dimension}; base grid {config.grid}; fragment catalogues {list(config.covers)}; known point-noise variance {config.noise_variance}.",
        "",
        "## Targets and chronology",
        "",
        "The source has a fixed infinite power-law spectrum. Deterministic scale and first-two-mode rotations vary its covariance, held constant within each week. The target for week i is the transported normalized innovation covariance V_i, evaluated on the declared grid. These are clean normalized innovations observed through noisy fragments, rather than innovations reconstructed from sparse/noisy price levels. The forcing cutoff is a finite reference with an explicit remaining tail.",
        "",
        f"Static fits use intervals 1–{config.static_window}. Fixed fits use the last {config.fixed_window} observations. Each method separately selects among windows {list(config.windows)} at origins {list(config.validation_origins)}, on the following {config.horizon} observations per origin. Exact ties choose the longest window. No candidate is dropped for failure; if any predeclared validation window is unavailable, that method's selected rule is unavailable.",
        "",
        f"All validation outcomes precede test origins {list(config.test_origins)}. Each issue is frozen for the following horizons 1–{config.horizon}, with no use of intervening outcomes. These are predictions of each future weekly covariance, rather than the covariance of a multiweek accumulated return.",
        "",
        "One parent measurement pool precedes every method and cover within replication/scenario. A seed-shuffled fragment catalogue repeats cyclically, so every training window contains every declared clique. This is an explicit design choice, not a claim of design parity with older studies. Shared random numbers couple methods; independent replications define MCSE.",
        "",
        f"Historical rank-two raw completion and its separate weighted global PSD projection are compared with coherent partial fitting and full-support completion. Coherent η={config.relative_regularization} is fixed; the weighted-coordinate floor is γ/Q² with γ={config.floor_gamma:g} and actual augmented grid size Q. The numerical observed-partial optimizer-distance budget is {config.numerical_error_budget:g}. No additional hyperparameter tuning or forecast model is introduced.",
        "",
        "## Observed forecast scores",
        "",
        "![Paired forecast comparisons](forecast-comparison.png)",
        "",
        "The reduced observed quadratic score omits a forecast-independent measurement constant and may be negative. Lower differences favor the first rule. Comparisons use identical future fragments and known-noise correction; their magnitudes must not be ranked across fragment designs. The mean and MCSE below average future dates/origins within replication before averaging independent replications.",
        "",
        "| Scenario | Fragments | Method | Selected minus static score | Available pairs | Selected minus fixed score | Available pairs |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for scenario in config.scenarios:
        for cover in config.covers:
            for method in METHODS:
                rows = [
                    r
                    for r in results["paired_differences"]
                    if r["scenario"] == scenario
                    and r["cover"] == cover
                    and r["method"] == method
                    and r["rule"] == "selected"
                ]
                a = next(r for r in rows if r["comparison"] == "static")
                b = next(r for r in rows if r["comparison"] == "fixed")
                lines.append(
                    f"| {scenario} | {cover} | {method} | {_value(a, 'observed_score')} | {a['observed_score_available']}/{config.replications} | {_value(b, 'observed_score')} | {b['observed_score_available']}/{config.replications} |"
                )
    lines += [
        "",
        "If a method is unavailable in some replications, its reported mean/MCSE is conditional on the explicitly counted complete pairs. Such a comparison is not the unconditional performance of all declared replications. Baseline unavailability is retained and does not prevent acceptance of a fully accounted numerical experiment.",
        "",
        "## Latent diagnostics and risk-score availability",
        "",
        "![Forecast horizons](forecast-horizons.png)",
        "",
        "![Risk errors](forecast-risks.png)",
        "",
        "Kernel and risk errors use the generating finite-reference V_i only after forecasts are issued. They are simulation diagnostics, not selection inputs. A scalar risk proxy exists only on future dates whose one fragment contains the entire point-functional exposure. Long-spread risk therefore has latent MSE but no operational score; pooled coverage does not substitute for joint observation.",
        "",
        "| Scenario | Fragments | Exposure | Coherent selected risk MSE | Score-available replications |",
        "|---|---:|---|---:|---:|",
    ]
    for row in results["risk_summary"]:
        if row["method"] == "coherent" and row["rule"] == "selected":
            lines.append(
                f"| {row['scenario']} | {row['cover']} | {row['target']} | {_value(row, 'risk_squared_error')} | {row['risk_score_available']}/{config.replications} |"
            )
    lines += ["", "## Acceptance, provenance and limits", ""]
    for name, value in results["acceptance"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines += [
        "",
        f"Descriptive availability: {results['availability']['unavailable_fits']} unavailable fits; {results['availability']['unavailable_test_rows']} unavailable future score rows. These are retained, with no fallback.",
    ]
    lines += [
        "",
        f"All {results['fit_count']} issued fits and {results['evaluation_count']} future score rows are retained, including unavailable cases. Source/configuration manifests and replication checkpoints record provenance; successful worker results are assembled in seed order.",
        "",
        "The optimizer certificate controls only the fitted observed partial kernel. It does not control completion amplification, sampling, omitted source modes or future drift. Grid, forcing cutoff and the floor remain fixed in this prediction study; joint-resolution work is separate. Constant-source and changing-source scenarios can favor different windows, and selected windows need not win. No market calibration, statistical confidence interval or source-loading recovery is claimed.",
        "",
        "[Configuration](config.json) · [Manifest](manifest.json) · [Model and tails](model.json) · [Results](results.json) · [Fits](fits.json) · [Selections](selections.json) · [Validation](validation.csv) · [Future scores](forecasts.csv) · [Risks](risks.csv) · [Replication averages](replication_averages.csv) · [Paired differences](paired.csv)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
