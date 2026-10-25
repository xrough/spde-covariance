"""Plain research figures and scope-aware statistical completion reporting."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

WORLD_LABELS = {
    "source_minus": "SPDE source −",
    "source_plus": "SPDE source +",
    "canonical_control": "Canonical Gaussian control",
}
METHOD_LABELS = {"gaussian_selected": "Graph Gaussian", "coherent_moment": "Coherent moments"}
COLORS = {"source_minus": "tab:blue", "source_plus": "tab:orange", "canonical_control": "tab:green"}
STYLES = {"gaussian_selected": "-", "coherent_moment": "--"}


def _number(value):
    return "unavailable" if value is None else f"{value:.5g}"


def _mean_se(value):
    return f"{_number(value['mean'])} ± {_number(value['mcse'])}"


def _figures(output, config, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = results["summary"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for world in WORLD_LABELS:
            for method in METHOD_LABELS:
                selected = sorted(
                    (r for r in summaries if r["world"] == world and r["method"] == method),
                    key=lambda r: r["training_count"],
                )
                selected = [r for r in selected if r["missing_error"]["mean"] is not None]
                for ax, metric in zip(axes, ("observed_error", "missing_error"), strict=True):
                    ax.errorbar(
                        [r["training_count"] for r in selected],
                        [r[metric]["mean"] for r in selected],
                        yerr=[r[metric]["mcse"] or 0 for r in selected],
                        color=COLORS[world],
                        linestyle=STYLES[method],
                        marker="o",
                        capsize=3,
                        label=f"{WORLD_LABELS[world]} · {METHOD_LABELS[method]}",
                    )
        for ax, title in zip(
            axes, ("Observed covariance entries", "Missing covariance entries"), strict=True
        ):
            ax.set(xlabel="Training observations", ylabel="Weighted kernel error", title=title)
            ax.set_xscale("log", base=2)
            ax.set_xticks(config.training_counts, [str(n) for n in config.training_counts])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=7.5, loc="best")
        fig.suptitle("Fixed true-point completion: means and replication MCSE")
        fig.savefig(output / "statistical_completion_errors.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for method in METHOD_LABELS:
            selected = sorted(
                (
                    r
                    for r in summaries
                    if r["world"] == "canonical_control"
                    and r["method"] == method
                    and r["native_nll_excess"]["mean"] is not None
                ),
                key=lambda r: r["training_count"],
            )
            axes[0].errorbar(
                [r["training_count"] for r in selected],
                [r["native_nll_excess"]["mean"] for r in selected],
                yerr=[r["native_nll_excess"]["mcse"] or 0 for r in selected],
                linestyle=STYLES[method],
                marker="o",
                capsize=3,
                label=METHOD_LABELS[method],
            )
        for world in WORLD_LABELS:
            for method in METHOD_LABELS:
                selected = sorted(
                    (
                        r
                        for r in summaries
                        if r["world"] == world
                        and r["method"] == method
                        and r["bridge_nll_excess"]["mean"] is not None
                    ),
                    key=lambda r: r["training_count"],
                )
                axes[1].errorbar(
                    [r["training_count"] for r in selected],
                    [r["bridge_nll_excess"]["mean"] for r in selected],
                    yerr=[r["bridge_nll_excess"]["mcse"] or 0 for r in selected],
                    color=COLORS[world],
                    linestyle=STYLES[method],
                    marker="o",
                    capsize=3,
                    label=f"{WORLD_LABELS[world]} · {METHOD_LABELS[method]}",
                )
        for ax, title in zip(
            axes,
            (
                "Native fragments: identical in every world",
                "Full-panel audit: previously missing pairs",
            ),
            strict=True,
        ):
            ax.set(xlabel="Training observations", ylabel="NLL minus truth-oracle NLL", title=title)
            ax.set_xscale("log", base=2)
            ax.set_xticks(config.training_counts, [str(n) for n in config.training_counts])
            ax.axhline(0, color="0.5", linewidth=0.7)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7.5, loc="best")
        fig.suptitle("Independent test scores; oracle subtraction is simulation-only")
        fig.savefig(output / "statistical_completion_scores.png", dpi=180)
        plt.close(fig)

        coverage = results["coverage"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        # First declared replications, never selected by their outcomes.
        display = coverage["records"][:30]
        for r in display:
            if not r["interval_available"]:
                axes[0].plot(0, r["replication"], marker="x", color="tab:red")
                continue
            axes[0].plot(
                [r["lower"], r["upper"]], [r["replication"]] * 2, color="0.65", linewidth=1
            )
        for index, (name, value) in enumerate(coverage["truth_risks"].items()):
            axes[0].axvline(value, label=f"{name}: {value:.2f}", linestyle="--", color=f"C{index}")
        population = coverage["population_identification_range"]
        axes[0].axvspan(
            population["lower"],
            population["upper"],
            color="tab:blue",
            alpha=0.08,
            label="Exact-partial identified range",
        )
        axes[0].set(
            xlabel="Variance of (X₀ − X₂)/√2",
            ylabel="Coverage replication index",
            title="PSD intervals: first 30 replications (× unavailable)",
        )
        axes[0].legend(fontsize=8, loc="best")
        entries = [("Observed\npartial", coverage["partial_coverage"])] + [
            (f"Risk\n{name}", row) for name, row in coverage["risk_coverage_all"].items()
        ]
        for i, (label, row) in enumerate(entries):
            if row["rate"] is not None:
                lo, hi = row["interval_95"]
                axes[1].errorbar(
                    i,
                    row["rate"],
                    yerr=[[row["rate"] - lo], [hi - row["rate"]]],
                    marker="o",
                    capsize=5,
                    color="tab:blue",
                )
        axes[1].axhline(
            config.confidence_level,
            linestyle="--",
            color="0.3",
            label=f"Nominal {config.confidence_level:.0%}",
        )
        axes[1].set(
            xticks=np.arange(len(entries)),
            xticklabels=[x[0] for x in entries],
            ylabel="Coverage proportion",
            ylim=(0, 1.025),
            title="Independent replications; exact binomial intervals",
        )
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.2)
        strata = coverage.get("resolution_study", (coverage,))
        for design, color in (("native", "tab:blue"), ("complete", "tab:orange")):
            cells = sorted(
                (r for r in strata if r["design"] == design), key=lambda r: r["training_count"]
            )
            cells = [r for r in cells if r["mean_width"] is not None]
            if cells:
                axes[2].errorbar(
                    [r["training_count"] for r in cells],
                    [r["mean_width"] for r in cells],
                    yerr=[r["width_mcse"] or 0 for r in cells],
                    marker="o",
                    capsize=3,
                    label=design,
                    color=color,
                )
        axes[2].axhline(
            population["upper"] - population["lower"],
            color="tab:blue",
            linestyle="--",
            label="Native exact-partial width",
        )
        axes[2].set(
            xlabel="Training observations",
            ylabel="Mean risk-interval width",
            title="Nested samples and added joint observations",
        )
        axes[2].set_xscale("log", base=2)
        counts = sorted({r["training_count"] for r in strata})
        axes[2].set_xticks(counts, [str(n) for n in counts])
        axes[2].set_ylim(bottom=0)
        axes[2].grid(alpha=0.2)
        axes[2].legend(fontsize=8)
        fig.suptitle("Identification-inclusive uncertainty; canonical and noncanonical worlds")
        fig.savefig(output / "statistical_completion_uncertainty.png", dpi=180)
        plt.close(fig)


def write_statistical_completion_report(output, config, results):
    output = Path(output)
    _figures(output, config, results)
    population, coverage = results["population"], results["coverage"]
    lines = [
        "# Statistical completion at fixed true points",
        "",
        "This study separates observed covariance estimation, a canonical completion convention, and uncertainty over all PSD completions. It makes no empirical-market or hedge-performance claim.",
        "",
        f"Acceptance: **{'PASS' if results['acceptance']['passed'] else 'FAIL'}**. There is no superiority or nominal-coverage pass/fail gate. Failed candidates: {results['failed_candidate_count']}; unavailable selected world/method rows: {results['unavailable_selected_count']}.",
        "",
        "## Observation model and chronology",
        "",
        f"Seven fixed true point maturities {list(np.asarray(population['grid']))}; two native cliques [0,1,2,3] and [3,4,5,6]. The target is a normalized innovation point kernel at Δ={config.delta:.8g} years. Source −/+ use a PSD perturbation of a fixed trace-one power-law source, with a common tail integrated through {config.reference_dimension} modes. Point evaluation occurs before estimator projection. The canonical Gaussian control completes their shared observed partial kernel; no SPDE-source membership is asserted for this control.",
        "",
        f"Each of {config.replications} independent point-study replications uses nested training counts {config.training_counts}. Validation uses [{max(config.training_counts)}, {max(config.training_counts) + config.validation_count}); the untouched full-panel test uses the following {config.test_count} observations. Native fragment rows alternate by date. No test or latent truth enters fitting, the training diagonal reference, or penalty selection. Complete test observations are a separately disclosed measurement augmentation, not native risk availability.",
        "",
        f"The known fresh Gaussian observation-noise variance is {config.noise_variance}. Conditional Gaussian parent coupling makes native data exactly equal across all three worlds: **{results['native_inputs_identical']}**. Population native-block discrepancy: {_number(population['observed_law_max_difference'])}. Fits, selections and issued kernels are consequently shared, not independently retuned using each world's hidden entries.",
        "",
        "The Gaussian model has zeros in the precision entries outside the fixed two-clique graph. It is fitted by multistart graph-constrained EM and has a local stationarity criterion, not a global optimization certificate. The coherent-moment comparator uses the existing shared PSD-clique objective followed by full-support canonical completion. Both use the same training-only positive diagonal reference and candidate pseudo-counts; their KL and Frobenius penalties remain different objectives. A selected failed fit is unavailable, and failed candidates remain in the archive.",
        "",
        f"Pseudo-count candidates {config.prior_counts} produce Gaussian λ=pseudo-count/N. The coherent λ also multiplies maximum normalized pair coverage. The positive training variance floor is {config.reference_variance_floor} times the largest absolute corrected training variance; the applied floor is recorded. Coherent weighted eigenvalue floor: {config.moment_eigenvalue_floor:g}; observed-partial optimizer-distance budget: {config.numerical_error_budget:g}. Those numerical bounds do not bound completed-kernel or sampling error.",
        "",
        "Selection minimizes native validation mean marginal Gaussian negative log likelihood, including each measurement dimension's normalizing constant. Ties choose the larger pseudo-count. The native graph stays fixed for full-panel scoring. Native scores cannot test the unobserved conditional-independence convention. Full-panel audit data can reveal its consequences; oracle subtraction below is a simulation-only diagnostic that removes different world entropies.",
        "",
        "## Population identification and structural misspecification",
        "",
        "| World | Canonical weighted kernel error | Canonical full-panel Gaussian KL |",
        "|---|---:|---:|",
    ]
    for world in WORLD_LABELS:
        lines.append(
            f"| {WORLD_LABELS[world]} | {_number(population['canonical_structural_error'][world])} | {_number(population['canonical_bridge_gaussian_kl'][world])} |"
        )
    lines.extend(
        [
            "",
            "These are deterministic population diagnostics, not fitted sampling performance. Native population covariances are equal while missing-block truths differ.",
            "",
            "| Fixed exposure | Exact-partial PSD lower | Upper | Source − / + / canonical truth |",
            "|---|---:|---:|---|",
        ]
    )
    for row in population["identification_ranges"]:
        truths = " / ".join(_number(row["world_values"][world]) for world in WORLD_LABELS)
        lines.append(
            f"| {row['target']} | {_number(row['lower'])} | {_number(row['upper'])} | {truths} |"
        )
    lines.extend(
        [
            "",
            "Exposure coefficients are normalized using the full weighted-Sobolev evaluation-representer Gram. These algebraic two-block ranges concern all PSD grid completions of exact observed entries. They are neither confidence intervals nor ranges restricted to transport sources.",
            "",
            "## Point estimation and independent scores",
            "",
            "![Observed and missing errors](statistical_completion_errors.png)",
            "",
            "Errors use the same normalized physical trapezoidal grid weights in every world/method. These grid-kernel norms are distinct from H-coordinate operator norms. Means ± MCSE use independent replication clusters, not dates, nodes or method/world rows.",
            "",
            "| World | N | Method | Available | Observed error | Missing error | Full kernel error |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for r in results["summary"]:
        lines.append(
            f"| {WORLD_LABELS[r['world']]} | {r['training_count']} | {METHOD_LABELS[r['method']]} | {r['available']}/{r['replications']} | {_mean_se(r['observed_error'])} | {_mean_se(r['missing_error'])} | {_mean_se(r['weighted_kernel_error'])} |"
        )
    lines.extend(
        [
            "",
            "![Native and full-panel audit likelihood](statistical_completion_scores.png)",
            "",
            "| World | N | Method | Native NLL | Full-panel NLL | Full-panel excess over truth oracle |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for r in results["summary"]:
        lines.append(
            f"| {WORLD_LABELS[r['world']]} | {r['training_count']} | {METHOD_LABELS[r['method']]} | {_mean_se(r['native_nll'])} | {_mean_se(r['bridge_nll'])} | {_mean_se(r['bridge_nll_excess'])} |"
        )
    lines.extend(
        [
            "",
            "Raw likelihoods for different measurement dimensions or different truth distributions are not directly interchangeable. Within-world paired method differences cancel the same measurement-only/oracle terms. Quadratic scores omit the measurement-only squared-product constant; all are archived.",
            "",
            "| World | N | Paired metric: Gaussian minus moments | Mean difference ± MCSE | Paired replications |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for r in results["paired"]:
        lines.append(
            f"| {WORLD_LABELS[r['world']]} | {r['training_count']} | {r['metric']} | {_number(r['mean'])} ± {_number(r['mcse'])} | {r['replications']} |"
        )
    lines.extend(
        [
            "",
            "## Fixed exposures and scoring availability",
            "",
            "| World | N | Method | Exposure | Risk MSE ± MCSE | Native test rows per replication |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for world in WORLD_LABELS:
        for count in config.training_counts:
            for method in METHOD_LABELS:
                selected = [
                    r
                    for r in results["rows"]
                    if r["world"] == world
                    and r["training_count"] == count
                    and r["method"] == method
                    and r["available"]
                ]
                for target in population["exposures"]:
                    values = [r["risks"][target]["squared_error"] for r in selected]
                    mean = float(np.mean(values)) if values else None
                    se = (
                        float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else None
                    )
                    counts = sorted({r["risks"][target]["native_count"] for r in selected})
                    lines.append(
                        f"| {WORLD_LABELS[world]} | {count} | {METHOD_LABELS[method]} | {target} | {_number(mean)} ± {_number(se)} | {counts} |"
                    )
    lines.extend(
        [
            "",
            "No same-date native proxy exists for cross-fragment exposures even when a completion predicts them. Full-panel noise-corrected risk scores are separate audit measurements, archived in every point row.",
            "",
            "## Independent uncertainty experiment",
            "",
            "![Identification-inclusive intervals and coverage](statistical_completion_uncertainty.png)",
            "",
            "The separate three-node Gaussian control observes cliques [0,1] and [1,2]. Canonical and noncanonical covariances share both observed blocks; their missing covariance is 0.16 versus 0.65. The variance of (X₀−X₂)/√2 is 0.84 versus 0.35. The graph Gaussian convention does not identify which world generated native data.",
            "",
            f"Each of {config.coverage_replications} independent replications uses {config.coverage_training_count} observations and known fresh Gaussian noise variance {config.coverage_noise_variance}. The exact Gaussian scalar variance intervals for node/sum/difference statistics share a single Bonferroni error budget, yielding a simultaneous observed-moment region at nominal level {config.confidence_level:.0%}. Its PSD risk interval includes identification ambiguity; it is not a confidence interval for the canonical point completion alone.",
            "",
            f"Nominal coverage-proportion MCSE is {_number(coverage['nominal_coverage_mcse'])}; target {_number(coverage['precision_target'])}, met: **{coverage['precision_goal_met']}**. Smoke settings may intentionally miss that precision target. No observed coverage percentage is a pass/fail gate. Exact binomial 95% intervals describe Monte Carlo uncertainty.",
            "",
            "| Coverage quantity | Success / denominator | Rate | MCSE | Exact binomial interval |",
            "|---|---:|---:|---:|---|",
        ]
    )
    entries = (
        [("Simultaneous observed partial", coverage["partial_coverage"])]
        + [
            (f"Risk {name}; unavailable counted as misses", row)
            for name, row in coverage["risk_coverage_all"].items()
        ]
        + [
            (f"Risk {name}; available intervals only", row)
            for name, row in coverage["risk_coverage_available"].items()
        ]
    )
    for label, row in entries:
        bounds = (
            " / ".join(_number(x) for x in row["interval_95"])
            if row["interval_95"]
            else "unavailable"
        )
        lines.append(
            f"| {label} | {row['successes']}/{row['count']} | {_number(row['rate'])} | {_number(row['mcse'])} | {bounds} |"
        )
    if "resolution_study" in coverage:
        lines.extend(
            [
                "",
                "Nested sample-size and observation-design controls use the same parent observations per replication. Each stratum has its own Monte Carlo summary; counts/designs are paired, not independent extra replications. Complete views are generated under the canonical covariance and therefore assess coverage for that world only. Their law distinguishes the noncanonical alternative.",
                "",
                "| Design | N | Available / replications | Partial coverage | Canonical-risk coverage | Noncanonical-risk coverage | Mean width ± MCSE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for stratum in coverage["resolution_study"]:
            risks = stratum["risk_coverage_all"]
            noncanonical = risks.get("noncanonical", {}).get("rate")
            lines.append(
                f"| {stratum['design']} | {stratum['training_count']} | {stratum['interval_available_count']}/{config.coverage_replications} | {_number(stratum['partial_coverage']['rate'])} | {_number(risks['canonical']['rate'])} | {_number(noncanonical) if noncanonical is not None else 'not applicable'} | {_number(stratum['mean_width'])} ± {_number(stratum['width_mcse'])} |"
            )
    lines.extend(
        [
            "",
            f"Available numerical intervals: {coverage['interval_available_count']}; unavailable: {coverage['interval_unavailable_count']}; mean width {_number(coverage['mean_width'])}. The exact-partial population risk range is [{_number(coverage['population_identification_range']['lower'])}, {_number(coverage['population_identification_range']['upper'])}]. It persists with infinite data. Numerical SDP endpoints have solver/residual checks, not exact arithmetic certificates; coverage rows retain endpoint status and tolerance.",
            "",
            "## Numerical audit, selection and artifacts",
            "",
            "| Method | Selected pseudo-count | Number of issued fits |",
            "|---|---:|---:|",
        ]
    )
    selections = Counter(
        (r["method"], r["selected_prior_count"]) for r in results["issued"] if r["available"]
    )
    for (method, pseudo), count in sorted(selections.items()):
        lines.append(f"| {METHOD_LABELS[method]} | {pseudo:g} | {count} |")
    gaussian = [
        r["diagnostics"]
        for r in results["candidates"]
        if r["method"] == "gaussian_selected" and "diagnostics" in r
    ]
    moment = [
        r["diagnostics"]
        for r in results["candidates"]
        if r["method"] == "coherent_moment" and "diagnostics" in r
    ]
    lines.extend(
        [
            "",
            f"Maximum candidate iterations: Gaussian {max((r.get('iterations', 0) for r in gaussian), default=0)}, coherent {max((r.get('iterations', 0) for r in moment), default=0)}. Maximum coherent observed-partial error bound: {_number(max((r.get('partial_error_bound', 0) for r in moment), default=0))}. Total descriptive runtime: {results['runtime_seconds']:.1f} seconds on the recorded environment; this is not a hardware-independent speed claim.",
            "",
            "Convergence, local stationarity, coherent objective gaps, actual regularization, reference floors, multistart attempts, PSD/conditioning diagnostics, all candidate failures and selected kernels remain in the raw artifacts. Gaussian local convergence does not prove a global optimum. Source-tail/integration checks are separate fixed-source numerical diagnostics, not calibrated confidence statements or uniform infinite-class exclusion.",
            "",
            "- [Configuration](config.json), [source/environment manifest](manifest.json), [status](status.json).",
            "- [Population controls and exact-partial ranges](population.json).",
            "- [All point rows](point_rows.json), [candidate fits](candidates.json), [issued kernels and chronology](issued_kernels.json).",
            "- [All independent coverage rows](coverage.json), [complete results](results.json).",
            "",
            "The experiment requires independent centered Gaussian innovations and independent known Gaussian measurement noise. Neither these controls nor observed validation establish robustness to unknown noise, informative missingness, serial dependence, non-Gaussian observations or nonstationarity. Future curve calibration and hedging require additional observation and decision models.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
