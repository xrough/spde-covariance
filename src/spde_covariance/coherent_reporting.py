"""Plain Matplotlib evidence for the PSD-consistent completion follow-up."""

import numpy as np


def write_coherent_report(config, output, results, examples, cone_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summary = results["summary"]
    q, variance = max(config.grids), max(config.noise_variances)
    policies = ("pairwise_rank2", "pairwise_rank2_psd", "coherent_fixed", "coherent_selected")
    labels = ("Pairwise rank 2", "Pairwise + PSD", "Coherent fixed", "Coherent selected")
    fig, axes = plt.subplots(
        len(config.models), 3, figsize=(12, 3.5 * len(config.models)), squeeze=False
    )
    for i, model in enumerate(config.models):
        for j, region in enumerate(("observed", "missing", "complete")):
            ax = axes[i, j]
            metric = region + "_weighted_error"
            for policy, label in zip(policies, labels, strict=True):
                rows = sorted(
                    (
                        r
                        for r in summary
                        if r["model"] == model
                        and r["base_grid"] == q
                        and r["cover"] == min(config.covers)
                        and r["noise_variance"] == variance
                        and r["policy"] == policy
                    ),
                    key=lambda r: r["N"],
                )
                ax.errorbar(
                    [r["N"] for r in rows],
                    [r[metric + "_mean"] for r in rows],
                    yerr=[r[metric + "_mcse"] or 0 for r in rows],
                    marker="o",
                    ms=3,
                    capsize=2,
                    label=label,
                )
            ax.set(
                xlabel="Training observations",
                ylabel="Weighted kernel error",
                title=f"{model}: {region}",
            )
            ax.set_xscale("log", base=2)
            ax.set_xticks(config.training_counts, [str(n) for n in config.training_counts])
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Base grid {q}; {min(config.covers)} fragments; noise variance {variance:g}; mean ± MCSE"
    )
    fig.tight_layout()
    fig.savefig(output / "kernel-errors.png", dpi=150)
    plt.close(fig)

    selected_examples = [
        e
        for e in examples
        if e["base_grid"] == q
        and e["model"] == config.models[0]
        and e["cover"] == max(config.covers)
    ]
    fig, axes = plt.subplots(1, len(policies), figsize=(13, 3), squeeze=False)
    scale = max((float(np.max(np.abs(e["kernel"]))) for e in selected_examples), default=1)
    for ax, policy, label in zip(axes[0], policies, labels, strict=True):
        record = next((e for e in selected_examples if e["policy"] == policy), None)
        if record is not None:
            image = ax.pcolormesh(
                record["nodes"],
                record["nodes"],
                record["kernel"],
                shading="auto",
                vmin=-scale,
                vmax=scale,
                cmap="RdBu_r",
            )
            ax.set(xlabel="Maturity (years)", ylabel="Maturity (years)", title=label)
    if selected_examples:
        fig.colorbar(image, ax=list(axes[0]), shrink=0.8, label="Innovation kernel")
    fig.suptitle("Illustrative replication zero: same observations for every method")
    fig.subplots_adjust(top=0.77, bottom=0.2, left=0.05, right=0.88, wspace=0.4)
    fig.savefig(output / "kernel-comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(config.models), figsize=(6 * len(config.models), 3.5), squeeze=False
    )
    for ax, model in zip(axes[0], config.models, strict=True):
        for target, label in (
            ("generating", "Generating kernel"),
            ("canonical", "Population canonical"),
        ):
            records = sorted(
                (
                    r
                    for r in cone_rows
                    if r["scope"] == "population finite grid"
                    and r["base_grid"] == q
                    and r["cover"] == min(config.covers)
                    and r["model"] == model
                    and r["target"] == target
                ),
                key=lambda r: r["J"],
            )
            ax.plot(
                [r["J"] for r in records],
                [r["distance_upper"] for r in records],
                "o-",
                label=label + " upper",
            )
            ax.plot(
                [r["J"] for r in records],
                [r["distance_lower"] for r in records],
                "--",
                label=label + " lower",
            )
        ax.set(
            xlabel="Declared constant-source dimension J",
            ylabel="Weighted kernel distance",
            title=model,
        )
        ax.set_xticks(config.admissibility_dimensions)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Finite source cone: base grid {q}, {min(config.covers)} fragments")
    fig.tight_layout()
    fig.savefig(output / "source-cone.png", dpi=150)
    plt.close(fig)

    lines = [
        "# PSD-consistent covariance completion: paired follow-up",
        "",
        "This study fits compatible PSD observed fragments jointly, then completes them using full numerical support. The historical pairwise rank-two estimator and its separate global PSD projection remain baselines. No superiority or identification claim is an acceptance condition.",
        "",
        f"Seed `{config.seed}`; {config.replications} paired replications; forcing cutoff {config.reference_dimension}; weekly intervals {config.delta:.8g}; training counts {list(config.training_counts)}; base grids {list(config.grids)}; covers {list(config.covers)}; noise variances {list(config.noise_variances)}.",
        "",
        "## Protocol and chronology",
        "",
        f"One parent observation pool is generated per replication/model before method, grid, noise and training-prefix views. Training ends at N; validation occupies positions {max(config.training_counts) + 1}–{max(config.training_counts) + config.validation_count}; the following {config.test_count} observations are held out. Selection sees corrected observed statistics only and does not refit on validation.",
        "",
        f"Relative partial penalties {list(config.partial_regularizations)} multiply the maximum training pair coverage. The fixed policy uses the candidate nearest 0.001; exact validation ties prefer the larger penalty. Weighted-coordinate eigenvalue floor {config.eigenvalue_floor:g}, support cutoff {config.support_rtol:g}, observed-partial optimizer-distance budget {config.numerical_error_budget:g}. The floor is a finite-grid regularizer, not a fixed continuum white-noise operator.",
        "",
        "The weighted fitting objective differs from the original-domain Frobenius validation score. Upstream penalties alter observed predictions, so validation can compare them. Hard completion ranks remain invisible to a score restricted to unchanged observed entries. Risk scores are available only when one future fragment measures the whole exposure jointly.",
        "",
        "## Kernel errors",
        "",
        "![Weighted kernel errors](kernel-errors.png)",
        "",
        "Means ± replication-level MCSE. Complete error is against the generating kernel; the machine-readable tables also separate error against its population canonical completion and the canonical-versus-generating discrepancy.",
        "",
        f"Representative base grid {q}, {min(config.covers)} fragments, noise {variance:g}:",
        "",
        "| Model | N | Policy | Observed error | Missing error | Complete error |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        if (
            row["base_grid"] == q
            and row["cover"] == min(config.covers)
            and row["noise_variance"] == variance
            and row["N"] in (min(config.training_counts), max(config.training_counts))
        ):

            def value(metric):
                return f"{row[metric + '_mean']:.5g} ± {row[metric + '_mcse'] or 0:.2g}"

            lines.append(
                f"| {row['model']} | {row['N']} | {row['policy']} | {value('observed_weighted_error')} | {value('missing_weighted_error')} | {value('complete_weighted_error')} |"
            )
    lines += [
        "",
        "![Representative kernels](kernel-comparison.png)",
        "",
        f"These kernel images use one common colour scale and the same observations: {config.models[0]}, base grid {q}, {max(config.covers)} fragments, N={config.training_counts[len(config.training_counts) // 2]}, noise {variance:g}, replication zero. Nonuniform augmented maturity coordinates are retained. This illustration does not replace the paired summaries.",
        "",
        "## Numerical feasibility and certificates",
        "",
    ]
    for key, value in results["acceptance"].items():
        lines.append(f"- `{key}`: `{value}`.")
    lines += [
        "",
        f"All {results['candidate_count']} declared partial candidates are retained in the audit. Their certified distance covers only the weighted observed-partial optimization problem. Completion inverse amplification, sampling, representation and source truncation errors are separate. The shared diagonal restoration is explicitly recorded; coherent completion does not apply a global PSD projection. The historical PSD baseline can modify observed entries and risks.",
        "",
        "Comparisons concern whole strategies: coherent fitting changes PSD constraints, upstream penalty, eigenvalue floor and completion support together. They do not isolate a causal benefit from PSD coherence alone. Reported runtime includes projection for the historical PSD baseline and all candidates for selected coherent fitting; the fixed strategy reports its single-candidate cost.",
        "",
        "## Constant-source SPDE diagnostic",
        "",
        "![Finite source-cone distances](source-cone.png)",
        "",
        "The diagnostic minimizes distance to kernels generated by a constant PSD source covariance in the declared finite Laguerre space and source trace cap. The returned source is an optimization witness, not an identified physical loading. The convex gap yields a distance interval even if the requested gap tolerance is not attained.",
        "",
        "The source-cone audit separately reports the best weighted symmetric kernel approximation in the evaluation-basis span (SVD relative threshold 1e-12). Total finite-class distance includes that representation mismatch as well as the source PSD/trace restrictions; it cannot be attributed entirely to a transport obstruction.",
        "",
        "| Two-dimensional control (interval one year) | Distance lower | Distance upper |",
        "|---|---:|---:|",
    ]
    for row in results["source_controls"]:
        lines.append(
            f"| {row['control']} | {row['distance_lower']:.8g} | {row['distance_upper']:.8g} |"
        )
    lines += [
        "",
        f"{results['cone_checks']} finite-grid diagnostics are saved; {results['cone_unconverged']} did not reach the requested objective-gap tolerance and retain their valid distance bounds. Population checks cover generating and canonical targets; empirical diagnostics use only the declared illustrative replication zero, middle training count and largest noise variance. They are not Monte Carlo evidence of a universal obstruction.",
        "",
        "A strictly positive lower distance rejects only the declared finite constant-source class with its trace cap and numerical integration map. It does not exclude an infinite-dimensional SPDE, a larger source support or a time-varying source. Grid/quadrature and omitted forcing uncertainty are not covered by the optimizer gap. The positive-definite two-dimensional control shows why PSD alone need not imply membership in this restricted source cone.",
        "",
        "## Artifacts and limits",
        "",
        "[Configuration](config.json) · [Source/environment manifest](manifest.json) · [Results](results.json) · [Candidates](candidates.json) · [Selections](selections.json) · [Estimates](estimates.csv) · [Risks](risks.csv) · [Paired differences](paired.csv) · [Source-cone audit](source_cone.json) · [Geometry](geometry.json) · [Representative kernels](representative_kernels.json)",
        "",
        "This focused follow-up preserves the historical 100-replication paper controls and 20-replication SPDE fragment evidence. It adds a PSD-consistent strategy and finite source-class diagnostics; it does not establish infinite-dimensional estimation rates, canonical identification, market calibration or forecast performance.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
