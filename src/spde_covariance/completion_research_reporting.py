"""Plain scientific figures for completion ablation and refinement evidence.

The reporter consumes saved summaries and population audits only. It performs no
fitting, candidate selection, simulation or statistical reaggregation.
"""

import csv
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("report inputs must use None for unavailable, not nonfinite values")
    return value


def _save_table(output, stem, rows):
    data = _jsonable(rows)
    (output / (stem + ".json")).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    fields = list(dict.fromkeys(key for row in data for key in row))
    with (output / (stem + ".csv")).open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
                for row in data
            )


def _number(value):
    return "unavailable" if value is None else f"{value:.5g}"


def _mean(row, metric):
    if row is None or row.get(metric + "_mean") is None:
        return "unavailable"
    value = _number(row[metric + "_mean"])
    error = row.get(metric + "_mcse")
    return value + (f" ± {error:.2g}" if error is not None else " (MCSE unavailable)")


def _curve(ax, rows, x, metric, *, label, marker="o", color=None):
    valid = sorted(
        (row for row in rows if row.get(x) is not None and row.get(metric + "_mean") is not None),
        key=lambda row: row[x],
    )
    if not valid:
        return
    ax.errorbar(
        [row[x] for row in valid],
        [row[metric + "_mean"] for row in valid],
        yerr=[row.get(metric + "_mcse") or 0 for row in valid],
        marker=marker,
        markersize=3,
        linewidth=1,
        capsize=2,
        label=label,
        color=color,
    )


def _population_curves(ax, rows, *, x, y, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        if row.get(x) is not None and row.get(y) is not None:
            grouped[tuple(row.get(field) for field in group_fields)].append(row)
    for key, records in grouped.items():
        # Identical repeated records do not imply independent measurements.
        values = sorted({(row[x], row[y]) for row in records})
        label = ", ".join(
            f"{field}={value}"
            for field, value in zip(group_fields, key, strict=True)
            if value is not None
        )
        ax.plot([v[0] for v in values], [v[1] for v in values], "o-", ms=3, lw=1, label=label or y)


def _ablation_figure(plt, config, output, summary):
    panels = [(model, cover) for model in config.models for cover in config.covers]
    policies = list(dict.fromkeys(row["policy"] for row in summary))
    q = max(config.grids)
    n = config.training_counts[len(config.training_counts) // 2]
    central = [row for row in summary if row["base_grid"] == q and row["N"] == n]
    positive = []
    for row in central:
        mean, error = (
            row.get("complete_weighted_error_mean"),
            row.get("complete_weighted_error_mcse") or 0,
        )
        if mean is not None and mean > 0:
            positive.append(mean)
            if mean - error > 0:
                positive.append(mean - error)
    lower_limit = min(positive, default=1e-8) / 2
    upper_limit = (
        max(
            (
                row["complete_weighted_error_mean"] + (row.get("complete_weighted_error_mcse") or 0)
                for row in central
                if row.get("complete_weighted_error_mean") is not None
            ),
            default=1.0,
        )
        * 1.2
    )
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(7, 4 * len(panels)), max(9, 0.35 * len(policies) + 1.2)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for number, ((model, cover), ax) in enumerate(zip(panels, axes[0], strict=True)):
        for position, policy in enumerate(policies):
            row = next(
                (
                    row
                    for row in central
                    if row["model"] == model and row["cover"] == cover and row["policy"] == policy
                ),
                None,
            )
            mean = None if row is None else row.get("complete_weighted_error_mean")
            if mean is None:
                ax.text(
                    lower_limit * 1.2, position, "unavailable", va="center", fontsize=7, color="0.4"
                )
            elif mean <= 0:
                ax.text(
                    lower_limit * 1.2, position, "0 (outside log axis)", va="center", fontsize=7
                )
            else:
                error = row.get("complete_weighted_error_mcse") or 0
                left = max(mean - error, lower_limit)
                ax.errorbar(
                    mean,
                    position,
                    xerr=[[mean - left], [error]],
                    fmt="o",
                    markersize=3,
                    linewidth=1,
                    capsize=2,
                    color="tab:blue",
                )
        ax.set(
            xscale="log",
            xlim=(lower_limit, max(upper_limit, lower_limit * 10)),
            xlabel="Weighted kernel error",
            title=f"{model}, {cover} fragments\nBase grid {q}, N={n}",
        )
        ax.set_yticks(np.arange(len(policies)), policies, fontsize=7)
        ax.tick_params(axis="y", labelleft=number == 0)
        ax.grid(axis="x", alpha=0.2)
    axes[0, 0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "empirical-errors.png", dpi=150)
    plt.close(fig)


def _empirical_figure(plt, config, output, summary):
    if config.experiment == "ablation":
        _ablation_figure(plt, config, output, summary)
        return
    panels = [(model, cover) for model in config.models for cover in config.covers]
    policies = list(dict.fromkeys(row["policy"] for row in summary))
    legend_height = 0.22 * int(np.ceil(len(policies) / 3)) + 0.15
    height = 3.0 * len(panels) + legend_height
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, height), squeeze=False)
    largest_q, largest_n = max(config.grids), max(config.training_counts)
    for (model, cover), row_axes in zip(panels, axes, strict=True):
        for policy_index, policy in enumerate(policies):
            records = [
                row
                for row in summary
                if row["model"] == model and row["cover"] == cover and row["policy"] == policy
            ]
            _curve(
                row_axes[0],
                [row for row in records if row["base_grid"] == largest_q],
                "N",
                "complete_weighted_error",
                label=policy,
                color=plt.get_cmap("tab10")(policy_index % 10),
            )
            _curve(
                row_axes[1],
                [row for row in records if row["N"] == largest_n],
                "base_grid",
                "complete_weighted_error",
                label=policy,
                color=plt.get_cmap("tab10")(policy_index % 10),
            )
        for ax, xlabel, title in (
            (
                row_axes[0],
                "Training observations",
                f"{model}, {cover} fragments; base grid {largest_q}",
            ),
            (row_axes[1], "Base maturity-grid nodes", f"{model}, {cover} fragments; N={largest_n}"),
        ):
            ax.set(
                xlabel=xlabel,
                ylabel="Weighted error to generating kernel",
                title=title,
                ylim=(0, None),
            )
            ax.grid(alpha=0.2)
        row_axes[0].set_xticks(config.training_counts)
        row_axes[1].set_xticks(config.grids)
    legend = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend.setdefault(label, handle)
    if legend:
        fig.legend(
            legend.values(),
            legend,
            loc="upper center",
            ncol=min(3, len(legend)),
            fontsize=8,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 1 - legend_height / height))
    fig.savefig(output / "empirical-errors.png", dpi=150)
    plt.close(fig)


def _population_figure(plt, config, output, population):
    fig, axes = plt.subplots(
        len(config.models), 3, figsize=(13, 3.3 * len(config.models)), squeeze=False
    )
    for model, row_axes in zip(config.models, axes, strict=True):
        records = [row for row in population if row["model"] == model]
        views = (
            (
                "M",
                "forcing_error",
                ("base_grid", "cover", "J"),
                "Forcing cutoff M",
                "Forcing-reference error",
            ),
            (
                "J",
                "representation_error",
                ("base_grid", "cover", "M"),
                "Representation dimension J",
                "Kernel representation error",
            ),
            (
                "base_grid",
                "canonical_reference_error",
                ("cover", "M", "J"),
                "Base maturity-grid nodes",
                "Canonical reference-grid error",
            ),
        )
        for ax, (x, metric, groups, xlabel, ylabel) in zip(row_axes, views, strict=True):
            _population_curves(ax, records, x=x, y=metric, group_fields=groups)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=model, ylim=(0, None))
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=7, frameon=False)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Not configured / unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
    fig.tight_layout()
    fig.savefig(output / "population-refinement.png", dpi=150)
    plt.close(fig)


def _source_figure(plt, config, output, source_rows):
    panels = [(model, cover) for model in config.models for cover in config.covers]
    dimensions = tuple(config.source_dimensions) or (None,)
    fig, axes = plt.subplots(
        len(panels),
        len(dimensions),
        figsize=(5.2 * len(dimensions), 3.2 * len(panels)),
        squeeze=False,
    )
    for i, (model, cover) in enumerate(panels):
        for j, output_dimension in enumerate(dimensions):
            ax = axes[i, j]
            groups = defaultdict(list)
            for row in source_rows:
                if row["model"] == model and row["cover"] == cover and row["J"] == output_dimension:
                    groups[row["target"]].append(row)
            for target, records in groups.items():
                records = sorted(records, key=lambda row: row["M"])
                valid = [
                    row
                    for row in records
                    if row.get("distance_lower") is not None
                    and row.get("distance_upper") is not None
                ]
                if not valid:
                    continue
                label = target
                (line,) = ax.plot(
                    [row["M"] for row in valid],
                    [row["distance_upper"] for row in valid],
                    "o-",
                    ms=3,
                    lw=1,
                    label=label + " upper",
                )
                ax.plot(
                    [row["M"] for row in valid],
                    [row["distance_lower"] for row in valid],
                    ":",
                    lw=1,
                    color=line.get_color(),
                    label=label + " lower",
                )
                nested = [row for row in valid if row.get("nested_witness_upper") is not None]
                if nested:
                    ax.plot(
                        [row["M"] for row in nested],
                        [row["nested_witness_upper"] for row in nested],
                        "--",
                        lw=1,
                        color=line.get_color(),
                        label=label + " nested witness",
                    )
            ax.set(
                xlabel="Forcing dimension M (output J fixed)",
                ylabel="Weighted kernel distance",
                title=f"{model}, {cover} fragments, J={output_dimension}"
                if output_dimension is not None
                else f"{model}, {cover} fragments: source study skipped",
                ylim=(0, None),
            )
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=7, frameon=False)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Not configured / unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
    fig.tight_layout()
    fig.savefig(output / "source-nesting.png", dpi=150)
    plt.close(fig)


def write_completion_research_report(config, output, results, population, source_rows):
    """Write tables and three figures without consulting raw paths or truth selection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = results["summary"]
    paired = results["paired_differences"]
    stability = results["stability_summary"]
    perturbations = results.get("separator_perturbations", ())
    # Validate serialization before writing any report artifacts.
    for value in (
        summary,
        paired,
        stability,
        population,
        source_rows,
        results["acceptance"],
        results.get("risk_summary", ()),
        results.get("source_controls", ()),
        perturbations,
    ):
        _jsonable(value)
    for records in (summary, paired, stability, population, source_rows):
        for row in records:
            if row.get("model") is not None and row["model"] not in config.models:
                raise ValueError("report row has an undeclared model")
            if row.get("cover") is not None and row["cover"] not in config.covers:
                raise ValueError("report row has an undeclared fragment cover")
            permitted_grids = config.population_grids if records is population else config.grids
            if row.get("base_grid") is not None and row["base_grid"] not in permitted_grids:
                raise ValueError("report row has an undeclared maturity grid")
    for stem, records in (
        ("completion_summary", summary),
        ("paired_summary", paired),
        ("stability_summary", stability),
        ("risk_summary", results.get("risk_summary", ())),
        ("population", population),
        ("source_nesting", source_rows),
        ("separator_perturbations", perturbations),
    ):
        _save_table(output, stem, records)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        _empirical_figure(plt, config, output, summary)
        _population_figure(plt, config, output, population)
        _source_figure(plt, config, output, source_rows)
    profile = config.experiment
    middle_n = config.training_counts[len(config.training_counts) // 2]
    largest_q = max(config.grids)
    empirical_caption = (
        f"Ablation dot plots fix the declared central case: base grid {largest_q}, N={middle_n}; every configured model/cover panel uses the same policy rows and logarithmic error axis. Points show mean weighted error and horizontal intervals show replication-level MCSE where available. An interval reaching zero is clipped to the common positive plotting limit; zero means and unavailable results are labelled explicitly. Exact means and MCSE are retained in the tables."
        if config.experiment == "ablation"
        else "The left column fixes the largest declared base grid; the right fixes the largest declared training count. Error bars show replication-level MCSE where available; a missing MCSE is not zero sampling uncertainty. Curves with only one configured point are controls, not refinement evidence."
    )
    lines = [
        "# Covariance completion: ablations and refinement",
        "",
        f"Declared profile: `{profile}`. Seed `{config.seed}`; {config.replications} paired replications; base grids {list(config.grids)}; training counts {list(config.training_counts)}; fragment covers {list(config.covers)}; noise variance {config.noise_variance:g}.",
        "",
        "The ablation profile compares explicit partial-estimation, floor and completion choices. The refinement profile studies changes in grid, representation and forcing cutoff. These questions have different targets; a favorable empirical error is not a population convergence proof.",
        "",
        "## Information and numerical boundaries",
        "",
        "Every strategy uses the same supplied training observations for its declared case. Partial means include the known fresh point-noise correction and may be indefinite. Upstream shrinkage and coherent PSD constraints can change observed entries; support/rank completion preserves those fitted entries. Selection uses observed validation scores and any predeclared training-split admissibility guard. No latent error, source label, held-out test score or future origin selects a candidate.",
        "",
        "Observed validation is blind to differences confined to never-observed pairs. Training split stability and separator inverse amplification are separate diagnostics, not missing-pair risk guarantees or proofs of identification. Unavailable half-fits, conditioning and numerical certificates remain in the audit.",
        "",
        "Shared upstream fits are cached across completion rules. A policy visited first bears the upstream solve time; a reused policy bears only its incremental completion time. The recorded runtimes are cache/order-dependent execution diagnostics, not a fair speed ranking of separately executed estimators.",
        "",
        "The resolution floor is gamma/Q² in sqrt(W) K sqrt(W) coordinates. Its diagonal baseline trace is gamma/Q; the legacy constant floor and zero floor are explicit ablations. Vanishing baseline trace alone establishes no continuum convergence after potentially ill-conditioned separator inversion. Weighted grid coordinates are not orthonormal H coordinates.",
        "",
        "The optimizer gap certifies only the weighted observed-partial optimization error. It excludes sampling, completion amplification, grid, representation, forcing truncation and SPDE-source errors. Raw or rank-truncated completed kernels can remain indefinite; no global PSD projection occurs implicitly.",
        "",
        "## Empirical kernel error",
        "",
        "![Empirical kernel errors](empirical-errors.png)",
        "",
        empirical_caption
        + " Generating-kernel error, canonical-estimation error and the canonical-versus-generating discrepancy are distinct quantities. Unavailable policies remain in the saved summaries and the central table.",
        "",
        f"Central table: base grid {largest_q}, N={middle_n}. Unavailable metrics are not replaced by zero.",
        "",
        "| Model | Fragments | Policy | Complete error | Missing error | Canonical error | Observed modification |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["base_grid"] == largest_q and row["N"] == middle_n:
            lines.append(
                f"| {row['model']} | {row['cover']} | {row['policy']} | {_mean(row, 'complete_weighted_error')} | {_mean(row, 'missing_weighted_error')} | {_mean(row, 'canonical_weighted_error')} | {_mean(row, 'observed_entry_change')} |"
            )
    lines += [
        "",
        "## Paired observed scores and stability",
        "",
        "Paired comparisons are computed within each replication before aggregation. A lower observed score concerns observed-pair covariance prediction; it does not validate missing cross-blocks. The complete machine-readable paired and stability tables retain availability counts, the declared guard and rejected candidates.",
        "",
    ]
    if paired:
        fields = [key for key in paired[0] if key.endswith("_mean")]
        lines += [
            "| Model | Fragments | N | Policy | Metric | Paired mean ± MCSE | Available replications |",
            "|---|---:|---:|---|---|---:|---:|",
        ]
        for row in paired:
            if row.get("base_grid") == largest_q and row.get("N") == middle_n:
                for field in fields:
                    metric = field.removesuffix("_mean")
                    lines.append(
                        f"| {row['model']} | {row['cover']} | {row['N']} | {row.get('policy', row.get('comparison_rule', 'declared comparison'))} | {metric} | {_mean(row, metric)} | {row.get(metric + '_available', 'unavailable')} |"
                    )
    if stability:
        grouped = defaultdict(list)
        for row in stability:
            grouped[(row["model"], row["base_grid"], row["cover"], row["N"], row["policy"])].append(
                row
            )
        lines += [
            "",
            "Training-only selection audits (counts, not extra independent samples):",
            "",
            "| Model | Grid | Fragments | N | Policy | Available / recorded replications | Selected-candidate counts |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
        for (model, q, cover, n, policy), records in grouped.items():
            counts = defaultdict(int)
            for row in records:
                counts[row.get("selected") or "unavailable"] += 1
            selected_counts = "; ".join(
                f"{name}: {count}" for name, count in sorted(counts.items())
            )
            lines.append(
                f"| {model} | {q} | {cover} | {n} | {policy} | {sum(bool(row['available']) for row in records)}/{len(records)} | {selected_counts} |"
            )
    risks = results.get("risk_summary", ())
    if risks:
        lines += [
            "",
            "## Exposure errors and operational score availability",
            "",
            f"Central case: base grid {largest_q}, N={middle_n}. Exposure error below is absolute variance error against the generating grid kernel. The operational corrected quadratic score requires a future fragment that observes the entire exposure jointly. A latent error can be evaluated in simulation even when that score is unavailable; the two must not be conflated. Continuous delivery risk, when recorded separately in the raw risk table, also has a grid-quadrature approximation error.",
            "",
            "| Model | Fragments | Policy | Exposure | Absolute variance error | Operational risk score | Score-available replications |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
        for row in risks:
            if row["base_grid"] == largest_q and row["N"] == middle_n:
                count = row.get("risk_score_available", 0)
                score = (
                    _mean(row, "risk_score")
                    if count
                    else "unavailable: no joint future exposure observation"
                )
                lines.append(
                    f"| {row['model']} | {row['cover']} | {row['policy']} | {row['target']} | {_mean(row, 'risk_error')} | {score} | {count}/{row.get('replications', config.replications)} |"
                )
    lines += [
        "",
        "## Population refinement",
        "",
        "![Population refinement](population-refinement.png)",
        "",
        "Population audits distinguish forcing-cutoff error, finite-basis representation error and comparison to the finest available declared canonical population grid restricted to the common anchor grid. The canonical-versus-generating discrepancy can persist even when numerical refinement errors decrease. Reference availability, numerical support cutoff, inverse amplification and omitted forcing bounds remain separate columns in the population table; an unavailable reference is not evidence of convergence.",
        "",
        "Relative canonical reference error above `1e-4` is a numerical-sensitivity warning, not a statistical acceptance requirement or a continuum-error bound. The actual reference grid used is recorded explicitly; a coarser available reference does not silently become the finest configured grid.",
        "",
        "| Model | Type | Grid | Fragments | M | J | Forcing error | Representation error | Canonical reference error | Reference grid | Relative reference error | Canonical/generating discrepancy | Tail bound | Reference available |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in population:
        lines.append(
            f"| {row['model']} | {row.get('record_type', '')} | {row.get('base_grid', '')} | {row.get('cover', '')} | {row.get('M', '')} | {row.get('J', '')} | {_number(row.get('forcing_error'))} | {_number(row.get('representation_error'))} | {_number(row.get('canonical_reference_error'))} | {row.get('canonical_reference_grid', '')} | {_number(row.get('canonical_reference_relative_error'))} | {_number(row.get('canonical_generating_discrepancy'))} | {_number(row.get('tail_bound'))} | {row.get('reference_available', 'unavailable')} |"
        )
    if perturbations:
        joins = [row for row in perturbations if row.get("record_type") == "population_first_join"]
        controls = [row for row in perturbations if row.get("record_type") == "control"]
        lines += [
            "",
            "### Finite separator perturbation audit",
            "",
            f"The first-join SPD bound is available for {sum(bool(row['available']) for row in joins)}/{len(joins)} recorded population comparisons. Independent finite-matrix controls are available for {sum(bool(row['available']) for row in controls)}/{len(controls)} cases. Availability requires the separator perturbation to be smaller than its protected positive spectral gap; otherwise the bound is explicitly unavailable.",
            "",
            "These are protected **population** kernel blocks with the declared weighted diagonal floor. The comparison changes the forcing cutoff from the lowest declared M to the fixed reference cutoff. The calculation bounds A S⁻¹ B at the first join only. It is not a bound on empirically fitted kernels, subsequent recursive joins, global completed-kernel error, or continuum error. A floor can improve this finite gap condition while also changing the target; neither effect is concealed.",
            "",
            "| Model | Grid | Fragments | Floor | M / reference | Separator gap | Separator perturbation | Bound available | Cross-block bound | Actual cross-block difference |",
            "|---|---:|---:|---|---:|---:|---:|---|---:|---:|",
        ]
        for row in joins:
            lines.append(
                f"| {row['model']} | {row['base_grid']} | {row['cover']} | {row['floor_policy']} | {row['M']} / {row['reference_M']} | {_number(row.get('gap'))} | {_number(row.get('separator_perturbation'))} | {row['available']} | {_number(row.get('bound'))} | {_number(row.get('actual'))} |"
            )
    lines += [
        "",
        "## Nested finite SPDE source classes",
        "",
        "![Finite-source distance bounds](source-nesting.png)",
        "",
        "Each panel fixes output representation J, target grid, model and cover, and increases only the forcing dimension M. Varying output J changes the target representation and is not the nested-class comparison. Lower and upper bounds concern distance of that fixed target kernel to the declared finite constant-PSD-source class. Original upper bounds are feasible-witness distances; unfinished solves can have nonmonotone upper bounds even when the exact distance to truly nested classes cannot increase. The transported nested witness and map-comparison allowance make changes in the numerical transport maps explicit. Increasing unfinished upper bounds are not evidence that the exact nested-class distance increased; convergence and availability flags remain in the table.",
        "",
        "A positive finite-class lower bound is not an exclusion of an infinite-dimensional or time-varying SPDE. For empirically fitted targets this is a compatibility diagnostic of an estimate, not a sampling-calibrated rejection of the generating process. Fixed generating tail bounds and numerical map allowances do not become optimizer-gap certificates.",
        "",
        "| Model | Fragments | Target | M | J | Lower | Upper | Nested witness upper | Map allowance | Fixed generating tail | Converged | Available | Unavailable reason |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['model']} | {row['cover']} | {row['target']} | {row.get('M', '')} | {row['J']} | {_number(row.get('distance_lower'))} | {_number(row.get('distance_upper'))} | {_number(row.get('nested_witness_upper'))} | {_number(row.get('map_comparison_allowance'))} | {_number(row.get('fixed_generating_tail_bound'))} | {row.get('converged', 'unavailable')} | {row.get('available', True)} | {row.get('unavailable_reason') or ''} |"
        )
    lines += ["", "## Acceptance and retained failures", ""]
    for key, value in results["acceptance"].items():
        lines.append(f"- `{key}`: `{value}`.")
    for key in ("unavailable_count", "coherent_fit_count", "failed_certificate_count"):
        lines.append(f"- `{key}`: `{results.get(key, 'unavailable')}`.")
    controls = results.get("source_controls", ())
    if isinstance(controls, Mapping):
        lines += [
            "",
            f"Independent source-class controls passed: `{controls.get('passed', 'unavailable')}`. The complete control audit is retained in `results.json`.",
        ]
        controls = controls.get("rows", controls.get("controls", ()))
        if isinstance(controls, Mapping):
            controls = ()
    if controls:
        lines += [
            "",
            "Finite-source controls:",
            "",
            "| Control | Lower distance | Upper distance |",
            "|---|---:|---:|",
        ]
        for row in controls:
            lines.append(
                f"| {row.get('control', row.get('name', 'declared control'))} | {_number(row.get('distance_lower'))} | {_number(row.get('distance_upper'))} |"
            )
    lines += [
        "",
        "## Saved evidence and limits",
        "",
        "[Configuration](config.json) · [Source/environment manifest](manifest.json) · [Results](results.json) · [Empirical summaries](completion_summary.json) · [Paired summaries](paired_summary.json) · [Training stability](stability_summary.json) · [Exposure summaries](risk_summary.json) · [Population audit](population.json) · [Finite separator perturbations](separator_perturbations.json) · [Finite-source nesting](source_nesting.json)",
        "",
        "Each summary also has a CSV counterpart. This report reads saved summaries and audits without redoing fits, selecting favorable cases, or pooling dates as independent Monte Carlo replications. These experiments do not establish canonical identification from a fragment cover, uniform continuum stability, market calibration, confidence intervals or unrestricted SPDE admissibility.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
