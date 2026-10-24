"""Plain scientific reports for the shared dense-fragment workflow.

Figures use declared central grid/sample levels and primary ranks, never the
smallest realized error. All available method/resolution cells remain in tables.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _get(config, name):
    return config[name] if isinstance(config, dict) else getattr(config, name)


def _number(value):
    if value is None:
        return "unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}" if np.isfinite(value) else "undefined"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _primary(model):
    return "rank4" if model == "finite_rank" else "rank2"


def _risk_display(row):
    """A fixed display rule; all policies remain in machine-readable artifacts."""
    return row["method"] == "ridge" or (
        row["method"] == "canonical"
        and row["policy"] == _primary(row["model"])
        and row["variant"] in ("raw", "psd")
    )


def _estimate(row, metric, reference=False):
    center, spread = (
        ("median", "mean_absolute_deviation_from_median") if reference else ("mean", "mcse")
    )
    value, uncertainty = row.get(f"{metric}_{center}"), row.get(f"{metric}_{spread}")
    if value is None:
        return "unavailable"
    return _number(value) if uncertainty is None else f"{_number(value)} ± {_number(uncertainty)}"


def _finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        _finite(value.item())
    elif isinstance(value, (tuple, list, np.ndarray)):
        for item in value:
            _finite(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("reported measurements must be finite or None")


def _validate(config, results, examples):
    for name in ("summary", "risk_summary", "paired_differences"):
        if name not in results:
            raise ValueError(f"results must contain {name}")
        _finite(results[name])
    models, grids, counts = (_get(config, name) for name in ("models", "grids", "training_counts"))
    for row in results["summary"]:
        if (
            row["model"] not in models
            or row["base_grid"] not in grids
            or row["N"] not in counts
            or row["cover"] not in _get(config, "covers")
            or row["noise_variance"] not in _get(config, "noise_variances")
        ):
            raise ValueError("summary contains a resolution not declared by the configuration")
        if not 0 < row["replications"] <= _get(config, "replications"):
            raise ValueError("summary replication counts exceed the declared independent paths")
    for example in examples:
        nodes, kernel = np.asarray(example["nodes"]), np.asarray(example["kernel"])
        if (
            nodes.ndim != 1
            or not len(nodes)
            or not np.isfinite(nodes).all()
            or np.any(np.diff(nodes) <= 0)
            or kernel.shape != (len(nodes), len(nodes))
            or not np.isfinite(kernel).all()
            or np.asarray(example["truth"]).shape != kernel.shape
            or not np.isfinite(example["truth"]).all()
        ):
            raise ValueError("representative kernels must match a finite increasing grid")
        if example.get("canonical") is not None and (
            np.asarray(example["canonical"]).shape != kernel.shape
            or not np.isfinite(example["canonical"]).all()
        ):
            raise ValueError("population reference must match the representative kernel grid")


def _style(axis, positive=False):
    axis.grid(axis="y", color=".88", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    if positive:
        series = [np.asarray(line.get_ydata(), dtype=float) for line in axis.lines]
        centers = [
            np.asarray(line.get_ydata(), dtype=float)
            for line in axis.lines
            if line.get_marker() not in ("_", "|")
        ]
        values = np.concatenate(series) if series else np.array([])
        central = np.concatenate(centers) if centers else np.array([])
        positive_values = values[np.isfinite(values) & (values > 0)]
        if len(central) and np.all(central > 0):
            axis.set_yscale("log", nonpositive="clip")
            axis.set_ylim(float(positive_values.min()) * 0.7, float(positive_values.max()) * 1.4)
        else:
            threshold = float(positive_values.min()) / 10 if len(positive_values) else 1e-12
            axis.set_yscale("symlog", linthresh=threshold)
            axis.set_ylim(bottom=0.0)


def _save(figure, path):
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def _title(figure, text):
    figure.suptitle(fill(text, width=max(48, int(figure.get_figwidth() * 10))), fontsize=11)


def _legend(figure, axes, columns=3):
    found = {}
    for axis in np.asarray(axes).flat:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            found.setdefault(label, handle)
    if found:
        figure.legend(
            found.values(),
            found.keys(),
            loc="outside lower center",
            frameon=False,
            ncol=columns,
            fontsize=8,
        )


def _errors(config, results, output):
    models, grids = tuple(_get(config, "models")), tuple(_get(config, "grids"))
    selected_grid = grids[len(grids) // 2]
    reference = _get(config, "profile") == "reference"
    center, spread = (
        ("median", "mean_absolute_deviation_from_median") if reference else ("mean", "mcse")
    )
    figure, axes = plt.subplots(
        len(models), 2, figsize=(12, 3.1 * len(models)), squeeze=False, layout="constrained"
    )
    colors = plt.get_cmap("tab10")
    for row_index, model in enumerate(models):
        for cover_index, cover in enumerate(_get(config, "covers")):
            for noise_index, noise in enumerate(_get(config, "noise_variances")):
                for variant in ("raw", "psd"):
                    rows = sorted(
                        [
                            row
                            for row in results["summary"]
                            if row["model"] == model
                            and row["base_grid"] == selected_grid
                            and row["cover"] == cover
                            and row["noise_variance"] == noise
                            and row["method"] == "canonical"
                            and row["policy"] == _primary(model)
                            and row["variant"] == variant
                        ],
                        key=lambda row: row["N"],
                    )
                    for column, region in enumerate(("observed", "missing")):
                        metric = region + (
                            "_integrated_squared_error" if reference else "_weighted_error"
                        )
                        selected = [
                            row for row in rows if row.get(f"{metric}_{center}") is not None
                        ]
                        if selected:
                            axes[row_index, column].errorbar(
                                [row["N"] for row in selected],
                                [row[f"{metric}_{center}"] for row in selected],
                                yerr=[row.get(f"{metric}_{spread}") or 0 for row in selected],
                                color=colors(cover_index),
                                linestyle="-" if noise_index == 0 else "--",
                                marker="o" if variant == "raw" else "s",
                                markersize=4,
                                capsize=2,
                                label=f"m={cover}; noise={noise:g}; {variant}",
                            )
        for column, region in enumerate(("observed", "missing")):
            axis = axes[row_index, column]
            axis.set_title(f"{model}: {region} region ({_primary(model)})")
            axis.set_xlabel("Training fragments N")
            axis.set_ylabel(
                "Integrated squared kernel error" if reference else "Weighted kernel L2 error"
            )
            axis.set_xscale("log")
            _style(axis, positive=True)
            if not axis.lines:
                axis.text(
                    0.5,
                    0.5,
                    "No available fits at this predeclared selection",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=9,
                )
    _legend(figure, axes, columns=4)
    _title(
        figure,
        f"Dense-fragment errors: predeclared central base grid {selected_grid}; "
        + (
            "median ± mean absolute deviation from the median"
            if reference
            else "mean ± one independent-replication MCSE"
        )
        + ". All other grids and inverse policies remain in the tables.",
    )
    _save(figure, output / "fragment_errors.png")


def _selected_example(config, examples, model, variant):
    grids, counts = tuple(_get(config, "grids")), tuple(_get(config, "training_counts"))
    matches = [
        row
        for row in examples
        if row["model"] == model
        and row["replication"] == 0
        and row["base_grid"] == grids[len(grids) // 2]
        and row["N"] == counts[len(counts) // 2]
        and row["cover"] == min(_get(config, "covers"))
        and row["noise_variance"] == min(_get(config, "noise_variances"))
        and row["method"] == "canonical"
        and row["policy"] == _primary(model)
        and row["variant"] == variant
    ]
    if len(matches) > 1:
        raise ValueError("representative selection contains duplicate kernels")
    return matches[0] if matches else None


def _kernels(config, examples, output):
    models = tuple(_get(config, "models"))
    figure, axes = plt.subplots(
        len(models), 4, figsize=(14, 3 * len(models)), squeeze=False, layout="constrained"
    )
    for row_index, model in enumerate(models):
        raw, projected = (
            _selected_example(config, examples, model, variant) for variant in ("raw", "psd")
        )
        arrays = [
            raw["truth"] if raw else None,
            raw.get("canonical") if raw else None,
            raw["kernel"] if raw else None,
            projected["kernel"] if projected else None,
        ]
        limit = max(
            (float(np.max(np.abs(value))) for value in arrays if value is not None), default=1.0
        )
        for column, (value, label) in enumerate(
            zip(
                arrays,
                (
                    "Generating kernel",
                    "Population support-cutoff completion",
                    "Raw sample completion",
                    "Separate PSD projection",
                ),
                strict=True,
            )
        ):
            axis = axes[row_index, column]
            axis.set_title(f"{model}\n{label}", fontsize=9)
            if value is None:
                axis.text(
                    0.5,
                    0.5,
                    "Unavailable at the\npredeclared selection",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=9,
                )
            else:
                example = projected if column == 3 else raw
                nodes = np.asarray(example["nodes"])
                image = axis.pcolormesh(
                    nodes, nodes, value, shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit
                )
                figure.colorbar(image, ax=axis, shrink=0.7)
                mask = np.asarray(example["mask"], dtype=float)
                if mask.min() != mask.max():
                    axis.contour(nodes, nodes, mask, levels=[0.5], colors=".2", linewidths=0.5)
            axis.set_xlabel(
                "Maturity (years)" if _get(config, "profile") == "spde" else "Domain coordinate"
            )
            axis.set_ylabel(
                "Maturity (years)" if _get(config, "profile") == "spde" else "Domain coordinate"
            )
    _title(
        figure,
        "Representative kernels: replication 0; central declared grid and N; smallest cover/noise; "
        "primary rank. Shared color scale within each model. The finite population completion is not continuum truth.",
    )
    _save(figure, output / "fragment_kernels.png")


def _risks(config, results, output):
    models, grids = tuple(_get(config, "models")), tuple(_get(config, "grids"))
    q, cover, noise = (
        grids[len(grids) // 2],
        min(_get(config, "covers")),
        max(_get(config, "noise_variances")),
    )
    figure, axes = plt.subplots(
        len(models), 2, figsize=(12, 3.8 * len(models)), squeeze=False, layout="constrained"
    )
    colors = plt.get_cmap("tab10")
    for row_index, model in enumerate(models):
        candidates = [
            row
            for row in results["risk_summary"]
            if row["model"] == model
            and row["base_grid"] == q
            and row["cover"] == cover
            and row["noise_variance"] == noise
        ]
        targets = sorted({row["target"] for row in candidates})
        counts = []
        for target_index, target in enumerate(targets):
            for variant in ("raw", "psd"):
                rows = sorted(
                    [
                        row
                        for row in candidates
                        if row["target"] == target
                        and row["method"] == "canonical"
                        and row["policy"] == _primary(model)
                        and row["variant"] == variant
                    ],
                    key=lambda row: row["N"],
                )
                rows = [row for row in rows if row.get("risk_squared_error_mean") is not None]
                if rows:
                    axes[row_index, 0].errorbar(
                        [row["N"] for row in rows],
                        [row["risk_squared_error_mean"] for row in rows],
                        yerr=[row.get("risk_squared_error_mcse") or 0 for row in rows],
                        color=colors(target_index % 10),
                        linestyle="-" if variant == "raw" else "--",
                        marker="o",
                        markersize=3,
                        capsize=2,
                        label=f"{target}: {variant}",
                    )
                    if variant == "raw":
                        counts.append(
                            f"{target}: {rows[0].get('available_test_dates_mean', 0):g} test dates"
                        )
        paired = [
            row
            for row in results["paired_differences"]
            if row["model"] == model
            and row["base_grid"] == q
            and row["cover"] == cover
            and row["noise_variance"] == noise
            and (
                (
                    row["method"] == "canonical"
                    and row["policy"] == _primary(model)
                    and row["variant"] == "psd"
                )
                or row["method"] == "ridge"
            )
        ]
        choices = sorted({(row["method"], row["policy"], row["variant"]) for row in paired})
        for method, policy, variant in choices:
            rows = sorted(
                [
                    row
                    for row in paired
                    if (row["method"], row["policy"], row["variant"]) == (method, policy, variant)
                    and row["score_difference_mean"] is not None
                ],
                key=lambda row: row["N"],
            )
            axes[row_index, 1].errorbar(
                [row["N"] for row in rows],
                [row["score_difference_mean"] for row in rows],
                yerr=[row.get("score_difference_mcse") or 0 for row in rows],
                fmt="o-",
                capsize=2,
                label=f"{method}/{policy}/{variant}",
            )
        axes[row_index, 1].axhline(0.0, color=".35", linewidth=0.7)
        axes[row_index, 0].set_title(f"{model}: risk MSE ± one MCSE")
        axes[row_index, 1].set_title(f"{model}: paired observed-score difference")
        axes[row_index, 0].set_ylabel("Risk MSE")
        axes[row_index, 1].set_ylabel("Paired score difference")
        for column in range(2):
            axes[row_index, column].set_xlabel("Training fragments N")
            axes[row_index, column].set_xscale("log")
            _style(axes[row_index, column], positive=column == 0)
        if counts:
            axes[row_index, 1].text(
                0.02,
                0.98,
                "\n".join(counts),
                va="top",
                fontsize=7,
                transform=axes[row_index, 1].transAxes,
            )
    _legend(figure, axes, columns=3)
    _title(
        figure,
        f"Predeclared risk/score comparison: central base grid {q}, m={cover}, noise variance={noise:g}. "
        "Risk errors may be evaluated in simulation even when future proxy scoring is unavailable.",
    )
    _save(figure, output / "fragment_risks.png")


def _table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_number(value) for value in row) + " |" for row in rows),
    ]


def _audit_tables(output):
    path = output / "completion_audit.json"
    if not path.exists():
        return [
            "Detailed rank and projection diagnostics are unavailable in this report invocation.",
            "",
        ]
    rows = json.loads(path.read_text())
    failures, ranks, changes, joins = Counter(), Counter(), [], []
    for row in rows:
        diagnostic = row.get("diagnostics")
        if diagnostic:
            if not diagnostic.get("available", True):
                failures[str(diagnostic.get("reason", "unspecified numerical unavailability"))] += 1
            for join in diagnostic.get("joins", []):
                joins.append(join)
                if join.get("selected_rank") is not None:
                    ranks[(row.get("policy"), join["selected_rank"])] += 1
        if row.get("projection"):
            changes.append(row["projection"])
    lines = [
        "### Inverse availability and PSD-projection changes",
        "",
        "Rank counts below count separator solves, not independent paths. Unavailable inverses remain unavailable; they are not replaced by another rank.",
        "",
    ]
    lines += _table(
        ["Policy", "Selected separator rank", "Solves"],
        [(policy, rank, count) for (policy, rank), count in sorted(ranks.items())],
    )
    lines += [""] + _table(["Unavailable reason", "Fits"], sorted(failures.items())) + [""]
    lines += _table(
        ["Projection diagnostic", "Maximum over recorded projections"],
        [
            (
                key,
                max(
                    (
                        row[key]
                        for row in changes
                        if row.get(key) is not None and np.isfinite(row[key])
                    ),
                    default=None,
                ),
            )
            for key in (
                "weighted_hs_change",
                "observed_weighted_hs_change",
                "observed_maximum_change",
                "weighted_trace_change",
                "kernel_diagonal_sum_change",
            )
        ],
    ) + [""]
    numerical = []
    for key in ("retained_condition_number", "inverse_norm", "inverse_perturbation_multiplier"):
        values = [row[key] for row in joins if row.get(key) is not None and np.isfinite(row[key])]
        numerical.append((key, max(values) if values else None, len(joins) - len(values)))
    lines += _table(
        ["Separator diagnostic", "Maximum finite value", "Undefined/unrecorded solves"], numerical
    ) + [""]
    return lines


def _runtime_table(results):
    groups = {}
    for row in results["summary"]:
        value = row.get("runtime_seconds_mean")
        if value is not None:
            key = tuple(row[name] for name in ("method", "policy", "variant", "base_grid"))
            groups.setdefault(key, []).append(value)
    return [
        "### Recorded wall-clock times",
        "",
        "These are minimum/maximum cell mean wall times across configured sources, noise levels, covers and training counts. They describe this recorded environment, not hardware-independent speed. The CSV retains each cell and replication.",
        "",
        *_table(
            [
                "Method",
                "Policy",
                "Variant",
                "Grid",
                "Cells",
                "Min mean seconds",
                "Max mean seconds",
            ],
            [
                (*key, len(values), min(values), max(values))
                for key, values in sorted(groups.items())
            ],
        ),
        "",
    ]


def write_fragment_report(config, output, results, examples, population):
    """Write figures and a complete-method report without choosing favorable cases."""
    output = Path(output)
    _validate(config, results, examples)
    output.mkdir(parents=True, exist_ok=True)
    _errors(config, results, output)
    _kernels(config, examples, output)
    _risks(config, results, output)
    reference = _get(config, "profile") == "reference"
    label = (
        "median ± mean absolute deviation from the median"
        if reference
        else "mean ± independent-replication MCSE"
    )
    lines = [
        "# Dense-fragment covariance kernels",
        "",
        "This study extends the existing observation workflow: parent Gaussian measurements are generated before grid/fragment/method views; fitting uses corrected observed products. The [protocol](https://github.com/xrough/spde-covariance/blob/main/docs/canonical-completion.md) distinguishes H-coordinate ridge covariance from covariance kernels and their weighted L2 operators.",
        "",
        f"Profile: {_get(config, 'profile')}; seed: {_get(config, 'seed')}; independent replications: {_get(config, 'replications')}. Training N={tuple(_get(config, 'training_counts'))}; base grids={tuple(_get(config, 'grids'))}; covers={tuple(_get(config, 'covers'))}; noise variances={tuple(_get(config, 'noise_variances'))}.",
        "",
        "Assumptions: centered Gaussian curves, common covariance across independent fragments, balanced randomized fragment assignment, dense point measurements, and known fresh independent measurement-noise variance. This is not extraction from sparse/noisy curve levels or a market-prediction experiment. Smaller training counts share prefixes; grids share parent curves. Replication MCSE never counts dates, grids, ranks or covers as independent experiments.",
        "",
        "The reference profile uses equal 1/Q grid weights on [0,1]. SPDE kernels use physical trapezoidal weights on [0,3] maturity years and a fixed normalized-innovation interval; their kernel values have curve-unit squared per year. Kernel L2 norms include the declared maturity quadrature. They are not H-coordinate operator norms. Continuous delivery risk and its discretized grid approximation remain separately labeled.",
        "",
        "## Numerical acceptance and scope",
        "",
    ]
    lines += _table(["Check", "Value"], list(results.get("acceptance", {}).items())) + [""]
    lines += [
        f"Available recorded fits: {results.get('available_fits', 'unrecorded')}; unavailable fits: {results.get('unavailable_fits', 'unrecorded')}; raw indefinite fits: {results.get('raw_indefinite_fits', results.get('acceptance', {}).get('raw_indefinite_fits', 'unrecorded'))}.",
        "",
        "Raw truncated completion may be indefinite. The separate PSD projection changes the estimator and may alter observed entries; it is not called an exact canonical completion. Only ridge retains the existing optimization-distance certificate. Integration agreement, support thresholds and source-tail checks are numerical diagnostics, not a completion-error certificate.",
        "",
        "![Observed and missing-region errors](fragment_errors.png)",
        "",
        "## Every configured method and resolution",
        "",
        f"Entries use {label}. The paper states mean absolute deviation without specifying its center; this report uses the median explicitly, and the archive also retains deviation from the mean and median absolute deviation. Regional values are ISE in the reference profile and L2 norms in the SPDE profile; complete ISE is always squared. These descriptive uncertainty summaries are not confidence intervals. Unavailable fits are labeled rather than silently assigned zero error.",
        "",
    ]
    lines += _table(
        [
            "Model",
            "Noise",
            "Grid",
            "m",
            "N",
            "Method/policy/variant",
            "Available/unavailable reps",
            "Observed ISE" if reference else "Observed L2",
            "Missing ISE" if reference else "Missing L2",
            "Complete ISE",
            "Error to population completion (L2)",
        ],
        [
            (
                row["model"],
                row["noise_variance"],
                row["base_grid"],
                row["cover"],
                row["N"],
                f"{row['method']}/{row['policy']}/{row['variant']}",
                f"{row.get('available_replications', row['replications'])}/{row.get('unavailable_replications', 0)}",
                _estimate(
                    row,
                    "observed_integrated_squared_error" if reference else "observed_weighted_error",
                    reference,
                ),
                _estimate(
                    row,
                    "missing_integrated_squared_error" if reference else "missing_weighted_error",
                    reference,
                ),
                _estimate(row, "integrated_squared_error", reference),
                _estimate(row, "canonical_weighted_error", reference),
            )
            for row in results["summary"]
        ],
    ) + [""]
    lines += [
        "## Population structure and representative kernels",
        "",
        "![Predeclared representative kernel panels](fragment_kernels.png)",
        "",
        "The population support-cutoff completion is a finite-grid reference. Its discrepancy from the generating kernel is structural/truncation/discretization discrepancy, separate from sample error against that completion. A canonical completion is an identifiable completion rule; without uniqueness it need not equal the generating covariance. Population inverse failure or sensitivity must remain visible.",
        "",
    ]
    controls = population.get("controls", {})
    lines += _table(
        ["Model", "m", "Grid", "Rank", "Population missing ISE", "Observed preservation"],
        [
            (
                row["model"],
                row["cover"],
                row["grid_size"],
                row["rank"],
                row["missing_ise"],
                row["observed_max_abs_error"],
            )
            for row in controls.get("paper", [])
        ],
    ) + [""]
    for name in ("noncanonical", "truncation"):
        lines += [
            f"### {name.capitalize()} identification control",
            "",
            "```json",
            json.dumps(
                controls.get(name, {}), indent=2, default=lambda value: np.asarray(value).tolist()
            ),
            "```",
            "",
        ]
    lines += [
        "### Forcing/grid/overlap/support sensitivity",
        "",
        "Comparisons below use common physical evaluation targets and the predeclared finest forcing/grid, support cutoff 1e-12 as numerical reference. Agreement is not proof of continuum convergence; disagreement is not a sampling effect.",
        "",
    ]
    lines += _table(
        [
            "Model",
            "Overlap",
            "Base/actual grid",
            "Forcing M",
            "Support cutoff",
            "Available",
            "Relative reference difference",
            "Unstable flag",
        ],
        [
            (
                row["model"],
                row["overlap_fraction"],
                f"{row['base_grid']}/{row['actual_grid']}",
                row["forcing_modes"],
                row["support_rtol"],
                row["available"],
                row.get("relative_weighted_reference_difference"),
                row.get("numerical_reference_unstable"),
            )
            for row in population.get("sensitivity", [])
        ],
    ) + [""]
    lines += _audit_tables(output)
    lines += _runtime_table(results)
    lines += [
        "## Exposure risks and held-out scores",
        "",
        "![Exposure and paired-score comparisons](fragment_risks.png)",
        "",
        "Risk error is evaluated against the generating grid kernel in simulation. Future risk scores require the complete exposure support in one actual fragment. Unavailable scoring is not filled using latent truth. Continuous delivery integrals have no exact proxy from finite point measurements. Raw completion ranks preserve the same observed partial covariance, so ordinary held-out observed scores cannot tune those ranks. Ridge alone uses observed validation and remains frozen for test evaluation.",
        "",
        "To keep this report readable, the following risk and paired-score tables use the predeclared primary rank (4 for the finite-rank control, 2 otherwise), both raw and PSD variants, and ridge wherever configured, across every grid/cover/noise/N condition. This display rule does not inspect outcomes. All inverse ranks and FVE remain in the covariance table above and the complete [risk archive](risks.csv), [risk summary](risk_summary.csv), and [paired-score archive](paired.csv).",
        "",
    ]
    lines += _table(
        [
            "Model",
            "Noise",
            "Grid",
            "m",
            "N",
            "Method/policy/variant",
            "Exposure",
            "Available test dates",
            "Risk MSE",
            "Future risk score",
        ],
        [
            (
                row["model"],
                row["noise_variance"],
                row["base_grid"],
                row["cover"],
                row["N"],
                f"{row['method']}/{row['policy']}/{row['variant']}",
                row["target"],
                row.get("available_test_dates_mean"),
                _estimate(row, "risk_squared_error", reference),
                _estimate(row, "risk_score", reference),
            )
            for row in results["risk_summary"]
            if _risk_display(row)
        ],
    ) + [""]
    lines += [
        "Paired score differences compare each method with the primary-rank raw completion on the same observations. Scores can be negative because a common measurement-only constant is omitted. Raw values across grids, covers or noise settings have different scales; they are not a cross-design ranking.",
        "",
    ]
    lines += _table(
        [
            "Model",
            "Noise",
            "Grid",
            "m",
            "N",
            "Method/policy/variant",
            "Paired reps",
            "Score difference mean ± MCSE",
        ],
        [
            (
                row["model"],
                row["noise_variance"],
                row["base_grid"],
                row["cover"],
                row["N"],
                f"{row['method']}/{row['policy']}/{row['variant']}",
                row["paired_replications"],
                f"{_number(row['score_difference_mean'])} ± {_number(row['score_difference_mcse'])}",
            )
            for row in results["paired_differences"]
            if _risk_display(row)
        ],
    ) + [""]
    lines += [
        "## Artifacts and provenance",
        "",
        "The unchanged [manifest](manifest.json) and [configuration](config.json) identify the source revision, environment, methods and quadrature convention. Full numerical arrays and diagnostics accompany the report:",
        "",
        "- [Results](results.json), [population controls and sensitivity](population.json), [grid geometry](geometry.json).",
        "- [Completion and projection audits](completion_audit.json), [representative kernels](representative_kernels.json).",
        "- [Every fit](estimates.csv), [risk evaluations](risks.csv), [summary](summary.csv), [paired differences](paired.csv).",
        "- [Ridge candidates](ridge_candidates.json) and [frozen validation selections](ridge_selection.json).",
        "",
        "The author-code reference is restricted to explicit-rank recurrence parity. Corrected FVE uses the consistently scaled overlap trace; the legacy R normalization is retained only as a documented reference discrepancy. No statistical superiority or general recovery theorem is inferred from this finite study.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return output / "report.md"
