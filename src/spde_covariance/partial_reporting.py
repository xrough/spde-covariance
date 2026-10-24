"""Fixed-panel reports for partial observations of one infinite-model innovation.

Finite-sieve geometry, full-functional proxy availability, representation bias
and latent evaluation errors are separate objects throughout the report.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_TARGETS = ("point_spread", "delivery_spread", "offgrid_spread")
_LABELS = {
    "powerlaw": "Power-law source",
    "diagonal": "Power-law source",
    "rotated": "Adjacent-pair rotation",
    "fixed_points": "Two points",
    "fixed_deliveries": "Two deliveries",
    "nested8": "Nested 8",
    "nested16": "Nested 16",
    "rotating4": "Rotating 4",
    "point_spread": "Point spread",
    "delivery_spread": "Delivery spread",
    "offgrid_spread": "Off-grid spread",
}


def _value(config: Any, name: str, default: Any = None) -> Any:
    return (
        config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)
    )


def _label(value: str) -> str:
    return _LABELS.get(value, value.replace("_", " "))


def _number(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{value:.5g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _estimate(row: dict | None, metric: str) -> str:
    if row is None or row.get(metric + "_mean") is None:
        return "unavailable"
    error = row.get(metric + "_mcse")
    return _number(row[metric + "_mean"]) + (f" ± {_number(error)}" if error is not None else "")


def _finite(value: Any, context: str = "results") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{context}.{key}")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        _finite(value.item(), context)
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            _finite(item, context)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{context} must be finite or None")


def _declared(config: Any) -> tuple[tuple, tuple, tuple, tuple]:
    values = tuple(
        tuple(_value(config, name, ()))
        for name in ("scenarios", "designs", "training_counts", "geometry_dimensions")
    )
    if any(not group or len(set(group)) != len(group) for group in values):
        raise ValueError("report configuration must declare unique nonempty cases and dimensions")
    return values[0], values[1], tuple(sorted(values[2])), tuple(sorted(values[3]))


def _case(row: dict) -> tuple:
    return row["scenario"], row["design"], row["N"]


def _validate(config: Any, results: dict) -> None:
    scenarios, designs, sizes, dimensions = _declared(config)
    _finite(results)
    cases = set(product(scenarios, designs, sizes))
    for name in ("selected_summary", "selections", "risk_summary", "paired_differences"):
        if any(_case(row) not in cases for row in results.get(name, [])):
            raise ValueError(f"{name} contains an undeclared case")
    selected = [(*_case(row), row["rule"]) for row in results["selected_summary"]]
    expected = {(*case, rule) for case in cases for rule in ("selected", "fixed")}
    if len(selected) != len(set(selected)) or set(selected) != expected:
        raise ValueError("selected_summary must retain every declared case and both rules")
    geometry = [(row["design"], row["N"], row["J"]) for row in results["geometry"]]
    expected_geometry = set(product(designs, sizes, dimensions))
    if len(geometry) != len(set(geometry)) or set(geometry) != expected_geometry:
        raise ValueError("geometry must retain every declared design, size and dimension")
    spans = results["full_h_geometry"]
    union_keys = [
        (row["design"], row["N"], row["target"])
        for row in spans
        if row["kind"] == "training_union_functional_span"
    ]
    if len(union_keys) != len(set(union_keys)) or set(union_keys) != set(
        product(designs, sizes, _TARGETS)
    ):
        raise ValueError("full-H union spans must retain every declared design, size and exposure")
    same_date = [
        (row["design"], tuple(row["contract_indices"]), row["target"])
        for row in spans
        if row["kind"] == "same_date"
    ]
    groups = {(design, indices) for design, indices, _ in same_date}
    if (
        {design for design, _ in groups} != set(designs)
        or set(same_date)
        != {(design, indices, target) for design, indices in groups for target in _TARGETS}
        or len(same_date) != len(set(same_date))
    ):
        raise ValueError("same-date full-H spans must retain all targets on every recorded group")
    approximation = [(*_case(row), row["J"]) for row in results["observation_approximation"]]
    expected_approximation = {(*case, dimension) for case in cases for dimension in dimensions}
    if (
        len(approximation) != len(set(approximation))
        or set(approximation) != expected_approximation
    ):
        raise ValueError("observation_approximation must retain all declared cases and dimensions")
    for row in results["selected_summary"]:
        if row["replications"] != _value(config, "replications"):
            raise ValueError("summary replication counts must match the configuration")
    for row in results["selections"]:
        if row["selected_J"] not in _value(config, "candidate_dimensions"):
            raise ValueError("selected J must belong to the declared candidate grid")
        if row["selected_relative_regularization"] not in _value(
            config, "relative_regularizations"
        ):
            raise ValueError("selected penalty must belong to the declared candidate grid")
    selection_keys = [(*_case(row), row["replication"]) for row in results["selections"]]
    expected_selections = {
        (*case, replication)
        for case in cases
        for replication in range(_value(config, "replications"))
    }
    if (
        len(selection_keys) != len(set(selection_keys))
        or set(selection_keys) != expected_selections
    ):
        raise ValueError("selections must contain every declared case and replication")
    candidate_keys = [
        (*_case(row), row["J"], row["relative_regularization"])
        for row in results["candidate_summary"]
    ]
    expected_candidates = {
        (*case, dimension, penalty)
        for case in cases
        for dimension in _value(config, "candidate_dimensions")
        for penalty in _value(config, "relative_regularizations")
    }
    if (
        len(candidate_keys) != len(set(candidate_keys))
        or set(candidate_keys) != expected_candidates
    ):
        raise ValueError("candidate_summary must retain the complete declared fit grid")
    risk_keys = [(*_case(row), row["rule"], row["target"]) for row in results["risk_summary"]]
    expected_risks = {
        (*case, rule, target)
        for case in cases
        for rule in ("selected", "fixed", "direct")
        for target in _TARGETS
    }
    if len(risk_keys) != len(set(risk_keys)) or set(risk_keys) != expected_risks:
        raise ValueError("risk_summary must retain all exposures, including unavailable scores")
    for row in results["risk_summary"]:
        if row["available_test_dates"] == 0 and row.get("risk_score_mean") is not None:
            raise ValueError("an unavailable future proxy cannot have a risk score")
    for row in results["paired_differences"]:
        if (
            row.get("available_replications", row.get("replications", 0)) == 0
            and row.get("mean_difference") is not None
        ):
            raise ValueError("unavailable paired comparisons must not be filled with zero")
    paired = [
        (*_case(row), row["metric"], row.get("target"))
        for row in results["paired_differences"]
        if row.get("comparison_rule", "selected") == "selected"
        and row.get("baseline_rule", "fixed") == "fixed"
    ]
    expected_paired = {
        (*case, metric, target)
        for case in cases
        for metric, target in (
            ("observed_score", None),
            *(("risk_score", target) for target in _TARGETS),
        )
    }
    if len(paired) != len(set(paired)) or set(paired) != expected_paired:
        raise ValueError(
            "paired differences must retain every declared score, including unavailable risks"
        )


def _style(axis) -> None:
    axis.grid(axis="y", color="0.88", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def _nonnegative_axis(axis) -> None:
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in axis.lines])
    positive = values[values > 0]
    if len(positive) and np.all(values > 0):
        axis.set_yscale("log")
    elif len(positive):
        axis.set_yscale("symlog", linthresh=float(min(positive) / 10))
        axis.set_ylim(bottom=0)


def _title(figure, *paragraphs: str) -> None:
    width = max(32, round(9 * figure.get_figwidth()))
    figure.suptitle(
        "\n".join(fill(paragraph, width=width) for paragraph in paragraphs), fontsize=11
    )


def _save(figure, output: Path, name: str) -> None:
    figure.savefig(output / name, dpi=160, facecolor="white")
    plt.close(figure)


def _legend(figure, axes, columns: int = 3) -> None:
    handles, labels = [], []
    for axis in np.asarray(axes).flat:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=columns, frameon=False)


def _geometry_figure(config: Any, results: dict, output: Path) -> None:
    _, designs, sizes, dimensions = _declared(config)
    lookup = {(row["design"], row["N"], row["J"]): row for row in results["geometry"]}
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for index, design in enumerate(designs):
        rows = [lookup[(design, max(sizes), dimension)] for dimension in dimensions]
        axes[0, 0].plot(
            dimensions,
            [row["rank"] / row["dimension"] for row in rows],
            "o-",
            label=_label(design),
            color=f"C{index}",
        )
        axes[0, 1].plot(
            dimensions,
            [row["smallest_positive_eigenvalue"] for row in rows],
            "o-",
            color=f"C{index}",
        )
    axes[0, 0].set_title("Finite symmetric-coordinate Gram rank")
    axes[0, 0].set_ylabel("Numerical rank / symmetric dimension")
    axes[0, 0].set_ylim(-0.02, 1.05)
    axes[0, 1].set_title("Smallest retained Gram eigenvalue")
    axes[0, 1].set_ylabel("Eigenvalue above numerical rank threshold")
    axes[0, 1].set_yscale("log")
    for axis in axes[0]:
        axis.set_xlabel("Covariance sieve dimension J")
        axis.set_xticks(dimensions)
        _style(axis)
    matrix = np.array(
        [
            [
                lookup[(design, max(sizes), dimension)]["targets"][target][
                    "nullspace_relative_residual"
                ]
                for dimension in dimensions
            ]
            for design, target in product(designs, _TARGETS)
        ]
    )
    image = axes[1, 0].imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="cividis")
    axes[1, 0].set_xticks(range(len(dimensions)), dimensions)
    axes[1, 0].set_yticks(
        range(len(designs) * len(_TARGETS)),
        [f"{_label(design)} / {_label(target)}" for design, target in product(designs, _TARGETS)],
        fontsize=7,
    )
    axes[1, 0].set_xlabel("Sieve dimension J")
    axes[1, 0].set_title("Finite-J lifted target range residual")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.7, label="Relative nullspace residual")
    risk_lookup = {
        (row["design"], row["target"]): row
        for row in results["risk_summary"]
        if row["N"] == max(sizes)
        and row["scenario"] == _declared(config)[0][0]
        and row["rule"] == "selected"
    }
    availability = np.array(
        [
            [
                risk_lookup[(design, target)]["available_test_dates"]
                / _value(config, "test_count", 128)
                for target in _TARGETS
            ]
            for design in designs
        ]
    )
    image = axes[1, 1].imshow(availability, aspect="auto", vmin=0, vmax=1, cmap="cividis")
    for row, column in product(range(len(designs)), range(len(_TARGETS))):
        axes[1, 1].text(
            column,
            row,
            f"{availability[row, column]:.0%}",
            ha="center",
            va="center",
            color="white" if availability[row, column] < 0.5 else "black",
        )
    axes[1, 1].set_xticks(
        range(len(_TARGETS)), [_label(target) for target in _TARGETS], rotation=20
    )
    axes[1, 1].set_yticks(range(len(designs)), [_label(design) for design in designs])
    axes[1, 1].set_title("Full-H same-date risk-proxy availability")
    figure.colorbar(image, ax=axes[1, 1], shrink=0.7, label="Fraction of test dates")
    _legend(figure, axes, columns=5)
    _title(
        figure,
        f"Observation geometry at declared training N={max(sizes)}; source independent",
        "Finite Gram/range diagnostics do not establish full-H or source-constrained identification. Same-date proxy absence is not a pooled nonidentification proof.",
    )
    _save(figure, output, "geometry_observability.png")


def _approximation_figure(config: Any, results: dict, output: Path) -> None:
    scenarios, designs, sizes, dimensions = _declared(config)
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(6 * len(scenarios), 4.8), squeeze=False, layout="constrained"
    )
    for column, scenario in enumerate(scenarios):
        axis = axes[0, column]
        for index, design in enumerate(designs):
            rows = sorted(
                (
                    row
                    for row in results["observation_approximation"]
                    if row["scenario"] == scenario
                    and row["design"] == design
                    and row["N"] == max(sizes)
                ),
                key=lambda row: row["J"],
            )
            axis.plot(
                [row["J"] for row in rows],
                [row["relative_hs_error"] for row in rows],
                "o-",
                color=f"C{index}",
                label=_label(design),
            )
        axis.set_title(_label(scenario))
        axis.set_xlabel("Covariance sieve dimension J")
        axis.set_ylabel("Relative observed-covariance approximation HS error")
        axis.set_xticks(dimensions)
        _nonnegative_axis(axis)
        _style(axis)
    _legend(figure, axes, columns=5)
    _title(
        figure,
        f"One full observation covariance versus its J-sieve prediction, N={max(sizes)} design frequencies",
        "Both omitted-omitted and retained-omitted cross terms are included. This finite-reference representation error is separate from measurement noise and forcing-tail bounds.",
    )
    _save(figure, output, "observation_approximation.png")


def _performance_figure(config: Any, results: dict, output: Path) -> None:
    scenarios, designs, sizes, _ = _declared(config)
    figure, axes = plt.subplots(
        len(scenarios),
        4,
        figsize=(17, 3.5 * len(scenarios) + 1),
        squeeze=False,
        layout="constrained",
    )
    for line, scenario in enumerate(scenarios):
        for column, target in enumerate((None, *_TARGETS)):
            axis = axes[line, column]
            axis.axhline(0, color="0.3", linestyle="--", linewidth=0.8)
            available = set()
            for index, size in enumerate(sizes):
                rows = [
                    row
                    for row in results["paired_differences"]
                    if row["scenario"] == scenario
                    and row["N"] == size
                    and row["metric"] == ("observed_score" if target is None else "risk_score")
                    and row.get("target") == target
                    and row.get("comparison_rule", "selected") == "selected"
                    and row.get("baseline_rule", "fixed") == "fixed"
                    and row.get("mean_difference") is not None
                ]
                available.update(row["design"] for row in rows)
                axis.errorbar(
                    [
                        designs.index(row["design"]) + (index - (len(sizes) - 1) / 2) * 0.16
                        for row in rows
                    ],
                    [row["mean_difference"] for row in rows],
                    yerr=[row.get("mcse") or 0 for row in rows],
                    fmt=("o", "s", "^")[index % 3],
                    color=f"C{index}",
                    capsize=2,
                    label=f"Training N={size}",
                )
            for index, design in enumerate(designs):
                if design not in available:
                    axis.text(
                        index,
                        0.6,
                        "unavailable",
                        transform=axis.get_xaxis_transform(),
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=7,
                        color="0.4",
                    )
            axis.set_xticks(
                range(len(designs)), [_label(design) for design in designs], rotation=30, ha="right"
            )
            axis.set_xlim(-0.5, len(designs) - 0.5)
            axis.set_title(
                f"{_label(scenario)}\n{_label(target) if target else 'Observed covariance score'}",
                fontsize=9,
            )
            axis.set_ylabel("Selected minus fixed score")
            _style(axis)
    _legend(figure, axes)
    _title(
        figure,
        "Frozen selected-versus-fixed fits on the same subsequent observations",
        "Negative favors selected within that design/exposure. Error bars: ±1 paired replication MCSE. Unavailable full-H risk proxies are omitted, never filled with zero.",
    )
    _save(figure, output, "selected_performance.png")


def _record_lines(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        return [
            line
            for key in sorted(value)
            for line in _record_lines(value[key], f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            line
            for index, item in enumerate(value)
            for line in _record_lines(item, f"{prefix}[{index}]")
        ]
    return [f"| {_number(prefix)} | {_number(value)} |"]


def _model_summary(model: dict) -> dict:
    """Keep large coefficient/matrix arrays in machine-readable results."""
    output = {
        key: model[key] for key in ("basis_family_id", "sources", "contracts") if key in model
    }
    output["tail_bounds"] = {
        source: {key: value for key, value in bounds.items() if key != "observed_matrix"}
        for source, bounds in model.get("tail_bounds", {}).items()
    }
    output["fixed_grid_witness"] = {
        key: value
        for key, value in model.get("fixed_grid_witness", {}).items()
        if key not in ("vector", "rank_one_covariance")
    }
    return output


def _tables(config: Any, results: dict) -> list[str]:
    scenarios, designs, sizes, _ = _declared(config)
    selected = {(*_case(row), row["rule"]): row for row in results["selected_summary"]}
    risks = {(*_case(row), row["rule"], row["target"]): row for row in results["risk_summary"]}
    lines = [
        "<details>",
        "<summary>All configured population geometry and observation-approximation rows</summary>",
        "",
        "Geometry uses normalized date weights. Rank and target range diagnostics depend on a declared numerical threshold, not sample count as an unnormalized multiplier. Approximation HS values are root-mean-square norms across the actual date maps; their relative value divides by the corresponding RMS full observation-covariance norm. Tail-tail and cross-term norms are shown separately; their norms do not add to the norm of their sum.",
        "",
        "| Design | N | J | Gram rank/dimension | Smallest retained eigenvalue |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in sorted(
        results["geometry"], key=lambda row: (designs.index(row["design"]), row["N"], row["J"])
    ):
        lines.append(
            f"| {_label(row['design'])} | {row['N']} | {row['J']} | {row['rank']}/{row['dimension']} | {_number(row['smallest_positive_eigenvalue'])} |"
        )
    lines += [
        "",
        "Full-H rows below use analytic functional Gram calculations, independent of J. Same-date availability concerns a scalar proxy; union membership concerns linear functionals. Neither alone proves pooled covariance identification or SPDE source admissibility. Coefficients and precise residuals remain in full_h_geometry.json.",
        "",
        "| Design | Scope | N | Contract indices | Exposure | In span | Relative full-H residual |",
        "| --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for row in results["full_h_geometry"]:
        lines.append(
            f"| {_label(row['design'])} | {row['kind']} | {_number(row.get('N'))} | {_number(row['contract_indices'])} | {_label(row['target'])} | {row['available']} | {_number(row['relative_residual'])} |"
        )
    lines += [
        "",
        "| Source | Design | N | J | Full-minus-sieve RMS HS | Relative RMS HS | Omitted-omitted RMS HS | Cross-sum RMS HS | Observation reference-tail bound |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        results["observation_approximation"], key=lambda row: (*_case(row), row["J"])
    ):
        lines.append(
            f"| {_label(row['scenario'])} | {_label(row['design'])} | {row['N']} | {row['J']} | "
            + " | ".join(
                _number(row.get(key))
                for key in (
                    "hs_error",
                    "relative_hs_error",
                    "tail_tail_hs_error",
                    "cross_sum_hs_error",
                    "observed_reference_tail_bound",
                )
            )
            + " |"
        )
    lines += [
        "",
        "</details>",
        "",
        "## Frozen fits on subsequent observations",
        "",
        "![Paired selected-minus-fixed performance](selected_performance.png)",
        "",
        "Scores omit candidate-independent constants and can be negative. Each entry is an independent-replication mean ± MCSE. Compare rules on the same future observations; absolute scores across different designs are not a common-information ranking. Selected/fixed matrices use training only and remain frozen after validation.",
        "",
        "| Source | Design | N | Selected test score | Fixed test score | Selected latent HS error | Fixed latent HS error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in product(scenarios, designs, sizes):
        choice, fixed = selected[(*case, "selected")], selected[(*case, "fixed")]
        lines.append(
            f"| {_label(case[0])} | {_label(case[1])} | {case[2]} | {_estimate(choice, 'test_observed_score')} | {_estimate(fixed, 'test_observed_score')} | {_estimate(choice, 'operator_hs_error')} | {_estimate(fixed, 'operator_hs_error')} |"
        )
    lines += [
        "",
        "## Candidate choices",
        "",
        f"Validation selects observed-space prediction loss, not latent operator/risk error. Exact ties prefer smaller J, then larger relative penalty. The fixed baseline is J={max(_value(config, 'candidate_dimensions'))}, relative penalty {_number(min(_value(config, 'relative_regularizations')))}, from the same candidate fits; neither rule refits through validation.",
        "",
        "| Source | Design | N | Selected (J, relative penalty): replication counts |",
        "| --- | --- | ---: | --- |",
    ]
    for case in product(scenarios, designs, sizes):
        counts = Counter(
            (row["selected_J"], row["selected_relative_regularization"])
            for row in results["selections"]
            if _case(row) == case
        )
        counts_text = "; ".join(
            f"({dimension}, {_number(penalty)}): {count}"
            for (dimension, penalty), count in sorted(counts.items())
        )
        lines.append(f"| {_label(case[0])} | {_label(case[1])} | {case[2]} | {counts_text} |")
    lines += [
        "",
        "## Fixed full-H exposure risks",
        "",
        "Future scalar proxy scores require a same-date full-H functional representation. Failure of this availability test is not a pooled covariance nonidentification proof. Plugin risks restrict a fixed full-H exposure to the fitted J without renormalization; their latent errors use full-reference V risk. Direct risk estimates use training proxies on available dates only. Latent errors never select a candidate.",
        "",
        "<details>",
        "<summary>Every source/design/training-size/exposure/rule result</summary>",
        "",
        "| Source | Design | N | Exposure | Rule | Available train/test dates | Future risk score ± MCSE | Latent risk MSE ± MCSE |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for case in product(scenarios, designs, sizes):
        for target, rule in product(_TARGETS, ("selected", "fixed", "direct")):
            row = risks[(*case, rule, target)]
            lines.append(
                f"| {_label(case[0])} | {_label(case[1])} | {case[2]} | {_label(target)} | {rule} | {row.get('available_train_dates', 'unrecorded')}/{row['available_test_dates']} | {_estimate(row, 'risk_score')} | {_estimate(row, 'risk_mse')} |"
            )
    lines += [
        "",
        "</details>",
        "",
        "## Numerical acceptance and reference scope",
        "",
        "An optimizer certificate concerns the finite constrained objective in floating point. Analytic forcing-tail bounds are distinct from quadrature checks, candidate-J observation approximation and Monte Carlo sampling error. No winner, full-H identification, source-C recovery or spatially uniform theorem is an acceptance requirement.",
        "",
        "| Check | Recorded value |",
        "| --- | --- |",
        *_record_lines(results.get("acceptance", {})),
        "",
        "<details>",
        "<summary>Reference-solver checks and model/protocol metadata</summary>",
        "",
        "| Field | Recorded value |",
        "| --- | --- |",
        *_record_lines(results.get("reference_checks", []), "reference_checks"),
        *_record_lines(_model_summary(results.get("model", {})), "model"),
        *_record_lines(results.get("numerical_checks", {}), "numerical_checks"),
        *_record_lines(results.get("protocol", {}), "protocol"),
        "",
        "</details>",
    ]
    return lines


def write_report(output: str | Path, config: Any, results: dict) -> Path:
    """Write complete fixed panels after validating the declared experiment coverage."""
    _validate(config, results)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _geometry_figure(config, results, output)
    _approximation_figure(config, results, output)
    _performance_figure(config, results, output)
    validation_start = max(_value(config, "training_counts")) + 1
    validation_end = validation_start + _value(config, "validation_count", 128) - 1
    test_end = validation_end + _value(config, "test_count", 128)
    lines = [
        "# Partial-innovation observations: fixed-interval study",
        "",
        "This study estimates normalized innovation covariance VΔ at a fixed interval (weekly in the full protocol). It does not estimate source C or reconstruct innovations from sparse/noisy price levels. The [protocol](https://github.com/xrough/spde-covariance/blob/main/docs/partial-observations.md) separates finite Gram geometry, full-H functional spans, same-date risk-proxy availability and observation approximation.",
        "",
        f"Study: {_number(_value(config, 'name'))}; seed: {_value(config, 'seed')}; independent replications: {_value(config, 'replications')}. Preserve config.json, source/environment manifest and raw candidate/selection/replication outcomes with this report.",
        "",
        f"Reference M={_number(_value(config, 'reference_dimension'))}; Δ={_number(_value(config, 'delta'))} years; candidate J={_value(config, 'candidate_dimensions')}; relative penalties={_value(config, 'relative_regularizations')}. Finite solver covariance budget={_number(_value(config, 'numerical_error_budget'))}; numerical rank relative threshold={_number(_value(config, 'rank_rtol'))}; analytic reference-tail budget={_number(_value(config, 'tail_budget'))}. These budgets concern different errors.",
        "",
        f"Training prefixes: {_value(config, 'training_counts')}; validation dates {validation_start}–{validation_end}; disjoint test dates {validation_end + 1}–{test_end}. Fits remain frozen after validation, with no validation refit. The same full 16-contract parent observations supply every candidate J.",
        "",
        "## Population observation geometry",
        "",
        "![Finite geometry and full-H proxy availability](geometry_observability.png)",
        "",
        "The geometry panel uses the largest declared training size and retains every design and geometry dimension. Geometry does not depend on the source covariance. Finite Gram range statements concern the unrestricted fitted covariance sieve; generic covariance witnesses need not arise from a positive SPDE source. The proxy panel uses analytic full-H functional spans on actual test-date contracts, not finite-coordinate coincidence.",
        "",
        "## Observation approximation before sampling error",
        "",
        "![Observed-covariance representation error](observation_approximation.png)",
        "",
        "All source/design/dimension combinations are retained, using the largest declared training size's date-map frequencies. One full contract observation is generated before any candidate J. Population full-minus-sieve covariance includes retained-omitted cross terms and may be indefinite; it is not independent omitted-coordinate noise. Finite-source tails remain separate from this representation error.",
        "",
        *_tables(config, results),
        "",
        "The experiment uses known transport and fresh independent innovation-observation noise. No source-constrained identification theorem, uniform regularity rate, confidence-interval coverage or empirical futures forecast is established here.",
        "",
    ]
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
