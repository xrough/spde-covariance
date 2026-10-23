"""Fixed-selection reports for the infinite-rank transport foundation benchmark.

Finite-reference discrepancies, analytic tail bounds and Monte Carlo errors
remain separate. No outcome chooses a forcing level, projection or exposure.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullLocator


def _value(config: Any, name: str) -> Any:
    return config[name] if isinstance(config, Mapping) else getattr(config, name)


def _number(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _estimate(row: dict, name: str) -> str:
    mean, error = row.get(name + "_mean"), row.get(name + "_mcse")
    if mean is None:
        return "unavailable"
    return f"{_number(mean)} ± {_number(error)}" if error is not None else _number(mean)


def _declared(config: Any) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    dimensions = []
    for name in ("forcing_dimensions", "projection_dimensions"):
        values = tuple(_value(config, name))
        if not values or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in values
        ):
            raise ValueError(f"{name} must contain positive integers")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must be unique")
        dimensions.append(tuple(sorted(values)))
    deltas = tuple(float(value) for value in _value(config, "deltas"))
    if not deltas or any(not np.isfinite(value) or value <= 0 for value in deltas):
        raise ValueError("deltas must be finite and positive")
    if len(set(deltas)) != len(deltas):
        raise ValueError("deltas must be unique")
    if max(dimensions[1]) > max(dimensions[0]):
        raise ValueError("every projection must fit inside the forcing reference")
    return dimensions[0], dimensions[1], tuple(sorted(deltas))


def _key(row: dict) -> tuple[int, int, float]:
    return row["M"], row["J"], row["delta"]


def _finite(value: Any, context: str) -> None:
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


def _validate(config: Any, results: dict) -> None:
    forcing, projections, deltas = _declared(config)
    expected = {case for case in product(forcing, projections, deltas) if case[1] <= case[0]}
    _finite(results, "results")
    required = {
        "population": (
            "full_source_tail_bound",
            "full_innovation_tail_bound",
            "full_state_mse_tail_bound",
            "projected_innovation_tail_bound",
            "operational_covariance_tail_bound_max",
            "forcing_innovation_hs_error",
            "forcing_state_hs_error",
            "projection_innovation_hs_error",
            "source_to_innovation_hs_error",
            "leakage_mean_hs",
            "leakage_terminal_hs",
            "tail_resolved",
        ),
        "summary": (
            "oracle_covariance_mse_mean",
            "operational_covariance_mse_mean",
            "operational_centered_covariance_mse_mean",
            "projection_terminal_error_squared_mean",
            "oracle_gaussian_mse",
        ),
    }
    for name, fields in required.items():
        rows = results.get(name, [])
        keys = [_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name} contains duplicate resolution cases")
        if set(keys) != expected:
            raise ValueError(f"{name} must contain every declared resolution case exactly once")
        for row in rows:
            if any(field not in row or row[field] is None for field in fields):
                raise ValueError(f"{name} is missing a required report metric")
            if name == "population" and row.get("reference_M") != max(forcing):
                raise ValueError("population forcing reference must match the configuration")
            for key, value in row.items():
                if value is not None and (
                    key.endswith("_mcse") or (key.endswith("_mse_mean") and "_minus_" not in key)
                ):
                    if value < 0:
                        raise ValueError("MSE and MCSE values must be nonnegative")
            if name == "summary" and row["replications"] != _value(config, "replications"):
                raise ValueError("summary replication counts must match the configuration")
    exposures = sorted({row["exposure"] for row in results.get("risk_summary", [])})
    if not exposures:
        raise ValueError("risk_summary must declare at least one exposure")
    risk_keys = [(*_key(row), row["exposure"]) for row in results["risk_summary"]]
    expected_risks = {(*case, exposure) for case in expected for exposure in exposures}
    if len(risk_keys) != len(set(risk_keys)) or set(risk_keys) != expected_risks:
        raise ValueError("risk_summary must contain every declared case and exposure exactly once")
    for row in results["risk_summary"]:
        if row["replications"] != _value(config, "replications"):
            raise ValueError("risk replication counts must match the configuration")
        for name in ("oracle_risk_mse_mean", "operational_risk_mse_mean"):
            if name not in row or row[name] is None or row[name] < 0:
                raise ValueError("risk MSE must be present and nonnegative")
        for name in (
            "projected_risk_truth",
            "reference_risk_truth",
            "projection_bias",
            "forcing_bias",
            "leakage_bias",
        ):
            if name not in row or row[name] is None:
                raise ValueError("risk targets and signed biases must be present")
    kernels = results["kernels"]
    grid = np.asarray(kernels["maturities"], dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("kernel maturities must be a strictly increasing grid")
    if kernels["reference_M"] != max(forcing) or kernels["delta"] != max(deltas):
        raise ValueError("kernels must use the predeclared largest forcing level and interval")
    if kernels["horizon"] != _value(config, "horizon"):
        raise ValueError("kernel state horizon must match the configuration")
    for name in ("source", "innovation", "state"):
        if np.asarray(kernels[name]).shape != (len(grid), len(grid)):
            raise ValueError("kernel matrices must match the declared maturity grid")


def _style(axis, *, positive: bool = False) -> None:
    axis.grid(axis="y", color="0.88", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    if positive:
        values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in axis.lines])
        positive_values = values[values > 0]
        if len(positive_values) and np.all(values > 0):
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5), numticks=8))
            axis.yaxis.set_major_formatter(
                LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf))
            )
        elif len(positive_values):
            # Preserve an exact reference zero without an artificial decade-sized gap.
            axis.set_yscale("symlog", linthresh=float(min(positive_values) / 10))
            if np.all(values >= 0):
                axis.set_ylim(bottom=0)


def _interval_axis(axis, deltas: tuple[float, ...]) -> None:
    axis.set_xscale("log")
    labels = []
    for delta in deltas:
        frequency = round(1 / delta)
        labels.append(f"1/{frequency}" if np.isclose(delta * frequency, 1) else f"{delta:.4g}")
    axis.set_xticks(deltas, labels)
    axis.xaxis.set_minor_locator(NullLocator())


def _title(figure, *paragraphs: str) -> None:
    width = max(32, round(9 * figure.get_figwidth()))
    figure.suptitle("\n".join(fill(text, width=width) for text in paragraphs), fontsize=11)


def _save(figure, output: Path, name: str) -> None:
    figure.savefig(output / name, dpi=160, facecolor="white")
    plt.close(figure)


def _legend(figure, axes, columns: int = 4) -> None:
    handles, labels = [], []
    for axis in np.asarray(axes).flat:
        new_handles, new_labels = axis.get_legend_handles_labels()
        for handle, label in zip(new_handles, new_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=columns, frameon=False)


def _kernel_figure(results: dict, output: Path) -> None:
    kernels = results["kernels"]
    grid = np.asarray(kernels["maturities"])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, name, title in zip(
        axes,
        ("source", "innovation", "state"),
        ("Source C", "Normalized innovation V", "State K"),
        strict=True,
    ):
        plot = axis.pcolormesh(
            grid, grid, np.asarray(kernels[name]), shading="auto", cmap="viridis"
        )
        axis.set_title(title)
        axis.set_xlabel("Remaining maturity (years)")
        axis.set_ylabel("Remaining maturity (years)")
        colorbar = figure.colorbar(plot, ax=axis, shrink=0.8)
        colorbar.set_label("Curve-unit²" if name == "state" else "Curve-unit²/year")
    _title(
        figure,
        f"Evaluated covariance kernels at finite reference M={kernels['reference_M']}",
        f"Innovation interval Δ={kernels['delta']:.6g} years; state time t={kernels['horizon']:.6g} years. Separate color scales and units; infinite tail remains.",
    )
    _save(figure, output, "covariance_kernels.png")


def _resolution_figure(config: Any, results: dict, output: Path) -> None:
    forcing, projections, deltas = _declared(config)
    population = {_key(row): row for row in results["population"]}
    summaries = {_key(row): row for row in results["summary"]}
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")
    for index, (projection, delta) in enumerate(product(projections, deltas)):
        valid_forcing = [level for level in forcing if level >= projection]
        rows = [population[(level, projection, delta)] for level in valid_forcing]
        axes[0].plot(
            valid_forcing,
            [row["forcing_innovation_hs_error"] for row in rows],
            marker="o",
            linewidth=1,
            color=colors(index % 10),
            linestyle=("-", "--", ":")[index // 10 % 3],
            label=f"Forcing: J={projection}, Δ={delta:.4g}",
        )
    axes[0].set_title("Forcing cutoff (every J and Δ)")
    axes[0].set_xlabel("Retained forcing modes M")
    axes[0].set_ylabel("Projected V: HS discrepancy to reference M")
    axes[0].set_xticks(forcing)
    for index, delta in enumerate(deltas):
        rows = [population[(max(forcing), projection, delta)] for projection in projections]
        axes[1].plot(
            projections,
            [row["projection_innovation_hs_error"] for row in rows],
            "o-",
            color=colors(index % 10),
            linewidth=1,
            label=f"Projection: Δ={delta:.4g}",
        )
    axes[1].set_title("Projection at reference forcing")
    axes[1].set_xlabel("Retained curve coordinates J")
    axes[1].set_ylabel("Full-reference V: HS projection error")
    axes[1].set_xticks(projections)
    axes[2].plot(
        deltas,
        [
            population[(max(forcing), max(projections), delta)]["source_to_innovation_hs_error"]
            for delta in deltas
        ],
        "o-",
        color="0.2",
    )
    axes[2].set_title("Finite-interval transport effect at reference forcing")
    axes[2].set_xlabel("Observation interval Δ (years)")
    axes[2].set_ylabel("Full finite-reference HS norm of VΔ − C")
    _interval_axis(axes[2], deltas)
    for index, projection in enumerate(projections):
        rows = [summaries[(max(forcing), projection, delta)] for delta in deltas]
        axes[3].errorbar(
            deltas,
            [row["oracle_covariance_mse_mean"] for row in rows],
            yerr=[row.get("oracle_covariance_mse_mcse") or 0 for row in rows],
            color=colors(index % 10),
            fmt="o-",
            capsize=2,
            label=f"J={projection}: Monte Carlo",
        )
        axes[3].plot(
            deltas,
            [row["oracle_gaussian_mse"] for row in rows],
            color=colors(index % 10),
            linestyle="--",
            label=f"J={projection}: Gaussian expectation",
        )
    axes[3].set_title("Interval and sample size at reference forcing")
    axes[3].set_xlabel("Observation interval Δ (years)")
    axes[3].set_ylabel("Oracle innovation covariance HS MSE")
    _interval_axis(axes[3], deltas)
    for axis in axes:
        _style(axis, positive=True)
    _legend(figure, axes, columns=5)
    _title(
        figure,
        f"Separate resolution controls; reference M={max(forcing)}, fixed horizon={_value(config, 'horizon'):g} years",
        "All configured levels retained. Sampling panel has N=T/Δ and target VΔ; it is not a source-C estimation rate. Error bars: ±1 replication MCSE.",
    )
    _save(figure, output, "resolution_errors.png")


def _leakage_figure(config: Any, results: dict, output: Path) -> None:
    forcing, projections, deltas = _declared(config)
    population = {_key(row): row for row in results["population"]}
    risks = {(*_key(row), row["exposure"]): row for row in results["risk_summary"]}
    exposures = sorted({row["exposure"] for row in results["risk_summary"]})
    figure, axes = plt.subplots(
        2, len(deltas), figsize=(5 * len(deltas), 7), squeeze=False, layout="constrained"
    )
    colors = plt.get_cmap("tab10")
    for column, delta in enumerate(deltas):
        rows = [population[(max(forcing), projection, delta)] for projection in projections]
        for name, label, linestyle in (
            ("leakage_mean_hs", "Mean-time leakage", "-"),
            ("leakage_terminal_hs", "Last-increment leakage", "--"),
        ):
            axes[0, column].plot(
                projections, [row[name] for row in rows], "o", linestyle=linestyle, label=label
            )
        for index, exposure in enumerate(exposures):
            selected = [
                risks[(max(forcing), projection, delta, exposure)] for projection in projections
            ]
            for prefix, linestyle, method in (
                ("oracle", "-", "oracle innovation"),
                ("operational", "--", "projected-level residual"),
            ):
                axes[1, column].errorbar(
                    projections,
                    [row[f"{prefix}_risk_mse_mean"] for row in selected],
                    yerr=[row.get(f"{prefix}_risk_mse_mcse") or 0 for row in selected],
                    marker="o",
                    linestyle=linestyle,
                    color=colors(index % 10),
                    linewidth=1,
                    capsize=2,
                    label=f"{exposure.replace('_', ' ')}: {method}",
                )
        axes[0, column].set_title(f"Δ={delta:.6g} years")
        axes[0, column].set_ylabel("Population leakage covariance HS norm")
        axes[1, column].set_ylabel("Risk MSE against full-exposure reference")
        for axis in axes[:, column]:
            axis.set_xlabel("Retained curve coordinates J")
            axis.set_xticks(projections)
            _style(axis, positive=True)
    _legend(figure, axes, columns=3)
    _title(
        figure,
        f"Projection leakage and every declared exposure at reference M={max(forcing)}",
        "Risk target retains the full fixed exposure at the finite forcing reference. Projection/forcing biases remain; error bars: ±1 independent-replication MCSE.",
    )
    _save(figure, output, "leakage_risk.png")


def _check_lines(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        lines = []
        for key in sorted(value):
            lines.extend(_check_lines(value[key], f"{prefix}.{key}" if prefix else key))
        return lines
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value):
            lines.extend(_check_lines(item, f"{prefix}[{index}]"))
        return lines
    return [f"| {_number(prefix)} | {_number(value)} |"]


def _tables(config: Any, results: dict) -> list[str]:
    forcing, projections, deltas = _declared(config)
    population = {_key(row): row for row in results["population"]}
    summaries = {_key(row): row for row in results["summary"]}
    risk_lookup = {(*_key(row), row["exposure"]): row for row in results["risk_summary"]}
    exposures = sorted({row["exposure"] for row in results["risk_summary"]})
    cases = [case for case in product(forcing, projections, deltas) if case[1] <= case[0]]
    lines = [
        "## Population remainders and finite-reference differences",
        "",
        "Analytic truncation inequalities are evaluated in floating point. They are distinct from numerical quadrature/block-exponential agreement, which is a cross-check rather than an interval-arithmetic certificate. Full trace or state-MSE bounds and finite-J covariance HS bounds have different scopes and units. A finite reference retains a nonzero infinite tail.",
        "",
        "| M | Δ (years) | Full source trace tail | Full V trace tail bound | Full state MSE tail bound at T |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for level, delta in product(forcing, deltas):
        matching = [
            row for row in population.values() if row["M"] == level and row["delta"] == delta
        ]
        if not matching:
            continue
        row = matching[0]
        lines.append(
            f"| {level} | {_number(delta)} | {_number(row['full_source_tail_bound'])} | "
            f"{_number(row['full_innovation_tail_bound'])} | {_number(row['full_state_mse_tail_bound'])} |"
        )
    lines += [
        "",
        "Finite-J bounds below control omitted-source covariance in the declared observation projection; they are not full-state trace bounds. The operational bound includes both current innovations and unresolved previous-state leakage, with its maximum at the last preceding-state time T−Δ. Tail resolution is assessed by the runner against the configured budget, separately from numerical identity checks.",
        "",
        "| M | J | Δ | Projected V tail HS bound | Maximum operational tail HS bound | Tail resolved |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        row = population[case]
        lines.append(
            f"| {row['M']} | {row['J']} | {_number(row['delta'])} | "
            f"{_number(row['projected_innovation_tail_bound'])} | "
            f"{_number(row['operational_covariance_tail_bound_max'])} | {row['tail_resolved']} |"
        )
    lines += [
        "",
        "![Separate forcing, projection and interval effects](resolution_errors.png)",
        "",
        "The forcing-panel HS discrepancies are between J-projected finite covariances and the largest configured forcing reference. The projection panel reports the full finite-reference HS difference between V and its zero-extended J projection. A separate interval panel shows the full finite-reference HS difference between VΔ and C. The sampling panel uses oracle innovation sample covariance with N=T/Δ and compares its HS MSE with the exact Gaussian expectation. It changes both the finite-interval target V and the sample count; it is not a theorem about estimation of C. Only configured cases with J≤M are simulated.",
        "",
        "<details>",
        "<summary>All finite-reference population covariance differences</summary>",
        "",
        "| M | J | Δ | Projected V forcing difference | V projection error | Projected state K forcing difference | Mean leakage HS | Terminal leakage HS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        row = population[case]
        lines.append(
            f"| {row['M']} | {row['J']} | {_number(row['delta'])} | "
            + " | ".join(
                _number(row[key])
                for key in (
                    "forcing_innovation_hs_error",
                    "projection_innovation_hs_error",
                    "forcing_state_hs_error",
                    "leakage_mean_hs",
                    "leakage_terminal_hs",
                )
            )
            + " |"
        )
    lines += [
        "",
        "</details>",
        "",
        "## Monte Carlo covariance diagnostics",
        "",
        "Each MSE is a squared HS error, not a root error. Entries are independent-replication means ± MCSE. Within-path dates and coupled resolutions are not counted as independent replications. Oracle sample covariance is compared with projected V at the same M,J,Δ. Operational residual sample second moments are reported against both this innovation target and their correct time-average population covariance Rbar. The latter residuals are generally dependent and have a past-dependent conditional mean; the oracle iid Gaussian formula is not applied to them.",
        "",
        "| M | J | Δ | Oracle vs projected V | Gaussian expectation | Operational vs projected V | Operational vs Rbar |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        row = summaries[case]
        lines.append(
            f"| {row['M']} | {row['J']} | {_number(row['delta'])} | "
            f"{_estimate(row, 'oracle_covariance_mse')} | {_number(row['oracle_gaussian_mse'])} | "
            f"{_estimate(row, 'operational_covariance_mse')} | "
            f"{_estimate(row, 'operational_centered_covariance_mse')} |"
        )
    lines += [
        "",
        "<details>",
        "<summary>Coupled terminal-state and sample-leakage diagnostics</summary>",
        "",
        "State forcing coupling compares the M path with its shared-forcing largest-M reference; the projected coupling compares their J projections. State projection compares the M path with its own J projection. These are state squared-norm errors, distinct from covariance HS errors and from the remaining infinite-state mean-square bound.",
        "",
        "| M | J | Δ | State forcing coupling MSE | Projected forcing coupling MSE | State projection MSE | Mean squared residual leakage |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        row = summaries[case]
        lines.append(
            f"| {row['M']} | {row['J']} | {_number(row['delta'])} | "
            + " | ".join(
                _estimate(row, name)
                for name in (
                    "state_terminal_coupling_error_squared",
                    "projected_state_terminal_error_squared",
                    "projection_terminal_error_squared",
                    "leakage_sample_mse",
                )
            )
            + " |"
        )
    lines += [
        "",
        "</details>",
        "",
        "## Leakage and fixed-exposure risk",
        "",
        "![Leakage and exposure-risk diagnostics](leakage_risk.png)",
        "",
        "An exposure is fixed in infinite H and restricted to J coordinates without renormalization. Both risk MSEs below use the full-exposure risk at the largest finite forcing reference, not the smaller projected risk. Signed projection bias is projected risk minus full-exposure risk at the same M; signed forcing bias is full-exposure risk at M minus full-exposure reference risk. Leakage bias is the exposure quadratic form of the time-average leakage covariance. These are different sources of error, and finite-reference agreement does not remove the remaining infinite tail.",
        "",
        "### Paired error differences at reference forcing",
        "",
        "Each entry is operational minus oracle MSE, averaged from within-path loss differences, then summarized across independent replications. A positive value means larger operational error against the same target; covariance targets projected V, while risks target the full-exposure reference. MCSE is paired, not a combination of separate standard errors. Differences may have either sign and are not a winner acceptance gate.",
        "",
        "| J | Δ | Covariance MSE difference ± MCSE | "
        + " | ".join(
            f"{exposure.replace('_', ' ')} risk MSE difference ± MCSE" for exposure in exposures
        )
        + " |",
        "| ---: | ---: | ---: | " + " | ".join("---:" for _ in exposures) + " |",
    ]
    for projection, delta in product(projections, deltas):
        covariance_row = summaries[(max(forcing), projection, delta)]
        values = [
            _estimate(covariance_row, "operational_minus_oracle_covariance_mse"),
            *[
                _estimate(
                    risk_lookup[(max(forcing), projection, delta, exposure)],
                    "operational_minus_oracle_risk_mse",
                )
                for exposure in exposures
            ],
        ]
        lines.append(f"| {projection} | {_number(delta)} | " + " | ".join(values) + " |")
    lines += [
        "",
        "<details>",
        "<summary>Every forcing/projection/interval/exposure risk result</summary>",
        "",
        "| M | J | Δ | Exposure | Oracle risk MSE ± MCSE | Operational risk MSE ± MCSE | Projection bias | Forcing bias | Leakage bias |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(results["risk_summary"], key=lambda row: (*_key(row), row["exposure"])):
        lines.append(
            f"| {row['M']} | {row['J']} | {_number(row['delta'])} | {_number(row['exposure'])} | "
            f"{_estimate(row, 'oracle_risk_mse')} | {_estimate(row, 'operational_risk_mse')} | "
            f"{_number(row.get('projection_bias'))} | {_number(row.get('forcing_bias'))} | "
            f"{_number(row.get('leakage_bias'))} |"
        )
    lines += [
        "",
        "</details>",
        "",
        "## Numerical checks and acceptance",
        "",
        "Acceptance is the runner's predeclared numerical and reference-resolution audit. It does not demand superiority of any estimator or claim new statistical consistency. An unresolved tail in a smoke run remains explicitly unresolved; it is not upgraded to a passed full-study tail condition.",
        "",
        "| Check | Recorded value |",
        "| --- | --- |",
        *_check_lines(results.get("acceptance", {})),
        "",
        "<details>",
        "<summary>Numerical cross-checks and restricted-dimension fits</summary>",
        "",
        "Covariance fits are limited to the configured small fit dimensions; population and direct sample-covariance diagnostics at large J do not imply a lifted large-J optimization. Solver certificates concern the finite objective in floating point, not statistical error or an infinite-dimensional proof.",
        "",
        "| Check | Recorded value |",
        "| --- | --- |",
        *_check_lines(results.get("numerical_checks", {})),
        *_check_lines(results.get("reference_checks", []), "reference_checks"),
        *_check_lines(results.get("fit_summary", []), "fit_summary"),
        "",
        "</details>",
    ]
    return lines


def write_report(output: str | Path, config: Any, results: dict) -> Path:
    """Write one report and three conventional figures without selecting outcomes."""
    _validate(config, results)
    forcing, projections, deltas = _declared(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _kernel_figure(results, output)
    _resolution_figure(config, results, output)
    _leakage_figure(config, results, output)
    lines = [
        "# Fixed infinite-rank transport: foundation benchmark",
        "",
        "This is a reproducible finite-representation study of one fixed infinite-rank additive transport model. It is not a new estimator theorem. Source C, normalized innovation VΔ and state Kt have different targets and units. The protocol and projection/tail formulas are in [the methods protocol](https://github.com/xrough/spde-covariance/blob/main/docs/infinite-transport.md); retain the machine-readable configuration and provenance alongside this report.",
        "",
        f"Study: {_number(_value(config, 'name'))}; seed: {_value(config, 'seed')}; independent path replications: {_value(config, 'replications')}; horizon: {_number(_value(config, 'horizon'))} years.",
        "",
        f"Declared forcing modes M: {forcing}; curve projection dimensions J: {projections}; intervals Δ in years: {tuple(_number(delta) for delta in deltas)}. Source exponent: {_number(_value(config, 'source_exponent'))}; infinite source trace: {_number(_value(config, 'source_trace'))}. Prefix weights are never renormalized.",
        "",
        "## Covariance kernels and fixed figure selections",
        "",
        "![Finite-reference evaluated covariance kernels](covariance_kernels.png)",
        "",
        f"Kernels use full forcing reference M={max(forcing)}, the largest configured interval Δ={_number(max(deltas))}, the declared state horizon and a common maturity grid. They are not J-projected kernels and not an infinite model computed exactly. Color scales are separate: C and V have covariance-per-year units; K has state-covariance units. Evaluated kernels describe curve evaluations, not automatically a Lebesgue kernel for the Sobolev-space operator.",
        "",
        "Figure choices use configured resolutions, never smallest realized errors. The forcing panel retains every valid J/Δ combination. Projection, interval, leakage and risk panels use the largest forcing reference and every configured projection/frequency/exposure. Positive errors use logarithmic axes; exact zeros remain visible on symmetric-log axes. Error bars denote one Monte Carlo standard error rather than confidence intervals.",
        "",
        *_tables(config, results),
        "",
        "## Limits of the experiment",
        "",
        "Known transport, zero initial state, no extra drift, deterministic trace-class source and clean projected levels are built into this foundation. No price-level measurement-noise inverse problem, unknown generator, stochastic volatility, new inference rate or market forecasting claim is tested here. Finite-reference coupling discrepancies and analytic infinite remainders are complementary diagnostics, not interchangeable guarantees.",
        "",
    ]
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
