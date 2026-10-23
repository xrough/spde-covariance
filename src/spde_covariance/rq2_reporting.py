"""Deterministic reports for the local-estimation research experiments.

Panels are selected by declared scenario, endpoint and window, never by errors
against latent truth. Error bars are Monte Carlo standard errors, not intervals
for an unknown covariance or coverage claims.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCENARIOS = ("constant", "scalar", "rotating")
_DESIGNS = ("full", "dense", "split", "joint8", "joint64")
_METHODS = ("constrained_ridge", "projected_ridge", "direct", "empirical")
_LABELS = {
    "constant": "Constant covariance",
    "scalar": "Changing scale",
    "rotating": "Rotating covariance",
    "abrupt": "Abrupt change (stress)",
    "full": "Identity control",
    "dense": "Joint every date",
    "split": "Separate pairs",
    "joint8": "Joint every 8",
    "joint64": "Joint every 64",
    "constrained_ridge": "Constrained ridge",
    "projected_ridge": "Projected ridge",
    "direct": "Direct risk",
    "empirical": "Empirical control",
    "near_spread": "Within-pair spread",
    "cross_spread": "Cross-pair spread",
    "butterfly": "Slope-neutral butterfly",
}
_COLORS = dict(zip(_DESIGNS, plt.get_cmap("tab10").colors[:5], strict=True))
_MARKERS = {"constrained_ridge": "o", "projected_ridge": "s", "direct": "^", "empirical": "D"}


def _value(config: Any, name: str, default: Any = None) -> Any:
    return (
        config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)
    )


def _label(value: str) -> str:
    return _LABELS.get(value, value.replace("_", " "))


def _primary(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("primary", True)]


def _number(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.5g}" if np.isfinite(value) else str(value)
    return str(value)


def _mean_mcse(row: dict, metric: str, *, suffix: bool = True) -> str:
    mean = row.get(metric + "_mean" if suffix else metric)
    mcse = row.get(metric + "_mcse")
    if mean is None:
        return "unavailable"
    return f"{_number(mean)} ± {_number(mcse)}"


def _selection(config: Any, results: dict) -> tuple[int, int]:
    endpoints = _value(config, "endpoints", ())
    windows = _value(config, "windows", ())
    if not endpoints:
        endpoints = [row["endpoint"] for row in _primary(results.get("summary", []))]
    if not windows:
        windows = [row["window"] for row in _primary(results.get("summary", []))]
    endpoint = max(endpoints)
    window = 64 if 64 in windows else max(windows)
    return endpoint, window


def _errorbar(ax, rows, metric, *, label, color, marker="o", linestyle="-"):
    usable = [row for row in rows if row.get(metric + "_mean") is not None]
    if not usable:
        return
    usable.sort(key=lambda row: row["window"])
    ax.errorbar(
        [row["window"] for row in usable],
        [row[metric + "_mean"] for row in usable],
        yerr=[row.get(metric + "_mcse") or 0 for row in usable],
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.3,
        markersize=4,
        capsize=2,
        label=label,
    )


def _window_figure(config: Any, results: dict, output: Path) -> None:
    endpoint, _ = _selection(config, results)
    rows = [
        row
        for row in _primary(results.get("summary", []))
        if row["endpoint"] == endpoint and row["method"] == "constrained_ridge"
    ]
    scenarios = [
        scenario for scenario in _SCENARIOS if any(r["scenario"] == scenario for r in rows)
    ]
    fig, axes = plt.subplots(2, max(len(scenarios), 1), figsize=(13, 7), squeeze=False)
    for column, scenario in enumerate(scenarios):
        for line, (metric, ylabel) in enumerate(
            (
                ("operator_hs_relative_error", "Relative covariance HS error"),
                ("exposure_risk_rmse", "Exposure-risk RMSE"),
            )
        ):
            ax = axes[line, column]
            for design in _DESIGNS:
                matching = [
                    row for row in rows if row["scenario"] == scenario and row["design"] == design
                ]
                _errorbar(ax, matching, metric, label=_label(design), color=_COLORS[design])
            ax.set_title(_label(scenario))
            ax.set_xlabel("Trailing window (intervals)")
            if column == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(_value(config, "windows", sorted({row["window"] for row in rows})))
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(
        f"Current-covariance estimation at endpoint {endpoint}; primary ridge\nError bars: ±1 Monte Carlo standard error",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.92))
    fig.savefig(output / "window_error.png", dpi=160)
    plt.close(fig)


def _coverage_figure(config: Any, results: dict, output: Path) -> None:
    geometry = [row for row in _primary(results.get("geometry", [])) if row["design"] == "joint64"]
    rows = [
        row
        for row in _primary(results.get("risk_summary", []))
        if row["design"] == "joint64" and row["target"] == "cross_spread"
    ]
    scenarios = [
        scenario for scenario in _SCENARIOS if any(r["scenario"] == scenario for r in rows)
    ]
    fig, axes = plt.subplots(1, 1 + len(scenarios), figsize=(15, 4.5), squeeze=False)
    rank_ax = axes[0, 0]
    endpoints = sorted(_value(config, "endpoints", {row["endpoint"] for row in geometry}))
    windows = sorted(_value(config, "windows", {row["window"] for row in geometry}))
    rank_scenario = next(
        (
            scenario
            for scenario in _SCENARIOS
            if any(row["scenario"] == scenario for row in geometry)
        ),
        None,
    )
    rank_values = np.full((len(endpoints), len(windows)), np.nan)
    for row in geometry:
        if (
            row["scenario"] == rank_scenario
            and row["endpoint"] in endpoints
            and row["window"] in windows
        ):
            rank_values[endpoints.index(row["endpoint"]), windows.index(row["window"])] = row[
                "rank"
            ]
    if rank_values.size:
        image = rank_ax.imshow(
            rank_values, origin="upper", aspect="auto", vmin=0, vmax=10, cmap="viridis"
        )
        for index in np.ndindex(rank_values.shape):
            if np.isfinite(rank_values[index]):
                rank_ax.text(
                    index[1],
                    index[0],
                    f"{rank_values[index]:.0f}/10",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        fig.colorbar(image, ax=rank_ax, label="Lifted Gram rank", shrink=0.75)
    rank_ax.set_xticks(range(len(windows)), windows)
    rank_ax.set_yticks(range(len(endpoints)), endpoints)
    rank_ax.set_xlabel("Trailing window (intervals)")
    rank_ax.set_ylabel("Endpoint (interval)")
    rank_ax.set_title("Local observation coverage")
    endpoint_colors = plt.get_cmap("tab10").colors
    for column, scenario in enumerate(scenarios, start=1):
        ax = axes[0, column]
        for index, endpoint in enumerate(endpoints):
            for method, linestyle, marker in (
                ("constrained_ridge", "-", "o"),
                ("direct", "--", "^"),
            ):
                matching = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["endpoint"] == endpoint
                    and row["method"] == method
                    and (method != "direct" or row.get("identified", True))
                ]
                _errorbar(
                    ax,
                    [{**row, "mse_mean": row.get("mse")} for row in matching],
                    "mse",
                    label=f"t={endpoint}" if method == "constrained_ridge" else "_nolegend_",
                    color=endpoint_colors[index % len(endpoint_colors)],
                    marker=marker,
                    linestyle=linestyle,
                )
        ax.set_title(_label(scenario))
        ax.set_xlabel("Trailing window (intervals)")
        ax.set_ylabel("Cross-pair risk MSE")
        ax.set_xticks(windows)
        ax.grid(alpha=0.25)
    if scenarios:
        if axes[0, -1].get_legend_handles_labels()[0]:
            axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Joint observations every 64 dates: coverage and current spread risk", fontsize=12)
    fig.text(
        0.5,
        0.02,
        "Solid circles: constrained ridge. Dashed triangles: direct risk, where available. Error bars: ±1 MCSE. Rank depends only on the observation design.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(output / "coverage.png", dpi=160)
    plt.close(fig)


def _risk_figure(config: Any, results: dict, output: Path) -> None:
    endpoint, window = _selection(config, results)
    rows = [
        row
        for row in _primary(results.get("risk_summary", []))
        if row["endpoint"] == endpoint and row["window"] == window
    ]
    scenarios = [
        scenario for scenario in _SCENARIOS if any(row["scenario"] == scenario for row in rows)
    ]
    targets = [
        target
        for target in ("near_spread", "cross_spread", "butterfly")
        if any(row["target"] == target for row in rows)
    ]
    fig, axes = plt.subplots(
        max(len(targets), 1),
        max(len(scenarios), 1),
        figsize=(13, max(3.2 * len(targets), 4)),
        squeeze=False,
    )
    method_colors = dict(zip(_METHODS, plt.get_cmap("tab10").colors[:4], strict=True))
    for line, target in enumerate(targets):
        for column, scenario in enumerate(scenarios):
            ax = axes[line, column]
            for index, method in enumerate(_METHODS):
                matching = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["target"] == target
                    and row["method"] == method
                    and (method != "direct" or row.get("identified", True))
                    and row.get("mse") is not None
                ]
                if not matching:
                    continue
                ax.errorbar(
                    [_DESIGNS.index(row["design"]) + (index - 1.5) * 0.13 for row in matching],
                    [row["mse"] for row in matching],
                    yerr=[row.get("mse_mcse") or 0 for row in matching],
                    fmt=_MARKERS[method],
                    color=method_colors[method],
                    markersize=4,
                    capsize=2,
                    label=_label(method),
                )
            ax.set_xticks(
                range(len(_DESIGNS)),
                ["Identity", "Joint", "Split", "Joint 8", "Joint 64"],
                rotation=30,
            )
            ax.set_title(f"{_label(scenario)} / {_label(target)}", fontsize=10)
            if column == 0:
                ax.set_ylabel("Current exposure-risk MSE")
            ax.grid(axis="y", alpha=0.25)
    handles, labels = [], []
    for ax in axes.flat:
        new_handles, new_labels = ax.get_legend_handles_labels()
        for handle, label in zip(new_handles, new_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(
        f"Direct versus covariance-based risk; endpoint {endpoint}, window {window}\nPrimary ridge; ±1 MCSE. Unavailable direct targets are omitted.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    fig.savefig(output / "risk_comparison.png", dpi=160)
    plt.close(fig)


def write_report(output: Path, config: Any, results: dict) -> None:
    """Write a Markdown report and three figures from completed experiment results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    endpoint, window = _selection(config, results)
    for renderer in (_window_figure, _coverage_figure, _risk_figure):
        renderer(config, results, output)
    acceptance = results.get("acceptance", {})
    lines = [
        "# Research question 2: local covariance and targeted risk",
        "",
        f"Configuration `{_value(config, 'name', 'rq2')}`: {_value(config, 'replications')} paired replications, {_value(config, 'intervals')} independent innovations, Δ={_value(config, 'delta', 1 / 52):.8g} years, seed `{_value(config, 'seed')}`.",
        "",
        "The target at endpoint n is the current finite-interval innovation covariance V_n, or a declared exposure variance aᵀV_n a. The experiment directly samples Gaussian innovations; transport extraction is covered by the separate Stage 1 benchmark. No instantaneous diffusion covariance or forecast is fitted here.",
        "",
        "All windows and ridge strengths are declared before simulation. Latent truths enter evaluation and analytic controls only, never parameter selection. Observations and noise are paired across methods, windows, and nested maturity designs. The noiseless identity design is a separate information control.",
        "",
        "## Locality and coverage",
        "",
        "![Window length and current covariance error](window_error.png)",
        "",
        "The first figure uses the latest declared endpoint and the primary constrained ridge. Each curve is a separate observation design; panels distinguish scale variation from rotation with unchanged eigenvalues. Error bars are ±1 Monte Carlo standard error (MCSE), not confidence intervals. The abrupt-change scenario is a separate stress case in the tables and raw results.",
        "",
        "![Local pair coverage and cross-pair risk](coverage.png)",
        "",
        "The rank heatmap uses joint observations every 64 dates and every declared endpoint/window. Rank is a property of the design and is the same across covariance paths. Risk panels retain separate endpoint curves. Full pooled rank identifies a common covariance; it does not identify the terminal covariance of an unrestricted changing path. A missing direct estimate remains unavailable; regularized covariance-based risks are still reported for unidentified targets and must be interpreted through their diagnostics.",
        "",
        "For A_i(B)=O_i B O_iᵀ, Q=Σ_i w_i A_i* A_i. The observation-weighted moment bias is b=Σ_i w_i A_i* A_i(V_i−V_n). Before PSD/trace constraints, the population ridge solution obeys (Q+λI)(B°−V_n)=b−λV_n. The saved moment-bias norm is measured in this normal-equation space; it is not itself a covariance error.",
        "",
        f"## Declared snapshot: endpoint {endpoint}, window {window}",
        "",
        "The following table reports every design and scenario for the primary constrained ridge at this predetermined snapshot. The machine-readable outputs retain all endpoints, windows, methods and ridge sensitivity settings.",
        "",
        "| Scenario | Design | Relative covariance HS error ± MCSE | Observed covariance RMSE ± MCSE | Exposure-risk RMSE ± MCSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    summary = [
        row
        for row in _primary(results.get("summary", []))
        if row["endpoint"] == endpoint
        and row["window"] == window
        and row["method"] == "constrained_ridge"
    ]
    for row in summary:
        values = [
            _mean_mcse(row, metric)
            for metric in (
                "operator_hs_relative_error",
                "observed_covariance_rmse",
                "exposure_risk_rmse",
            )
        ]
        lines.append(
            f"| {_label(row['scenario'])} | {_label(row['design'])} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "HS error uses the declared orthonormal H coordinates. Observed covariance RMSE projects terminal covariance error through each local observation map: (Σ_i w_i ‖O_i(B−V_n)O_iᵀ‖²_F)¹ᐟ². It measures the visible part of the terminal error. Exposures are normalized to unit H norm. Finite-sample risk estimates may be negative for the linear direct estimator.",
        "",
        "## Local geometry and temporal moment bias",
        "",
        "This is the same declared snapshot. Target availability is the common-covariance range check at the configured rank threshold. The moment-bias norm is deterministic and therefore has no MCSE.",
        "",
        "| Scenario | Design | Gram rank | Moment bias HS | Targets passing range check |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in _primary(results.get("geometry", [])):
        if row["endpoint"] == endpoint and row["window"] == window:
            available = (
                ", ".join(
                    _label(name)
                    for name, target in row.get("targets", {}).items()
                    if target.get("identified")
                )
                or "none"
            )
            lines.append(
                f"| {_label(row['scenario'])} | {_label(row['design'])} | {row['rank']}/10 | {_number(row.get('moment_bias_hs'))} | {available} |"
            )
    lines += [
        "",
        "## Direct versus covariance-based risks",
        "",
        "![Direct and covariance plug-in risk estimates](risk_comparison.png)",
        "",
        "For T=aaᵀ in ran(Q), set u=Q†svec(T) and U_i=smat(A_i u). The direct estimator is Σ_i w_i〈U_i,Z_i〉. Its exact bias against the terminal target is Σ_i w_i〈U_i,A_i(V_i−V_n)〉. Under the stated independent Gaussian model, its variance is 2Σ_i w_i² tr(U_iΣ_iU_iΣ_i), with Σ_i=O_iV_iO_iᵀ+N_i. Saved analytic MSE is bias²+variance. This range check is a common-covariance observability check; under a changing path, a direct estimate need not be unbiased for terminal risk.",
        "",
    ]
    risk_rows = [
        row
        for row in _primary(results.get("risk_summary", []))
        if row["endpoint"] == endpoint and row["window"] == window
    ]
    scenario = "rotating" if any(row["scenario"] == "rotating" for row in risk_rows) else "constant"
    lines += [
        f"Direct-estimator diagnostics below use the predeclared {_label(scenario).lower()} scenario at the same snapshot. Values are empirical bias/MSE with MCSE, followed by exact Gaussian-model values; discrepancies are Monte Carlo variation, not a numerical certificate.",
        "",
        "| Design | Target | Identified | Bias ± MCSE | MSE ± MCSE | Analytic bias | Analytic variance | Analytic MSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in risk_rows:
        if row["scenario"] != scenario or row["method"] != "direct":
            continue
        values = [_mean_mcse(row, metric, suffix=False) for metric in ("bias", "mse")]
        values += [
            _number(row.get(metric))
            for metric in ("analytic_bias", "analytic_variance", "analytic_mse")
        ]
        lines.append(
            f"| {_label(row['design'])} | {_label(row['target'])} | {row.get('identified')} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "### Paired risk-error differences",
        "",
        f"These differences use the same {_label(scenario).lower()} snapshot. Positive MSE difference means the comparison method has larger squared error than constrained ridge. MCSE is calculated from replication-level paired differences, not from independent standard errors.",
        "",
        "| Design | Target | Comparison method | MSE difference ± MCSE |",
        "| --- | --- | --- | ---: |",
    ]
    for row in _primary(results.get("paired_differences", [])):
        if row["scenario"] == scenario and row["endpoint"] == endpoint and row["window"] == window:
            lines.append(
                f"| {_label(row['design'])} | {_label(row['target'])} | {_label(row['comparison_method'])} | {_number(row.get('mse_difference'))} ± {_number(row.get('mcse'))} |"
            )
    lines += [
        "",
        "## Numerical acceptance and reproducibility",
        "",
        f"Acceptance: **{'PASS' if acceptance.get('passed') else 'FAIL'}**.",
        "",
        "Run acceptance checks numerical feasibility and certificates and selected independent convex references. Separate tests check deterministic design ranks, target ranges, covariance paths and temporal boundaries. It does not require a statistical ranking of methods or a favorable risk result. Numerical covariance-error certificates bound distance to the constrained optimizer, not estimation error against latent truth.",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for key, value in acceptance.items():
        lines.append(f"| {key.replace('_', ' ')} | {_number(value)} |")
    lines += [
        "",
        f"Independent convex-reference cases recorded: {len(results.get('reference_checks', []))}.",
        "",
        "`results.json` retains covariance summaries, risk bias/MSE, paired differences, analytic direct-risk controls, observation spectra and target diagnostics, moment bias, reference checks and acceptance. Replication-level CSV files retain the paired statistical measurements. `manifest.json` records the resolved configuration, source hashes, revision, environment and solver settings; provenance must be interpreted with its dirty-source flag.",
        "",
        "The protocol is documented in [research-question-2.md](../../../docs/research-question-2.md) in a standard experiments/runs/<run> output directory. This finite-dimensional Gaussian study does not establish confidence-interval coverage, recovery of instantaneous C, robustness to price-level noise or market forecasting performance.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
