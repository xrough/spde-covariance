"""Observed-score reports for the predeclared frozen-block forecast experiment.

Figures use configured scenarios, designs and horizons. Latent simulation
diagnostics never choose a panel, rule or window. Uncertainty is supplied by the
experiment after aggregation within independent replication clusters.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCENARIOS = ("constant", "scalar", "rotating", "abrupt")
_DESIGNS = ("full", "dense", "split", "joint8", "joint64")
_RULES = ("static", "fixed", "selected")
_TARGETS = ("near_spread", "cross_spread", "butterfly")
_LABELS = {
    "constant": "Constant covariance",
    "scalar": "Changing scale",
    "rotating": "Rotating covariance",
    "abrupt": "Abrupt change (stress)",
    "full": "Identity",
    "dense": "Joint each date",
    "split": "Separate pairs",
    "joint8": "Joint every 8",
    "joint64": "Joint every 64",
    "near_spread": "Within-pair spread",
    "cross_spread": "Cross-pair spread",
    "butterfly": "Slope-neutral butterfly",
}
_COLORS = {"static": "0.45", "fixed": "tab:blue", "selected": "tab:orange"}
_MARKERS = {"static": "s", "selected": "o"}


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
    mean, mcse = row[metric + "_mean"], row.get(metric + "_mcse")
    if mcse is None:
        return f"{_number(mean)} (MCSE unavailable)"
    return f"{_number(mean)} ± {_number(mcse)}"


def _difference(row: dict | None) -> str:
    if row is None or row.get("mean_difference") is None:
        return "unavailable"
    if row.get("mcse") is None:
        return f"{_number(row['mean_difference'])} (MCSE unavailable)"
    return f"{_number(row['mean_difference'])} ± {_number(row['mcse'])}"


def _declared(config: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scenarios = tuple(_value(config, "scenarios", _SCENARIOS))
    designs = tuple(_value(config, "designs", _DESIGNS))
    if not scenarios or any(scenario not in _SCENARIOS for scenario in scenarios):
        raise ValueError("report scenarios must be predeclared RQ3 scenarios")
    if not designs or any(design not in _DESIGNS for design in designs):
        raise ValueError("report designs must be predeclared RQ3 designs")
    return (
        tuple(scenario for scenario in _SCENARIOS if scenario in scenarios),
        tuple(design for design in _DESIGNS if design in designs),
    )


def _validate(config: Any, results: dict) -> None:
    scenarios, designs = _declared(config)
    windows = tuple(_value(config, "windows", (16, 32, 64, 128)))
    test_origins = tuple(_value(config, "test_origins", (256, 272, 304, 448)))
    for name in (
        "summary",
        "risk_summary",
        "truth_summary",
        "paired_differences",
        "horizon_summary",
        "origin_summary",
        "selections",
        "geometry",
    ):
        for row in results.get(name, []):
            if row.get("scenario", scenarios[0]) not in scenarios:
                raise ValueError(f"{name} contains a scenario outside the declared experiment")
            if row.get("design", designs[0]) not in designs:
                raise ValueError(f"{name} contains a design outside the declared experiment")
            if "rule" in row and row["rule"] not in _RULES:
                raise ValueError(f"{name} contains an undeclared forecast rule")
            for key, value in row.items():
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError(f"{name}.{key} must be finite or None")
    for row in results.get("selections", []):
        if row.get("selected_window") not in windows:
            raise ValueError("selected window is outside the predeclared candidate set")
        if any(candidate["window"] not in windows for candidate in row.get("candidates", [])):
            raise ValueError("validation candidate is outside the predeclared window set")
        complete = row.get("selection_complete_interval")
        if complete is not None and complete > min(test_origins):
            raise ValueError("selection must be completed before test forecasts")
    for row in results.get("geometry", []):
        if row.get("training_last_interval", row["origin"]) != row["origin"]:
            raise ValueError("geometry training bounds must end at the forecast origin")
        if row.get("training_first_interval", row["origin"] - row["window"] + 1) != (
            row["origin"] - row["window"] + 1
        ):
            raise ValueError("geometry training bounds must match the declared trailing window")
    for row in results.get("risk_summary", []):
        if row.get("available_scores") == 0 and row.get("risk_score_mean") is not None:
            raise ValueError("an unavailable future risk proxy cannot have a reported score")


def _rule_label(rule: str, config: Any) -> str:
    if rule == "static":
        return f"Static ({_value(config, 'static_window', 128)})"
    if rule == "fixed":
        return f"Fixed local ({_value(config, 'fixed_window', 128)})"
    return "Selected window"


def _style_axis(axis) -> None:
    axis.grid(axis="y", color="0.87", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _save(fig, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=160, facecolor="white")
    plt.close(fig)


def _title(fig, *paragraphs: str) -> None:
    """Keep captions within the figure, including reduced CI configurations."""
    width = max(30, round(8 * fig.get_figwidth()))
    fig.suptitle("\n".join(fill(paragraph, width=width) for paragraph in paragraphs), fontsize=12)


def _paired_points(axis, rows: list[dict], designs: tuple[str, ...], config: Any) -> None:
    axis.axhline(0, color="0.25", linewidth=0.8, linestyle="--")
    for index, rule in enumerate(("static", "selected")):
        usable = [
            row
            for row in rows
            if row["comparison_rule"] == rule and row.get("mean_difference") is not None
        ]
        if usable:
            axis.errorbar(
                [designs.index(row["design"]) + (index - 0.5) * 0.16 for row in usable],
                [row["mean_difference"] for row in usable],
                yerr=[row.get("mcse") or 0 for row in usable],
                fmt=_MARKERS[rule],
                color=_COLORS[rule],
                markersize=4,
                capsize=2,
                label=_rule_label(rule, config),
            )
    axis.set_xticks(range(len(designs)), [_label(design) for design in designs], rotation=25)
    axis.set_xlim(-0.5, len(designs) - 0.5)
    _style_axis(axis)


def _legend(fig, axes) -> None:
    handles, labels = [], []
    for axis in np.asarray(axes).flat:
        new_handles, new_labels = axis.get_legend_handles_labels()
        for handle, label in zip(new_handles, new_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)


def _forecast_figure(config: Any, results: dict, output: Path) -> None:
    scenarios, designs = _declared(config)
    columns = min(2, len(scenarios))
    rows = (len(scenarios) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 3.8 * rows), squeeze=False, layout="constrained"
    )
    for axis, scenario in zip(axes.flat, scenarios, strict=False):
        matching = [
            row
            for row in results.get("paired_differences", [])
            if row["scenario"] == scenario
            and row.get("metric") == "observed_score"
            and row.get("baseline_rule", "fixed") == "fixed"
        ]
        _paired_points(axis, matching, designs, config)
        axis.set_title(_label(scenario))
        axis.set_ylabel("Observed-score difference")
    for axis in list(axes.flat)[len(scenarios) :]:
        axis.set_visible(False)
    _legend(fig, axes)
    _title(
        fig,
        f"Future observed covariance scores: comparison minus fixed local ({_value(config, 'fixed_window', 128)})",
        "Lower favors the comparison rule; error bars are ±1 replication-cluster MCSE",
    )
    _save(fig, output, "forecast_scores.png")


def _horizon_figure(config: Any, results: dict, output: Path) -> None:
    scenarios, designs = _declared(config)
    horizon = _value(config, "horizon", 16)
    fig, axes = plt.subplots(
        len(scenarios),
        len(designs),
        figsize=(3.1 * len(designs), 2.6 * len(scenarios)),
        squeeze=False,
        layout="constrained",
    )
    for line, scenario in enumerate(scenarios):
        for column, design in enumerate(designs):
            axis = axes[line, column]
            for rule in _RULES:
                matching = sorted(
                    (
                        row
                        for row in results.get("horizon_summary", [])
                        if row["scenario"] == scenario
                        and row["design"] == design
                        and row["rule"] == rule
                        and row.get("observed_score_mean") is not None
                    ),
                    key=lambda row: row["horizon"],
                )
                if not matching:
                    continue
                x = np.array([row["horizon"] for row in matching])
                means = np.array([row["observed_score_mean"] for row in matching])
                errors = np.array([row.get("observed_score_mcse") or 0 for row in matching])
                axis.plot(
                    x, means, color=_COLORS[rule], linewidth=1.2, label=_rule_label(rule, config)
                )
                axis.fill_between(
                    x, means - errors, means + errors, color=_COLORS[rule], alpha=0.12, linewidth=0
                )
            axis.set_title(f"{_label(scenario)} / {_label(design)}", fontsize=9)
            axis.set_xlim(1, max(horizon, 2))
            axis.set_xticks(sorted({1, (horizon + 1) // 2, horizon}))
            if line == len(scenarios) - 1:
                axis.set_xlabel("Weekly forecast horizon")
            if column == 0:
                axis.set_ylabel("Reduced observed score")
            _style_axis(axis)
    _legend(fig, axes)
    _title(
        fig,
        "Reduced future observed scores by weekly horizon",
        "Future measurement dimensions may vary; compare rules on the same future maps. Bands: ±1 cluster MCSE",
    )
    _save(fig, output, "horizon_sensitivity.png")


def _risk_figure(config: Any, results: dict, output: Path) -> None:
    scenarios, designs = _declared(config)
    fig, axes = plt.subplots(
        len(_TARGETS),
        len(scenarios),
        figsize=(3.6 * len(scenarios), 9),
        squeeze=False,
        layout="constrained",
    )
    for line, target in enumerate(_TARGETS):
        for column, scenario in enumerate(scenarios):
            axis = axes[line, column]
            matching = [
                row
                for row in results.get("paired_differences", [])
                if row["scenario"] == scenario
                and row.get("metric") == "risk_score"
                and row.get("target") == target
                and row.get("baseline_rule", "fixed") == "fixed"
            ]
            _paired_points(axis, matching, designs, config)
            availability = {
                row["design"]: row.get("available_scores", 0)
                for row in results.get("risk_summary", [])
                if row["scenario"] == scenario
                and row["target"] == target
                and row["rule"] == "fixed"
            }
            for index, design in enumerate(designs):
                if availability.get(design, 0) == 0:
                    axis.annotate(
                        "unavailable",
                        (index, 0),
                        xytext=(0, 4),
                        textcoords="offset points",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="0.4",
                    )
            axis.set_title(f"{_label(scenario)}\n{_label(target)}", fontsize=9)
            if column == 0:
                axis.set_ylabel("Risk-score difference")
    _legend(fig, axes)
    _title(
        fig,
        f"Future exposure-risk scores: comparison minus fixed local ({_value(config, 'fixed_window', 128)})",
        "±1 replication-cluster MCSE; unavailable future proxies are omitted, never filled with zero",
    )
    _save(fig, output, "risk_evaluation.png")


def _tables(config: Any, results: dict) -> list[str]:
    scenarios, designs = _declared(config)
    cases = [(scenario, design) for scenario in scenarios for design in designs]
    summary = {
        (row["scenario"], row["design"], row["rule"]): row for row in results.get("summary", [])
    }
    paired = {
        (row["scenario"], row["design"], row["comparison_rule"]): row
        for row in results.get("paired_differences", [])
        if row.get("metric") == "observed_score" and row.get("baseline_rule", "fixed") == "fixed"
    }
    rule_names = [_rule_label(rule, config) for rule in _RULES]
    lines = [
        "## Primary future observed scores",
        "",
        "Lower scores are better for the same future observation design. Absolute scores from different designs are not a common-information ranking. Each entry is a replication-cluster mean ± MCSE, averaged over the declared test origins and weekly horizons within each replication. Reduced quadratic scores can be negative.",
        "",
        "![Paired future observed-score differences](forecast_scores.png)",
        "",
        "| Scenario | Design | " + " | ".join(rule_names) + " |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for scenario, design in cases:
        values = [
            _estimate(summary.get((scenario, design, rule)), "observed_score") for rule in _RULES
        ]
        lines.append(f"| {_label(scenario)} | {_label(design)} | " + " | ".join(values) + " |")
    lines += [
        "",
        "### Paired differences against fixed local",
        "",
        "A positive difference means the comparison rule has a larger score than fixed local on the same future data. MCSE uses within-replication paired differences, not a combination of independent standard errors. No superiority of adaptation is an acceptance requirement.",
        "",
        "| Scenario | Design | Static minus fixed ± MCSE | Selected minus fixed ± MCSE |",
        "| --- | --- | ---: | ---: |",
    ]
    for scenario, design in cases:
        values = [
            _difference(paired.get((scenario, design, rule))) for rule in ("static", "selected")
        ]
        lines.append(f"| {_label(scenario)} | {_label(design)} | " + " | ".join(values) + " |")
    windows = sorted(_value(config, "windows", (16, 32, 64, 128)))
    lines += [
        "",
        "## Window selection before the test period",
        "",
        "The selected rule chooses one trailing-window length from past validation scores for each scenario/design/replication. It then refits that fixed choice at each test origin. Counts below are independent replication choices, not counts of dates or horizons. Latent covariance, risk and kernel errors do not select a window.",
        "",
        "| Scenario | Design | "
        + " | ".join(f"Window {window}" for window in windows)
        + " | Choices |",
        "| --- | --- | " + " | ".join("---:" for _ in windows) + " | ---: |",
    ]
    for scenario, design in cases:
        counts = Counter(
            row["selected_window"]
            for row in results.get("selections", [])
            if row["scenario"] == scenario and row["design"] == design
        )
        lines.append(
            f"| {_label(scenario)} | {_label(design)} | "
            + " | ".join(str(counts[window]) for window in windows)
            + f" | {sum(counts.values())} |"
        )
    origin_rows = {
        (row["scenario"], row["design"], row["origin"], row["rule"]): row
        for row in results.get("origin_summary", [])
    }
    lines += [
        "",
        "## Horizon and origin sensitivity",
        "",
        "![Observed scores by weekly forecast horizon](horizon_sensitivity.png)",
        "",
        "Every configured scenario/design panel is retained. These are raw reduced scores, not normalized by the number of future measurements. Future observation dimensions and maps can vary with horizon, especially for occasional joint measurements. Compare rules on the same future maps; score magnitudes across different horizons/designs are not a common-information ranking. The same issued covariance is used at all horizons in a block. Horizon-specific means first average the test origins within a replication; their MCSE is across independent replications. These are separate weekly covariance forecasts, not a covariance for an accumulated multiweek return. Bands show marginal MCSE, not simultaneous confidence bands or paired-difference uncertainty.",
        "",
        "<details>",
        "<summary>Scores at every predeclared test origin</summary>",
        "",
        "Within each origin, horizons are averaged inside each independent replication before calculating MCSE.",
        "",
        "| Scenario | Design | Origin | " + " | ".join(rule_names) + " |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario, design in cases:
        for origin in _value(config, "test_origins", (256, 272, 304, 448)):
            values = [
                _estimate(origin_rows.get((scenario, design, origin, rule)), "observed_score")
                for rule in _RULES
            ]
            lines.append(
                f"| {_label(scenario)} | {_label(design)} | {origin} | " + " | ".join(values) + " |"
            )
    lines += [
        "",
        "</details>",
        "",
        "## Operational future exposure-risk scores",
        "",
        "![Paired future exposure-risk scores](risk_evaluation.png)",
        "",
        "A future risk proxy is used only when that future observation map identifies the declared exposure variance. Full-rank past training geometry does not make a missing future cross-pair observation available. In particular, joint-every-64 training windows may have full rank while most of their future cross-spread proxies remain unavailable. Unavailable rows have no score; they are not zero errors.",
        "",
    ]
    risks = {
        (row["scenario"], row["design"], row["target"], row["rule"]): row
        for row in results.get("risk_summary", [])
    }
    total = (
        _value(config, "replications", 0)
        * len(_value(config, "test_origins", ()))
        * _value(config, "horizon", 16)
    )
    lines += [
        "Availability is the number of usable future date/horizon rows per rule, with independent replication clusters shown separately. Eligible dates per replication are the total available rows divided by the planned independent replication count; the future design is deterministic. The table uses the common future-design availability from fixed local; the same future maps score all rules. Each risk score averages available rows within each replication before MCSE. Availability varies by target and design, so comparison across different subsets requires care.",
        "",
        "| Scenario | Design | Target | Eligible dates/rep | Available rows / planned | Replication clusters | "
        + " | ".join(rule_names)
        + " |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, design in cases:
        for target in _TARGETS:
            fixed = risks.get((scenario, design, target, "fixed"), {})
            values = [
                _estimate(risks.get((scenario, design, target, rule)), "risk_score")
                for rule in _RULES
            ]
            lines.append(
                f"| {_label(scenario)} | {_label(design)} | {_label(target)} | "
                f"{_number(fixed.get('available_scores', 0) / max(_value(config, 'replications', 0), 1))} | "
                f"{fixed.get('available_scores', 0)} / {total} | {fixed.get('available_replications', 0)} | "
                + " | ".join(values)
                + " |"
            )
    return lines


def _diagnostics(config: Any, results: dict) -> list[str]:
    scenarios, designs = _declared(config)
    lines = [
        "## Simulation-only latent diagnostics",
        "",
        "The following diagnostics use latent future covariance or exposures and are unavailable in an observed-data deployment. They evaluate the issued frozen forecasts against each future interval, are kept separate from the primary observed scores, and never enter selection or acceptance as a statistical performance threshold. HS errors use the declared orthonormal H coordinates; a regularized full-operator estimate is still reported when the design leaves directions unidentified.",
        "",
        "| Scenario | Design | Rule | Relative operator HS error ± MCSE | Observed covariance error ± MCSE | Off-grid kernel RMSE ± MCSE | Exposure-risk RMSE ± MCSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    truth = {
        (row["scenario"], row["design"], row["rule"]): row
        for row in results.get("truth_summary", [])
    }
    for scenario in scenarios:
        for design in designs:
            for rule in _RULES:
                row = truth.get((scenario, design, rule))
                values = [
                    _estimate(row, metric)
                    for metric in (
                        "operator_hs_relative_error",
                        "observed_covariance_error",
                        "kernel_grid_rmse",
                        "exposure_risk_rmse",
                    )
                ]
                lines.append(
                    f"| {_label(scenario)} | {_label(design)} | {_rule_label(rule, config)} | "
                    + " | ".join(values)
                    + " |"
                )
    origins = sorted(
        set(
            (
                *_value(config, "validation_origins", (128, 192)),
                *_value(config, "test_origins", (256, 272, 304, 448)),
                _value(config, "static_window", 128),
            )
        )
    )
    windows = sorted(
        set(
            (
                *_value(config, "windows", (16, 32, 64, 128)),
                _value(config, "static_window", 128),
                _value(config, "fixed_window", 128),
            )
        )
    )
    geometry = {
        (row["design"], row["origin"], row["window"]): row for row in results.get("geometry", [])
    }
    lines += [
        "",
        "## Training geometry and numerical acceptance",
        "",
        "Training Gram ranks below describe recorded fits on the declared origin/window grid and are deterministic properties of the observation design, shared across covariance paths and replications. An unavailable cell means no fit was recorded for that cell; it is not a rank failure or a claim that its observations are unavailable. The denominator is the dimension of the symmetric parameter space. A trailing fit uses intervals origin−window+1 through origin; the static fit uses its first declared block once. Geometry spectra, thresholds and target-range diagnostics are retained in `geometry.json`, with exact training bounds and geometry references in `forecasts.json`.",
        "",
        "| Design | Window | " + " | ".join(f"Origin {origin}" for origin in origins) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in origins) + " |",
    ]
    for design in designs:
        for window in windows:
            cells = []
            for origin in origins:
                row = geometry.get((design, origin, window))
                cells.append(f"{row['rank']}/{len(row['eigenvalues'])}" if row else "unavailable")
            lines.append(f"| {_label(design)} | {window} | " + " | ".join(cells) + " |")
    acceptance = results.get("acceptance", {})
    lines += [
        "",
        f"Numerical acceptance: **{'PASS' if acceptance.get('passed') else 'FAIL'}**.",
        "",
        "Acceptance checks feasibility, optimizer error certificates and the declared independent convex references. A covariance-error certificate bounds numerical distance to the constrained optimizer, not sampling, approximation or forecasting error. MCSE is not a numerical certificate or a confidence-interval coverage claim. There is no statistical superiority gate.",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name in (
        "failed_fits",
        "feasibility_passed",
        "reference_passed",
        "reference_checks_requested",
        "reference_cases",
        "reference_cases_expected",
        "maximum_covariance_error_bound",
        "numerical_error_budget",
        "statistical_performance_gate",
    ):
        value = acceptance.get(name)
        display = (
            "none (no superiority requirement)"
            if name == "statistical_performance_gate" and value is None
            else _number(value)
        )
        lines.append(f"| {name.replace('_', ' ')} | {display} |")
    return lines


def write_report(output: Path, config: Any, results: dict) -> None:
    """Write an observed-score report and three fixed-selection scientific figures."""
    _validate(config, results)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (
        plt.style.context("default"),
        plt.rc_context(
            {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8}
        ),
    ):
        for renderer in (_forecast_figure, _horizon_figure, _risk_figure):
            renderer(config, results, output)
    validation = tuple(_value(config, "validation_origins", (128, 192)))
    tests = tuple(_value(config, "test_origins", (256, 272, 304, 448)))
    horizon = _value(config, "horizon", 16)
    windows = tuple(_value(config, "windows", (16, 32, 64, 128)))
    static = _value(config, "static_window", 128)
    fixed = _value(config, "fixed_window", 128)
    lines = [
        "# Research question 3: forecasting innovation covariance",
        "",
        f"Configuration `{_value(config, 'name', 'rq3')}`: {_value(config, 'replications')} independent paired replications, {_value(config, 'intervals')} weekly innovations, Δ={_value(config, 'delta', 1 / 52):.8g} years, seed `{_value(config, 'seed')}`.",
        "",
        "This study asks whether refitting a recent innovation covariance improves future observed quadratic scores, and when past-only window selection helps or hurts. The target is the covariance of each subsequent weekly innovation. It is not instantaneous diffusion covariance C, an accumulated multiweek-return covariance, or a prediction of the curve level. The experiment reuses the configured finite-dimensional RQ2 covariance paths and nested observation designs, directly sampling independent Gaussian innovations with known measurement-noise covariance. It does not establish market forecasting performance.",
        "",
        "## Issued forecasts and chronological separation",
        "",
        "| Rule | Training and update policy |",
        "| --- | --- |",
        f"| Static ({static}) | Fit intervals 1 through {static} once and reuse that same matrix at every test origin. |",
        f"| Fixed local ({fixed}) | Refit the trailing {fixed} past intervals at each test origin. |",
        f"| Selected window | Select once from {', '.join(map(str, windows))} using only the declared validation blocks; refit that chosen trailing window at each test origin. |",
        "",
        f"Validation origins are {', '.join(map(str, validation))}; their outcomes end at interval {max(validation) + horizon}. Selection is frozen before the first test origin {min(tests)}. The selected window is not retuned after observing test outcomes. At each test origin the fitted covariance is frozen for the next {horizon} separate weekly innovations. All rules face the same future observations; the noiseless identity design is a separate information control.",
        "",
        "| Test origin | Future intervals scored |",
        "| ---: | --- |",
    ]
    lines.extend(f"| {origin} | {origin + 1} through {origin + horizon} |" for origin in tests)
    lines += [
        "",
        "## Observed scoring rule",
        "",
        "For a future observation y with map O and known noise covariance N, the noise-corrected measurement is Z. The issued forecast B is scored without latent covariance:",
        "",
        "```math",
        r"Z=yy^\top-N,\qquad A(B)=OBO^\top,",
        r"s(B)=\|A(B)\|_F^2-2\langle Z,A(B)\rangle_F.",
        "```",
        "",
        "The forecast-independent squared norm of Z is omitted. Scores may therefore be negative. Expected score differences compare squared error in the future observed covariance space under the stated noise model. They do not identify every full-operator direction or provide a Gaussian likelihood score.",
        "",
        "For an identifiable future exposure, the operational risk score is forecast_variance² − 2 × forecast_variance × noise_corrected_future_risk_proxy. Its omitted proxy-squared constant is also common across forecast rules. Availability is checked at the future observation map, separately from training rank.",
        "",
        "All Monte Carlo standard errors (MCSE) treat one independent replication as a cluster. Origins and horizons are averaged within replication, including paired differences; they are not counted as independent replications. MCSE is unavailable when fewer than two independent clusters are present. Figures and tables use predeclared scenarios, designs, origins and horizons, with no selection of favorable outcomes.",
        "",
    ]
    lines += _tables(config, results)
    lines += _diagnostics(config, results)
    lines += [
        "",
        "## Artifacts and provenance",
        "",
        "[results.json](results.json) contains the clustered summaries, paired comparisons, selection audit, geometry, model settings and acceptance. [scores.csv](scores.csv) retains future observed scores; [risks.csv](risks.csv) retains operational risk scores and availability; [truth_metrics.csv](truth_metrics.csv) is simulation-only, including nullspace and weak-direction HS errors. [forecasts.json](forecasts.json) stores every issued covariance, its training bounds, basis, issue time and geometry reference. [selection_audit.json](selection_audit.json) retains candidate validation scores. [geometry.json](geometry.json) retains training spectra and target diagnostics; [fit_diagnostics.json](fit_diagnostics.json) retains per-fit certificates; [observation_schedule.json](observation_schedule.json) records the future maps and noise covariances.",
        "",
        "[manifest.json](manifest.json) records the resolved configuration, seed, source hashes/revision, environment, solver settings and dirty-source flag. Interpret provenance using that manifest rather than a release label alone. Statistical uncertainty, numerical certificates and simulation-only truth error answer different questions.",
        "",
        "The predeclared [RQ3 protocol](../../../docs/research-question-3.md) and [method definitions](../../../docs/methods.md) are repository links that also work when this report is copied into `experiments/results/rq3/`. The study assumes correctly specified centered innovations, known fresh observation-noise covariance and deterministic observation designs. It does not establish robustness to curve-level noise, general misspecification, confidence-interval coverage or market risk calibration.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
