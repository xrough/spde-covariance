"""One infinite-source model, separate forcing/projection/time refinements.

Finite forcing levels are coupled by independent forcing bands on the finest
time grid. Coarse innovations are exact compositions of the same fine steps.
Projecting a full-forcing trajectory is never replaced by an autonomous
J-dimensional process. Operational residuals contain a predictable hidden-state
term; they are not treated as independent common-covariance innovations.
"""

from __future__ import annotations

import csv
import json
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.integrate import quad

from .estimation import fit_covariance, fit_covariance_reference
from .infinite_config import InfiniteTransportConfig
from .laguerre import LaguerreTransport, PowerLawSource
from .linalg import validate_psd
from .observations import Observation, delivery_average_map
from .provenance import _jsonable, _manifest, _source_root

SOLVER_OPTIONS = {"absolute_tolerance": 1e-9, "relative_tolerance": 0.0, "max_iterations": 30_000}
STATISTICS = (
    "oracle_covariance_mse",
    "operational_covariance_mse",
    "operational_centered_covariance_mse",
    "state_terminal_coupling_error_squared",
    "projected_state_terminal_error_squared",
    "projection_terminal_error_squared",
    "leakage_sample_mse",
)


def _readonly(values):
    result = np.asarray(values, dtype=float).copy()
    result.flags.writeable = False
    return result


def _sym(matrix):
    return (matrix + matrix.T) / 2


def state_covariance_sum(shift, step_covariance, count):
    """Return K_count and sum(K_i,i=0..count-1) in O(log count) compositions.

    K_i is the zero-initial-state covariance of the exact discrete recursion.
    The sum includes the preceding state of every observed increment.
    """
    shift = np.asarray(shift, dtype=float)
    covariance = validate_psd(step_covariance, name="step covariance")
    if shift.shape != covariance.shape or not np.all(np.isfinite(shift)):
        raise ValueError("shift must be finite and match the step covariance")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    accumulated_shift = np.eye(len(shift))
    accumulated_covariance = np.zeros_like(covariance)
    accumulated_sum = np.zeros_like(covariance)
    block_shift, block_covariance = shift.copy(), covariance.copy()
    block_sum = np.zeros_like(covariance)
    block_length, remaining = 1, int(count)
    while remaining:
        if remaining & 1:
            accumulated_sum = _sym(
                accumulated_sum
                + block_length * accumulated_covariance
                + accumulated_shift @ block_sum @ accumulated_shift.T
            )
            accumulated_covariance = _sym(
                accumulated_covariance + accumulated_shift @ block_covariance @ accumulated_shift.T
            )
            accumulated_shift = accumulated_shift @ block_shift
        remaining >>= 1
        if remaining:
            block_sum = _sym(
                block_sum
                + block_length * block_covariance
                + block_shift @ block_sum @ block_shift.T
            )
            block_covariance = _sym(
                block_covariance + block_shift @ block_covariance @ block_shift.T
            )
            block_shift = block_shift @ block_shift
            block_length *= 2
    return accumulated_covariance, accumulated_sum


def fixed_exposures(model):
    """Fixed infinite-H unit exposures, without normalizing finite prefixes."""
    point_norm_squared = np.exp(-0.5) - np.exp(-1.5)
    point = (model.evaluate([0.5])[0] - model.evaluate([1.5])[0]) / np.sqrt(point_norm_squared)
    rows = delivery_average_map(model, [(0.5, 1.0), (1.5, 2.0)], quadrature_order=32)
    # Difference of delivery averages has zero total mass. Its derivative
    # representer is a ramp, a plateau, then a ramp; the H dual norm is exact
    # up to one-dimensional numerical quadrature of this elementary function.
    norm_squared = (
        quad(lambda x: np.exp(-x) * (2 * (x - 0.5)) ** 2, 0.5, 1.0, epsabs=1e-14)[0]
        + np.exp(-1.0)
        - np.exp(-1.5)
        + quad(lambda x: np.exp(-x) * (2 * (2.0 - x)) ** 2, 1.5, 2.0, epsabs=1e-14)[0]
    )
    return {
        "point_spread": _readonly(point),
        "delivery_spread": _readonly((rows[0] - rows[1]) / np.sqrt(norm_squared)),
    }


def _check(name, error, tolerance, **metadata):
    return {
        "name": name,
        **metadata,
        "error": float(error),
        "tolerance": float(tolerance),
        "passed": bool(np.isfinite(error) and error <= tolerance),
    }


def _negative_part_norm(matrix):
    eigenvalues = np.linalg.eigvalsh(_sym(matrix))
    return float(max(0.0, -eigenvalues[0]))


def _prepare_population(config):
    source = PowerLawSource(config.source_exponent, config.source_trace)
    models = {m: LaguerreTransport(m) for m in config.forcing_dimensions}
    reference_m = max(models)
    reference = models[reference_m]
    exposures = fixed_exposures(reference)
    grid_points = np.linspace(0.0, 3.0, 61)
    grid = reference.evaluate(grid_points)
    source_covariances, state_covariances, innovations, shifts, past_means = {}, {}, {}, {}, {}
    previous_terminal, checks, population, cases = {}, [], [], {}
    for m, model in models.items():
        covariance = source.finite_matrix(m)
        source_covariances[m] = covariance
        state = model.cumulative_covariance(covariance, config.horizon, config.quadrature_order)
        state_covariances[m] = state
        block_state = model.cumulative_covariance(
            covariance, config.horizon, config.quadrature_order, method="block"
        )
        checks.append(
            _check(
                "state_covariance_integration",
                np.linalg.norm(state - block_state),
                config.numerical_tolerance,
                M=m,
            )
        )
        first, second = 0.37 * config.horizon, 0.63 * config.horizon
        composition = model.matrix(first + second) - model.matrix(first) @ model.matrix(second)
        checks.append(
            _check(
                "transport_composition",
                np.linalg.norm(composition),
                config.transport_tolerance,
                M=m,
            )
        )
        evaluation = model.evaluate(grid_points + first) - model.evaluate(
            grid_points
        ) @ model.matrix(first)
        checks.append(
            _check(
                "transport_evaluation", np.max(np.abs(evaluation)), config.transport_tolerance, M=m
            )
        )
        checks.append(
            _check(
                "source_trace_and_tail",
                abs(np.trace(covariance) + source.tail_trace(m) - source.trace),
                config.numerical_tolerance,
                M=m,
            )
        )
        for delta in config.deltas:
            key = (m, delta)
            diagnostic = model.covariance_diagnostics(covariance, delta, config.quadrature_order)
            innovation = diagnostic["quadrature_innovation_covariance"]
            innovations[key] = innovation
            shifts[key] = model.matrix(delta)
            checks.append(
                _check(
                    "innovation_covariance_integration",
                    diagnostic["absolute_hs_discrepancy"],
                    config.numerical_tolerance,
                    M=m,
                    delta=delta,
                )
            )
            count = round(config.horizon / delta)
            terminal, state_sum = state_covariance_sum(shifts[key], delta * innovation, count)
            past_means[key] = state_sum / count
            previous_terminal[key] = model.cumulative_covariance(
                covariance,
                config.horizon - delta,
                config.quadrature_order,
            )
            checks.append(
                _check(
                    "composed_state_covariance",
                    np.linalg.norm(terminal - state),
                    config.numerical_tolerance,
                    M=m,
                    delta=delta,
                )
            )
    for m, model in models.items():
        for delta in config.deltas:
            innovation, shift = innovations[(m, delta)], shifts[(m, delta)]
            state, previous = state_covariances[m], previous_terminal[(m, delta)]
            reference_v = innovations[(reference_m, delta)]
            for j in config.projection_dimensions:
                if j > m:
                    continue
                low_shift = LaguerreTransport(j, cache_max_bytes=0).matrix(delta)
                leakage_map = shift[:j].copy()
                leakage_map[:, :j] -= low_shift
                leakage_mean = _sym(leakage_map @ past_means[(m, delta)] @ leakage_map.T / delta)
                leakage_terminal = _sym(leakage_map @ previous @ leakage_map.T / delta)
                projected = innovation[:j, :j]
                # A separate level-covariance calculation audits the hidden-term formula.
                cross = shift[:j] @ previous[:, :j]
                direct_operational = _sym(
                    (
                        state[:j, :j]
                        + low_shift @ previous[:j, :j] @ low_shift.T
                        - cross @ low_shift.T
                        - low_shift @ cross.T
                    )
                    / delta
                )
                checks.append(
                    _check(
                        "operational_leakage_formula",
                        np.linalg.norm(direct_operational - projected - leakage_terminal),
                        config.numerical_tolerance,
                        M=m,
                        J=j,
                        delta=delta,
                    )
                )
                projected_bound = source.projected_tail_bound(j, m, delta)
                operational_bound = source.operational_tail_bound(
                    j, m, delta, config.horizon - delta
                )
                ref_low_shift = low_shift
                ref_leakage_map = shifts[(reference_m, delta)][:j].copy()
                ref_leakage_map[:, :j] -= ref_low_shift
                ref_terminal_leakage = _sym(
                    ref_leakage_map
                    @ previous_terminal[(reference_m, delta)]
                    @ ref_leakage_map.T
                    / delta
                )
                checks.append(
                    _check(
                        "projected_tail_domination",
                        _negative_part_norm(
                            projected_bound["matrix"] - (reference_v[:j, :j] - projected)
                        ),
                        config.numerical_tolerance,
                        M=m,
                        J=j,
                        delta=delta,
                    )
                )
                checks.append(
                    _check(
                        "operational_tail_domination",
                        _negative_part_norm(
                            operational_bound["matrix"]
                            - (
                                reference_v[:j, :j]
                                + ref_terminal_leakage
                                - projected
                                - leakage_terminal
                            )
                        ),
                        config.numerical_tolerance,
                        M=m,
                        J=j,
                        delta=delta,
                    )
                )
                row = {
                    "M": m,
                    "J": j,
                    "delta": delta,
                    "intervals": round(config.horizon / delta),
                    "reference_M": reference_m,
                    "source_trace": float(np.trace(source_covariances[m])),
                    "innovation_trace": float(np.trace(innovation)),
                    "state_trace": float(np.trace(state)),
                    "full_source_tail_bound": source.tail_trace(m),
                    "full_innovation_tail_bound": source.innovation_tail_bound(m, delta),
                    "full_state_mse_tail_bound": source.state_mse_tail_bound(m, config.horizon),
                    "projected_innovation_tail_bound": projected_bound["hs_bound"],
                    "operational_covariance_tail_bound_max": operational_bound["hs_bound"],
                    "forcing_innovation_hs_error": float(
                        np.linalg.norm(reference_v[:j, :j] - projected)
                    ),
                    "projection_innovation_hs_error": float(
                        np.linalg.norm(innovation - np.pad(projected, ((0, m - j), (0, m - j))))
                    ),
                    "forcing_state_hs_error": float(
                        np.linalg.norm(state_covariances[reference_m][:j, :j] - state[:j, :j])
                    ),
                    "source_to_innovation_hs_error": float(
                        np.linalg.norm(innovation - source_covariances[m])
                    ),
                    "leakage_mean_hs": float(np.linalg.norm(leakage_mean)),
                    "leakage_terminal_hs": float(np.linalg.norm(leakage_terminal)),
                    "leakage_mean_trace": float(np.trace(leakage_mean)),
                    "oracle_gaussian_mse": float(
                        (np.sum(projected**2) + np.trace(projected) ** 2)
                        / round(config.horizon / delta)
                    ),
                    "tail_resolved": bool(
                        max(projected_bound["hs_bound"], operational_bound["hs_bound"])
                        <= config.tail_budget
                    ),
                }
                population.append(row)
                risks = {}
                for name, full_exposure in exposures.items():
                    a, a_m = full_exposure[:j], full_exposure[:m]
                    projected_risk = float(a @ projected @ a)
                    full_risk = float(a_m @ innovation @ a_m)
                    reference_risk = float(full_exposure @ reference_v @ full_exposure)
                    risks[name] = {
                        "projected_risk_truth": projected_risk,
                        "forcing_risk_truth": full_risk,
                        "reference_risk_truth": reference_risk,
                        "projection_bias": projected_risk - full_risk,
                        "forcing_bias": full_risk - reference_risk,
                        "leakage_bias": float(a @ leakage_mean @ a),
                        "risk_reference_tail_bound": source.innovation_tail_bound(
                            reference_m, delta
                        ),
                        "oracle_gaussian_risk_mse": (
                            2 * projected_risk**2 / row["intervals"]
                            + (projected_risk - reference_risk) ** 2
                        ),
                    }
                cases[(m, j, delta)] = {
                    "population": row,
                    "innovation": projected,
                    "operational_mean": projected + leakage_mean,
                    "leakage_map": leakage_map,
                    "leakage_mean": leakage_mean,
                    "leakage_terminal": leakage_terminal,
                    "risks": risks,
                }
    selected_delta = max(config.deltas)
    kernels = {
        "maturities": grid_points,
        "reference_M": reference_m,
        "delta": selected_delta,
        "horizon": config.horizon,
        "source": grid @ source_covariances[reference_m] @ grid.T,
        "innovation": grid @ innovations[(reference_m, selected_delta)] @ grid.T,
        "state": grid @ state_covariances[reference_m] @ grid.T,
    }
    return {
        "source": source,
        "models": models,
        "reference_M": reference_m,
        "source_covariances": source_covariances,
        "state_covariances": state_covariances,
        "innovations": innovations,
        "shifts": shifts,
        "cases": cases,
        "population": population,
        "numerical_checks": checks,
        "kernels": kernels,
        "exposures": exposures,
    }


def population_audit(config: InfiniteTransportConfig):
    """Population-only checkpoint; no Monte Carlo samples or covariance fitting."""
    prepared = _prepare_population(config)
    return {name: prepared[name] for name in ("population", "numerical_checks", "kernels")}


def _covariance_root(covariance, tolerance):
    values, vectors = np.linalg.eigh(_sym(covariance))
    scale = max(float(np.linalg.norm(covariance)), np.finfo(float).tiny)
    if values[0] < -tolerance * scale:
        raise ValueError("forcing-band covariance is not numerically positive semidefinite")
    root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    return root, {
        "minimum_eigenvalue": float(values[0]),
        "clipped_negative_trace": float(-np.sum(values[values < 0])),
        "factorization_error": float(np.linalg.norm(root @ root.T - covariance)),
        "covariance_scale": scale,
    }


def _prepare_bands(config, prepared):
    lower, bands = 0, []
    for upper in config.forcing_dimensions:
        covariance = prepared["source"].finite_matrix(upper)
        covariance[:lower, :lower] = 0.0
        integrated = prepared["models"][upper].cumulative_covariance(
            covariance,
            config.fine_delta,
            config.quadrature_order,
        )
        root, diagnostic = _covariance_root(integrated, config.numerical_tolerance)
        bands.append({"lower": lower, "upper": upper, "covariance": integrated, "root": root})
        prepared["numerical_checks"].append(
            _check(
                "forcing_band_factorization",
                diagnostic["factorization_error"],
                config.numerical_tolerance * diagnostic["covariance_scale"],
                lower=lower,
                upper=upper,
                **diagnostic,
            )
        )
        lower = upper
    return bands


@dataclass(frozen=True)
class CoupledSimulation:
    """Read-only paths and unnormalized fine increments, indexed by forcing M."""

    times: np.ndarray
    paths: dict
    fine_increments: dict
    band_increments: tuple

    def __post_init__(self):
        object.__setattr__(self, "times", _readonly(self.times))
        for name in ("paths", "fine_increments"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {key: _readonly(value) for key, value in getattr(self, name).items()}
                ),
            )
        object.__setattr__(
            self,
            "band_increments",
            tuple(
                MappingProxyType(
                    {
                        "lower": row["lower"],
                        "upper": row["upper"],
                        "values": _readonly(row["values"]),
                    }
                )
                for row in self.band_increments
            ),
        )


def simulate_coupled_paths(config, replication, *, prepared=None, bands=None):
    """Couple forcing prefixes through independent source bands, not shared seeds alone.

    A band's integrated increments live in its upper-dimensional invariant
    prefix. Reusing that exact increment in every larger model gives the joint
    covariance dictated by common Brownian forcing modes.
    """
    if (
        isinstance(replication, bool)
        or not isinstance(replication, (int, np.integer))
        or not 0 <= replication < config.replications
    ):
        raise ValueError("replication must be a configured zero-based integer")
    prepared = _prepare_population(config) if prepared is None else prepared
    bands = _prepare_bands(config, prepared) if bands is None else bands
    draws = []
    for band in bands:
        sequence = np.random.SeedSequence(
            [config.seed, int(replication), band["lower"], band["upper"]]
        )
        rng = np.random.default_rng(sequence)
        values = rng.standard_normal((config.fine_intervals, band["upper"])) @ band["root"].T
        draws.append({"lower": band["lower"], "upper": band["upper"], "values": values})
    paths, increments = {}, {}
    for m, model in prepared["models"].items():
        innovations = np.zeros((config.fine_intervals, m))
        for band in draws:
            if band["upper"] <= m:
                innovations[:, : band["upper"]] += band["values"]
        shift = model.matrix(config.fine_delta)
        values = np.zeros((config.fine_intervals + 1, m))
        for index, innovation in enumerate(innovations):
            values[index + 1] = shift @ values[index] + innovation
        paths[m], increments[m] = values, innovations
    return CoupledSimulation(
        np.arange(config.fine_intervals + 1) * config.fine_delta, paths, increments, tuple(draws)
    )


def coarsen_time_grid(path, fine_increments, model, *, fine_delta, delta):
    """Extract exact coarse innovations from the same fine path and audit composition."""
    values, fine = np.asarray(path, dtype=float), np.asarray(fine_increments, dtype=float)
    if (
        values.ndim != 2
        or fine.shape != (len(values) - 1, model.dimension)
        or values.shape[1] != model.dimension
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(fine))
    ):
        raise ValueError("path and fine increments must be matching finite coefficient arrays")
    if not np.isfinite(fine_delta) or not np.isfinite(delta) or fine_delta <= 0 or delta <= 0:
        raise ValueError("time intervals must be finite and positive")
    stride = round(delta / fine_delta)
    if (
        stride < 1
        or not np.isclose(delta / fine_delta, stride, rtol=1e-12, atol=0)
        or len(fine) % stride
    ):
        raise ValueError("delta must coarsen the complete fine grid by an integer factor")
    states = values[::stride]
    extracted = (states[1:] - states[:-1] @ model.matrix(delta).T) / np.sqrt(delta)
    composed = np.zeros_like(extracted)
    shift = model.matrix(fine_delta)
    for offset in range(stride):
        composed = composed @ shift.T + fine[offset::stride]
    composed /= np.sqrt(delta)
    return {
        "states": _readonly(states),
        "innovations": _readonly(extracted),
        "composition_error": float(np.max(np.abs(extracted - composed))),
        "intervals": len(extracted),
        "stride": stride,
    }


def _mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    )


def _groups(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _summaries(rows, risks, fits, config):
    summary = []
    for key, records in _groups(rows, ("M", "J", "delta")).items():
        item = dict(zip(("M", "J", "delta"), key, strict=True))
        item.update(
            replications=len(records),
            oracle_gaussian_mse=records[0]["oracle_gaussian_mse"],
            tail_resolved=records[0]["tail_resolved"],
        )
        for metric in STATISTICS:
            item[metric + "_mean"], item[metric + "_mcse"] = _mean_mcse(
                [row[metric] for row in records]
            )
        (
            item["operational_minus_oracle_covariance_mse_mean"],
            item["operational_minus_oracle_covariance_mse_mcse"],
        ) = _mean_mcse(
            [row["operational_covariance_mse"] - row["oracle_covariance_mse"] for row in records]
        )
        mcse = item["oracle_covariance_mse_mcse"]
        item["oracle_gaussian_mse_within_4mcse"] = bool(
            mcse is not None
            and abs(item["oracle_covariance_mse_mean"] - item["oracle_gaussian_mse"]) <= 4 * mcse
        )
        summary.append(item)
    risk_summary = []
    for key, records in _groups(risks, ("M", "J", "delta", "exposure")).items():
        item = dict(zip(("M", "J", "delta", "exposure"), key, strict=True))
        item.update(replications=len(records))
        for name in (
            "projected_risk_truth",
            "forcing_risk_truth",
            "reference_risk_truth",
            "projection_bias",
            "forcing_bias",
            "leakage_bias",
            "risk_reference_tail_bound",
            "oracle_gaussian_risk_mse",
        ):
            item[name] = records[0][name]
        for metric in (
            "oracle_risk_mse",
            "operational_risk_mse",
            "oracle_risk_error",
            "operational_risk_error",
        ):
            item[metric + "_mean"], item[metric + "_mcse"] = _mean_mcse(
                [row[metric] for row in records]
            )
        (
            item["operational_minus_oracle_risk_mse_mean"],
            item["operational_minus_oracle_risk_mse_mcse"],
        ) = _mean_mcse([row["operational_risk_mse"] - row["oracle_risk_mse"] for row in records])
        item["reference_tail_resolved"] = item["risk_reference_tail_bound"] <= config.tail_budget
        risk_summary.append(item)
    fit_summary = []
    for key, records in _groups(fits, ("J", "delta", "input_kind")).items():
        item = dict(zip(("J", "delta", "input_kind"), key, strict=True))
        item["replications"] = len(records)
        for metric in ("operator_hs_error", "centered_hs_error", "covariance_error_bound"):
            item[metric + "_mean"], item[metric + "_mcse"] = _mean_mcse(
                [row[metric] for row in records]
            )
        fit_summary.append(item)
    paired = []
    reference_m = max(config.forcing_dimensions)
    for key, records in _groups(rows, ("J", "delta")).items():
        baseline = {row["replication"]: row for row in records if row["M"] == reference_m}
        for m, comparisons in _groups(records, ("M",)).items():
            if m[0] == reference_m:
                continue
            for metric in ("oracle_covariance_mse", "operational_covariance_mse"):
                values = [row[metric] - baseline[row["replication"]][metric] for row in comparisons]
                mean, mcse = _mean_mcse(values)
                paired.append(
                    {
                        "M": m[0],
                        "reference_M": reference_m,
                        "J": key[0],
                        "delta": key[1],
                        "metric": metric,
                        "mean_difference": mean,
                        "mcse": mcse,
                        "replications": len(values),
                    }
                )
    return {
        "summary": summary,
        "risk_summary": risk_summary,
        "fit_summary": fit_summary,
        "paired_differences": paired,
    }


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_infinite_transport_benchmark(config, output=None):
    """Run all numerical gates and paired simulations into a fresh directory."""
    if isinstance(config, (str, Path)):
        with Path(config).open("rb") as stream:
            config = InfiniteTransportConfig(**tomllib.load(stream))
    if not isinstance(config, InfiniteTransportConfig):
        raise TypeError("config must be InfiniteTransportConfig or a TOML path")
    root = _source_root()
    output = (
        Path(output)
        if output is not None
        else (root or Path.cwd()) / "experiments" / "runs" / config.name
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _run(config, output, root)
    except (Exception, KeyboardInterrupt) as error:
        _write_json(
            output / "status.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _run(config, output, root):
    manifest = _manifest(config, root)
    manifest.update(
        experiment="coupled finite-forcing approximations of one infinite-rank Laguerre transport SPDE",
        space={
            "metric": "f(0)g(0)+integral_0^infinity exp(x) f'(x)g'(x) dx",
            "maturity_unit": "year",
            "maturity_cutoff": None,
            "basis": "e0=1; ej=integral_0^x exp(-u)L_(j-1)(u)du",
        },
        solver={
            "population": "positive quadrature and independent scaled block exponential",
            "finite_J_covariance": "accelerated projected gradient; J<=16 only",
            **SOLVER_OPTIONS,
            "numerical_error_budget": config.numerical_error_budget,
        },
        coupling="independent forcing-band Gaussian fine increments; coarse grids composed from same paths",
        operational_residuals="dependent and time varying; low-level covariance contrast only",
    )
    _write_json(output / "config.json", asdict(config))
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "status.json", {"status": "running"})
    prepared = _prepare_population(config)
    bands = _prepare_bands(config, prepared)
    _write_json(output / "population.json", prepared["population"])
    _write_json(output / "numerical_checks.json", prepared["numerical_checks"])
    if not all(row["passed"] for row in prepared["numerical_checks"]):
        raise RuntimeError("population or forcing-band numerical acceptance failed")
    rows, risk_rows, fit_rows, references = [], [], [], []
    composition_errors, leakage_errors = [], []
    reference_m = prepared["reference_M"]
    reference_j = min(config.fit_dimensions) if config.fit_dimensions else None
    for replication in range(config.replications):
        simulation = simulate_coupled_paths(config, replication, prepared=prepared, bands=bands)
        reference_terminal = simulation.paths[reference_m][-1]
        for m, model in prepared["models"].items():
            terminal = simulation.paths[m][-1]
            state_error = reference_terminal - np.pad(terminal, (0, reference_m - m))
            for delta in config.deltas:
                coarse = coarsen_time_grid(
                    simulation.paths[m],
                    simulation.fine_increments[m],
                    model,
                    fine_delta=config.fine_delta,
                    delta=delta,
                )
                composition_errors.append(coarse["composition_error"])
                count = coarse["intervals"]
                for j in config.projection_dimensions:
                    if j > m:
                        continue
                    case = prepared["cases"][(m, j, delta)]
                    true = coarse["innovations"][:, :j]
                    low_shift = LaguerreTransport(j, cache_max_bytes=0).matrix(delta)
                    states = coarse["states"]
                    operational = (states[1:, :j] - states[:-1, :j] @ low_shift.T) / np.sqrt(delta)
                    leakage = states[:-1] @ case["leakage_map"].T / np.sqrt(delta)
                    error = float(np.max(np.abs(operational - true - leakage)))
                    leakage_errors.append(error)
                    oracle_covariance, operational_covariance = (
                        true.T @ true / count,
                        operational.T @ operational / count,
                    )
                    common = {"replication": replication, "M": m, "J": j, "delta": delta}
                    rows.append(
                        {
                            **common,
                            "oracle_covariance_mse": float(
                                np.sum((oracle_covariance - case["innovation"]) ** 2)
                            ),
                            "operational_covariance_mse": float(
                                np.sum((operational_covariance - case["innovation"]) ** 2)
                            ),
                            "operational_centered_covariance_mse": float(
                                np.sum((operational_covariance - case["operational_mean"]) ** 2)
                            ),
                            "state_terminal_coupling_error_squared": float(np.sum(state_error**2)),
                            "projected_state_terminal_error_squared": float(
                                np.sum(state_error[:j] ** 2)
                            ),
                            "projection_terminal_error_squared": float(np.sum(terminal[j:] ** 2)),
                            "leakage_sample_mse": float(np.mean(np.sum(leakage**2, axis=1))),
                            "oracle_gaussian_mse": case["population"]["oracle_gaussian_mse"],
                            "tail_resolved": case["population"]["tail_resolved"],
                            "leakage_identity_error": error,
                        }
                    )
                    for name, full_exposure in prepared["exposures"].items():
                        a = full_exposure[:j]
                        truth = case["risks"][name]["reference_risk_truth"]
                        oracle_risk = float(a @ oracle_covariance @ a)
                        operational_risk = float(a @ operational_covariance @ a)
                        risk_rows.append(
                            {
                                **common,
                                "exposure": name,
                                **case["risks"][name],
                                "oracle_risk_estimate": oracle_risk,
                                "operational_risk_estimate": operational_risk,
                                "oracle_risk_error": oracle_risk - truth,
                                "operational_risk_error": operational_risk - truth,
                                "oracle_risk_mse": (oracle_risk - truth) ** 2,
                                "operational_risk_mse": (operational_risk - truth) ** 2,
                            }
                        )
                    if m == reference_m and j in config.fit_dimensions:
                        for kind, values in (("oracle", true), ("operational", operational)):
                            observations = tuple(
                                Observation(
                                    float((index + 1) * delta),
                                    value,
                                    np.eye(j),
                                    np.zeros((j, j)),
                                )
                                for index, value in enumerate(values)
                            )
                            estimate = fit_covariance(
                                observations,
                                regularization=config.relative_regularization,
                                trace_cap=config.trace_cap,
                                numerical_error_budget=config.numerical_error_budget,
                                **SOLVER_OPTIONS,
                            )
                            centered = (
                                case["innovation"] if kind == "oracle" else case["operational_mean"]
                            )
                            fit_rows.append(
                                {
                                    **common,
                                    "input_kind": kind,
                                    "covariance": estimate.covariance,
                                    "basis_id": LaguerreTransport(j).basis_id,
                                    "basis_family_id": model.family_id,
                                    "gram_operator_norm": 1.0,
                                    "relative_regularization": config.relative_regularization,
                                    "regularization": estimate.regularization,
                                    "target": (
                                        "projected innovation covariance"
                                        if kind == "oracle"
                                        else "pooled operational second moment; dependent contrast"
                                    ),
                                    "operator_hs_error": float(
                                        np.linalg.norm(estimate.covariance - case["innovation"])
                                    ),
                                    "centered_hs_error": float(
                                        np.linalg.norm(estimate.covariance - centered)
                                    ),
                                    "covariance_error_bound": estimate.covariance_error_bound,
                                    "objective_gap_bound": estimate.objective_gap_bound,
                                    "converged": estimate.converged,
                                    "status": estimate.status,
                                    "iterations": estimate.iterations,
                                    "trace": estimate.trace,
                                    "minimum_eigenvalue": estimate.min_eigenvalue,
                                }
                            )
                            if (
                                config.reference_checks
                                and replication == 0
                                and j == reference_j
                                and delta == max(config.deltas)
                            ):
                                reference = fit_covariance_reference(
                                    observations,
                                    regularization=config.relative_regularization,
                                    trace_cap=config.trace_cap,
                                    numerical_error_budget=config.numerical_error_budget,
                                    solver_tolerance=1e-10,
                                )
                                references.append(
                                    {
                                        **common,
                                        "input_kind": kind,
                                        "converged": reference.converged,
                                        "matrix_discrepancy": float(
                                            np.linalg.norm(
                                                reference.covariance - estimate.covariance
                                            )
                                        ),
                                        "covariance_error_bound": reference.covariance_error_bound,
                                        "reference_refinements": reference.reference_refinements,
                                        "reference_solver_tolerance": reference.reference_solver_tolerance,
                                    }
                                )
        print(
            f"Infinite transport replication {replication + 1}/{config.replications} complete",
            flush=True,
        )
    population = prepared["population"]
    reference_rows = [row for row in population if row["M"] == reference_m]
    expected_references = 2 if config.reference_checks and config.fit_dimensions else 0
    reference_passed = len(references) == expected_references and all(
        row["converged"]
        and row["covariance_error_bound"] <= config.numerical_error_budget
        and row["matrix_discrepancy"] <= 2 * config.numerical_error_budget
        for row in references
    )
    failed_fits = [
        row
        for row in fit_rows
        if not row["converged"]
        or row["covariance_error_bound"] > config.numerical_error_budget
        or row["minimum_eigenvalue"] < -config.numerical_tolerance
        or row["trace"] > config.trace_cap + config.numerical_tolerance
    ]
    max_composition = max(composition_errors, default=0.0)
    max_leakage = max(leakage_errors, default=0.0)
    numerical_passed = (
        all(row["passed"] for row in prepared["numerical_checks"])
        and max_composition <= config.transport_tolerance
        and max_leakage <= config.transport_tolerance
    )
    resolution_passed = all(
        row["tail_resolved"]
        and max(
            row["full_source_tail_bound"],
            row["full_innovation_tail_bound"],
            row["full_state_mse_tail_bound"],
        )
        <= config.tail_budget
        for row in reference_rows
    )
    results = {
        **_summaries(rows, risk_rows, fit_rows, config),
        "population": population,
        "numerical_checks": prepared["numerical_checks"],
        "reference_checks": references,
        "kernels": prepared["kernels"],
        "acceptance": {
            "passed": numerical_passed and not failed_fits and reference_passed,
            "numerical_checks_passed": numerical_passed,
            "failed_fits": len(failed_fits),
            "reference_passed": reference_passed,
            "reference_cases": len(references),
            "reference_resolution_passed": resolution_passed,
            "tail_budget": config.tail_budget,
            "maximum_time_composition_error": max_composition,
            "maximum_leakage_identity_error": max_leakage,
            "maximum_covariance_error_bound": max(
                [row["covariance_error_bound"] for row in fit_rows], default=0.0
            ),
            "statistical_performance_gate": None,
        },
        "model": {
            "basis_family_id": prepared["models"][reference_m].family_id,
            "source_family_id": prepared["source"].family_id,
            "source_eigenvalues_at_reference": prepared["source"].eigenvalues(reference_m),
            "basis_ids": {m: model.basis_id for m, model in prepared["models"].items()},
            "initial_state": "zero deterministic curve",
            "exposures": prepared["exposures"],
            "exposure_normalization": "one fixed unit-H infinite exposure; no J-dependent normalization",
            "forcing_bands": [{"lower": band["lower"], "upper": band["upper"]} for band in bands],
        },
        "protocol": {
            "source": "one fixed infinite diagonal spectrum; unrenormalized finite forcing prefixes",
            "time": "one finest exact Gaussian recursion; coarser innovations compose its same increments",
            "projection": "J-coordinate projection of each M-forcing process, never autonomous J replacement",
            "innovation_target": "P_J V_delta,M P_J; distinct from source C and level covariance K_t",
            "operational_target": "intended V target plus time-varying predictable-state leakage",
            "operational_centering": "pooled operational covariance centered on V plus mean preceding-state leakage",
            "gaussian_mse": "analytic independent-Gaussian control applies only to oracle innovation sample covariance",
            "risk_target": "fixed full-H exposure variance at finite reference M; analytic infinite-source tail retained",
            "mcse": "independent replications, paired by common forcing bands and common fine paths",
            "resolution": "tail flags are separate from numerical run acceptance; smoke may remain unresolved",
        },
    }
    _write_json(output / "fit_results.json", fit_rows)
    _write_json(output / "numerical_checks.json", prepared["numerical_checks"])
    _write_csv(output / "replications.csv", rows)
    _write_csv(output / "risks.csv", risk_rows)
    matrices = {}
    for index, ((m, j, delta), case) in enumerate(prepared["cases"].items()):
        for name in ("innovation", "operational_mean", "leakage_mean", "leakage_terminal"):
            matrices[f"case{index}_{name}"] = case[name]
        case["population"]["matrix_key_prefix"] = f"case{index}"
    np.savez_compressed(output / "population_covariances.npz", **matrices)
    # Include matrix-key provenance in the population table itself.
    _write_json(output / "population.json", population)
    _write_json(output / "results.json", results)
    from .infinite_reporting import write_report

    write_report(output, config, results)
    _write_json(
        output / "status.json",
        {"status": "complete" if results["acceptance"]["passed"] else "failed"},
    )
    if not results["acceptance"]["passed"]:
        raise RuntimeError(
            f"Infinite-transport numerical checks failed; inspect {output / 'report.md'}"
        )
    return output
