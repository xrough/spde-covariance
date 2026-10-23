"""Bounded deterministic-design stress checks for the numerical foundations.

This is a diagnostic harness, not a scalability claim. Runtime and Python-
traced memory exclude data generation and do not measure all native BLAS memory.
"""

from __future__ import annotations

import platform
import time
import tracemalloc
from collections.abc import Mapping

import numpy as np

from .bridge import integrate_innovation_covariance
from .estimation import fit_covariance
from .simulation import simulate_innovations
from .spaces import SplineSpace
from .trajectories import InnovationBatch, extract_innovations, simulate_transport
from .transport import ExponentialTransport, ProjectedSplineTransport
from .workflow import fit_innovation_covariance, interval_diagnostics


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("stress diagnostics must contain finite numerical values")
    return value


def _bridge_cases(model, source, budget):
    factor = np.linalg.cholesky(source)
    rows = []
    for dt in (1 / 12, 1 / 52, 1 / 252):
        analytic = model.innovation_covariance(source, dt)
        quadrature = integrate_innovation_covariance(
            lambda time: factor,
            start=0,
            delta=dt,
            shifted_source=lambda time, lag: model.matrix(lag) @ factor,
            quadrature_order=24,
        ).innovation_covariance
        rows.append(
            {
                "dt": dt,
                "v_minus_c_hs": np.linalg.norm(analytic - source),
                "analytic_vs_gauss_hs": np.linalg.norm(analytic - quadrature),
                "quadrature_order": 24,
            }
        )
    lengths = np.array([0.01, 0.03, 0.02])
    batch = InnovationBatch(
        np.r_[0, np.cumsum(lengths)], np.zeros((3, model.dimension)), model.basis_id, model.kind
    )
    covariances = [model.innovation_covariance(source, dt) for dt in lengths]
    rejected, reason = False, None
    try:
        fit_innovation_covariance(
            batch,
            regularization=0.001,
            trace_cap=2,
            covariance_assumption="common",
            numerical_error_budget=budget,
        )
    except ValueError as error:
        rejected, reason = True, str(error)
    return rows, {
        "diagnostics": interval_diagnostics(batch),
        "constant_source_covariance": source,
        "innovation_covariance_span_hs": max(
            np.linalg.norm(left - right) for left in covariances for right in covariances
        ),
        "pooling_rejected": rejected,
        "pooling_rejection_reason": reason,
    }


def _omitted_tail_squared(model, length):
    rates = model.rates
    denominator = rates[:, None] + rates[None, :] - 1 / model.tau
    raw = np.outer(rates, rates) * np.exp(-denominator * length) / denominator
    return float(np.trace(model.transform.T @ raw @ model.transform))


def _projection_cases(model):
    rows = []
    grid, shift = np.linspace(0, 3, 61), 0.1
    for length in (4.0, 8.0):
        for dimension in (4, 8, 16):
            space = SplineSpace(n_basis=dimension, length=length)
            coarse = space.project(model.evaluate, model.derivative, quadrature_order=12)
            projected = space.project(model.evaluate, model.derivative, quadrature_order=24)
            nodes, weights = space.quadrature(order=24, weighted=True)
            anchor_squared = float(
                np.sum((model.evaluate([0]) - space.evaluate([0]) @ projected) ** 2)
            )
            derivative_squared = float(
                np.sum(
                    weights[:, None]
                    * (model.derivative(nodes) - space.derivative(nodes) @ projected) ** 2
                )
            )
            tail_squared = _omitted_tail_squared(model, length)
            error_squared = anchor_squared + derivative_squared + tail_squared
            shift_matrix = space.shift_matrix(shift)
            directly_shifted = space.project(
                lambda x: model.evaluate(x + shift),
                lambda x: model.derivative(x + shift),
                quadrature_order=24,
            )
            grid_difference = space.evaluate(grid) @ shift_matrix @ projected - model.evaluate(
                grid + shift
            )
            metric_defect = space.transform.T @ space.raw_gram @ space.transform - np.eye(dimension)
            rows.append(
                {
                    "basis_size": dimension,
                    "padding_length": length,
                    "tau": space.tau,
                    "projection_error_h": np.sqrt(max(0.0, error_squared)),
                    "anchor_error": np.sqrt(anchor_squared),
                    "interior_derivative_error_h": np.sqrt(derivative_squared),
                    "omitted_tail_error_h": np.sqrt(max(0.0, tail_squared)),
                    "projection_quadrature_coefficient_error": np.linalg.norm(coarse - projected),
                    "projection_identity_discrepancy": abs(
                        error_squared - (model.dimension - np.sum(projected**2))
                    ),
                    "shift_function_rmse": np.sqrt(np.mean(grid_difference**2)),
                    "projected_shift_coefficient_error_h": np.linalg.norm(
                        shift_matrix @ projected - directly_shifted
                    ),
                    "semigroup_defect": np.linalg.norm(
                        shift_matrix @ shift_matrix - space.shift_matrix(2 * shift)
                    ),
                    "raw_gram_condition": np.linalg.cond(space.raw_gram),
                    "metric_orthogonality_error": np.linalg.norm(metric_defect),
                    "propagation_method": "projected_mild_recursion",
                }
            )
    return {
        "reference": "four exact orthonormal exponential functions on the half-line",
        "projection_metric": "anchored weighted H; summed squared errors over four functions",
        "boundary": "spline constant tails; omitted exact derivative energy integrated analytically",
        "grid": grid,
        "shift": shift,
        "quadrature_orders": [12, 24],
        "rows": rows,
        "interpretation": "padding changes both boundary error and resolution at fixed basis size",
    }


def _conditioning_case(model):
    close_rates = np.array([1.0, 1.0 + 1e-8, 1.0 + 2e-8, 1.0 + 3e-8])
    rejected, message = False, None
    try:
        ExponentialTransport(rates=close_rates)
    except ValueError as error:
        rejected, message = True, str(error)
    return {
        "default_rates": model.rates,
        "default_raw_gram_condition": model.gram_condition_number,
        "close_rates": close_rates,
        "close_rates_rejected": rejected,
        "rejection_reason": message,
    }


def _temporal_cases(model, seed, smoke, budget):
    count, dt = (128 if smoke else 512), 1 / 52
    windows = [8, 32, 64] if smoke else [32, 128, 256]
    first, second = np.diag([0.4, 0.3, 0.2, 0.1]), np.diag([0.1, 0.2, 0.3, 0.4])
    sources = np.repeat(first[None, :, :], count, axis=0)
    sources[count // 2 :] = second
    simulation = simulate_transport(model, np.arange(count + 1) * dt, sources, seed=seed)
    innovations = extract_innovations(simulation.trajectory, model)
    post_change_endpoint = count // 2 + windows[0]
    rows = []
    grid_map = model.evaluate(np.linspace(0, 3, 31))
    exposure = model.evaluate([1, 2])[0] - model.evaluate([1, 2])[1]
    for endpoint, label in ((count, "terminal"), (post_change_endpoint, "post_change")):
        available = InnovationBatch(
            innovations.times[: endpoint + 1],
            innovations.values[:endpoint],
            model.basis_id,
            model.kind,
            curve_unit=innovations.curve_unit,
        )
        target = simulation.innovation_covariances[endpoint - 1]
        for window in windows:
            width = (window - 0.5) * dt
            fit = fit_innovation_covariance(
                available,
                regularization=0.001,
                trace_cap=2,
                covariance_assumption="local",
                numerical_error_budget=budget,
                trailing_window=width,
            )
            actual_window = np.flatnonzero(fit.estimate.weights > 0)
            mean_covariance = np.einsum(
                "i,ijk->jk", fit.estimate.weights, simulation.innovation_covariances[:endpoint]
            )
            error = fit.covariance - target
            rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "window_intervals": window,
                    "selected_intervals": len(actual_window),
                    "trailing_window_years": width,
                    "window_covariance_bias_hs": np.linalg.norm(mean_covariance - target),
                    "actual_covariance_error_hs": np.linalg.norm(error),
                    "observed_covariance_rmse": np.linalg.norm(error),
                    "off_grid_kernel_rmse": np.sqrt(np.mean((grid_map @ error @ grid_map.T) ** 2)),
                    "spread_risk_absolute_error": abs(exposure @ error @ exposure),
                    "operator_norm_error": np.linalg.norm(error, ord=2),
                    "relative_operator_error": np.linalg.norm(error, ord=2)
                    / np.linalg.norm(target, ord=2),
                    "solver_iterations": fit.estimate.iterations,
                    "objective_gap_bound": fit.estimate.objective_gap_bound,
                    "numerical_covariance_error_bound": fit.estimate.covariance_error_bound,
                    "converged": fit.estimate.converged,
                    "covariance_assumption": "local",
                    "target": fit.target,
                }
            )
    return {
        "seed": seed,
        "replications": 1,
        "interval_count": count,
        "dt": dt,
        "change_after_interval": count // 2,
        "source_before": first,
        "source_after": second,
        "source_regime_change_hs": np.linalg.norm(second - first),
        "windows": windows,
        "regularization": 0.001,
        "regularization_relative_to_qmax": 0.001,
        "trace_cap": 2,
        "rows": rows,
        "terminal_window_note": "requested terminal windows are wholly within the second regime",
        "post_change_note": "same windows evaluated soon after the change to expose localization bias",
        "assumption": "local covariance approximation declared; no automatic stationarity validation",
        "metric_conventions": {
            "observed_covariance_rmse": "Frobenius error for the full identity observation map",
            "off_grid_kernel_rmse": "entrywise RMSE on the predeclared 31-point [0,3] maturity grid",
            "spread_risk_absolute_error": "absolute variance error of predeclared f(1)-f(2)",
        },
    }


def _performance_cases(seed, smoke, budget):
    rows = []
    for dimension in [4] if smoke else [4, 8, 16, 24]:
        space = SplineSpace(n_basis=dimension, length=6)
        propagator = ProjectedSplineTransport(space)
        eigenvalues = 1 / np.arange(1, dimension + 1, dtype=float) ** 2
        source = np.diag(eigenvalues / np.sum(eigenvalues))
        covariance = propagator.innovation_covariance(source, 1 / 52, quadrature_order=24)
        operator = np.diag(np.linspace(0.5, 1, dimension))
        grid_map = space.evaluate(np.linspace(0, 3, 31))
        exposure = space.evaluate([1, 2])[0] - space.evaluate([1, 2])[1]
        exposure = exposure / np.linalg.norm(exposure)
        generated = simulate_innovations(
            covariance, [operator] * 64, times=np.arange(1, 65) / 52, seed=seed
        )
        own_trace = not tracemalloc.is_tracing()
        if own_trace:
            tracemalloc.start()
        initial_memory, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        try:
            fit = fit_covariance(
                generated.observations,
                regularization=0.001,
                trace_cap=2,
                numerical_error_budget=budget,
                absolute_tolerance=1e-12,
                relative_tolerance=0,
            )
            runtime = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if own_trace:
                tracemalloc.stop()
        error = fit.covariance - covariance
        rows.append(
            {
                "basis_size": dimension,
                "samples": 64,
                "seed": seed,
                "runtime_seconds": runtime,
                "python_tracemalloc_peak_bytes": max(0, peak - initial_memory),
                "solver_iterations": fit.iterations,
                "converged": fit.converged,
                "objective_gap_bound": fit.objective_gap_bound,
                "numerical_covariance_error_bound": fit.covariance_error_bound,
                "actual_covariance_error_hs": np.linalg.norm(error),
                "observed_covariance_rmse": np.linalg.norm(operator @ error @ operator.T),
                "off_grid_kernel_rmse": np.sqrt(np.mean((grid_map @ error @ grid_map.T) ** 2)),
                "spread_risk_absolute_error": abs(exposure @ error @ exposure),
                "operator_norm_error": np.linalg.norm(error, ord=2),
                "relative_operator_error": np.linalg.norm(error, ord=2)
                / np.linalg.norm(covariance, ord=2),
                "observation_singular_value_min": 0.5,
                "observation_singular_value_max": 1.0,
                "q_min": fit.q_min,
                "q_max": fit.lipschitz - fit.regularization,
                "propagation_method": propagator.kind,
            }
        )
    return {
        "rows": rows,
        "dt": 1 / 52,
        "padding_length": 6,
        "quadrature_order": 24,
        "regularization": 0.001,
        "trace_cap": 2,
        "solver": "accelerated_projected_gradient",
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 0,
        "max_iterations": 20000,
        "observation_design": "common diagonal O=diag(linspace(0.5,1,J)); 64 independent innovations",
        "source_spectrum": "full diagonal j^-2 spectrum normalized to trace one",
        "replications": 1,
        "timing_scope": "fit_covariance only; observations generated once per dimension before timing",
        "memory_scope": "Python tracemalloc peak increase; native BLAS allocations not fully measured",
        "interpretation": "bounded solver harness at a common certificate budget; no universal scalability claim",
        "metric_conventions": {
            "observed_covariance_rmse": "Frobenius error; the same diagonal observation map is used on all 64 dates",
            "off_grid_kernel_rmse": "entrywise RMSE on the predeclared 31-point [0,3] maturity grid",
            "spread_risk_absolute_error": "absolute risk error for f(1)-f(2), normalized to unit H-representer norm before data generation",
        },
    }


def run_stress_suite(*, seed=1729, smoke=False, numerical_error_budget=1e-4) -> dict:
    """Return JSON-serializable bridge, discretization, localization and solver checks."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be a boolean")
    raw_budget = np.asarray(numerical_error_budget)
    if (
        raw_budget.ndim != 0
        or np.iscomplexobj(raw_budget)
        or raw_budget.dtype.kind not in "iuf"
        or not np.isfinite(raw_budget)
        or raw_budget <= 0
    ):
        raise ValueError("numerical_error_budget must be a finite strictly positive real scalar")
    budget, seed = float(raw_budget), int(seed)
    model = ExponentialTransport(rates=[0.6, 1.0, 2.0, 4.0])
    source = np.diag([0.4, 0.3, 0.2, 0.1])
    bridge, irregular = _bridge_cases(model, source, budget)
    return _json_ready(
        {
            "settings": {
                "seed": seed,
                "smoke": smoke,
                "numerical_error_budget": budget,
                "exact_rates": model.rates,
                "tau": model.tau,
                "time_unit": "year",
                "curve_unit": "arbitrary curve unit",
                "bridge_source_covariance": source,
                "observation_noise": "none; clean normalized innovations",
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "randomness": "independent centered Gaussian innovations; deterministic designs",
            },
            "dt_bridge": bridge,
            "irregular_intervals": irregular,
            "spline_projection": _projection_cases(model),
            "conditioning": _conditioning_case(model),
            "temporal_variation": _temporal_cases(model, seed, smoke, budget),
            "performance": _performance_cases(seed, smoke, budget),
        }
    )
