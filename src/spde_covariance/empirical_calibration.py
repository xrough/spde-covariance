"""Finite transport calibration conditional on reconstructed settlement curves.

These are quasi-model fits to reconstructed states, not latent-source recovery
or fresh-noise deconvolution. Exact finite Laguerre transport, actual interval
lengths and original price units remain explicit. Callers supply a training
prefix only; no future outcome or latent truth is accepted by these APIs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from .admissibility import _integer, _scalar, covariance_integration_matrix
from .geometry import smat, svec
from .hedging import _snapshot
from .laguerre import LaguerreTransport
from .linalg import _real_array, project_psd_trace
from .trajectories import _times


@dataclass(frozen=True, slots=True, eq=False)
class CrossSectionFit:
    coefficients: np.ndarray
    fitted_prices: np.ndarray
    residuals: np.ndarray
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("coefficients", "fitted_prices", "residuals", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))

    @property
    def available(self):
        return self.diagnostics["available"]


def fit_cross_section(tenors, prices, dimension=3, smoothing=1e-6, *, rank_rtol=1e-10):
    """Shape-ridge fit in the fixed orthonormal Laguerre basis.

    Minimize mean squared quotation error / 2 plus lambda times the squared
    nonconstant coefficients / 2. The constant level is unpenalized, so adding
    a constant to every quotation shifts only that state coordinate. lambda
    equals smoothing times the evaluation normal-matrix spectral norm.
    Rank-deficient data retain a candidate but are explicitly unavailable;
    regularization does not create observed state directions.
    """
    dimension = _integer(dimension, "dimension")
    smoothing = _scalar(smoothing, "smoothing")
    rank_rtol = _scalar(rank_rtol, "rank_rtol", positive=True)
    if rank_rtol >= 1:
        raise ValueError("rank_rtol must be below one")
    tenors = _real_array(tenors, name="tenors")
    prices = _real_array(prices, name="prices")
    if tenors.ndim != 1 or not len(tenors) or np.any(tenors < 0) or prices.shape != tenors.shape:
        raise ValueError(
            "tenors and prices must be matching nonempty vectors with nonnegative tenors"
        )
    evaluation = LaguerreTransport(dimension).evaluate(tenors)
    normal = evaluation.T @ evaluation / len(prices)
    geometry_norm = float(np.linalg.eigvalsh(normal)[-1])
    regularization = smoothing * geometry_norm
    penalty = np.eye(dimension)
    penalty[0, 0] = 0
    design = np.vstack([evaluation / np.sqrt(len(prices)), np.sqrt(regularization) * penalty])
    # Centering makes the translation invariance numerically robust for large
    # quoted price levels while leaving the unpenalized intercept exact.
    price_level = float(np.mean(prices))
    response = np.r_[(prices - price_level) / np.sqrt(len(prices)), np.zeros(dimension)]
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    coefficients[0] += price_level
    fitted = evaluation @ coefficients
    residuals = fitted - prices
    singular = np.linalg.svd(evaluation, compute_uv=False)
    threshold = rank_rtol * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    hessian = normal + regularization * penalty
    effective_df = float(np.trace(np.linalg.pinv(hessian, rcond=1e-12) @ normal))
    gradient = evaluation.T @ residuals / len(prices) + regularization * penalty @ coefficients
    normal_residual = float(
        np.linalg.norm(gradient) / (1 + np.linalg.norm(evaluation.T @ prices / len(prices)))
    )
    return CrossSectionFit(
        coefficients,
        fitted,
        residuals,
        {
            "available": bool(rank == dimension and normal_residual <= 1e-8),
            "dimension": dimension,
            "basis_id": LaguerreTransport(dimension).basis_id,
            "observed_contract_count": len(prices),
            "design_rank": rank,
            "rank_threshold": float(threshold),
            "design_singular_values": singular,
            "design_condition_number": float(singular[0] / singular[-1])
            if rank == dimension
            else float("inf"),
            "effective_degrees_of_freedom": effective_df,
            "smoothing_relative": smoothing,
            "regularization": regularization,
            "evaluation_gram_norm": geometry_norm,
            "constant_penalized": False,
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "maximum_absolute_residual": float(np.max(np.abs(residuals))),
            "relative_normal_equation_residual": normal_residual,
            "tenor_min": float(np.min(tenors)),
            "tenor_max": float(np.max(tenors)),
            "scope": "regularized cross-section reconstruction; no independent observation-noise estimate",
        },
    )


@lru_cache(maxsize=512)
def _dynamics(dimension, delta, transport_adjustment):
    if not transport_adjustment:
        size = dimension * (dimension + 1) // 2
        return _snapshot((np.eye(dimension), delta * np.eye(dimension), np.eye(size)))
    model = LaguerreTransport(dimension, cache_max_bytes=0)
    block = np.zeros((2 * dimension, 2 * dimension))
    block[:dimension, :dimension] = model.generator
    block[:dimension, dimension:] = np.eye(dimension)
    drift_map = expm(delta * block)[:dimension, dimension:]
    integration = covariance_integration_matrix(model, delta)
    return _snapshot((model.matrix(delta), drift_map, integration))


def integrated_drift_matrix(delta, dimension, *, transport_adjustment=True):
    """Return integral_0^delta S(r) dr, including the constant generator mode."""
    delta = _scalar(delta, "delta", positive=True)
    dimension = _integer(dimension, "dimension")
    if not isinstance(transport_adjustment, (bool, np.bool_)):
        raise ValueError("transport_adjustment must be boolean")
    return _dynamics(dimension, delta, bool(transport_adjustment))[1]


def _trajectory(coefficients, times):
    values = _real_array(coefficients, name="coefficients")
    dates = _times(times)
    if values.ndim != 2 or values.shape[0] != len(dates) or not values.shape[1]:
        raise ValueError("coefficients must have one nonempty state row per time")
    return values, dates


def extract_coefficient_innovations(coefficients, times, *, drift=None, transport_adjustment=True):
    """Exact finite transport residuals divided by each actual sqrt(delta).

    drift is one constant physical-time coefficient vector (price/year).
    Reconstruction errors remain in the residuals and may be dependent.
    """
    values, dates = _trajectory(coefficients, times)
    if not isinstance(transport_adjustment, (bool, np.bool_)):
        raise ValueError("transport_adjustment must be boolean")
    dimension = values.shape[1]
    mean = np.zeros(dimension) if drift is None else _real_array(drift, name="drift")
    if mean.shape != (dimension,):
        raise ValueError("drift must have one coefficient per state coordinate")
    result = []
    for previous, following, delta in zip(values[:-1], values[1:], np.diff(dates), strict=True):
        transition, drift_map, _ = _dynamics(dimension, float(delta), bool(transport_adjustment))
        result.append((following - transition @ previous - drift_map @ mean) / np.sqrt(delta))
    return _snapshot(np.asarray(result))


@dataclass(frozen=True, slots=True, eq=False)
class EmpiricalSourceFit:
    source_covariance: np.ndarray
    drift: np.ndarray
    bridge_covariance: np.ndarray
    normalized_residuals: np.ndarray
    diagnostics: Mapping

    def __post_init__(self):
        for name in (
            "source_covariance",
            "drift",
            "bridge_covariance",
            "normalized_residuals",
            "diagnostics",
        ):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))

    @property
    def covariance(self):
        """The fitted finite source C, not a pooled innovation V."""
        return self.source_covariance

    @property
    def converged(self):
        return self.diagnostics["converged"]

    @property
    def available(self):
        return self.converged


def _gap(matrix, hessian, rhs, cap):
    gradient = smat(hessian @ svec(matrix) - rhs)
    return max(
        0.0, float(np.sum(gradient * matrix) - cap * min(np.linalg.eigvalsh(gradient)[0], 0.0))
    )


def fit_source_covariance(
    coefficients,
    times,
    window=None,
    *,
    regularization_relative=1e-3,
    trace_cap=None,
    trace_cap_multiplier=100.0,
    drift="constant",
    drift_regularization_relative=1e-6,
    bridge_delta=None,
    transport_adjustment=True,
    numerical_error_budget=1e-5,
    max_iterations=20_000,
):
    """Certified convex source moment fit on a supplied training prefix.

    window is a trailing number of intervals. Constant drift is estimated by
    interval-normalized ridge least squares using only those intervals; zero
    drift is an explicit alternative. Fit C>=0, tr(C)<=cap through each
    T_delta(C)=delta^-1 integral S C S.T. Raw increments and identity dynamics
    form the explicitly misspecified no-transport ablation when requested.

    The numerical error budget is relative to covariance_scale, a training-only
    residual scale. Diagnostics also report the bound in original covariance
    units. It concerns the constrained moment optimizer conditional on the
    fitted drift/reconstructed states, not statistical error or a latent C.
    Default trace cap is 100 times a training trace proxy, not a confidence
    bound. Explicit cap zero, including a zero-residual fit, is supported.
    """
    values, dates = _trajectory(coefficients, times)
    if not isinstance(transport_adjustment, (bool, np.bool_)):
        raise ValueError("transport_adjustment must be boolean")
    transport_adjustment = bool(transport_adjustment)
    count = len(dates) - 1 if window is None else _integer(window, "window")
    if count < 2 or count >= len(dates):
        raise ValueError("window must select at least two available trailing intervals")
    values, dates = values[-count - 1 :], dates[-count - 1 :]
    dimension = values.shape[1]
    deltas = np.diff(dates)
    relative = _scalar(regularization_relative, "regularization_relative", positive=True)
    drift_relative = _scalar(drift_regularization_relative, "drift_regularization_relative")
    budget = _scalar(numerical_error_budget, "numerical_error_budget", positive=True)
    multiplier = _scalar(trace_cap_multiplier, "trace_cap_multiplier", positive=True)
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 0
    ):
        raise ValueError("max_iterations must be a nonnegative integer")
    if drift not in ("constant", "zero"):
        raise ValueError("drift must be 'constant' or 'zero'")
    raw = extract_coefficient_innovations(values, dates, transport_adjustment=transport_adjustment)
    dynamics = [_dynamics(dimension, float(delta), transport_adjustment) for delta in deltas]
    drift_design = np.vstack(
        [item[1] / np.sqrt(delta) for item, delta in zip(dynamics, deltas, strict=True)]
    )
    drift_gram = drift_design.T @ drift_design / count
    drift_penalty = drift_relative * float(np.linalg.eigvalsh(drift_gram)[-1])
    if drift == "constant":
        augmented = np.vstack(
            [drift_design / np.sqrt(count), np.sqrt(drift_penalty) * np.eye(dimension)]
        )
        response = np.r_[raw.reshape(-1) / np.sqrt(count), np.zeros(dimension)]
        fitted_drift = np.linalg.lstsq(augmented, response, rcond=None)[0]
    else:
        fitted_drift = np.zeros(dimension)
    residuals = extract_coefficient_innovations(
        values, dates, drift=fitted_drift, transport_adjustment=transport_adjustment
    )
    maps = [item[2] for item in dynamics]
    gram = sum(item.T @ item for item in maps) / count
    spectrum = np.linalg.eigvalsh(gram)
    regularization = relative * float(spectrum[-1])
    hessian = gram + regularization * np.eye(len(gram))
    eigenvalues = np.linalg.eigvalsh(hessian)
    strong, lipschitz = float(eigenvalues[0]), float(eigenvalues[-1])
    trace_gain = smat(sum(item.T @ svec(np.eye(dimension)) for item in maps) / count)
    minimum_trace_gain = float(np.linalg.eigvalsh(trace_gain)[0])
    with np.errstate(over="ignore", invalid="ignore"):
        empirical_trace = float(np.sum(residuals**2) / count)
    if not np.isfinite(empirical_trace) or minimum_trace_gain <= 0:
        raise ValueError("residual covariance scale or transport trace gain is not representable")
    scale = empirical_trace / minimum_trace_gain if empirical_trace > 0 else 1.0
    cap = (
        multiplier * empirical_trace / minimum_trace_gain
        if trace_cap is None
        else _scalar(trace_cap, "trace_cap")
    )
    scaled_cap = cap / scale
    if not np.isfinite(scale) or not np.isfinite(cap) or not np.isfinite(scaled_cap):
        raise ValueError("source trace scale exceeds floating-point range; rescale price units")
    scaled_residuals = residuals / np.sqrt(scale)
    moments = [svec(np.outer(row, row)) for row in scaled_residuals]
    rhs = sum(item.T @ moment for item, moment in zip(maps, moments, strict=True)) / count
    constant = sum(float(moment @ moment) for moment in moments) / (2 * count)
    candidate = np.zeros((dimension, dimension))
    extrapolated = candidate.copy()
    momentum = (np.sqrt(lipschitz) - np.sqrt(strong)) / (np.sqrt(lipschitz) + np.sqrt(strong))
    gap = float("inf")
    for iteration in range(int(max_iterations) + 1):
        gap = _gap(candidate, hessian, rhs, scaled_cap)
        if gap <= 0.5 * strong * budget**2:
            break
        if iteration == max_iterations:
            break
        gradient = smat(hessian @ svec(extrapolated) - rhs)
        following = project_psd_trace(extrapolated - gradient / lipschitz, scaled_cap)
        old_vector, new_vector = svec(candidate), svec(following)
        old_objective = 0.5 * old_vector @ hessian @ old_vector - rhs @ old_vector
        new_objective = 0.5 * new_vector @ hessian @ new_vector - rhs @ new_vector
        if new_objective > old_objective + 1e-14 * max(1.0, abs(old_objective)):
            gradient = smat(hessian @ old_vector - rhs)
            following = project_psd_trace(candidate - gradient / lipschitz, scaled_cap)
            extrapolated = following.copy()
        else:
            extrapolated = following + momentum * (following - candidate)
        candidate = following
    covariance = scale * candidate
    bridge_delta = (
        float(deltas[-1])
        if bridge_delta is None
        else _scalar(bridge_delta, "bridge_delta", positive=True)
    )
    bridge = smat(_dynamics(dimension, bridge_delta, transport_adjustment)[2] @ svec(covariance))
    vector = svec(candidate)
    objective_scaled = float(0.5 * vector @ hessian @ vector - rhs @ vector + constant)
    converged = bool(gap <= 0.5 * strong * budget**2)
    error_relative = float(np.sqrt(2 * gap / strong))
    threshold = float(spectrum[-1] * 1e-10)
    residual_eigenvalues = np.linalg.eigvalsh(residuals.T @ residuals / count)
    return EmpiricalSourceFit(
        covariance,
        fitted_drift,
        bridge,
        residuals,
        {
            "converged": converged,
            "status": "converged" if converged else "maximum_iterations",
            "iterations": iteration,
            "dimension": dimension,
            "interval_count": count,
            "training_start_time": float(dates[0]),
            "training_end_time": float(dates[-1]),
            "interval_lengths": deltas,
            "minimum_interval": float(deltas.min()),
            "maximum_interval": float(deltas.max()),
            "bridge_delta": bridge_delta,
            "transport_adjustment": transport_adjustment,
            "dynamics": "exact finite Laguerre transport"
            if transport_adjustment
            else "A=0 no-transport ablation",
            "target": "finite source covariance C conditional on reconstructed states and fitted drift",
            "bridge_target": "normalized innovation V_delta; raw change covariance is delta * V_delta",
            "drift_policy": drift,
            "fitted_drift_parameter_count": dimension if drift == "constant" else 0,
            "drift_regularization_relative": drift_relative,
            "drift_regularization": drift_penalty,
            "drift_design_rank": int(np.linalg.matrix_rank(drift_design)),
            "regularization_relative": relative,
            "regularization": regularization,
            "covariance_scale": scale,
            "empirical_residual_trace": empirical_trace,
            "normalized_residual_mean": np.mean(residuals, axis=0),
            "minimum_mean_transport_trace_gain": minimum_trace_gain,
            "trace_cap": cap,
            "trace_cap_policy": "training residual scale times declared multiplier"
            if trace_cap is None
            else "explicit supplied source trace cap",
            "trace_cap_multiplier": multiplier,
            "source_trace": float(np.trace(covariance)),
            "minimum_source_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
            "objective": objective_scaled * scale**2,
            "objective_scaled": objective_scaled,
            "objective_gap_bound": gap * scale**2,
            "objective_gap_bound_scaled": gap,
            "strong_convexity": strong,
            "lipschitz": lipschitz,
            "relative_covariance_error_bound": error_relative,
            "covariance_error_bound": scale * error_relative,
            "numerical_error_budget": budget,
            "linear_map_rank": int(np.count_nonzero(spectrum > threshold)),
            "linear_map_coordinate_dimension": len(spectrum),
            "linear_map_rank_threshold": threshold,
            "linear_map_condition_number": float(spectrum[-1] / spectrum[0])
            if spectrum[0] > threshold
            else float("inf"),
            "empirical_residual_rank": int(
                np.count_nonzero(residual_eigenvalues > max(residual_eigenvalues[-1], 0) * 1e-10)
            ),
            "observability_scope": "source-to-coefficient covariance map, conditional on reconstructed states; not latent identifiability from settlements",
            "certificate_scope": "floating-point convex optimizer bound conditional on reconstructed states and fitted drift; not statistical error",
            "latent_fresh_noise_claim": False,
            "source_recovery_claim": False,
        },
    )


def forecast_contract_changes(origin_prices, end_tenors, source_fit, delta):
    """Anchor next same-contract mean at observed origin prices.

    End tenors are for the same contracts at the next observation date. The
    returned covariance is in raw price-change units squared, not V units.
    No future settlement, reconstructed-origin residual or fake mean-reversion
    adjustment enters this forecast. This is a conditional plug-in model.
    """
    if not isinstance(source_fit, EmpiricalSourceFit) or not source_fit.converged:
        raise ValueError("source_fit must be a converged EmpiricalSourceFit")
    prices = _real_array(origin_prices, name="origin_prices")
    tenors = _real_array(end_tenors, name="end_tenors")
    if prices.ndim != 1 or not len(prices) or tenors.shape != prices.shape or np.any(tenors < 0):
        raise ValueError(
            "origin_prices and nonnegative end_tenors must be matching nonempty vectors"
        )
    delta = _scalar(delta, "delta", positive=True)
    dimension = len(source_fit.drift)
    adjustment = source_fit.diagnostics["transport_adjustment"]
    _, drift_map, integration = _dynamics(dimension, delta, adjustment)
    evaluation = LaguerreTransport(dimension).evaluate(tenors)
    mean_changes = evaluation @ drift_map @ source_fit.drift
    covariance = delta * evaluation @ smat(integration @ svec(source_fit.covariance)) @ evaluation.T
    return _snapshot(
        {
            "mean_changes": mean_changes,
            "mean_prices": prices + mean_changes,
            "covariance": (covariance + covariance.T) / 2,
            "delta": delta,
            "end_tenors": tenors,
            "anchored_at_observed_origin_prices": True,
            "target": "same-contract raw price changes; covariance in quotation units squared",
            "parameter_uncertainty_included": False,
            "reconstruction_error_included": False,
        }
    )


def source_risk_error_bound(source_fit, end_tenors, positions, delta, multiplier=1.0):
    """Propagate the source optimizer-distance bound to a raw portfolio variance.

    Positions are signed contract counts; multiplier converts quotation changes
    to currency P&L. This excludes sampling, reconstruction and drift/model
    errors and is conditional on the numerical source integration map.
    """
    if not isinstance(source_fit, EmpiricalSourceFit):
        raise ValueError("source_fit must be EmpiricalSourceFit")
    tenors = _real_array(end_tenors, name="end_tenors")
    positions = _real_array(positions, name="positions")
    if tenors.ndim != 1 or not len(tenors) or positions.shape != tenors.shape or np.any(tenors < 0):
        raise ValueError("positions and nonnegative end_tenors must be matching nonempty vectors")
    delta = _scalar(delta, "delta", positive=True)
    multiplier = _scalar(multiplier, "multiplier", positive=True)
    dimension = len(source_fit.drift)
    exposure = multiplier * LaguerreTransport(dimension).evaluate(tenors).T @ positions
    integration = _dynamics(dimension, delta, source_fit.diagnostics["transport_adjustment"])[2]
    functional = delta * integration.T @ svec(np.outer(exposure, exposure))
    return float(np.linalg.norm(functional) * source_fit.diagnostics["covariance_error_bound"])
