"""A provenance-preserving workflow for common-step innovation covariance.

This layer requires an explicit covariance assumption and a numerical-error
budget. Irregular intervals are diagnosed but never silently pooled. Curve
extraction and fresh measurement-noise correction remain separate operations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimation import CovarianceEstimate, fit_covariance, uniform_trailing_weights
from .geometry import diagnose_design
from .linalg import _real_array
from .observations import Observation
from .trajectories import InnovationBatch, _label, _metadata_snapshot, _readonly, _times

FloatArray = NDArray[np.float64]
_INTERVAL_RTOL = 1e-10


@dataclass(frozen=True, slots=True)
class ObservedInnovationBatch:
    """Observed innovations and declared basis provenance, without latent values.

    Observation times must match the right endpoints. Operators must have a
    common latent dimension, though their row counts may differ. The basis ID
    declares the coordinate convention: individual Observation objects carry
    dimensions rather than a separate coordinate identifier.
    """

    times: FloatArray
    observations: tuple[Observation, ...]
    basis_id: str
    propagation_method: str
    time_unit: str = "year"
    curve_unit: str = "arbitrary curve unit"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        times = _times(self.times)
        observations = tuple(self.observations)
        if len(observations) != len(times) - 1 or not all(
            isinstance(item, Observation) for item in observations
        ):
            raise ValueError("observations must contain one Observation per interval")
        dimension = observations[0].dimension
        if any(item.dimension != dimension for item in observations):
            raise ValueError("observation operators must share a common basis dimension")
        if not np.allclose([item.time for item in observations], times[1:], rtol=1e-12, atol=1e-14):
            raise ValueError("observation times must match the interval right endpoints")
        if self.time_unit != "year":
            raise ValueError("time_unit must be 'year'; convert timestamps before fitting")
        for name in ("basis_id", "propagation_method", "curve_unit"):
            object.__setattr__(self, name, _label(getattr(self, name), name))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        provenance = {
            "basis_id": self.basis_id,
            "propagation_method": self.propagation_method,
            "time_unit": self.time_unit,
            "curve_unit": self.curve_unit,
            "innovation_unit": self.innovation_unit,
        }
        for key, expected in provenance.items():
            if key in self.metadata and (
                not isinstance(self.metadata[key], str) or self.metadata[key] != expected
            ):
                raise ValueError(f"metadata {key} contradicts the declared batch provenance")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "metadata", _metadata_snapshot(self.metadata))

    @property
    def dimension(self) -> int:
        return self.observations[0].dimension

    @property
    def n_intervals(self) -> int:
        return len(self.observations)

    @property
    def interval_lengths(self) -> FloatArray:
        return _readonly(np.diff(self.times), name="interval_lengths")

    @property
    def end_times(self) -> FloatArray:
        return self.times[1:]

    @property
    def start_times(self) -> FloatArray:
        return self.times[:-1]

    @property
    def innovation_unit(self) -> str:
        return f"{self.curve_unit}/sqrt(year)"


def observe_innovations(
    batch: InnovationBatch,
    operators: Sequence[ArrayLike] | ArrayLike | None = None,
    *,
    noise_covariances: Sequence[ArrayLike] | ArrayLike | None = None,
    seed: int = 0,
) -> ObservedInnovationBatch:
    """Observe a batch while retaining interval, basis, and physical-unit labels."""
    if not isinstance(batch, InnovationBatch):
        raise ValueError("batch must be an InnovationBatch")
    metadata = dict(batch.metadata)
    metadata.update(
        {
            "drift_convention": batch.drift_convention,
            "observation_noise": "fresh independent Gaussian noise added after extraction",
            "observation_seed": seed,
        }
    )
    return ObservedInnovationBatch(
        batch.times,
        batch.observe(operators, noise_covariances=noise_covariances, seed=seed),
        batch.basis_id,
        batch.propagation_method,
        time_unit=batch.time_unit,
        curve_unit=batch.curve_unit,
        metadata=metadata,
    )


def interval_diagnostics(data: InnovationBatch | ObservedInnovationBatch) -> dict[str, Any]:
    """Describe transport-bridge interval targets without performing any pooling.

    Different dt generally give different normalized V_i even under constant
    diffusion covariance C. Approximate equality uses relative tolerance 1e-10
    and zero absolute tolerance, so very small unequal intervals remain distinct.
    """
    if not isinstance(data, (InnovationBatch, ObservedInnovationBatch)):
        raise ValueError("data must be an InnovationBatch or ObservedInnovationBatch")
    lengths = data.interval_lengths
    common = bool(np.all(np.isclose(lengths, lengths[0], rtol=_INTERVAL_RTOL, atol=0)))
    minimum, maximum = float(np.min(lengths)), float(np.max(lengths))
    return {
        "n_intervals": data.n_intervals,
        "basis_id": data.basis_id,
        "propagation_method": data.propagation_method,
        "time_unit": data.time_unit,
        "curve_unit": data.curve_unit,
        "innovation_unit": data.innovation_unit,
        "interval_lengths": lengths,
        "delta_min": minimum,
        "delta_max": maximum,
        "relative_interval_variation": (maximum - minimum) / minimum,
        "equal_intervals": common,
        "interval_rtol": _INTERVAL_RTOL,
        "target": "common-step innovation covariance V_delta"
        if common
        else "interval-specific innovation covariances V_i; no automatic pooling",
        "bridge": "V_i = integral_0^dt_i S_r C_i S_r* dr / dt_i, distinct from C_i",
    }


class ConvergenceError(RuntimeError):
    """The solver failed its requested certificate; the feasible iterate is retained."""

    def __init__(self, estimate: CovarianceEstimate):
        self.estimate = estimate
        super().__init__(
            f"covariance fit did not converge: {estimate.status} after {estimate.iterations} "
            f"iterations; certified covariance error bound={estimate.covariance_error_bound:.6g}"
        )


@dataclass(frozen=True, slots=True)
class InnovationCovarianceFit:
    """A certified estimate with its covariance assumption and target preserved."""

    estimate: CovarianceEstimate
    diagnostics: Mapping[str, Any]
    basis_id: str
    interval_length: float
    target: str
    covariance_assumption: str
    propagation_method: str
    times: FloatArray
    curve_unit: str
    time_unit: str = "year"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "estimate",
            replace(
                self.estimate,
                covariance=_readonly(self.estimate.covariance, name="covariance"),
                weights=_readonly(self.estimate.weights, name="weights"),
            ),
        )
        object.__setattr__(self, "times", _times(self.times))
        object.__setattr__(self, "diagnostics", _metadata_snapshot(self.diagnostics))
        object.__setattr__(self, "metadata", _metadata_snapshot(self.metadata))

    @property
    def covariance(self) -> FloatArray:
        return self.estimate.covariance

    @property
    def innovation_unit(self) -> str:
        return f"{self.curve_unit}/sqrt(year)"


def _date_weights(data, weights, trailing_window):
    if weights is not None and trailing_window is not None:
        raise ValueError("weights and trailing_window are mutually exclusive")
    if trailing_window is not None:
        raw = np.asarray(trailing_window)
        if raw.ndim != 0 or np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
            raise ValueError("trailing_window must be a finite nonnegative width in years")
        return uniform_trailing_weights(data.end_times, float(raw))
    if weights is None:
        return np.full(data.n_intervals, 1 / data.n_intervals)
    values = _real_array(weights, name="weights")
    if values.shape != (data.n_intervals,) or np.any(values < 0) or not np.any(values > 0):
        raise ValueError("weights must be nonnegative, nonzero, and match the intervals")
    # Rescale before summation to preserve finite normalization for large weights.
    values = values / np.max(values)
    return values / np.sum(values)


def fit_innovation_covariance(
    data: InnovationBatch | ObservedInnovationBatch,
    *,
    regularization: float,
    trace_cap: float,
    covariance_assumption: str,
    numerical_error_budget: float,
    weights: ArrayLike | None = None,
    trailing_window: float | None = None,
    targets=None,
    rank_rtol: float = 1e-10,
    expected_basis_id: str | None = None,
    max_iterations: int = 20_000,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-8,
    initial: ArrayLike | None = None,
) -> InnovationCovarianceFit:
    """Fit V_delta for equal-step innovations under an explicitly declared assumption.

    ``covariance_assumption='common'`` declares covariance constant across the
    weighted dates; ``'local'`` declares a local approximation at the last date.
    Neither assumption is verified statistically by this routine. A trailing
    window is a calendar width in years; otherwise all dates receive equal
    weights by default. Irregular intervals require an interval-aware extension
    and are rejected even if source covariance might be constant.

    Full-coordinate InnovationBatch input is observed through the identity map.
    Partial observations enter only through ObservedInnovationBatch. Noise-
    corrected outer products remain valid when indefinite. Numerical failure
    raises ConvergenceError carrying the feasible low-level estimate.
    """
    if not isinstance(data, (InnovationBatch, ObservedInnovationBatch)):
        raise ValueError("data must be an InnovationBatch or ObservedInnovationBatch")
    if not isinstance(covariance_assumption, str) or covariance_assumption not in (
        "common",
        "local",
    ):
        raise ValueError("covariance_assumption must be explicitly 'common' or 'local'")
    raw_budget = np.asarray(numerical_error_budget)
    if (
        raw_budget.ndim != 0
        or np.iscomplexobj(raw_budget)
        or raw_budget.dtype.kind not in "iuf"
        or not np.isfinite(raw_budget)
        or raw_budget <= 0
    ):
        raise ValueError("numerical_error_budget must be a finite strictly positive real scalar")
    budget = float(raw_budget)
    if expected_basis_id is not None and expected_basis_id != data.basis_id:
        raise ValueError("data basis_id does not match expected_basis_id")
    intervals = interval_diagnostics(data)
    if not intervals["equal_intervals"]:
        raise ValueError(
            "irregular intervals have interval-specific V_i targets; pooling is unsupported. "
            "Use interval_diagnostics to inspect the transport bridge."
        )
    observed = observe_innovations(data) if isinstance(data, InnovationBatch) else data
    date_weights = _date_weights(observed, weights, trailing_window)
    diagnostics = diagnose_design(
        [item.operator for item in observed.observations],
        weights=date_weights,
        targets=targets,
        regularization=regularization,
        rank_rtol=rank_rtol,
    )
    estimate = fit_covariance(
        observed.observations,
        regularization=regularization,
        trace_cap=trace_cap,
        weights=date_weights,
        numerical_error_budget=budget,
        max_iterations=max_iterations,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        initial=initial,
    )
    if not estimate.converged or estimate.covariance_error_bound > budget:
        raise ConvergenceError(estimate)
    target = (
        "common-step innovation covariance V_delta"
        if covariance_assumption == "common"
        else "local approximation to terminal innovation covariance V_delta"
    )
    diagnostics.update(
        {
            "intervals": intervals,
            "covariance_assumption": covariance_assumption,
            "assumption_status": "declared by caller; not statistically verified",
            "weights": estimate.weights,
        }
    )
    metadata = dict(observed.metadata)
    metadata.update(
        {
            "covariance_assumption": covariance_assumption,
            "assumption_status": "declared by caller; not statistically verified",
            "target": target,
            "instantaneous_covariance_recovery": False,
            "numerical_error_budget": budget,
            "time_unit": observed.time_unit,
            "curve_unit": observed.curve_unit,
            "innovation_unit": observed.innovation_unit,
            "covariance_unit": f"({observed.curve_unit})^2/year",
            "trailing_window_years": trailing_window,
        }
    )
    return InnovationCovarianceFit(
        estimate,
        diagnostics,
        observed.basis_id,
        float(observed.interval_lengths[0]),
        target,
        covariance_assumption,
        observed.propagation_method,
        observed.times,
        observed.curve_unit,
        observed.time_unit,
        metadata,
    )
