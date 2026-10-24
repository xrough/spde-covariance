"""Kernel-facing adapters; grid kernels never masquerade as Hilbert matrices."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .coherent import CoherentConvergenceError, fit_coherent_partial
from .completion import complete_kernel, weighted_psd_projection
from .fragments import FragmentLayout
from .observation_statistics import summarize_observations
from .trajectories import InnovationBatch, _metadata_snapshot, _readonly
from .workflow import (
    InnovationCovarianceFit,
    ObservedInnovationBatch,
    fit_innovation_covariance,
    interval_diagnostics,
    observe_innovations,
)


@dataclass(frozen=True, slots=True)
class KernelCovarianceFit:
    """A grid kernel in curve²/year, with its measure and explicit estimand.

    ``kernel`` is the raw method output. ``projected_kernel`` is a separately
    requested weighted PSD repair, never an implicit replacement. Only ridge
    has an ``operator_fit`` and its Hilbert-coordinate numerical certificate.
    Coherent completion certifies its observed-partial optimization separately.
    """

    kernel: np.ndarray
    nodes: np.ndarray
    weights: np.ndarray
    grid_id: str
    method: str
    estimand: str
    diagnostics: Mapping
    metadata: Mapping
    operator_fit: InnovationCovarianceFit | None = None
    projected_kernel: np.ndarray | None = None

    def __post_init__(self):
        for name in ("kernel", "nodes", "weights"):
            object.__setattr__(self, name, _readonly(getattr(self, name), name=name))
        if self.kernel.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("kernel and node shapes disagree")
        if self.projected_kernel is not None:
            object.__setattr__(
                self, "projected_kernel", _readonly(self.projected_kernel, name="projected kernel")
            )
        for name in ("diagnostics", "metadata"):
            object.__setattr__(self, name, _metadata_snapshot(getattr(self, name)))


def fit_covariance_kernel(data, *, layout, method, method_options, covariance_assumption):
    """Estimate one common-step kernel from observations, without latent truth.

    Ridge options are passed unchanged to ``fit_innovation_covariance``.
    Canonical options: rank or fve, support_rtol, minimum_pair_count (default1),
    weights (date weights), and psd_projection (defaultFalse). Canonical input
    is dense point fragments only, with validated row-to-grid correspondence.
    Coherent options: regularization, eigenvalue_floor, numerical_error_budget,
    rho, max_iterations, support_rtol and the same date/count options. It fits
    compatible PSD cliques jointly before untruncated supported completion.
    """
    if not isinstance(layout, FragmentLayout):
        raise TypeError("layout must be a FragmentLayout")
    if not isinstance(method_options, Mapping):
        raise TypeError("method_options must be a mapping")
    options = dict(method_options)
    if not isinstance(data, (InnovationBatch, ObservedInnovationBatch)):
        raise TypeError("data must preserve innovation interval and basis provenance")
    observed = observe_innovations(data) if isinstance(data, InnovationBatch) else data
    validation = layout.validate_observations(
        observed.observations, basis_id=observed.basis_id, require_operator_validation=True
    )
    metadata = {
        **dict(observed.metadata),
        "basis_id": observed.basis_id,
        "propagation_method": observed.propagation_method,
        "time_unit": observed.time_unit,
        "maturity_unit": "year",
        "covariance_unit": f"({observed.curve_unit})^2/year",
        "grid_id": layout.grid_id,
        "layout_id": layout.layout_id,
        "covariance_assumption": covariance_assumption,
        "interval_lengths": observed.interval_lengths,
        "times": observed.times,
        "assumption_status": "declared by caller; not statistically verified",
    }
    projected = None
    operator_fit = None
    if method == "ridge":
        operator_fit = fit_innovation_covariance(
            observed, covariance_assumption=covariance_assumption, **options
        )
        evaluation = layout.evaluation_matrix
        kernel = evaluation @ operator_fit.covariance @ evaluation.T
        estimand = f"grid evaluation of {operator_fit.target} in the declared H basis"
        diagnostics = {
            "layout_validation": validation,
            "operator": operator_fit.diagnostics,
            "certificate_scope": "H-coordinate ridge optimizer only",
        }
    elif method in ("canonical", "coherent"):
        if covariance_assumption != "common":
            raise ValueError("canonical completion initially requires a common covariance")
        intervals = interval_diagnostics(observed)
        if not intervals["equal_intervals"]:
            raise ValueError("canonical completion requires equal innovation intervals")
        date_weights = options.pop("weights", None)
        minimum = options.pop("minimum_pair_count", 1)
        repair = options.pop("psd_projection", False)
        if isinstance(minimum, bool) or not isinstance(minimum, (int, np.integer)) or minimum < 1:
            raise ValueError("minimum_pair_count must be a positive integer")
        if not isinstance(repair, bool):
            raise ValueError("psd_projection must be boolean")
        allowed = (
            {"rank", "fve", "support_rtol"}
            if method == "canonical"
            else {
                "regularization",
                "eigenvalue_floor",
                "numerical_error_budget",
                "rho",
                "max_iterations",
                "support_rtol",
            }
        )
        if not set(options) <= allowed:
            raise ValueError(f"unknown {method} method options")
        statistics = summarize_observations(
            observed.observations, weights=date_weights, layout=layout
        )
        if np.any(statistics.pair_counts[layout.declared_mask] < minimum):
            raise ValueError(
                "insufficient observed pair counts for the declared dense fragment cover"
            )
        if np.any(statistics.pair_weight_sums[layout.declared_mask] <= 0):
            raise ValueError("every declared observed pair needs positive date-weight coverage")
        for i, (a, b) in enumerate(layout.intervals[1:], start=1):
            if not any(max(a, c) < min(b, d) for c, d in layout.intervals[:i]):
                raise ValueError("canonical fragment estimation requires positive-length overlap")
        cliques = layout.cliques
        partial = np.where(layout.declared_mask, statistics.partial_kernel, np.nan)
        coherent = None
        if method == "coherent":
            support_rtol = options.pop("support_rtol", 1e-12)
            if options.get("regularization", 0) <= 0:
                raise ValueError("coherent fitting requires positive regularization")
            if "numerical_error_budget" not in options:
                raise ValueError("coherent fitting requires an explicit numerical_error_budget")
            coherent = fit_coherent_partial(
                partial,
                pair_weights=statistics.pair_weight_sums,
                cliques=cliques,
                weights=layout.weights,
                **options,
            )
            if not coherent.converged:
                raise CoherentConvergenceError(coherent)
            partial = coherent.partial_kernel
            options = {"population": True, "support_rtol": support_rtol}
        completion = complete_kernel(partial, cliques=cliques, weights=layout.weights, **options)
        if completion.kernel is None:
            raise ValueError(
                f"canonical completion unavailable: {completion.diagnostics['reason']}"
            )
        kernel = completion.kernel
        diagnostics = {
            "layout_validation": validation,
            "completion": completion.diagnostics,
            "pair_counts": statistics.pair_counts,
            "weighted_coverage": statistics.weighted_counts,
            "canonical_assumption": "conditional independence of separated fragments given overlaps",
            "certificate_scope": "no H-operator optimization certificate",
        }
        if coherent is not None:
            diagnostics["coherent_partial"] = coherent.diagnostics
            diagnostics["certificate_scope"] = (
                "weighted observed-partial optimizer distance only; excludes completion amplification"
            )
        if repair:
            projection = weighted_psd_projection(
                kernel, weights=layout.weights, observed_mask=layout.declared_mask
            )
            projected = projection.kernel
            diagnostics["weighted_psd_projection"] = projection.diagnostics
        estimand = "finite-grid canonical completion of the observed common innovation kernel"
        if coherent is not None:
            estimand = (
                "canonical completion of a regularized compatible PSD partial innovation kernel"
            )
    else:
        raise ValueError("method must be ridge, canonical or coherent")
    return KernelCovarianceFit(
        kernel,
        layout.nodes,
        layout.weights,
        layout.grid_id,
        method,
        estimand,
        diagnostics,
        metadata,
        operator_fit,
        projected,
    )
