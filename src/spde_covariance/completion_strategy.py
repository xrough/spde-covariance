"""Observed-data-only partial-kernel strategies and declared stability guards.

This layer composes existing moment, coherent-fitting and completion routines.
It neither converts grid kernels to H operators nor silently repairs a completed
kernel. The coherent certificate concerns the observed partial optimization;
split stability is a training diagnostic, not a completed-kernel error bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np

from .coherent import fit_coherent_partial
from .completion import _cover, _snapshot, _weights, complete_kernel
from .fragments import _row_indices
from .observation_statistics import (
    PartialKernelStatistics,
    kernel_quadratic_score_from_statistics,
    partial_kernel_from_arrays,
)


def _nonnegative(value, name, *, positive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class CompletionStrategySettings:
    """One predeclared partial estimator, floor policy and completion rule.

    Actual lambda is relative_regularization times the largest normalized date
    pair coverage. The resolution floor is gamma/Q**2 in sqrt(W) K sqrt(W)
    coordinates. A positive floor is an explicit finite-grid restriction.
    """

    partial_method: str = "pairwise"
    relative_regularization: float = 0.0
    completion_rule: str = "support"
    floor_policy: str = "zero"
    floor_gamma: float = 1e-5 * 46**2
    constant_floor: float = 1e-5
    numerical_error_budget: float = 1e-4
    rho: float = 0.2
    max_iterations: int = 100000
    support_rtol: float = 1e-12

    def __post_init__(self):
        if self.partial_method not in {"pairwise", "ridge", "coherent"}:
            raise ValueError("partial_method must be pairwise, ridge or coherent")
        if self.completion_rule not in {"support", "rank2"}:
            raise ValueError("completion_rule must be support or rank2")
        if self.floor_policy not in {"zero", "resolution", "constant"}:
            raise ValueError("floor_policy must be zero, resolution or constant")
        for name in ("relative_regularization", "floor_gamma", "constant_floor"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in ("numerical_error_budget", "rho"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name, positive=True))
        object.__setattr__(
            self, "max_iterations", _positive_integer(self.max_iterations, "max_iterations")
        )
        value = _nonnegative(self.support_rtol, "support_rtol", positive=True)
        if value >= 1:
            raise ValueError("support_rtol must be less than one")
        object.__setattr__(self, "support_rtol", value)
        if self.partial_method == "pairwise" and self.relative_regularization != 0:
            raise ValueError("pairwise uses no regularization; use partial_method='ridge'")
        if self.partial_method != "coherent" and self.floor_policy != "zero":
            raise ValueError("a weighted PSD eigenvalue floor is only defined for coherent fitting")

    def eigenvalue_floor(self, size):
        """Return the declared weighted-coordinate floor without changing units."""
        size = _positive_integer(size, "size")
        if self.floor_policy == "resolution":
            return self.floor_gamma / size**2
        if self.floor_policy == "constant":
            return self.constant_floor
        return 0.0


@dataclass(frozen=True, slots=True)
class CompletionStrategyResult:
    available: bool
    reason: str | None
    partial_kernel: np.ndarray
    kernel: np.ndarray | None
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("partial_kernel", "kernel", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))

    @property
    def converged(self):
        return bool(self.diagnostics["converged"])


@dataclass(frozen=True, slots=True)
class FragmentSplit:
    available: bool
    reason: str | None
    first_indices: np.ndarray
    second_indices: np.ndarray
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("first_indices", "second_indices", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class CompletionStabilityResult:
    available: bool
    reason: str | None
    split: FragmentSplit
    first_fit: CompletionStrategyResult | None
    second_fit: CompletionStrategyResult | None
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", _snapshot(self.diagnostics))


@dataclass(frozen=True, slots=True)
class CompletionStabilityGuard:
    """Optional, predeclared admissibility constraints; no automatic tuning.

    A missing diagnostic cannot pass its requested threshold. With no guard,
    observed validation alone selects among available fits; stability remains
    visible in the audit even when it is unavailable.
    """

    max_relative_difference: float | None = None
    max_inverse_norm: float | None = None
    max_condition_number: float | None = None
    require_available: bool = True

    def __post_init__(self):
        for name in ("max_relative_difference", "max_inverse_norm", "max_condition_number"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if not isinstance(self.require_available, (bool, np.bool_)):
            raise ValueError("require_available must be boolean")


@dataclass(frozen=True, slots=True)
class CompletionSelection:
    available: bool
    reason: str | None
    selected_name: str | None
    selected: CompletionStrategyResult | None
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", _snapshot(self.diagnostics))


def _statistics_geometry(statistics, cliques, weights):
    if not isinstance(statistics, PartialKernelStatistics):
        raise TypeError("statistics must be PartialKernelStatistics from observed training dates")
    target = np.asarray(statistics.partial_kernel)
    if target.ndim != 2 or target.shape[0] != target.shape[1] or not len(target):
        raise ValueError("partial statistics must contain a square kernel")
    groups, mask = _cover(cliques, len(target))
    quadrature_weights = _weights(weights, len(target))
    coverage = np.asarray(statistics.pair_weight_sums)
    measured = np.asarray(statistics.observed_mask, dtype=bool)
    if coverage.shape != target.shape or measured.shape != target.shape:
        raise ValueError("statistics coverage must match the kernel")
    if (
        not np.all(np.isfinite(target))
        or not np.all(np.isfinite(coverage))
        or np.any(coverage < 0)
        or not np.array_equal(measured, coverage > 0)
        or np.max(np.abs(target - target.T)) > 1e-12 * np.max(np.abs(target))
        or not np.array_equal(coverage, coverage.T)
    ):
        raise ValueError("statistics must contain finite symmetric moments and pair coverage")
    if np.any(measured & ~mask):
        raise ValueError("training observations include pairs outside the declared clique cover")
    partial = np.where(measured, (target + target.T) / 2, np.nan)
    return partial, coverage, groups, mask, quadrature_weights


def _conditioning(diagnostics):
    joins = [join for join in diagnostics.get("joins", ()) if not join.get("skipped", False)]
    conditions = [join.get("retained_condition_number") for join in joins]
    return {
        "conditioning_available": all(
            value is not None and np.isfinite(value) for value in conditions
        ),
        "maximum_inverse_norm": max((join.get("inverse_norm", 0.0) for join in joins), default=0.0),
        "maximum_retained_condition_number": max(conditions, default=1.0)
        if all(value is not None and np.isfinite(value) for value in conditions)
        else None,
        "maximum_inverse_perturbation_multiplier": max(
            (join.get("inverse_perturbation_multiplier", 0.0) for join in joins), default=0.0
        ),
        "separator_count": len(joins),
        "conditioning_scope": "retained weighted separator spectra; not an error certificate",
    }


def fit_completion_strategy(statistics, *, cliques, weights, settings):
    """Fit and complete a grid kernel using observed training moments only.

    Missing coverage, failed optimizer certificates and unavailable completion
    ranks return a flagged result. Malformed inputs raise. Coherent numerical
    support completion invokes the existing population compatibility checks;
    this flag checks finite-grid PSD/support, not that data are population truth.
    """
    if not isinstance(settings, CompletionStrategySettings):
        raise TypeError("settings must be CompletionStrategySettings")
    partial, coverage, groups, mask, weights = _statistics_geometry(statistics, cliques, weights)
    size = len(partial)
    floor = settings.eigenvalue_floor(size)
    regularization_scale = float(np.max(coverage))
    regularization = settings.relative_regularization * regularization_scale
    diagnostics = {
        "settings": asdict(settings),
        "n_training_dates": statistics.n_observations,
        "observed_pair_count": int(np.sum(mask)),
        "missing_training_pair_count": int(np.sum(mask & ~statistics.observed_mask)),
        "relative_regularization": settings.relative_regularization,
        "regularization_scale": regularization_scale,
        "regularization": regularization,
        "eigenvalue_floor": floor,
        "weighted_floor_diagonal_trace": size * floor,
        "floor_coordinate_convention": "sqrt(W) K sqrt(W); not orthonormal H coordinates",
        "floor_units": "kernel covariance units times maturity quadrature units",
        "continuum_convergence_claim": False,
        "target": "completed finite-grid kernel; no H/source covariance identification claim",
        "converged": False,
        "objective_gap_bound": None,
        "partial_error_bound": None,
        "partial_fit": None,
        "completion": None,
        "observed_modification": None,
        "conditioning_available": False,
        "maximum_inverse_norm": None,
        "maximum_retained_condition_number": None,
        "maximum_inverse_perturbation_multiplier": None,
    }
    if diagnostics["missing_training_pair_count"]:
        return CompletionStrategyResult(
            False, "training does not cover all declared clique pairs", partial, None, diagnostics
        )
    original = partial.copy()
    if settings.partial_method == "coherent":
        fit = fit_coherent_partial(
            partial,
            pair_weights=coverage,
            cliques=groups,
            weights=weights,
            regularization=regularization,
            eigenvalue_floor=floor,
            numerical_error_budget=settings.numerical_error_budget,
            rho=settings.rho,
            max_iterations=settings.max_iterations,
        )
        partial = fit.partial_kernel
        certified = (
            fit.converged
            and np.isfinite(fit.objective_gap_bound)
            and fit.objective_gap_bound >= 0
            and np.isfinite(fit.partial_error_bound)
            and 0 <= fit.partial_error_bound <= settings.numerical_error_budget
        )
        diagnostics.update(
            partial_fit=fit.diagnostics,
            converged=bool(certified),
            objective_gap_bound=fit.objective_gap_bound,
            partial_error_bound=fit.partial_error_bound,
        )
    else:
        partial = original.copy()
        if settings.partial_method == "ridge":
            partial[mask] *= coverage[mask] / (coverage[mask] + regularization)
        diagnostics.update(
            converged=True,
            objective_gap_bound=0.0,
            partial_error_bound=0.0,
            partial_fit={
                "solver": "entrywise closed form",
                "certificate_arithmetic": "floating point",
            },
        )
    root = np.sqrt(weights)
    difference = np.where(mask, partial - original, 0.0)
    diagnostics["observed_modification"] = {
        "weighted_hs": float(np.linalg.norm(root[:, None] * difference * root[None, :])),
        "unweighted_frobenius": float(np.linalg.norm(difference)),
        "maximum_absolute": float(np.max(np.abs(difference))),
        "weighted_diagonal_trace_change": float(np.sum(weights * np.diag(difference))),
    }
    if not diagnostics["converged"]:
        return CompletionStrategyResult(
            False,
            "coherent observed-partial certificate exceeds budget",
            partial,
            None,
            diagnostics,
        )
    completion = complete_kernel(
        partial,
        cliques=groups,
        weights=weights,
        rank=2 if settings.completion_rule == "rank2" else None,
        support_rtol=settings.support_rtol,
        population=settings.partial_method == "coherent" and settings.completion_rule == "support",
    )
    diagnostics["completion"] = completion.diagnostics
    diagnostics.update(_conditioning(completion.diagnostics))
    return CompletionStrategyResult(
        completion.available,
        completion.diagnostics["reason"],
        partial,
        completion.kernel,
        diagnostics,
    )


def balanced_fragment_split(row_to_nodes, *, size, seed=0):
    """Complementary training positions, balanced within each fragment support.

    Rows with different permutations of the same support belong to one group.
    The split uses support labels and a seed, never values, time beyond the
    supplied training sequence, scenario labels or latent covariance.
    """
    size = _positive_integer(size, "size")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rows = _row_indices(row_to_nodes, size=size)
    groups = {}
    for position, row in enumerate(rows):
        groups.setdefault(tuple(sorted(row)), []).append(position)
    rng = np.random.default_rng(seed)
    first, second, audit = [], [], []
    extra_first = bool(rng.integers(2))
    for support in sorted(groups):
        positions = np.asarray(groups[support], dtype=int)
        shuffled = rng.permutation(positions)
        take = len(positions) // 2
        if len(positions) % 2:
            take += int(extra_first)
            extra_first = not extra_first
        first.extend(shuffled[:take])
        second.extend(shuffled[take:])
        audit.append(
            {
                "nodes": support,
                "count": len(positions),
                "first_count": take,
                "second_count": len(positions) - take,
            }
        )
    available = all(item["first_count"] > 0 and item["second_count"] > 0 for item in audit)
    return FragmentSplit(
        available,
        None if available else "a fragment support occurs fewer than twice in training",
        np.array(sorted(first), dtype=int),
        np.array(sorted(second), dtype=int),
        {
            "seed": int(seed),
            "n_training_dates": len(rows),
            "groups": audit,
            "partition_scope": "supplied training positions only",
        },
    )


def split_completion_stability(
    values,
    row_to_nodes,
    *,
    size,
    cliques,
    weights,
    settings,
    noise_variance=0.0,
    seed=0,
    date_weights=None,
):
    """Fit complementary balanced training halves; report completion sensitivity.

    Noise variance is the known variance of fresh independent point noise, as
    in partial_kernel_from_arrays. Date weights, if supplied, are restricted to
    each half before its moment summary and relative penalty are recomputed.
    """
    values = tuple(values)
    rows = _row_indices(row_to_nodes, size=size, count=len(values))
    # Validate observations/noise/date weights even when a split is unavailable.
    partial_kernel_from_arrays(values, rows, noise_variance, date_weights, size)
    split = balanced_fragment_split(rows, size=size, seed=seed)
    diagnostics = {
        "relative_difference": None,
        "observed_relative_difference": None,
        "missing_relative_difference": None,
        "scope": "training split sensitivity; not a sampling or completed-kernel error certificate",
    }
    if not split.available:
        return CompletionStabilityResult(False, split.reason, split, None, None, diagnostics)
    fits = []
    raw_weights = None if date_weights is None else np.asarray(date_weights)
    for positions in (split.first_indices, split.second_indices):
        if raw_weights is not None and not np.any(raw_weights[positions] > 0):
            return CompletionStabilityResult(
                False,
                "one training half has no positive date weight",
                split,
                None,
                None,
                diagnostics,
            )
        statistics = partial_kernel_from_arrays(
            tuple(values[index] for index in positions),
            tuple(rows[index] for index in positions),
            noise_variance,
            None if raw_weights is None else raw_weights[positions],
            size,
        )
        fits.append(
            fit_completion_strategy(statistics, cliques=cliques, weights=weights, settings=settings)
        )
    first, second = fits
    if not first.available or not second.available:
        diagnostics["half_availability"] = (first.available, second.available)
        return CompletionStabilityResult(
            False, "a training-half fit is unavailable", split, first, second, diagnostics
        )
    _, mask = _cover(cliques, size)
    root = np.sqrt(_weights(weights, size))
    a = root[:, None] * first.kernel * root[None, :]
    b = root[:, None] * second.kernel * root[None, :]
    for prefix, region in (("", np.ones_like(mask)), ("observed_", mask), ("missing_", ~mask)):
        absolute = float(np.linalg.norm((a - b)[region]))
        scale = max(float(np.linalg.norm(a[region])), float(np.linalg.norm(b[region])))
        diagnostics[prefix + "absolute_difference"] = absolute
        diagnostics[prefix + "relative_difference"] = absolute / scale if scale else 0.0
        diagnostics[prefix + "normalization"] = scale
    diagnostics["half_availability"] = (True, True)
    return CompletionStabilityResult(True, None, split, first, second, diagnostics)


def _guard_failures(fit, stability, guard):
    if guard is None:
        return []
    reasons = []
    if guard.require_available and (stability is None or not stability.available):
        reasons.append("training split stability unavailable")
    checks = (
        (
            guard.max_relative_difference,
            None if stability is None else stability.diagnostics.get("relative_difference"),
            "training split relative difference",
        ),
        (
            guard.max_inverse_norm,
            fit.diagnostics.get("maximum_inverse_norm"),
            "separator inverse norm",
        ),
        (
            guard.max_condition_number,
            fit.diagnostics.get("maximum_retained_condition_number"),
            "separator condition number",
        ),
    )
    for threshold, value, label in checks:
        if threshold is not None:
            if value is None or not np.isfinite(value):
                reasons.append(label + " unavailable")
            elif value > threshold:
                reasons.append(label + " exceeds declared threshold")
    return reasons


def select_completion_strategy(candidates, validation_statistics, *, stability=None, guard=None):
    """Choose by corrected observed validation score after a declared guard.

    Exact score ties retain candidate mapping insertion order. No test data,
    latent truth or scenario parameters enter this interface. Every failed,
    rejected and eligible candidate remains in the returned audit.
    """
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a nonempty ordered mapping of names to fits")
    if guard is not None and not isinstance(guard, CompletionStabilityGuard):
        raise TypeError("guard must be CompletionStabilityGuard or None")
    if stability is not None and not isinstance(stability, Mapping):
        raise TypeError("stability must be a mapping of candidate names to split diagnostics")
    if stability is not None and set(stability) - set(candidates):
        raise ValueError("stability contains unknown candidate names")
    if not isinstance(validation_statistics, PartialKernelStatistics):
        raise TypeError("validation_statistics must be PartialKernelStatistics")
    audit, winner, best_score = [], None, np.inf
    for name, fit in candidates.items():
        if not isinstance(name, str) or not name or not isinstance(fit, CompletionStrategyResult):
            raise ValueError("each candidate must have a nonempty name and strategy result")
        split = None if stability is None else stability.get(name)
        if split is not None and not isinstance(split, CompletionStabilityResult):
            raise TypeError("stability entries must be CompletionStabilityResult")
        reasons = [] if fit.available else [fit.reason or "fit unavailable"]
        reasons.extend(_guard_failures(fit, split, guard))
        score = (
            kernel_quadratic_score_from_statistics(fit.kernel, validation_statistics)
            if fit.available
            else None
        )
        audit.append(
            {
                "name": name,
                "fit_available": fit.available,
                "fit_reason": fit.reason,
                "converged": fit.converged,
                "stability_available": None if split is None else split.available,
                "stability_reason": None if split is None else split.reason,
                "validation_score": score,
                "admissible": not reasons,
                "rejection_reasons": reasons,
            }
        )
        if not reasons and score < best_score:
            winner, best_score = name, score
    return CompletionSelection(
        winner is not None,
        None
        if winner is not None
        else "no candidate satisfies numerical availability and the declared guard",
        winner,
        None if winner is None else candidates[winner],
        {
            "candidates": audit,
            "guard": None if guard is None else asdict(guard),
            "tie_rule": "first in candidate mapping order for exactly equal observed scores",
            "selection_scope": "observed validation score and declared training-only diagnostics",
            "validation_count": validation_statistics.n_observations,
            "selected_score": None if winner is None else best_score,
        },
    )
