"""Finite Gaussian Markov completion with known observation noise.

The observed-data objective is generally nonconvex. EM uses an exact chordal
Gaussian M-step, and reports monotonicity and local stationarity, never a global
optimization certificate. Kernel matrices use physical grid-value coordinates;
maturity quadrature is not a likelihood weight.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .completion import _snapshot

_MAX_GROUPS = 1024
_MAX_GROUP_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True)
class GaussianCompletionFit:
    kernel: np.ndarray | None
    precision: np.ndarray | None
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("kernel", "precision", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))

    @property
    def converged(self):
        return bool(self.diagnostics["converged"])


@dataclass(frozen=True, slots=True)
class GaussianConditional:
    mean: np.ndarray
    covariance: np.ndarray
    target_nodes: np.ndarray
    diagnostics: Mapping

    def __post_init__(self):
        for name in ("mean", "covariance", "target_nodes", "diagnostics"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))


def _scalar(value, name, *, positive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(raw, dtype=float, copy=True)


def _matrix(value, name, *, positive=False):
    matrix = _array(value, name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError(f"{name} must be a nonempty square matrix")
    scale = float(np.max(np.abs(matrix)))
    if np.max(np.abs(matrix - matrix.T)) > 1e-12 * scale:
        raise ValueError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) / 2
    if positive:
        _chol(matrix, name)
    elif np.linalg.eigvalsh(matrix)[0] < -1e-12 * len(matrix) * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _chol(matrix, name):
    try:
        return cho_factor(matrix, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"{name} is not numerically positive definite; no jitter was added"
        ) from error


def _inverse(matrix, name):
    result = cho_solve(_chol(matrix, name), np.eye(len(matrix)), check_finite=False)
    return (result + result.T) / 2


def _nodes(value, size, *, empty=False):
    nodes = np.asarray(value)
    if empty and nodes.ndim == 1 and not len(nodes):
        return np.array([], dtype=int)
    if (
        nodes.ndim != 1
        or nodes.dtype.kind not in "iu"
        or (not len(nodes) and not empty)
        or np.any(nodes < 0)
        or np.any(nodes >= size)
        or len(np.unique(nodes)) != len(nodes)
    ):
        raise ValueError("node indices must be unique integers inside the kernel grid")
    return np.array(nodes, dtype=int, copy=True)


def _graph(cliques, size):
    try:
        declared = tuple(tuple(sorted(_nodes(group, size))) for group in cliques)
    except TypeError as error:
        raise ValueError("cliques must be an ordered cover") from error
    if not declared or set().union(*(set(group) for group in declared)) != set(range(size)):
        raise ValueError("cliques must cover every kernel node")
    unique = list(dict.fromkeys(declared))
    maximal = [group for group in unique if not any(set(group) < set(other) for other in unique)]
    covered, previous, separators = set(), [], []
    mask = np.zeros((size, size), dtype=bool)
    for group in maximal:
        separator = covered.intersection(group)
        if separator and not any(separator.issubset(old) for old in previous):
            raise ValueError("maximal cliques must have an ordered running-intersection property")
        separators.append(np.array(sorted(separator), dtype=int))
        mask[np.ix_(group, group)] = True
        previous.append(set(group))
        covered.update(group)
    return tuple(np.array(group, dtype=int) for group in maximal), tuple(separators), mask, declared


def _noise(value, size):
    if np.ndim(value) == 0:
        scalar = value.item() if isinstance(value, np.ndarray) else value
        return np.eye(size) * _scalar(scalar, "noise variance")
    noise = _matrix(value, "noise covariance")
    if noise.shape != (size, size):
        raise ValueError("noise covariance must match its observed row count")
    return noise


def _noises(value, rows):
    if isinstance(value, Real) or (isinstance(value, np.ndarray) and value.ndim == 0):
        common = {size: _noise(value, size) for size in set(map(len, rows))}
        return tuple(common[len(row)] for row in rows)
    try:
        raw = np.asarray(value)
    except ValueError:
        raw = None
    if (
        raw is not None
        and raw.ndim == 2
        and raw.dtype.kind in "iuf"
        and raw.shape[0] == raw.shape[1]
        and all(len(row) == raw.shape[0] for row in rows)
    ):
        common = _noise(raw, raw.shape[0])
        return (common,) * len(rows)
    try:
        sequence = tuple(value)
    except TypeError as error:
        raise ValueError("noise_covariances must be common or one per observation") from error
    if len(sequence) != len(rows):
        raise ValueError("one noise covariance is required per observation")
    return tuple(_noise(item, len(row)) for item, row in zip(sequence, rows, strict=True))


@dataclass(slots=True)
class _Data:
    values: tuple
    rows: tuple
    noises: tuple
    weights: np.ndarray
    groups: tuple | None
    diagnostics: dict

    def iter_groups(self):
        if self.groups is not None:
            yield from self.groups
        else:
            for y, row, noise, weight in zip(
                self.values, self.rows, self.noises, self.weights, strict=True
            ):
                if weight:
                    yield row, noise, float(weight), weight * np.outer(y, y)


def _data(values, row_to_nodes, noise_covariances, weights, size):
    samples = tuple(_array(value, "observed values") for value in values)
    rows = tuple(_nodes(row, size) for row in row_to_nodes)
    if (
        not samples
        or len(samples) != len(rows)
        or any(y.shape != (len(row),) for y, row in zip(samples, rows, strict=True))
    ):
        raise ValueError("each observed vector must match one nonempty row map")
    noises = _noises(noise_covariances, rows)
    date_weights = np.ones(len(samples)) if weights is None else _array(weights, "date weights")
    if (
        date_weights.shape != (len(samples),)
        or np.any(date_weights < 0)
        or not np.any(date_weights > 0)
    ):
        raise ValueError("date weights must be nonnegative, nonzero and match observations")
    date_weights /= date_weights.max()
    date_weights /= date_weights.sum()
    grouped, used_bytes, fallback = {}, 0, False
    for y, row, noise, weight in zip(samples, rows, noises, date_weights, strict=True):
        if not weight:
            continue
        key = (tuple(row), noise.tobytes())
        if key not in grouped:
            # Raw buffers, byte key, and conservative Python key overhead.
            needed = 64 * len(row) + 3 * noise.nbytes + 512
            if len(grouped) >= _MAX_GROUPS or used_bytes + needed > _MAX_GROUP_BYTES:
                fallback = True
                break
            grouped[key] = [row, noise, 0.0, np.zeros_like(noise)]
            used_bytes += needed
        group = grouped[key]
        group[2] += weight
        with np.errstate(over="raise", invalid="raise"):
            try:
                group[3] += weight * np.outer(y, y)
            except FloatingPointError as error:
                raise ValueError("raw observed moments overflowed; rescale observations") from error
    return _Data(
        samples,
        rows,
        noises,
        date_weights,
        None if fallback else tuple(tuple(group) for group in grouped.values()),
        {
            "n_observations": len(samples),
            "positive_weight_observations": int(np.sum(date_weights > 0)),
            "grouping_mode": "streaming observations"
            if fallback
            else "row-and-noise grouped raw moments",
            "cached_groups": 0 if fallback else len(grouped),
            "group_cache_bytes": 0 if fallback else used_bytes,
            "group_cache_limit_bytes": _MAX_GROUP_BYTES,
            "group_cache_limit_count": _MAX_GROUPS,
            "noise_grouping": "exact row order and known noise covariance",
        },
    )


def _posterior(kernel, row, noise):
    sigma = kernel[np.ix_(row, row)] + noise
    factor = _chol(sigma, "predicted observed covariance")
    gain = cho_solve(factor, kernel[row], check_finite=False).T
    residual = np.eye(len(kernel))
    residual[:, row] -= gain
    # Joseph form remains valid with zero or rank-deficient known noise.
    posterior = residual @ kernel @ residual.T + gain @ noise @ gain.T
    return gain, (posterior + posterior.T) / 2, factor


def _likelihood(kernel, data, *, posterior=False):
    value, expected = 0.0, np.zeros_like(kernel) if posterior else None
    for row, noise, weight, second in data.iter_groups():
        if posterior:
            gain, conditional, factor = _posterior(kernel, row, noise)
            expected += weight * conditional + gain @ second @ gain.T
        else:
            factor = _chol(kernel[np.ix_(row, row)] + noise, "predicted observed covariance")
        value += 0.5 * (
            weight * (2 * np.log(np.diag(factor[0])).sum() + len(row) * np.log(2 * np.pi))
            + np.trace(cho_solve(factor, second, check_finite=False))
        )
    if not np.isfinite(value) or (posterior and not np.all(np.isfinite(expected))):
        raise ValueError("Gaussian likelihood or conditional moments exceeded floating-point range")
    return float(value), None if expected is None else (expected + expected.T) / 2


def gaussian_kernel_score(kernel, values, row_to_nodes, *, noise_covariances, weights=None):
    """Mean raw-data Gaussian NLL, including normalizing constants.

    A PSD latent kernel is permitted when each positively weighted observed
    covariance K_II+N is SPD. Corrected outer products are not likelihood data.
    """
    matrix = _matrix(kernel, "kernel")
    data = _data(values, row_to_nodes, noise_covariances, weights, len(matrix))
    return _likelihood(matrix, data)[0]


def condition_gaussian_kernel(
    kernel, values, observed_nodes, *, noise_covariance, target_nodes=None
):
    """Conditional latent Gaussian mean/covariance; no parameter-uncertainty CI.

    By default return all unobserved nodes. Explicit targets may include observed
    nodes. Joseph covariance conditioning supports known singular/zero noise as
    long as the observed covariance is SPD; no inverse or PSD repair is hidden.
    """
    matrix = _matrix(kernel, "kernel")
    observed = _nodes(observed_nodes, len(matrix), empty=True)
    value = _array(values, "observed values")
    if value.shape != (len(observed),):
        raise ValueError("values must match observed_nodes in the supplied order")
    target = (
        np.array(sorted(set(range(len(matrix))) - set(observed)), dtype=int)
        if target_nodes is None
        else _nodes(target_nodes, len(matrix), empty=True)
    )
    if len(observed) or np.ndim(noise_covariance) == 0:
        noise = _noise(noise_covariance, len(observed))
    else:
        noise = _array(noise_covariance, "noise covariance")
        if noise.shape != (0, 0):
            raise ValueError("noise covariance must match zero observed rows")
    if len(observed):
        gain, posterior, factor = _posterior(matrix, observed, noise)
        mean = gain @ value
        condition = float(np.linalg.cond(matrix[np.ix_(observed, observed)] + noise))
        logdet = float(2 * np.log(np.diag(factor[0])).sum())
    else:
        mean, posterior, condition, logdet = np.zeros(len(matrix)), matrix.copy(), None, 0.0
    covariance = posterior[np.ix_(target, target)]
    minimum = float(np.linalg.eigvalsh(covariance)[0]) if len(target) else 0.0
    tolerance = (
        100
        * np.finfo(float).eps
        * max(len(matrix), 1)
        * max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    )
    if minimum < -tolerance:
        raise ValueError("Joseph conditional covariance failed its PSD roundoff check")
    return GaussianConditional(
        mean[target],
        covariance,
        target,
        {
            "observed_nodes": observed,
            "posterior_minimum_eigenvalue": minimum,
            "posterior_psd_tolerance": tolerance,
            "observed_covariance_condition": condition,
            "observed_covariance_logdet": logdet,
            "conditioning_method": "Joseph form; Cholesky observed solve",
            "scope": "latent conditional distribution at a fixed fitted kernel; excludes parameter uncertainty",
            "silent_psd_repair": False,
        },
    )


def _m_step(second, cliques, separators):
    """Exact chordal M-step, constructed from conditional Cholesky factors."""
    size = len(second)
    regression, residual_root = np.eye(size), np.zeros((size, size))
    covered = set()
    for clique, separator in zip(cliques, separators, strict=True):
        new = np.array(sorted(set(clique) - covered), dtype=int)
        ordering = np.r_[separator, new]
        block = second[np.ix_(ordering, ordering)]
        factor = np.linalg.cholesky(block)
        count = len(separator)
        if count:
            beta = cho_solve(
                (factor[:count, :count], True), second[np.ix_(separator, new)], check_finite=False
            ).T
            regression[np.ix_(new, separator)] = -beta
        residual_root[np.ix_(new, new)] = factor[count:, count:]
        covered.update(clique)
    covariance_factor = np.linalg.solve(regression, residual_root)
    precision_factor = np.linalg.solve(residual_root, regression)
    kernel = covariance_factor @ covariance_factor.T
    precision = precision_factor.T @ precision_factor
    _chol(kernel, "M-step kernel")
    _chol(precision, "M-step precision")
    return (kernel + kernel.T) / 2, (precision + precision.T) / 2


def _evaluate(kernel, precision, data, reference, regularization, mask):
    nll, expected = _likelihood(kernel, data, posterior=True)
    logdet_q = 2 * np.log(np.diag(_chol(precision, "precision")[0])).sum()
    logdet_t = 2 * np.log(np.diag(_chol(reference, "reference kernel")[0])).sum()
    penalty = (
        regularization / 2 * (np.sum(reference * precision) - logdet_q - logdet_t - len(kernel))
    )
    gradient = (expected + regularization * reference - (1 + regularization) * kernel) / 2
    free_gradient = np.where(mask, gradient, 0.0)
    scale = np.sqrt(np.diag(reference))
    relative = float(
        np.linalg.norm((free_gradient / scale[:, None] / scale[None, :])[mask])
        / np.sqrt(np.sum(mask))
    )
    if (
        not np.isfinite(nll + penalty)
        or not np.isfinite(relative)
        or not np.all(np.isfinite(gradient))
    ):
        raise ValueError("penalized Gaussian objective or gradient exceeded floating-point range")
    return {
        "objective": float(nll + penalty),
        "negative_loglikelihood": nll,
        "penalty": float(penalty),
        "expected_second_moment": expected,
        "gradient": gradient,
        "stationarity_norm": float(np.linalg.norm(free_gradient)),
        "relative_stationarity": relative,
    }


def fit_gaussian_completion(
    values,
    row_to_nodes,
    *,
    noise_covariances,
    cliques,
    reference_kernel,
    regularization,
    weights=None,
    max_iterations=2000,
    stationarity_tolerance=1e-6,
    relative_tolerance=1e-8,
    initial_kernels=None,
):
    """Penalized observed Gaussian likelihood under a fixed chordal graph.

    The mean is declared zero. Regularization is positive, date weights sum to
    one, and the penalty is lambda/2*(tr(TQ)-logdet(TQ)-d). The exact M-step uses
    S_eff=(E[xxT|observations]+lambda*T)/(1+lambda). Positive T and lambda make
    each M-step SPD without eigenvalue floors. The observed objective remains
    nonconvex: convergence means local free-precision stationarity plus a small
    relative objective change. All starts/failures remain in the audit.
    """
    reference = _matrix(reference_kernel, "reference kernel", positive=True)
    regularization = _scalar(regularization, "regularization", positive=True)
    max_iterations = _integer(max_iterations, "max_iterations")
    stationarity_tolerance = _scalar(
        stationarity_tolerance, "stationarity_tolerance", positive=True
    )
    relative_tolerance = _scalar(relative_tolerance, "relative_tolerance", positive=True)
    groups, separators, mask, declared = _graph(cliques, len(reference))
    data = _data(values, row_to_nodes, noise_covariances, weights, len(reference))
    if initial_kernels is None:
        prior_kernel, _ = _m_step(reference, groups, separators)
        adjacency = mask.astype(float) - np.eye(len(reference))
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        scale = np.sqrt(np.diag(reference))
        correlated = (
            scale[:, None]
            * _inverse(np.eye(len(reference)) + 0.25 * laplacian, "graph-correlated initializer")
            * scale[None, :]
        )
        starts = (
            ("graph-prior", prior_kernel),
            ("scaled-graph-prior", 2 * prior_kernel),
            ("graph-correlated-prior", correlated),
        )
    elif isinstance(initial_kernels, Mapping):
        starts = tuple(initial_kernels.items())
    else:
        try:
            array = np.asarray(initial_kernels)
            values = (initial_kernels,) if array.ndim == 2 else tuple(initial_kernels)
        except (TypeError, ValueError) as error:
            raise ValueError("initial_kernels must contain square kernel matrices") from error
        starts = tuple((f"initial-{i}", value) for i, value in enumerate(values))
    if not starts:
        raise ValueError("at least one initial kernel is required")
    attempts, candidates = [], []
    for name, initial in starts:
        started = time.perf_counter()
        history, nll_history, penalty_history = [], [], []
        state, kernel, precision = None, None, None
        status, converged, iterations, change = "initialization_failed", False, 0, None
        allowance, rejected, initial_digest = 0.0, None, None
        try:
            kernel = _matrix(initial, "initial kernel", positive=True)
            initial_digest = hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest()
            if kernel.shape != reference.shape:
                raise ValueError("initial kernel must match reference dimensions")
            precision = _inverse(kernel, "initial kernel")
            outside = float(np.max(np.abs(precision[~mask]))) if np.any(~mask) else 0.0
            if outside > 1e-10 * np.max(np.abs(precision)):
                raise ValueError("initial precision violates the declared graph")
            state = _evaluate(kernel, precision, data, reference, regularization, mask)
            history.append(state["objective"])
            nll_history.append(state["negative_loglikelihood"])
            penalty_history.append(state["penalty"])
            status = "maximum_iterations"
            if state["relative_stationarity"] <= stationarity_tolerance:
                converged, status = True, "local_stationarity"
            else:
                for iteration in range(1, max_iterations + 1):
                    second = (state["expected_second_moment"] + regularization * reference) / (
                        1 + regularization
                    )
                    proposed_kernel, proposed_precision = _m_step(second, groups, separators)
                    proposed = _evaluate(
                        proposed_kernel, proposed_precision, data, reference, regularization, mask
                    )
                    allowance = (
                        256
                        * np.finfo(float).eps
                        * max(1.0, abs(state["objective"]), abs(proposed["objective"]))
                        * len(reference)
                    )
                    if proposed["objective"] > state["objective"] + allowance:
                        status = "objective_increase_rejected"
                        rejected = proposed["objective"]
                        break
                    change = abs(proposed["objective"] - state["objective"]) / max(
                        1.0, abs(state["objective"])
                    )
                    kernel, precision, state, iterations = (
                        proposed_kernel,
                        proposed_precision,
                        proposed,
                        iteration,
                    )
                    history.append(state["objective"])
                    nll_history.append(state["negative_loglikelihood"])
                    penalty_history.append(state["penalty"])
                    if (
                        state["relative_stationarity"] <= stationarity_tolerance
                        and change <= relative_tolerance
                    ):
                        converged, status = True, "local_stationarity"
                        break
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            status = "initialization_failed" if state is None else "numerical_failure"
            reason = str(error)
        else:
            reason = None if converged else status
        attempt = {
            "name": str(name),
            "initial_kernel_sha256": initial_digest,
            "converged": converged,
            "status": status,
            "reason": reason,
            "iterations": iterations,
            "objective": None if state is None else state["objective"],
            "negative_loglikelihood": None if state is None else state["negative_loglikelihood"],
            "penalty": None if state is None else state["penalty"],
            "stationarity_norm": None if state is None else state["stationarity_norm"],
            "relative_stationarity": None if state is None else state["relative_stationarity"],
            "relative_objective_change": change,
            "objective_history": history,
            "negative_loglikelihood_history": nll_history,
            "penalty_history": penalty_history,
            "maximum_objective_increase": max(np.diff(history), default=0.0),
            "monotonicity_roundoff_allowance": allowance,
            "rejected_proposed_objective": rejected,
            "runtime_seconds": time.perf_counter() - started,
        }
        attempts.append(attempt)
        if state is not None and np.isfinite(state["objective"]):
            candidates.append((state["objective"], len(attempts) - 1, kernel, precision, state))
    diagnostics = {
        **data.diagnostics,
        "converged": False,
        "status": "all_initializations_failed",
        "regularization": regularization,
        "effective_prior_fraction": regularization / (1 + regularization),
        "effective_prior_pseudocount_at_uniform_weights": regularization * len(data.values),
        "effective_date_count": float(1 / np.sum(data.weights**2)),
        "normalized_date_weights": data.weights,
        "reference_kernel": reference,
        "declared_cliques": declared,
        "cliques": groups,
        "separators": separators,
        "free_precision_mask": mask,
        "attempts": attempts,
        "distinct_initial_kernel_count": len(
            {
                attempt["initial_kernel_sha256"]
                for attempt in attempts
                if attempt["initial_kernel_sha256"] is not None
            }
        ),
        "selected_attempt": None,
        "iterations": 0,
        "objective": None,
        "negative_loglikelihood": None,
        "penalty": None,
        "stationarity_norm": None,
        "relative_stationarity": None,
        "objective_history": (),
        "negative_loglikelihood_history": (),
        "penalty_history": (),
        "max_iterations": max_iterations,
        "stationarity_tolerance": stationarity_tolerance,
        "relative_tolerance": relative_tolerance,
        "mean_convention": "known zero mean; no automatic centering",
        "coordinate_convention": "physical grid covariance; no maturity quadrature likelihood weights",
        "objective_convention": "mean observed Gaussian NLL + lambda/2*(tr(TQ)-logdet(TQ)-d)",
        "certificate_scope": "local stationarity only; no global optimizer or statistical error bound",
        "silent_psd_repair": False,
        "m_step": "full-support chordal conditional Cholesky factors; no eigenvalue truncation",
    }
    if not candidates:
        return GaussianCompletionFit(None, None, diagnostics)
    _, selected, kernel, precision, state = min(candidates, key=lambda item: (item[0], item[1]))
    attempt = attempts[selected]
    diagnostics.update(
        {
            **{key: value for key, value in state.items() if key != "expected_second_moment"},
            "converged": attempt["converged"],
            "status": attempt["status"],
            "iterations": attempt["iterations"],
            "objective_history": attempt["objective_history"],
            "negative_loglikelihood_history": attempt["negative_loglikelihood_history"],
            "penalty_history": attempt["penalty_history"],
            "maximum_objective_increase": attempt["maximum_objective_increase"],
            "monotonicity_roundoff_allowance": attempt["monotonicity_roundoff_allowance"],
            "rejected_proposed_objective": attempt["rejected_proposed_objective"],
            "selected_attempt": selected,
            "relative_objective_change": attempt["relative_objective_change"],
            "kernel_condition": float(np.linalg.cond(kernel)),
            "precision_condition": float(np.linalg.cond(precision)),
            "kernel_minimum_eigenvalue": float(np.linalg.eigvalsh(kernel)[0]),
            "precision_minimum_eigenvalue": float(np.linalg.eigvalsh(precision)[0]),
            "outside_graph_precision_max": float(np.max(np.abs(precision[~mask])))
            if np.any(~mask)
            else 0.0,
            "inverse_backward_residual": float(
                np.linalg.norm(kernel @ precision - np.eye(len(kernel)))
                / max(1.0, np.linalg.norm(kernel) * np.linalg.norm(precision))
            ),
        }
    )
    return GaussianCompletionFit(kernel, precision, diagnostics)
