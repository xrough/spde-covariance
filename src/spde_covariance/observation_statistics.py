"""Shared corrected observation moments, coverage and quadratic scoring.

Scores omit the same measurement-only constant as the existing ridge study.
Kernel predictions may be indefinite; their numerical validity and their PSD
status are distinct diagnostics. Grouped default-weight arithmetic deliberately
retains the legacy observation-score summation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .fragments import FragmentLayout, _observations, _real_values, _row_indices, _snapshot
from .linalg import validate_symmetric


def _weights(weights, count):
    if weights is None:
        return np.ones(count), float(count), True
    values = _real_values(weights, "weights")
    if values.shape != (count,) or np.any(values < 0) or not np.any(values > 0):
        raise ValueError("weights must be nonnegative, nonzero and match observations")
    values = values / np.max(values)
    return values, float(np.sum(values)), False


@dataclass(frozen=True, slots=True)
class ObservationGroup:
    """One repeated map; corrected_sum uses raw scaled date weights."""

    operator: np.ndarray
    count: int
    weight_sum: float
    corrected_sum: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "operator", _snapshot(self.operator))
        object.__setattr__(self, "corrected_sum", _snapshot(self.corrected_sum))


@dataclass(frozen=True, slots=True)
class KernelObservationGroup:
    row_to_nodes: np.ndarray
    count: int
    weight_sum: float
    corrected_sum: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "row_to_nodes", _snapshot(self.row_to_nodes, dtype=np.int64))
        object.__setattr__(self, "corrected_sum", _snapshot(self.corrected_sum))


@dataclass(frozen=True, slots=True)
class PartialKernelStatistics:
    """Pairwise corrected means and observed support, without latent operators.

    Missing entries of partial_kernel are zero placeholders, always accompanied
    by observed_mask. Pair weights are normalized across dates; raw pair_counts
    count actual measurements, including zero-weight dates. Group moments use
    raw scaled weights and are divided by weight_sum when scored.
    """

    partial_kernel: np.ndarray
    pair_counts: np.ndarray
    pair_weight_sums: np.ndarray
    pair_effective_counts: np.ndarray
    pair_corrected_sums: np.ndarray
    observed_mask: np.ndarray
    weights: np.ndarray
    groups: tuple[KernelObservationGroup, ...]
    weight_sum: float
    n_observations: int
    uniform_weights: bool

    def __post_init__(self):
        for name in (
            "partial_kernel",
            "pair_counts",
            "pair_weight_sums",
            "pair_effective_counts",
            "pair_corrected_sums",
            "observed_mask",
            "weights",
        ):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))
        object.__setattr__(self, "groups", tuple(self.groups))

    @property
    def weighted_counts(self):
        """Normalized date-weight coverage for every observed pair."""
        return self.pair_weight_sums


@dataclass(frozen=True, slots=True)
class ObservationStatistics:
    groups: tuple[ObservationGroup, ...]
    weights: np.ndarray
    weight_sum: float
    n_observations: int
    uniform_weights: bool
    partial: PartialKernelStatistics | None = None
    layout_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "weights", _snapshot(self.weights))

    @property
    def partial_kernel(self):
        return None if self.partial is None else self.partial.partial_kernel

    @property
    def pair_counts(self):
        return None if self.partial is None else self.partial.pair_counts

    @property
    def pair_weight_sums(self):
        return None if self.partial is None else self.partial.pair_weight_sums

    @property
    def weighted_counts(self):
        return self.pair_weight_sums

    @property
    def pair_effective_counts(self):
        return None if self.partial is None else self.partial.pair_effective_counts

    @property
    def pair_corrected_sums(self):
        return None if self.partial is None else self.partial.pair_corrected_sums

    @property
    def observed_mask(self):
        return None if self.partial is None else self.partial.observed_mask


def _group_observations(data, weights):
    raw_weights, total, uniform = _weights(weights, len(data))
    groups = {}
    for item, weight in zip(data, raw_weights, strict=True):
        key = (item.operator.shape, item.operator.tobytes())
        if key not in groups:
            groups[key] = [item.operator, 0, 0.0, np.zeros((item.n_contracts, item.n_contracts))]
        group = groups[key]
        group[1] += 1
        group[2] += weight
        group[3] += item.measurement if uniform else weight * item.measurement
    if any(not np.all(np.isfinite(group[3])) for group in groups.values()):
        raise ValueError("corrected observation moments exceed floating-point range")
    return tuple(ObservationGroup(*group) for group in groups.values()), raw_weights, total, uniform


def _partial_statistics(values, row_to_nodes, noises, weights, size):
    rows = _row_indices(row_to_nodes, size=size, count=len(values))
    raw_weights, total, uniform = _weights(weights, len(values))
    grouped = {}
    for value, row, noise, weight in zip(values, rows, noises, raw_weights, strict=True):
        if value.shape != (len(row),):
            raise ValueError("each value vector must match its observation's node indices")
        key = tuple(row)
        if key not in grouped:
            grouped[key] = [row, [], [], []]
        grouped[key][1].append(value)
        grouped[key][2].append(weight)
        grouped[key][3].append(noise)
    counts = np.zeros((size, size), dtype=np.int64)
    pair_weights, squared_weights, sums = (np.zeros((size, size)) for _ in range(3))
    groups = []
    for row, samples, date_weights, group_noises in grouped.values():
        samples, date_weights = np.asarray(samples), np.asarray(date_weights)
        weight_sum = float(np.sum(date_weights))
        corrected = (
            samples.T @ samples if uniform else samples.T @ (date_weights[:, None] * samples)
        )
        for noise, weight in zip(group_noises, date_weights, strict=True):
            if np.ndim(noise) == 0:
                corrected.flat[:: len(row) + 1] -= weight * noise
            else:
                corrected -= weight * noise
        indices = np.ix_(row, row)
        counts[indices] += len(samples)
        pair_weights[indices] += weight_sum
        squared_weights[indices] += float(np.sum(date_weights**2))
        sums[indices] += corrected
        groups.append(KernelObservationGroup(row, len(samples), weight_sum, corrected))
    if not np.all(np.isfinite(sums)):
        raise ValueError("corrected pair moments exceed floating-point range")
    observed = pair_weights > 0
    partial = np.divide(sums, pair_weights, out=np.zeros_like(sums), where=observed)
    effective = np.divide(
        pair_weights**2, squared_weights, out=np.zeros_like(pair_weights), where=squared_weights > 0
    )
    return PartialKernelStatistics(
        partial,
        counts,
        pair_weights / total,
        effective,
        sums / total,
        observed,
        raw_weights / total,
        tuple(groups),
        total,
        len(values),
        uniform,
    )


def partial_kernel_from_arrays(values, row_to_nodes, noise_variance=0.0, weights=None, size=None):
    """Summarize dense fragments without constructing any latent operators.

    values is a sequence of one-dimensional observed vectors, one per row map
    (a rectangular N-by-m array is also a sequence of such vectors). It is not
    interpreted as an unmasked complete N-by-grid parent array. Noise variance
    is a known common variance of fresh independent point noise.
    """
    samples = tuple(_real_values(value, "fragment values") for value in values)
    if not samples:
        raise ValueError("at least one observed fragment is required")
    if (
        isinstance(noise_variance, (bool, np.bool_))
        or not isinstance(noise_variance, Real)
        or not np.isfinite(noise_variance)
        or noise_variance < 0
    ):
        raise ValueError("noise_variance must be finite and nonnegative")
    try:
        raw_rows = tuple(row_to_nodes)
    except TypeError as error:
        raise ValueError("row_to_nodes must be a sequence") from error
    if size is None:
        if not raw_rows or any(np.asarray(row).size == 0 for row in raw_rows):
            raise ValueError("node indices must be nonempty")
        try:
            size = max(int(np.max(row)) for row in raw_rows) + 1
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("node indices must be nonnegative integer arrays") from error
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    return _partial_statistics(
        samples, raw_rows, (float(noise_variance),) * len(samples), weights, int(size)
    )


def summarize_observations(observations, weights=None, *, layout=None):
    """Group corrected products and optionally compute point-grid pair coverage."""
    data = _observations(observations)
    groups, raw_weights, total, uniform = _group_observations(data, weights)
    partial, layout_id = None, None
    if layout is not None:
        if not isinstance(layout, FragmentLayout):
            raise TypeError("layout must be a FragmentLayout")
        layout.validate_observations(data)
        partial = _partial_statistics(
            tuple(item.y for item in data),
            layout.row_to_nodes,
            tuple(item.noise_covariance for item in data),
            weights,
            layout.n_nodes,
        )
        layout_id = layout.layout_id
    return ObservationStatistics(
        groups, raw_weights / total, total, len(data), uniform, partial, layout_id
    )


def _score(groups, total, predict, uniform):
    value = 0.0
    for group in groups:
        predicted = predict(group)
        count = group.count if uniform else group.weight_sum
        value += count * np.sum(predicted**2) - 2 * np.sum(predicted * group.corrected_sum)
    value = float(value / total)
    if not np.isfinite(value):
        raise ValueError("observed score exceeds floating-point range")
    return value


def covariance_quadratic_score(covariance, observations, weights=None):
    """Score a symmetric finite coordinate covariance; PSD validation is caller-owned."""
    matrix = validate_symmetric(covariance, name="candidate covariance")
    data = _observations(observations)
    if any(item.dimension != len(matrix) for item in data):
        raise ValueError("score observations must match the candidate basis")
    groups, _, total, uniform = _group_observations(data, weights)
    return _score(groups, total, lambda group: group.operator @ matrix @ group.operator.T, uniform)


def prediction_quadratic_score(predictions, observations, weights=None):
    """Score one symmetric finite prediction per date, permitting indefiniteness."""
    data = _observations(observations)
    matrices = tuple(
        validate_symmetric(value, name="predicted observation covariance") for value in predictions
    )
    if len(matrices) != len(data) or any(
        matrix.shape != (item.n_contracts, item.n_contracts) for matrix, item in zip(matrices, data)
    ):
        raise ValueError("one matching predicted covariance is required per observation")
    raw_weights, total, uniform = _weights(weights, len(data))
    groups = {}
    for matrix, item, weight in zip(matrices, data, raw_weights, strict=True):
        key = (matrix.shape, matrix.tobytes())
        if key not in groups:
            groups[key] = [matrix, 0, 0.0, np.zeros_like(matrix)]
        group = groups[key]
        group[1] += 1
        group[2] += weight
        group[3] += item.measurement if uniform else weight * item.measurement
    values = tuple(ObservationGroup(*group) for group in groups.values())
    return _score(values, total, lambda group: group.operator, uniform)


def kernel_quadratic_score_from_statistics(kernel, statistics):
    """Score a kernel repeatedly from precomputed fragment sufficient statistics."""
    matrix = validate_symmetric(kernel, name="candidate kernel")
    if (
        not isinstance(statistics, PartialKernelStatistics)
        or matrix.shape != statistics.partial_kernel.shape
    ):
        raise ValueError("kernel shape must match PartialKernelStatistics")
    return _score(
        statistics.groups,
        statistics.weight_sum,
        lambda group: matrix[np.ix_(group.row_to_nodes, group.row_to_nodes)],
        statistics.uniform_weights,
    )


def kernel_quadratic_score(kernel, observations, row_to_nodes, weights=None):
    """Corrected point-fragment score; the kernel may be symmetric indefinite."""
    matrix = validate_symmetric(kernel, name="candidate kernel")
    data = _observations(observations)
    statistics = _partial_statistics(
        tuple(item.y for item in data),
        row_to_nodes,
        tuple(item.noise_covariance for item in data),
        weights,
        len(matrix),
    )
    return kernel_quadratic_score_from_statistics(matrix, statistics)


def risk_proxy_summary(exposure_grid_weights, observations, row_to_nodes):
    """Same-date grid-functional proxies and availability, with known-noise correction.

    These are grid functional weights, not automatically an H-unit exposure.
    Unavailable dates are represented by an explicit mask rather than NaNs.
    Failure is not a general pooled-covariance nonidentification statement.
    """
    exposure = _real_values(exposure_grid_weights, "exposure grid weights")
    if exposure.ndim != 1 or not len(exposure):
        raise ValueError("exposure grid weights must be a nonempty vector")
    data = _observations(observations)
    rows = _row_indices(row_to_nodes, size=len(exposure), count=len(data))
    support = set(np.flatnonzero(exposure != 0))
    available, values, indices = np.zeros(len(data), dtype=bool), [], []
    for index, (item, row) in enumerate(zip(data, rows, strict=True)):
        if len(row) != item.n_contracts:
            raise ValueError("observed values must match their point-node rows")
        if support.issubset(row):
            coefficient = exposure[row]
            value = float(
                (coefficient @ item.y) ** 2 - coefficient @ item.noise_covariance @ coefficient
            )
            if not np.isfinite(value):
                raise ValueError("risk proxy exceeds floating-point range")
            available[index] = True
            values.append(value)
            indices.append(index)
    return {
        "values": _snapshot(values, dtype=float),
        "available_mask": _snapshot(available),
        "available_indices": _snapshot(indices, dtype=np.int64),
        "n_available": len(values),
        "n_observations": len(data),
        "mean": float(np.mean(values)) if values else None,
        "scope": "same-date grid-functional proxy; not H normalization or pooled identification",
    }
