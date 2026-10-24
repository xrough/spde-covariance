"""Immutable maturity-grid layouts attached to existing point observations.

A layout describes observation rows; it does not reinterpret their latent
basis or turn delivery averages into point measurements. Quadrature weights
retain their supplied physical scale rather than being normalized silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .observations import Observation


def _real_values(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite real numbers")
    return np.asarray(raw, dtype=float)


def _snapshot(value, *, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _row_indices(row_to_nodes, *, size, count=None):
    if isinstance(row_to_nodes, (str, bytes)):
        raise ValueError("row_to_nodes must contain one integer array per observation")
    try:
        rows = tuple(row_to_nodes)
    except TypeError as error:
        raise ValueError("row_to_nodes must be a sequence") from error
    if not rows or (count is not None and len(rows) != count):
        raise ValueError("row_to_nodes must match a nonempty observation sequence")
    result = []
    for row in rows:
        indices = np.asarray(row)
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= size)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("node indices must be unique, integer, nonempty and inside the grid")
        result.append(_snapshot(indices, dtype=np.int64))
    return tuple(result)


def _observations(observations):
    data = tuple(observations)
    if not data or any(not isinstance(item, Observation) for item in data):
        raise ValueError("observations must be a nonempty sequence of Observation objects")
    return data


def _digest(*arrays, labels=()):
    digest = hashlib.sha256()
    for label in labels:
        encoded = label.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for array in arrays:
        value = np.asarray(array)
        dtype = "<i8" if value.dtype.kind in "iu" else "<f8"
        canonical = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class FragmentLayout:
    """Dense point fragments on one finite quadrature grid.

    ``intervals`` is the catalogue of declared fragment supports. Each
    observation's ``row_to_nodes`` must contain exactly the nodes in one
    catalogue interval, although their row order may be permuted. Interval
    membership allows one floating-point ULP at either endpoint. Connected
    overlap and ordered serrated-domain eligibility are completion checks,
    not assumptions imposed on all observation layouts.

    An optional ``evaluation_matrix`` maps the declared latent coordinates to
    the grid. Without it, generic curve-reference controls can still compute
    coverage and moments, but operator-map validation is explicitly unavailable.
    """

    nodes: np.ndarray
    weights: np.ndarray
    intervals: tuple[tuple[float, float], ...]
    row_to_nodes: tuple[np.ndarray, ...]
    evaluation_matrix: np.ndarray | None = None
    basis_id: str | None = None
    _cliques: tuple[np.ndarray, ...] = field(init=False, repr=False)
    _declared_mask: np.ndarray = field(init=False, repr=False)
    _grid_id: str = field(init=False, repr=False)
    _layout_id: str = field(init=False, repr=False)

    def __post_init__(self):
        nodes = _real_values(self.nodes, "nodes")
        weights = _real_values(self.weights, "quadrature weights")
        if nodes.ndim != 1 or not len(nodes) or np.any(np.diff(nodes) <= 0):
            raise ValueError("nodes must be nonempty, one-dimensional and strictly increasing")
        if weights.shape != nodes.shape or np.any(weights <= 0):
            raise ValueError("quadrature weights must be strictly positive and match the nodes")
        if isinstance(self.intervals, (str, bytes)):
            raise ValueError("intervals must be a nonempty sequence of endpoint pairs")
        try:
            raw_intervals = tuple(self.intervals)
        except TypeError as error:
            raise ValueError("intervals must be a nonempty sequence") from error
        if not raw_intervals:
            raise ValueError("at least one declared fragment interval is required")
        intervals, cliques = [], []
        for interval in raw_intervals:
            try:
                endpoints = tuple(interval)
            except TypeError as error:
                raise ValueError("each interval must have two finite real endpoints") from error
            if len(endpoints) != 2 or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                for value in endpoints
            ):
                raise ValueError("each interval must have two finite real endpoints")
            start, end = map(float, endpoints)
            if end <= start:
                raise ValueError("fragment intervals require start < end")
            indices = np.flatnonzero(
                (nodes >= np.nextafter(start, -np.inf)) & (nodes <= np.nextafter(end, np.inf))
            )
            if not len(indices):
                raise ValueError("every declared fragment interval must contain a grid node")
            intervals.append((start, end))
            cliques.append(_snapshot(indices, dtype=np.int64))
        rows = _row_indices(self.row_to_nodes, size=len(nodes))
        allowed = {tuple(value) for value in cliques}
        if any(tuple(np.sort(row)) not in allowed for row in rows):
            raise ValueError("each point fragment must contain all nodes of one declared interval")
        evaluation = self.evaluation_matrix
        if evaluation is not None:
            evaluation = _real_values(evaluation, "evaluation matrix")
            if evaluation.ndim != 2 or evaluation.shape[0] != len(nodes) or not evaluation.shape[1]:
                raise ValueError("evaluation_matrix must have shape (n_nodes, basis_dimension)")
            evaluation = _snapshot(evaluation)
        if self.basis_id is not None and (
            not isinstance(self.basis_id, str) or not self.basis_id.strip()
        ):
            raise ValueError("basis_id must be None or a nonempty string")
        mask = np.zeros((len(nodes), len(nodes)), dtype=bool)
        for clique in cliques:
            mask[np.ix_(clique, clique)] = True
        grid_id = "maturity-grid:" + _digest(nodes, weights)
        layout_arrays = (nodes, weights, np.asarray(intervals), *rows)
        if evaluation is not None:
            layout_arrays += (evaluation,)
        layout_id = "fragment-layout:" + _digest(
            *layout_arrays,
            labels=(self.basis_id or "", "point_evaluation"),
        )
        for name, value in (
            ("nodes", _snapshot(nodes)),
            ("weights", _snapshot(weights)),
            ("intervals", tuple(intervals)),
            ("row_to_nodes", rows),
            ("evaluation_matrix", evaluation),
            ("_cliques", tuple(cliques)),
            ("_declared_mask", _snapshot(mask)),
            ("_grid_id", grid_id),
            ("_layout_id", layout_id),
        ):
            object.__setattr__(self, name, value)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_observations(self):
        return len(self.row_to_nodes)

    @property
    def cliques(self):
        return self._cliques

    @property
    def declared_mask(self):
        return self._declared_mask

    @property
    def grid_id(self):
        return self._grid_id

    @property
    def layout_id(self):
        return self._layout_id

    def validate_observations(
        self,
        observations: Sequence[Observation],
        *,
        basis_id=None,
        require_operator_validation=False,
        rtol=1e-10,
        atol=1e-12,
    ):
        """Validate row semantics and, when supplied, the actual basis-to-grid map."""
        data = _observations(observations)
        if len(data) != self.n_observations or any(
            item.n_contracts != len(row) for item, row in zip(data, self.row_to_nodes)
        ):
            raise ValueError("observation rows must match row_to_nodes")
        if not isinstance(require_operator_validation, bool):
            raise ValueError("require_operator_validation must be a boolean")
        if (
            not np.isscalar(rtol)
            or not np.isscalar(atol)
            or not np.isfinite(rtol)
            or not np.isfinite(atol)
            or rtol < 0
            or atol < 0
        ):
            raise ValueError("map tolerances must be finite and nonnegative")
        if basis_id is not None and (not isinstance(basis_id, str) or not basis_id.strip()):
            raise ValueError("basis_id must be a nonempty string when supplied")
        if self.basis_id is not None and basis_id is not None and self.basis_id != basis_id:
            raise ValueError("layout basis_id does not match observation provenance")
        if self.evaluation_matrix is None:
            if require_operator_validation:
                raise ValueError("operator validation requires an explicit evaluation_matrix")
        else:
            for item, row in zip(data, self.row_to_nodes, strict=True):
                expected = self.evaluation_matrix[row]
                if item.operator.shape != expected.shape or not np.allclose(
                    item.operator,
                    expected,
                    rtol=rtol,
                    atol=atol,
                ):
                    raise ValueError("observation operator does not match its declared point rows")
        return {
            "n_observations": len(data),
            "layout_id": self.layout_id,
            "grid_id": self.grid_id,
            "operator_maps_validated": self.evaluation_matrix is not None,
            "basis_id_checked": self.basis_id is not None and basis_id is not None,
        }
