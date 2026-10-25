"""Fixed true-point completion, observed likelihood and identification uncertainty.

Simulation truth is confined to parent construction and evaluation. The issued
kernel is fitted once from native fragments and is reused in all equal-law
worlds. A full-panel test is an additional audit, never native tuning data.
"""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .coherent import fit_coherent_partial
from .completion import complete_kernel, two_block_risk_range
from .fragment_benchmark import quadrature_weights
from .hedging_true_points import true_point_hedge_control
from .linalg import validate_psd
from .observation_statistics import (
    kernel_quadratic_score_from_statistics,
    partial_kernel_from_arrays,
)
from .provenance import _jsonable, _manifest, _source_root

NATIVE_CLIQUES = ((0, 1, 2, 3), (3, 4, 5, 6))
METHODS = ("gaussian_selected", "coherent_moment")
WORLDS = ("source_minus", "source_plus", "canonical_control")


@dataclass(frozen=True)
class StatisticalCompletionConfig:
    name: str = "statistical-completion"
    study: str = "statistical_completion"
    seed: int = 20261030
    replications: int = 20
    training_counts: tuple[int, ...] = (128, 512)
    validation_count: int = 128
    test_count: int = 512
    source_dimension: int = 16
    reference_dimension: int = 512
    quadrature_order: int = 24
    delta: float = 1 / 52
    noise_variance: float = 0.001
    prior_counts: tuple[float, ...] = (0.5, 2.0, 8.0)
    reference_variance_floor: float = 1e-6
    moment_eigenvalue_floor: float = 1e-9
    numerical_error_budget: float = 1e-5
    moment_max_iterations: int = 100000
    em_max_iterations: int = 2000
    stationarity_tolerance: float = 1e-6
    relative_tolerance: float = 1e-8
    confidence_level: float = 0.95
    coverage_replications: int = 500
    coverage_training_count: int = 512
    coverage_training_counts: tuple[int, ...] = ()
    coverage_noise_variance: float = 0.001
    coverage_mcse_target: float = 0.01
    interval_solver: str = "CLARABEL"
    interval_solver_tolerance: float = 1e-8
    interval_max_iterations: int = 100000
    coverage_precision_required: bool = True

    def __post_init__(self):
        if (
            self.study != "statistical_completion"
            or not isinstance(self.name, str)
            or not self.name
        ):
            raise ValueError("invalid statistical completion study/name")
        for name in (
            "seed",
            "replications",
            "validation_count",
            "test_count",
            "source_dimension",
            "reference_dimension",
            "quadrature_order",
            "moment_max_iterations",
            "em_max_iterations",
            "coverage_replications",
            "coverage_training_count",
            "interval_max_iterations",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < (name != "seed")
            ):
                raise ValueError(f"invalid {name}")
        counts = tuple(self.training_counts)
        if (
            not counts
            or tuple(sorted(set(counts))) != counts
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 8 or n % 2 for n in counts)
        ):
            raise ValueError("training_counts must be increasing even integers >= 8")
        object.__setattr__(self, "training_counts", counts)
        coverage_counts = tuple(self.coverage_training_counts)
        if coverage_counts and (
            tuple(sorted(set(coverage_counts))) != coverage_counts
            or self.coverage_training_count not in coverage_counts
            or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 8 or n % 2
                for n in coverage_counts
            )
        ):
            raise ValueError(
                "coverage_training_counts must be increasing even counts including coverage_training_count"
            )
        object.__setattr__(self, "coverage_training_counts", coverage_counts)
        if any(
            getattr(self, n) % 2
            for n in ("validation_count", "test_count", "coverage_training_count")
        ):
            raise ValueError("balanced native segments require even observation counts")
        if not 8 <= self.source_dimension <= self.reference_dimension:
            raise ValueError("require 8 <= source_dimension <= reference_dimension")
        if any(isinstance(x, (bool, np.bool_)) for x in self.prior_counts):
            raise ValueError("prior_counts must be real positive numbers, not booleans")
        candidates = tuple(float(x) for x in self.prior_counts)
        if (
            not candidates
            or tuple(sorted(set(candidates))) != candidates
            or any(not np.isfinite(x) or x <= 0 for x in candidates)
        ):
            raise ValueError("prior_counts must be distinct increasing positive values")
        object.__setattr__(self, "prior_counts", candidates)
        for name in (
            "delta",
            "reference_variance_floor",
            "moment_eigenvalue_floor",
            "numerical_error_budget",
            "stationarity_tolerance",
            "relative_tolerance",
            "coverage_mcse_target",
            "interval_solver_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("noise_variance", "coverage_noise_variance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie in (0,1)")
        if not isinstance(self.coverage_precision_required, bool):
            raise ValueError("coverage_precision_required must be boolean")
        if self.interval_solver not in {"CLARABEL", "SCS"}:
            raise ValueError("interval_solver must be CLARABEL or SCS")

    @property
    def total_count(self):
        return max(self.training_counts) + self.validation_count + self.test_count


def load_statistical_completion_config(path):
    with Path(path).open("rb") as handle:
        return StatisticalCompletionConfig(**tomllib.load(handle))


def _root(matrix):
    eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if eigenvalues[0] < -1e-10 * max(1.0, eigenvalues[-1]):
        raise ValueError("a conditional Gaussian covariance is materially indefinite")
    return vectors * np.sqrt(np.maximum(eigenvalues, 0))


def paired_completion_parents(kernels, cliques, *, count, noise_variance, seed):
    """Noisy Gaussian parents with common native observations in every world.

    Conditions each world's missing coordinates on a single common native
    draw. Requires equal native covariance blocks, to floating point. This
    coupling does not identify, or leak, unobserved values into fitting.
    """
    matrices = tuple(validate_psd(k, name="world kernel") for k in kernels)
    if not matrices:
        raise ValueError("at least one world kernel is required")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(noise_variance, bool) or not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    dimension = len(matrices[0])
    if not cliques or set().union(*(set(group) for group in cliques)) != set(range(dimension)):
        raise ValueError("cliques must cover the grid")
    if any(not group or len(set(group)) != len(group) for group in cliques):
        raise ValueError("cliques must be nonempty and contain unique node indices")
    if any(k.shape != (dimension, dimension) for k in matrices):
        raise ValueError("all world kernels must share a square shape")
    for group in cliques:
        ix = np.ix_(group, group)
        if any(
            not np.allclose(k[ix], matrices[0][ix], rtol=1e-11, atol=1e-12) for k in matrices[1:]
        ):
            raise ValueError("world kernels do not have the same native observation law")
    rng = np.random.default_rng(seed)
    parents = [np.empty((count, dimension)) for _ in matrices]
    for parity, group in enumerate(cliques):
        dates = np.arange(parity, count, len(cliques))
        observed = np.asarray(group, dtype=int)
        missing = np.asarray(sorted(set(range(dimension)) - set(group)), dtype=int)
        block = matrices[0][np.ix_(observed, observed)] + noise_variance * np.eye(len(group))
        visible = rng.normal(size=(len(dates), len(observed))) @ _root(block).T
        hidden_standard = rng.normal(size=(len(dates), len(missing)))
        for parent, kernel in zip(parents, matrices, strict=True):
            cross = kernel[np.ix_(missing, observed)]
            regression = np.linalg.solve(block, cross.T).T
            conditional = (
                kernel[np.ix_(missing, missing)]
                + noise_variance * np.eye(len(missing))
                - regression @ cross.T
            )
            parent[np.ix_(dates, observed)] = visible
            parent[np.ix_(dates, missing)] = (
                visible @ regression.T + hidden_standard @ _root(conditional).T
            )
    return tuple(parents)


def _native_values(parent, start, count, cliques=NATIVE_CLIQUES):
    rows = tuple(
        np.asarray(cliques[i % len(cliques)], dtype=int) for i in range(start, start + count)
    )
    values = tuple(
        parent[i, nodes].copy() for i, nodes in zip(range(start, start + count), rows, strict=True)
    )
    return values, rows


def _hash_arrays(values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def training_reference(statistics, *, relative_floor):
    """Diagonal reference from noise-corrected training variances only."""
    diagonal = np.diag(statistics.partial_kernel)
    scale = max(float(np.max(np.abs(diagonal))), np.finfo(float).tiny)
    floor = max(relative_floor * scale, np.finfo(float).eps * max(scale, 1.0))
    return np.diag(np.maximum(diagonal, floor)), {"variance_scale": scale, "variance_floor": floor}


def fit_statistical_candidates(
    training, training_rows, validation, validation_rows, *, config, weights
):
    """Observed-only model fitting and tuning; no world/truth argument exists."""
    from .gaussian_completion import fit_gaussian_completion, gaussian_kernel_score

    statistics = partial_kernel_from_arrays(
        training, training_rows, noise_variance=config.noise_variance, size=7
    )
    reference, reference_metadata = training_reference(
        statistics, relative_floor=config.reference_variance_floor
    )
    fits, candidates, selections = {}, [], {}
    for method in METHODS:
        available = []
        for pseudo_count in config.prior_counts:
            relative = pseudo_count / len(training)
            regularization = (
                relative
                if method == "gaussian_selected"
                else relative * float(np.max(statistics.pair_weight_sums))
            )
            started = time.perf_counter()
            record = {
                "method": method,
                "prior_count": pseudo_count,
                "relative_regularization": relative,
                "regularization": regularization,
                "reference": reference_metadata,
                "reference_sha256": _hash_arrays((reference,)),
                "available": False,
            }
            kernel = None
            try:
                if method == "gaussian_selected":
                    fit = fit_gaussian_completion(
                        training,
                        training_rows,
                        noise_covariances=config.noise_variance,
                        cliques=NATIVE_CLIQUES,
                        reference_kernel=reference,
                        regularization=regularization,
                        max_iterations=config.em_max_iterations,
                        stationarity_tolerance=config.stationarity_tolerance,
                        relative_tolerance=config.relative_tolerance,
                    )
                    record["diagnostics"] = _jsonable(fit.diagnostics)
                    if fit.converged:
                        kernel = fit.kernel
                else:
                    fit = fit_coherent_partial(
                        np.where(statistics.observed_mask, statistics.partial_kernel, np.nan),
                        pair_weights=statistics.pair_weight_sums,
                        cliques=NATIVE_CLIQUES,
                        weights=weights,
                        regularization=regularization,
                        reference_kernel=reference,
                        eigenvalue_floor=config.moment_eigenvalue_floor,
                        numerical_error_budget=config.numerical_error_budget,
                        max_iterations=config.moment_max_iterations,
                    )
                    record["diagnostics"] = _jsonable(fit.diagnostics)
                    if fit.converged:
                        completion = complete_kernel(
                            fit.partial_kernel,
                            cliques=NATIVE_CLIQUES,
                            weights=weights,
                            population=True,
                        )
                        record["completion_diagnostics"] = _jsonable(completion.diagnostics)
                        if completion.available:
                            kernel = completion.kernel
                if kernel is not None:
                    score = gaussian_kernel_score(
                        kernel, validation, validation_rows, noise_covariances=config.noise_variance
                    )
                    record.update(
                        available=True,
                        validation_nll=float(score),
                        kernel_sha256=_hash_arrays((kernel,)),
                    )
                    available.append((score, -pseudo_count, kernel, len(candidates)))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                record["error"] = str(error)
            record["runtime_seconds"] = time.perf_counter() - started
            candidates.append(record)
        if available:
            _, _, kernel, candidate_index = min(available, key=lambda item: item[:2])
            fits[method] = kernel
            selections[method] = candidate_index
        else:
            fits[method], selections[method] = None, None
    return fits, candidates, selections


def _exposures(points):
    representer_gram = 2 - np.exp(-np.minimum.outer(points, points))
    raw = {
        "within_left": np.eye(7)[0] - np.eye(7)[1],
        "across_fragments": np.eye(7)[0] - np.eye(7)[6],
        "spread_difference": np.eye(7)[0] - np.eye(7)[1] - np.eye(7)[5] + np.eye(7)[6],
    }
    return {name: a / np.sqrt(a @ representer_gram @ a) for name, a in raw.items()}


def population_completion_controls(config):
    control = true_point_hedge_control(
        source_dimension=config.source_dimension,
        reference_dimension=config.reference_dimension,
        quadrature_order=config.quadrature_order,
        delta=config.delta,
    )
    minus, plus = (np.asarray(k) for k in control["true_point_pair"]["kernels"])
    mask = np.asarray(control["observed_mask"], dtype=bool)
    points = np.asarray(control["grid"])
    weights = quadrature_weights(points)
    weights /= np.sum(weights)
    completion = complete_kernel(
        np.where(mask, minus, np.nan), cliques=NATIVE_CLIQUES, weights=weights, population=True
    )
    if not completion.available:
        raise RuntimeError("population canonical control is unavailable")
    kernels = (minus, plus, completion.kernel)
    exposures = _exposures(points)
    ranges = []
    for name, exposure in exposures.items():
        interval = two_block_risk_range(
            np.where(mask, minus, np.nan), exposure, cliques=NATIVE_CLIQUES, weights=weights
        )
        ranges.append(
            {
                "target": name,
                "lower": interval.lower,
                "upper": interval.upper,
                "identified": interval.identified,
                "available": interval.diagnostics["available"],
                "world_values": {
                    w: float(exposure @ k @ exposure) for w, k in zip(WORLDS, kernels, strict=True)
                },
                "scope": "exact-partial ordinary PSD two-block completions; no confidence or source-class claim",
            }
        )
    root = np.sqrt(weights)
    floors = {
        w: float(np.linalg.norm(root[:, None] * (k - completion.kernel) * root[None, :]))
        for w, k in zip(WORLDS, kernels, strict=True)
    }
    noisy_canonical = completion.kernel + config.noise_variance * np.eye(7)
    bridge_kl = {}
    for world, kernel in zip(WORLDS, kernels, strict=True):
        noisy_truth = kernel + config.noise_variance * np.eye(7)
        bridge_kl[world] = float(
            max(
                0,
                (
                    np.trace(np.linalg.solve(noisy_canonical, noisy_truth))
                    - 7
                    + np.linalg.slogdet(noisy_canonical)[1]
                    - np.linalg.slogdet(noisy_truth)[1]
                )
                / 2,
            )
        )
    return {
        "grid": points,
        "weights": weights,
        "observed_mask": mask,
        "kernels": dict(zip(WORLDS, kernels, strict=True)),
        "exposures": exposures,
        "identification_ranges": ranges,
        "canonical_structural_error": floors,
        "canonical_bridge_gaussian_kl": bridge_kl,
        "observed_law_max_difference": max(
            float(np.max(np.abs((k - minus)[mask]))) for k in kernels
        ),
        "source_pair": control,
        "canonical_control_scope": "grid Gaussian canonical completion; no asserted transport-source membership",
    }


def _risk_records(kernel, truth, full_values, native_values, native_rows, exposures, noise):
    result = {}
    for name, a in exposures.items():
        predicted, actual = float(a @ kernel @ a), float(a @ truth @ a)
        full_proxy = (full_values @ a) ** 2 - noise * (a @ a)
        support = np.flatnonzero(a)
        native_proxies = [
            float((a[rows] @ value) ** 2 - noise * (a[rows] @ a[rows]))
            for value, rows in zip(native_values, native_rows, strict=True)
            if np.all(np.isin(support, rows))
        ]
        result[name] = {
            "prediction": predicted,
            "truth": actual,
            "squared_error": (predicted - actual) ** 2,
            "full_audit_score": float(predicted**2 - 2 * predicted * np.mean(full_proxy)),
            "full_audit_count": len(full_values),
            "native_available": bool(native_proxies),
            "native_count": len(native_proxies),
            "native_score": float(predicted**2 - 2 * predicted * np.mean(native_proxies))
            if native_proxies
            else None,
        }
    return result


def _evaluate(kernel, truth, full_values, native, native_rows, population, config):
    from .gaussian_completion import gaussian_kernel_score

    full_rows = (np.arange(7),) * len(full_values)
    full_tuple = tuple(full_values)
    statistics = partial_kernel_from_arrays(
        native, native_rows, noise_variance=config.noise_variance, size=7
    )
    full_statistics = partial_kernel_from_arrays(
        full_tuple, full_rows, noise_variance=config.noise_variance, size=7
    )
    weighted_error = np.sqrt(population["weights"][:, None] * population["weights"][None, :]) * (
        kernel - truth
    )
    mask = population["observed_mask"]
    native_nll = float(
        gaussian_kernel_score(kernel, native, native_rows, noise_covariances=config.noise_variance)
    )
    bridge_nll = float(
        gaussian_kernel_score(
            kernel, full_tuple, full_rows, noise_covariances=config.noise_variance
        )
    )
    return {
        "native_nll": native_nll,
        "bridge_nll": bridge_nll,
        "native_nll_excess": native_nll
        - float(
            gaussian_kernel_score(
                truth, native, native_rows, noise_covariances=config.noise_variance
            )
        ),
        "bridge_nll_excess": bridge_nll
        - float(
            gaussian_kernel_score(
                truth, full_tuple, full_rows, noise_covariances=config.noise_variance
            )
        ),
        "native_quadratic_score": kernel_quadratic_score_from_statistics(kernel, statistics),
        "bridge_quadratic_score": kernel_quadratic_score_from_statistics(kernel, full_statistics),
        "weighted_kernel_error": float(np.linalg.norm(weighted_error)),
        "observed_error": float(np.linalg.norm(weighted_error[mask])),
        "missing_error": float(np.linalg.norm(weighted_error[~mask])),
        "minimum_kernel_eigenvalue": float(np.linalg.eigvalsh(kernel)[0]),
        "risks": _risk_records(
            kernel,
            truth,
            full_values,
            native,
            native_rows,
            population["exposures"],
            config.noise_variance,
        ),
    }


def _point_replication(config, population, replication):
    seed = np.random.SeedSequence([config.seed, 1, replication])
    kernels = tuple(population["kernels"][w] for w in WORLDS)
    parents = paired_completion_parents(
        kernels,
        NATIVE_CLIQUES,
        count=config.total_count,
        noise_variance=config.noise_variance,
        seed=seed,
    )
    validation_start = max(config.training_counts)
    test_start = validation_start + config.validation_count
    validation, validation_rows = _native_values(
        parents[0], validation_start, config.validation_count
    )
    audits, issued, records = [], [], []
    for count in config.training_counts:
        training, training_rows = _native_values(parents[0], 0, count)
        fits, candidates, selections = fit_statistical_candidates(
            training,
            training_rows,
            validation,
            validation_rows,
            config=config,
            weights=population["weights"],
        )
        training_hash = _hash_arrays((*training, *training_rows))
        for candidate in candidates:
            audits.append(
                {
                    "replication": replication,
                    "training_count": count,
                    "training_sha256": training_hash,
                    **candidate,
                }
            )
        for method, kernel in fits.items():
            selected_index = selections[method]
            selected = None if selected_index is None else candidates[selected_index]
            kernel_hash = None if kernel is None else _hash_arrays((kernel,))
            issued.append(
                {
                    "replication": replication,
                    "training_count": count,
                    "method": method,
                    "training_first": 0,
                    "training_stop": count,
                    "validation_first": validation_start,
                    "validation_stop": test_start,
                    "test_first": test_start,
                    "test_stop": config.total_count,
                    "training_sha256": training_hash,
                    "kernel_sha256": kernel_hash,
                    "selected_prior_count": None if selected is None else selected["prior_count"],
                    "kernel": kernel,
                    "available": kernel is not None,
                }
            )
            for world, parent, truth in zip(WORLDS, parents, kernels, strict=True):
                native, rows = _native_values(parent, test_start, config.test_count)
                native_equal = all(
                    np.array_equal(a, b)
                    for a, b in zip(training, _native_values(parent, 0, count)[0], strict=True)
                )
                if not native_equal:
                    raise RuntimeError("equal-law native parent coupling failed")
                record = {
                    "replication": replication,
                    "world": world,
                    "training_count": count,
                    "method": method,
                    "available": kernel is not None,
                    "training_sha256": training_hash,
                    "kernel_sha256": kernel_hash,
                    "selected_prior_count": None if selected is None else selected["prior_count"],
                    "native_test_sha256": _hash_arrays(native),
                }
                if kernel is not None:
                    record.update(
                        _evaluate(
                            kernel, truth, parent[test_start:], native, rows, population, config
                        )
                    )
                records.append(record)
    return {"replication": replication, "rows": records, "candidates": audits, "issued": issued}


def _binomial_summary(successes, count):
    if not count:
        return {
            "successes": successes,
            "count": count,
            "rate": None,
            "mcse": None,
            "interval_95": None,
        }
    rate = successes / count
    return {
        "successes": successes,
        "count": count,
        "rate": rate,
        "mcse": float(np.sqrt(rate * (1 - rate) / count)),
        "interval_95": [
            0.0 if successes == 0 else float(beta.ppf(0.025, successes, count - successes + 1)),
            1.0 if successes == count else float(beta.ppf(0.975, successes + 1, count - successes)),
        ],
    }


def coverage_experiment(config, *, complete=False):
    """Independent, well-conditioned 3-node band and risk-interval assessment."""
    from .completion_uncertainty import completion_risk_interval, gaussian_completion_region

    canonical = np.array([[1.0, 0.4, 0.16], [0.4, 1.0, 0.4], [0.16, 0.4, 1.0]])
    alternative = canonical.copy()
    alternative[0, 2] = alternative[2, 0] = 0.65
    exposure = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    cliques = ((0, 1), (1, 2))
    rows = tuple(
        np.arange(3) if complete else np.asarray(cliques[i % 2])
        for i in range(config.coverage_training_count)
    )
    truth_risks = {
        "canonical": float(exposure @ canonical @ exposure),
        "noncanonical": float(exposure @ alternative @ exposure),
    }
    if complete:
        # Complete observations have a different law in the noncanonical world.
        # Their coverage is assessed against their own canonical generating law only.
        truth_risks.pop("noncanonical")
    parent_root = _root(canonical + config.coverage_noise_variance * np.eye(3))
    records = []
    for replication in range(config.coverage_replications):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2, replication]))
        parents = rng.normal(size=(config.coverage_training_count, 3)) @ parent_root.T
        values = tuple(parents[i, ids] for i, ids in enumerate(rows))
        region = gaussian_completion_region(
            values,
            rows,
            noise_covariances=config.coverage_noise_variance,
            size=3,
            confidence_level=config.confidence_level,
        )
        mask = region.observed_mask
        covered = bool(
            region.available
            and np.all(canonical[mask] >= region.lower[mask])
            and np.all(canonical[mask] <= region.upper[mask])
        )
        record = {
            "replication": replication,
            "training_count": config.coverage_training_count,
            "design": "complete" if complete else "native",
            "region_available": region.available,
            "partial_truth_covered": covered,
            "region_reason": region.reason,
            "region_diagnostics": _compact_region_diagnostics(region.diagnostics),
            "interval_available": False,
        }
        if region.available:
            interval = completion_risk_interval(
                region,
                exposure,
                solver=config.interval_solver,
                tolerance=config.interval_solver_tolerance,
                max_iterations=config.interval_max_iterations,
            )
            record.update(
                interval_available=interval.available,
                lower=interval.lower,
                upper=interval.upper,
                interval_diagnostics=_jsonable(interval.diagnostics),
                minimum_status=interval.minimizer.diagnostics["status"],
                maximum_status=interval.maximizer.diagnostics["status"],
                minimum_diagnostics=_jsonable(interval.minimizer.diagnostics),
                maximum_diagnostics=_jsonable(interval.maximizer.diagnostics),
                inner_lower=interval.inner_lower,
                inner_upper=interval.inner_upper,
            )
            if interval.available:
                record["width"] = interval.upper - interval.lower
                record["risk_covered"] = {
                    name: bool(interval.lower <= risk <= interval.upper)
                    for name, risk in truth_risks.items()
                }
        records.append(record)
    available = [r for r in records if r["interval_available"]]
    p = config.confidence_level
    nominal_mcse = float(np.sqrt(p * (1 - p) / config.coverage_replications))
    mask = np.zeros((3, 3), dtype=bool)
    for group in cliques:
        mask[np.ix_(group, group)] = True
    population_range = two_block_risk_range(
        np.where(mask, canonical, np.nan), exposure, cliques=cliques
    )
    widths = np.asarray([r["width"] for r in available])
    return {
        "records": records,
        "kernels": {"canonical": canonical, "noncanonical": alternative},
        "truth_risks": truth_risks,
        "exposure": exposure,
        "partial_coverage": _binomial_summary(
            sum(r["partial_truth_covered"] for r in records), len(records)
        ),
        "risk_coverage_available": {
            name: _binomial_summary(sum(r["risk_covered"][name] for r in available), len(available))
            for name in truth_risks
        },
        "risk_coverage_all": {
            name: _binomial_summary(sum(r["risk_covered"][name] for r in available), len(records))
            for name in truth_risks
        },
        "interval_available_count": len(available),
        "interval_unavailable_count": len(records) - len(available),
        "mean_width": float(np.mean([r["width"] for r in available])) if available else None,
        "width_mcse": float(np.std(widths, ddof=1) / np.sqrt(len(widths)))
        if len(widths) > 1
        else None,
        "training_count": config.coverage_training_count,
        "design": "complete" if complete else "native",
        "nominal_coverage_mcse": nominal_mcse,
        "precision_target": config.coverage_mcse_target,
        "precision_goal_met": nominal_mcse <= config.coverage_mcse_target,
        "population_identification_range": {
            "lower": truth_risks["canonical"] if complete else population_range.lower,
            "upper": truth_risks["canonical"] if complete else population_range.upper,
        },
        "scope": "Bonferroni Gaussian observed-moment region; numerical all-PSD risk endpoints, not canonical confidence intervals",
        "sampling": "independent coverage replications and separate seed namespace; no point-study fitting or tuning",
    }


def _compact_region_diagnostics(diagnostics):
    result = _jsonable(diagnostics)
    # Group counts/sums/noise and bounds are sufficient to audit these fixed
    # designs. Membership can be regenerated from the recorded seed and mask.
    for interval in result.get("scalar_group_intervals", ()):
        interval.pop("date_indices", None)
    return result


def coverage_resolution_study(config):
    """Nested prefixes and complete views share parents; MCSE stays per stratum."""
    counts = config.coverage_training_counts
    if not counts:
        return coverage_experiment(config)
    strata = []
    primary = None
    for count in counts:
        local = replace(config, coverage_training_count=count, coverage_training_counts=())
        for complete in (False, True):
            result = coverage_experiment(local, complete=complete)
            if count == config.coverage_training_count and not complete:
                primary = result
            strata.append(result)
    if primary is None:
        raise ValueError("coverage study needs its declared primary native sample count")
    return {
        **primary,
        "resolution_study": strata,
        "paired_nested_sample_sizes": True,
        "paired_complete_observation_views": True,
        "monte_carlo_replication_unit": "replication index within each design/count; do not pool strata",
    }


def _summary(rows):
    metrics = (
        "native_nll",
        "bridge_nll",
        "native_nll_excess",
        "bridge_nll_excess",
        "native_quadratic_score",
        "bridge_quadratic_score",
        "weighted_kernel_error",
        "observed_error",
        "missing_error",
    )
    result = []
    for world, count, method in sorted(
        {(r["world"], r["training_count"], r["method"]) for r in rows}
    ):
        group = [
            r
            for r in rows
            if (r["world"], r["training_count"], r["method"]) == (world, count, method)
        ]
        available = [r for r in group if r["available"]]
        record = {
            "world": world,
            "training_count": count,
            "method": method,
            "replications": len(group),
            "available": len(available),
            "unavailable": len(group) - len(available),
        }
        for metric in metrics:
            values = [r[metric] for r in available]
            record[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "mcse": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else None,
            }
        result.append(record)
    return result


def _paired(rows):
    result = []
    index = {(r["replication"], r["world"], r["training_count"], r["method"]): r for r in rows}
    for world, count in sorted({(r["world"], r["training_count"]) for r in rows}):
        for metric in ("native_nll", "bridge_nll", "weighted_kernel_error", "missing_error"):
            differences = []
            for rep in sorted({r["replication"] for r in rows}):
                a, b = (index[(rep, world, count, m)] for m in METHODS)
                if a["available"] and b["available"]:
                    differences.append(a[metric] - b[metric])
            result.append(
                {
                    "world": world,
                    "training_count": count,
                    "metric": metric,
                    "comparison": "gaussian_selected minus coherent_moment",
                    "replications": len(differences),
                    "mean": float(np.mean(differences)) if differences else None,
                    "mcse": float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
                    if len(differences) > 1
                    else None,
                }
            )
    return result


def _json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def run_statistical_completion(config, output=None):
    """Run a new output directory; never overwrite or publish existing evidence."""
    if isinstance(config, (str, Path)):
        config = load_statistical_completion_config(config)
    if not isinstance(config, StatisticalCompletionConfig):
        raise TypeError("config must be StatisticalCompletionConfig or a TOML path")
    directory = Path(output) if output is not None else Path("experiments/runs") / config.name
    directory.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(config, _source_root())
    manifest["experiment"] = (
        "fixed true-point equal-native-law completion study and independent Gaussian coverage controls"
    )
    manifest["solver"] = {
        "gaussian": "multistart graph-constrained EM; local stationarity only",
        "moment": "coherent ADMM; observed-partial primal-dual certificate",
        "uncertainty": "Gaussian simultaneous moment bands and numerical PSD support intervals",
    }
    _json(directory / "config.json", asdict(config))
    _json(directory / "manifest.json", manifest)
    _json(directory / "status.json", {"status": "running", "completed_replications": 0})
    started = time.perf_counter()
    try:
        population = population_completion_controls(config)
        _json(directory / "population.json", population)
        checkpoints = []
        manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        for replication in range(config.replications):
            record = _point_replication(config, population, replication)
            record["parent_manifest_sha256"] = manifest_hash
            _json(directory / f"replication-{replication:04d}.json", record)
            checkpoints.append(record)
            _json(
                directory / "status.json",
                {
                    "status": "running",
                    "completed_replications": len(checkpoints),
                    "phase": "point_estimation",
                },
            )
        rows = [r for c in checkpoints for r in c["rows"]]
        candidates = [r for c in checkpoints for r in c["candidates"]]
        issued = [r for c in checkpoints for r in c["issued"]]
        _json(directory / "point_rows.json", rows)
        _json(directory / "candidates.json", candidates)
        _json(directory / "issued_kernels.json", issued)
        _json(
            directory / "status.json",
            {"status": "running", "completed_replications": len(checkpoints), "phase": "coverage"},
        )
        coverage = coverage_resolution_study(config)
        _json(directory / "coverage.json", coverage)
        native_equal = all(
            len(
                {
                    r["native_test_sha256"]
                    for r in rows
                    if r["replication"] == rep
                    and r["training_count"] == count
                    and r["method"] == method
                }
            )
            == 1
            for rep in range(config.replications)
            for count in config.training_counts
            for method in METHODS
        )
        unavailable = sum(not r["available"] for r in rows)
        failures = []
        if population["observed_law_max_difference"] > 1e-10 or not native_equal:
            failures.append("native-law/data equality control failed")
        if unavailable:
            failures.append("one or more selected point estimates are unavailable")
        if config.coverage_precision_required and not coverage["precision_goal_met"]:
            failures.append("coverage Monte Carlo precision target not met")
        for stratum in coverage.get("resolution_study", (coverage,)):
            total = config.coverage_replications
            if stratum["interval_available_count"] < 0.99 * total:
                failures.append(
                    f"uncertainty intervals unavailable in more than 1% of the declared well-conditioned control: {stratum.get('design')}/{stratum.get('training_count')}"
                )
        results = {
            "summary": _summary(rows),
            "paired": _paired(rows),
            "rows": rows,
            "candidates": candidates,
            "issued": issued,
            "coverage": coverage,
            "population": population,
            "native_inputs_identical": native_equal,
            "failed_candidate_count": sum(not r["available"] for r in candidates),
            "unavailable_selected_count": unavailable,
            "acceptance": {
                "passed": not failures,
                "failures": failures,
                "statistical_superiority_gate": None,
                "nominal_coverage_pass_fail_gate": None,
            },
            "runtime_seconds": time.perf_counter() - started,
            "protocol": {
                "chronology": "train prefixes; validation after largest prefix; untouched test after validation; independent coverage seeds",
                "selection": "native validation mean Gaussian marginal NLL; ties choose larger pseudo-count; failed candidates retained and excluded",
                "graph": "same two-clique precision graph for all fits and full-bridge tests; no automatic graph changes",
                "canonical_control": population["canonical_control_scope"],
                "mcse": "independent replication clusters; point and coverage experiments separate",
                "target": "normalized weekly innovation point kernel, not instantaneous source covariance",
                "uncertainty": coverage["scope"],
                "weights": "normalized physical trapezoidal grid weights; not H-coordinate matrix weights",
            },
        }
        _json(directory / "results.json", results)
        from .statistical_completion_reporting import write_statistical_completion_report

        write_statistical_completion_report(directory, config, results)
        _json(
            directory / "status.json",
            {
                "status": "complete",
                "acceptance": results["acceptance"],
                "completed_replications": config.replications,
                "coverage_replications": config.coverage_replications,
            },
        )
        return results
    except BaseException as error:
        _json(directory / "status.json", {"status": "failed", "error": str(error)})
        raise
