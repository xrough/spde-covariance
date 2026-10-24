"""Declared population controls for covariance-kernel completion.

The paper kernels are evaluation kernels on [0, 1], not matrices in the
transport model's Sobolev coordinate metric. Author-code parity is restricted
to explicit ranks; it deliberately does not adopt the legacy FVE normalization.
"""

import csv
import json
from pathlib import Path

import numpy as np

AUTHOR_REVISION = "0d332d99583315b60e17b6fd83b6c2e0b74e8da0"
AUTHOR_SOURCE_URL = (
    f"https://raw.githubusercontent.com/kgwstat/covcomp/{AUTHOR_REVISION}/R/covcomp.R"
)
AUTHOR_SOURCE_SHA256 = "a38acc2ed2149a0b727f8d8bf1a760e1f7819d7bc0eddccd3dc5117af5596ebe"
PAPER_MODELS = ("finite_rank", "brownian", "analytic")


def _nodes(nodes):
    values = np.asarray(nodes, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("nodes must be a nonempty finite one-dimensional array")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("paper control nodes must lie in [0, 1]")
    return values


def paper_kernel(name, nodes):
    """Evaluate the three kernels in Waghmare--Panaretos (2022), Section 10.

    ``finite_rank``/``K1`` has rank at most four; ``brownian``/``K2`` is
    min(s,t); ``analytic``/``K3`` is 10*s*t*exp(-10*(s-t)**2).
    Node order and repetitions are preserved.
    """
    x = _nodes(nodes)
    aliases = {"K1": "finite_rank", "K2": "brownian", "K3": "analytic"}
    name = aliases.get(name, name)
    if name == "finite_rank":
        phi = np.column_stack(
            (
                np.ones_like(x),
                np.sqrt(3) * (2 * x - 1),
                np.sqrt(5) * (6 * x**2 - 6 * x + 1),
                np.sqrt(7) * (20 * x**3 - 30 * x**2 + 12 * x - 1),
            )
        )
        return (phi * np.array([1, 0.5, 0.25, 0.125])) @ phi.T
    if name == "brownian":
        return np.minimum.outer(x, x)
    if name == "analytic":
        return 10 * np.outer(x, x) * np.exp(-10 * (x[:, None] - x[None, :]) ** 2)
    raise ValueError(f"unknown paper kernel {name!r}; choose one of {PAPER_MODELS}")


def paper_cover(count, domain=(0, 1)):
    """Paper covers with 2, 3, 5, 9 or 17 intervals, affinely rescaled.

    Each interval has 60% of the domain length. Rescaling the cover does not
    rescale ``paper_kernel``, whose original domain remains [0, 1].
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("cover count must be an integer")
    if count not in (2, 3, 5, 9, 17):
        raise ValueError("paper cover count must be 2, 3, 5, 9 or 17")
    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0]:
        raise ValueError("domain must be two finite increasing bounds")
    left, right = bounds
    length = right - left
    starts = left + np.linspace(0, 0.4, count) * length
    return tuple((float(a), float(a + 0.6 * length)) for a in starts)


def noncanonical_control(nodes, rho=0.5):
    """A PSD kernel family with identical cover-2 observations for |rho|<=1.

    The two private factors vanish on the overlap [.4, .6]. Their residual
    correlation rho is invisible on either observed square. rho=0 is the
    canonical completion; endpoint-spread variance is 2-2*rho.
    """
    x = _nodes(nodes)
    if not np.isscalar(rho) or not np.isfinite(rho) or abs(rho) > 1:
        raise ValueError("rho must be a finite scalar in [-1, 1]")
    left = np.maximum(1 - x / 0.4, 0)
    right = np.maximum((x - 0.6) / 0.4, 0)
    return (
        np.ones((len(x), len(x)))
        + np.outer(left, left)
        + np.outer(right, right)
        + rho * (np.outer(left, right) + np.outer(right, left))
    )


def cover_cliques(nodes, intervals):
    """Integer grid cliques using the same closed-interval tolerance as layout."""
    x = _nodes(nodes)
    return tuple(np.flatnonzero((x >= a - 1e-12) & (x <= b + 1e-12)) for a, b in intervals)


def clique_mask(size, cliques):
    mask = np.zeros((size, size), dtype=bool)
    for clique in cliques:
        mask[np.ix_(clique, clique)] = True
    return mask


def truncation_psd_control():
    """Return an exact PSD matrix whose rank-one completion is indefinite.

    Variables are (A, B1, B2, C), Var(B1)=2 and A=B2=C with Var(B2)=1.
    Keeping only B1 in the overlap inverse incorrectly fills Cov(A,C)=0.
    """
    kernel = np.array([[1, 0, 1, 1], [0, 2, 0, 0], [1, 0, 1, 1], [1, 0, 1, 1]], float)
    cliques = (np.array([0, 1, 2]), np.array([1, 2, 3]))
    return {"kernel": kernel, "cliques": cliques, "mask": clique_mask(4, cliques)}


def _require_completion(partial, cliques, **options):
    from .completion import complete_kernel

    result = complete_kernel(partial, cliques=cliques, **options)
    if not result.available or result.kernel is None:
        raise RuntimeError(f"reference completion failed: {result.diagnostics}")
    return result


def _completion_summary(diagnostics):
    """Keep the control result JSON-small; full arrays belong to fit artifacts."""
    keys = (
        "available",
        "input_cliques_psd",
        "output_psd",
        "support_compatible",
        "observed_preservation_residual",
        "maximum_inverse_norm",
        "support_rtol",
    )
    join_keys = (
        "clique_number",
        "selected_rank",
        "support_rank",
        "retained_condition_number",
        "inverse_norm",
        "available",
        "skipped",
    )
    return {
        **{key: diagnostics[key] for key in keys if key in diagnostics},
        "joins": [
            {key: row[key] for key in join_keys if key in row} for row in diagnostics["joins"]
        ],
    }


def population_controls(grid_size=41, *, support_rtol=1e-12):
    """Reproducible finite-grid audits, with no sampling or hidden target tuning.

    These compare manual paper ranks 4/2 to the known generating grid kernel.
    Missing-region error includes both truncation and finite-grid completion
    bias; it is not a claim about a continuum asymptotic rate.
    """
    if (
        isinstance(grid_size, bool)
        or not isinstance(grid_size, (int, np.integer))
        or grid_size < 21
    ):
        raise ValueError("population controls require an integer grid_size >= 21")
    nodes = np.linspace(0, 1, grid_size)
    weights = np.full(grid_size, 1 / (grid_size - 1))
    weights[[0, -1]] *= 0.5
    measure = np.outer(weights, weights)
    rows = []
    for name in PAPER_MODELS:
        kernel = paper_kernel(name, nodes)
        rank = 4 if name == "finite_rank" else 2
        for count in (3, 9):
            cliques = cover_cliques(nodes, paper_cover(count))
            mask = clique_mask(grid_size, cliques)
            result = _require_completion(
                np.where(mask, kernel, np.nan), cliques, rank=rank, support_rtol=support_rtol
            )
            difference = result.kernel - kernel
            rows.append(
                {
                    "model": name,
                    "cover": count,
                    "grid_size": grid_size,
                    "rank": rank,
                    "observed_max_abs_error": float(np.max(np.abs(difference[mask]))),
                    "missing_ise": float(np.sum(measure[~mask] * difference[~mask] ** 2)),
                    "true_min_eigenvalue": float(np.linalg.eigvalsh(kernel)[0]),
                    "completed_min_eigenvalue": float(np.linalg.eigvalsh(result.kernel)[0]),
                    "completion_diagnostics": _completion_summary(result.diagnostics),
                }
            )
    cliques = cover_cliques(nodes, paper_cover(2))
    mask = clique_mask(grid_size, cliques)
    canonical = noncanonical_control(nodes, 0)
    family = [noncanonical_control(nodes, rho) for rho in (-1, 0, 1)]
    result = _require_completion(np.where(mask, canonical, np.nan), cliques, rank=1)
    from .completion import two_block_risk_range

    exposure = np.zeros(grid_size)
    exposure[0], exposure[-1] = 1, -1
    risk_range = two_block_risk_range(
        np.where(mask, canonical, np.nan), exposure, cliques=cliques, support_rtol=support_rtol
    )
    if not risk_range.available:
        raise RuntimeError(f"reference risk range unavailable: {risk_range.diagnostics}")
    control = truncation_psd_control()
    partial = np.where(control["mask"], control["kernel"], np.nan)
    truncated = _require_completion(partial, control["cliques"], rank=1)
    exact = _require_completion(partial, control["cliques"], rank=2)
    return {
        "paper": rows,
        "noncanonical": {
            "rho_values": [-1, 0, 1],
            "observed_difference_max": float(
                max(np.max(np.abs(k[mask] - canonical[mask])) for k in family)
            ),
            "endpoint_spread_variances": [
                float(k[0, 0] + k[-1, -1] - 2 * k[0, -1]) for k in family
            ],
            "canonical_completion_error": float(np.max(np.abs(result.kernel - canonical))),
            "risk_range": {
                "lower": risk_range.lower,
                "upper": risk_range.upper,
                "canonical": risk_range.canonical,
                "identified": bool(risk_range.identified),
                "lower_witness_min_eigenvalue": float(
                    np.linalg.eigvalsh(risk_range.lower_kernel)[0]
                ),
                "upper_witness_min_eigenvalue": float(
                    np.linalg.eigvalsh(risk_range.upper_kernel)[0]
                ),
            },
            "scope": "known three-factor population family; unrestricted completion nonuniqueness",
        },
        "truncation": {
            "rank1_min_eigenvalue": float(np.linalg.eigvalsh(truncated.kernel)[0]),
            "rank2_min_eigenvalue": float(np.linalg.eigvalsh(exact.kernel)[0]),
            "rank1_missing_value": float(truncated.kernel[0, 3]),
            "rank2_missing_value": float(exact.kernel[0, 3]),
        },
        "author_reference": {
            "revision": AUTHOR_REVISION,
            "url": AUTHOR_SOURCE_URL,
            "sha256": AUTHOR_SOURCE_SHA256,
            "scope": "explicit-rank matrix recurrence only; FVE deliberately separate",
        },
    }


def export_author_fixtures(output):
    """Write CSV inputs and Python outputs for base-R explicit-rank parity.

    Uniform unit weights reproduce the author's ordinary matrix eigenproblem.
    No downloaded author code is vendored or executed here.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    nodes = np.linspace(0, 1, 41)
    fixtures = []
    for name in PAPER_MODELS:
        for count in (3, 9):
            rank = 4 if name == "finite_rank" else 2
            kernel = paper_kernel(name, nodes)
            cliques = cover_cliques(nodes, paper_cover(count))
            fixtures.append((f"{name}-m{count}", kernel, cliques, rank))
    control = truncation_psd_control()
    for rank in (1, 2):
        fixtures.append((f"psd-control-r{rank}", control["kernel"], control["cliques"], rank))
    rows = []
    for name, kernel, cliques, rank in fixtures:
        mask = clique_mask(len(kernel), cliques)
        result = _require_completion(np.where(mask, kernel, np.nan), cliques, rank=rank)
        for suffix, values in (
            ("input", np.where(mask, kernel, 0)),
            ("mask", mask.astype(int)),
            ("expected", result.kernel),
        ):
            np.savetxt(output / f"{name}-{suffix}.csv", values, delimiter=",", fmt="%.17g")
        rows.append({"name": name, "rank": rank, "joins": len(cliques) - 1})
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("name", "rank", "joins"))
        writer.writeheader()
        writer.writerows(rows)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "revision": AUTHOR_REVISION,
                "url": AUTHOR_SOURCE_URL,
                "sha256": AUTHOR_SOURCE_SHA256,
                "scope": "explicit ranks; unit grid weights",
            },
            indent=2,
        )
        + "\n"
    )
    return rows
