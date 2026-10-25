"""Write a deterministic true-point identification annex from committed source.

Run after committing numerical source, from the repository root:
    python examples/true_point_identification.py --output /tmp/true-point-annex-new

The output is a new raw run accepted by scripts/publish_benchmark_evidence.py.
This is a separate deterministic control, not a rerun or reinterpretation of
the projected J=8 Monte Carlo study. It needs no CVXPY or network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from spde_covariance.hedging_true_points import true_point_hedge_control
from spde_covariance.provenance import _manifest, _source_root


@dataclass(frozen=True)
class TruePointAnnexConfig:
    name: str = "true-point-identification-annex"
    source_dimension: int = 16
    reference_dimension: int = 512
    quadrature_order: int = 24
    delta: float = 1 / 52
    safety: float = 0.95
    rank_rtol: float = 1e-12
    numerical_error_budget: float = 1e-10


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _acceptance(control, tolerance):
    pair, checks = control["true_point_pair"], control["integration_diagnostics"]
    original = control["original_projected_witness"]
    conditions = {
        "finite_source_blocks_positive": min(pair["minimum_finite_source_eigenvalues"]) > 0,
        "full_source_traces_one": max(abs(value - 1) for value in pair["full_source_traces"])
        < tolerance,
        "true_point_observed_pairs_agree": pair["observed_svec_difference"] < tolerance,
        "material_hedge_gap": abs(pair["hedge_gap"]) > 1,
        "material_wrong_hedge_excess": min(pair["wrong_hedge_excess_fraction_unhedged"]) > 0.05,
        "quadrature_refinement_small": checks["map_frobenius_refinement_discrepancy"] < tolerance,
        "same_witness_refined_pairs_agree": checks["same_witness_refined_observed_difference"]
        < tolerance,
        "independent_block_check_small": max(
            checks["independent_block_finite_kernel_discrepancies"]
        )
        < tolerance,
        "fixed_tail_allowance_small_relative_to_gap": max(
            control["tail_diagnostics"]["optimal_hedge_absolute_error_bounds"]
        )
        < abs(pair["hedge_gap"]) / 1000,
        "original_projected_witness_is_not_true_point_alias": (
            original["evaluated_as_true_points"]["observed_svec_difference"] > 100 * tolerance
        ),
    }
    return {
        "passed": all(conditions.values()),
        **conditions,
        "scope": "declared deterministic numerical checks; not exact integration or statistical coverage",
    }


def _report(control, acceptance):
    pair, original = control["true_point_pair"], control["original_projected_witness"]
    projected_in_points = original["evaluated_as_true_points"]
    tails, checks = control["tail_diagnostics"], control["integration_diagnostics"]
    return f"""# True-point observation identification annex

This deterministic annex observes true point values of the full shifted curve:
the positive quadrature factors are evaluation of the Laguerre basis at
`grid+r`. It applies no output projection before evaluation. The numerical
source map integrates an unknown 16-mode source block. The common fixed
power-law source tail is evaluated through mode 512, with the omitted tail
bounded separately. The infinite-dimensional sources themselves are
`diag(q_k) +/- 0.95*embed_16(D)`, with `q_k=(k+1)^-3/zeta(3)`; prefixes are
never renormalized. Positive finite blocks and the common strictly positive
diagonal tail make both full sources PSD, trace one and infinite rank.

The seven maturities are 0, 0.5, ..., 3 years. Native fragments are indices
{{0,1,2,3}} and {{3,4,5,6}}. The exposure is node 0 minus node 1; its sole
hedge is node 5 minus node 6. A new, trace-preserving source perturbation is
constructed for this true-point observation map. The 19 observed covariance
pairs agree to svec residual {pair["observed_svec_difference"]:.6g}; the optimal
hedges are {pair["optimal_hedges"][0]:.9f} and {pair["optimal_hedges"][1]:.9f}.
Choosing the other world's hedge adds variance
{pair["wrong_hedge_excess"][0]:.9f}, or
{100 * pair["wrong_hedge_excess_fraction_unhedged"][0]:.4f}% of unhedged
target variance. These are population decisions in a deterministic control,
not sampling or market-performance results.

The original projected J=8 witness is **not** an observational alias under
true point observations: its observed svec discrepancy is
{projected_in_points["observed_svec_difference"]:.9g}, with maximum observed
entry difference {projected_in_points["observed_entry_max_difference"]:.9g}.
The original J=8 Monte Carlo study remains a projected-observation study.
This annex establishes a new pair; it does not relabel that study as observing
full-curve point values, transfer its finite-sample results, or claim uniform
identification over all infinite-dimensional sources.

The 24-to-48-point quadrature map discrepancy is
{checks["map_frobenius_refinement_discrepancy"]:.6g}; the largest independent
finite-source block-integration discrepancy is
{max(checks["independent_block_finite_kernel_discrepancies"]):.6g}.
The separate continuous-time tail allowance follows from `C_tail <= q_512 I_H`
and the translated RKHS evaluation Gram. It bounds optimal hedge errors by
{tails["optimal_hedge_absolute_error_bounds"][0]:.6g} and
{tails["optimal_hedge_absolute_error_bounds"][1]:.6g}. Formulas and residuals
are evaluated in floating point, without an exact numerical-error certificate.

Acceptance passed: `{acceptance["passed"]}`. All source matrices, factors,
kernel values, checks and thresholds are retained in [results](results.json),
with [configuration](config.json), [source provenance](manifest.json) and
[completion status](status.json).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new raw-run directory")
    arguments = parser.parse_args()
    config = TruePointAnnexConfig()
    root = _source_root()
    manifest = _manifest(config, root)
    if root is None or manifest["git_dirty"] is not False:
        raise RuntimeError(
            "commit the numerical source and run from clean source before writing evidence"
        )
    relative_script = Path(__file__).resolve().relative_to(root).as_posix()
    manifest["project_file_sha256"][relative_script] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest["solver"] = {
        "name": "deterministic whitened SVD null projection and positive Gauss quadrature",
        "rank_rtol": config.rank_rtol,
        "quadrature_order": config.quadrature_order,
        "refined_quadrature_order": 2 * config.quadrature_order,
        "uses_sdp": False,
        "numerical_check_threshold": config.numerical_error_budget,
        "exact_numerical_error_certificate": False,
    }
    manifest["experiment"] = (
        "deterministic true-point identification annex; new witness; fixed common p=3 infinite tail; no Monte Carlo"
    )
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", asdict(config))
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "status.json", {"status": "running"})
    control = true_point_hedge_control(
        source_dimension=config.source_dimension,
        reference_dimension=config.reference_dimension,
        quadrature_order=config.quadrature_order,
        delta=config.delta,
        safety=config.safety,
        rank_rtol=config.rank_rtol,
    )
    acceptance = _acceptance(control, config.numerical_error_budget)
    _write_json(output / "results.json", {"acceptance": acceptance, "control": control})
    (output / "report.md").write_text(_report(control, acceptance), encoding="utf-8")
    _write_json(output / "status.json", {"status": "complete"})
    if not acceptance["passed"]:
        raise RuntimeError(f"deterministic checks failed; diagnostics retained in {output}")
    print(output)


if __name__ == "__main__":
    main()
