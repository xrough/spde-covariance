"""Aggregate post hoc diagnostics for reconstructed same-contract risk.

These diagnostics use realized future prices and reconstructed future states.
They issue no forecasts, fit no parameters, and return no individual quotes,
contract identifiers, dates, coefficients or outcomes.
"""

from __future__ import annotations

import numpy as np

from .laguerre import LaguerreTransport


def _section(section, *, maturity):
    """Validate the array contract used by ``prepare_daily_curves``."""
    ids = np.asarray(section["ids"])
    raw = np.asarray(section["prices"])
    if ids.ndim != 1 or raw.shape != ids.shape or raw.dtype.kind not in "iuf":
        raise ValueError("section ids and real prices must be matching vectors")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("section contract ids must be unique")
    result = {"ids": ids, "prices": raw.astype(float)}
    if maturity:
        dates = np.asarray(section["maturity"], dtype="datetime64[D]")
        if dates.shape != ids.shape or np.any(np.isnat(dates)):
            raise ValueError("section maturity dates must match contract ids")
        result["maturity"] = dates
    return result


def reconstruction_risk_decomposition(daily, config):
    """Decompose realized cash second moments on the frozen test schedule.

    Contracts are selected once from the eligible origin section, using the
    study's selection rule. Their next settlements are looked up by the same
    identifiers in ``settlement_sections``, including quotes that cease to
    satisfy reconstruction liquidity filters. Missing legs invalidate only
    exposures holding those legs. Both endpoint coefficient vectors must be
    available; no state is refitted or filled.

    For interval length ``dt`` in ACT/365.25 years, let ``r = a_next-M_dt a``.
    The modeled contract change is ``E_end r``. Exact restricted transport
    gives ``E_end M_dt = E_origin``, so the remaining change is the difference
    of endpoint reconstruction residuals for these same contracts. Cash
    changes satisfy ``Y=S+U`` and ``E[Y²]=E[S²]+E[U²]+2E[SU]`` empirically.
    All moments are uncentered sample averages in squared CNY, including the
    configured contract multiplier. The residual is not assumed independent,
    fresh noise; its cross share may be negative and component shares may
    exceed one. Shares are unavailable when the actual second moment is zero.

    The diagnostic includes eligible test origins independently of covariance
    fit convergence. It does not change or condition on forecast availability.
    ``max_identity_error`` is the largest per-origin squared-CNY arithmetic
    discrepancy; ``identity_relative_error`` checks the aggregate identity
    relative to its absolute component scale.
    """
    # Deferred because the study runner imports this helper at module scope.
    from .empirical_study import EXPOSURES, select_contracts

    dates = np.asarray(daily["dates"], dtype="datetime64[D]")
    raw_coefficients = np.asarray(daily["coefficients"])
    if dates.ndim != 1 or np.any(np.isnat(dates)) or np.any(np.diff(dates).astype(int) <= 0):
        raise ValueError("daily dates must be strictly increasing finite session dates")
    if (
        raw_coefficients.ndim != 2
        or raw_coefficients.shape[0] != len(dates)
        or raw_coefficients.shape[1] < 1
        or raw_coefficients.dtype.kind not in "iuf"
        or np.any(np.isinf(raw_coefficients))
    ):
        raise ValueError("daily coefficients must be a real session-by-dimension array")
    if len(daily["sections"]) != len(dates) or len(daily["settlement_sections"]) != len(dates):
        raise ValueError("daily sections must match the session dates")
    coefficients = raw_coefficients.astype(float)
    model = LaguerreTransport(coefficients.shape[1])
    values = {name: [] for name in EXPOSURES}
    unavailable = {name: {} for name in EXPOSURES}
    basis_errors, reconstruction_errors = [], []
    candidate_count = 0

    def omit(reason, names=EXPOSURES):
        for name in names:
            counts = unavailable[name]
            counts[reason] = counts.get(reason, 0) + 1

    for i in range(max(config.windows), len(dates) - 1, config.origin_stride):
        origin, end = dates[i : i + 2]
        if origin <= np.datetime64(config.validation_end) or end > np.datetime64(config.test_end):
            continue
        candidate_count += 1
        chosen = select_contracts(
            _section(daily["sections"][i], maturity=True), origin, config.target_tenors_days
        )
        if chosen is None:
            omit("origin_contract_selection_unavailable")
            continue
        if not np.all(np.isfinite(coefficients[i : i + 2])):
            omit("endpoint_reconstruction_unavailable")
            continue
        end_tenors = (chosen["maturity"] - end).astype(int) / 365.25
        if np.any(end_tenors < 0):
            omit("selected_contract_expired")
            continue
        dt = float((end - origin).astype(int)) / 365.25
        matrix = model.matrix(dt)
        end_evaluation = model.evaluate(end_tenors)
        origin_evaluation = model.evaluate((chosen["maturity"] - origin).astype(int) / 365.25)
        basis_errors.append(float(np.max(np.abs(end_evaluation @ matrix - origin_evaluation))))
        coefficient_residual = coefficients[i + 1] - matrix @ coefficients[i]
        modeled_changes = end_evaluation @ coefficient_residual
        reconstructed_changes = (
            end_evaluation @ coefficients[i + 1] - origin_evaluation @ coefficients[i]
        )
        reconstruction_errors.append(float(np.max(np.abs(modeled_changes - reconstructed_changes))))
        future = _section(daily["settlement_sections"][i + 1], maturity=False)
        lookup = dict(zip(future["ids"], future["prices"], strict=True))
        actual_changes = (
            np.array([lookup.get(key, np.nan) for key in chosen["ids"]]) - chosen["prices"]
        )
        for name, exposure in EXPOSURES.items():
            support = exposure != 0
            if not np.all(np.isfinite(actual_changes[support])):
                omit("held_contract_settlement_unavailable", (name,))
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                actual = float(config.multiplier * exposure[support] @ actual_changes[support])
                modeled = float(config.multiplier * exposure[support] @ modeled_changes[support])
                residual = actual - modeled
                moments = np.array([actual**2, modeled**2, residual**2, 2 * modeled * residual])
            if not np.all(np.isfinite(moments)):
                omit("nonfinite_decomposition", (name,))
                continue
            values[name].append(moments)

    rows = []
    for name, observations in values.items():
        count = len(observations)
        row = {
            "exposure": name,
            "count": count,
            "unavailable_count": candidate_count - count,
            "unavailable_reasons": unavailable[name],
            **dict.fromkeys(
                (
                    "actual_second_moment",
                    "modeled_second_moment",
                    "residual_second_moment",
                    "twice_cross_moment",
                    "modeled_share",
                    "residual_share",
                    "cross_share",
                    "max_identity_error",
                    "identity_relative_error",
                )
            ),
        }
        if count:
            sample = np.asarray(observations)
            # Dividing before summing also avoids unnecessary moment-sum overflow.
            actual, modeled, residual, cross = np.sum(sample / count, axis=0)
            row.update(
                actual_second_moment=float(actual),
                modeled_second_moment=float(modeled),
                residual_second_moment=float(residual),
                twice_cross_moment=float(cross),
                max_identity_error=float(
                    np.max(np.abs(sample[:, 0] - sample[:, 1] - sample[:, 2] - sample[:, 3]))
                ),
            )
            if actual > 0:
                row.update(
                    modeled_share=float(modeled / actual),
                    residual_share=float(residual / actual),
                    cross_share=float(cross / actual),
                )
            scale = max(actual, modeled + residual + abs(cross))
            row["identity_relative_error"] = (
                float(abs(actual - modeled - residual - cross) / scale) if scale > 0 else 0.0
            )
        rows.append(row)
    return {
        "scope": {
            "purpose": "post_hoc_reconstruction_diagnostic_no_forecast_refit",
            "candidate_origin_count": candidate_count,
            "dimension": model.dimension,
            "origin_schedule": "frozen_test_origins_independent_of_covariance_fit_availability",
            "portfolio_selection": "origin_eligible_contracts_held_by_same_id_to_next_session",
            "interval_convention": "ACT/365.25",
            "moments": "uncentered_cash_second_moments",
            "units": "CNY_squared",
            "multiplier": float(config.multiplier),
            "residual_interpretation": "endpoint_reconstruction_error_change_not_fresh_noise",
            "basis_identity_max_abs_error": max(basis_errors, default=None),
            "reconstruction_change_max_abs_error_cny_per_tonne": max(
                reconstruction_errors, default=None
            ),
        },
        "exposures": rows,
    }
