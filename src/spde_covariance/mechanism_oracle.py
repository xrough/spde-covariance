"""Analytical controls for reconstruction/source and residual attribution.

These are two distinct synthetic experiments.  The first has an exact finite
transport trajectory and spatially perturbed quotes.  The second is a Gaussian
increment model with known correlated factor/residual blocks; it makes no claim
that its residuals arise from a level-consistent SPDE.  Neither reads SHFE data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spde_covariance.empirical_calibration import fit_cross_section, fit_source_covariance
from spde_covariance.empirical_short_history_models import (
    residual_maturity_features,
    schur_residual_forecast,
)
from spde_covariance.laguerre import LaguerreTransport


def _relative_error(estimate, target):
    return float(np.linalg.norm(estimate - target) / np.linalg.norm(target))


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "mcse": float(values.std(ddof=1) / np.sqrt(len(values))),
        "median": float(np.median(values)),
        "replications": len(values),
    }


def transport_oracle_control(*, replications=20, intervals=252, seed=20261028):
    """Separate observable reconstruction and source errors against known truth.

    Twelve quotes at fixed remaining tenors reconstruct each latent coefficient
    vector independently.  They are spatial snapshots, not rolling-contract
    returns.  Source fitting always uses the original adjacent state calendar.
    The direct-H policy averages normalized innovations of unequal intervals.
    """
    if replications < 2 or intervals < 20:
        raise ValueError("at least two replications and twenty intervals are required")
    model = LaguerreTransport(3, cache_max_bytes=0)
    source = np.array([[0.7, 0.08, 0.02], [0.08, 0.3, 0.03], [0.02, 0.03, 0.15]])
    day_counts = np.resize(np.array([1, 1, 1, 1, 3]), intervals)
    deltas = day_counts / 365.25
    times = np.r_[0.0, np.cumsum(deltas)]
    tenors = np.linspace(30, 365, 12) / 365.25
    evaluation = model.evaluate(tenors)
    risk_evaluation = model.evaluate(np.array([90, 180, 270]) / 365.25)
    weights = np.exp2(-np.arange(intervals - 1, -1, -1) / 63)
    weights /= weights.sum()
    shifts = {days: model.matrix(days / 365.25) for days in (1, 3)}
    covariances = {days: model.cumulative_covariance(source, days / 365.25) for days in (1, 3)}
    draws, coefficient_errors, numerical = defaultdict(list), [], []
    for replication in range(replications):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replication, 0]))
        states = np.zeros((intervals + 1, 3))
        states[0, 0] = 10.0
        for index, days in enumerate(day_counts):
            states[index + 1] = shifts[int(days)] @ states[index] + rng.multivariate_normal(
                np.zeros(3), covariances[int(days)]
            )
        quotes = states @ evaluation.T + rng.normal(0, 1e-4, size=(intervals + 1, 12))
        reconstructed = np.asarray(
            [
                fit_cross_section(tenors, row, dimension=3, smoothing=1e-6).coefficients
                for row in quotes
            ]
        )
        coefficient_errors.append(_relative_error(reconstructed, states))
        for representation, coefficients in (
            ("known_coefficients", states),
            ("reconstructed_coefficients", reconstructed),
        ):
            fit = fit_source_covariance(
                coefficients,
                times,
                interval_weights=weights,
                drift="zero",
                regularization_relative=1e-3,
                trace_cap_multiplier=100,
                numerical_error_budget=1e-5,
            )
            if not fit.converged:
                raise RuntimeError("synthetic source fit did not meet its numerical budget")
            numerical.append(float(fit.diagnostics["relative_covariance_error_bound"]))
            innovations = np.array(
                [
                    (right - shifts[int(days)] @ left) / np.sqrt(delta)
                    for left, right, days, delta in zip(
                        coefficients[:-1], coefficients[1:], day_counts, deltas, strict=True
                    )
                ]
            )
            empirical = innovations.T @ (weights[:, None] * innovations)
            for days in (1, 3):
                delta = days / 365.25
                target = risk_evaluation @ covariances[days] @ risk_evaluation.T
                forecasts = {
                    "source_C": risk_evaluation
                    @ model.cumulative_covariance(fit.source_covariance, delta)
                    @ risk_evaluation.T,
                    "direct_H": delta * risk_evaluation @ empirical @ risk_evaluation.T,
                }
                for policy, forecast in forecasts.items():
                    draws[(representation, policy, days)].append(_relative_error(forecast, target))
    rows = [
        {
            "representation": representation,
            "covariance_policy": policy,
            "future_calendar_days": days,
            "relative_kernel_error": _summary(values),
        }
        for (representation, policy, days), values in sorted(draws.items())
    ]
    return {
        "scope": "exact finite transport with known latent coefficients and source C",
        "settings": {
            "dimension": 3,
            "spatial_quotes": 12,
            "spatial_quote_noise_sd": 1e-4,
            "spatial_quote_policy": "fixed remaining-tenor snapshots, not contract returns",
            "calendar_day_pattern": [1, 1, 1, 1, 3],
            "intervals": intervals,
            "half_life_sessions": 63,
            "source": source.tolist(),
            "reconstruction_smoothing": 1e-6,
            "source_relative_ridge": 1e-3,
            "source_numerical_budget": 1e-5,
        },
        "coefficient_relative_error": _summary(coefficient_errors),
        "maximum_relative_covariance_certificate": max(numerical),
        "oracle_C_relative_kernel_error": 0.0,
        "rows": rows,
    }


def residual_oracle_control(*, replications=20, intervals=252, seed=20261028):
    """Known correlated Gaussian blocks demonstrate what cross terms repair."""
    if replications < 2 or intervals < 20:
        raise ValueError("at least two replications and twenty intervals are required")
    model = LaguerreTransport(3, cache_max_bytes=0)
    b = model.evaluate(np.array([90, 180, 270]) / 365.25)
    psi = residual_maturity_features(np.array([90, 180, 270]))
    h = np.diag([0.7, 0.3, 0.15])
    a = np.array([[0.3, 0.1, 0.0], [-0.2, 0.2, 0.0], [0.1, -0.1, 0.2], [0.2, 0, -0.1]])
    omega = np.diag([0.09, 0.07, 0.05, 0.03])
    g, r = h @ a.T, a @ h @ a.T + omega
    joint = np.block([[h, g], [g.T, r]])
    diagonal = np.array([0.004, 0.003, 0.002])
    truth = (b + psi @ a) @ h @ (b + psi @ a).T + psi @ omega @ psi.T + np.diag(diagonal)
    exact, _, diagnostics = schur_residual_forecast(b, psi, joint, h, diagonal=diagonal)
    exposures = {
        "outright": np.array([1.0, 0.0, 0.0]),
        "spread": np.array([1.0, -1.0, 0.0]),
        "butterfly": np.array([1.0, -2.0, 1.0]),
    }
    controls = {
        "oracle_full_joint": exact,
        "oracle_omit_cross": b @ h @ b.T + psi @ r @ psi.T + np.diag(diagonal),
        "oracle_factor_only": b @ h @ b.T,
        "oracle_no_diagonal": truth - np.diag(diagonal),
    }
    oracle_rows = [
        {
            "policy": name,
            "relative_kernel_error": _relative_error(value, truth),
            "risk_ratio": {
                exposure: float(vector @ value @ vector / (vector @ truth @ vector))
                for exposure, vector in exposures.items()
            },
        }
        for name, value in controls.items()
    ]
    errors = []
    for replication in range(replications):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replication, 1]))
        sample = rng.multivariate_normal(np.zeros(7), joint, size=intervals)
        moment = sample.T @ sample / intervals
        estimated, _, _ = schur_residual_forecast(b, psi, moment, moment[:3, :3], diagonal=diagonal)
        errors.append(_relative_error(estimated, truth))
    return {
        "scope": "Gaussian increment control; no level-consistent residual SPDE is asserted",
        "settings": {"intervals": intervals, "main_dimension": 3, "residual_features": 4},
        "oracle_identity_relative_error": _relative_error(exact, truth),
        "oracle_component_relative_error": diagnostics["component_sum_relative_residual"],
        "sampled_full_joint_relative_kernel_error": _summary(errors),
        "rows": oracle_rows,
    }


def run_oracle_controls(*, replications=20, intervals=252, seed=20261028):
    return {
        "schema": "mechanism-oracle-v1",
        "seed": seed,
        "replications": replications,
        "scope": "synthetic known-truth diagnostics, not empirical copper attribution",
        "transport": transport_oracle_control(
            replications=replications, intervals=intervals, seed=seed
        ),
        "residual": residual_oracle_control(
            replications=replications, intervals=intervals, seed=seed
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("output must be a new file")
    result = run_oracle_controls(
        replications=2 if arguments.smoke else 20,
        intervals=40 if arguments.smoke else 252,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(arguments.output)


if __name__ == "__main__":
    main()
