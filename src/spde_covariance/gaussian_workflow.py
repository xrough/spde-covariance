"""Observed-data selection for the finite-grid Gaussian completion model.

Validation scores each observed vector once with its known noise covariance.
The validation layout can include bridge observations without changing the
training model graph. Selection never tests a structural assumption in a pair
that neither training nor validation observes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .fragments import FragmentLayout
from .trajectories import InnovationBatch, _metadata_snapshot
from .workflow import ObservedInnovationBatch, interval_diagnostics, observe_innovations

if TYPE_CHECKING:
    from .kernel_workflow import KernelCovarianceFit


class GaussianCompletionConvergenceError(RuntimeError):
    """A failed local fit remains accessible for numerical audit."""

    def __init__(self, fit):
        self.fit = fit
        super().__init__(
            f"Gaussian completion did not converge: {fit.diagnostics['status']}; "
            f"relative stationarity={fit.diagnostics.get('relative_stationarity')}"
        )


@dataclass(frozen=True, slots=True)
class GaussianCompletionSelection:
    available: bool
    selected_regularization: float | None
    fit: KernelCovarianceFit | None
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", _metadata_snapshot(self.diagnostics))


def _observed(data, layout):
    if not isinstance(layout, FragmentLayout):
        raise TypeError("layout must be a FragmentLayout")
    if isinstance(data, InnovationBatch):
        data = observe_innovations(data)
    if not isinstance(data, ObservedInnovationBatch):
        raise TypeError("data must be an innovation batch with interval and basis provenance")
    layout.validate_observations(
        data.observations,
        basis_id=data.basis_id,
        require_operator_validation=True,
    )
    if not interval_diagnostics(data)["equal_intervals"]:
        raise ValueError("Gaussian completion requires equal innovation intervals")
    return data


def gaussian_kernel_validation_score(fit, data, *, layout, weights=None):
    """Mean observed Gaussian NLL, including noise, on a compatible grid."""
    from .gaussian_completion import gaussian_kernel_score

    observed = _observed(data, layout)
    if fit.grid_id != layout.grid_id or fit.metadata["basis_id"] != observed.basis_id:
        raise ValueError("fit and validation grid/basis must agree")
    if fit.metadata["covariance_unit"] != f"({observed.curve_unit})^2/year":
        raise ValueError("fit and validation covariance units must agree")
    fit_intervals = np.asarray(fit.metadata["interval_lengths"])
    if not np.isclose(observed.interval_lengths[0], fit_intervals[0], rtol=1e-10, atol=0):
        raise ValueError("fit and validation innovation intervals must agree")
    return gaussian_kernel_score(
        fit.kernel,
        tuple(item.y for item in observed.observations),
        layout.row_to_nodes,
        noise_covariances=tuple(item.noise_covariance for item in observed.observations),
        weights=weights,
    )


def select_gaussian_completion(
    training,
    validation,
    *,
    training_layout,
    validation_layout,
    reference_kernel,
    regularizations,
    covariance_assumption,
    fit_options=None,
):
    """Choose a predeclared penalty on strictly subsequent observed data.

    The reference kernel must be external or computed from training only.
    The returned fit uses training only; validation is never silently included
    in a refit. Exact score ties choose the larger penalty. Failures remain in
    the audit, and an all-failed selection returns ``available=False``.
    """
    from .kernel_workflow import fit_covariance_kernel

    if covariance_assumption != "common":
        raise ValueError("Gaussian completion selection requires common covariance")
    train, valid = _observed(training, training_layout), _observed(validation, validation_layout)
    if training_layout.grid_id != validation_layout.grid_id:
        raise ValueError("training and validation must use the same maturity grid and measure")
    if train.basis_id != valid.basis_id or train.curve_unit != valid.curve_unit:
        raise ValueError("training and validation basis/units must agree")
    if not np.array_equal(training_layout.evaluation_matrix, validation_layout.evaluation_matrix):
        raise ValueError("training and validation grid evaluation maps must agree")
    if train.times[-1] > valid.times[0]:
        raise ValueError("validation intervals must start at or after the training endpoint")
    if not np.isclose(train.interval_lengths[0], valid.interval_lengths[0], rtol=1e-10, atol=0):
        raise ValueError("training and validation interval lengths must agree")
    raw = np.asarray(regularizations)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not len(raw):
        raise ValueError("regularizations must be a nonempty sequence of positive finite numbers")
    penalties = np.asarray(raw, dtype=float)
    if (
        np.any(~np.isfinite(penalties))
        or np.any(penalties <= 0)
        or len(np.unique(penalties)) != len(penalties)
    ):
        raise ValueError("regularizations must be distinct, positive and finite")
    if fit_options is not None and not isinstance(fit_options, Mapping):
        raise TypeError("fit_options must be a mapping")
    options = {} if fit_options is None else dict(fit_options)
    if {"reference_kernel", "regularization"} & set(options):
        raise ValueError("reference_kernel and regularization have separate selection arguments")
    audits, candidates = [], []
    for penalty in sorted(penalties):
        try:
            fit = fit_covariance_kernel(
                train,
                layout=training_layout,
                method="gaussian",
                covariance_assumption="common",
                method_options={
                    **options,
                    "reference_kernel": reference_kernel,
                    "regularization": float(penalty),
                },
            )
            score = gaussian_kernel_validation_score(fit, valid, layout=validation_layout)
        except GaussianCompletionConvergenceError as error:
            audits.append(
                {
                    "regularization": float(penalty),
                    "available": False,
                    "reason": str(error),
                    "fit": error.fit.diagnostics,
                }
            )
            continue
        candidates.append((score, -float(penalty), fit))
        audits.append(
            {
                "regularization": float(penalty),
                "available": True,
                "validation_negative_loglikelihood": float(score),
                "fit": fit.diagnostics["gaussian_model"],
            }
        )
    chosen = min(candidates, key=lambda item: item[:2]) if candidates else None
    return GaussianCompletionSelection(
        bool(chosen),
        None if chosen is None else -chosen[1],
        None if chosen is None else chosen[2],
        {
            "candidates": audits,
            "training_end": float(train.times[-1]),
            "validation_start": float(valid.times[0]),
            "validation_end": float(valid.times[-1]),
            "reference_source": "caller-supplied; training-only/external provenance required",
            "selection_score": "mean per-date observed Gaussian NLL including known fresh noise",
            "tie_break": "larger regularization",
            "refitted_on_validation": False,
            "unobserved_conditional_independence_validated": False,
        },
    )
