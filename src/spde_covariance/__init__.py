"""Transport-SPDE innovation covariance in orthonormal Hilbert coordinates."""

from .admissibility import AdmissibilityDiagnostic, diagnose_constant_source_admissibility
from .coherent import CoherentConvergenceError, CoherentPartialFit, fit_coherent_partial
from .completion_uncertainty import (
    CompletionRiskInterval,
    GaussianCompletionRegion,
    completion_risk_interval,
    gaussian_completion_region,
)
from .fragments import FragmentLayout
from .futures import ContractSpec, SettlementChanges, SettlementPanel
from .gaussian_completion import (
    GaussianCompletionFit,
    GaussianConditional,
    condition_gaussian_kernel,
    fit_gaussian_completion,
    gaussian_kernel_score,
)
from .gaussian_workflow import (
    GaussianCompletionConvergenceError,
    GaussianCompletionSelection,
    gaussian_kernel_validation_score,
    select_gaussian_completion,
)
from .hedging import CovarianceAmbiguity, solve_robust_hedge
from .kernel_workflow import KernelCovarianceFit, fit_covariance_kernel
from .laguerre import LaguerreTransport, PowerLawSource
from .source_classes import projected_covariance_integration_matrix
from .trajectories import CurveTrajectory, InnovationBatch, extract_innovations, simulate_transport
from .transport import ExponentialTransport, ProjectedSplineTransport, TransportPropagator
from .workflow import (
    ConvergenceError,
    InnovationCovarianceFit,
    ObservedInnovationBatch,
    fit_innovation_covariance,
    interval_diagnostics,
    observe_innovations,
)

__all__ = [
    "AdmissibilityDiagnostic",
    "CoherentConvergenceError",
    "CoherentPartialFit",
    "CompletionRiskInterval",
    "ConvergenceError",
    "ContractSpec",
    "CovarianceAmbiguity",
    "CurveTrajectory",
    "ExponentialTransport",
    "FragmentLayout",
    "GaussianCompletionConvergenceError",
    "GaussianCompletionFit",
    "GaussianCompletionRegion",
    "GaussianCompletionSelection",
    "GaussianConditional",
    "InnovationBatch",
    "InnovationCovarianceFit",
    "KernelCovarianceFit",
    "LaguerreTransport",
    "ObservedInnovationBatch",
    "PowerLawSource",
    "ProjectedSplineTransport",
    "SettlementChanges",
    "SettlementPanel",
    "TransportPropagator",
    "extract_innovations",
    "completion_risk_interval",
    "condition_gaussian_kernel",
    "diagnose_constant_source_admissibility",
    "fit_coherent_partial",
    "fit_gaussian_completion",
    "fit_innovation_covariance",
    "fit_covariance_kernel",
    "interval_diagnostics",
    "gaussian_completion_region",
    "gaussian_kernel_score",
    "gaussian_kernel_validation_score",
    "observe_innovations",
    "projected_covariance_integration_matrix",
    "simulate_transport",
    "select_gaussian_completion",
    "solve_robust_hedge",
]

__version__ = "0.3.0"
