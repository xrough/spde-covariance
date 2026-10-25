"""Transport-SPDE innovation covariance in orthonormal Hilbert coordinates."""

from .admissibility import AdmissibilityDiagnostic, diagnose_constant_source_admissibility
from .coherent import CoherentConvergenceError, CoherentPartialFit, fit_coherent_partial
from .fragments import FragmentLayout
from .futures import ContractSpec, SettlementChanges, SettlementPanel
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
    "ConvergenceError",
    "ContractSpec",
    "CovarianceAmbiguity",
    "CurveTrajectory",
    "ExponentialTransport",
    "FragmentLayout",
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
    "diagnose_constant_source_admissibility",
    "fit_coherent_partial",
    "fit_innovation_covariance",
    "fit_covariance_kernel",
    "interval_diagnostics",
    "observe_innovations",
    "projected_covariance_integration_matrix",
    "simulate_transport",
    "solve_robust_hedge",
]

__version__ = "0.3.0"
