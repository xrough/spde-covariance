"""Transport-SPDE innovation covariance in orthonormal Hilbert coordinates."""

from .laguerre import LaguerreTransport, PowerLawSource
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
    "ConvergenceError",
    "CurveTrajectory",
    "ExponentialTransport",
    "InnovationBatch",
    "InnovationCovarianceFit",
    "LaguerreTransport",
    "ObservedInnovationBatch",
    "PowerLawSource",
    "ProjectedSplineTransport",
    "TransportPropagator",
    "extract_innovations",
    "fit_innovation_covariance",
    "interval_diagnostics",
    "observe_innovations",
    "simulate_transport",
]

__version__ = "0.3.0"
