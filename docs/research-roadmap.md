# Three questions, from a toy model to curve dynamics

The public project begins with a finite-dimensional introductory laboratory.
It develops intuition, working software and numerical validation before
studying an infinite-dimensional statistical problem. Its current estimators
combine established covariance, inverse-problem and convex-optimization
methods. No new statistical estimator, new theorem or infinite-dimensional
recovery guarantee is claimed.

The three questions remain the organizing framework:

| Question | Current finite target | Continuation toward infinite dynamics |
| --- | --- | --- |
| **RQ1: Identification** | Which finite-coordinate covariance matrices and exposure risks give distinguishable observations? | Which operator quantities remain distinguishable and stably observed under increasingly rich point/delivery designs? |
| **RQ2: Current estimation** | Estimate terminal innovation covariance and declared risks under finite sampling, temporal change and changing pair coverage | Separate sampling, temporal and representation errors when spatial resolution increases |
| **RQ3: Forecast evaluation** | Score fitted covariance forecasts on strictly later observed innovations | Evaluate functional uncertainty forecasts while accounting for dynamics, observation design and approximation |

This table states directions, not results. Identifiability, conditioning,
statistical error and forecast performance are separate questions even when
the same covariance representation is used.

## The current finite pass

**Stage 1: numerical foundations.** The exact four-mode transport model supplies
clean coefficient trajectories, analytical normalized innovation covariance,
partial linear observations and independent numerical controls. The spline
backend makes projection and boundary approximations visible. The
[Stage 1 report](../experiments/results/stage1/report.md) records the evidence.

**RQ1: observation diagnostics.** Lifted Gram spectra, target range checks and
simple missing-pair examples distinguish absent from weak information.
Strictly convex regularization selects one fitted matrix; it does not establish
that observations identify every covariance direction. These are established
finite-dimensional facts implemented and tested here.

**RQ2: local estimation.** The [RQ2 protocol](research-question-2.md) isolates
changing covariance and local maturity-pair coverage by sampling Gaussian
innovations directly. It compares covariance fitting and direct risk
estimation, with latent truth excluded from fitting and selection. Its
covariance paths are finite experiments, not an asserted smoothness theorem
or an optimal window rule.

**RQ3: subsequent observations.** The [RQ3 protocol](research-question-3.md)
holds each matrix forecast fixed over a future block. Historical validation
selects window counts; future corrected outer products supply observable
scores. The [full report](../experiments/results/rq3/report.md) records 20 paired
replications, including no demonstrated advantage from validation-selected
windows over the fixed local policy. This closes a first finite cycle without
claiming a new covariance dynamics model or market forecasts.

The distinction between source covariance $C$, finite-interval innovation
covariance $V_\Delta$, and nonunique Brownian loading $L$ persists throughout.
RQ2 and RQ3 prescribe innovation-covariance paths directly; they do not recover
$C$ or assert that those paths are exact trajectories of a specified SPDE.

## What a later infinite-dimensional study must specify

A coherent spatial-refinement study starts from **one fixed underlying
Hilbert-space model**, rather than treating unrelated finite models as a
convergence experiment. It must declare the generator and semigroup, state
space and metric, diffusion covariance, initial condition, drift and
observation functionals. Increasing finite spaces then approximate that same
model, with the relation to projection and omitted tails made explicit.

Three resolution choices must remain distinct:

- Curve/Galerkin dimension controls the represented curve degrees of freedom.
- Noise-mode truncation controls retained Brownian forcing directions.
- A mollifier scale, if a future model introduces one, controls spatial
  smoothing; smoothing does not generally make covariance finite rank.

The current equality of four curve modes and four Brownian channels is a
choice of analytical control. It is not a definition of spatial resolution.
Likewise, exact transport in an invariant exponential span does not make every
finite projection an exact semigroup. Projected spline transport already
illustrates this difference.

For each research question, later work must distinguish identification of a
fixed exposure from full operator recovery. A spatially smoother covariance,
denser observations, or a larger basis does not by itself establish a useful
statistical convergence rate. Assumptions and approximation errors need to be
specified before such a result can be assessed.

## Ordered continuation and evidence gates

1. **Finite loop completed.** Preserve the existing RQ1/RQ2 controls and the
   predeclared RQ3 study as regression examples. Reports retain paired outcomes,
   including unavailable risks and non-improvements.
2. **Validate coherent refinement.** Couple approximations to one fixed curve
   model and check projection, tail, quadrature and transport errors separately.
   Keep fixed-interval innovation covariance distinct from source covariance.
   The [fixed infinite-rank transport benchmark](infinite-transport.md) implements
   this step with a Laguerre basis, unrenormalized forcing truncations, coupled
   exact finite trajectories and explicit hidden-mode leakage diagnostics.
3. **Revisit the same three questions.** Study observation stability, local
   estimation and future risk evaluation as spatial and temporal resolution
   change. New theoretical conclusions require their own assumptions and
   mathematical validation; finite numerical rank is insufficient.
4. **Introduce practical observation and dynamics models.** Sparse/noisy price
   levels, stochastic volatility, irregular sampling and futures data require
   explicit additional modelling. They are not consequences of the current
   clean-innovation experiments.

Nonlinear dynamics and stochastic volatility are optional later extensions. A
well-posed linear model with genuinely infinite-dimensional forcing already
raises the operator identification and approximation questions above.

Detailed operational mathematics stays in [methods](methods.md); callable
interfaces are in the [API](api.md). The [portfolio note](portfolio.md)
separates delivered numerical evidence from future ambitions. Public notes
describe implemented methods and reproducible protocols; later research
directions remain distinct from validated outcomes.
