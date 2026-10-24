# Transport-SPDE covariance laboratory

[![Verification](https://github.com/xrough/spde-covariance/actions/workflows/verify.yml/badge.svg)](https://github.com/xrough/spde-covariance/actions/workflows/verify.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Python research toolkit for transport-SPDE covariance. A **four-dimensional
toy laboratory** develops identification, estimation and forecast evaluation;
a fixed infinite-rank benchmark separates forcing truncation, curve projection,
time resolution and partial-observation error. Estimators use established methods. No new
estimator or infinite-dimensional statistical guarantee is claimed.

The longer-term application is futures-curve risk and prediction. Three questions
organize both the finite studies and their continuation:

| Question | Current finite study | Later direction |
| --- | --- | --- |
| **1. Identification** | Which covariance entries and exposure risks do partial observations reveal? | Observation geometry as spatial resolution increases |
| **2. Current-risk estimation** | How do sampling error, changing covariance and maturity-pair coverage interact? | Local operator/risk estimation with representation error |
| **3. Forecast evaluation** | How do estimates score on strictly subsequent observations? | Functional uncertainty forecasts under richer dynamics |

The [research roadmap](docs/research-roadmap.md) explains the progression and
the distinction between current numerical evidence and future research.

## See the toy model

![Transport-adjusted innovations and covariance estimation](assets/transport-spde-poster.png)

One seeded four-mode transport trajectory, its normalized innovations and the
estimated covariance against analytical truth. This illustration uses full,
clean coefficients, four Brownian sources and zero additional drift.

<details>
<summary>Show the animated Matplotlib figure</summary>

![Animated transport-adjusted innovations and covariance estimation](assets/transport-spde-preview.gif)

[Full-resolution MP4](assets/transport-spde.mp4) ·
[Rendering instructions](docs/visualizations.md)

</details>

## Quick start

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). No market data or GPU is needed.

```sh
git clone https://github.com/xrough/spde-covariance.git
cd spde-covariance
uv sync --locked --extra reference
uv run python examples/transport_pipeline.py
uv run pytest
```

A small executable example:

```python
import numpy as np
import spde_covariance as spde

model = spde.ExponentialTransport()
times = np.arange(513) / 52
source_C = np.diag([0.4, 0.3, 0.2, 0.1])
simulation = spde.simulate_transport(model, times, source_C, seed=1729)
innovations = spde.extract_innovations(simulation.trajectory, model)
fit = spde.fit_innovation_covariance(
    innovations,
    covariance_assumption="common",
    regularization=1e-3,
    trace_cap=2.0,
    numerical_error_budget=1e-5,
)
print(fit.covariance)  # normalized innovation covariance V_delta
```

The [full example](examples/transport_pipeline.py) also checks analytical truth;
the [notebook](notebooks/transport_covariance.ipynb) explores partial observations.

## What is estimated

For clean curve coefficients and a known transport matrix, the innovation is

```math
\xi_i=\frac{a_i-M_{\Delta_i}a_{i-1}-d_i}{\sqrt{\Delta_i}}.
```

The supplied $d_i$ is the integrated, transported drift. With deterministic
constant source covariance, the finite model gives

```math
C=LL^\top,\qquad
V_\Delta=\mathrm{Cov}(\xi_i)
=\frac{1}{\Delta}\int_0^\Delta M_r C M_r^\top dr.
```

**The fitted target is $V_\Delta$, not $C$ or $L$.** The simulator accepts $C$;
a compatible Brownian loading $L$ is nonunique. The default exact exponential
model has four Gaussian Brownian channels and no explicit spatial mollifier.
After transport adjustment, its covariance-fitting core is a generic problem
with partial linear observations.

The fitter receives $y_i=O_i\xi_i+\varepsilon_i$, known maps and fresh noise
covariances. It fits corrected outer products with PSD/trace-constrained ridge
and reports observation geometry and numerical convergence diagnostics.
The [methods](docs/methods.md) retain the full SPDE setup, Hilbert metric,
exact/projected distinction, objective and certificate formulas.

## What the numerical evidence says

The [Stage 1 report](experiments/results/stage1/report.md) uses 20 paired synthetic
replications. Relative Hilbert–Schmidt errors are Monte Carlo means ± MCSE:

| Information setting | Constrained ridge error | Numerical Gram rank |
| --- | ---: | ---: |
| Full noiseless coefficients | 0.1201 ± 0.0097 | 10/10 |
| Dense point/delivery observations | 0.3133 ± 0.0411 | 10/10 |
| Rotating sparse subsets | 0.8068 ± 0.1228 | 10/10 |
| Missing coordinate pairs | 0.1666 ± 0.0095 | 6/10 |

Full rank can coexist with weak information; modest error at one truth does not
identify missing directions. The full-data empirical baseline scores 0.1189 ±
0.0098, so ridge does not win every comparison. MCSE measures simulation precision;
optimization certificates bound numerical error to the constrained optimizer.

## Current studies and next steps

- [RQ2 protocol](docs/research-question-2.md) and [report](experiments/results/rq2/report.md):
  current covariance and risk under changing coverage, with directly sampled
  Gaussian innovations to isolate estimation.
- [RQ3 protocol](docs/research-question-3.md) and [report](experiments/results/rq3/report.md):
  20 paired replications of frozen forecasts on future observations; selected
  windows do not establish an advantage over the fixed-window policy.
- [Infinite-rank transport protocol](docs/infinite-transport.md) and [report](experiments/results/infinite-transport/report.md):
  20 coupled path replications. At 256 forcing modes, weekly covariance projection
  error falls from 1.72e-3 to 1.10e-5 at 8→64 coordinates; sampling error remains separate.
- [Partial-observation protocol](docs/partial-observations.md): fixed weekly innovations,
  shared observations across dimensions, and observed-data selection of dimension and ridge.

Current experiments assume known observation maps/noise and correct drift.
Sparse/noisy price-level reconstruction, stochastic volatility, confidence
intervals and empirical futures forecasting are not validated outcomes.
The spline backend is a projected approximation, with separate boundary and
quadrature diagnostics; dense computation targets moderate dimensions.

## Find details and reproduce

[API](docs/api.md) · [Workflow](docs/workflow.md) ·
[Methods](docs/methods.md) · [Verification](docs/verification.md) ·
[CV evidence](docs/portfolio.md) · [Contributing](CONTRIBUTING.md)

Reproduce the smoke studies:

```sh
uv run spde-covariance benchmark experiments/configs/stage1-smoke.toml --output experiments/runs/stage1-smoke
uv run spde-covariance benchmark-rq2 experiments/configs/rq2-smoke.toml --output experiments/runs/rq2-smoke
uv run spde-covariance benchmark-rq3 experiments/configs/rq3-smoke.toml --output experiments/runs/rq3-smoke
uv run spde-covariance benchmark-infinite-transport experiments/configs/infinite-transport-smoke.toml --output experiments/runs/infinite-smoke
uv run spde-covariance benchmark-partial experiments/configs/partial-observations-smoke.toml --output experiments/runs/partial-smoke
```

Protocols and the [roadmap](docs/research-roadmap.md) link reproducible evidence and next questions.
Runs preserve source/configuration manifests and machine-readable measurements.
Code, tests and analytical controls are licensed under [MIT](LICENSE).
