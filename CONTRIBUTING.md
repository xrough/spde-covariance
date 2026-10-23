# Contributing

Contributions to the MIT-licensed transport-SPDE covariance toolkit are welcome.
The finite-dimensional toy laboratory studies identification, local covariance
estimation and forecast evaluation. A [fixed infinite-rank transport benchmark](docs/infinite-transport.md)
adds coherent refinement and projection diagnostics. These implement established
methods with reproducible synthetic validation; it does not establish a novel
estimator or infinite-dimensional guarantees. Start with the [API guide](docs/api.md),
[workflow](docs/workflow.md), and [methods](docs/methods.md). The
[RQ2 protocol](docs/research-question-2.md) studies local estimation, and the
[RQ3 protocol](docs/research-question-3.md) evaluates frozen covariance forecasts
on future observations. Curve-level forecasting and market calibration remain
later application work; the [roadmap](docs/research-roadmap.md) carries the same
three questions into genuinely infinite-dimensional dynamics.

## Development and review

Use Python 3.13 and the repository's locked uv environment:

```sh
uv sync --locked --extra reference
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
uv run --no-sync python examples/transport_pipeline.py
uv run --no-sync spde-covariance benchmark experiments/configs/stage1-smoke.toml --output experiments/runs/contribution-smoke
uv run --no-sync spde-covariance benchmark-rq2 experiments/configs/rq2-smoke.toml --output experiments/runs/contribution-rq2
uv run --no-sync spde-covariance benchmark-rq3 experiments/configs/rq3-smoke.toml --output experiments/runs/contribution-rq3
uv run --no-sync spde-covariance benchmark-infinite-transport experiments/configs/infinite-transport-smoke.toml --output experiments/runs/contribution-infinite
```

The `reference` extra installs CVXPY/SCS for independent solver checks. Choose a
fresh benchmark output directory; nonempty results are never overwritten. Full
Monte Carlo studies are separate from small CI checks. Follow the root README
for any additional release checks.

Use a focused branch and pull request describing the problem, resulting behavior,
validation, and relevant issue. API changes to targets, coordinate meaning,
units, or statistical assumptions must update the corresponding documentation
and meaningful tests. Public interfaces are evolving; explain compatibility
effects instead of silently changing array interpretation.

## Numerical and statistical conventions

Use orthonormal Hilbert coordinates and preserve basis identifiers. Distinguish
instantaneous diffusion covariance `C`, normalized innovation covariance `V`,
observed covariance, and exposure risks. A projected spline shift remains a
numerical mild recursion, not an exact semigroup. Irregular intervals retain
interval-specific targets and must not be silently pooled.

Keep observed-data tuning separate from latent truth and evaluation outcomes.
Baselines receive the same information within each comparison; full-coordinate
controls and partial-observation cases have separate purposes. Declare grids,
windows, noise, caps, relative/actual penalties, and rank thresholds. Retain
paired replicates and Monte Carlo uncertainty, including cases where a simple
baseline performs better.

Forecast selection must precede testing, and issued forecasts must use only
observations through their issue origin. Keep latent truth out of fitting and
selection. Aggregate forecast dates within independent replications before
calculating Monte Carlo uncertainty. A past target-range check and the ability
to score that exposure from a future observation are different diagnostics.

Report convergence and feasibility. Solver certificates concern distance to the
constrained optimum, not estimation error to truth. Compare constrained numerical
changes with the independent reference formulation where relevant. Separate
optimization, quadrature, projection, boundary, localization, and statistical
errors. Performance claims require a recorded environment and measurement
definition; traced allocations are not process RSS.

## Method documentation, provenance, and licensing

The public project documents implemented methods, supported assumptions and
measured behavior. Keep simulation inputs, fitted targets and evaluation truth
distinct. The current fitted covariance is V_delta; diffusion covariance C and
loading L are different objects. Update the operational documentation when
changing these meanings. Cite primary sources for background methods and retain
configuration/source provenance for numerical results.

Code contributions are made under the repository's [MIT license](LICENSE).
Preserve appropriate attribution for third-party material and the permissions of
any contributed dataset. Generated runs and restricted data stay out of source
control unless a selected reproducibility artifact is explicitly reviewed.

## Equations and scientific figures

For Markdown display equations, use fenced `math` blocks as described in
[GitHub's documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
Keep inline expressions short, use explicit braces for fractions and roots,
and keep complex math out of tables and headings. Use `\mathrm{Cov}` and
`\mathrm{tr}` for the documented covariance and trace notation. Inspect the
rendered GitHub page before merging equation changes. Notebook Markdown uses
standard dollar-delimited inline and display math.

Use conventional Matplotlib scientific figures with white backgrounds, labelled
axes and units, restrained colors, readable legends and fixed comparison
scales. Static figures should stand alone; animations are supplementary.
Preserve seeds and numerical metadata when changing a figure's appearance.
