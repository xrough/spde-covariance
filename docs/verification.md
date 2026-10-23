# Software verification

The automated checks on Python 3.13 cover
transport and analytical covariance agreement, deterministic/drift cancellation,
coordinate invariance, irregular-time handling, observation adjoints, symmetric
vectorization, PSD/trace feasibility, numerical certificates, CVXPY/SCS reference
agreement, observed-data tuning and seeded reproducibility. The RQ2 checks add
paired observation coverage, time-varying covariance controls, exact Gaussian
direct-risk bias/variance, chronological windows, Monte Carlo aggregation and
failed/interrupted-run handling.

RQ3 adds forecast-origin boundaries, past-only validation and refitting,
future-observation and latent-truth isolation, deterministic window selection,
noise-corrected forecast scores, exposure scoring availability and independent
replication-level aggregation. Its three forecast rules use the existing
constrained estimator rather than introducing a new statistical method.

The Gaussian observation simulator and the shared source/environment manifest
remain independently checked. The benchmark CLI operates without research
drafts or a document compiler. The public RQ2 suite directly studies local
innovation-covariance estimation; original mathematical drafts remain separate.

Reproduce the software checks with:

```sh
uv sync --locked --extra reference
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python examples/transport_pipeline.py
uv run --no-sync python scripts/check_notebook.py
uv run --no-sync spde-covariance benchmark experiments/configs/stage1-smoke.toml --output experiments/runs/verification
uv run --no-sync spde-covariance benchmark-rq2 experiments/configs/rq2-smoke.toml --output experiments/runs/verification-rq2
uv run --no-sync spde-covariance benchmark-rq3 experiments/configs/rq3-smoke.toml --output experiments/runs/verification-rq3
```

The end-to-end example estimates V_delta from 1,024 weekly innovations. Its
seeded relative HS error is 0.058214 and its numerical certificate is 2.31e-8.
The notebook uses six executable code cells and an inspected three-panel kernel
comparison. The animation renderers use the same installed package and standard
scientific plotting tools; their parameters and output formats are recorded in
[visualization metadata](visualizations.md).

The [Monte Carlo report](../experiments/results/stage1/report.md) separately
records estimation accuracy. Its manifest states the source and environment
that generated the results. Numerical solver certificates concern distance to
the constrained optimizer, rather than statistical error against truth.

The [RQ2 protocol](research-question-2.md) records the new statistical experiment.
Its smoke study also passed from an installed wheel outside the source checkout,
using the locked environment's NumPy/SciPy/Matplotlib dependencies. Small
CVXPY/SCS comparisons must converge and meet the numerical-error budget as well
as agree with the implementation.

The [RQ3 protocol](research-question-3.md) fixes validation and test blocks before
simulation. Future observed scores and latent simulation errors are reported
separately; neither a small observed score nor a numerical certificate proves
recovery in unobserved directions. There is no acceptance condition requiring a
particular forecast rule to win, and MCSE is not a confidence interval for an
unknown market covariance.

CI runs the numerical checks, example, notebook, all three seeded smoke suites
and a clean wheel installation on Linux and macOS. Full Monte Carlo studies and
video encoding run separately. Selected full-study evidence records its source
revision and environment; numerical reports must be interpreted with that
provenance.
