# Software verification

The code-focused tree passes 159 automated checks on Python 3.13. They cover
transport and analytical covariance agreement, deterministic/drift cancellation,
coordinate invariance, irregular-time handling, observation adjoints, symmetric
vectorization, PSD/trace feasibility, numerical certificates, CVXPY/SCS reference
agreement, observed-data tuning and seeded reproducibility.

The Gaussian observation simulator and the shared source/environment manifest
remain independently checked. The benchmark CLI operates without research
drafts or a document compiler. Research-specific experiments are outside the
public software distribution.

Reproduce the software checks with:

```sh
uv sync --locked --extra reference
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python examples/transport_pipeline.py
uv run --no-sync python scripts/check_notebook.py
uv run --no-sync spde-covariance benchmark experiments/configs/stage1-smoke.toml --output experiments/runs/verification
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

CI runs the numerical checks, example, notebook, seeded benchmark and a clean
wheel installation on Linux and macOS. Full Monte Carlo studies and video
encoding run separately.
