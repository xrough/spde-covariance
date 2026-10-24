# Software verification

The suite has 417 passing tests on Python 3.13. Its checks cover
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
uv run --no-sync spde-covariance benchmark-infinite-transport experiments/configs/infinite-transport-smoke.toml --output experiments/runs/verification-infinite
uv run --no-sync spde-covariance benchmark-partial experiments/configs/partial-observations-smoke.toml --output experiments/runs/verification-partial
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
as agree with the implementation. A reference solve that meets SCS's stopping
criterion can still miss the covariance-distance budget. The independent
reference wrapper checks that certificate after feasibility projection and, if
needed, retries SCS at tighter tolerances; it retains failure if the budget is
still unmet. Regression tests cover both refinement and exhausted retries.

The [RQ3 protocol](research-question-3.md) fixes validation and test blocks before
simulation. Future observed scores and latent simulation errors are reported
separately; neither a small observed score nor a numerical certificate proves
recovery in unobserved directions. There is no acceptance condition requiring a
particular forecast rule to win, and MCSE is not a confidence interval for an
unknown market covariance.

The clean wheel also passed the transport example and a reduced RQ3 run outside
the source checkout, using only its declared core dependencies. Optional convex
references are checked separately in the reference-enabled test environment.
The [full RQ2](../experiments/results/rq2/report.md) and
[full RQ3](../experiments/results/rq3/report.md) reports record 20 paired
replications each from clean source revision
`07c7ba0a90c332dcd74340de05f4fc2302d00b98`. Their numerical-source hashes are
retained alongside complete measurements, with large artifacts gzip compressed.

Publish a completed, accepted run into a new evidence directory with
`python scripts/publish_benchmark_evidence.py SOURCE OUTPUT`. The publisher
checks the manifest against its recorded Git revision, preserves that provenance,
and writes compressed artifacts with original and published checksums. It
refuses to overwrite a destination or publish dirty-source or failed runs.

The [infinite-rank transport benchmark](infinite-transport.md) adds nested
Laguerre coordinates, exact shift identities, elementary two-mode covariance
integration, positive quadrature versus independently scaled block
exponentials, forcing-band coupling and projected-level leakage controls.
Numerical integration discrepancies are floating-point cross-checks. Analytic
forcing-tail bounds and optimization certificates are separate quantities;
none bounds sampling error to the infinite-dimensional covariance estimator.
The smoke configuration intentionally leaves some tail-resolution flags false.
Software acceptance checks the identities, feasibility and declared numerical
tolerances; the full study must additionally resolve its final reference tails.

The [recorded full transport study](../experiments/results/infinite-transport/report.md)
passes both checks from clean source `3f573c746d720aceda8861a4655cb1f7d27ed2ef`.
Its 240 constrained fits converge within budget; the two independent convex
comparisons agree below 1.8e-15 in matrix HS norm. A clean wheel also passed the
transport example and a reduced coupled benchmark outside the checkout, using
only declared core dependencies. The wheel's Python sources match the recorded
numerical implementation.

The [partial-observation study](partial-observations.md) adds common parent
observations across candidate dimensions, covariance omission/cross-term
checks, full-H contract geometry, exact fresh-noise correction, past-only
selection and test isolation. Independent replication-level paired summaries
retain unavailable direct-risk proxies and separate them from finite-sieve
range checks. Full-study reference-tail acceptance remains separate from the
smoke run's intentionally coarse forcing approximation.
A clean wheel passed a reduced two-source partial-observation run outside the
checkout using only its declared core dependencies; optional reference solves
were verified separately in the reference-enabled test environment.

CI runs the numerical checks, example, notebook, all five seeded smoke suites
and a clean wheel installation on Linux and macOS. Full Monte Carlo studies and
video encoding run separately. Selected full-study evidence records its source
revision and environment; numerical reports must be interpreted with that
provenance.
