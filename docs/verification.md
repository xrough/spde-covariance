# Software verification

The empirical research suite adds tests for matched-window chronology, future
price isolation, joint factor/residual cross terms, diffusion-clock separation,
independent convex hedge references, paired score identities and public/local
data boundaries. The published bundle is audited with
`python scripts/check_empirical_research.py`; optional `--verify-current-source`
also matches its recorded numerical-source hashes to the current checkout.
CI executes all six research notebooks and a synthetic suite example on Linux
and macOS, then repeats notebook/example execution from a clean core-only wheel.

The [statistical completion model](statistical-completion.md) adds independent
convex full/noiseless graphical-likelihood checks, precision-gradient finite
differences, EM monotonicity/local-stationarity checks, singular/heterogeneous
known-noise conditioning, units/permutation tests and bounded-group-cache
fallback. Workflow checks enforce observed-only chronological selection and
preserve the distinction between the declared graph and bridge observations.
Uncertainty tests check exact Gaussian probe intervals, global allocation,
heterogeneous-noise grouping and primal/dual PSD risk bounds. Equal-observed-law
controls must not become distinguishable through truth metadata or future data.
The example and `benchmark-statistical-completion` smoke run execute on both
CI platforms; the example also runs from a core-only clean wheel.

The executed [current findings notebook](../notebooks/current_findings.ipynb) reconstructs
five figures and paired numerical tables from checksummed committed evidence. Its nine
code cells run without Jupyter or optional convex solvers through `check_notebook.py`.
CI checks both notebooks from the checkout and from a core-only wheel outside it.
The runner resolves relative data from the notebook's directory and restores the caller's
working directory. The report does not rerun or silently retune the Monte Carlo studies.

The suite runs on Python 3.13. Its checks cover
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
uv run --no-sync python examples/canonical_pipeline.py
uv run --no-sync python examples/coherent_pipeline.py
uv run --no-sync python scripts/check_notebook.py
uv run --no-sync python scripts/check_notebook.py notebooks/current_findings.ipynb
uv run --no-sync spde-covariance benchmark experiments/configs/stage1-smoke.toml --output experiments/runs/verification
uv run --no-sync spde-covariance benchmark-rq2 experiments/configs/rq2-smoke.toml --output experiments/runs/verification-rq2
uv run --no-sync spde-covariance benchmark-rq3 experiments/configs/rq3-smoke.toml --output experiments/runs/verification-rq3
uv run --no-sync spde-covariance benchmark-infinite-transport experiments/configs/infinite-transport-smoke.toml --output experiments/runs/verification-infinite
uv run --no-sync spde-covariance benchmark-partial experiments/configs/partial-observations-smoke.toml --output experiments/runs/verification-partial
uv run --no-sync spde-covariance benchmark-partial experiments/configs/canonical-smoke.toml --output experiments/runs/verification-canonical
uv run --no-sync spde-covariance benchmark-partial experiments/configs/coherent-smoke.toml --output experiments/runs/verification-coherent
uv run --no-sync spde-covariance benchmark-partial experiments/configs/completion-research-smoke.toml --output experiments/runs/verification-research
uv run --no-sync spde-covariance benchmark-rq3 experiments/configs/completion-forecast-smoke.toml --output experiments/runs/verification-completion-forecast
```

The end-to-end example estimates V_delta from 1,024 weekly innovations. Its
seeded relative HS error is 0.058214 and its numerical certificate is 2.31e-8.
The transport notebook uses six executable code cells and an inspected three-panel kernel
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
The [full partial-observation report](../experiments/results/partial-observations/report.md)
records 3,600 converged fits and two independent convex comparisons from clean
source `6d2be636fe23bb5db9ad39e2296ae731abddda1a`. All finite numerical budgets
and the separately reported full-study reference-tail budgets pass. The
compressed evidence retains every candidate, selected/fixed comparison,
exposure-availability row and replication-level outcome with checksums.

The [coherent-completion follow-up](coherent-completion.md) adds a shared
observed-partial PSD fit before completion. `tests/test_coherent.py` checks
clique compatibility, weighted coordinates, primal/dual gap evaluation,
feasibility restoration, convergence failure and an optional independent
CVXPY/SCS formulation. `tests/test_coherent_controls.py` includes analytic
controls distinguishing partial PSD, completed-kernel PSD and transport-source
admissibility. Hard rank truncation can still spoil PSD after a compatible
partial fit, so it is tested separately from supported untruncated completion.

`tests/test_kernel_workflow.py` checks the `method="coherent"` adapter's
target/provenance preservation and rejection of failed partial certificates.
The low-level partial solver retains a feasible result with `converged=False`
when it misses the requested optimizer-distance budget; the high-level adapter
raises `CoherentConvergenceError` and retains that result on `.fit`. ADMM
residuals are recorded separately from the distance certificate. The bound
concerns the weighted observed-partial optimizer, not missing-entry truth or
error after completion. `tests/test_coherent_benchmark.py` checks that changing
validation observations changes selection scores without changing the
training-only candidate fits.

`tests/test_admissibility.py` checks normalized covariance-map polarization,
the adjoint for non-self-adjoint transport, weighted scale and coordinate
invariance, feasible controls, finite-source cone obstructions, immutable
results, degenerate cases and iteration-limit intervals. An optional CVXPY
comparison checks the same small constrained problem. In the actual
two-mode Laguerre control, a PSD innovation matrix has an indefinite inverse
constant source, distinguishing PSD covariance feasibility from membership
in that declared source cone.

The source diagnostic supplies lower and upper bounds on distance to a
finite, trace-capped, constant-source class. Its objective gap remains
available on an unconverged exit; reports must retain both distance endpoints,
the iteration status and the source cutoff. Positive residuals below the
solver uncertainty cannot establish separation. Integration and floating-point
cross-checks remain distinct from the convex numerical-map certificate.
Caller-supplied grid/tail allowances perturb the input kernel against the same
finite class; they do not certify exclusion from infinite-dimensional or
time-varying sources. No completed-source recovery claim follows from these
checks.

The [coherent example](../examples/coherent_pipeline.py) exercises known
transport, noisy point fragments, the compatible partial fit, canonical
completion and the finite-source diagnostic through the public API. The
`coherent-smoke.toml` command above checks the configured paired workflow and
archives certificates, selection audits and source-cone intervals. The smoke
configuration uses a deliberately smaller forcing reference than a full study;
its finite numerical acceptance is not a full-reference tail-resolution or
statistical-accuracy claim.

CI runs the numerical checks, examples, notebook, configured seeded smoke studies
and a clean wheel installation on Linux and macOS. Full Monte Carlo studies and
video encoding run separately. Selected full-study evidence records its source
revision and environment; numerical reports must be interpreted with that
provenance.

The [accepted coherent-completion study](../experiments/results/coherent-completion/report.md)
records 20 paired replications from clean numerical source
`566b07e30f30abd2edf60c8af7cf1df2d9ff2edc`. All 1,440 candidates meet the
$10^{-4}$ partial-distance budget and all 960 coherent outputs pass the PSD
check; the largest iteration count is 25,820. All twenty audit checkpoints
are bound to the parent manifest. Tests compare serial and spawned-worker
scientific outputs exactly, excluding runtime, and check failed-checkpoint
retention. These are audit records, not an automatic resume interface.
The clean wheel passes a two-worker reduced benchmark outside the checkout.
Independent replay and aggregation checks verify the reported means, paired
MCSEs and observed-validation selections; twelve unfinished population
source-cone diagnostics retain their valid intervals and explicit status.

The [continuation studies](completion-continuation-findings.md) add rectangular
source/output integration and adjoint/coordinate controls, an independent CVXPY
two-versus-three-mode source-support control, nested feasible witnesses,
fixed-source tail and numerical-map comparisons, factorial strategy reuse,
finite separator perturbation bounds, training-only split guards and
source-level dynamic forecast paths. Forecast tests change future data and
latent truth while preserving already issued fits and selections. Serial and
spawned-worker scientific results agree exactly apart from runtime.

Linux/macOS CI runs both additional smoke commands. A fresh installed wheel
outside the checkout passes all three examples and the projected-source controls
without optional CVXPY. Full studies remain separate from CI; their clean-source
manifests, solver diagnostics and independent replication checkpoints accompany
each published report. The postprocessing test verifies paired availability
and MCSE in `scripts/completion_contrasts.py`.
