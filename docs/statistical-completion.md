# Completion as a statistical model

This increment makes the finite-grid completion assumption a fitted probability
model. It prepares calibration and risk applications without claiming an
empirical futures model, a new estimator or a continuum statistical guarantee.
The historical moment-based and raw completion methods remain available.

## Model and estimand

For one common innovation interval, let $X_i$ be the latent innovation values
at the declared maturity nodes. The statistical model is

```math
X_i\sim N(0,K),\qquad Y_i=P_iX_i+\epsilon_i,
\qquad\epsilon_i\sim N(0,N_i).
```

The observations are independent across dates, centered with known zero mean,
and have known fresh Gaussian noise independent of the latent innovation.
Each $P_i$ selects the actually observed
nodes, without filling missing values. The noise can be correlated within a
date and differ between dates. Price-level differencing does not automatically
produce this model. Drift, serial dependence and unknown noise require later
extensions.

The completion model declares a decomposable precision graph:

```math
Q=K^{-1}\succ0,\qquad Q_{ab}=0\quad\text{outside the declared graph.}
```

For a serrated cover, this is the positive-definite finite-grid Gaussian
conditional-independence interpretation of canonical completion. The **model
graph and observation catalogue are separate**. A bridge observation can test
the model without automatically replacing its conditional-independence
assumption. Singular population completions remain supported by the older
algebraic methods; this likelihood model uses positive-definite signal kernels.

K is a point-kernel covariance in curve-squared/year. It is neither an
orthonormal H-coordinate matrix nor the instantaneous source C. Quadrature
weights define kernel evaluation metrics; they do not make individual pairs
independent likelihood observations. Gaussian conditional distributions are
available for missing innovation values, conditional on the fitted kernel and declared model;
they do not reconstruct sparse noisy price-level trajectories.

## Observed likelihood and regularization

With normalized date weights $p_i$, the mean negative log likelihood is

```math
\mathcal L(K)=\frac12\sum_i p_i
\left[\log\det\Sigma_i+Y_i^\top\Sigma_i^{-1}Y_i
+m_i\log(2\pi)\right],\qquad
\Sigma_i=P_iKP_i^\top+N_i.
```

Each observed vector contributes once. The fitter uses raw observations and
their own noise covariances. Noise-corrected outer products are valid for the
moment estimator, but are not substituted for likelihood observations.

An explicit positive-definite reference kernel T and positive penalty lambda
define

```math
\mathcal L(K)+\frac\lambda2
\left[\mathrm{tr}(TQ)-\log\det(TQ)-d\right].
```

The bracket is twice the Gaussian KL divergence from covariance T to K.
It is a declared shrinkage assumption. T must be supplied externally or
estimated from training data only. A fixed positive lambda changes the
population estimand. The benchmark uses predeclared pseudo-counts divided by
training count, and selects among them using subsequent observed validation
data; this is a practical policy, not a consistency theorem.

The EM E-step computes the complete latent second moment conditional on the
actual observations. Its covariance calculation uses a Joseph form to avoid
subtracting nearly equal covariance matrices. The M-step solves the
decomposable graphical Gaussian problem for

```math
S_{\mathrm{eff}}=
\frac{\sum_i p_i\,E[X_iX_i^\top\mid Y_i]+\lambda T}{1+\lambda}.
```

All supported positive directions are retained; no rank truncation, diagonal
jitter or completed-kernel PSD repair is silently introduced. The fit records
penalized-objective histories, local score residuals, conditioning, attempted
initializations and explicit failures. Partial/noisy observed likelihood is
not asserted to be globally convex. EM convergence is a local numerical
statement, distinct from the global certificate of the moment solver.

For comparison, the coherent moment solver also accepts a supplied shrinkage
reference. Its historical default remains zero. Targeted Frobenius and KL
penalties are different objectives even when they use the same T and candidate
pseudo-counts; their comparison does not isolate likelihood alone.

## Selection, prediction and uncertainty

`fit_covariance_kernel(..., method="gaussian", ...)` preserves existing
innovation, interval, basis and grid conventions. It requires explicit
`reference_kernel` and `regularization`. `model_cliques` can declare a graph
different from the layout's catalogue. Nonconvergence raises an informative
error retaining the numerical fit.

`select_gaussian_completion` chooses a penalty using strictly later observed
validation data. It returns a training-only fit and records all candidates;
it does not silently refit validation observations. Exact score ties prefer
the larger penalty. Validation can assess observed predictions, but cannot
validate a completion convention in pairs that remain unseen.

`condition_gaussian_kernel` returns latent conditional means/covariances for
specified nodes given a noisy fragment. These are model-conditional prediction
quantities, excluding parameter uncertainty; they are not sampling confidence intervals for K.

The uncertainty companion uses centered Gaussian variance chi-square intervals,
polarization and one Bonferroni allocation over scalar probes and known-noise
groups. It retains negative corrected bounds. The resulting partial-kernel
region does **not** impose the canonical graph. It admits all PSD completions
consistent with those bands, with a trace upper bound derived from the
observed diagonal bands rather than an extra source prior.

Exposure-risk ranges propagate this region through PSD optimization. Numerical
outer endpoints use repaired feasible dual bounds; primal endpoint witnesses,
gaps and feasibility checks remain available. Floating-point checks are not
interval-arithmetic proofs. An unavailable or infeasible numerical result is
retained rather than replaced by a successful-looking interval.

Sampling confidence for the partial kernel and ambiguity among its completions
remain separate. The fitted Markov kernel need not lie inside a finite-sample
region after penalization. A good native likelihood and a narrow conditional
prediction distribution do not identify never-observed covariances.

## Evidence and acceptance

Run the offline example and a new configured study directory:

```sh
uv sync --locked --extra reference
uv run python examples/statistical_completion.py
uv run spde-covariance benchmark-statistical-completion experiments/configs/statistical-completion-smoke.toml --output /tmp/statistical-completion-smoke-new
uv run spde-covariance benchmark-statistical-completion experiments/configs/statistical-completion.toml --output experiments/runs/statistical-completion-new
```

The [published numerical report](../experiments/results/statistical-completion/report.md)
retains the clean-source manifest, selections, conditional-model diagnostics,
failed candidates, all point-study rows and coverage strata. The report does
not turn a successful numerical fit into empirical calibration evidence.

The fixed seven-maturity study uses the verified true-point SPDE source pair,
which shares native fragment laws but has different missing covariance, and
a separate correctly specified Gaussian canonical control. Training, observed
validation and untouched test blocks are disjoint. Full-panel test observations
are explicitly a bridge audit; they are not available for native tuning.

The report separates observed/missing kernel error, native/bridge likelihood,
quadratic scores, exposure risks, model bias, numerical failures and paired
Monte Carlo uncertainty. A separate small-grid study assesses uncertainty
coverage with 500 independent replications per design/sample-size stratum.
Nested counts 128, 512 and 2048 share parent observations. Complete views of
the same parents provide a positive identification control. Native intervals
cover both equal-observed-law worlds; complete observations have different
laws and are evaluated against their own generating covariance. Monte Carlo
errors remain per stratum, without pooling dependent counts or designs.

Acceptance requires independent convex full/noiseless controls, conditional
Gaussian identities, graph/units invariance, explicit local convergence,
chronological selection and truth isolation. It also requires consistent
inference across equal-observed-law worlds and transparent bridge-audit
misspecification. Neither universal estimator superiority nor native recovery
of unidentified risks is required.

The mathematical ingredients are established: [Waghmare–Panaretos](https://arxiv.org/abs/2107.07350),
[EM](https://academic.oup.com/jrsssb/article/39/1/1/7027539), and
[decomposable Gaussian covariance estimation](https://arxiv.org/abs/0903.1283).
The contribution here is an integrated, auditable statistical workflow and
numerical evidence; later claims require their own assumptions and results.
