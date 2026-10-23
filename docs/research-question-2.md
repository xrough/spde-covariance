# Research question 2: local covariance and targeted risk

**How accurately can we estimate current curve-shock covariance and spread risk
when observation coverage and covariance both change over time?**

Research question 1 establishes which covariance quantities are distinguishable
under a common covariance. Question 2 adds finite sampling and temporal change:
more observations may recover missing maturity-pair information while including
older covariance regimes. This protocol studies that tradeoff in a fixed
four-dimensional space. It does not claim a new general identification theorem
or an optimal window-selection rule.

The [completed 20-replication report](../experiments/results/rq2/report.md)
retains all declared comparisons, figures and source provenance. Row-level
measurements and fitted matrices are included as gzip archives with checksums.

## Mathematical problem

In the declared orthonormal curve basis, independently across dates,

```math
\begin{aligned}
X_i&\sim N(0,V_i),\\
y_i&=O_iX_i+\varepsilon_i,\\
\varepsilon_i&\sim N(0,N_i),\\
Z_i&=y_iy_i^\top-N_i.
\end{aligned}
```

Observation maps and noise covariances are known and deterministic; noise is
independent of the innovations. The target at endpoint n is the current
finite-interval innovation covariance V_n, or the variance
θ_n(a)=aᵀV_n a of a predeclared exposure. V_i is not the instantaneous SPDE
source covariance C. The suite samples innovations directly to isolate local
estimation; Stage 1 independently checks transport extraction and the V–C
bridge.

Earlier dates provide information about V_n only through assumptions linking
covariances across time. A possible theoretical class $\mathcal{V}_\alpha(L,R)$ imposes

```math
\begin{aligned}
V_i&\succeq0,\qquad \mathrm{tr}(V_i)\le R,\\
\|V_i-V_j\|_F&\le L|t_i-t_j|^\alpha,\qquad 0<\alpha\le1.
\end{aligned}
```

The relevant loss is E[(θ̂_n−aᵀV_n a)²], potentially maximized over such a
class. The finite experiments below use explicit covariance paths rather than
claiming uniform or minimax guarantees. The abrupt path is a separate stress
case and does not satisfy a smooth covariance-path assumption across its jump.

For uniform weights on a trailing window of h observations, estimate

```math
\begin{aligned}
F_{n,h,\lambda}(B)
&=\frac12\sum_{i=n-h+1}^{n}w_i\|Z_i-O_iBO_i^\top\|_F^2
+\frac\lambda2\|B\|_F^2,\\
\widehat V_{n,h,\lambda}
&=\underset{B\succeq0,\ \mathrm{tr}(B)\le R}{\arg\min}
F_{n,h,\lambda}(B),\\
w_i&=1/h.
\end{aligned}
```

Let A_i(B)=O_iBO_iᵀ and Q=Σ_i w_i A_i* A_i. A common covariance is
identifiable when Q is injective. Its scalar risk with target T=aaᵀ is
identifiable when T lies in ran(Q). These are finite-dimensional,
common-covariance statements. A full-rank pooled Q does not identify an
unrestricted terminal covariance under a changing path.

The population unconstrained ridge solution satisfies

```math
(Q+\lambda I)(B^\circ-V_n)
=\underbrace{\sum_i w_i A_i^*A_i(V_i-V_n)}_{b_{n,h}}
-\lambda V_n.
```

The saved moment bias is ‖b_n,h‖_F. It is measured in normal-equation space,
not covariance space. With changing observation maps the population fit is
generally not the ordinary time average of V_i. PSD and trace constraints add
a further nonlinear effect.

For an identified scalar target, an alternative estimates risk directly. In
Frobenius-isometric symmetric coordinates, define

```math
\begin{aligned}
u&=Q^\dagger\mathrm{svec}(T),\\
U_i&=\mathrm{smat}(\mathbf{A}_i u),\\
\widehat\theta_{\mathrm{direct}}
&=\sum_iw_i\langle U_i,Z_i\rangle_F.
\end{aligned}
```

Here **A**_i is the matrix representation of A_i. This estimate is unbiased
under a common covariance. For a changing covariance its exact terminal bias
and Gaussian variance are

```math
\begin{aligned}
\mathrm{Bias}(\widehat\theta_{\mathrm{direct}})
 &=\sum_iw_i\langle U_i,A_i(V_i-V_n)\rangle_F,\\
\mathrm{Var}(\widehat\theta_{\mathrm{direct}})
 &=2\sum_iw_i^2\mathrm{tr}(U_i\Sigma_iU_i\Sigma_i),\\
\Sigma_i &=O_iV_iO_i^\top+N_i.
\end{aligned}
```

The analytic MSE is bias squared plus variance. A direct estimate can be
negative in a finite sample; it does not impose nonnegative scalar risk or fit
a PSD matrix. Unidentified targets are reported as unavailable.

## Predeclared experiments

The full configuration is [rq2.toml](../experiments/configs/rq2.toml); the
[smoke configuration](../experiments/configs/rq2-smoke.toml) exercises the same
pipeline at reduced replication count. Configuration values and source hashes
are recorded in every run; the full protocol is:

| Setting | Value |
| --- | --- |
| Dimension and basis | Four-mode exponential model, orthonormal in the declared H metric |
| Dates | 512 intervals, one week apart (Δ=1/52 year) |
| Replications and seed | 20 paired replications; base seed 20261023 |
| Endpoints n | 256, 272, 304, 512 |
| Trailing counts h | 16, 32, 64, 128; only observations through n |
| Base covariance | D=diag(0.4, 0.3, 0.2, 0.1) |
| Maturities | 1/12, 0.5, 1.5, 3 years; raw point evaluations |
| Observation noise | Known N_i=0.001 I for maturity observations |
| Constraint | PSD and trace at most 2 |
| Primary ridge | λ=10⁻³ ‖Q‖op, computed for each local design |
| Sensitivity ridge | 10⁻⁴ ‖Q‖op and 10⁻² ‖Q‖op, endpoint 512/window 64 |
| Numerical covariance-error budget | 10⁻⁴ for constrained fits |
| Rank threshold | Relative spectral threshold 10⁻¹⁰, reported explicitly |

The four covariance paths separate different sources of local bias:

- **Constant:** V_i=D; a control with no temporal bias.
- **Scalar:** V_i=s_i D, with s_i increasing linearly from 0.75 to 1.25.
- **Rotating:** V_i=R_i D R_iᵀ, with R_i rotating coordinates 1 and 4
  from angle 0 to π/2; trace and eigenvalues stay fixed.
- **Abrupt stress:** V_i=D through interval 256 and
  V_i=diag(0.1,0.2,0.3,0.4) afterward.

The five deterministic observation designs are compared separately:

| Design | Available data |
| --- | --- |
| `full` | All four coefficients, O_i=I, no noise; information control |
| `dense` | All four maturity evaluations jointly at every date |
| `split` | Alternating first/second and third/fourth maturity pairs |
| `joint8` | Split design with a joint observation every 8 dates |
| `joint64` | Split design with a joint observation every 64 dates |

Within each covariance-path replication, generate the full Gaussian shocks and
the dense maturity-noise draws once. Slice those same dense observations for
partial designs. This pairs observation subsets, windows, endpoints and methods;
it does not treat the noiseless identity control as an equal-information
competitor. Joint observations use their full corrected outer product, retaining
cross-pair information. Dates are numbered from 1; a joint date has its date
index divisible by its schedule period.

Declare the within-pair spread at the first/second maturities, the cross-pair
spread at the second/third maturities and the slope-neutral butterfly before
generating data. Each coefficient exposure is normalized to unit H norm.
The butterfly uses weights (12,−17,5) at maturities (1/12,0.5,1.5), so both its
weight sum and maturity-weighted sum vanish. Actual coefficient exposures and
the basis identifier are saved in the results.

Compare constrained ridge with unconstrained ridge followed by Euclidean
PSD/trace projection. Include empirical covariance only for the identity
control, and direct estimates only for targets passing the range check. All
predeclared ridge sensitivities are reported without choosing a winner using
latent covariance truth. These experiments estimate risk at the current
endpoint; they do not evaluate a forecast at a later date.

## Measurements, reporting and acceptance

The suite retains each replication and summarizes:

- Terminal covariance HS error, terminal covariance error projected through the
  local observation maps, fixed-grid kernel error, exposure-risk error and
  numerical certificates.
- Gram spectra, numerical rank, target range residuals and observability
  factors, effective sample size and observation-weighted temporal moment bias.
- Target-specific bias/MSE and their MCSE; exact direct-estimator bias,
  Gaussian variance and MSE; paired squared-error differences between methods.

MCSE quantifies simulation uncertainty in a reported mean or MSE. It is not a
confidence interval for an unknown covariance, and this experiment does not
assess confidence-interval coverage. Paired method comparisons use the MCSE of
replication-level differences. A small error at one latent truth does not imply
identifiability over a model class.

The three generated figures use fixed selections: latest endpoint for window
curves; all declared endpoint/window combinations for the joint64 coverage
comparison; and latest endpoint/window 64 (or the largest configured main
window) for direct versus covariance-based risk. Smooth-path scenarios are
separate panels; abrupt-change stress results remain separately labelled.
These selections do not depend on the realized ranking of errors.

Run acceptance checks PSD/trace feasibility, requested solver certificates,
and selected independent CVXPY/SCS references. Separate tests check expected
observation ranks and target ranges. Split pairs span 6 of
10 symmetric covariance directions; a joint date adds full rank in exact
arithmetic. Temporal tests check chronological windows and direct-risk
expectation/variance algebra. Statistical results may favor either method;
acceptance does not require a particular winner or decreasing error for every
window.

```sh
uv sync --locked --extra reference
uv run spde-covariance benchmark-rq2 experiments/configs/rq2-smoke.toml --output experiments/runs/rq2-smoke
# Use a fresh directory for the full predeclared study.
uv run spde-covariance benchmark-rq2 experiments/configs/rq2.toml --output experiments/runs/rq2
```

Each fresh output directory includes a Markdown report, `window_error.png`,
`coverage.png`, `risk_comparison.png`, machine-readable summaries and
replication measurements, and a manifest of configuration, source and solver
settings. Keep the full output together to preserve provenance.

## Relation to existing work and later questions

[Wong and Zhang, *Nonparametric Operator-Regularized Covariance Function
Estimation for Functional Data*](https://arxiv.org/abs/1701.06263) studies PSD
covariance estimation through operator regularization.
[Kurisu, *On the estimation of locally stationary functional time
series*](https://arxiv.org/abs/2105.11873) studies kernel estimation of
time-varying covariance operators.
[Waghmare and Panaretos, *The Completion of Covariance
Kernels*](https://arxiv.org/abs/2107.07350) studies covariance completion and
conditions for uniqueness from partial kernels. These are related foundations;
the suite does not claim to reproduce their estimators or theoretical results.

The next theoretical step is to relate target-specific estimation error to
both temporal regularity and local observation geometry. Later experiments can
add representation/discretization changes or recover C from V under explicit
assumptions. [Research question 3](research-question-3.md) uses strictly chronological
forecast evaluation on future observations, a separate task from
current-covariance estimation. The [roadmap](research-roadmap.md) retains all
three questions as the project progresses beyond this finite introductory model.
No market prediction claim follows from this synthetic study.
