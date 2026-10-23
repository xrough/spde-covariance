# Model, estimation target, and numerical method

This note describes the mathematical model implemented by the public package,
the information supplied to its estimators, and the interpretation of its
diagnostics. The fitted object is a finite-dimensional, normalized
transport-innovation covariance. The continuous curve model motivates that
object; the package does not reconstruct an unrestricted covariance operator
from sparse price levels.

## 1. Curve transport and the coordinate model

Let $x \geq 0$ denote time to maturity and let $t$ denote calendar time, both
measured in years. A transport SPDE with a finite Brownian source has the form

```math
df_t(x)
= \left(\partial_x f_t(x)+\beta_t(x)\right)dt
+ \sum_{k=1}^{m}\ell_k(x)dW_t^k.
```

Here $W^1,\ldots,W^m$ are independent standard Brownian motions. The shift
$S_r f(x)=f(x+r)$ accounts for the passage of time along a maturity curve.
The functions $\ell_k$ are spatial loadings: the same Brownian increment can
move many maturities together.

The implementation represents a curve in an orthonormal basis of a chosen
$J$-dimensional subspace of a weighted Sobolev space:

```math
\begin{aligned}
\langle f,g\rangle_H
&= f(0)g(0)
+ \int_0^\infty e^{x/\tau}f'(x)g'(x)dx,\\
f_t(x)&=\phi(x)^\top a_t,\\
\phi(x)&=\begin{pmatrix}\phi_1(x)\\ \vdots\\ \phi_J(x)\end{pmatrix}.
\end{aligned}
```

The functions are absolutely continuous with finite displayed norm. The
default scale is $\tau=1$ year. The anchor at zero and the derivative weight
are part of the model geometry, rather than an empirical choice of an
ordinary Euclidean maturity grid. Orthonormal coordinates give
$\|f_t\|_H=\|a_t\|_2$. A covariance matrix in these coordinates therefore has
Frobenius norm equal to the Hilbert--Schmidt norm of its finite-rank operator.

When the chosen span is shift invariant and the initial curve, drift and
loadings lie in that span, the curve equation has the exact coordinate version

```math
\begin{aligned}
da_t &= (A a_t+b_t)dt+L dW_t,\\
M(r)&=e^{rA},\\
C&=LL^\top.
\end{aligned}
```

The columns of $L$ contain the coefficients of the loading functions.
$C$ is the instantaneous covariance of the Brownian forcing in the declared
basis. It differs from the covariance of a finite-interval innovation below.
The simulation API takes $C$, rather than separately fitted loading functions.
For example, replacing $L$ by $LU$ for an orthogonal matrix $U$ leaves $C$
unchanged. Covariance data alone do not select a unique loading factor.

The default analytical control has $J=m=4$ and four independent Brownian
sources, with deterministic, time-constant $C$. Its spatial correlation comes
from the finite loading functions and their covariance. It does not introduce
an explicit spatial mollifier or an infinite-dimensional spatial white-noise
source.

## 2. Exact exponential transport

`ExponentialTransport` uses raw functions $g_j(x)=e^{-\rho_j x}$. Distinct
rates satisfying $\rho_j>1/(2\tau)$ ensure finite weighted derivative energy.
Their Gram matrix and the orthonormalization are

```math
\begin{aligned}
G_{jk}&=1+\frac{\rho_j\rho_k}{\rho_j+\rho_k-1/\tau},\\
G&=KK^\top,\qquad T=K^{-\top},\\
\phi(x)^\top&=g(x)^\top T.
\end{aligned}
```

$K$ here is a Cholesky factor of the basis Gram matrix, not a Brownian loading
matrix. The transport matrix in orthonormal coordinates is

```math
M(r)=T^{-1}
\mathrm{diag}\left(e^{-\rho_1 r},\ldots,e^{-\rho_J r}\right)T.
```

The finite span is invariant under the shift, so this backend has the
semigroup property. Its matrix is generally nonnormal in the orthonormal
$H$ coordinates; using a diagonal matrix in those coordinates would change
the model. The implementation checks rate admissibility and rejects an
excessively ill-conditioned raw Gram matrix before constructing the basis.
These controls make the finite model exact; they do not make an arbitrary
real curve an exact member of that model.

See [transport.py](../src/spde_covariance/transport.py) for the formulas and
[spaces.py](../src/spde_covariance/spaces.py) for the alternative spline space.

## 3. Mild increments and the covariance being estimated

For endpoint times $t_i<t_{i+1}$, put $\Delta_i=t_{i+1}-t_i$. With the correct
integrated, transported drift contribution, the mild recursion is

```math
\begin{aligned}
a_{i+1}&=M(\Delta_i)a_i+d_i+\sqrt{\Delta_i}\xi_i,\\
d_i&=\int_{t_i}^{t_{i+1}}M(t_{i+1}-s)b_s ds,\\
\xi_i&=\frac{a_{i+1}-M(\Delta_i)a_i-d_i}{\sqrt{\Delta_i}}.
\end{aligned}
```

`extract_innovations` starts from clean coefficient trajectories. Its drift
argument is $d_i$, already integrated and transported in the same basis.
Omitting it declares zero mild drift. It does not estimate an unknown drift
or justify ignoring one. A wrong drift can give the residual a nonzero mean
and contaminate its outer products.

For deterministic, constant $L$, the innovations over disjoint intervals are
independent centered Gaussian vectors. If the interval length is $\Delta$,
their covariance is

```math
\begin{aligned}
\xi_i&=\frac{1}{\sqrt{\Delta}}
\int_{t_i}^{t_{i+1}}M(t_{i+1}-s)LdW_s,\\
V_\Delta&=\mathrm{Cov}(\xi_i)\\
&=\frac{1}{\Delta}\int_0^\Delta M(r)C M(r)^\top dr.
\end{aligned}
```

Thus $V_\Delta$ is a transport average of $C$, not $C$ itself. Normalization
removes the leading interval-length scale but does not remove transport.
In this finite continuous model $V_\Delta$ tends to $C$ as $\Delta$ tends to
zero. The package's fitted output is $V_\Delta$ at the declared interval,
not an estimate of $C$ or of $L$.

For the exponential backend the integral is evaluated analytically. Writing
$\widetilde{C}=TCT^\top$ for the raw-basis coefficient covariance gives

```math
\begin{aligned}
(H_\Delta)_{jk}
&=\frac{1-e^{-(\rho_j+\rho_k)\Delta}}{(\rho_j+\rho_k)\Delta},\\
V_\Delta&=T^{-1}
\left(\widetilde{C}\odot H_\Delta\right)T^{-\top}.
\end{aligned}
```

where $\odot$ is entrywise multiplication. The code uses `expm1` to evaluate
the small-interval numerator accurately. Independent quadrature checks this
analytical control in the benchmark.

Times, basis identity, propagation method and original endpoints remain in
the trajectory and innovation objects. Innovations have units of curve units
per square root of a year; $C$ and $V_\Delta$ have units of curve units squared
per year. Irregular intervals retain different targets $V_{\Delta_i}$ even
when $C$ is constant. Extraction supports those intervals, but the current
high-level common-target fitter rejects pooling them.

With deterministic diffusion held constant separately on each interval, the
same formula applies with that interval's $C_i$. A fit across changing
covariances needs an explicit approximation assumption: `"common"` declares
one $V_\Delta$ across the weighted observations; `"local"` treats the selected
window as an approximation to its terminal interval's covariance. The code
records the choice and does not establish stationarity or a localization
error bound.

## 4. Projected spline transport

`SplineSpace` uses clamped cubic B-splines on a finite padded domain
$[0,\ell]$, orthonormalized in the same $H$ metric. Each raw spline is extended
continuously and constantly beyond $\ell$, so its derivative is zero on that
tail. The construction does not impose a zero endpoint derivative on every
function in the span.

If $P_J$ is the $H$-orthogonal projection onto this span, the spline transport
matrix represents

```math
\begin{aligned}
M_J(r)&=P_J S_r P_J,\\
V_{\Delta,J}
&=\frac{1}{\Delta}\int_0^\Delta M_J(r)C_J M_J(r)^\top dr.
\end{aligned}
```

In general, $M_J(r+s)$ differs from $M_J(r)M_J(s)$. Consequently the spline
simulation is labelled `projected_mild_recursion`: each step uses the
declared projected transport and Gaussian innovation covariance. It is not
labelled an exact trajectory of the infinite-dimensional SPDE. Quadrature
is split at relevant spline breakpoints and shift-induced breakpoints.

Refining the basis at fixed padding, enlarging the domain at fixed basis
size, and increasing quadrature accuracy address different errors. A
constant-tail approximation cannot reproduce derivative energy of a reference
curve beyond $\ell$; at fixed padding this gives a tail error floor. More
padding can also worsen interior resolution at fixed $J$. The stress suite
reports projection, omitted-tail, shift, quadrature and semigroup-defect
measurements separately, without asserting uniform convergence from a short
dimension ladder.

## 5. Observations and corrected covariance measurements

The estimator receives one `Observation` per interval:

```math
\begin{aligned}
y_i&=O_i\xi_i+\varepsilon_i,\\
\mathbb{E}[\varepsilon_i]&=0,\qquad
\mathrm{Cov}(\varepsilon_i)=N_i,\\
Z_i&=y_i y_i^\top-N_i.
\end{aligned}
```

The benchmark uses deterministic observation maps, fresh Gaussian noise
independent of the innovation and other dates, and known noise covariances.
Under these assumptions and a common centered innovation covariance,

```math
\mathbb{E}[Z_i]=O_i V_\Delta O_i^\top.
```

The moment identity only needs the corresponding centered second-moment and
uncorrelated-noise assumptions; Gaussianity supplies the particular synthetic
sampling model. The package does not infer those assumptions from the data.
Corrected measurements can have negative eigenvalues. They are retained
without a positivity projection, which would change their expectation.

Rows of $O_i$ may represent points, delivery averages, or other supplied
linear functionals. A point at $x$ has row $\phi(x)^\top$; a uniform delivery
average on $[u,v]$ has row

```math
\frac{1}{v-u}\int_u^v\phi(x)^\top dx.
```

The evaluated covariance surface is
$k_V(x,z)=\phi(x)^\top V_\Delta\phi(z)$. This is the covariance of curve
evaluations. With the Sobolev inner product, it should not be interpreted
automatically as a Lebesgue integral kernel for the covariance operator.

Noise here is added after extracting clean innovations. Differencing noisy
price levels creates different noise covariances and temporal dependence;
the correction $N_i$ is not an automatic solution to that inverse problem.
Sparse or noisy level reconstruction, asynchronous market quotes, and
unknown noise estimation are outside this workflow. Matching matrix
dimensions alone also does not establish matching bases: the high-level
objects carry an explicit basis identifier and units.

## 6. PSD-constrained ridge estimation

Let $w_i\geq0$ be date weights normalized to sum to one. The fitted matrix
minimizes

```math
\begin{aligned}
F(B)&=\frac{1}{2}\sum_i w_i
\left\|Z_i-O_iBO_i^\top\right\|_F^2
+\frac{\lambda}{2}\|B\|_F^2,\\
B&\in\mathcal{K}_R,\\
\mathcal{K}_R&=
\left\{B=B^\top,\ B\succeq0,\ \mathrm{tr}(B)\leq R\right\}.
\end{aligned}
```

The regularization parameter $\lambda$ is strictly positive and the trace cap
$R$ is finite and nonnegative. For orthonormal coordinates the trace bounds
total innovation variance in $H$. There is no imposed rank constraint.
Positive ridge makes the finite optimization problem strictly convex, with
a unique optimizer; it also shrinks poorly observed directions. Neither that
uniqueness nor PSD feasibility proves that the observations identify the
true covariance.

With $\mathcal{A}_i(B)=O_iBO_i^\top$, define the weighted design operator

```math
\begin{aligned}
\mathcal{Q}&=\sum_i w_i\mathcal{A}_i^*\mathcal{A}_i,\\
\mathcal{A}_i^*(D)&=O_i^\top D O_i,\\
\nabla F(B)&=\mathcal{Q}(B)
-\sum_i w_iO_i^\top Z_iO_i+\lambda B.
\end{aligned}
```

The code uses an isometric symmetric vectorization: diagonal entries are
unchanged and off-diagonal entries are multiplied by $\sqrt{2}$. Hence the
matrix representation $Q$ of $\mathcal{Q}$ preserves the Frobenius geometry.
Its dimension is $J(J+1)/2$, rather than $J$.

The main solver is accelerated projected gradient with an objective-increase
restart. Projection onto $\mathcal{K}_R$ is a Euclidean spectral projection:
negative eigenvalues are removed and, when necessary, the remaining
eigenvalues are projected onto the nonnegative trace simplex. The optional
CVXPY/SCS reference solves the same constrained objective on small problems;
it is a numerical cross-check, not statistical ground truth.

The post-projected-ridge baseline first solves the unconstrained ridge
system and then applies this projection. It generally differs from minimizing
the constrained objective because its final projection uses Euclidean
distance instead of the objective's Hessian geometry. Empirical covariance
and linear shrinkage baselines accept complete coefficient observations
$O_i=I$ only. Their noise-corrected outputs are not silently PSD projected
or trace capped.

## 7. Identification and exposure-risk diagnostics

For a fixed design, the unobserved symmetric directions are

```math
\begin{aligned}
\mathcal{N}
&=\left\{D=D^\top:\ O_iDO_i^\top=0
\text{ whenever }w_i>0\right\}\\
&=\mathrm{ker}(\mathcal{Q}).
\end{aligned}
```

The moment map identifies every covariance in the full PSD trace class with
$R>0$ precisely when this nullspace is zero. This is a global statement over
that class. A particular boundary covariance may be uniquely constrained even
when the map has a nonzero symmetric nullspace. The software does not make
that stronger, point-specific conclusion from a Gram rank check.

For a symmetric target $P$, the linear quantity
$\theta_P=\langle P,V_\Delta\rangle_F$ is globally determined by the moments
when $P$ is orthogonal to $\mathcal{N}$. In particular, an exposure vector $p$
has variance target $p^\top V_\Delta p$ with $P=pp^\top$. A risk can be
identified even if the entire covariance is not.

The diagnostics inspect eigenvalues of $Q$. Eigenvalues above a declared
relative threshold times its largest eigenvalue are treated as numerically
resolved; the remaining eigenvectors define a numerical nullspace. The
reported target residual is the relative norm of the target's projection onto
that space. Small positive eigenvalues can amplify sampling noise even when
the numerical rank is full. Rank, conditioning, and a target's alignment
with weak directions therefore need separate interpretation.

`estimate_risk` uses the numerical pseudoinverse only when the target passes
its range check. In exact arithmetic, for a target in the range, put

```math
\begin{aligned}
D&=\mathcal{Q}^\dagger(P),\\
\widehat\theta_P&=\sum_i w_i
\left\langle O_i D O_i^\top,Z_i\right\rangle_F.
\end{aligned}
```

For deterministic designs and weights, correct noise correction, and common
$V_\Delta$, this estimates $\theta_P$ without first fitting a PSD covariance.
The exact-range identity gives unbiasedness under that model. A finite
eigenvalue threshold qualifies the implemented range decision. The estimate
can be negative even for a variance target, and the API supplies no confidence
interval. Clipping it to zero would change its expectation.

All of these diagnostics concern the chosen finite basis and supplied
observation maps. They do not establish identification of an unrestricted
infinite-dimensional covariance, the source matrix $C$, or individual
Brownian loadings.

## 8. Meaning of the numerical certificates

For a feasible iterate $B$, write $G_B=\nabla F(B)$. The implementation computes
the following first-order objective-gap bound:

```math
g(B)=\langle G_B,B\rangle_F
-R\min\left\{\lambda_{\min}(G_B),0\right\}.
```

In exact arithmetic, convexity and the minimum of a linear functional over
the PSD trace ball give

```math
\begin{aligned}
0&\leq F(B)-F(B_*)\leq g(B),\\
\|B-B_*\|_F
&\leq\sqrt{\frac{2g(B)}{q_{\min}+\lambda}},\\
q_{\min}&=\lambda_{\min}(Q)\geq0.
\end{aligned}
```

Here $B_*$ is the constrained, regularized optimizer for the observed sample.
The reported `objective_gap_bound` clips a slightly negative computed gap to
zero, and `q_min` clips the computed minimum Gram eigenvalue to zero.
`covariance_error_bound` uses the displayed distance formula. These are
exact-arithmetic bounds evaluated with floating-point operations, not formal
interval-arithmetic certificates.

Stopping requires both the configured absolute/relative objective tolerance
and, when supplied, the requested covariance-distance budget.
The low-level solver retains a feasible iterate with `converged=False` on an
iteration-limit exit. The high-level workflow raises `ConvergenceError` if
the requested numerical budget is not met, retaining the estimate for
inspection. A small optimization bound says nothing by itself about sampling
error, model error, wrong drift, wrong noise correction, basis truncation, or
distance to the true $V_\Delta$.

See [estimation.py](../src/spde_covariance/estimation.py) and
[geometry.py](../src/spde_covariance/geometry.py) for the implemented constants
and diagnostic conventions.

## 9. Tuning, benchmark protocol, and supported conclusions

Tuning uses observed validation measurements only. For a candidate $B$, the
score is

```math
\frac{1}{n_{\mathrm{val}}}\sum_{i\in\mathrm{val}}
\left(
\|O_iBO_i^\top\|_F^2
-2\left\langle Z_i,O_iBO_i^\top\right\rangle_F
\right).
```

This is squared prediction loss with a candidate-independent term omitted.
Under the common moment model its expectation ranks candidates by observed
covariance error; it does not expose latent covariance truth. Indefinite
corrected validation measurements are retained.

Ridge candidates use $\lambda=\eta\|Q_{\mathrm{window}}\|_2$, where $\eta$ is
a relative strength. The design norm is recomputed for each training window
and again on refit. Benchmark window candidates are trailing observation
counts. By contrast, the high-level workflow's `trailing_window` is a calendar
width in years. After validation selects a method's parameters, refitting
uses that selected trailing count through the validation endpoint. The
stored validation score remains the selection audit; it is not relabelled
as a score of the refitted covariance.

The [full benchmark configuration](../experiments/configs/stage1.toml) declares
20 paired replications of an exact four-dimensional exponential model with
weekly intervals: 256 training, 128 validation, and 128 evaluation intervals.
It uses a trace cap of 2, relative ridge candidates of 0.0001, 0.001 and 0.01,
window counts of 128 and 256, and a numerical covariance-distance budget of
0.0001. Four observation settings share the same latent trajectory:

- Complete noiseless coefficient observations provide a recovery control and
  the valid input for empirical and shrinkage baselines.
- Dense observations contain six point rows and two delivery-average rows,
  individually scaled to unit $H$ functional norm.
- Sparse observations take nested, rotating two-row subsets of those eight
  rows, reusing the corresponding noise draws.
- Missing-pair observations alternate coordinate blocks, leaving cross-block
  covariance directions unobserved.

Within a partial-data setting, competing methods receive the same observed
maps, values, and noise corrections. The complete-data control has more
information and is interpreted separately. Source covariances and latent
innovation covariances are reserved for simulation and evaluation, outside
selection APIs.
The ordinary-increment ablation uses the same trajectory, maps, and noise
but omits transport adjustment and tunes on its own observed increments.
It is a diagnostic for transport contamination, not another valid
$V_\Delta$ estimator under the stated model. At a small weekly interval,
its finite-sample errors can resemble those of transport adjustment without
making the targets equal.

Evaluation reports full-coordinate covariance error, observed covariance
error, a fixed maturity-grid covariance error, predeclared exposure risks,
resolved/weak-direction errors, feasibility and optimization diagnostics.
Direct risks are unavailable when their range check fails. Monte Carlo
standard errors describe the uncertainty of the reported simulation means;
they are not confidence intervals for a market covariance.

Separate stress cases examine the $V_\Delta$ versus $C$ bridge, spline
projection and padding, quadrature, semigroup defects, conditioning, and
temporal-window bias after a change in source covariance. A dimension ladder
through $J=24$ measures a single fit at a common numerical budget. The current
algorithm forms dense lifted matrices, so these measurements support
moderate-dimensional computation rather than a large-scale complexity claim.

Here numerical robustness means input/provenance checks, explicit failure
states, observed-only tuning, conditioning diagnostics and repeatable
numerical comparisons. It does not mean robustness to outliers, arbitrary
misspecification, unknown level noise, or changing market microstructure.
The present synthetic results do not establish futures-market performance,
curve calibration or prediction, or superiority of ridge over every baseline.

The [API](api.md) and [workflow](workflow.md) describe callable interfaces and
their units. The [benchmark report](../experiments/results/stage1/report.md)
contains measured outcomes, acceptance criteria, configuration and provenance;
the equations above describe what those experiments estimate.
