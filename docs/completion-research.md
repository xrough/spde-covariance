# Completion after the first coherent study

This increment separates sampling, regularization, overlap inversion and model
representation. It adds numerical studies, not an infinite-dimensional theorem.
The source is the same fixed half-line Laguerre transport model documented in
[the model note](infinite-transport.md); the power-law eigenvalues are never
renormalized as a numerical cutoff changes. The second source applies the fixed
adjacent-pair rotation to those same eigenvalues.

## Controlled ablation

The primary ablation uses 20 paired replications, base grid 41 (46 actual nodes),
512 training observations, three/nine fragments and known fresh noise variance
0.001. There are 128 subsequent validation and 512 held-out observations.
One parent observation pool precedes every method view.

Sixteen strategies cross upstream estimation and completion support:

- Pairwise moments or entrywise ridge at relative penalty 0.001, each followed
  by rank-two or numerical-support completion.
- Joint PSD fragment fitting at relative penalties 0 and 0.001, with zero,
  constant or resolution-dependent floors, each followed by either completion.

The actual penalty is the relative value times maximum normalized pair coverage.
The PSD floor is in weighted kernel coordinates. The constant floor is
`1e-5`; the resolution policy is `0.02116 / Q**2`. These coincide at Q=46,
which supplies an exact anchor control. The weighted diagonal-floor trace is
`0.02116 / Q` under refinement; this alone says nothing about the completed
kernel after separator inversion. Identical upstream objectives are solved once.
Per-policy times therefore depend on cache/order and do not rank method speed.

A separately labelled weighted PSD projection of the raw rank-two result is
retained. It can change observed entries and is not called a canonical completion.
Zero-floor and hard-rank alternatives can be unavailable. Every declared case,
reason and numerical certificate is retained; there is no silent fallback.

## Joint refinement of a fixed model

The primary empirical study crosses base grids 21/41/81, sample counts
128/512/2048, both fixed sources and three/nine fragments, with 20 paired
replications. It compares raw rank-two, separately projected rank-two and the
two coherent floor policies at relative penalty 0.001. The generating source
uses 512 forcing modes at every empirical resolution. Fixed exposure maturities
are inserted in every grid, so point-risk targets do not move.

Population checks separately vary forcing cutoffs 64/128/256/512, output
projection dimensions 4/8/16/32 and grids through 161/321. Forcing errors are
measured against the 512-mode reference and accompanied by a fixed-source tail
allowance. Representation error projects the **full-forcing innovation
covariance**, not a different source restricted to the same output dimension.
Grid completions are compared on common physical anchor nodes with the finest
available reference. An unavailable reference is recorded, not zero-filled.

The report separates error to the generating kernel from error to its canonical
completion. An observed score cannot distinguish completions with identical
observed entries. Errors in missing regions and exposure risks remain simulation
diagnostics. Delivery estimates are grid quadrature; continuous delivery truth
and quadrature error are reported separately in the risk table.

Finite SPD two-block perturbation controls bound the change in `A S^-1 B`
using the separator spectral gap and the changes in all three blocks. They
explicitly become unavailable if a perturbation reaches that gap. General
recursive, statistical and continuum bounds remain future research.

## Training-only stability and source classes

At the coarsest empirical grid and largest training count, three observed
validation candidates use relative penalties 0.0001/0.001/0.01. Each training
fragment is split into complementary balanced halves. The observed-only rule
is compared with a predeclared guard: relative split-completion difference at
most 1 and retained separator condition number at most `1e8`. Thresholds are
not selected from latent errors or held-out data. Missing split diagnostics
make the guarded rule unavailable. A failed declared candidate blocks selection.
This is a stability diagnostic and abstention policy, not an error certificate.

[Finite source-class diagnostics](source-classes.md) now distinguish output
dimension J from forcing support M. The refinement report compares M=J,2J,4J
for J=4,8,16. Embedded feasible source witnesses provide nested upper-distance
envelopes; solver lower bounds always belong to their own finite cone.
Quadrature comparisons and generating-source tail allowances are labelled
separately. No finite-cone exclusion is an exclusion of all infinite-rank sources.

## Reproduce

```sh
spde-covariance benchmark-partial experiments/configs/completion-research-smoke.toml --output experiments/runs/research-check
spde-covariance benchmark-partial experiments/configs/completion-ablation.toml --output experiments/runs/ablation-new
spde-covariance benchmark-partial experiments/configs/completion-refinement.toml --output experiments/runs/refinement-new
```

Commit numerical source before full runs. Manifests, paired replication
checkpoints, configurations, failures and solver audits accompany the evidence.
MCSE is calculated over independent replication-level paired differences.
Acceptance requires complete accounting, feasible available primary outputs,
valid available certificates and independent numerical controls; it does not
require any method to win or every ablation to be available.

The subsequent [completion forecast study](completion-forecast.md) uses these
same strategies with frozen forecasts and the existing RQ3 chronology.
