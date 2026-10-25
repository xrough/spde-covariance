# True-point observation identification annex

This deterministic annex observes true point values of the full shifted curve:
the positive quadrature factors are evaluation of the Laguerre basis at
`grid+r`. It applies no output projection before evaluation. The numerical
source map integrates an unknown 16-mode source block. The common fixed
power-law source tail is evaluated through mode 512, with the omitted tail
bounded separately. The infinite-dimensional sources themselves are
`diag(q_k) +/- 0.95*embed_16(D)`, with `q_k=(k+1)^-3/zeta(3)`; prefixes are
never renormalized. Positive finite blocks and the common strictly positive
diagonal tail make both full sources PSD, trace one and infinite rank.

The seven maturities are 0, 0.5, ..., 3 years. Native fragments are indices
{0,1,2,3} and {3,4,5,6}. The exposure is node 0 minus node 1; its sole
hedge is node 5 minus node 6. A new, trace-preserving source perturbation is
constructed for this true-point observation map. The 19 observed covariance
pairs agree to svec residual 5.92507e-16; the optimal
hedges are 0.816780208 and 6.079174152.
Choosing the other world's hedge adds variance
0.006136506, or
31.7258% of unhedged
target variance. These are population decisions in a deterministic control,
not sampling or market-performance results.

The original projected J=8 witness is **not** an observational alias under
true point observations: its observed svec discrepancy is
0.000220079507, with maximum observed
entry difference 9.54547316e-05.
The original J=8 Monte Carlo study remains a projected-observation study.
This annex establishes a new pair; it does not relabel that study as observing
full-curve point values, transfer its finite-sample results, or claim uniform
identification over all infinite-dimensional sources.

The 24-to-48-point quadrature map discrepancy is
2.3976e-15; the largest independent
finite-source block-integration discrepancy is
4.354e-15.
The separate continuous-time tail allowance follows from `C_tail <= q_512 I_H`
and the translated RKHS evaluation Gram. It bounds optimal hedge errors by
3.83143e-06 and
8.51263e-06. Formulas and residuals
are evaluated in floating point, without an exact numerical-error certificate.

Acceptance passed: `True`. All source matrices, factors,
kernel values, checks and thresholds are retained in [results](results.json.gz),
with [configuration](config.json), [source provenance](manifest.json) and
[completion status](status.json).

## Compressed evidence

This evidence bundle retains the complete accepted run. CSV files and measurement/audit JSON files are compressed with gzip; artifact links point to those archives. Use `gzip -dc FILE.gz > FILE` to read an uncompressed copy. [Bundle checksums](artifact_sha256.json) record original and published bytes. The unchanged [manifest](manifest.json) retains the originating clean revision and source hashes.
