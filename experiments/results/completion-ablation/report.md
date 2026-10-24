# Covariance completion: ablations and refinement

Declared profile: `ablation`. Seed `20261028`; 20 paired replications; base grids [41]; training counts [512]; fragment covers [3, 9]; noise variance 0.001.

The ablation profile compares explicit partial-estimation, floor and completion choices. The refinement profile studies changes in grid, representation and forcing cutoff. These questions have different targets; a favorable empirical error is not a population convergence proof.

## Information and numerical boundaries

Every strategy uses the same supplied training observations for its declared case. Partial means include the known fresh point-noise correction and may be indefinite. Upstream shrinkage and coherent PSD constraints can change observed entries; support/rank completion preserves those fitted entries. Selection uses observed validation scores and any predeclared training-split admissibility guard. No latent error, source label, held-out test score or future origin selects a candidate.

Observed validation is blind to differences confined to never-observed pairs. Training split stability and separator inverse amplification are separate diagnostics, not missing-pair risk guarantees or proofs of identification. Unavailable half-fits, conditioning and numerical certificates remain in the audit.

Shared upstream fits are cached across completion rules. A policy visited first bears the upstream solve time; a reused policy bears only its incremental completion time. The recorded runtimes are cache/order-dependent execution diagnostics, not a fair speed ranking of separately executed estimators.

The resolution floor is gamma/Q² in sqrt(W) K sqrt(W) coordinates. Its diagonal baseline trace is gamma/Q; the legacy constant floor and zero floor are explicit ablations. Vanishing baseline trace alone establishes no continuum convergence after potentially ill-conditioned separator inversion. Weighted grid coordinates are not orthonormal H coordinates.

The optimizer gap certifies only the weighted observed-partial optimization error. It excludes sampling, completion amplification, grid, representation, forcing truncation and SPDE-source errors. Raw or rank-truncated completed kernels can remain indefinite; no global PSD projection occurs implicitly.

## Empirical kernel error

![Empirical kernel errors](empirical-errors.png)

Ablation dot plots fix the declared central case: base grid 41, N=512; every configured model/cover panel uses the same policy rows and logarithmic error axis. Points show mean weighted error and horizontal intervals show replication-level MCSE where available. An interval reaching zero is clipped to the common positive plotting limit; zero means and unavailable results are labelled explicitly. Exact means and MCSE are retained in the tables. Generating-kernel error, canonical-estimation error and the canonical-versus-generating discrepancy are distinct quantities. Unavailable policies remain in the saved summaries and the central table.

Central table: base grid 41, N=512. Unavailable metrics are not replaced by zero.

| Model | Fragments | Policy | Complete error | Missing error | Canonical error | Observed modification |
|---|---:|---|---:|---:|---:|---:|
| powerlaw | 3 | pairwise_rank2 | 0.22635 ± 0.026 | 0.082445 ± 0.011 | 0.22635 ± 0.026 | 0 ± 0 |
| powerlaw | 3 | pairwise_support | 2.0939 ± 0.82 | 2.0397 ± 0.83 | 2.0939 ± 0.82 | 0 ± 0 |
| powerlaw | 3 | ridge_rank2 | 0.22621 ± 0.026 | 0.083111 ± 0.011 | 0.22621 ± 0.026 | 0.090499 ± 0.0015 |
| powerlaw | 3 | ridge_support | 1.9769 ± 0.76 | 1.9226 ± 0.76 | 1.9769 ± 0.76 | 0.090499 ± 0.0015 |
| powerlaw | 3 | coherent_eta0_zero_rank2 | 0.22692 ± 0.026 | 0.092838 ± 0.012 | 0.22692 ± 0.026 | 0.45769 ± 0.067 |
| powerlaw | 3 | coherent_eta0_zero_support | 0.2272 ± 0.026 | 0.093423 ± 0.012 | 0.2272 ± 0.026 | 0.45769 ± 0.067 |
| powerlaw | 3 | coherent_eta0_constant_rank2 | 0.22691 ± 0.026 | 0.092835 ± 0.012 | 0.22691 ± 0.026 | 0.45791 ± 0.067 |
| powerlaw | 3 | coherent_eta0_constant_support | 0.2272 ± 0.026 | 0.09342 ± 0.012 | 0.2272 ± 0.026 | 0.45791 ± 0.067 |
| powerlaw | 3 | coherent_eta0_resolution_rank2 | 0.22691 ± 0.026 | 0.092835 ± 0.012 | 0.22691 ± 0.026 | 0.45791 ± 0.067 |
| powerlaw | 3 | coherent_eta0_resolution_support | 0.2272 ± 0.026 | 0.09342 ± 0.012 | 0.2272 ± 0.026 | 0.45791 ± 0.067 |
| powerlaw | 3 | coherent_eta0.001_zero_rank2 | 0.22687 ± 0.026 | 0.093411 ± 0.012 | 0.22687 ± 0.026 | 0.4737 ± 0.065 |
| powerlaw | 3 | coherent_eta0.001_zero_support | 0.2272 ± 0.026 | 0.094082 ± 0.012 | 0.2272 ± 0.026 | 0.4737 ± 0.065 |
| powerlaw | 3 | coherent_eta0.001_constant_rank2 | 0.22687 ± 0.026 | 0.093406 ± 0.012 | 0.22687 ± 0.026 | 0.4739 ± 0.065 |
| powerlaw | 3 | coherent_eta0.001_constant_support | 0.22719 ± 0.026 | 0.094085 ± 0.012 | 0.22719 ± 0.026 | 0.4739 ± 0.065 |
| powerlaw | 3 | coherent_eta0.001_resolution_rank2 | 0.22687 ± 0.026 | 0.093406 ± 0.012 | 0.22687 ± 0.026 | 0.4739 ± 0.065 |
| powerlaw | 3 | coherent_eta0.001_resolution_support | 0.22719 ± 0.026 | 0.094085 ± 0.012 | 0.22719 ± 0.026 | 0.4739 ± 0.065 |
| powerlaw | 3 | pairwise_rank2_psd | 0.22276 ± 0.026 | 0.083089 ± 0.011 | 0.22276 ± 0.026 | 0.47115 ± 0.067 |
| powerlaw | 3 | validation_only | 0.21902 ± 0.025 | 0.090579 ± 0.011 | 0.21902 ± 0.025 | 0.69222 ± 0.089 |
| powerlaw | 3 | stability_guarded | 0.21902 ± 0.025 | 0.090579 ± 0.011 | 0.21902 ± 0.025 | 0.69222 ± 0.089 |
| powerlaw | 9 | pairwise_rank2 | 0.31551 ± 0.021 | 0.17105 ± 0.011 | 0.31551 ± 0.021 | 0 ± 0 |
| powerlaw | 9 | pairwise_support | 40.717 ± 16 | 40.71 ± 16 | 40.717 ± 16 | 0 ± 0 |
| powerlaw | 9 | ridge_rank2 | 0.31648 ± 0.02 | 0.17425 ± 0.011 | 0.31648 ± 0.02 | 0.16557 ± 0.0032 |
| powerlaw | 9 | ridge_support | 42.298 ± 12 | 42.293 ± 12 | 42.298 ± 12 | 0.16557 ± 0.0032 |
| powerlaw | 9 | coherent_eta0_zero_rank2 | 0.27964 ± 0.02 | 0.13166 ± 0.0093 | 0.27964 ± 0.02 | 1.0864 ± 0.077 |
| powerlaw | 9 | coherent_eta0_zero_support | 0.27952 ± 0.02 | 0.13137 ± 0.0087 | 0.27952 ± 0.02 | 1.0864 ± 0.077 |
| powerlaw | 9 | coherent_eta0_constant_rank2 | 0.27963 ± 0.02 | 0.13165 ± 0.0093 | 0.27963 ± 0.02 | 1.0867 ± 0.077 |
| powerlaw | 9 | coherent_eta0_constant_support | 0.2795 ± 0.02 | 0.13134 ± 0.0087 | 0.2795 ± 0.02 | 1.0867 ± 0.077 |
| powerlaw | 9 | coherent_eta0_resolution_rank2 | 0.27963 ± 0.02 | 0.13165 ± 0.0093 | 0.27963 ± 0.02 | 1.0867 ± 0.077 |
| powerlaw | 9 | coherent_eta0_resolution_support | 0.2795 ± 0.02 | 0.13134 ± 0.0087 | 0.2795 ± 0.02 | 1.0867 ± 0.077 |
| powerlaw | 9 | coherent_eta0.001_zero_rank2 | 0.27961 ± 0.02 | 0.13267 ± 0.0091 | 0.27961 ± 0.02 | 1.0991 ± 0.076 |
| powerlaw | 9 | coherent_eta0.001_zero_support | 0.27948 ± 0.02 | 0.1323 ± 0.0086 | 0.27948 ± 0.02 | 1.0991 ± 0.076 |
| powerlaw | 9 | coherent_eta0.001_constant_rank2 | 0.2796 ± 0.02 | 0.13266 ± 0.0091 | 0.2796 ± 0.02 | 1.0995 ± 0.076 |
| powerlaw | 9 | coherent_eta0.001_constant_support | 0.27946 ± 0.02 | 0.13226 ± 0.0086 | 0.27946 ± 0.02 | 1.0995 ± 0.076 |
| powerlaw | 9 | coherent_eta0.001_resolution_rank2 | 0.2796 ± 0.02 | 0.13266 ± 0.0091 | 0.2796 ± 0.02 | 1.0995 ± 0.076 |
| powerlaw | 9 | coherent_eta0.001_resolution_support | 0.27946 ± 0.02 | 0.13226 ± 0.0086 | 0.27946 ± 0.02 | 1.0995 ± 0.076 |
| powerlaw | 9 | pairwise_rank2_psd | 0.30395 ± 0.02 | 0.1566 ± 0.01 | 0.30395 ± 0.02 | 1.1081 ± 0.081 |
| powerlaw | 9 | validation_only | 0.26464 ± 0.018 | 0.12579 ± 0.0082 | 0.26464 ± 0.018 | 1.3888 ± 0.13 |
| powerlaw | 9 | stability_guarded | 0.26464 ± 0.018 | 0.12579 ± 0.0082 | 0.26464 ± 0.018 | 1.3888 ± 0.13 |
| rotated | 3 | pairwise_rank2 | 0.30685 ± 0.034 | 0.10666 ± 0.014 | 0.30685 ± 0.034 | 0 ± 0 |
| rotated | 3 | pairwise_support | 0.77994 ± 0.22 | 0.68386 ± 0.22 | 0.77994 ± 0.22 | 0 ± 0 |
| rotated | 3 | ridge_rank2 | 0.30751 ± 0.034 | 0.10854 ± 0.014 | 0.30751 ± 0.034 | 0.12533 ± 0.0019 |
| rotated | 3 | ridge_support | 0.77764 ± 0.22 | 0.68216 ± 0.22 | 0.77764 ± 0.22 | 0.12533 ± 0.0019 |
| rotated | 3 | coherent_eta0_zero_rank2 | 0.30757 ± 0.034 | 0.12485 ± 0.015 | 0.30757 ± 0.034 | 0.74221 ± 0.1 |
| rotated | 3 | coherent_eta0_zero_support | 0.30756 ± 0.034 | 0.12486 ± 0.015 | 0.30756 ± 0.034 | 0.74221 ± 0.1 |
| rotated | 3 | coherent_eta0_constant_rank2 | 0.30757 ± 0.034 | 0.12485 ± 0.015 | 0.30757 ± 0.034 | 0.74243 ± 0.1 |
| rotated | 3 | coherent_eta0_constant_support | 0.30756 ± 0.034 | 0.12487 ± 0.015 | 0.30756 ± 0.034 | 0.74243 ± 0.1 |
| rotated | 3 | coherent_eta0_resolution_rank2 | 0.30757 ± 0.034 | 0.12485 ± 0.015 | 0.30757 ± 0.034 | 0.74243 ± 0.1 |
| rotated | 3 | coherent_eta0_resolution_support | 0.30756 ± 0.034 | 0.12487 ± 0.015 | 0.30756 ± 0.034 | 0.74243 ± 0.1 |
| rotated | 3 | coherent_eta0.001_zero_rank2 | 0.30855 ± 0.034 | 0.12658 ± 0.015 | 0.30855 ± 0.034 | 0.76208 ± 0.1 |
| rotated | 3 | coherent_eta0.001_zero_support | 0.30853 ± 0.034 | 0.12657 ± 0.015 | 0.30853 ± 0.034 | 0.76208 ± 0.1 |
| rotated | 3 | coherent_eta0.001_constant_rank2 | 0.30855 ± 0.034 | 0.12658 ± 0.015 | 0.30855 ± 0.034 | 0.76229 ± 0.1 |
| rotated | 3 | coherent_eta0.001_constant_support | 0.30854 ± 0.034 | 0.12658 ± 0.015 | 0.30854 ± 0.034 | 0.76229 ± 0.1 |
| rotated | 3 | coherent_eta0.001_resolution_rank2 | 0.30855 ± 0.034 | 0.12658 ± 0.015 | 0.30855 ± 0.034 | 0.76229 ± 0.1 |
| rotated | 3 | coherent_eta0.001_resolution_support | 0.30854 ± 0.034 | 0.12658 ± 0.015 | 0.30854 ± 0.034 | 0.76229 ± 0.1 |
| rotated | 3 | pairwise_rank2_psd | 0.30089 ± 0.033 | 0.10933 ± 0.014 | 0.30089 ± 0.033 | 0.75709 ± 0.1 |
| rotated | 3 | validation_only | 0.30202 ± 0.033 | 0.12499 ± 0.015 | 0.30202 ± 0.033 | 1.0038 ± 0.12 |
| rotated | 3 | stability_guarded | 0.30202 ± 0.033 | 0.12499 ± 0.015 | 0.30202 ± 0.033 | 1.0038 ± 0.12 |
| rotated | 9 | pairwise_rank2 | 0.41538 ± 0.032 | 0.21713 ± 0.015 | 0.41538 ± 0.032 | 0 ± 0 |
| rotated | 9 | pairwise_support | 314.8 ± 1.5e+02 | 314.79 ± 1.5e+02 | 314.8 ± 1.5e+02 | 0 ± 0 |
| rotated | 9 | ridge_rank2 | 0.41648 ± 0.031 | 0.22033 ± 0.015 | 0.41648 ± 0.031 | 0.22308 ± 0.0043 |
| rotated | 9 | ridge_support | 204.1 ± 84 | 204.09 ± 84 | 204.1 ± 84 | 0.22308 ± 0.0043 |
| rotated | 9 | coherent_eta0_zero_rank2 | 0.36968 ± 0.032 | 0.16584 ± 0.013 | 0.36968 ± 0.032 | 1.4575 ± 0.095 |
| rotated | 9 | coherent_eta0_zero_support | 0.36905 ± 0.032 | 0.16463 ± 0.013 | 0.36905 ± 0.032 | 1.4575 ± 0.095 |
| rotated | 9 | coherent_eta0_constant_rank2 | 0.36967 ± 0.032 | 0.16583 ± 0.013 | 0.36967 ± 0.032 | 1.4578 ± 0.095 |
| rotated | 9 | coherent_eta0_constant_support | 0.36904 ± 0.032 | 0.16461 ± 0.013 | 0.36904 ± 0.032 | 1.4578 ± 0.095 |
| rotated | 9 | coherent_eta0_resolution_rank2 | 0.36967 ± 0.032 | 0.16583 ± 0.013 | 0.36967 ± 0.032 | 1.4578 ± 0.095 |
| rotated | 9 | coherent_eta0_resolution_support | 0.36904 ± 0.032 | 0.16461 ± 0.013 | 0.36904 ± 0.032 | 1.4578 ± 0.095 |
| rotated | 9 | coherent_eta0.001_zero_rank2 | 0.37002 ± 0.032 | 0.16702 ± 0.014 | 0.37002 ± 0.032 | 1.4746 ± 0.094 |
| rotated | 9 | coherent_eta0.001_zero_support | 0.36948 ± 0.032 | 0.16596 ± 0.013 | 0.36948 ± 0.032 | 1.4746 ± 0.094 |
| rotated | 9 | coherent_eta0.001_constant_rank2 | 0.37002 ± 0.032 | 0.16701 ± 0.014 | 0.37002 ± 0.032 | 1.475 ± 0.094 |
| rotated | 9 | coherent_eta0.001_constant_support | 0.36947 ± 0.032 | 0.16594 ± 0.013 | 0.36947 ± 0.032 | 1.475 ± 0.094 |
| rotated | 9 | coherent_eta0.001_resolution_rank2 | 0.37002 ± 0.032 | 0.16701 ± 0.014 | 0.37002 ± 0.032 | 1.475 ± 0.094 |
| rotated | 9 | coherent_eta0.001_resolution_support | 0.36947 ± 0.032 | 0.16594 ± 0.013 | 0.36947 ± 0.032 | 1.475 ± 0.094 |
| rotated | 9 | pairwise_rank2_psd | 0.39889 ± 0.031 | 0.19481 ± 0.013 | 0.39889 ± 0.031 | 1.5161 ± 0.11 |
| rotated | 9 | validation_only | 0.35586 ± 0.029 | 0.16174 ± 0.012 | 0.35586 ± 0.029 | 1.8172 ± 0.17 |
| rotated | 9 | stability_guarded | 0.35586 ± 0.029 | 0.16174 ± 0.012 | 0.35586 ± 0.029 | 1.8172 ± 0.17 |

## Paired observed scores and stability

Paired comparisons are computed within each replication before aggregation. A lower observed score concerns observed-pair covariance prediction; it does not validate missing cross-blocks. The complete machine-readable paired and stability tables retain availability counts, the declared guard and rejected candidates.

| Model | Fragments | N | Policy | Metric | Paired mean ± MCSE | Available replications |
|---|---:|---:|---|---|---:|---:|
| powerlaw | 3 | 512 | pairwise_rank2 | test_observed_score | 0 ± 0 | 20 |
| powerlaw | 3 | 512 | pairwise_rank2 | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_rank2 | complete_weighted_error | 0 ± 0 | 20 |
| powerlaw | 3 | 512 | pairwise_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_rank2 | missing_weighted_error | 0 ± 0 | 20 |
| powerlaw | 3 | 512 | pairwise_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_support | test_observed_score | 0 ± 0 | 20 |
| powerlaw | 3 | 512 | pairwise_support | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_support | complete_weighted_error | 1.8676 ± 0.83 | 20 |
| powerlaw | 3 | 512 | pairwise_support | complete_weighted_error_mean_absolute_deviation_from | 2.38 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_support | missing_weighted_error | 1.9572 ± 0.83 | 20 |
| powerlaw | 3 | 512 | pairwise_support | missing_weighted_error_mean_absolute_deviation_from | 2.386 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | ridge_rank2 | test_observed_score | -0.063984 ± 0.058 | 20 |
| powerlaw | 3 | 512 | ridge_rank2 | test_observed_score_mean_absolute_deviation_from | 0.20857 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | ridge_rank2 | complete_weighted_error | -0.00013664 ± 0.0012 | 20 |
| powerlaw | 3 | 512 | ridge_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0048148 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | ridge_rank2 | missing_weighted_error | 0.00066645 ± 0.0011 | 20 |
| powerlaw | 3 | 512 | ridge_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.0042965 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | ridge_support | test_observed_score | -0.063984 ± 0.058 | 20 |
| powerlaw | 3 | 512 | ridge_support | test_observed_score_mean_absolute_deviation_from | 0.20857 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | ridge_support | complete_weighted_error | 1.7505 ± 0.76 | 20 |
| powerlaw | 3 | 512 | ridge_support | complete_weighted_error_mean_absolute_deviation_from | 2.2334 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | ridge_support | missing_weighted_error | 1.8402 ± 0.76 | 20 |
| powerlaw | 3 | 512 | ridge_support | missing_weighted_error_mean_absolute_deviation_from | 2.2393 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_zero_rank2 | test_observed_score | -0.16749 ± 0.056 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.1748 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error | 0.00056968 ± 0.0015 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0055493 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error | 0.010393 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.015542 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_zero_support | test_observed_score | -0.16749 ± 0.056 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_zero_support | test_observed_score_mean_absolute_deviation_from | 0.1748 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_zero_support | complete_weighted_error | 0.00085423 ± 0.0015 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.0057997 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_zero_support | missing_weighted_error | 0.010979 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.015587 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_constant_rank2 | test_observed_score | -0.16751 ± 0.056 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.17484 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error | 0.00056659 ± 0.0015 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0055467 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error | 0.01039 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.015532 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_constant_support | test_observed_score | -0.16751 ± 0.056 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_constant_support | test_observed_score_mean_absolute_deviation_from | 0.17484 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_constant_support | complete_weighted_error | 0.0008485 ± 0.0015 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.0057865 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_constant_support | missing_weighted_error | 0.010975 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.015564 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_resolution_rank2 | test_observed_score | -0.16751 ± 0.056 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.17484 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error | 0.00056659 ± 0.0015 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0055467 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error | 0.01039 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.015532 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_resolution_support | test_observed_score | -0.16751 ± 0.056 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.17484 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_resolution_support | complete_weighted_error | 0.0008485 ± 0.0015 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.0057865 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0_resolution_support | missing_weighted_error | 0.010975 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.015564 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score | -0.22599 ± 0.081 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.27776 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error | 0.0005217 ± 0.002 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0078065 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error | 0.010967 ± 0.0041 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.016014 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_support | test_observed_score | -0.22599 ± 0.081 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_support | test_observed_score_mean_absolute_deviation_from | 0.27776 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_support | complete_weighted_error | 0.00084784 ± 0.0021 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.0080207 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_support | missing_weighted_error | 0.011637 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.015983 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score | -0.22601 ± 0.081 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.27778 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error | 0.00051759 ± 0.002 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0078035 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error | 0.010961 ± 0.0041 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.016005 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_support | test_observed_score | -0.22601 ± 0.081 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_support | test_observed_score_mean_absolute_deviation_from | 0.27778 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_support | complete_weighted_error | 0.00084446 ± 0.0021 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.0080055 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_support | missing_weighted_error | 0.011641 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.015954 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score | -0.22601 ± 0.081 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.27778 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error | 0.00051759 ± 0.002 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0078035 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error | 0.010961 ± 0.0041 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.016005 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_support | test_observed_score | -0.22601 ± 0.081 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.27778 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error | 0.00084446 ± 0.0021 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.0080055 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error | 0.011641 ± 0.0042 | 20 |
| powerlaw | 3 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.015954 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_rank2_psd | test_observed_score | -0.12309 ± 0.088 | 20 |
| powerlaw | 3 | 512 | pairwise_rank2_psd | test_observed_score_mean_absolute_deviation_from | 0.25422 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_rank2_psd | complete_weighted_error | -0.0035931 ± 0.00083 | 20 |
| powerlaw | 3 | 512 | pairwise_rank2_psd | complete_weighted_error_mean_absolute_deviation_from | 0.0025293 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | pairwise_rank2_psd | missing_weighted_error | 0.00064451 ± 0.00056 | 20 |
| powerlaw | 3 | 512 | pairwise_rank2_psd | missing_weighted_error_mean_absolute_deviation_from | 0.0021442 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | validation_only | test_observed_score | -0.6799 ± 0.4 | 20 |
| powerlaw | 3 | 512 | validation_only | test_observed_score_mean_absolute_deviation_from | 1.1704 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | validation_only | complete_weighted_error | -0.0073239 ± 0.0068 | 20 |
| powerlaw | 3 | 512 | validation_only | complete_weighted_error_mean_absolute_deviation_from | 0.021569 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | validation_only | missing_weighted_error | 0.0081348 ± 0.0056 | 20 |
| powerlaw | 3 | 512 | validation_only | missing_weighted_error_mean_absolute_deviation_from | 0.019918 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | stability_guarded | test_observed_score | -0.6799 ± 0.4 | 20 |
| powerlaw | 3 | 512 | stability_guarded | test_observed_score_mean_absolute_deviation_from | 1.1704 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | stability_guarded | complete_weighted_error | -0.0073239 ± 0.0068 | 20 |
| powerlaw | 3 | 512 | stability_guarded | complete_weighted_error_mean_absolute_deviation_from | 0.021569 (MCSE unavailable) | unavailable |
| powerlaw | 3 | 512 | stability_guarded | missing_weighted_error | 0.0081348 ± 0.0056 | 20 |
| powerlaw | 3 | 512 | stability_guarded | missing_weighted_error_mean_absolute_deviation_from | 0.019918 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_rank2 | test_observed_score | 0 ± 0 | 20 |
| powerlaw | 9 | 512 | pairwise_rank2 | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_rank2 | complete_weighted_error | 0 ± 0 | 20 |
| powerlaw | 9 | 512 | pairwise_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_rank2 | missing_weighted_error | 0 ± 0 | 20 |
| powerlaw | 9 | 512 | pairwise_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_support | test_observed_score | 0 ± 0 | 20 |
| powerlaw | 9 | 512 | pairwise_support | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_support | complete_weighted_error | 40.401 ± 16 | 20 |
| powerlaw | 9 | 512 | pairwise_support | complete_weighted_error_mean_absolute_deviation_from | 41.18 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_support | missing_weighted_error | 40.539 ± 16 | 20 |
| powerlaw | 9 | 512 | pairwise_support | missing_weighted_error_mean_absolute_deviation_from | 41.189 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | ridge_rank2 | test_observed_score | -0.082845 ± 0.059 | 20 |
| powerlaw | 9 | 512 | ridge_rank2 | test_observed_score_mean_absolute_deviation_from | 0.21945 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | ridge_rank2 | complete_weighted_error | 0.00097048 ± 0.002 | 20 |
| powerlaw | 9 | 512 | ridge_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0073134 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | ridge_rank2 | missing_weighted_error | 0.0032047 ± 0.0018 | 20 |
| powerlaw | 9 | 512 | ridge_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.0064726 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | ridge_support | test_observed_score | -0.082845 ± 0.059 | 20 |
| powerlaw | 9 | 512 | ridge_support | test_observed_score_mean_absolute_deviation_from | 0.21945 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | ridge_support | complete_weighted_error | 41.982 ± 12 | 20 |
| powerlaw | 9 | 512 | ridge_support | complete_weighted_error_mean_absolute_deviation_from | 40.059 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | ridge_support | missing_weighted_error | 42.122 ± 12 | 20 |
| powerlaw | 9 | 512 | ridge_support | missing_weighted_error_mean_absolute_deviation_from | 40.052 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_zero_rank2 | test_observed_score | -0.26721 ± 0.06 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.20388 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error | -0.035869 ± 0.0058 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.020606 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error | -0.039383 ± 0.0084 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.029839 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_zero_support | test_observed_score | -0.26721 ± 0.06 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_zero_support | test_observed_score_mean_absolute_deviation_from | 0.20388 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_zero_support | complete_weighted_error | -0.035988 ± 0.0057 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.01997 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_zero_support | missing_weighted_error | -0.039672 ± 0.0083 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.028385 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_constant_rank2 | test_observed_score | -0.26721 ± 0.06 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.20391 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error | -0.035877 ± 0.0058 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.020605 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error | -0.039395 ± 0.0084 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.029834 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_constant_support | test_observed_score | -0.26721 ± 0.06 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_constant_support | test_observed_score_mean_absolute_deviation_from | 0.20391 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_constant_support | complete_weighted_error | -0.036007 ± 0.0057 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.01997 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_constant_support | missing_weighted_error | -0.039704 ± 0.0083 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.028382 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_resolution_rank2 | test_observed_score | -0.26721 ± 0.06 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.20391 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error | -0.035877 ± 0.0058 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.020605 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error | -0.039395 ± 0.0084 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.029834 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_resolution_support | test_observed_score | -0.26721 ± 0.06 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.20391 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_resolution_support | complete_weighted_error | -0.036007 ± 0.0057 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.01997 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0_resolution_support | missing_weighted_error | -0.039704 ± 0.0083 | 20 |
| powerlaw | 9 | 512 | coherent_eta0_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.028382 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score | -0.33998 ± 0.097 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.33834 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error | -0.035896 ± 0.0065 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.022973 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error | -0.038374 ± 0.0087 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.031237 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_support | test_observed_score | -0.33998 ± 0.097 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_support | test_observed_score_mean_absolute_deviation_from | 0.33834 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_support | complete_weighted_error | -0.036031 ± 0.0064 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.02244 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_support | missing_weighted_error | -0.03875 ± 0.0087 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.0301 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score | -0.33999 ± 0.097 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.33836 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error | -0.035905 ± 0.0065 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.022971 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error | -0.038386 ± 0.0087 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.031231 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_support | test_observed_score | -0.33999 ± 0.097 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_support | test_observed_score_mean_absolute_deviation_from | 0.33836 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_support | complete_weighted_error | -0.036051 ± 0.0064 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.022436 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_support | missing_weighted_error | -0.038783 ± 0.0087 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.030088 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score | -0.33999 ± 0.097 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.33836 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error | -0.035905 ± 0.0065 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.022971 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error | -0.038386 ± 0.0087 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.031231 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_support | test_observed_score | -0.33999 ± 0.097 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.33836 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error | -0.036051 ± 0.0064 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.022436 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error | -0.038783 ± 0.0087 | 20 |
| powerlaw | 9 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.030088 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_rank2_psd | test_observed_score | 0.34221 ± 0.17 | 20 |
| powerlaw | 9 | 512 | pairwise_rank2_psd | test_observed_score_mean_absolute_deviation_from | 0.55699 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_rank2_psd | complete_weighted_error | -0.011557 ± 0.0014 | 20 |
| powerlaw | 9 | 512 | pairwise_rank2_psd | complete_weighted_error_mean_absolute_deviation_from | 0.0050445 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | pairwise_rank2_psd | missing_weighted_error | -0.014442 ± 0.0031 | 20 |
| powerlaw | 9 | 512 | pairwise_rank2_psd | missing_weighted_error_mean_absolute_deviation_from | 0.01057 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | validation_only | test_observed_score | -0.91791 ± 0.39 | 20 |
| powerlaw | 9 | 512 | validation_only | test_observed_score_mean_absolute_deviation_from | 1.2791 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | validation_only | complete_weighted_error | -0.050866 ± 0.012 | 20 |
| powerlaw | 9 | 512 | validation_only | complete_weighted_error_mean_absolute_deviation_from | 0.045051 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | validation_only | missing_weighted_error | -0.045251 ± 0.011 | 20 |
| powerlaw | 9 | 512 | validation_only | missing_weighted_error_mean_absolute_deviation_from | 0.037887 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | stability_guarded | test_observed_score | -0.91791 ± 0.39 | 20 |
| powerlaw | 9 | 512 | stability_guarded | test_observed_score_mean_absolute_deviation_from | 1.2791 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | stability_guarded | complete_weighted_error | -0.050866 ± 0.012 | 20 |
| powerlaw | 9 | 512 | stability_guarded | complete_weighted_error_mean_absolute_deviation_from | 0.045051 (MCSE unavailable) | unavailable |
| powerlaw | 9 | 512 | stability_guarded | missing_weighted_error | -0.045251 ± 0.011 | 20 |
| powerlaw | 9 | 512 | stability_guarded | missing_weighted_error_mean_absolute_deviation_from | 0.037887 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_rank2 | test_observed_score | 0 ± 0 | 20 |
| rotated | 3 | 512 | pairwise_rank2 | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_rank2 | complete_weighted_error | 0 ± 0 | 20 |
| rotated | 3 | 512 | pairwise_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_rank2 | missing_weighted_error | 0 ± 0 | 20 |
| rotated | 3 | 512 | pairwise_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_support | test_observed_score | 0 ± 0 | 20 |
| rotated | 3 | 512 | pairwise_support | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_support | complete_weighted_error | 0.47308 ± 0.22 | 20 |
| rotated | 3 | 512 | pairwise_support | complete_weighted_error_mean_absolute_deviation_from | 0.47135 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_support | missing_weighted_error | 0.5772 ± 0.22 | 20 |
| rotated | 3 | 512 | pairwise_support | missing_weighted_error_mean_absolute_deviation_from | 0.49022 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | ridge_rank2 | test_observed_score | -0.12733 ± 0.11 | 20 |
| rotated | 3 | 512 | ridge_rank2 | test_observed_score_mean_absolute_deviation_from | 0.38739 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | ridge_rank2 | complete_weighted_error | 0.0006606 ± 0.0015 | 20 |
| rotated | 3 | 512 | ridge_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0059227 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | ridge_rank2 | missing_weighted_error | 0.0018754 ± 0.0011 | 20 |
| rotated | 3 | 512 | ridge_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.0044302 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | ridge_support | test_observed_score | -0.12733 ± 0.11 | 20 |
| rotated | 3 | 512 | ridge_support | test_observed_score_mean_absolute_deviation_from | 0.38739 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | ridge_support | complete_weighted_error | 0.47079 ± 0.22 | 20 |
| rotated | 3 | 512 | ridge_support | complete_weighted_error_mean_absolute_deviation_from | 0.4667 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | ridge_support | missing_weighted_error | 0.57549 ± 0.22 | 20 |
| rotated | 3 | 512 | ridge_support | missing_weighted_error_mean_absolute_deviation_from | 0.48396 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_zero_rank2 | test_observed_score | -0.42011 ± 0.14 | 20 |
| rotated | 3 | 512 | coherent_eta0_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.46912 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error | 0.00071772 ± 0.0025 | 20 |
| rotated | 3 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0084023 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error | 0.01819 ± 0.0056 | 20 |
| rotated | 3 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.020723 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_zero_support | test_observed_score | -0.42011 ± 0.14 | 20 |
| rotated | 3 | 512 | coherent_eta0_zero_support | test_observed_score_mean_absolute_deviation_from | 0.46912 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_zero_support | complete_weighted_error | 0.0007068 ± 0.0025 | 20 |
| rotated | 3 | 512 | coherent_eta0_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.0083648 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_zero_support | missing_weighted_error | 0.018196 ± 0.0056 | 20 |
| rotated | 3 | 512 | coherent_eta0_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.020727 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_constant_rank2 | test_observed_score | -0.42014 ± 0.14 | 20 |
| rotated | 3 | 512 | coherent_eta0_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.46917 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error | 0.00071662 ± 0.0025 | 20 |
| rotated | 3 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0084006 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error | 0.018188 ± 0.0056 | 20 |
| rotated | 3 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.020727 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_constant_support | test_observed_score | -0.42014 ± 0.14 | 20 |
| rotated | 3 | 512 | coherent_eta0_constant_support | test_observed_score_mean_absolute_deviation_from | 0.46917 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_constant_support | complete_weighted_error | 0.00071145 ± 0.0025 | 20 |
| rotated | 3 | 512 | coherent_eta0_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.0083727 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_constant_support | missing_weighted_error | 0.018204 ± 0.0056 | 20 |
| rotated | 3 | 512 | coherent_eta0_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.020713 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_resolution_rank2 | test_observed_score | -0.42014 ± 0.14 | 20 |
| rotated | 3 | 512 | coherent_eta0_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.46917 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error | 0.00071662 ± 0.0025 | 20 |
| rotated | 3 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0084006 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error | 0.018188 ± 0.0056 | 20 |
| rotated | 3 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.020727 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_resolution_support | test_observed_score | -0.42014 ± 0.14 | 20 |
| rotated | 3 | 512 | coherent_eta0_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.46917 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_resolution_support | complete_weighted_error | 0.00071145 ± 0.0025 | 20 |
| rotated | 3 | 512 | coherent_eta0_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.0083727 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0_resolution_support | missing_weighted_error | 0.018204 ± 0.0056 | 20 |
| rotated | 3 | 512 | coherent_eta0_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.020713 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score | -0.53452 ± 0.16 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.60787 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error | 0.0016997 ± 0.0029 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.01031 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error | 0.019919 ± 0.0054 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.019771 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_zero_support | test_observed_score | -0.53452 ± 0.16 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_zero_support | test_observed_score_mean_absolute_deviation_from | 0.60787 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_zero_support | complete_weighted_error | 0.0016788 ± 0.0029 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.010253 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_zero_support | missing_weighted_error | 0.019905 ± 0.0054 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.019743 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score | -0.53455 ± 0.16 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.60789 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error | 0.001698 ± 0.0029 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.01031 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error | 0.019915 ± 0.0054 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.019769 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_constant_support | test_observed_score | -0.53455 ± 0.16 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_constant_support | test_observed_score_mean_absolute_deviation_from | 0.60789 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_constant_support | complete_weighted_error | 0.0016849 ± 0.0029 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.010267 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_constant_support | missing_weighted_error | 0.019915 ± 0.0054 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.019751 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score | -0.53455 ± 0.16 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.60789 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error | 0.001698 ± 0.0029 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.01031 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error | 0.019915 ± 0.0054 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.019769 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_resolution_support | test_observed_score | -0.53455 ± 0.16 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.60789 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error | 0.0016849 ± 0.0029 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.010267 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error | 0.019915 ± 0.0054 | 20 |
| rotated | 3 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.019751 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_rank2_psd | test_observed_score | -0.24421 ± 0.19 | 20 |
| rotated | 3 | 512 | pairwise_rank2_psd | test_observed_score_mean_absolute_deviation_from | 0.54929 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_rank2_psd | complete_weighted_error | -0.005959 ± 0.0011 | 20 |
| rotated | 3 | 512 | pairwise_rank2_psd | complete_weighted_error_mean_absolute_deviation_from | 0.0039106 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | pairwise_rank2_psd | missing_weighted_error | 0.0026621 ± 0.0012 | 20 |
| rotated | 3 | 512 | pairwise_rank2_psd | missing_weighted_error_mean_absolute_deviation_from | 0.0044727 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | validation_only | test_observed_score | -0.62955 ± 0.66 | 20 |
| rotated | 3 | 512 | validation_only | test_observed_score_mean_absolute_deviation_from | 1.7872 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | validation_only | complete_weighted_error | -0.0048323 ± 0.0084 | 20 |
| rotated | 3 | 512 | validation_only | complete_weighted_error_mean_absolute_deviation_from | 0.025012 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | validation_only | missing_weighted_error | 0.018329 ± 0.0068 | 20 |
| rotated | 3 | 512 | validation_only | missing_weighted_error_mean_absolute_deviation_from | 0.02308 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | stability_guarded | test_observed_score | -0.62955 ± 0.66 | 20 |
| rotated | 3 | 512 | stability_guarded | test_observed_score_mean_absolute_deviation_from | 1.7872 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | stability_guarded | complete_weighted_error | -0.0048323 ± 0.0084 | 20 |
| rotated | 3 | 512 | stability_guarded | complete_weighted_error_mean_absolute_deviation_from | 0.025012 (MCSE unavailable) | unavailable |
| rotated | 3 | 512 | stability_guarded | missing_weighted_error | 0.018329 ± 0.0068 | 20 |
| rotated | 3 | 512 | stability_guarded | missing_weighted_error_mean_absolute_deviation_from | 0.02308 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_rank2 | test_observed_score | 0 ± 0 | 20 |
| rotated | 9 | 512 | pairwise_rank2 | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_rank2 | complete_weighted_error | 0 ± 0 | 20 |
| rotated | 9 | 512 | pairwise_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_rank2 | missing_weighted_error | 0 ± 0 | 20 |
| rotated | 9 | 512 | pairwise_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_support | test_observed_score | 0 ± 0 | 20 |
| rotated | 9 | 512 | pairwise_support | test_observed_score_mean_absolute_deviation_from | 0 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_support | complete_weighted_error | 314.38 ± 1.5e+02 | 20 |
| rotated | 9 | 512 | pairwise_support | complete_weighted_error_mean_absolute_deviation_from | 425.62 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_support | missing_weighted_error | 314.58 ± 1.5e+02 | 20 |
| rotated | 9 | 512 | pairwise_support | missing_weighted_error_mean_absolute_deviation_from | 425.64 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | ridge_rank2 | test_observed_score | -0.12558 ± 0.12 | 20 |
| rotated | 9 | 512 | ridge_rank2 | test_observed_score_mean_absolute_deviation_from | 0.42944 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | ridge_rank2 | complete_weighted_error | 0.0011026 ± 0.0023 | 20 |
| rotated | 9 | 512 | ridge_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.0085739 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | ridge_rank2 | missing_weighted_error | 0.0031991 ± 0.002 | 20 |
| rotated | 9 | 512 | ridge_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.007483 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | ridge_support | test_observed_score | -0.12558 ± 0.12 | 20 |
| rotated | 9 | 512 | ridge_support | test_observed_score_mean_absolute_deviation_from | 0.42944 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | ridge_support | complete_weighted_error | 203.68 ± 84 | 20 |
| rotated | 9 | 512 | ridge_support | complete_weighted_error_mean_absolute_deviation_from | 278.7 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | ridge_support | missing_weighted_error | 203.87 ± 84 | 20 |
| rotated | 9 | 512 | ridge_support | missing_weighted_error_mean_absolute_deviation_from | 278.72 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_zero_rank2 | test_observed_score | -0.46422 ± 0.13 | 20 |
| rotated | 9 | 512 | coherent_eta0_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.37684 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error | -0.0457 ± 0.0077 | 20 |
| rotated | 9 | 512 | coherent_eta0_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.023678 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error | -0.051292 ± 0.011 | 20 |
| rotated | 9 | 512 | coherent_eta0_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.03323 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_zero_support | test_observed_score | -0.46422 ± 0.13 | 20 |
| rotated | 9 | 512 | coherent_eta0_zero_support | test_observed_score_mean_absolute_deviation_from | 0.37684 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_zero_support | complete_weighted_error | -0.046328 ± 0.0071 | 20 |
| rotated | 9 | 512 | coherent_eta0_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.02264 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_zero_support | missing_weighted_error | -0.0525 ± 0.0099 | 20 |
| rotated | 9 | 512 | coherent_eta0_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.030614 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_constant_rank2 | test_observed_score | -0.46422 ± 0.13 | 20 |
| rotated | 9 | 512 | coherent_eta0_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.37688 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error | -0.045708 ± 0.0077 | 20 |
| rotated | 9 | 512 | coherent_eta0_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.023676 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error | -0.051304 ± 0.011 | 20 |
| rotated | 9 | 512 | coherent_eta0_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.033223 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_constant_support | test_observed_score | -0.46422 ± 0.13 | 20 |
| rotated | 9 | 512 | coherent_eta0_constant_support | test_observed_score_mean_absolute_deviation_from | 0.37688 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_constant_support | complete_weighted_error | -0.04634 ± 0.0071 | 20 |
| rotated | 9 | 512 | coherent_eta0_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.022643 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_constant_support | missing_weighted_error | -0.05252 ± 0.0099 | 20 |
| rotated | 9 | 512 | coherent_eta0_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.030613 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_resolution_rank2 | test_observed_score | -0.46422 ± 0.13 | 20 |
| rotated | 9 | 512 | coherent_eta0_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.37688 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error | -0.045708 ± 0.0077 | 20 |
| rotated | 9 | 512 | coherent_eta0_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.023676 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error | -0.051304 ± 0.011 | 20 |
| rotated | 9 | 512 | coherent_eta0_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.033223 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_resolution_support | test_observed_score | -0.46422 ± 0.13 | 20 |
| rotated | 9 | 512 | coherent_eta0_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.37688 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_resolution_support | complete_weighted_error | -0.04634 ± 0.0071 | 20 |
| rotated | 9 | 512 | coherent_eta0_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.022643 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0_resolution_support | missing_weighted_error | -0.05252 ± 0.0099 | 20 |
| rotated | 9 | 512 | coherent_eta0_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.030613 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score | -0.57233 ± 0.21 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_zero_rank2 | test_observed_score_mean_absolute_deviation_from | 0.67997 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error | -0.045354 ± 0.0084 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_zero_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.027688 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error | -0.050116 ± 0.011 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_zero_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.035735 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_zero_support | test_observed_score | -0.57233 ± 0.21 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_zero_support | test_observed_score_mean_absolute_deviation_from | 0.67997 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_zero_support | complete_weighted_error | -0.045903 ± 0.0079 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_zero_support | complete_weighted_error_mean_absolute_deviation_from | 0.026692 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_zero_support | missing_weighted_error | -0.051176 ± 0.01 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_zero_support | missing_weighted_error_mean_absolute_deviation_from | 0.033056 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score | -0.57232 ± 0.21 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_constant_rank2 | test_observed_score_mean_absolute_deviation_from | 0.67999 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error | -0.045363 ± 0.0084 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_constant_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.027685 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error | -0.050128 ± 0.011 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_constant_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.035728 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_constant_support | test_observed_score | -0.57232 ± 0.21 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_constant_support | test_observed_score_mean_absolute_deviation_from | 0.67999 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_constant_support | complete_weighted_error | -0.045911 ± 0.0079 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_constant_support | complete_weighted_error_mean_absolute_deviation_from | 0.026693 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_constant_support | missing_weighted_error | -0.051189 ± 0.01 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_constant_support | missing_weighted_error_mean_absolute_deviation_from | 0.033059 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score | -0.57232 ± 0.21 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_resolution_rank2 | test_observed_score_mean_absolute_deviation_from | 0.67999 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error | -0.045363 ± 0.0084 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_resolution_rank2 | complete_weighted_error_mean_absolute_deviation_from | 0.027685 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error | -0.050128 ± 0.011 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_resolution_rank2 | missing_weighted_error_mean_absolute_deviation_from | 0.035728 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_resolution_support | test_observed_score | -0.57232 ± 0.21 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_resolution_support | test_observed_score_mean_absolute_deviation_from | 0.67999 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error | -0.045911 ± 0.0079 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_resolution_support | complete_weighted_error_mean_absolute_deviation_from | 0.026693 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error | -0.051189 ± 0.01 | 20 |
| rotated | 9 | 512 | coherent_eta0.001_resolution_support | missing_weighted_error_mean_absolute_deviation_from | 0.033059 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_rank2_psd | test_observed_score | 0.64317 ± 0.26 | 20 |
| rotated | 9 | 512 | pairwise_rank2_psd | test_observed_score_mean_absolute_deviation_from | 0.83923 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_rank2_psd | complete_weighted_error | -0.016488 ± 0.0025 | 20 |
| rotated | 9 | 512 | pairwise_rank2_psd | complete_weighted_error_mean_absolute_deviation_from | 0.0082214 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | pairwise_rank2_psd | missing_weighted_error | -0.022325 ± 0.0053 | 20 |
| rotated | 9 | 512 | pairwise_rank2_psd | missing_weighted_error_mean_absolute_deviation_from | 0.016597 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | validation_only | test_observed_score | -1.0148 ± 0.78 | 20 |
| rotated | 9 | 512 | validation_only | test_observed_score_mean_absolute_deviation_from | 2.0743 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | validation_only | complete_weighted_error | -0.059521 ± 0.013 | 20 |
| rotated | 9 | 512 | validation_only | complete_weighted_error_mean_absolute_deviation_from | 0.046736 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | validation_only | missing_weighted_error | -0.055391 ± 0.011 | 20 |
| rotated | 9 | 512 | validation_only | missing_weighted_error_mean_absolute_deviation_from | 0.035872 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | stability_guarded | test_observed_score | -1.0148 ± 0.78 | 20 |
| rotated | 9 | 512 | stability_guarded | test_observed_score_mean_absolute_deviation_from | 2.0743 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | stability_guarded | complete_weighted_error | -0.059521 ± 0.013 | 20 |
| rotated | 9 | 512 | stability_guarded | complete_weighted_error_mean_absolute_deviation_from | 0.046736 (MCSE unavailable) | unavailable |
| rotated | 9 | 512 | stability_guarded | missing_weighted_error | -0.055391 ± 0.011 | 20 |
| rotated | 9 | 512 | stability_guarded | missing_weighted_error_mean_absolute_deviation_from | 0.035872 (MCSE unavailable) | unavailable |

Training-only selection audits (counts, not extra independent samples):

| Model | Grid | Fragments | N | Policy | Available / recorded replications | Selected-candidate counts |
|---|---:|---:|---:|---|---:|---|
| powerlaw | 41 | 3 | 512 | validation_only | 20/20 | eta0.0001: 12; eta0.01: 8 |
| powerlaw | 41 | 3 | 512 | stability_guarded | 20/20 | eta0.0001: 12; eta0.01: 8 |
| powerlaw | 41 | 9 | 512 | validation_only | 20/20 | eta0.0001: 11; eta0.001: 1; eta0.01: 8 |
| powerlaw | 41 | 9 | 512 | stability_guarded | 20/20 | eta0.0001: 11; eta0.001: 1; eta0.01: 8 |
| rotated | 41 | 3 | 512 | validation_only | 20/20 | eta0.0001: 12; eta0.001: 1; eta0.01: 7 |
| rotated | 41 | 3 | 512 | stability_guarded | 20/20 | eta0.0001: 12; eta0.001: 1; eta0.01: 7 |
| rotated | 41 | 9 | 512 | validation_only | 20/20 | eta0.0001: 13; eta0.01: 7 |
| rotated | 41 | 9 | 512 | stability_guarded | 20/20 | eta0.0001: 13; eta0.01: 7 |

## Exposure errors and operational score availability

Central case: base grid 41, N=512. Exposure error below is absolute variance error against the generating grid kernel. The operational corrected quadratic score requires a future fragment that observes the entire exposure jointly. A latent error can be evaluated in simulation even when that score is unavailable; the two must not be conflated. Continuous delivery risk, when recorded separately in the raw risk table, also has a grid-quadrature approximation error.

| Model | Fragments | Policy | Exposure | Absolute variance error | Operational risk score | Score-available replications |
|---|---:|---|---|---:|---:|---:|
| powerlaw | 3 | pairwise_rank2 | point_spread | 0.16456 ± 0.034 | 0.046086 ± 0.016 | 20/20 |
| powerlaw | 3 | pairwise_rank2 | offgrid_spread | 0.2024 ± 0.035 | 0.06293 ± 0.023 | 20/20 |
| powerlaw | 3 | pairwise_rank2 | long_spread | 0.043381 ± 0.01 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | pairwise_rank2 | grid_delivery_spread | 0.15244 ± 0.025 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | pairwise_support | point_spread | 0.16456 ± 0.034 | 0.046086 ± 0.016 | 20/20 |
| powerlaw | 3 | pairwise_support | offgrid_spread | 0.2024 ± 0.035 | 0.06293 ± 0.023 | 20/20 |
| powerlaw | 3 | pairwise_support | long_spread | 3.47 ± 1.6 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | pairwise_support | grid_delivery_spread | 0.55754 ± 0.22 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | ridge_rank2 | point_spread | 0.16407 ± 0.033 | 0.04574 ± 0.016 | 20/20 |
| powerlaw | 3 | ridge_rank2 | offgrid_spread | 0.20321 ± 0.036 | 0.063715 ± 0.024 | 20/20 |
| powerlaw | 3 | ridge_rank2 | long_spread | 0.045814 ± 0.01 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | ridge_rank2 | grid_delivery_spread | 0.15275 ± 0.025 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | ridge_support | point_spread | 0.16407 ± 0.033 | 0.04574 ± 0.016 | 20/20 |
| powerlaw | 3 | ridge_support | offgrid_spread | 0.20321 ± 0.036 | 0.063715 ± 0.024 | 20/20 |
| powerlaw | 3 | ridge_support | long_spread | 3.2367 ± 1.4 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | ridge_support | grid_delivery_spread | 0.53365 ± 0.21 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_zero_rank2 | point_spread | 0.10635 ± 0.034 | 0.030611 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0_zero_rank2 | offgrid_spread | 0.17794 ± 0.04 | 0.06039 ± 0.026 | 20/20 |
| powerlaw | 3 | coherent_eta0_zero_rank2 | long_spread | 0.064567 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_zero_rank2 | grid_delivery_spread | 0.13582 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_zero_support | point_spread | 0.10635 ± 0.034 | 0.030611 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0_zero_support | offgrid_spread | 0.17794 ± 0.04 | 0.06039 ± 0.026 | 20/20 |
| powerlaw | 3 | coherent_eta0_zero_support | long_spread | 0.064871 ± 0.016 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_zero_support | grid_delivery_spread | 0.1358 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_constant_rank2 | point_spread | 0.10639 ± 0.034 | 0.030717 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0_constant_rank2 | offgrid_spread | 0.17803 ± 0.04 | 0.060519 ± 0.026 | 20/20 |
| powerlaw | 3 | coherent_eta0_constant_rank2 | long_spread | 0.064791 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_constant_rank2 | grid_delivery_spread | 0.13583 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_constant_support | point_spread | 0.10639 ± 0.034 | 0.030717 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0_constant_support | offgrid_spread | 0.17803 ± 0.04 | 0.060519 ± 0.026 | 20/20 |
| powerlaw | 3 | coherent_eta0_constant_support | long_spread | 0.065092 ± 0.016 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_constant_support | grid_delivery_spread | 0.1358 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_resolution_rank2 | point_spread | 0.10639 ± 0.034 | 0.030717 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0_resolution_rank2 | offgrid_spread | 0.17803 ± 0.04 | 0.060519 ± 0.026 | 20/20 |
| powerlaw | 3 | coherent_eta0_resolution_rank2 | long_spread | 0.064791 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_resolution_rank2 | grid_delivery_spread | 0.13583 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_resolution_support | point_spread | 0.10639 ± 0.034 | 0.030717 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0_resolution_support | offgrid_spread | 0.17803 ± 0.04 | 0.060519 ± 0.026 | 20/20 |
| powerlaw | 3 | coherent_eta0_resolution_support | long_spread | 0.065092 ± 0.016 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0_resolution_support | grid_delivery_spread | 0.1358 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_zero_rank2 | point_spread | 0.10733 ± 0.034 | 0.031186 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_zero_rank2 | offgrid_spread | 0.18071 ± 0.04 | 0.061988 ± 0.027 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_zero_rank2 | long_spread | 0.066228 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_zero_rank2 | grid_delivery_spread | 0.13763 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_zero_support | point_spread | 0.10733 ± 0.034 | 0.031186 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_zero_support | offgrid_spread | 0.18071 ± 0.04 | 0.061988 ± 0.027 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_zero_support | long_spread | 0.066831 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_zero_support | grid_delivery_spread | 0.13761 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_constant_rank2 | point_spread | 0.10744 ± 0.034 | 0.031294 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_constant_rank2 | offgrid_spread | 0.18081 ± 0.04 | 0.062119 ± 0.027 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_constant_rank2 | long_spread | 0.066482 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_constant_rank2 | grid_delivery_spread | 0.13763 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_constant_support | point_spread | 0.10744 ± 0.034 | 0.031294 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_constant_support | offgrid_spread | 0.18081 ± 0.04 | 0.062119 ± 0.027 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_constant_support | long_spread | 0.067085 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_constant_support | grid_delivery_spread | 0.13761 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_rank2 | point_spread | 0.10744 ± 0.034 | 0.031294 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_rank2 | offgrid_spread | 0.18081 ± 0.04 | 0.062119 ± 0.027 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_rank2 | long_spread | 0.066482 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_rank2 | grid_delivery_spread | 0.13763 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_support | point_spread | 0.10744 ± 0.034 | 0.031294 ± 0.019 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_support | offgrid_spread | 0.18081 ± 0.04 | 0.062119 ± 0.027 | 20/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_support | long_spread | 0.067085 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | coherent_eta0.001_resolution_support | grid_delivery_spread | 0.13761 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | pairwise_rank2_psd | point_spread | 0.11138 ± 0.034 | 0.032595 ± 0.019 | 20/20 |
| powerlaw | 3 | pairwise_rank2_psd | offgrid_spread | 0.18939 ± 0.046 | 0.074844 ± 0.034 | 20/20 |
| powerlaw | 3 | pairwise_rank2_psd | long_spread | 0.072238 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | pairwise_rank2_psd | grid_delivery_spread | 0.14135 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | validation_only | point_spread | 0.1097 ± 0.034 | 0.031885 ± 0.019 | 20/20 |
| powerlaw | 3 | validation_only | offgrid_spread | 0.19597 ± 0.042 | 0.070999 ± 0.027 | 20/20 |
| powerlaw | 3 | validation_only | long_spread | 0.073976 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | validation_only | grid_delivery_spread | 0.14552 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | stability_guarded | point_spread | 0.1097 ± 0.034 | 0.031885 ± 0.019 | 20/20 |
| powerlaw | 3 | stability_guarded | offgrid_spread | 0.19597 ± 0.042 | 0.070999 ± 0.027 | 20/20 |
| powerlaw | 3 | stability_guarded | long_spread | 0.073976 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 3 | stability_guarded | grid_delivery_spread | 0.14552 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | pairwise_rank2 | point_spread | 0.10891 ± 0.018 | 0.016384 ± 0.0055 | 20/20 |
| powerlaw | 9 | pairwise_rank2 | offgrid_spread | 0.23939 ± 0.039 | 0.084227 ± 0.027 | 20/20 |
| powerlaw | 9 | pairwise_rank2 | long_spread | 0.19214 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | pairwise_rank2 | grid_delivery_spread | 0.13076 ± 0.023 | 0.025316 ± 0.0081 | 20/20 |
| powerlaw | 9 | pairwise_support | point_spread | 0.10891 ± 0.018 | 0.016384 ± 0.0055 | 20/20 |
| powerlaw | 9 | pairwise_support | offgrid_spread | 0.23939 ± 0.039 | 0.084227 ± 0.027 | 20/20 |
| powerlaw | 9 | pairwise_support | long_spread | 53.54 ± 19 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | pairwise_support | grid_delivery_spread | 0.13076 ± 0.023 | 0.025316 ± 0.0081 | 20/20 |
| powerlaw | 9 | ridge_rank2 | point_spread | 0.10815 ± 0.018 | 0.016155 ± 0.0054 | 20/20 |
| powerlaw | 9 | ridge_rank2 | offgrid_spread | 0.24004 ± 0.039 | 0.084434 ± 0.027 | 20/20 |
| powerlaw | 9 | ridge_rank2 | long_spread | 0.19256 ± 0.031 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | ridge_rank2 | grid_delivery_spread | 0.13058 ± 0.023 | 0.025311 ± 0.0081 | 20/20 |
| powerlaw | 9 | ridge_support | point_spread | 0.10815 ± 0.018 | 0.016155 ± 0.0054 | 20/20 |
| powerlaw | 9 | ridge_support | offgrid_spread | 0.24004 ± 0.039 | 0.084434 ± 0.027 | 20/20 |
| powerlaw | 9 | ridge_support | long_spread | 56.992 ± 21 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | ridge_support | grid_delivery_spread | 0.13058 ± 0.023 | 0.025311 ± 0.0081 | 20/20 |
| powerlaw | 9 | coherent_eta0_zero_rank2 | point_spread | 0.093873 ± 0.021 | 0.015549 ± 0.0073 | 20/20 |
| powerlaw | 9 | coherent_eta0_zero_rank2 | offgrid_spread | 0.20199 ± 0.043 | 0.075032 ± 0.031 | 20/20 |
| powerlaw | 9 | coherent_eta0_zero_rank2 | long_spread | 0.14672 ± 0.018 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0_zero_rank2 | grid_delivery_spread | 0.11139 ± 0.027 | 0.02428 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0_zero_support | point_spread | 0.093873 ± 0.021 | 0.015549 ± 0.0073 | 20/20 |
| powerlaw | 9 | coherent_eta0_zero_support | offgrid_spread | 0.20199 ± 0.043 | 0.075032 ± 0.031 | 20/20 |
| powerlaw | 9 | coherent_eta0_zero_support | long_spread | 0.13343 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0_zero_support | grid_delivery_spread | 0.11139 ± 0.027 | 0.02428 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0_constant_rank2 | point_spread | 0.094164 ± 0.021 | 0.015659 ± 0.0073 | 20/20 |
| powerlaw | 9 | coherent_eta0_constant_rank2 | offgrid_spread | 0.20234 ± 0.043 | 0.075195 ± 0.031 | 20/20 |
| powerlaw | 9 | coherent_eta0_constant_rank2 | long_spread | 0.14697 ± 0.018 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0_constant_rank2 | grid_delivery_spread | 0.11141 ± 0.027 | 0.024289 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0_constant_support | point_spread | 0.094164 ± 0.021 | 0.015659 ± 0.0073 | 20/20 |
| powerlaw | 9 | coherent_eta0_constant_support | offgrid_spread | 0.20234 ± 0.043 | 0.075195 ± 0.031 | 20/20 |
| powerlaw | 9 | coherent_eta0_constant_support | long_spread | 0.13371 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0_constant_support | grid_delivery_spread | 0.11141 ± 0.027 | 0.024289 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0_resolution_rank2 | point_spread | 0.094164 ± 0.021 | 0.015659 ± 0.0073 | 20/20 |
| powerlaw | 9 | coherent_eta0_resolution_rank2 | offgrid_spread | 0.20234 ± 0.043 | 0.075195 ± 0.031 | 20/20 |
| powerlaw | 9 | coherent_eta0_resolution_rank2 | long_spread | 0.14697 ± 0.018 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0_resolution_rank2 | grid_delivery_spread | 0.11141 ± 0.027 | 0.024289 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0_resolution_support | point_spread | 0.094164 ± 0.021 | 0.015659 ± 0.0073 | 20/20 |
| powerlaw | 9 | coherent_eta0_resolution_support | offgrid_spread | 0.20234 ± 0.043 | 0.075195 ± 0.031 | 20/20 |
| powerlaw | 9 | coherent_eta0_resolution_support | long_spread | 0.13371 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0_resolution_support | grid_delivery_spread | 0.11141 ± 0.027 | 0.024289 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_zero_rank2 | point_spread | 0.095732 ± 0.021 | 0.016111 ± 0.0074 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_zero_rank2 | offgrid_spread | 0.20648 ± 0.044 | 0.077755 ± 0.032 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_zero_rank2 | long_spread | 0.15083 ± 0.018 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0.001_zero_rank2 | grid_delivery_spread | 0.11358 ± 0.027 | 0.02497 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_zero_support | point_spread | 0.095732 ± 0.021 | 0.016111 ± 0.0074 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_zero_support | offgrid_spread | 0.20648 ± 0.044 | 0.077755 ± 0.032 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_zero_support | long_spread | 0.13735 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0.001_zero_support | grid_delivery_spread | 0.11358 ± 0.027 | 0.02497 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_constant_rank2 | point_spread | 0.096022 ± 0.021 | 0.016223 ± 0.0075 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_constant_rank2 | offgrid_spread | 0.20683 ± 0.044 | 0.077922 ± 0.032 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_constant_rank2 | long_spread | 0.15108 ± 0.018 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0.001_constant_rank2 | grid_delivery_spread | 0.11359 ± 0.027 | 0.024979 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_constant_support | point_spread | 0.096022 ± 0.021 | 0.016223 ± 0.0075 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_constant_support | offgrid_spread | 0.20683 ± 0.044 | 0.077922 ± 0.032 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_constant_support | long_spread | 0.13763 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0.001_constant_support | grid_delivery_spread | 0.11359 ± 0.027 | 0.024979 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_rank2 | point_spread | 0.096022 ± 0.021 | 0.016223 ± 0.0075 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_rank2 | offgrid_spread | 0.20683 ± 0.044 | 0.077922 ± 0.032 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_rank2 | long_spread | 0.15108 ± 0.018 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_rank2 | grid_delivery_spread | 0.11359 ± 0.027 | 0.024979 ± 0.01 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_support | point_spread | 0.096022 ± 0.021 | 0.016223 ± 0.0075 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_support | offgrid_spread | 0.20683 ± 0.044 | 0.077922 ± 0.032 | 20/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_support | long_spread | 0.13763 ± 0.015 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | coherent_eta0.001_resolution_support | grid_delivery_spread | 0.11359 ± 0.027 | 0.024979 ± 0.01 | 20/20 |
| powerlaw | 9 | pairwise_rank2_psd | point_spread | 0.19113 ± 0.033 | 0.05567 ± 0.021 | 20/20 |
| powerlaw | 9 | pairwise_rank2_psd | offgrid_spread | 0.30152 ± 0.044 | 0.12646 ± 0.04 | 20/20 |
| powerlaw | 9 | pairwise_rank2_psd | long_spread | 0.22089 ± 0.03 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | pairwise_rank2_psd | grid_delivery_spread | 0.14548 ± 0.033 | 0.039393 ± 0.016 | 20/20 |
| powerlaw | 9 | validation_only | point_spread | 0.10204 ± 0.023 | 0.019153 ± 0.0088 | 20/20 |
| powerlaw | 9 | validation_only | offgrid_spread | 0.22858 ± 0.047 | 0.092083 ± 0.036 | 20/20 |
| powerlaw | 9 | validation_only | long_spread | 0.1507 ± 0.016 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | validation_only | grid_delivery_spread | 0.12451 ± 0.03 | 0.030214 ± 0.012 | 20/20 |
| powerlaw | 9 | stability_guarded | point_spread | 0.10204 ± 0.023 | 0.019153 ± 0.0088 | 20/20 |
| powerlaw | 9 | stability_guarded | offgrid_spread | 0.22858 ± 0.047 | 0.092083 ± 0.036 | 20/20 |
| powerlaw | 9 | stability_guarded | long_spread | 0.1507 ± 0.016 | unavailable: no joint future exposure observation | 0/20 |
| powerlaw | 9 | stability_guarded | grid_delivery_spread | 0.12451 ± 0.03 | 0.030214 ± 0.012 | 20/20 |
| rotated | 3 | pairwise_rank2 | point_spread | 0.27231 ± 0.049 | 0.10696 ± 0.038 | 20/20 |
| rotated | 3 | pairwise_rank2 | offgrid_spread | 0.32054 ± 0.054 | 0.14585 ± 0.043 | 20/20 |
| rotated | 3 | pairwise_rank2 | long_spread | 0.081443 ± 0.016 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | pairwise_rank2 | grid_delivery_spread | 0.23682 ± 0.039 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | pairwise_support | point_spread | 0.27231 ± 0.049 | 0.10696 ± 0.038 | 20/20 |
| rotated | 3 | pairwise_support | offgrid_spread | 0.32054 ± 0.054 | 0.14585 ± 0.043 | 20/20 |
| rotated | 3 | pairwise_support | long_spread | 1.1985 ± 0.44 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | pairwise_support | grid_delivery_spread | 0.29148 ± 0.061 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | ridge_rank2 | point_spread | 0.2708 ± 0.049 | 0.10603 ± 0.037 | 20/20 |
| rotated | 3 | ridge_rank2 | offgrid_spread | 0.32252 ± 0.054 | 0.14666 ± 0.043 | 20/20 |
| rotated | 3 | ridge_rank2 | long_spread | 0.082734 ± 0.016 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | ridge_rank2 | grid_delivery_spread | 0.23722 ± 0.039 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | ridge_support | point_spread | 0.2708 ± 0.049 | 0.10603 ± 0.037 | 20/20 |
| rotated | 3 | ridge_support | offgrid_spread | 0.32252 ± 0.054 | 0.14666 ± 0.043 | 20/20 |
| rotated | 3 | ridge_support | long_spread | 1.1826 ± 0.43 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | ridge_support | grid_delivery_spread | 0.29276 ± 0.061 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_zero_rank2 | point_spread | 0.18464 ± 0.047 | 0.064816 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0_zero_rank2 | offgrid_spread | 0.25185 ± 0.053 | 0.10481 ± 0.038 | 20/20 |
| rotated | 3 | coherent_eta0_zero_rank2 | long_spread | 0.10063 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_zero_rank2 | grid_delivery_spread | 0.18862 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_zero_support | point_spread | 0.18464 ± 0.047 | 0.064816 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0_zero_support | offgrid_spread | 0.25185 ± 0.053 | 0.10481 ± 0.038 | 20/20 |
| rotated | 3 | coherent_eta0_zero_support | long_spread | 0.10061 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_zero_support | grid_delivery_spread | 0.18854 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_constant_rank2 | point_spread | 0.1847 ± 0.048 | 0.064977 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0_constant_rank2 | offgrid_spread | 0.25203 ± 0.053 | 0.105 ± 0.038 | 20/20 |
| rotated | 3 | coherent_eta0_constant_rank2 | long_spread | 0.10084 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_constant_rank2 | grid_delivery_spread | 0.18863 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_constant_support | point_spread | 0.1847 ± 0.048 | 0.064977 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0_constant_support | offgrid_spread | 0.25203 ± 0.053 | 0.105 ± 0.038 | 20/20 |
| rotated | 3 | coherent_eta0_constant_support | long_spread | 0.10065 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_constant_support | grid_delivery_spread | 0.18858 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_resolution_rank2 | point_spread | 0.1847 ± 0.048 | 0.064977 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0_resolution_rank2 | offgrid_spread | 0.25203 ± 0.053 | 0.105 ± 0.038 | 20/20 |
| rotated | 3 | coherent_eta0_resolution_rank2 | long_spread | 0.10084 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_resolution_rank2 | grid_delivery_spread | 0.18863 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_resolution_support | point_spread | 0.1847 ± 0.048 | 0.064977 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0_resolution_support | offgrid_spread | 0.25203 ± 0.053 | 0.105 ± 0.038 | 20/20 |
| rotated | 3 | coherent_eta0_resolution_support | long_spread | 0.10065 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0_resolution_support | grid_delivery_spread | 0.18858 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_zero_rank2 | point_spread | 0.18637 ± 0.048 | 0.066096 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_zero_rank2 | offgrid_spread | 0.25657 ± 0.054 | 0.10802 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_zero_rank2 | long_spread | 0.10185 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_zero_rank2 | grid_delivery_spread | 0.19084 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_zero_support | point_spread | 0.18637 ± 0.048 | 0.066096 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_zero_support | offgrid_spread | 0.25657 ± 0.054 | 0.10802 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_zero_support | long_spread | 0.10156 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_zero_support | grid_delivery_spread | 0.19075 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_constant_rank2 | point_spread | 0.18644 ± 0.048 | 0.06626 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_constant_rank2 | offgrid_spread | 0.25674 ± 0.054 | 0.10821 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_constant_rank2 | long_spread | 0.10206 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_constant_rank2 | grid_delivery_spread | 0.19085 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_constant_support | point_spread | 0.18644 ± 0.048 | 0.06626 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_constant_support | offgrid_spread | 0.25674 ± 0.054 | 0.10821 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_constant_support | long_spread | 0.1016 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_constant_support | grid_delivery_spread | 0.19078 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_resolution_rank2 | point_spread | 0.18644 ± 0.048 | 0.06626 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_resolution_rank2 | offgrid_spread | 0.25674 ± 0.054 | 0.10821 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_resolution_rank2 | long_spread | 0.10206 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_resolution_rank2 | grid_delivery_spread | 0.19085 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_resolution_support | point_spread | 0.18644 ± 0.048 | 0.06626 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_resolution_support | offgrid_spread | 0.25674 ± 0.054 | 0.10821 ± 0.039 | 20/20 |
| rotated | 3 | coherent_eta0.001_resolution_support | long_spread | 0.1016 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | coherent_eta0.001_resolution_support | grid_delivery_spread | 0.19078 ± 0.046 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | pairwise_rank2_psd | point_spread | 0.18871 ± 0.048 | 0.067321 ± 0.039 | 20/20 |
| rotated | 3 | pairwise_rank2_psd | offgrid_spread | 0.26197 ± 0.06 | 0.12352 ± 0.047 | 20/20 |
| rotated | 3 | pairwise_rank2_psd | long_spread | 0.11544 ± 0.023 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | pairwise_rank2_psd | grid_delivery_spread | 0.19275 ± 0.048 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | validation_only | point_spread | 0.18938 ± 0.049 | 0.069148 ± 0.039 | 20/20 |
| rotated | 3 | validation_only | offgrid_spread | 0.27353 ± 0.056 | 0.12201 ± 0.04 | 20/20 |
| rotated | 3 | validation_only | long_spread | 0.10613 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | validation_only | grid_delivery_spread | 0.20043 ± 0.047 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | stability_guarded | point_spread | 0.18938 ± 0.049 | 0.069148 ± 0.039 | 20/20 |
| rotated | 3 | stability_guarded | offgrid_spread | 0.27353 ± 0.056 | 0.12201 ± 0.04 | 20/20 |
| rotated | 3 | stability_guarded | long_spread | 0.10613 ± 0.028 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 3 | stability_guarded | grid_delivery_spread | 0.20043 ± 0.047 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | pairwise_rank2 | point_spread | 0.16191 ± 0.024 | 0.025379 ± 0.011 | 20/20 |
| rotated | 9 | pairwise_rank2 | offgrid_spread | 0.34458 ± 0.059 | 0.17237 ± 0.058 | 20/20 |
| rotated | 9 | pairwise_rank2 | long_spread | 0.24957 ± 0.041 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | pairwise_rank2 | grid_delivery_spread | 0.19214 ± 0.034 | 0.04589 ± 0.018 | 20/20 |
| rotated | 9 | pairwise_support | point_spread | 0.16191 ± 0.024 | 0.025379 ± 0.011 | 20/20 |
| rotated | 9 | pairwise_support | offgrid_spread | 0.34458 ± 0.059 | 0.17237 ± 0.058 | 20/20 |
| rotated | 9 | pairwise_support | long_spread | 484.32 ± 3.3e+02 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | pairwise_support | grid_delivery_spread | 0.19214 ± 0.034 | 0.04589 ± 0.018 | 20/20 |
| rotated | 9 | ridge_rank2 | point_spread | 0.16122 ± 0.024 | 0.024942 ± 0.011 | 20/20 |
| rotated | 9 | ridge_rank2 | offgrid_spread | 0.34547 ± 0.059 | 0.1728 ± 0.058 | 20/20 |
| rotated | 9 | ridge_rank2 | long_spread | 0.25128 ± 0.04 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | ridge_rank2 | grid_delivery_spread | 0.19255 ± 0.034 | 0.045969 ± 0.018 | 20/20 |
| rotated | 9 | ridge_support | point_spread | 0.16122 ± 0.024 | 0.024942 ± 0.011 | 20/20 |
| rotated | 9 | ridge_support | offgrid_spread | 0.34547 ± 0.059 | 0.1728 ± 0.058 | 20/20 |
| rotated | 9 | ridge_support | long_spread | 329.51 ± 2.2e+02 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | ridge_support | grid_delivery_spread | 0.19255 ± 0.034 | 0.045969 ± 0.018 | 20/20 |
| rotated | 9 | coherent_eta0_zero_rank2 | point_spread | 0.13002 ± 0.028 | 0.02161 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0_zero_rank2 | offgrid_spread | 0.28747 ± 0.063 | 0.1442 ± 0.064 | 20/20 |
| rotated | 9 | coherent_eta0_zero_rank2 | long_spread | 0.20154 ± 0.031 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0_zero_rank2 | grid_delivery_spread | 0.1723 ± 0.039 | 0.046391 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0_zero_support | point_spread | 0.13002 ± 0.028 | 0.02161 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0_zero_support | offgrid_spread | 0.28747 ± 0.063 | 0.1442 ± 0.064 | 20/20 |
| rotated | 9 | coherent_eta0_zero_support | long_spread | 0.19056 ± 0.03 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0_zero_support | grid_delivery_spread | 0.1723 ± 0.039 | 0.046391 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0_constant_rank2 | point_spread | 0.13029 ± 0.028 | 0.02176 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0_constant_rank2 | offgrid_spread | 0.28792 ± 0.063 | 0.14443 ± 0.064 | 20/20 |
| rotated | 9 | coherent_eta0_constant_rank2 | long_spread | 0.20175 ± 0.031 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0_constant_rank2 | grid_delivery_spread | 0.1723 ± 0.039 | 0.046405 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0_constant_support | point_spread | 0.13029 ± 0.028 | 0.02176 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0_constant_support | offgrid_spread | 0.28792 ± 0.063 | 0.14443 ± 0.064 | 20/20 |
| rotated | 9 | coherent_eta0_constant_support | long_spread | 0.19082 ± 0.03 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0_constant_support | grid_delivery_spread | 0.1723 ± 0.039 | 0.046405 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0_resolution_rank2 | point_spread | 0.13029 ± 0.028 | 0.02176 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0_resolution_rank2 | offgrid_spread | 0.28792 ± 0.063 | 0.14443 ± 0.064 | 20/20 |
| rotated | 9 | coherent_eta0_resolution_rank2 | long_spread | 0.20175 ± 0.031 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0_resolution_rank2 | grid_delivery_spread | 0.1723 ± 0.039 | 0.046405 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0_resolution_support | point_spread | 0.13029 ± 0.028 | 0.02176 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0_resolution_support | offgrid_spread | 0.28792 ± 0.063 | 0.14443 ± 0.064 | 20/20 |
| rotated | 9 | coherent_eta0_resolution_support | long_spread | 0.19082 ± 0.03 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0_resolution_support | grid_delivery_spread | 0.1723 ± 0.039 | 0.046405 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0.001_zero_rank2 | point_spread | 0.13302 ± 0.029 | 0.022761 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0.001_zero_rank2 | offgrid_spread | 0.29366 ± 0.064 | 0.14952 ± 0.066 | 20/20 |
| rotated | 9 | coherent_eta0.001_zero_rank2 | long_spread | 0.20444 ± 0.031 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0.001_zero_rank2 | grid_delivery_spread | 0.17516 ± 0.039 | 0.047974 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0.001_zero_support | point_spread | 0.13302 ± 0.029 | 0.022761 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0.001_zero_support | offgrid_spread | 0.29366 ± 0.064 | 0.14952 ± 0.066 | 20/20 |
| rotated | 9 | coherent_eta0.001_zero_support | long_spread | 0.19399 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0.001_zero_support | grid_delivery_spread | 0.17516 ± 0.039 | 0.047974 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0.001_constant_rank2 | point_spread | 0.13329 ± 0.029 | 0.022916 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0.001_constant_rank2 | offgrid_spread | 0.29406 ± 0.064 | 0.14976 ± 0.066 | 20/20 |
| rotated | 9 | coherent_eta0.001_constant_rank2 | long_spread | 0.20466 ± 0.031 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0.001_constant_rank2 | grid_delivery_spread | 0.17517 ± 0.039 | 0.047988 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0.001_constant_support | point_spread | 0.13329 ± 0.029 | 0.022916 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0.001_constant_support | offgrid_spread | 0.29406 ± 0.064 | 0.14976 ± 0.066 | 20/20 |
| rotated | 9 | coherent_eta0.001_constant_support | long_spread | 0.19424 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0.001_constant_support | grid_delivery_spread | 0.17517 ± 0.039 | 0.047988 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0.001_resolution_rank2 | point_spread | 0.13329 ± 0.029 | 0.022916 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0.001_resolution_rank2 | offgrid_spread | 0.29406 ± 0.064 | 0.14976 ± 0.066 | 20/20 |
| rotated | 9 | coherent_eta0.001_resolution_rank2 | long_spread | 0.20466 ± 0.031 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0.001_resolution_rank2 | grid_delivery_spread | 0.17517 ± 0.039 | 0.047988 ± 0.023 | 20/20 |
| rotated | 9 | coherent_eta0.001_resolution_support | point_spread | 0.13329 ± 0.029 | 0.022916 ± 0.014 | 20/20 |
| rotated | 9 | coherent_eta0.001_resolution_support | offgrid_spread | 0.29406 ± 0.064 | 0.14976 ± 0.066 | 20/20 |
| rotated | 9 | coherent_eta0.001_resolution_support | long_spread | 0.19424 ± 0.029 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | coherent_eta0.001_resolution_support | grid_delivery_spread | 0.17517 ± 0.039 | 0.047988 ± 0.023 | 20/20 |
| rotated | 9 | pairwise_rank2_psd | point_spread | 0.25071 ± 0.044 | 0.089491 ± 0.036 | 20/20 |
| rotated | 9 | pairwise_rank2_psd | offgrid_spread | 0.43257 ± 0.065 | 0.2538 ± 0.082 | 20/20 |
| rotated | 9 | pairwise_rank2_psd | long_spread | 0.31035 ± 0.042 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | pairwise_rank2_psd | grid_delivery_spread | 0.21958 ± 0.048 | 0.08136 ± 0.038 | 20/20 |
| rotated | 9 | validation_only | point_spread | 0.1393 ± 0.03 | 0.026441 ± 0.014 | 20/20 |
| rotated | 9 | validation_only | offgrid_spread | 0.31308 ± 0.068 | 0.17156 ± 0.075 | 20/20 |
| rotated | 9 | validation_only | long_spread | 0.20648 ± 0.03 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | validation_only | grid_delivery_spread | 0.18892 ± 0.042 | 0.057604 ± 0.027 | 20/20 |
| rotated | 9 | stability_guarded | point_spread | 0.1393 ± 0.03 | 0.026441 ± 0.014 | 20/20 |
| rotated | 9 | stability_guarded | offgrid_spread | 0.31308 ± 0.068 | 0.17156 ± 0.075 | 20/20 |
| rotated | 9 | stability_guarded | long_spread | 0.20648 ± 0.03 | unavailable: no joint future exposure observation | 0/20 |
| rotated | 9 | stability_guarded | grid_delivery_spread | 0.18892 ± 0.042 | 0.057604 ± 0.027 | 20/20 |

## Population refinement

![Population refinement](population-refinement.png)

Population audits distinguish forcing-cutoff error, finite-basis representation error and comparison to the finest available declared canonical population grid restricted to the common anchor grid. The canonical-versus-generating discrepancy can persist even when numerical refinement errors decrease. Reference availability, numerical support cutoff, inverse amplification and omitted forcing bounds remain separate columns in the population table; an unavailable reference is not evidence of convergence.

Relative canonical reference error above `1e-4` is a numerical-sensitivity warning, not a statistical acceptance requirement or a continuum-error bound. The actual reference grid used is recorded explicitly; a coarser available reference does not silently become the finest configured grid.

| Model | Type | Grid | Fragments | M | J | Forcing error | Representation error | Canonical reference error | Reference grid | Relative reference error | Canonical/generating discrepancy | Tail bound | Reference available |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| powerlaw | forcing |  |  | 64 |  | 8.1136e-09 | unavailable | unavailable |  | unavailable | unavailable | 0.00050954 | True |
| powerlaw | forcing |  |  | 128 |  | 6.1186e-10 | unavailable | unavailable |  | unavailable | unavailable | 0.00012838 | True |
| powerlaw | forcing |  |  | 256 |  | 4.7147e-11 | unavailable | unavailable |  | unavailable | unavailable | 3.2222e-05 | True |
| powerlaw | forcing |  |  | 512 |  | 0 | unavailable | unavailable |  | unavailable | unavailable | 8.0712e-06 | True |
| powerlaw | representation |  |  |  | 4 | unavailable | 0.00026427 | unavailable |  | unavailable | unavailable | unavailable | True |
| powerlaw | representation |  |  |  | 8 | unavailable | 1.7435e-05 | unavailable |  | unavailable | unavailable | unavailable | True |
| powerlaw | representation |  |  |  | 16 | unavailable | 1.2291e-06 | unavailable |  | unavailable | unavailable | unavailable | True |
| powerlaw | representation |  |  |  | 32 | unavailable | 1.0173e-07 | unavailable |  | unavailable | unavailable | unavailable | True |
| powerlaw | grid | 321 | 3 |  |  | unavailable | unavailable | 0 | 321 | 0 | 3.1879e-06 | unavailable | True |
| powerlaw | grid | 321 | 9 |  |  | unavailable | unavailable | 0 | 321 | 0 | 3.1918e-07 | unavailable | True |
| powerlaw | grid | 161 | 3 |  |  | unavailable | unavailable | 1.0008e-07 | 321 | 3.776e-08 | 3.0881e-06 | unavailable | True |
| powerlaw | grid | 161 | 9 |  |  | unavailable | unavailable | 7.9018e-09 | 321 | 2.9813e-09 | 3.1232e-07 | unavailable | True |
| powerlaw | grid | 81 | 3 |  |  | unavailable | unavailable | 2.6999e-07 | 321 | 1.0187e-07 | 2.9184e-06 | unavailable | True |
| powerlaw | grid | 81 | 9 |  |  | unavailable | unavailable | 4.2964e-08 | 321 | 1.621e-08 | 2.7947e-07 | unavailable | True |
| powerlaw | grid | 41 | 3 |  |  | unavailable | unavailable | 5.059e-07 | 321 | 1.9087e-07 | 2.6825e-06 | unavailable | True |
| powerlaw | grid | 41 | 9 |  |  | unavailable | unavailable | 4.8637e-07 | 321 | 1.835e-07 | 7.7387e-07 | unavailable | True |
| powerlaw | grid | 21 | 3 |  |  | unavailable | unavailable | 3.0417e-06 | 321 | 1.1476e-06 | 3.56e-07 | unavailable | True |
| powerlaw | grid | 21 | 9 |  |  | unavailable | unavailable | 7.491e-07 | 321 | 2.8263e-07 | 4.3647e-07 | unavailable | True |
| rotated | forcing |  |  | 64 |  | 8.2507e-09 | unavailable | unavailable |  | unavailable | unavailable | 0.00050954 | True |
| rotated | forcing |  |  | 128 |  | 6.1708e-10 | unavailable | unavailable |  | unavailable | unavailable | 0.00012838 | True |
| rotated | forcing |  |  | 256 |  | 4.7348e-11 | unavailable | unavailable |  | unavailable | unavailable | 3.2222e-05 | True |
| rotated | forcing |  |  | 512 |  | 0 | unavailable | unavailable |  | unavailable | unavailable | 8.0712e-06 | True |
| rotated | representation |  |  |  | 4 | unavailable | 0.00030888 | unavailable |  | unavailable | unavailable | unavailable | True |
| rotated | representation |  |  |  | 8 | unavailable | 1.9414e-05 | unavailable |  | unavailable | unavailable | unavailable | True |
| rotated | representation |  |  |  | 16 | unavailable | 1.3058e-06 | unavailable |  | unavailable | unavailable | unavailable | True |
| rotated | representation |  |  |  | 32 | unavailable | 1.051e-07 | unavailable |  | unavailable | unavailable | unavailable | True |
| rotated | grid | 321 | 3 |  |  | unavailable | unavailable | 0 | 321 | 0 | 3.604e-06 | unavailable | True |
| rotated | grid | 321 | 9 |  |  | unavailable | unavailable | 0 | 321 | 0 | 7.6193e-07 | unavailable | True |
| rotated | grid | 161 | 3 |  |  | unavailable | unavailable | 1.1009e-07 | 321 | 2.9761e-08 | 3.4943e-06 | unavailable | True |
| rotated | grid | 161 | 9 |  |  | unavailable | unavailable | 1.4665e-06 | 321 | 3.9643e-07 | 8.2064e-07 | unavailable | True |
| rotated | grid | 81 | 3 |  |  | unavailable | unavailable | 2.9355e-07 | 321 | 7.9352e-08 | 3.3113e-06 | unavailable | True |
| rotated | grid | 81 | 9 |  |  | unavailable | unavailable | 1.4681e-06 | 321 | 3.9687e-07 | 8.2893e-07 | unavailable | True |
| rotated | grid | 41 | 3 |  |  | unavailable | unavailable | 4.2763e-06 | 321 | 1.156e-06 | 9.1438e-07 | unavailable | True |
| rotated | grid | 41 | 9 |  |  | unavailable | unavailable | 5.597e-07 | 321 | 1.513e-07 | 2.4438e-07 | unavailable | True |
| rotated | grid | 21 | 3 |  |  | unavailable | unavailable | 3.0816e-06 | 321 | 8.3303e-07 | 6.002e-07 | unavailable | True |
| rotated | grid | 21 | 9 |  |  | unavailable | unavailable | 1.5944e-06 | 321 | 4.3102e-07 | 8.8386e-07 | unavailable | True |

### Finite separator perturbation audit

The first-join SPD bound is available for 40/40 recorded population comparisons. Independent finite-matrix controls are available for 6/6 cases. Availability requires the separator perturbation to be smaller than its protected positive spectral gap; otherwise the bound is explicitly unavailable.

These are protected **population** kernel blocks with the declared weighted diagonal floor. The comparison changes the forcing cutoff from the lowest declared M to the fixed reference cutoff. The calculation bounds A S⁻¹ B at the first join only. It is not a bound on empirically fitted kernels, subsequent recursive joins, global completed-kernel error, or continuum error. A floor can improve this finite gap condition while also changing the target; neither effect is concealed.

| Model | Grid | Fragments | Floor | M / reference | Separator gap | Separator perturbation | Bound available | Cross-block bound | Actual cross-block difference |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|
| powerlaw | 321 | 3 | constant | 64 / 512 | 1e-05 | 3.7672e-09 | True | 20.83 | 1.0446e-08 |
| powerlaw | 321 | 3 | resolution | 64 / 512 | 1.991e-07 | 3.7672e-09 | True | 53536 | 1.4528e-07 |
| powerlaw | 321 | 9 | constant | 64 / 512 | 1e-05 | 5.2981e-09 | True | 9.6397 | 5.5268e-10 |
| powerlaw | 321 | 9 | resolution | 64 / 512 | 1.991e-07 | 5.2981e-09 | True | 24968 | 9.53e-10 |
| powerlaw | 161 | 3 | constant | 64 / 512 | 1e-05 | 3.7877e-09 | True | 21.018 | 9.5413e-09 |
| powerlaw | 161 | 3 | resolution | 64 / 512 | 7.6789e-07 | 3.7877e-09 | True | 3580.8 | 3.3273e-08 |
| powerlaw | 161 | 9 | constant | 64 / 512 | 1e-05 | 5.3095e-09 | True | 9.555 | 5.1338e-10 |
| powerlaw | 161 | 9 | resolution | 64 / 512 | 7.6789e-07 | 5.3095e-09 | True | 1630.8 | 7.9197e-10 |
| powerlaw | 81 | 3 | constant | 64 / 512 | 1e-05 | 3.8297e-09 | True | 21.4 | 7.8602e-09 |
| powerlaw | 81 | 3 | resolution | 64 / 512 | 2.861e-06 | 3.8297e-09 | True | 261.69 | 6.8843e-09 |
| powerlaw | 81 | 9 | constant | 64 / 512 | 1e-05 | 5.3416e-09 | True | 9.3865 | 4.355e-10 |
| powerlaw | 81 | 9 | resolution | 64 / 512 | 2.861e-06 | 5.3416e-09 | True | 114.83 | 4.4856e-10 |
| powerlaw | 41 | 3 | constant | 64 / 512 | 1e-05 | 3.9109e-09 | True | 22.135 | 5.0238e-09 |
| powerlaw | 41 | 3 | resolution | 64 / 512 | 1e-05 | 3.9109e-09 | True | 22.135 | 5.0238e-09 |
| powerlaw | 41 | 9 | constant | 64 / 512 | 1e-05 | 5.4928e-09 | True | 8.7352 | 2.6919e-10 |
| powerlaw | 41 | 9 | resolution | 64 / 512 | 1e-05 | 5.4928e-09 | True | 8.7352 | 2.6919e-10 |
| powerlaw | 21 | 3 | constant | 64 / 512 | 1e-05 | 3.9587e-09 | True | 23.024 | 2.3021e-09 |
| powerlaw | 21 | 3 | resolution | 64 / 512 | 3.1302e-05 | 3.9587e-09 | True | 2.3493 | 1.4527e-09 |
| powerlaw | 21 | 9 | constant | 64 / 512 | 1e-05 | 5.9957e-09 | True | 6.8993 | 1.0724e-10 |
| powerlaw | 21 | 9 | resolution | 64 / 512 | 3.1302e-05 | 5.9957e-09 | True | 0.70387 | 5.7253e-12 |
| rotated | 321 | 3 | constant | 64 / 512 | 1e-05 | 3.831e-09 | True | 36.519 | 1.158e-08 |
| rotated | 321 | 3 | resolution | 64 / 512 | 1.991e-07 | 3.831e-09 | True | 93890 | 1.6405e-07 |
| rotated | 321 | 9 | constant | 64 / 512 | 1e-05 | 5.3906e-09 | True | 14.446 | 5.7252e-10 |
| rotated | 321 | 9 | resolution | 64 / 512 | 1.991e-07 | 5.3906e-09 | True | 37435 | 9.6182e-10 |
| rotated | 161 | 3 | constant | 64 / 512 | 1e-05 | 3.8517e-09 | True | 36.824 | 1.0544e-08 |
| rotated | 161 | 3 | resolution | 64 / 512 | 7.6789e-07 | 3.8517e-09 | True | 6273.9 | 4.1523e-08 |
| rotated | 161 | 9 | constant | 64 / 512 | 1e-05 | 5.4022e-09 | True | 14.302 | 5.3004e-10 |
| rotated | 161 | 9 | resolution | 64 / 512 | 7.6789e-07 | 5.4022e-09 | True | 2441.4 | 8.2947e-10 |
| rotated | 81 | 3 | constant | 64 / 512 | 1e-05 | 3.8944e-09 | True | 37.438 | 8.6286e-09 |
| rotated | 81 | 3 | resolution | 64 / 512 | 2.861e-06 | 3.8944e-09 | True | 457.81 | 6.6615e-09 |
| rotated | 81 | 9 | constant | 64 / 512 | 1e-05 | 5.4349e-09 | True | 14.013 | 4.4645e-10 |
| rotated | 81 | 9 | resolution | 64 / 512 | 2.861e-06 | 5.4349e-09 | True | 171.42 | 4.5766e-10 |
| rotated | 41 | 3 | constant | 64 / 512 | 1e-05 | 3.9769e-09 | True | 38.603 | 5.4291e-09 |
| rotated | 41 | 3 | resolution | 64 / 512 | 1e-05 | 3.9769e-09 | True | 38.603 | 5.4291e-09 |
| rotated | 41 | 9 | constant | 64 / 512 | 1e-05 | 5.5889e-09 | True | 12.951 | 2.7107e-10 |
| rotated | 41 | 9 | resolution | 64 / 512 | 1e-05 | 5.5889e-09 | True | 12.951 | 2.7107e-10 |
| rotated | 21 | 3 | constant | 64 / 512 | 1e-05 | 4.0255e-09 | True | 40.173 | 2.4561e-09 |
| rotated | 21 | 3 | resolution | 64 / 512 | 3.1302e-05 | 4.0255e-09 | True | 4.0991 | 1.645e-09 |
| rotated | 21 | 9 | constant | 64 / 512 | 1e-05 | 6.1002e-09 | True | 9.9832 | 1.2751e-10 |
| rotated | 21 | 9 | resolution | 64 / 512 | 3.1302e-05 | 6.1002e-09 | True | 1.0185 | 7.3884e-12 |

## Nested finite SPDE source classes

![Finite-source distance bounds](source-nesting.png)

Each panel fixes output representation J, target grid, model and cover, and increases only the forcing dimension M. Varying output J changes the target representation and is not the nested-class comparison. Lower and upper bounds concern distance of that fixed target kernel to the declared finite constant-PSD-source class. Original upper bounds are feasible-witness distances; unfinished solves can have nonmonotone upper bounds even when the exact distance to truly nested classes cannot increase. The transported nested witness and map-comparison allowance make changes in the numerical transport maps explicit. Increasing unfinished upper bounds are not evidence that the exact nested-class distance increased; convergence and availability flags remain in the table.

A positive finite-class lower bound is not an exclusion of an infinite-dimensional or time-varying SPDE. For empirically fitted targets this is a compatibility diagnostic of an estimate, not a sampling-calibrated rejection of the generating process. Fixed generating tail bounds and numerical map allowances do not become optimizer-gap certificates.

| Model | Fragments | Target | M | J | Lower | Upper | Nested witness upper | Map allowance | Fixed generating tail | Converged | Available | Unavailable reason |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|

## Acceptance and retained failures

- `all_declared_cases_recorded`: `True`.
- `all_declared_estimates_recorded`: `True`.
- `declared_primary_available`: `True`.
- `available_primary_outputs_psd`: `True`.
- `available_certificates_within_budget`: `True`.
- `finite_separator_bound_controls`: `True`.
- `available_population_separator_bounds`: `True`.
- `source_controls`: `True`.
- `finite_numeric_outputs`: `True`.
- `passed`: `True`.
- `unavailable_count`: `0`.
- `coherent_fit_count`: `320`.
- `failed_certificate_count`: `0`.

Independent source-class controls passed: `True`. The complete control audit is retained in `results.json`.

Finite-source controls:

| Control | Lower distance | Upper distance |
|---|---:|---:|
| projected_forcing_support | 0.039315 | 0.039315 |
| projected_forcing_support | 0 | 2.5438e-16 |
| projected_forcing_support | 0 | 2.7756e-16 |

## Saved evidence and limits

[Configuration](config.json) · [Source/environment manifest](manifest.json) · [Results](results.json.gz) · [Empirical summaries](completion_summary.json.gz) · [Paired summaries](paired_summary.json.gz) · [Training stability](stability_summary.json.gz) · [Exposure summaries](risk_summary.json.gz) · [Population audit](population.json.gz) · [Finite separator perturbations](separator_perturbations.json.gz) · [Finite-source nesting](source_nesting.json.gz)

Each summary also has a CSV counterpart. This report reads saved summaries and audits without redoing fits, selecting favorable cases, or pooling dates as independent Monte Carlo replications. These experiments do not establish canonical identification from a fragment cover, uniform continuum stability, market calibration, confidence intervals or unrestricted SPDE admissibility.

## Compressed evidence

This evidence bundle retains the complete accepted run. CSV files and measurement/audit JSON files are compressed with gzip; artifact links point to those archives. Use `gzip -dc FILE.gz > FILE` to read an uncompressed copy. [Bundle checksums](artifact_sha256.json) record original and published bytes. The unchanged [manifest](manifest.json) retains the originating clean revision and source hashes.
