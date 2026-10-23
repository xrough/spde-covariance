# Stage 1 synthetic benchmark report

Configuration `stage1`; 20 paired replications, 256/128/128 training/validation/evaluation intervals; Δ=0.019230769 years; dimension 4.

Source revision: `bf64c1d01ba5ea34394df2e33c9ef67d43bf864d`; dirty=False. `manifest.json` records source hashes, lockfile, environment and solver settings.

The exact exponential model supplies analytic V independently of covariance fitting. Candidates use corrected observed validation outer products; latent truth is used only for evaluation. Ridge candidates are relative to the training Gram norm and are rescaled on refit. Refit uses the selected trailing count through validation. The evaluation block is untouched by tuning.

## Paired comparisons

Entries are Monte Carlo mean ± standard error, not confidence intervals. Full and partial settings have different information and are compared separately.

| Setting | Method | Relative HS error | Kernel grid RMSE | Exposure-risk RMSE |
| --- | --- | ---: | ---: | ---: |
| full | constrained_ridge | 0.1201 ± 0.0097 | 0.003197 ± 0.00035 | 0.02456 ± 0.0032 |
| full | projected_ridge | 0.1201 ± 0.0097 | 0.003197 ± 0.00035 | 0.02456 ± 0.0032 |
| full | empirical | 0.1189 ± 0.0098 | 0.00308 ± 0.00034 | 0.02385 ± 0.0032 |
| full | linear_shrinkage | 0.1491 ± 0.016 | 0.004101 ± 0.00062 | 0.03574 ± 0.0055 |
| full | unadjusted_increment_ablation | 0.1174 ± 0.0094 | 0.002888 ± 0.00032 | 0.0225 ± 0.003 |
| dense | constrained_ridge | 0.3133 ± 0.041 | 0.003856 ± 0.0005 | 0.04651 ± 0.0057 |
| dense | projected_ridge | 0.3134 ± 0.041 | 0.003853 ± 0.0005 | 0.04652 ± 0.0057 |
| dense | unadjusted_increment_ablation | 0.2957 ± 0.039 | 0.003921 ± 0.00057 | 0.04494 ± 0.0054 |
| sparse | constrained_ridge | 0.8068 ± 0.12 | 0.006213 ± 0.00056 | 0.07339 ± 0.0088 |
| sparse | projected_ridge | 0.7009 ± 0.087 | 0.006036 ± 0.00057 | 0.07115 ± 0.0087 |
| sparse | unadjusted_increment_ablation | 0.7538 ± 0.099 | 0.006364 ± 0.00054 | 0.07596 ± 0.0091 |
| missing_pair | constrained_ridge | 0.1666 ± 0.0095 | 0.004329 ± 0.00074 | 0.03376 ± 0.0041 |
| missing_pair | projected_ridge | 0.1666 ± 0.0095 | 0.004329 ± 0.00074 | 0.03376 ± 0.0041 |
| missing_pair | unadjusted_increment_ablation | 0.1774 ± 0.0096 | 0.00484 ± 0.00083 | 0.03842 ± 0.0044 |

![Paired benchmark comparison](comparison.png)

## Interpretation and diagnostics

Dense rows are six point evaluations and two delivery averages, each scaled to unit H norm. Sparse observations are nested rotating two-row subsets, including the exact same noise draws. The full case observes coefficients with O=I and zero noise; it is a recovery control, not a fair competitor to partial data. The missing-pair design alternates coordinate blocks (0,1) and (2,3), so cross-block covariance is unidentified. Designs are deterministic.

The transport ablation uses ordinary increments of the same trajectory with the same observation maps and noise. It tunes against its own unadjusted validation increments and is evaluated against V to expose transport bias. It is not an additional valid V estimator under the stated model.

Kernel error is RMSE on a fixed 101-point maturity grid [0,3] years. Exposures are coordinate 0 and the 0.5-versus-2-year point spread normalized to unit H norm. Both are declared before simulation. Direct risk estimates use the largest declared trailing count and are reported only when numerically identified; they need not produce a PSD full covariance.

`replications.csv` includes absolute/relative HS and spectral operator errors, observed covariance, kernel and risk errors, resolved/weak-direction errors, rank at 1e-8/1e-10/1e-12, feasibility, certificates and timings. `tuning.json` preserves all candidate scores and selected relative/actual penalties. `results.json` includes target nullspace residuals, paired method differences with Monte Carlo standard errors, direct risks, independent convex references and all stress/performance results.

Optimization certificates bound error to the constrained optimum, not truth. Runtime includes tuning and refitting; peak traced allocation excludes process RSS and native allocations not registered with Python tracing. The separate dimension ladder measures one fit at a common certificate budget. This is a dense moderate-dimensional implementation.

## Stress cases

The stress suite separates analytic V−C bridge error, quadrature discrepancy, spline interior projection and omitted-tail error, projected semigroup defect, padding, conditioning, temporal-window bias, and J=4/8/16/24 fit performance. Temporal stress fits are labelled local approximations; irregular intervals retain interval-specific targets and common-V fitting is rejected.

Acceptance: **PASS**. The predeclared O=I mean relative-HS limit is 0.30, selected/refitted constrained fits must meet their numerical budget, reference discrepancies must be within twice that budget, and missing-pair rank must be deficient. This does not require ridge to outperform every baseline. Failed tuning candidates are recorded and skipped.

These synthetic experiments do not establish empirical market performance, robustness to outliers or model misspecification, or futures-curve prediction.

## Direct exposure-risk estimates

Bias and MSE have Monte Carlo standard errors. Unidentified targets are reported as unavailable rather than estimated by regularization.

| Setting | Target | Identified | Bias ± MCSE | MSE ± MCSE |
| --- | --- | --- | ---: | ---: |
| full | coordinate_0 | True | -0.01246 ± 0.0059 | 0.0008196 ± 0.00025 |
| full | unit_H_spread_0.5_2 | True | 0.008655 ± 0.0049 | 0.0005278 ± 0.00019 |
| dense | coordinate_0 | True | -0.01254 ± 0.0061 | 0.0008712 ± 0.00026 |
| dense | unit_H_spread_0.5_2 | True | 0.008593 ± 0.0049 | 0.0005237 ± 0.00018 |
| sparse | coordinate_0 | True | -0.02215 ± 0.02 | 0.007829 ± 0.0021 |
| sparse | unit_H_spread_0.5_2 | True | 0.01457 ± 0.015 | 0.004751 ± 0.0014 |
| missing_pair | coordinate_0 | True | -0.01728 ± 0.009 | 0.001839 ± 0.00044 |
| missing_pair | unit_H_spread_0.5_2 | False | unavailable | unavailable |

## Bridge and numerical stress measurements

| Δ (years) | ‖V−C‖ HS | Analytic versus quadrature |
| ---: | ---: | ---: |
| 0.0833333 | 0.056785 | 1.14e-15 |
| 0.0192308 | 0.01606 | 1.11e-15 |
| 0.00396825 | 0.0034978 | 6.93e-16 |

Spline errors below concern the four smooth reference basis functions, summed in squared H norm. A finite constant-tail domain leaves a tail error floor. More padding at fixed dimension can reduce tail error while worsening interior resolution; whole-sieve semigroup defects need not decrease with dimension.

| J | Padding (years) | H projection error | Omitted-tail H error | Shift grid RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 1.5021 | 1.0145 | 0.099515 |
| 8 | 4 | 1.0597 | 1.0145 | 0.010568 |
| 16 | 4 | 1.0148 | 1.0145 | 0.00027062 |
| 4 | 8 | 1.5917 | 0.77718 | 0.27736 |
| 8 | 8 | 1.1461 | 0.77718 | 0.04201 |
| 16 | 8 | 0.79453 | 0.77718 | 0.0048008 |

| Regime checkpoint | Window | Window-target bias HS | Actual HS error |
| --- | ---: | ---: | ---: |
| terminal | 32 | 3.7556e-15 | 0.17464 |
| terminal | 128 | 2.8984e-15 | 0.10437 |
| terminal | 256 | 2.8204e-15 | 0.081064 |
| post_change | 32 | 2.3719e-16 | 0.13252 |
| post_change | 128 | 0.32223 | 0.34843 |
| post_change | 256 | 0.37594 | 0.37824 |

| J | Fit time (s) | Peak traced MiB | Iterations | Certified error bound |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.06432 | 0.01038 | 97 | 5.33e-06 |
| 8 | 0.08065 | 0.05463 | 97 | 5.05e-06 |
| 16 | 0.1092 | 0.7127 | 97 | 5.13e-06 |
| 24 | 0.1586 | 3.378 | 97 | 5.13e-06 |

Performance entries are single diagnostic runs on the environment in the manifest, with 64 observations and a common singular-value range [0.5,1]. They do not provide Monte Carlo uncertainty for timing. Statistical comparisons above use paired replications.
