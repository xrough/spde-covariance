# Scientific figures and supplementary animations

Both videos are generated from seeded numerical models with Matplotlib and
ffmpeg. They contain no external footage or market data. The MP4 files use
H.264, yuv420p pixels and faststart metadata for browser playback. GIF previews
use fewer frames. The PNG figures use conventional white Matplotlib subplots,
labelled axes and units, ordinary legends, and fixed comparison scales. They
are the primary illustrations in the README; animations are supplementary.
The transport GIF compresses 20 seconds into 8 seconds; the
observation companion keeps its 12-second pace.

## Transport and covariance estimation

[Play the 20-second transport-SPDE video](../assets/transport-spde.mp4).

![Transport and innovation covariance](../assets/transport-spde-poster.png)

The exact four-dimensional exponential model uses four independent Brownian
sources, constant source covariance C, zero additional drift and known transport.
The video follows a curve, its transport prediction, and the normalized
innovations. Existing package routines fit a running PSD, trace-capped ridge
estimate of V_delta. The video labels C as a simulation input and V_delta as the
estimated target. The six panels show the curve/transport comparison, analytical
and fitted covariance matrices on a shared color scale, the normalized
innovation, relative HS error, and the exposure-variance ratio. Actual discrete
simulation frames are used; curve paths are
not presented as interpolated continuous Brownian samples.

```sh
uv sync --locked
uv run --no-sync python examples/render_transport_video.py --output assets/transport-spde.mp4 --seed 1729 --fps 16 --duration 20
# A short rendering check, saved separately from the selected asset
uv run --no-sync python examples/render_transport_video.py --preview --output experiments/runs/video-preview.mp4
```

Time and maturity are measured in years. Curves use arbitrary units. The
[manifest](../assets/transport-spde.json) records the seed, interval length,
basis, source covariance, regularization, certificates and encoding settings.

## Observation geometry and spread risk

[Play the 12-second observation-diagnostics companion](../assets/observability.mp4).

![Unpaired Gaussian observations](../assets/observability-poster.png)

This is a Gaussian innovation illustration. The latent models have covariance
matrices with unit diagonal and correlations +0.8 and -0.8. Observing one
coordinate per independent draw produces the same marginal distributions.
Their spread variances are 0.4 and 3.6. The five panels show latent reference
clouds with 95% Gaussian contours, identical marginal densities with scalar
observations, and the two spread variances. The static figure shows the unpaired
design. The animation later changes the design explicitly to paired observations
and marks the newly observed pairs. Latent reference clouds are labelled and
are not supplied as data under the unpaired design.

```sh
uv run --no-sync python examples/render_observability_video.py --output-dir assets
```

The [manifest](../assets/observability.json) records seeds, assumptions, matrices,
the exposure and the observation-design change. This companion illustrates the
information available to an estimator; its hidden clouds are not supplied to a
partial-observation fit.

## Rendering requirements

Encoding requires an `ffmpeg` executable with the `libx264` encoder on PATH.
The Python requirements are the package's existing NumPy/Matplotlib environment.
Each renderer also writes a PNG poster and an optimized GIF beside the MP4.
Numerical manifests permit reproduction of the model and frame sequence;
encoded bytes can differ across ffmpeg/font versions.
