"""Render an exact finite-model transport-SPDE covariance illustration.

Run ``uv run python examples/render_transport_video.py`` for the full video or
pass ``--preview --output /tmp/transport-qa.mp4`` for a short rendering check.
Every video frame shows an actual simulated discrete observation. Neither the
MP4 nor the subsampled GIF interpolates Brownian motion between observations.
Importing this module does not simulate, fit, or render anything.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from spde_covariance.estimation import fit_covariance
from spde_covariance.trajectories import extract_innovations, simulate_transport
from spde_covariance.transport import ExponentialTransport

NAVY = "#08121F"
CARD = "#0F2135"
WHITE = "#F3F7FC"
MUTED = "#A6B9CD"
CYAN = "#56D4E8"
CORAL = "#FF8D80"
GOLD = "#F5C66A"


@dataclass(frozen=True)
class VideoConfig:
    seed: int = 1729
    fps: int = 16
    duration: float = 20.0
    width: int = 1280
    height: int = 720
    delta_years: float = 1 / 52
    minimum_samples: int = 16
    fit_every: int = 4
    regularization: float = 1e-3
    trace_cap: float = 2.0
    numerical_error_budget: float = 1e-6

    @property
    def steps(self):
        return round(self.fps * self.duration)


def _precompute(config):
    model = ExponentialTransport()
    source_covariance = np.diag([0.4, 0.3, 0.2, 0.1])
    # The initial curve is also in the exact shift-invariant span. These are
    # raw exponential coefficients, converted to orthonormal H coordinates.
    initial = np.linalg.solve(model.transform, np.array([1.0, 0.5, -0.2, 0.12]))
    times = np.arange(config.steps + 1) * config.delta_years
    simulation = simulate_transport(
        model, times, source_covariance, initial=initial, seed=config.seed
    )
    innovations = extract_innovations(simulation.trajectory, model)
    observations = innovations.observe()
    truth = model.innovation_covariance(source_covariance, config.delta_years)
    maturities = np.linspace(0, 4, 181)
    evaluation = model.evaluate(maturities)
    curves = simulation.trajectory.coefficients @ evaluation.T
    propagated = (
        simulation.trajectory.coefficients[:-1] @ model.matrix(config.delta_years).T
    ) @ evaluation.T
    innovation_curves = innovations.values @ evaluation.T
    exposure = model.evaluate([0.5])[0] - model.evaluate([2.0])[0]
    exposure /= np.linalg.norm(exposure)
    true_risk = float(exposure @ truth @ exposure)
    estimates, errors, risks, certificates = {}, {}, {}, {}
    initial_fit = None
    fit_counts = list(range(config.minimum_samples, config.steps + 1, config.fit_every))
    if fit_counts[-1] != config.steps:
        fit_counts.append(config.steps)
    for count in fit_counts:
        result = fit_covariance(
            observations[:count],
            regularization=config.regularization,
            trace_cap=config.trace_cap,
            numerical_error_budget=config.numerical_error_budget,
            absolute_tolerance=1e-12,
            relative_tolerance=0,
            initial=initial_fit,
        )
        if not result.converged:
            raise RuntimeError(f"covariance fit at n={count} failed: {result.status}")
        initial_fit = result.covariance
        estimates[count] = result.covariance.copy()
        errors[count] = float(np.linalg.norm(result.covariance - truth) / np.linalg.norm(truth))
        risks[count] = float(exposure @ result.covariance @ exposure)
        certificates[count] = result.covariance_error_bound
    return {
        "model": model,
        "source_covariance": source_covariance,
        "truth": truth,
        "times": times,
        "maturities": maturities,
        "curves": curves,
        "propagated": propagated,
        "innovation_curves": innovation_curves,
        "estimates": estimates,
        "errors": errors,
        "risks": risks,
        "true_risk": true_risk,
        "certificates": certificates,
        "fit_counts": np.array(fit_counts),
    }


def _scene(config, data):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spde-video-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon, Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": WHITE,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#28415B",
            "savefig.facecolor": NAVY,
        }
    )
    fig = plt.figure(figsize=(config.width / 100, config.height / 100), dpi=100, facecolor=NAVY)
    fig.text(0.047, 0.94, "From a noisy curve to its covariance", fontsize=23, weight="bold")
    fig.text(
        0.047,
        0.887,
        r"$df_t=\partial_x f_t\,dt+L\,dW_t$    |    $C=LL^*$    |    "
        "4 Brownian channels · exact finite model · zero additional drift",
        color=MUTED,
        fontsize=11,
    )

    def panel(bounds):
        axis = fig.add_axes(bounds, facecolor=CARD)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9, length=3)
        axis.grid(color="#54708D", alpha=0.17, linewidth=0.6)
        return axis

    curve_axis = panel([0.073, 0.50, 0.425, 0.30])
    innovation_axis = panel([0.073, 0.20, 0.425, 0.18])
    metric_axis = panel([0.588, 0.20, 0.363, 0.14])
    fig.text(
        0.073, 0.835, "01  TRANSPORT AND RANDOM FORCING", color=CYAN, fontsize=10, weight="bold"
    )
    fig.text(
        0.588,
        0.835,
        "02  COVARIANCE OF NORMALIZED INNOVATIONS",
        color=CYAN,
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.588,
        0.37,
        "03  LEARNING FROM THE OBSERVED INCREMENTS",
        color=CYAN,
        fontsize=10,
        weight="bold",
    )
    x = data["maturities"]
    values = np.concatenate((data["curves"].ravel(), data["propagated"].ravel()))
    margin = 0.12 * np.ptp(values)
    curve_axis.set(xlim=(0, 4), ylim=(values.min() - margin, values.max() + margin))
    curve_axis.set_ylabel("Curve value (arbitrary units)")
    curve_axis.tick_params(labelbottom=False)
    trail = [
        curve_axis.plot([], [], color=MUTED, alpha=0.05 + 0.035 * i, lw=1)[0] for i in range(6)
    ]
    (current,) = curve_axis.plot([], [], color=CYAN, lw=2.6, label=r"Observed $f_i$")
    (predicted,) = curve_axis.plot(
        [], [], color=CORAL, lw=1.8, ls="--", label=r"Transport $S_\Delta f_{i-1}$"
    )
    gap = Polygon(np.zeros((2 * len(x), 2)), closed=True, color=CORAL, alpha=0.13, linewidth=0)
    curve_axis.add_patch(gap)
    curve_axis.legend(
        loc="upper right", facecolor=CARD, edgecolor="none", labelcolor=WHITE, fontsize=9
    )
    curve_clock = curve_axis.text(
        0.025, 0.94, "", transform=curve_axis.transAxes, color=GOLD, fontsize=10
    )

    innovation_limit = float(np.max(np.abs(data["innovation_curves"]))) * 1.08
    innovation_axis.set(xlim=(0, 4), ylim=(-innovation_limit, innovation_limit))
    innovation_axis.set_xlabel("Maturity x (years)", labelpad=6)
    innovation_axis.set_ylabel(r"Curve unit / $\sqrt{\mathrm{year}}$")
    innovation_axis.axhline(0, color=MUTED, lw=0.7, alpha=0.5)
    (innovation_line,) = innovation_axis.plot([], [], color=CYAN, lw=1.7)
    positive = Polygon(np.zeros((2 * len(x), 2)), color=CYAN, alpha=0.2, linewidth=0)
    negative = Polygon(np.zeros((2 * len(x), 2)), color=CORAL, alpha=0.2, linewidth=0)
    innovation_axis.add_patch(positive)
    innovation_axis.add_patch(negative)
    fig.text(
        0.073,
        0.435,
        r"Realized innovation: $X_i=(f_i-S_\Delta f_{i-1})/\sqrt{\Delta}$",
        fontsize=11,
    )

    color_limit = (
        max(
            float(np.max(np.abs(data["truth"]))),
            *(float(np.max(np.abs(v))) for v in data["estimates"].values()),
        )
        * 1.05
    )
    cmap = LinearSegmentedColormap.from_list("covariance", [CORAL, CARD, CYAN])
    truth_axis = fig.add_axes([0.592, 0.55, 0.157, 0.245], facecolor=CARD)
    estimate_axis = fig.add_axes([0.79, 0.55, 0.157, 0.245], facecolor=CARD)
    for axis, title in (
        (truth_axis, r"Analytic $V_\Delta$"),
        (estimate_axis, r"Estimated $\widehat V_\Delta(n)$"),
    ):
        axis.set_title(title, fontsize=12, pad=10)
        axis.set_xticks(range(4), ["1", "2", "3", "4"])
        axis.set_yticks(range(4), ["1", "2", "3", "4"])
        axis.tick_params(length=0, labelsize=9)
        axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        axis.grid(which="minor", color=NAVY, linewidth=1.5)
        axis.tick_params(which="minor", length=0)
        axis.set_xlabel("H-basis mode", fontsize=9, labelpad=5)
        for spine in axis.spines.values():
            spine.set_visible(False)
    truth_axis.set_ylabel("H-basis mode", fontsize=9)
    truth_axis.imshow(data["truth"], cmap=cmap, vmin=-color_limit, vmax=color_limit)
    estimate_image = estimate_axis.imshow(
        np.zeros((4, 4)), cmap=cmap, vmin=-color_limit, vmax=color_limit
    )
    truth_annotations, estimate_annotations = [], []
    for row in range(4):
        for column in range(4):
            number = data["truth"][row, column]
            label = f"{0 if abs(number) < 0.0005 else number:.3f}"
            truth_annotations.append(
                truth_axis.text(
                    column, row, label, ha="center", va="center", color=WHITE, fontsize=8
                )
            )
            estimate_annotations.append(
                estimate_axis.text(
                    column, row, "", ha="center", va="center", color=WHITE, fontsize=8
                )
            )
    awaiting = estimate_axis.text(
        0.5,
        0.5,
        f"Awaiting {config.minimum_samples}\nincrements",
        transform=estimate_axis.transAxes,
        ha="center",
        va="center",
        color=GOLD,
        fontsize=11,
    )
    colorbar_axis = fig.add_axes([0.625, 0.475, 0.29, 0.012])
    colorbar = fig.colorbar(estimate_image, cax=colorbar_axis, orientation="horizontal")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.set_label(
        "Covariance (curve unit² / year) · fixed color scale", fontsize=8, labelpad=2
    )
    fit_status = fig.text(0.588, 0.402, "", fontsize=10, color=GOLD)

    fit_counts = data["fit_counts"]
    all_errors = np.array([data["errors"][int(n)] for n in fit_counts])
    all_risk_ratios = np.array([data["risks"][int(n)] / data["true_risk"] for n in fit_counts])
    upper = max(1.3, float(np.max(all_errors)) * 1.1, float(np.max(all_risk_ratios)) * 1.1)
    metric_axis.set(xlim=(0, data["times"][-1]), ylim=(0, upper))
    metric_axis.set_xlabel("Physical time t (years)", labelpad=6)
    metric_axis.set_ylabel("Dimensionless")
    metric_axis.axhline(1, color=MUTED, lw=0.8, ls=":", alpha=0.65)
    (error_line,) = metric_axis.plot([], [], color=CORAL, lw=1.8, label="Relative HS error")
    (risk_line,) = metric_axis.plot([], [], color=CYAN, lw=1.8, label="Risk / truth")
    metric_axis.legend(
        loc="upper right",
        bbox_to_anchor=(0.97, 0.98),
        facecolor=CARD,
        edgecolor="none",
        labelcolor=WHITE,
        fontsize=7.5,
        labelspacing=0.25,
        handlelength=1.6,
        borderaxespad=0.4,
    )
    fig.text(
        0.588, 0.11, "Risk: unit-H spread between x = 0.5 and 2 years", fontsize=9, color=MUTED
    )
    summary = fig.text(0.073, 0.11, "", fontsize=11, color=WHITE)
    fig.text(
        0.047,
        0.048,
        "Discrete observations · no Brownian-time interpolation · synthetic illustration",
        color=MUTED,
        fontsize=9,
    )
    fig.text(
        0.953,
        0.048,
        r"$V_\Delta=\mathrm{Cov}(X_i)$; $C=LL^*$ is the source covariance",
        color=MUTED,
        fontsize=9,
        ha="right",
    )
    progress_axis = fig.add_axes([0.047, 0.082, 0.906, 0.004], facecolor=CARD)
    progress_axis.set(xlim=(0, 1), ylim=(0, 1))
    progress_axis.axis("off")
    progress = Rectangle((0, 0), 0, 1, color=CYAN)
    progress_axis.add_patch(progress)

    def polygon(top, bottom):
        return np.vstack((np.column_stack((x, top)), np.column_stack((x[::-1], bottom[::-1]))))

    def update(frame):
        count = frame + 1
        for offset, line in enumerate(trail):
            previous = count - len(trail) + offset
            line.set_data(x, data["curves"][previous] if previous >= 0 else np.full_like(x, np.nan))
        curve = data["curves"][count]
        transport = data["propagated"][frame]
        current.set_data(x, curve)
        predicted.set_data(x, transport)
        gap.set_xy(polygon(curve, transport))
        realized = data["innovation_curves"][frame]
        innovation_line.set_data(x, realized)
        positive.set_xy(polygon(np.maximum(realized, 0), np.zeros_like(x)))
        negative.set_xy(polygon(np.minimum(realized, 0), np.zeros_like(x)))
        curve_clock.set_text(f"t = {data['times'][count]:.2f} years")
        available = fit_counts[fit_counts <= count]
        if len(available):
            last = int(available[-1])
            matrix = data["estimates"][last]
            estimate_image.set_data(matrix)
            awaiting.set_visible(False)
            for annotation, number in zip(estimate_annotations, matrix.ravel(), strict=True):
                annotation.set_text(f"{0 if abs(number) < 0.0005 else number:.3f}")
            fit_status.set_text(f"PSD ridge · expanding window n = {last}")
            error_line.set_data(
                data["times"][available], [data["errors"][int(n)] for n in available]
            )
            risk_line.set_data(
                data["times"][available],
                [data["risks"][int(n)] / data["true_risk"] for n in available],
            )
            summary.set_text(
                f"Relative operator error {data['errors'][last]:.3f}    |    n = {count:03d} / {config.steps}"
            )
        else:
            awaiting.set_visible(True)
            fit_status.set_text("Collecting observed innovations before the first fit")
            summary.set_text(
                f"n = {count:03d} / {config.steps} · covariance needs repeated observations"
            )
        progress.set_width(count / config.steps)

    if config.width != 1280:
        from matplotlib.text import Text

        # Preview dimensions retain the final composition's relative typography.
        fig.canvas.draw()
        for text in fig.findobj(match=Text):
            text.set_fontsize(text.get_fontsize() * config.width / 1280)
    return fig, update


def render(output, config):
    output = Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must be an .mp4 path")
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError("ffmpeg is required to encode the scientific animation")
    data = _precompute(config)
    figure, update = _scene(config, data)
    import matplotlib
    from matplotlib.animation import FFMpegWriter

    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
    writer = FFMpegWriter(
        fps=config.fps,
        codec="libx264",
        bitrate=-1,
        extra_args=["-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={
            "title": "Transport-SPDE innovation covariance",
            "comment": "Exact finite-model synthetic illustration; no market-data claim.",
        },
    )
    with writer.saving(figure, str(output), dpi=100):
        for frame in range(config.steps):
            update(frame)
            writer.grab_frame(facecolor=NAVY)
    poster = output.with_name(output.stem + "-poster.png")
    update(config.steps - 1)
    figure.savefig(poster, dpi=100, facecolor=NAVY)
    gif = output.with_name(output.stem + "-preview.gif")
    gif_duration = min(8.0, config.steps / config.fps)
    speed = (config.steps / config.fps) / gif_duration
    filters = (
        f"setpts=PTS/{speed:.10g},fps=8,scale=768:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=96:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output),
            "-filter_complex",
            filters,
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}
    manifest = {
        "illustration": "exact finite-model transport SPDE; deterministic source, four Brownian channels",
        "configuration": asdict(config),
        "seed": config.seed,
        "steps": config.steps,
        "physical_horizon_years": float(data["times"][-1]),
        "model": {
            "kind": data["model"].kind,
            "basis_id": data["model"].basis_id,
            "rates_per_year": data["model"].rates.tolist(),
            "tau_years": data["model"].tau,
        },
        "source_covariance_C": data["source_covariance"].tolist(),
        "source_factor_L": np.sqrt(data["source_covariance"]).tolist(),
        "brownian_channels": 4,
        "integrated_additional_drift": "zero",
        "observation": "full clean coefficients O=I; no measurement noise",
        "target": "V_delta = Cov((f_i-S_delta f_(i-1))/sqrt(delta)); distinct from source C and factor L",
        "analytic_V_delta": data["truth"].tolist(),
        "estimator": {
            "function": "spde_covariance.estimation.fit_covariance",
            "window": "expanding, beginning at 16 increments",
            "fit_count": len(data["estimates"]),
            "maximum_certified_numerical_error": max(data["certificates"].values()),
            "final_relative_hilbert_schmidt_error": data["errors"][config.steps],
        },
        "risk": "unit-H point-evaluation spread x=.5 minus x=2 years",
        "rendering": {
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "duration_seconds": config.steps / config.fps,
            "codec": "H.264",
            "pixel_format": "yuv420p",
            "faststart": True,
            "temporal_interpolation": "none; each MP4 frame is one actual discrete observation",
            "gif": "8fps subsampling of the discrete frames, compressed to at most 8seconds",
        },
        "files": {"video": output.name, "poster": poster.name, "preview": gif.name},
        "environment": {"python": platform.python_version(), "packages": versions},
        "scope": "synthetic illustration; no empirical market, calibration, or prediction claim",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/transport-spde.mp4"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument(
        "--preview", action="store_true", help="short 960x540 rendering for visual QA"
    )
    args = parser.parse_args(argv)
    if args.seed < 0 or args.fps < 1 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("seed must be nonnegative; fps and duration must be positive")
    config = VideoConfig(
        seed=args.seed,
        fps=min(args.fps, 12) if args.preview else args.fps,
        duration=min(args.duration, 4) if args.preview else args.duration,
        width=960 if args.preview else 1280,
        height=540 if args.preview else 720,
    )
    if config.steps < config.minimum_samples:
        parser.error("render at least 16 discrete frames to show the first covariance fit")
    print(render(args.output, config).resolve())


if __name__ == "__main__":
    main()
