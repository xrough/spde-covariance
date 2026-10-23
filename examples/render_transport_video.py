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

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(config.width / 100, config.height / 100),
        dpi=100,
        layout="constrained",
        gridspec_kw={"width_ratios": (1.1, 1, 1)},
    )
    curve_axis, truth_axis, estimate_axis = axes[0]
    innovation_axis, error_axis, risk_axis = axes[1]
    fig.suptitle(
        "Transport-SPDE innovation covariance\n"
        r"Exact four-mode model; $\Delta=1/52$ year; full observations; no measurement noise",
        fontsize=12,
    )
    fig.supxlabel(
        r"$C=LL^*=\operatorname{diag}(0.4,0.3,0.2,0.1)$; four Brownian channels; "
        "zero additional drift.\n"
        f"Synthetic data, seed {config.seed}. Every animation frame is a discrete observation; "
        "no interpolation in physical time.",
        fontsize=8,
    )

    for axis in (curve_axis, innovation_axis, error_axis, risk_axis):
        axis.grid(color="0.85", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    x = data["maturities"]
    values = np.concatenate((data["curves"].ravel(), data["propagated"].ravel()))
    margin = 0.12 * np.ptp(values)
    curve_axis.set(
        title="(a) Curve and one-step transport",
        xlim=(0, 4),
        ylim=(values.min() - margin, values.max() + margin),
        xlabel="Maturity x (years)",
        ylabel="Curve value (arbitrary units)",
    )
    (current,) = curve_axis.plot([], [], color="tab:blue", lw=1.6, label=r"$f_i$")
    (predicted,) = curve_axis.plot(
        [], [], color="tab:orange", lw=1.4, ls="--", label=r"$S_\Delta f_{i-1}$"
    )
    curve_axis.legend(loc="upper right", frameon=True)
    curve_clock = curve_axis.text(
        0.04, 0.96, "", transform=curve_axis.transAxes, va="top", fontsize=9
    )

    innovation_limit = float(np.max(np.abs(data["innovation_curves"]))) * 1.08
    innovation_axis.set(
        title="(d) Normalized innovation\n"
        r"$X_i=(f_i-S_\Delta f_{i-1})/\sqrt{\Delta}$",
        xlim=(0, 4),
        ylim=(-innovation_limit, innovation_limit),
        xlabel="Maturity x (years)",
        ylabel=r"Curve units / $\sqrt{\mathrm{year}}$",
    )
    innovation_axis.axhline(0, color="0.35", lw=0.6)
    (innovation_line,) = innovation_axis.plot([], [], color="tab:blue", lw=1.4)

    color_limit = (
        max(
            float(np.max(np.abs(data["truth"]))),
            *(float(np.max(np.abs(v))) for v in data["estimates"].values()),
        )
        * 1.05
    )
    for axis, title in (
        (truth_axis, r"(b) Analytic covariance $V_\Delta$"),
        (estimate_axis, r"(c) PSD-constrained estimate $\widehat V_\Delta$"),
    ):
        axis.set_title(title)
        axis.set_xticks(range(4), ["1", "2", "3", "4"])
        axis.set_yticks(range(4), ["1", "2", "3", "4"])
        axis.set_xlabel(r"$H$-basis mode")
        axis.set_ylabel(r"$H$-basis mode")
        axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.6)
        axis.tick_params(which="minor", length=0)
    truth_axis.imshow(data["truth"], cmap="RdBu_r", vmin=-color_limit, vmax=color_limit)
    estimate_image = estimate_axis.imshow(
        np.zeros((4, 4)), cmap="RdBu_r", vmin=-color_limit, vmax=color_limit
    )

    def annotation_color(number):
        return "white" if abs(number) > 0.6 * color_limit else "black"

    estimate_annotations = []
    for row in range(4):
        for column in range(4):
            number = data["truth"][row, column]
            truth_axis.text(
                column,
                row,
                f"{0 if abs(number) < 0.0005 else number:.3f}",
                ha="center",
                va="center",
                color=annotation_color(number),
                fontsize=8,
            )
            estimate_annotations.append(
                estimate_axis.text(column, row, "", ha="center", va="center", fontsize=8)
            )
    awaiting = estimate_axis.text(
        0.5,
        0.5,
        f"Fewer than {config.minimum_samples}\nincrements",
        transform=estimate_axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    estimate_title = r"(c) PSD-constrained estimate $\widehat V_\Delta$"
    fit_sample = estimate_axis.set_title(estimate_title + "\nExpanding window")
    colorbar = fig.colorbar(
        estimate_image, ax=[truth_axis, estimate_axis], fraction=0.035, pad=0.04
    )
    colorbar.set_label(r"Covariance (curve units$^2$ / year)")
    colorbar.ax.tick_params(labelsize=8)

    fit_counts = data["fit_counts"]
    all_errors = np.array([data["errors"][int(n)] for n in fit_counts])
    all_risk_ratios = np.array([data["risks"][int(n)] / data["true_risk"] for n in fit_counts])
    error_axis.set(
        title="(e) Covariance estimation error",
        xlim=(0, data["times"][-1]),
        ylim=(0, max(0.1, float(np.max(all_errors)) * 1.12)),
        xlabel="Physical time t (years)",
        ylabel="Relative HS error",
    )
    (error_line,) = error_axis.plot(
        [], [], color="tab:blue", lw=1.4, label=r"$\|\widehat V_\Delta-V_\Delta\|/\|V_\Delta\|$"
    )
    error_axis.legend(loc="upper right")
    risk_axis.set(
        title="(f) Maturity-spread variance\nUnit-H spread: 0.5 minus 2 years",
        xlim=(0, data["times"][-1]),
        ylim=(0, max(1.3, float(np.max(all_risk_ratios)) * 1.12)),
        xlabel="Physical time t (years)",
        ylabel="Estimated / analytic variance",
    )
    risk_axis.axhline(1, color="0.4", lw=1, ls="--", label="Analytic variance")
    (risk_line,) = risk_axis.plot([], [], color="tab:blue", lw=1.4, label="Estimated variance")
    risk_axis.legend(loc="lower right")

    def update(frame):
        count = frame + 1
        current.set_data(x, data["curves"][count])
        predicted.set_data(x, data["propagated"][frame])
        innovation_line.set_data(x, data["innovation_curves"][frame])
        curve_clock.set_text(f"t = {data['times'][count]:.2f} years")
        available = fit_counts[fit_counts <= count]
        if len(available):
            last = int(available[-1])
            matrix = data["estimates"][last]
            estimate_image.set_data(matrix)
            awaiting.set_visible(False)
            fit_sample.set_text(estimate_title + f"\nExpanding window: n = {last} increments")
            for annotation, number in zip(estimate_annotations, matrix.ravel(), strict=True):
                annotation.set_text(f"{0 if abs(number) < 0.0005 else number:.3f}")
                annotation.set_color(annotation_color(number))
            error_line.set_data(
                data["times"][available], [data["errors"][int(n)] for n in available]
            )
            risk_line.set_data(
                data["times"][available],
                [data["risks"][int(n)] / data["true_risk"] for n in available],
            )
        else:
            awaiting.set_visible(True)
            fit_sample.set_text(estimate_title + "\nExpanding window")
            for annotation in estimate_annotations:
                annotation.set_text("")

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
            writer.grab_frame(facecolor="white")
    poster = output.with_name(output.stem + "-poster.png")
    update(config.steps - 1)
    figure.savefig(poster, dpi=100, facecolor="white")
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
            "visual_style": "white Matplotlib research figure; standard labeled subplots",
            "shared_covariance_color_scale": "fixed over the complete discrete experiment",
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
