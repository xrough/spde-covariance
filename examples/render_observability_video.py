"""Render the Gaussian covariance-identification companion animation.

Run: python examples/render_observability_video.py
Requires the core NumPy/Matplotlib dependencies and ffmpeg with libx264.
No market data, internet access, or optional Python dependencies are needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Ellipse, FancyBboxPatch

BACKGROUND = "#08121F"
CARD = "#0F2135"
EDGE = "#20364D"
TEXT = "#F3F7FC"
MUTED = "#A7BACF"
CYAN = "#56D4E8"
CORAL = "#FF8D80"
GOLD = "#F5C66A"


def _style_axis(axis):
    axis.set_facecolor("none")
    axis.tick_params(colors=MUTED, labelsize=10, length=0, pad=5)
    for spine in axis.spines.values():
        spine.set_color(EDGE)
    axis.grid(color=EDGE, linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)


def _scene(seed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    figure.text(
        0.035,
        0.946,
        "GAUSSIAN INNOVATIONS  /  OBSERVATION GEOMETRY",
        fontsize=11,
        color=MUTED,
        weight="bold",
    )
    figure.text(
        0.035,
        0.882,
        "What the observations cannot tell you",
        fontsize=28,
        color=TEXT,
        weight="bold",
    )
    subtitle = figure.text(
        0.035,
        0.833,
        "Identical marginals. Opposite covariance. Very different spread risk.",
        fontsize=14,
        color=MUTED,
    )
    for left, width in ((0.035, 0.425), (0.48, 0.255), (0.755, 0.21)):
        figure.add_artist(
            FancyBboxPatch(
                (left, 0.25),
                width,
                0.50,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                transform=figure.transFigure,
                facecolor=CARD,
                edgecolor=EDGE,
                linewidth=1.2,
                zorder=-10,
            )
        )
    hidden_title = figure.text(
        0.2475, 0.711, "HIDDEN JOINT TRUTH", ha="center", fontsize=14, weight="bold"
    )
    figure.text(0.6075, 0.711, "SAME OBSERVED MARGINALS", ha="center", fontsize=12.5, weight="bold")
    figure.text(0.86, 0.711, "SPREAD VARIANCE", ha="center", fontsize=14, weight="bold")

    rng = np.random.default_rng(seed)
    matrices = (np.array([[1.0, 0.8], [0.8, 1.0]]), np.array([[1.0, -0.8], [-0.8, 1.0]]))
    latent_clouds = [rng.multivariate_normal(np.zeros(2), matrix, size=210) for matrix in matrices]
    joint_samples = [rng.multivariate_normal(np.zeros(2), matrix, size=95) for matrix in matrices]
    # This stream is generated independently of every latent joint cloud. A
    # displayed scalar is never paired with another coordinate from its draw.
    independent_scalar_draws = rng.standard_normal(80)
    scans, latent_scatters, observed_scatters, badges = [], [], [], []
    for index, (left, color, covariance) in enumerate(
        zip((0.062, 0.272), (CYAN, CORAL), matrices, strict=True)
    ):
        axis = figure.add_axes((left, 0.365, 0.16, 0.285))
        _style_axis(axis)
        axis.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
        axis.set_aspect("equal")
        axis.text(
            0.97, 0.035, "$X_1$", transform=axis.transAxes, ha="right", color=MUTED, fontsize=10
        )
        axis.set_ylabel("$X_2$", color=MUTED, fontsize=11, labelpad=1)
        axis.axhline(0, color=EDGE, linewidth=0.8)
        axis.axvline(0, color=EDGE, linewidth=0.8)
        latent = axis.scatter(
            *latent_clouds[index].T, s=8, color=color, alpha=0.22, edgecolors="none", zorder=2
        )
        latent_scatters.append(latent)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        axis.add_patch(
            Ellipse(
                (0, 0),
                3.6 * np.sqrt(eigenvalues[-1]),
                3.6 * np.sqrt(eigenvalues[0]),
                angle=angle,
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                alpha=0.9,
            )
        )
        vertical = axis.axvline(0, color=GOLD, linewidth=2.3, alpha=0, zorder=4)
        horizontal = axis.axhline(0, color=GOLD, linewidth=2.3, alpha=0, zorder=4)
        scans.append((vertical, horizontal))
        observed_scatters.append(
            axis.scatter([], [], s=20, color=color, alpha=0, edgecolors="none", zorder=5)
        )
        badges.append(
            axis.text(
                0.5,
                0.96,
                "HIDDEN",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                color=MUTED,
                weight="bold",
                bbox={"facecolor": CARD, "edgecolor": "none", "pad": 2.5},
                zorder=8,
            )
        )
        sign = "+0.8" if index == 0 else "−0.8"
        figure.text(
            left + 0.08,
            0.315,
            f"$V_{'+' if index == 0 else '-'}$   covariance {sign}",
            ha="center",
            fontsize=12,
            color=color,
        )
    hidden_note = figure.text(
        0.2475,
        0.273,
        "One measured axis per independent draw",
        ha="center",
        fontsize=11,
        color=MUTED,
    )

    locations = np.linspace(-3.2, 3.2, 301)
    density = np.exp(-(locations**2) / 2) / np.sqrt(2 * np.pi)
    marginal_axes, markers = [], []
    for coordinate, bottom in enumerate((0.52, 0.325)):
        axis = figure.add_axes((0.505, bottom, 0.205, 0.115))
        _style_axis(axis)
        axis.set(xlim=(-3.2, 3.2), ylim=(0, 0.45), xticks=[-2, 0, 2], yticks=[])
        axis.spines[["top", "left", "right"]].set_visible(False)
        axis.plot(locations, density, color=CYAN, linewidth=3.8, solid_capstyle="round")
        axis.plot(locations, density, color=CORAL, linewidth=2, linestyle=(0, (3, 3)))
        axis.text(
            0,
            1.11,
            f"Coordinate $X_{coordinate + 1}$",
            transform=axis.transAxes,
            fontsize=12,
            color=MUTED,
        )
        (marker,) = axis.plot(
            [],
            [],
            "o",
            color=GOLD,
            markersize=8,
            markeredgecolor=TEXT,
            markeredgewidth=0.8,
            zorder=5,
        )
        markers.append(marker)
        marginal_axes.append(axis)
    figure.text(
        0.6075, 0.273, "$V_+$  /  $V_-$: the same $N(0,1)$", ha="center", fontsize=11, color=MUTED
    )

    risk_axis = figure.add_axes((0.795, 0.37, 0.14, 0.255))
    _style_axis(risk_axis)
    risk_axis.set(
        ylim=(0, 4.35),
        xlim=(-0.65, 1.65),
        yticks=[0, 2, 4],
        xticks=[0, 1],
        xticklabels=["$V_+$", "$V_-$"],
    )
    risk_axis.spines[["top", "right", "left"]].set_visible(False)
    risk_axis.tick_params(axis="x", labelsize=14)
    risks = np.array([0.4, 3.6])
    bars = risk_axis.bar([0, 1], risks, color=[CYAN, CORAL], width=0.55, zorder=3)
    for position, value in enumerate(risks):
        risk_axis.text(
            position,
            value + 0.14,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=20,
            weight="bold",
            color=TEXT,
        )
    figure.text(
        0.86,
        0.656,
        r"$\mathrm{Var}(X_1-X_2)=2-2\,\mathrm{Cov}(X_1,X_2)$",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )
    figure.text(
        0.86, 0.289, "9× spread variance", ha="center", fontsize=16, weight="bold", color=TEXT
    )

    phase_title = figure.text(0.035, 0.173, "", fontsize=17, color=GOLD, weight="bold")
    phase_note = figure.text(0.035, 0.128, "", fontsize=13, color=MUTED)
    progress_background = plt.Line2D(
        [0.035, 0.965],
        [0.087, 0.087],
        transform=figure.transFigure,
        color=EDGE,
        linewidth=3,
        solid_capstyle="round",
    )
    progress = plt.Line2D(
        [0.035, 0.035],
        [0.087, 0.087],
        transform=figure.transFigure,
        color=GOLD,
        linewidth=3,
        solid_capstyle="round",
    )
    figure.add_artist(progress_background)
    figure.add_artist(progress)
    figure.text(
        0.035,
        0.035,
        "COMPANION DIAGNOSTIC  •  Gaussian innovation geometry",
        fontsize=10,
        color=MUTED,
    )
    figure.text(0.965, 0.035, "Known marginal variances = 1", ha="right", fontsize=10, color=MUTED)

    def update(fraction):
        fraction = float(np.clip(fraction, 0, 1))
        joint_stage = fraction >= 0.76
        progress.set_xdata([0.035, 0.035 + 0.93 * fraction])
        if not joint_stage:
            draw_position = fraction * 24
            draw_index = min(int(draw_position), len(independent_scalar_draws) - 1)
            coordinate = draw_index % 2
            scalar = independent_scalar_draws[draw_index]
            pulse = np.sin(np.pi * (draw_position % 1)) ** 0.7
            phase_title.set_text(
                f"Independent draw {draw_index + 1:02d}: observe $X_{coordinate + 1}$ only"
            )
            phase_note.set_text(
                "The other coordinate is missing. No within-draw cross-product is recorded."
            )
            subtitle.set_text(
                "Identical marginals. Opposite covariance. Very different spread risk."
            )
            hidden_title.set_text("HIDDEN JOINT TRUTH")
            hidden_note.set_text("One measured axis per independent draw")
            for pair in scans:
                pair[0].set_xdata([scalar, scalar])
                pair[1].set_ydata([scalar, scalar])
                pair[0].set_alpha(0.7 * pulse if coordinate == 0 else 0)
                pair[1].set_alpha(0.7 * pulse if coordinate == 1 else 0)
            for index, marker in enumerate(markers):
                marker.set_data([scalar], [np.exp(-(scalar**2) / 2) / np.sqrt(2 * np.pi)])
                marker.set_alpha(pulse if index == coordinate else 0)
                marginal_axes[index].patch.set_alpha(0)
            for cloud, observed, badge in zip(
                latent_scatters, observed_scatters, badges, strict=True
            ):
                cloud.set_alpha(0.22)
                observed.set_alpha(0)
                badge.set_text("HIDDEN")
                badge.set_color(MUTED)
        else:
            phase = (fraction - 0.76) / 0.24
            count = max(3, min(95, int(95 * min(1, phase * 1.5))))
            phase_title.set_text("Change the design: observe $X_1$ and $X_2$ together")
            phase_note.set_text(
                "Joint observations supply the missing cross-products and reveal the orientation."
            )
            subtitle.set_text(
                "The marginal curves still match. Paired observations distinguish the models."
            )
            hidden_title.set_text("NOW: JOINT OBSERVATIONS")
            hidden_note.set_text("A new observation design reveals covariance")
            for pair in scans:
                pair[0].set_alpha(0)
                pair[1].set_alpha(0)
            for marker in markers:
                marker.set_alpha(0)
            for index, (cloud, observed, badge) in enumerate(
                zip(latent_scatters, observed_scatters, badges, strict=True)
            ):
                cloud.set_alpha(0.07)
                observed.set_offsets(joint_samples[index][:count])
                observed.set_alpha(0.88)
                badge.set_text("PAIRED DATA")
                badge.set_color(GOLD)
        for bar in bars:
            bar.set_alpha(0.92)

    return figure, update, matrices, risks


def render_observability_video(output_dir=None, *, seed=1729, duration=12.0, fps=24):
    """Create MP4, compact GIF, poster and explicit scientific metadata."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(duration) or not 10 <= duration <= 15:
        raise ValueError("duration must lie between 10 and 15 seconds")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 12:
        raise ValueError("fps must be an integer of at least 12")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render the video and GIF preview")
    output_dir = (
        Path(__file__).resolve().parents[1] / "assets" if output_dir is None else Path(output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4 = output_dir / "observability.mp4"
    preview = output_dir / "observability-preview.gif"
    poster = output_dir / "observability-poster.png"
    figure, update, covariances, risks = _scene(seed)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={
            "title": "What the observations cannot tell you",
            "comment": "Gaussian innovation geometry; explicit independent unpaired and paired designs",
        },
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "19",
            "-preset",
            "medium",
        ],
    )
    frame_count = round(duration * fps)
    with writer.saving(figure, str(mp4), dpi=100):
        for frame in range(frame_count):
            update(frame / (frame_count - 1))
            writer.grab_frame(facecolor=BACKGROUND)
    update(0.52)
    figure.savefig(poster, dpi=100, facecolor=BACKGROUND)
    plt.close(figure)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-filter_complex",
            "[0:v]fps=10,scale=768:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(preview),
        ],
        check=True,
    )
    exposure = np.array([1.0, -1.0])
    np.testing.assert_allclose(
        [exposure @ covariance @ exposure for covariance in covariances], risks
    )
    metadata = {
        "title": "What the observations cannot tell you",
        "kind": "scientific animation; Gaussian innovation geometry companion diagnostic",
        "seed": seed,
        "duration_seconds": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": 1280,
        "height": 720,
        "codec": "H.264 / libx264",
        "pixel_format": "yuv420p",
        "faststart": True,
        "models": {"V_plus": covariances[0].tolist(), "V_minus": covariances[1].tolist()},
        "marginals": "Each coordinate is N(0,1) under both centered Gaussian models.",
        "initial_observation_design": "One coordinate is measured per fresh independent draw; no paired values or within-draw cross-products are observed.",
        "scan_construction": "Independent scalar N(0,1) draws, generated separately from all latent joint clouds; exactly one scan axis and one marginal marker visible per draw.",
        "latent_clouds": "Illustrative hidden joint truth, explicitly labelled; not the data available to the unpaired design.",
        "final_design_change_at_seconds": 0.76 * duration,
        "final_design": "Both coordinates observed jointly on each new draw; paired samples reveal covariance orientation while marginal laws remain unchanged.",
        "spread_exposure": exposure.tolist(),
        "spread_variance_formula": "a.T @ V @ a = 2 - 2*Cov(X1,X2), a=(1,-1)",
        "spread_variances": risks.tolist(),
        "risk_variance_ratio": 9,
        "scope": "Toy centered Gaussian innovation experiment, not an SPDE trajectory simulation, curve calibration, forecasting result, or novelty claim.",
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "files": {"video": mp4.name, "preview": preview.name, "poster": poster.name},
        "file_sizes_bytes": {path.name: path.stat().st_size for path in (mp4, preview, poster)},
        "reproduction": f"python examples/render_observability_video.py --seed {seed}",
    }
    (output_dir / "observability.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    arguments = parser.parse_args()
    metadata = render_observability_video(arguments.output_dir, seed=arguments.seed)
    print(
        json.dumps(
            {"title": metadata["title"], "file_sizes_bytes": metadata["file_sizes_bytes"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
