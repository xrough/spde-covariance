"""Plot covariance nonidentifiability under unpaired Gaussian observations.

Run: python examples/render_observability_video.py
Uses NumPy, Matplotlib, and ffmpeg with libx264. The MP4, GIF, static figure,
and model/observation metadata are written to assets by default.
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
from matplotlib.patches import Ellipse

COLORS = ("#1f77b4", "#ff7f0e")
JOINT_START = 0.76


def _scene(seed):
    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    layout = figure.add_gridspec(
        2,
        3,
        width_ratios=(1, 1, 0.82),
        left=0.065,
        right=0.97,
        bottom=0.19,
        top=0.86,
        wspace=0.42,
        hspace=0.46,
    )
    figure.suptitle("Unpaired Gaussian observations and covariance identification", fontsize=15)

    rng = np.random.default_rng(seed)
    matrices = (np.array([[1.0, 0.8], [0.8, 1.0]]), np.array([[1.0, -0.8], [-0.8, 1.0]]))
    latent = [rng.multivariate_normal(np.zeros(2), matrix, size=210) for matrix in matrices]
    paired = [rng.multivariate_normal(np.zeros(2), matrix, size=95) for matrix in matrices]
    # Independently generated scalar observations: alternating coordinates are
    # measured on different draws. Consecutive values never form a joint pair.
    unpaired = rng.standard_normal(200)
    observed_pairs, initial_legends, paired_legends = [], [], []
    contour_radius = np.sqrt(-2 * np.log(0.05))
    for index, (matrix, color) in enumerate(zip(matrices, COLORS, strict=True)):
        axis = figure.add_subplot(layout[index, 0])
        axis.set(
            xlim=(-3.2, 3.2),
            ylim=(-3.2, 3.2),
            xticks=[-2, 0, 2],
            yticks=[-2, 0, 2],
            xlabel="$X_1$",
            ylabel="$X_2$",
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.2, linewidth=0.6)
        sign, rho = ("+", 0.8) if index == 0 else ("-", -0.8)
        axis.set_title(f"({'a' if index == 0 else 'b'}) Latent $V_{sign}$, $\\rho={rho:g}$")
        cloud = axis.scatter(
            *latent[index].T, s=8, c="0.6", alpha=0.32, edgecolors="none", label="Latent draws"
        )
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        direction = eigenvectors[:, -1]
        ellipse = Ellipse(
            (0, 0),
            2 * contour_radius * np.sqrt(eigenvalues[-1]),
            2 * contour_radius * np.sqrt(eigenvalues[0]),
            angle=np.degrees(np.arctan2(direction[1], direction[0])),
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
            label="95% Gaussian contour",
        )
        axis.add_patch(ellipse)
        observed = axis.scatter(
            [], [], s=17, marker="x", c=color, linewidths=0.9, label="New paired data"
        )
        observed_pairs.append(observed)
        first_legend = axis.legend(
            handles=[cloud, ellipse],
            loc="upper left",
            fontsize=7.5,
            frameon=False,
            handlelength=1.6,
        )
        axis.add_artist(first_legend)
        second_legend = axis.legend(
            handles=[cloud, observed],
            loc="upper left",
            fontsize=7.5,
            frameon=False,
            handlelength=1.6,
        )
        second_legend.set_visible(False)
        initial_legends.append(first_legend)
        paired_legends.append(second_legend)

    locations = np.linspace(-3.2, 3.2, 301)
    density = np.exp(-(locations**2) / 2) / np.sqrt(2 * np.pi)
    rugs = []
    for coordinate in range(2):
        axis = figure.add_subplot(layout[coordinate, 1])
        axis.set(
            xlim=(-3.2, 3.2),
            ylim=(-0.065, 0.47),
            xticks=[-2, 0, 2],
            yticks=[0, 0.2, 0.4],
            xlabel=f"$X_{coordinate + 1}$",
            ylabel="Density",
        )
        axis.set_title(
            f"({'c' if coordinate == 0 else 'd'}) Marginal distribution of $X_{coordinate + 1}$"
        )
        axis.plot(locations, density, color=COLORS[0], linewidth=1.6, label="$V_+$")
        axis.plot(locations, density, color=COLORS[1], linewidth=1.6, linestyle="--", label="$V_-$")
        (rug,) = axis.plot(
            [],
            [],
            "|",
            color="0.3",
            alpha=0.65,
            markersize=6,
            markeredgewidth=0.8,
            label="Unpaired data",
        )
        rugs.append(rug)
        axis.axhline(0, color="0.7", linewidth=0.6)
        axis.legend(loc="upper right", frameon=False, fontsize=8)

    risk_axis = figure.add_subplot(layout[:, 2])
    risks = np.array([0.4, 3.6])
    risk_axis.bar([0, 1], risks, color=COLORS, width=0.55, alpha=0.82)
    risk_axis.set(
        xlim=(-0.7, 1.7),
        ylim=(0, 4.15),
        xticks=[0, 1],
        xticklabels=["$V_+$", "$V_-$"],
        yticks=[0, 1, 2, 3, 4],
        ylabel="$\\mathrm{Var}(X_1-X_2)$",
        title="(e) Spread variance",
    )
    risk_axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    risk_axis.set_axisbelow(True)
    for position, value in enumerate(risks):
        risk_axis.text(position, value + 0.08, f"{value:.1f}", ha="center", fontsize=11)
    risk_axis.text(
        0.5,
        0.95,
        "$\\mathrm{Var}(X_1-X_2)=2-2\\rho$",
        transform=risk_axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    design_caption = figure.text(0.065, 0.105, "", fontsize=10.5)
    figure.text(
        0.065,
        0.06,
        "Both models are centered with unit marginal variances. Latent joint samples are reference information.",
        fontsize=9,
        color="0.25",
    )
    figure.text(
        0.065,
        0.027,
        "Gaussian innovation geometry example; this figure is not an SPDE trajectory simulation.",
        fontsize=9,
        color="0.25",
    )

    def update(fraction):
        fraction = float(np.clip(fraction, 0, 1))
        scalar_count = min(200, int(200 * min(fraction / JOINT_START, 1)))
        for coordinate, rug in enumerate(rugs):
            values = unpaired[:scalar_count][coordinate::2]
            rug.set_data(values, np.full(len(values), -0.032))
        joint_stage = fraction >= JOINT_START
        for before, after in zip(initial_legends, paired_legends, strict=True):
            before.set_visible(not joint_stage)
            after.set_visible(joint_stage)
        if not joint_stage:
            for observed in observed_pairs:
                observed.set_offsets(np.empty((0, 2)))
            first_count, second_count = (scalar_count + 1) // 2, scalar_count // 2
            design_caption.set_text(
                f"Unpaired design: one coordinate per independent draw; "
                f"$n_1={first_count}$, $n_2={second_count}$. No within-draw cross-products."
            )
        else:
            count = min(95, max(1, int(95 * (fraction - JOINT_START) / (1 - JOINT_START))))
            for index, observed in enumerate(observed_pairs):
                observed.set_offsets(paired[index][:count])
            design_caption.set_text(
                f"New paired-observation design: both coordinates measured together; "
                f"$n={count}$ per model. Cross-products are available."
            )

    return figure, update, matrices, risks


def render_observability_video(output_dir=None, *, seed=1729, duration=12.0, fps=24):
    """Render fixed-scale scientific plots and save their observation assumptions."""
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
            "title": "Unpaired Gaussian observations and covariance identification",
            "comment": "Latent joint references, independent unpaired observations, and new paired observations",
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
            writer.grab_frame(facecolor="white")
    update(0.6)
    figure.savefig(poster, dpi=100, facecolor="white")
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
        "title": "Unpaired Gaussian observations and covariance identification",
        "kind": "Matplotlib scientific figure animation; Gaussian innovation geometry",
        "seed": seed,
        "duration_seconds": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": 1280,
        "height": 720,
        "codec": "H.264 / libx264",
        "pixel_format": "yuv420p",
        "faststart": True,
        "plot_style": "White Matplotlib figure, standard axes and legends, fixed scales, standard blue/orange colors",
        "models": {"V_plus": covariances[0].tolist(), "V_minus": covariances[1].tolist()},
        "marginals": "Each coordinate is N(0,1) under both centered Gaussian models.",
        "initial_observation_design": "One coordinate is measured per fresh independent draw; no paired values or within-draw cross-products are observed.",
        "unpaired_sample_construction": "200 independent scalar N(0,1) draws generated separately from all latent joint clouds; alternating draws observe different coordinates; rugs accumulate separately.",
        "latent_clouds": "210 reference samples per model, labelled latent. These joint pairs are not observed by the unpaired design.",
        "gaussian_contour_probability": 0.95,
        "final_design_change_at_seconds": JOINT_START * duration,
        "final_design": "95 new joint samples per model, added as x markers only after the design changes to paired observation. Marginal laws remain unchanged.",
        "spread_exposure": exposure.tolist(),
        "spread_variance_formula": "a.T @ V @ a = 2 - 2*Cov(X1,X2), a=(1,-1)",
        "spread_variances": risks.tolist(),
        "risk_variance_ratio": 9,
        "scope": "Centered Gaussian innovation geometry; not an SPDE trajectory, calibration or forecasting result.",
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
