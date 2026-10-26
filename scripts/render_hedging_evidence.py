"""Re-render hedge figures/report without recomputing or changing numerical results.

Usage: python scripts/render_hedging_evidence.py RAW_RUN NEW_DIRECTORY
The numerical manifest remains unchanged; rendering has its own clean-source
manifest. Publish the new directory with publish_benchmark_evidence.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from publish_benchmark_evidence import _json, _verify_source

from spde_covariance.hedging_pilot import HedgingPilotConfig
from spde_covariance.hedging_reporting import write_hedging_report
from spde_covariance.provenance import _manifest, _source_root

PRESENTATION = {"report.md", "identification.png", "future-risk.png", "hedge-positions.png"}


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_run(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"destination already exists: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("destination must be outside the original run")
    files = sorted(source.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("source must contain regular files only")
    if (source / "rendering_manifest.json").exists():
        raise ValueError("source must be an original raw run")
    numerical_manifest = _json(source / "manifest.json")
    _verify_source(source, numerical_manifest)
    results = _json(source / "results.json")
    if (
        results["acceptance"]["passed"] is not True
        or _json(source / "status.json")["status"] != "complete"
    ):
        raise ValueError("source run must be complete and accepted")
    config = HedgingPilotConfig(**_json(source / "config.json"))
    if numerical_manifest["config"] != _json(source / "config.json"):
        raise ValueError("configuration differs from the numerical manifest")
    root = _source_root()
    rendering = _manifest(config, root)
    if root is None or rendering["git_dirty"] is not False:
        raise ValueError("commit the rendering source before generating evidence")
    script = Path(__file__).resolve()
    rendering["project_file_sha256"][script.relative_to(root).as_posix()] = _hash(script)
    rendering["experiment"] = "presentation-only re-rendering of an unchanged accepted hedge run"
    rendering["solver"] = {"name": "Matplotlib rendering; no estimation or optimization"}
    rendering["numerical_revision"] = numerical_manifest["git_revision"]
    rendering["original_presentation_sha256"] = {
        name: _hash(source / name) for name in sorted(PRESENTATION)
    }
    rendering["unchanged_numerical_file_sha256"] = {
        path.name: _hash(path) for path in files if path.name not in PRESENTATION
    }
    shutil.copytree(source, output)
    write_hedging_report(
        output,
        config,
        _json(source / "control.json"),
        _json(source / "population.json"),
        results,
    )
    report = output / "report.md"
    report.write_text(
        report.read_text(encoding="utf-8")
        + "\nFigures and presentation were re-rendered from unchanged numerical artifacts. "
        "The [rendering manifest](rendering_manifest.json) records that separate source revision; "
        "the original numerical manifest is unchanged.\n",
        encoding="utf-8",
    )
    for name, expected in rendering["unchanged_numerical_file_sha256"].items():
        if _hash(output / name) != expected:
            raise RuntimeError(f"rendering changed a numerical artifact: {name}")
    rendering["rendered_presentation_sha256"] = {
        name: _hash(output / name) for name in sorted(PRESENTATION)
    }
    (output / "rendering_manifest.json").write_text(
        json.dumps(rendering, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(render_run(args.source, args.output))


if __name__ == "__main__":
    main()
