"""Publish a complete accepted benchmark run into a new compact evidence directory.

Usage: python scripts/publish_benchmark_evidence.py SOURCE OUTPUT
Source hashes are checked against the run's recorded local Git revision, which
need not be the current checkout. The original run and its manifest are preserved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value {value} in {path.name}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _git(directory: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments], capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(
            "recorded source revision or file is unavailable in the local Git repository"
        )
    return result.stdout


def _verify_source(source: Path, manifest: dict) -> None:
    if manifest.get("git_dirty") is not False:
        raise ValueError("only runs recorded from clean source can be published")
    revision = manifest.get("git_revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("manifest must record a full Git revision")
    try:
        repository = Path(_git(source, "rev-parse", "--show-toplevel").decode().strip())
    except ValueError:
        repository = Path(__file__).resolve().parents[1]
    if _git(repository, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("recorded Git revision must identify a commit")
    for field, prefix in (("source_sha256", "src"), ("project_file_sha256", "")):
        hashes = manifest.get(field)
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"manifest must contain nonempty {field}")
        for name, expected in hashes.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("manifest hash paths must stay inside the repository")
            data = _git(repository, "show", f"{revision}:{(Path(prefix) / relative).as_posix()}")
            if _sha256(data) != expected:
                raise ValueError(f"{field} does not match the recorded Git revision: {name}")


def _report(data: bytes, compressed: dict[str, str], published_names: set[str]) -> bytes:
    text = data.decode("utf-8")
    for original, published in compressed.items():
        text = text.replace(f"]({original})", f"]({published})")
        text = text.replace(f"](./{original})", f"]({published})")
    notice = (
        "This evidence bundle retains the complete accepted run. CSV files and measurement/audit "
        "JSON files are compressed with gzip; artifact links point to those archives. "
        "Use `gzip -dc FILE.gz > FILE` to read an uncompressed copy. "
        "[Bundle checksums](artifact_sha256.json) record original and published bytes. "
        "The unchanged [manifest](manifest.json) retains the originating clean revision and source hashes."
    )
    heading = "## Artifacts and provenance\n"
    if heading in text:
        text = text.replace(heading, heading + "\n" + notice + "\n", 1)
    else:
        text = text.rstrip() + "\n\n## Compressed evidence\n\n" + notice + "\n"
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http:", "https:", "#", "../")):
            continue
        if target.removeprefix("./").split("#", 1)[0] not in published_names:
            raise ValueError(f"report links to an artifact missing from the bundle: {target}")
    return text.encode("utf-8")


def publish_run(source: Path, output: Path) -> Path:
    """Validate and publish without overwriting a run or an existing destination."""
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"destination already exists: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("destination must be outside the source run")
    files = sorted(source.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("the source run must contain only regular, non-symlink files")
    if (source / "artifact_sha256.json").exists():
        raise ValueError("source must be an unpublished raw run")
    manifest = _json(source / "manifest.json")
    config = _json(source / "config.json")
    results = _json(source / "results.json")
    status = _json(source / "status.json")
    if (
        results.get("acceptance", {}).get("passed") is not True
        or status.get("status") != "complete"
    ):
        raise ValueError("only complete, accepted runs can be published")
    if manifest.get("config") != config:
        raise ValueError("config.json differs from the run manifest")
    _verify_source(source, manifest)
    if not (source / "report.md").is_file():
        raise ValueError("the completed run must contain report.md")
    for path in files:
        if path.suffix == ".json":
            _json(path)
    uncompressed = {"config.json", "manifest.json", "status.json"}
    compressed = {
        path.name: path.name + ".gz"
        for path in files
        if path.suffix in (".json", ".csv") and path.name not in uncompressed
    }
    names = {compressed.get(path.name, path.name) for path in files} | {"artifact_sha256.json"}
    if len(names) != len(files) + 1:
        raise ValueError("source filenames would collide after compression")
    report = _report((source / "report.md").read_bytes(), compressed, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Atomic refusal if another publisher created this destination.
    try:
        index = []
        for path in files:
            data = path.read_bytes()
            name = compressed.get(path.name, path.name)
            published = (
                gzip.compress(data, compresslevel=9, mtime=0) if path.name in compressed else data
            )
            if path.name == "report.md":
                published = report
            (output / name).write_bytes(published)
            index.append(
                {
                    "source_name": path.name,
                    "published_name": name,
                    "source_sha256": _sha256(data),
                    "published_sha256": _sha256(published),
                    "gzip": path.name in compressed,
                }
            )
        (output / "artifact_sha256.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(publish_run(args.source, args.output))


if __name__ == "__main__":
    main()
