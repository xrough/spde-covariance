"""Integrity checks for the dependency-free evidence publisher."""

import gzip
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

publish_run = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "publish_benchmark_evidence.py")
)["publish_run"]


def _git(root, *arguments):
    return (
        subprocess.check_output(["git", "-C", str(root), *arguments], stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )


@pytest.fixture
def raw_run(tmp_path):
    repository = tmp_path / "repository"
    package = repository / "src" / "spde_covariance"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n")
    (repository / "pyproject.toml").write_text("[project]\nname='example'\n")
    (repository / "uv.lock").write_text("version = 1\n")
    (repository / ".gitignore").write_text("experiments/runs/\n")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Recorded source",
    )
    revision = _git(repository, "rev-parse", "HEAD")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = {"name": "example", "replications": 2}
    manifest = {
        "git_revision": revision,
        "git_dirty": False,
        "config": config,
        "source_sha256": {"spde_covariance/module.py": digest(package / "module.py")},
        "project_file_sha256": {
            name: digest(repository / name) for name in ("pyproject.toml", "uv.lock")
        },
    }
    source = repository / "experiments" / "runs" / "complete"
    source.mkdir(parents=True)
    for name, value in (
        ("config", config),
        ("manifest", manifest),
        ("results", {"acceptance": {"passed": True}, "scores": [-1.0, 0.2]}),
        ("status", {"status": "complete"}),
        ("geometry", [{"rank": 1}]),
    ):
        (source / f"{name}.json").write_text(json.dumps(value) + "\n")
    (source / "scores.csv").write_text("score\n-1.0\n0.2\n")
    (source / "figure.png").write_bytes(b"unchanged figure bytes")
    (source / "report.md").write_text(
        "# Report\n\n[Results](results.json) · [Scores](scores.csv)\n\n"
        "![Figure](figure.png)\n\n[Protocol](../../../docs/protocol.md)\n"
    )
    return repository, source, tmp_path / "published"


def test_historical_revision_publication_preserves_run_and_verifies_every_byte(raw_run):
    repository, source, output = raw_run
    # A newer checkout must not invalidate a correctly recorded historical run.
    (repository / "src" / "spde_covariance" / "module.py").write_text("VALUE = 2\n")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Newer source",
    )
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    assert publish_run(source, output) == output
    index = json.loads((output / "artifact_sha256.json").read_text())
    assert len(index) == len(original)
    assert {entry["published_name"] for entry in index} | {"artifact_sha256.json"} == {
        path.name for path in output.iterdir()
    }
    for entry in index:
        packed = (output / entry["published_name"]).read_bytes()
        assert hashlib.sha256(original[entry["source_name"]]).hexdigest() == entry["source_sha256"]
        assert hashlib.sha256(packed).hexdigest() == entry["published_sha256"]
        if entry["gzip"]:
            assert gzip.decompress(packed) == original[entry["source_name"]]
            assert int.from_bytes(packed[4:8], "little") == 0
    assert (output / "manifest.json").read_bytes() == original["manifest.json"]
    assert (output / "figure.png").read_bytes() == original["figure.png"]
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    report = (output / "report.md").read_text()
    assert "](results.json.gz)" in report and "](scores.csv.gz)" in report
    assert "## Compressed evidence" in report
    assert "[Bundle checksums](artifact_sha256.json)" in report
    assert "[manifest](manifest.json)" in report
    with pytest.raises(FileExistsError):
        publish_run(source, output)
    assert (output / "manifest.json").read_bytes() == original["manifest.json"]


@pytest.mark.parametrize(
    "problem", ("dirty", "failed", "incomplete", "source_hash", "project_hash", "broken_link")
)
def test_invalid_evidence_is_rejected_without_destination(raw_run, problem):
    _, source, output = raw_run
    if problem in ("dirty", "source_hash", "project_hash"):
        path = source / "manifest.json"
        manifest = json.loads(path.read_text())
        if problem == "dirty":
            manifest["git_dirty"] = True
        else:
            field = "source_sha256" if problem == "source_hash" else "project_file_sha256"
            manifest[field][next(iter(manifest[field]))] = "0" * 64
        path.write_text(json.dumps(manifest))
    elif problem == "failed":
        (source / "results.json").write_text('{"acceptance":{"passed":false}}')
    elif problem == "incomplete":
        (source / "status.json").write_text('{"status":"running"}')
    else:
        (source / "report.md").write_text("[Missing data](missing.csv)\n")
    with pytest.raises(ValueError):
        publish_run(source, output)
    assert not output.exists()


def test_existing_provenance_heading_gets_notice_and_symlinks_are_rejected(raw_run):
    _, source, output = raw_run
    (source / "report.md").write_text(
        "# Report\n\n## Artifacts and provenance\n\n[Results](results.json)\n"
    )
    publish_run(source, output)
    report = (output / "report.md").read_text()
    assert "## Artifacts and provenance\n\nThis evidence bundle" in report
    assert "## Compressed evidence" not in report
    (source / "linked.csv").symlink_to(source / "scores.csv")
    with pytest.raises(ValueError, match="non-symlink"):
        publish_run(source, output.with_name("other"))
