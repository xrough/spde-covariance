from spde_covariance.benchmarks import BenchmarkConfig
from spde_covariance.provenance import _manifest


def test_installed_package_provenance_does_not_invent_git_checkout():
    manifest = _manifest(BenchmarkConfig(), None)
    assert manifest["source_sha256"]
    assert "spde_covariance/provenance.py" in manifest["source_sha256"]
    assert manifest["git_revision"] is None
    assert manifest["git_dirty"] is None
    assert manifest["project_file_sha256"] == {}
