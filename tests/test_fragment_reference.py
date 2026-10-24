import csv
import json

import numpy as np
import pytest

from spde_covariance.fragment_reference import (
    AUTHOR_REVISION,
    AUTHOR_SOURCE_SHA256,
    PAPER_MODELS,
    clique_mask,
    cover_cliques,
    export_author_fixtures,
    noncanonical_control,
    paper_cover,
    paper_kernel,
    population_controls,
    truncation_psd_control,
)


def test_original_kernel_formulas_and_finite_rank():
    nodes = np.linspace(0, 1, 21)
    assert paper_kernel("K1", [0])[0, 0] == pytest.approx(4.625)
    assert np.linalg.matrix_rank(paper_kernel("finite_rank", nodes), tol=1e-10) == 4
    np.testing.assert_array_equal(paper_kernel("K2", nodes), np.minimum.outer(nodes, nodes))
    assert paper_kernel("K3", [0.5, 1])[0, 1] == pytest.approx(5 * np.exp(-2.5))
    for name in PAPER_MODELS:
        kernel = paper_kernel(name, nodes)
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-14)
        assert np.linalg.eigvalsh(kernel)[0] >= -1e-12


@pytest.mark.parametrize("count", [2, 3, 5, 9, 17])
def test_cover_contains_paper_closed_intervals(count):
    intervals = np.asarray(paper_cover(count))
    np.testing.assert_allclose(intervals[:, 1] - intervals[:, 0], 0.6)
    np.testing.assert_allclose(intervals[:, 0], np.linspace(0, 0.4, count))
    scaled = np.asarray(paper_cover(count, domain=(2, 7)))
    np.testing.assert_allclose(scaled, 2 + 5 * intervals)


def test_noncanonical_family_same_observations_different_unseen_risk():
    nodes = np.linspace(0, 1, 41)
    mask = clique_mask(len(nodes), cover_cliques(nodes, paper_cover(2)))
    kernels = [noncanonical_control(nodes, rho) for rho in (-1, 0, 1)]
    for kernel in kernels:
        np.testing.assert_allclose(kernel[mask], kernels[1][mask], atol=1e-14, rtol=0)
        assert np.linalg.eigvalsh(kernel)[0] >= -1e-12
    risks = [k[0, 0] + k[-1, -1] - 2 * k[0, -1] for k in kernels]
    np.testing.assert_allclose(risks, [4, 2, 0], atol=1e-14)
    assert not mask[0, -1]


def test_exact_psd_control_is_valid_and_missing_pair_is_unobserved():
    control = truncation_psd_control()
    assert np.linalg.eigvalsh(control["kernel"])[0] >= -1e-14
    assert not control["mask"][0, 3]
    assert np.count_nonzero(~control["mask"]) == 2


@pytest.mark.parametrize("nodes", [[], [[0, 1]], [np.nan], [-0.1, 0.5], [1.01]])
def test_invalid_nodes_rejected(nodes):
    with pytest.raises(ValueError):
        paper_kernel("brownian", nodes)
    with pytest.raises(ValueError):
        noncanonical_control(nodes)


@pytest.mark.parametrize("count", [1, 4, True, 3.0])
def test_invalid_paper_cover_count_rejected(count):
    with pytest.raises(ValueError):
        paper_cover(count)


def test_other_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="unknown"):
        paper_kernel("unknown", [0, 1])
    for rho in (-1.01, 1.01, np.nan):
        with pytest.raises(ValueError):
            noncanonical_control([0, 1], rho)
    with pytest.raises(ValueError):
        paper_cover(3, (1, 0))
    with pytest.raises(ValueError):
        population_controls(10)


def test_population_controls_expose_bias_and_psd_failure():
    results = population_controls()
    assert {(r["model"], r["cover"]) for r in results["paper"]} == {
        (name, m) for name in PAPER_MODELS for m in (3, 9)
    }
    assert all(row["observed_max_abs_error"] == 0 for row in results["paper"])
    assert [row["rank"] for row in results["paper"]] == [4, 4, 2, 2, 2, 2]
    assert results["noncanonical"]["observed_difference_max"] < 1e-14
    assert results["noncanonical"]["canonical_completion_error"] < 1e-10
    risk_range = results["noncanonical"]["risk_range"]
    assert risk_range["lower"] == pytest.approx(0)
    assert risk_range["upper"] == pytest.approx(4)
    assert risk_range["canonical"] == pytest.approx(2)
    assert not risk_range["identified"]
    assert risk_range["lower_witness_min_eigenvalue"] >= -1e-11
    assert risk_range["upper_witness_min_eigenvalue"] >= -1e-11
    assert results["truncation"]["rank1_min_eigenvalue"] == pytest.approx(1 - np.sqrt(2))
    assert results["truncation"]["rank1_missing_value"] == pytest.approx(0)
    assert results["truncation"]["rank2_missing_value"] == pytest.approx(1)
    json.dumps(results, allow_nan=False)


def test_fixture_export_is_reproducible_and_declares_pinned_reference(tmp_path):
    rows = export_author_fixtures(tmp_path)
    assert len(rows) == 8
    with (tmp_path / "manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    assert len(manifest) == 8
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["revision"] == AUTHOR_REVISION
    assert provenance["sha256"] == AUTHOR_SOURCE_SHA256
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    export_author_fixtures(tmp_path)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for row in manifest:
        name = row["name"]
        expected = np.loadtxt(tmp_path / f"{name}-expected.csv", delimiter=",")
        original = np.loadtxt(tmp_path / f"{name}-input.csv", delimiter=",")
        mask = np.loadtxt(tmp_path / f"{name}-mask.csv", delimiter=",").astype(bool)
        np.testing.assert_array_equal(expected[mask], original[mask])
