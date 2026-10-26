"""Research-suite integration, provenance and public/local data boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_research import (
    ResearchSuiteConfig,
    data_and_target_diagnostics,
    export_research_evidence,
    load_research_config,
    load_research_evidence,
    representation_diagnostics,
    run_research_suite,
)
from spde_covariance.empirical_study import EmpiricalConfig, run_empirical_study
from spde_covariance.laguerre import LaguerreTransport


@pytest.fixture(scope="module")
def research_input():
    dates = np.arange(np.datetime64("2022-01-03"), np.datetime64("2022-04-01"))
    dates = dates[np.is_busday(dates)]
    months = np.arange(np.datetime64("2022-02", "M"), np.datetime64("2023-05", "M"))
    model = LaguerreTransport(3)
    rows = []
    for i, day in enumerate(dates):
        for j, month in enumerate(months):
            maturity = month.astype("datetime64[D]") + np.timedelta64(14, "D")
            if maturity <= day:
                continue
            a = np.array([60000 + 80 * np.sin(i), 1000 + 20 * np.cos(i / 3), 100 * np.sin(i / 2)])
            price = float(
                model.evaluate([(maturity - day).astype(int) / 365.25])[0] @ a + 4 * np.cos(i + j)
            )
            rows.append((day, "cu" + str(month)[2:4] + str(month)[5:7], maturity, price, 100, 1000))
    panel = ShfeCuPanel(
        *zip(*rows, strict=True), dates, provenance={"raw_prices": [99999], "synthetic": True}
    )
    config = EmpiricalConfig(
        training_end="2022-01-31",
        validation_end="2022-02-28",
        test_end="2022-03-31",
        windows=(8, 16),
        min_direct_count=3,
        bootstrap_replications=20,
        sensitivities=False,
    )
    return panel, config


def test_research_config_validation(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('name="smoke"\nscore_blocks=[2,4]\n[empirical]\nwindows=[8,16]\n')
    loaded = load_research_config(path)
    assert loaded.empirical.windows == (8, 16)
    assert loaded.score_blocks == (2, 4)
    for kwargs in (
        {"name": "../bad"},
        {"score_blocks": (0,)},
        {"representation_dimensions": (0,)},
        {"representation_smoothing": (float("nan"),)},
    ):
        with pytest.raises(ValueError):
            ResearchSuiteConfig(**kwargs)


def test_data_summary_is_aggregate_and_does_not_mutate(research_input):
    panel, config = research_input
    prices = panel.settlement.copy()
    result = data_and_target_diagnostics(panel, config)
    assert result["sessions"] == len(panel.session_dates)
    assert sum(r["rows"] for r in result["tenor_coverage"]) == len(panel.dates)
    assert result["selected_origins"] > 0
    assert all(
        row["count"] == result["selected_origins"] for row in result["same_contract_history"]
    )
    assert "raw_prices" not in json.dumps(result)
    assert "contract_ids" not in json.dumps(result)
    np.testing.assert_array_equal(prices, panel.settlement)


def test_representation_decomposition_identity(research_input):
    panel, empirical = research_input
    config = ResearchSuiteConfig(
        empirical=empirical, representation_dimensions=(2, 3), representation_smoothing=(1e-6,)
    )
    result = representation_diagnostics(panel, config)
    assert len(result["fits"]) == 2
    for row in result["fits"]:
        assert row["available_sessions"] == len(panel.session_dates)
        for exposure in row["decomposition"]["exposures"]:
            assert exposure["identity_relative_error"] < 1e-12


def test_midmonth_coordinate_is_not_expiry_and_audit_retains_negative_tenors():
    dates = np.array(["2022-02-16", "2022-02-17"], dtype="datetime64[D]")
    panel = ShfeCuPanel(
        dates,
        ["cu2202"] * 2,
        ["2022-02-15"] * 2,
        [60000.0, 60010.0],
        [100] * 2,
        [1000] * 2,
        dates,
        provenance={"synthetic": True},
    )
    result = data_and_target_diagnostics(panel, EmpiricalConfig())
    assert result["tenor_coverage"][0]["rows"] == 2
    assert sum(row["rows"] for row in result["tenor_coverage"]) == len(panel.dates)
    assert result["outside_tenor_bins_count"] == 0


def test_complete_research_suite_and_verified_export(research_input, tmp_path):
    panel, empirical = research_input
    baseline = tmp_path / "baseline"
    run_empirical_study(panel, empirical, baseline)
    config = ResearchSuiteConfig(
        empirical=empirical,
        score_blocks=(2, 4),
        representation_dimensions=(3,),
        representation_smoothing=(1e-6,),
    )
    local = run_research_suite(panel, config, tmp_path / "research", baseline_run=baseline)
    public = export_research_evidence(local, tmp_path / "public")
    assert len(list(public.glob("*.json"))) == 8
    assert load_research_evidence(public)["data"]["sessions"] == len(panel.session_dates)
    manifest = json.loads((public / "manifest.json").read_text())
    assert "raw_prices" not in json.dumps(manifest)
    assert "empirical_joint.py" in manifest["source_sha256"]
    protocol = json.loads((public / "protocol.json").read_text())
    assert "exploratory" in protocol["status"]
    assert "untouched" in protocol["holdout"]
    assert protocol["baseline_data_verification"]["unique_scored_targets_checked"] > 0
    # An unrelated panel must not be combined with the frozen score records.
    other_panel = ShfeCuPanel(
        panel.dates,
        panel.contract_ids,
        panel.contract_month_dates,
        panel.settlement * 1.01,
        panel.volume,
        panel.open_interest,
        panel.session_dates,
        provenance=panel.provenance,
    )
    with pytest.raises(ValueError, match="outcomes differ"):
        run_research_suite(other_panel, config, tmp_path / "wrong-data", baseline_run=baseline)
    manifest_path = baseline / "manifest.json"
    old_manifest = json.loads(manifest_path.read_text())
    changed_manifest = dict(
        old_manifest, data_provenance={"archives": [{"year": 2022, "sha256": "a" * 64, "bytes": 1}]}
    )
    manifest_path.write_text(json.dumps(changed_manifest))
    with pytest.raises(ValueError, match="archive identities"):
        run_research_suite(panel, config, tmp_path / "wrong-archive", baseline_run=baseline)
    manifest_path.write_text(json.dumps(old_manifest))
    (local / "data.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        export_research_evidence(local, tmp_path / "bad-public")
    (public / "joint.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_research_evidence(public)
    with pytest.raises(ValueError, match="protocol mismatch"):
        run_research_suite(
            panel,
            replace(config, empirical=replace(empirical, smoothing=1e-4)),
            tmp_path / "mismatch",
            baseline_run=baseline,
        )
