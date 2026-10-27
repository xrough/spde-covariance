"""Controlled contrasts change one covariance layer and preserve information."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spde_covariance.empirical_calibration import fit_cross_section
from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_mechanism_models import (
    _reconstruct,
    _source_donor_exclusions,
    known_source_kernel,
    mechanism_covariances,
)
from spde_covariance.empirical_short_history_data import (
    ShortHistoryConfig,
    baseline_covariances,
    build_short_history_view,
    daily_sections,
    make_short_history_episodes,
)
from spde_covariance.empirical_short_history_models import BoundedModelCache, model_covariances
from spde_covariance.laguerre import LaguerreTransport


@pytest.fixture(scope="module")
def panel_config():
    rng = np.random.default_rng(20261027)
    dates = np.busday_offset(np.datetime64("2023-09-01"), np.arange(55))
    months = np.arange(np.datetime64("2023-11", "M"), np.datetime64("2024-12", "M"))
    maturities = months.astype("datetime64[D]") + np.timedelta64(14, "D")
    model = LaguerreTransport(4, cache_max_bytes=0)
    state = np.array([65000.0, 1000.0, -400.0, 90.0])
    rows = []
    for i, date in enumerate(dates):
        if i:
            delta = int((date - dates[i - 1]).astype(int)) / 365.25
            state = model.matrix(delta) @ state + rng.normal(size=4) * [65, 40, 25, 15]
        days = (maturities - date).astype(int)
        prices = model.evaluate(np.maximum(days, 0) / 365.25) @ state
        prices += 15 * np.sin(days / 29 + i / 4)
        for month, maturity, price in zip(months, maturities, prices, strict=True):
            identifier = "cu" + str(month)[2:4] + str(month)[5:7]
            rows.append((date, identifier, maturity, price, 100, 1000))
    panel = ShfeCuPanel(*zip(*rows, strict=True), dates, provenance={"synthetic": True})
    config = ShortHistoryConfig(
        training_end="2023-08-01",
        validation_end="2023-09-01",
        test_end="2023-12-31",
        window=20,
        origin_stride=1,
        min_training_intervals=15,
        bootstrap_replications=10,
    )
    return panel, config


def _view(panel_config, budget=0, future_days=1):
    panel, config = panel_config
    episodes = make_short_history_episodes(panel, config)
    episode = [
        item for item in episodes if int((item.next_date - item.origin).astype(int)) == future_days
    ][-1]
    return build_short_history_view(panel, episode, budget, config), config


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_six_cells_psd_components_and_frozen_residual_statistics(panel_config, dimension):
    view, config = _view(panel_config)
    result = mechanism_covariances(view, config, dimension=dimension)
    assert len(result) == 7
    for name, fit in result.items():
        assert fit["available"], (name, fit["reason"])
        covariance = fit["covariance"]
        assert np.linalg.eigvalsh(covariance)[0] >= -1e-8 * np.linalg.norm(covariance)
        assert_allclose(sum(fit["components"].values()), covariance, atol=1e-7)
    for component in ("correlated_residual", "diagonal_prediction_error"):
        assert_allclose(
            result["source_diagonal"]["components"][component],
            result["empirical_diagonal"]["components"][component],
            atol=0,
        )
    assert_allclose(
        result["source_diagonal"]["covariance"] - result["source_correlated"]["covariance"],
        result["empirical_diagonal"]["covariance"] - result["empirical_correlated"]["covariance"],
        atol=1e-7,
    )
    source = result["source_factor"]["diagnostics"]["source_fit"]
    assert source["converged"]
    assert source["relative_covariance_error_bound"] <= config.numerical_error_budget
    assert source["positive_weight_interval_count"] == 20
    assert result["source_factor"]["diagnostics"]["reconstruction"]["continuum_tail_error"] is None


def test_primary_dimension_three_preserves_existing_source_policies(panel_config):
    view, config = _view(panel_config, budget=5)
    old = model_covariances(view, config)
    new = mechanism_covariances(view, config)
    for new_name, old_name in (
        ("source_factor", "spde_ewma"),
        ("source_correlated", "spde_joint_no_nugget"),
        ("source_diagonal", "spde_joint"),
    ):
        assert old[old_name]["available"]
        assert_allclose(
            new[new_name]["covariance"], old[old_name]["covariance"], rtol=2e-9, atol=2e-6
        )


@pytest.mark.parametrize("budget", [0, 5, 20, None])
def test_reconstructed_pool_has_exact_raw_baseline_count_and_weights(panel_config, budget):
    view, config = _view(panel_config, budget)
    raw = baseline_covariances(view, config)["pooled_shrink"]
    fit = mechanism_covariances(view, config)["reconstructed_pool"]
    assert fit["available"]
    diagnostics = fit["diagnostics"]
    for key in ("complete_changes", "pooled_changes", "short_blend_weight"):
        assert diagnostics[key] == raw["diagnostics"][key]
    assert_allclose(
        diagnostics["matched_raw_pool_covariance"], raw["covariance"], rtol=3e-15, atol=1e-9
    )
    assert diagnostics["reconstruction_identity_relative_residual"] < 1e-14
    for exposure in diagnostics["reconstruction_retention"].values():
        assert_allclose(
            exposure["raw_second_moment"],
            exposure["fitted_second_moment"]
            + exposure["residual_second_moment"]
            + exposure["signed_cross_second_moment"],
            rtol=2e-10,
            atol=1e-6,
        )


def test_one_day_policy_retains_original_ages_and_explicit_future_availability(panel_config):
    view, config = _view(panel_config)
    prepared = _reconstruct(view, config, 3)
    deltas = prepared["deltas"]
    selected = np.isclose(deltas, 1 / 365.25, atol=1e-14, rtol=0)
    weights = np.exp2(-np.arange(len(deltas) - 1, -1, -1) / config.half_life)
    weights[~selected] = 0
    weights /= weights.sum()
    result = mechanism_covariances(view, config, interval_policy="one_day")
    diagnostic = result["source_factor"]["diagnostics"]
    assert result["empirical_factor"]["available"]
    assert diagnostic["selected_training_interval_count"] == np.count_nonzero(selected)
    assert_allclose(diagnostic["effective_interval_count"], 1 / (weights @ weights))
    assert_allclose(diagnostic["source_fit"]["interval_weights"], weights, atol=1e-16)
    weekend, config = _view(panel_config, future_days=3)
    other = mechanism_covariances(weekend, config, interval_policy="one_day")
    assert other["source_factor"]["available"]
    for name in ("empirical_factor", "empirical_correlated", "empirical_diagonal"):
        assert not other[name]["available"]
        assert "multi-day" in other[name]["reason"]


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_batched_donor_exclusion_matches_independent_augmented_lstsq(panel_config, dimension):
    view, config = _view(panel_config)
    section = daily_sections(view)[-1]
    date = view.origin
    excluded, diagnostic = _source_donor_exclusions(section, date, config, dimension)
    assert diagnostic["normal_equation_residual_max"] < 1e-8
    tenors = (section["maturity"] - date).astype(int) / 365.25
    for k, identifier in enumerate(section["ids"]):
        keep = np.arange(len(tenors)) != k
        independent = fit_cross_section(
            tenors[keep], section["prices"][keep], dimension=dimension, smoothing=config.smoothing
        )
        assert independent.available
        assert_allclose(excluded[identifier], independent.coefficients, rtol=5e-8, atol=1e-5)
    changed_section = dict(section)
    prices = np.array(section["prices"])
    prices[2] += 10000
    changed_section["prices"] = prices
    changed, _ = _source_donor_exclusions(changed_section, date, config, dimension)
    assert_allclose(changed[section["ids"][2]], excluded[section["ids"][2]], rtol=5e-8, atol=1e-5)


def test_hidden_and_future_perturbations_cache_and_order_invariance(panel_config):
    panel, config = panel_config
    view, _ = _view(panel_config)
    prices = panel.settlement.copy()
    forbidden = ((panel.contract_ids == view.target_id) & (panel.dates < view.origin)) | (
        panel.dates > view.origin
    )
    prices[forbidden] += 50000
    changed = ShfeCuPanel(
        panel.dates,
        panel.contract_ids,
        panel.contract_month_dates,
        prices,
        panel.volume,
        panel.open_interest,
        panel.session_dates,
    )
    changed_view = build_short_history_view(changed, view.episode, 0, config)
    assert view.fingerprint == changed_view.fingerprint
    cache = BoundedModelCache(max_entries=3, max_bytes=2 * 1024**2)
    first = mechanism_covariances(view, config, cache=cache)
    mechanism_covariances(view, config, dimension=2, interval_policy="one_day", cache=cache)
    second = mechanism_covariances(changed_view, config, cache=cache)
    for name in first:
        assert first[name]["available"] == second[name]["available"]
        if first[name]["available"]:
            assert_allclose(first[name]["covariance"], second[name]["covariance"], atol=0)
    assert cache.entries <= 3
    assert cache.bytes <= 2 * 1024**2
    with pytest.raises(TypeError):
        first["source_factor"]["available"] = False
    with pytest.raises(ValueError):
        first["source_factor"]["covariance"][0, 0] = 0
    changed_config = replace(config, target_tenors_days=(60, 180, 300))
    other = mechanism_covariances(view, changed_config, cache=cache)
    reference = mechanism_covariances(view, changed_config)
    assert_allclose(
        other["reconstructed_pool"]["covariance"],
        reference["reconstructed_pool"]["covariance"],
        atol=0,
    )


def test_known_source_oracle_matches_independent_quadrature():
    source = np.array([[9.0, 0.8, 0.1], [0.8, 4.0, -0.3], [0.1, -0.3, 2.0]])
    tenors = np.array([80, 170, 260])
    delta = 4 / 365.25
    actual = known_source_kernel(source, tenors, delta)
    model = LaguerreTransport(3, cache_max_bytes=0)
    evaluation = model.evaluate(tenors / 365.25)
    points, weights = np.polynomial.legendre.leggauss(30)
    expected = np.zeros((3, 3))
    for point, weight in zip(points, weights, strict=True):
        shift = model.matrix((point + 1) * delta / 2)
        expected += weight * delta / 2 * evaluation @ shift @ source @ shift.T @ evaluation.T
    assert_allclose(actual, expected, atol=2e-17)
    with pytest.raises(ValueError, match="square"):
        known_source_kernel(np.ones((2, 3)), tenors, delta)


def test_explicit_unavailable_and_input_validation(panel_config):
    view, config = _view(panel_config)
    with pytest.raises(ValueError, match="dimensions"):
        mechanism_covariances(view, config, dimension=5)
    with pytest.raises(ValueError, match="dimensions"):
        mechanism_covariances(view, config, dimension=3.0)
    with pytest.raises(ValueError, match="interval_policy"):
        mechanism_covariances(view, config, interval_policy="compressed")
    unavailable = replace(view, available=False, reason="test-unavailable")
    assert all(
        item["reason"] == "test-unavailable"
        for item in mechanism_covariances(unavailable, config).values()
    )
    with pytest.raises(TypeError):
        mechanism_covariances(view, config, latent_truth=np.eye(3))


def test_covariance_rows_share_one_residual_fit_and_only_replace_h(panel_config, monkeypatch):
    import spde_covariance.empirical_mechanism_models as models

    view, config = _view(panel_config)
    original = models._residual_statistics
    calls = []

    def record_call(*args, **kwargs):
        calls.append(np.asarray(args[3]).copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(models, "_residual_statistics", record_call)
    first = models.mechanism_covariances(view, config)
    assert len(calls) == 1
    original_source = models.fit_source_covariance

    def changed_source(*args, **kwargs):
        fit = original_source(*args, **kwargs)
        return replace(fit, source_covariance=2 * fit.source_covariance)

    monkeypatch.setattr(models, "fit_source_covariance", changed_source)
    second = models.mechanism_covariances(view, config)
    assert len(calls) == 2
    for name in (
        "empirical_factor",
        "empirical_correlated",
        "empirical_diagonal",
        "reconstructed_pool",
    ):
        assert_allclose(first[name]["covariance"], second[name]["covariance"], atol=0)
    for component in ("correlated_residual", "diagonal_prediction_error"):
        assert_allclose(
            first["source_diagonal"]["components"][component],
            second["source_diagonal"]["components"][component],
            atol=0,
        )
    assert_allclose(
        second["source_factor"]["covariance"], 2 * first["source_factor"]["covariance"], atol=1e-7
    )


def test_unavailable_diagonal_does_not_hide_available_correlated_fit(panel_config, monkeypatch):
    import spde_covariance.empirical_mechanism_models as models

    view, config = _view(panel_config)
    original = models._source_donor_exclusions

    def failed_exclusion(*args, **kwargs):
        raise ValueError("deliberate donor reconstruction failure")

    monkeypatch.setattr(models, "_source_donor_exclusions", failed_exclusion)
    result = models.mechanism_covariances(view, config)
    for row in ("source", "empirical"):
        assert result[f"{row}_factor"]["available"]
        assert result[f"{row}_correlated"]["available"]
        assert not result[f"{row}_diagonal"]["available"]
        assert "donor reconstruction" in result[f"{row}_diagonal"]["reason"]
    monkeypatch.setattr(models, "_source_donor_exclusions", original)


def test_exact_shift_links_coefficient_innovations_to_same_id_fitted_changes(panel_config):
    view, config = _view(panel_config)
    for dimension in (2, 3, 4):
        prepared = _reconstruct(view, config, dimension)
        model = LaguerreTransport(dimension, cache_max_bytes=0)
        for j, record in enumerate(prepared["records"]):
            end_tenors = np.asarray(record["tenors_days"]) / 365.25
            delta = float(record["delta"])
            fitted_raw_change = (
                model.evaluate(end_tenors) @ prepared["states"][j + 1]
                - model.evaluate(end_tenors + delta) @ prepared["states"][j]
            )
            adjusted = model.evaluate(end_tenors) @ prepared["innovations"][j] * np.sqrt(delta)
            assert_allclose(fitted_raw_change, adjusted, atol=4e-11)


def test_source_failure_does_not_change_empirical_or_pool_controls(panel_config, monkeypatch):
    import spde_covariance.empirical_mechanism_models as models

    view, config = _view(panel_config)
    before = models.mechanism_covariances(view, config)

    def failed_source(*args, **kwargs):
        raise ValueError("deliberate source failure")

    monkeypatch.setattr(models, "fit_source_covariance", failed_source)
    after = models.mechanism_covariances(view, config)
    for name in (
        "empirical_factor",
        "empirical_correlated",
        "empirical_diagonal",
        "reconstructed_pool",
    ):
        assert after[name]["available"]
        assert_allclose(before[name]["covariance"], after[name]["covariance"], atol=0)
    for name in ("source_factor", "source_correlated", "source_diagonal"):
        assert not after[name]["available"]
        assert after[name]["reason"] == "deliberate source failure"
