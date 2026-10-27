"""Independent algebra and information-boundary checks for maturity risk transfer."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spde_covariance.empirical_data import ShfeCuPanel
from spde_covariance.empirical_short_history_data import (
    ShortHistoryConfig,
    build_short_history_view,
    make_short_history_episodes,
)
from spde_covariance.empirical_short_history_models import (
    _dfm_training,
    _residual_statistics,
    interval_weights,
    model_covariances,
    orthonormal_spline_basis,
    residual_maturity_features,
    schur_residual_forecast,
)
from spde_covariance.laguerre import LaguerreTransport


def test_spline_uniform_metric_and_hat_partition():
    nodes, weights = np.polynomial.legendre.leggauss(8)
    gram = np.zeros((6, 6))
    for left, right in ((30, 120), (120, 240), (240, 365)):
        x = left + (nodes + 1) * (right - left) / 2
        phi = orthonormal_spline_basis(x)
        gram += phi.T @ (weights[:, None] * phi) * (right - left) / 670
    assert_allclose(gram, np.eye(6), atol=3e-14)
    hats = residual_maturity_features(np.array([20.0, 30.0, 100.0, 250.0, 365.0, 370.0]))
    assert_allclose(hats.sum(axis=1), 1)
    assert np.all(hats >= 0)
    with pytest.raises(ValueError, match="30,365"):
        orthonormal_spline_basis([20])


def test_schur_forecast_equals_full_joint_kernel_including_cross_terms():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(200, 7))
    values[:, 3:] += values[:, :3] @ rng.normal(size=(3, 4))
    moment = values.T @ values / len(values)
    b, psi = rng.normal(size=(3, 3)), rng.normal(size=(3, 4))
    kernel, components, diagnostics = schur_residual_forecast(b, psi, moment, moment[:3, :3])
    direct = np.column_stack((b, psi)) @ moment @ np.column_stack((b, psi)).T
    assert_allclose(kernel, direct, rtol=2e-13, atol=3e-13)
    assert_allclose(sum(components.values()), kernel, atol=1e-13)
    assert diagnostics["relative_support_residual"] < 1e-14
    assert np.linalg.norm(components["cross_correction"]) > 0


def test_singular_compatible_support_and_material_incompatibility():
    z = np.array([[1.0, 0.0, 2.0], [-2.0, 0.0, 1.0], [3.0, 0.0, -1.0]])
    moment = z.T @ z / len(z)
    b, psi = np.eye(2), np.ones((2, 1))
    result, _, diagnostics = schur_residual_forecast(b, psi, moment, moment[:2, :2])
    assert diagnostics["factor_numerical_rank"] == 1
    assert np.linalg.eigvalsh(result)[0] >= -1e-13
    incompatible = np.array([[1.0, 0.0, 0.0], [0.0, 1e-12, 1e-6], [0.0, 1e-6, 1.0]])
    with pytest.raises(ValueError, match="support"):
        schur_residual_forecast(b, psi, incompatible, incompatible[:2, :2])
    with pytest.raises(ValueError, match="indefinite"):
        schur_residual_forecast(b, psi, np.diag([1.0, 1.0, -1.0]), np.eye(2))


def test_factor_rotation_and_sign_invariance():
    rng = np.random.default_rng(9)
    data = rng.normal(size=(100, 7))
    moment = data.T @ data / len(data)
    b, psi = rng.normal(size=(3, 3)), rng.normal(size=(3, 4))
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    transform = np.eye(7)
    transform[:3, :3] = q
    first = schur_residual_forecast(b, psi, moment, moment[:3, :3])[0]
    transformed = transform.T @ moment @ transform
    second = schur_residual_forecast(b @ q, psi, transformed, q.T @ moment[:3, :3] @ q)[0]
    assert_allclose(first, second, atol=3e-14)


def test_dfm_uncentered_pca_and_donor_exclusion():
    rng = np.random.default_rng(2)
    x = np.linspace(30.0, 365.0, 10)
    records = tuple({"tenors_days": x, "response": 100 + rng.normal(size=10)} for _ in range(20))
    weights = interval_weights(20, 6)
    loading, factors, main, exclude, diagnostics = _dfm_training(records, weights)
    assert not diagnostics["moment_centered"]
    assert diagnostics["coefficient_second_moment_eigenvalues"][0] > 5000
    # With frozen loading shapes, a removed donor's own response cannot affect
    # its leave-one-out factor score.
    first = exclude(2, 4)
    records[2]["response"][4] += 1e6
    second = exclude(2, 4)
    assert_allclose(first, second, atol=0)
    moment, diagonal, residual = _residual_statistics(
        records, factors, main, weights, main_loo=exclude
    )
    assert moment.shape == (7, 7)
    assert np.all(diagonal >= 0)
    assert residual["leave_one_donor_out_count"] == 200
    assert loading.shape == (6, 3)


@pytest.fixture(scope="module")
def panel_config():
    rng = np.random.default_rng(14)
    dates = np.busday_offset(np.datetime64("2023-09-01"), np.arange(65))
    months = np.arange(np.datetime64("2023-11", "M"), np.datetime64("2024-12", "M"))
    maturities = months.astype("datetime64[D]") + np.timedelta64(14, "D")
    model = LaguerreTransport(3)
    state = np.array([60000.0, 1000.0, -400.0])
    rows = []
    for i, date in enumerate(dates):
        if i:
            delta = int((date - dates[i - 1]).astype(int)) / 365.25
            state = model.matrix(delta) @ state + rng.normal(size=3) * np.array([70, 40, 25])
        days = (maturities - date).astype(int)
        prices = model.evaluate(np.maximum(days, 0) / 365.25) @ state
        prices += 10 * np.sin(days / 21 + i / 3)
        for month, maturity, price in zip(months, maturities, prices, strict=True):
            identifier = "cu" + str(month)[2:4] + str(month)[5:7]
            rows.append((date, identifier, maturity, price, 100, 1000))
    panel = ShfeCuPanel(*zip(*rows, strict=True), dates, provenance={"synthetic": True})
    config = ShortHistoryConfig(
        training_end="2023-08-01",
        validation_end="2023-09-01",
        test_end="2023-12-31",
        window=20,
        min_training_intervals=15,
        bootstrap_replications=10,
    )
    return panel, config


def test_end_to_end_masked_models_feasible_and_component_identity(panel_config):
    panel, config = panel_config
    episode = make_short_history_episodes(panel, config)[-3]
    view = build_short_history_view(panel, episode, 0, config)
    result = model_covariances(view, config)
    for name, payload in result.items():
        assert payload["available"], (name, payload["reason"])
        covariance = payload["covariance"]
        assert covariance.shape == (3, 3)
        assert np.linalg.eigvalsh(covariance)[0] >= -1e-8 * np.linalg.norm(covariance)
        assert_allclose(sum(payload["components"].values()), covariance, atol=1e-7)
    assert np.trace(result["dfm_joint"]["covariance"]) >= np.trace(
        result["dfm_joint_no_nugget"]["covariance"]
    )
    assert np.trace(result["spde_joint"]["covariance"]) >= np.trace(
        result["spde_joint_no_nugget"]["covariance"]
    )
    assert result["dfm_joint"]["diagnostics"]["leave_one_donor_out_count"] > 100


def test_future_and_hidden_prices_do_not_change_fits(panel_config):
    panel, config = panel_config
    episode = make_short_history_episodes(panel, config)[-6]
    view = build_short_history_view(panel, episode, 0, config)
    original = model_covariances(view, config)
    prices = panel.settlement.copy()
    forbidden = ((panel.contract_ids == episode.target_id) & (panel.dates < episode.origin)) | (
        panel.dates > episode.origin
    )
    prices[forbidden] += 123456.0
    changed = ShfeCuPanel(
        panel.dates,
        panel.contract_ids,
        panel.contract_month_dates,
        prices,
        panel.volume,
        panel.open_interest,
        panel.session_dates,
    )
    other_view = build_short_history_view(changed, episode, 0, config)
    assert other_view.fingerprint == view.fingerprint
    altered = model_covariances(other_view, config)
    for name in original:
        assert original[name]["available"] == altered[name]["available"]
        if original[name]["available"]:
            assert_allclose(original[name]["covariance"], altered[name]["covariance"], atol=0)


def test_ewma_is_dynamic_and_uniform_is_half_life_independent(panel_config):
    panel, config = panel_config
    episode = make_short_history_episodes(panel, config)[-3]
    view = build_short_history_view(panel, episode, 5, config)
    first, second = model_covariances(view, config, 5), model_covariances(view, config, 100)
    assert_allclose(
        first["dfm_uniform_joint"]["covariance"], second["dfm_uniform_joint"]["covariance"], atol=0
    )
    assert_allclose(
        first["spde_uniform"]["covariance"], second["spde_uniform"]["covariance"], atol=0
    )
    assert not np.allclose(first["dfm_joint"]["covariance"], second["dfm_joint"]["covariance"])


def test_nugget_predicts_exclusion_error_not_identified_noise():
    rng = np.random.default_rng(88)
    x = np.linspace(30.0, 365.0, 12)
    records = tuple({"tenors_days": x, "response": rng.normal(size=12)} for _ in range(400))
    factors = np.zeros((400, 1))
    main = [np.zeros((12, 1))] * 400
    _, diagonal, diagnostics = _residual_statistics(
        records, factors, main, interval_weights(400), main_loo=lambda interval, donor: np.zeros(1)
    )
    # Held-out prediction errors include estimating the four-feature curve
    # from eleven donors, so they exceed the unit latent variance here.
    mean_prediction_error = float(np.mean(residual_maturity_features(x) @ diagonal))
    assert 1.1 < mean_prediction_error < 2
    assert_allclose(
        mean_prediction_error, diagnostics["leave_one_donor_out_annual_mse"], rtol=1e-12
    )
    assert "not identified latent noise" in diagnostics["diagonal_scope"]


def test_bounded_cache_preserves_results_and_processing_order(panel_config):
    from spde_covariance.empirical_short_history_models import BoundedModelCache

    panel, config = panel_config
    episode = make_short_history_episodes(panel, config)[-3]
    first = build_short_history_view(panel, episode, 0, config)
    second = build_short_history_view(panel, episode, 5, config)
    cache = BoundedModelCache(max_entries=1, max_bytes=64 * 1024**2)
    result = model_covariances(first, config, 5, cache=cache)
    model_covariances(first, config, 100, cache=cache)
    assert cache.hits == 1
    model_covariances(second, config, 5, cache=cache)
    again = model_covariances(first, config, 5, cache=cache)
    assert cache.entries == 1
    assert cache.bytes <= cache.max_bytes
    for name in result:
        assert_allclose(result[name]["covariance"], again[name]["covariance"], atol=0)
    tiny = BoundedModelCache(max_entries=64, max_bytes=10)
    tiny.put("oversized", np.ones(100))
    assert tiny.entries == tiny.bytes == 0


def test_batched_source_exclusion_matches_independent_augmented_lstsq(panel_config):
    from spde_covariance.empirical_short_history_data import daily_sections
    from spde_covariance.empirical_short_history_models import (
        _source_donor_exclusions,
        _source_without_donor,
    )

    panel, config = panel_config
    episode = make_short_history_episodes(panel, config)[-3]
    view = build_short_history_view(panel, episode, 0, config)
    sections = daily_sections(view)
    for date, section in zip(view.panel.session_dates[::4], sections[::4], strict=True):
        batched = _source_donor_exclusions(section, date, config)
        for donor in section["ids"]:
            independent = _source_without_donor(section, date, donor, config)
            assert_allclose(batched[donor], independent, rtol=2e-10, atol=2e-6)
