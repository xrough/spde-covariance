"""Known-truth controls do not require market files or an exchange importer."""

import numpy as np
import pytest

from spde_covariance.mechanism_oracle import (
    residual_oracle_control,
    run_oracle_controls,
    transport_oracle_control,
)


def test_residual_oracle_retains_exact_correlated_blocks():
    result = residual_oracle_control(replications=2, intervals=40, seed=19)
    assert result["oracle_identity_relative_error"] < 1e-12
    assert result["oracle_component_relative_error"] < 1e-12
    rows = {row["policy"]: row for row in result["rows"]}
    assert rows["oracle_full_joint"]["relative_kernel_error"] < 1e-12
    assert rows["oracle_omit_cross"]["relative_kernel_error"] > 0.01
    assert rows["oracle_factor_only"]["risk_ratio"]["spread"] < 1
    assert result["sampled_full_joint_relative_kernel_error"]["replications"] == 2


def test_transport_oracle_uses_known_truth_and_is_reproducible():
    first = transport_oracle_control(replications=2, intervals=40, seed=19)
    second = transport_oracle_control(replications=2, intervals=40, seed=19)
    assert first == second
    assert first["oracle_C_relative_kernel_error"] == 0
    assert first["maximum_relative_covariance_certificate"] <= 1e-5
    assert first["coefficient_relative_error"]["mean"] > 0
    assert len(first["rows"]) == 8
    for row in first["rows"]:
        assert np.isfinite(row["relative_kernel_error"]["mean"])
        assert row["relative_kernel_error"]["mcse"] >= 0


def test_oracle_minimum_sample_requirements():
    with pytest.raises(ValueError, match="at least"):
        run_oracle_controls(replications=1)
