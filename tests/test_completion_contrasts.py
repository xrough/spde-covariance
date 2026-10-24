"""Postprocessing must preserve independent paired replications and availability."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "completion_contrasts", Path(__file__).parents[1] / "scripts/completion_contrasts.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_contrasts_pair_before_mcse_and_count_both_methods():
    rows = []
    for rep, difference in enumerate((1.0, 3.0, 7.0)):
        for policy, value in (
            ("pairwise_rank2", 1000.0 * rep),
            ("coherent_eta0_zero_rank2", 1000.0 * rep + difference),
        ):
            rows.append(
                dict(
                    replication=rep,
                    policy=policy,
                    model="control",
                    base_grid=41,
                    cover=3,
                    N=512,
                    available=rep != 2 or policy != "pairwise_rank2",
                    complete_weighted_error=value,
                )
            )
    row = next(
        r
        for r in module.contrasts(rows)
        if r["contrast"] == "PSD coherence at rank2" and r["metric"] == "complete_weighted_error"
    )
    assert row["available_pairs"] == 2
    assert row["mean"] == 2.0
    assert row["mcse"] == pytest.approx(np.std([1, 3], ddof=1) / np.sqrt(2))
    with pytest.raises(ValueError, match="duplicate"):
        module.contrasts(rows + rows[:1])
