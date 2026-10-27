"""The process runner reproduces serial forecasts using only masked views."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from spde_covariance.empirical_short_history import (
    EVIDENCE_FILES,
    export_short_history_evidence,
    load_short_history_evidence,
    run_short_history_benchmark,
)
from spde_covariance.empirical_short_history_data import ShortHistoryConfig


def _small_study():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples/short_history_pipeline.py"))
    config = ShortHistoryConfig(
        name="serial-process-runner-control",
        training_end="2022-01-31",
        validation_end="2022-02-28",
        test_end="2022-03-31",
        window=20,
        min_training_intervals=16,
        origin_stride=40,
        history_budgets=(0, None),
        bootstrap_replications=10,
        sensitivity_half_lives=(),
        cache_entries=4,
        cache_bytes=4 * 1024**2,
    )
    return example["synthetic_panel"](), config


def test_serial_process_and_repeat_runs_reproduce_forecasts_and_export(tmp_path, monkeypatch):
    # Child processes must use the same explicit numerical-thread policy.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    panel, config = _small_study()
    paths = [tmp_path / name for name in ("serial", "repeat", "parallel")]
    results = [
        run_short_history_benchmark(panel, config, path, workers=workers)
        for path, workers in zip(paths, (1, 1, 2), strict=True)
    ]
    initial = results[0]
    assert initial["manifest"]["episode_count"] == 6
    assert initial["manifest"]["issued_count"] == 120
    for result in results[1:]:
        assert result["summary"] == initial["summary"]
        assert result["paired"] == initial["paired"]
        assert result["protocol"] == initial["protocol"]
        assert result["diagnostics"]["methods"] == initial["diagnostics"]["methods"]
    hashes = []
    for path, result, workers in zip(paths, results, (1, 1, 2), strict=True):
        local = path / "private_predictions.jsonl"
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        hashes.append(digest)
        assert result["manifest"]["local_records_sha256"] == digest
        assert result["manifest"]["workers"] == workers
        verified = load_short_history_evidence(path)
        assert verified["summary"] == result["summary"]
        assert verified["paired"] == result["paired"]
        cache = result["diagnostics"]["cache"]
        assert cache["max_entries"] == config.cache_entries
        assert cache["max_bytes"] == config.cache_bytes
        if workers == 1:
            assert cache["entries"] <= config.cache_entries
            assert cache["bytes"] <= config.cache_bytes
        else:
            assert cache["workers"] == workers
            assert cache["max_entries_observed"] <= config.cache_entries
            assert cache["max_bytes_observed"] <= config.cache_bytes
        destination = tmp_path / f"published-{path.name}"
        export_short_history_evidence(path, destination)
        assert {file.name for file in destination.iterdir()} == {*EVIDENCE_FILES, "checksums.json"}
        assert not (destination / "private_predictions.jsonl").exists()
        assert load_short_history_evidence(destination) == verified
    assert len(set(hashes)) == 1
    # Scientific products are byte-identical; provenance keeps runtime/workers.
    for name in ("summary.json", "paired.json", "protocol.json"):
        assert len({(path / name).read_bytes() for path in paths}) == 1
    assert json.loads((paths[2] / "manifest.json").read_text())["workers"] == 2
    with pytest.raises(FileExistsError):
        run_short_history_benchmark(panel, config, paths[0])


@pytest.mark.parametrize("workers", [0, -1, 9, True, 1.5])
def test_runner_rejects_invalid_worker_counts_without_creating_output(tmp_path, workers):
    panel, config = _small_study()
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="workers"):
        run_short_history_benchmark(panel, config, output, workers=workers)
    assert not output.exists()
