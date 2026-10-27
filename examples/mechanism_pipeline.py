"""Offline synthetic mechanism workflow, without SHFE files.

Run from a source checkout. The panel generator is the same software smoke
fixture as the historical short-history example, not empirical evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from short_history_pipeline import synthetic_panel

from spde_covariance.empirical_mechanisms import (
    MechanismConfig,
    export_mechanism_evidence,
    run_mechanism_benchmark,
)
from spde_covariance.empirical_short_history_data import ShortHistoryConfig


def run(output, *, export=None):
    config = MechanismConfig(
        base=ShortHistoryConfig(
            name="copper-mechanisms-synthetic-smoke",
            training_end="2022-01-31",
            validation_end="2022-02-28",
            test_end="2022-03-31",
            window=20,
            min_training_intervals=16,
            origin_stride=10,
            sensitivity_half_lives=(),
            bootstrap_replications=50,
        ),
        oracle_replications=2,
    )
    result = run_mechanism_benchmark(synthetic_panel(), config, output)
    if export is not None:
        export_mechanism_evidence(output, export)
    print("Offline mechanism contrasts completed:", result["manifest"]["issued_count"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.output is None:
        with TemporaryDirectory(prefix="spde-mechanisms-") as folder:
            run(Path(folder) / "run", export=args.export)
    else:
        run(args.output, export=args.export)


if __name__ == "__main__":
    main()
