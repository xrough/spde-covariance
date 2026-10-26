"""Run reproducible transport-innovation covariance benchmarks."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("benchmark", "run the paired synthetic benchmark suite"),
        ("benchmark-rq2", "run local covariance and risk estimation experiments"),
        ("benchmark-rq3", "evaluate frozen covariance forecasts on future observations"),
        ("benchmark-hedging", "audit partial identification and finite-instrument hedging"),
        (
            "benchmark-statistical-completion",
            "evaluate Gaussian completion and uncertainty under partial observations",
        ),
        ("benchmark-infinite-transport", "audit a fixed infinite-rank transport model"),
        (
            "benchmark-partial",
            "study partial innovations, covariance selection and fragment completion",
        ),
        ("empirical-shfe", "calibrate local SHFE copper curves and evaluate subsequent risk"),
        ("empirical-research", "run the exploratory futures risk research notebook suite"),
    ):
        benchmark = commands.add_parser(command, help=help_text)
        benchmark.add_argument("config", type=Path)
        benchmark.add_argument("--output", type=Path)
        if command in ("empirical-shfe", "empirical-research"):
            benchmark.add_argument("--data", required=True, type=Path)
            benchmark.add_argument("--export", type=Path, help="export aggregate evidence only")
        if command == "empirical-research":
            benchmark.add_argument("--baseline-run", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "empirical-research":
            from .empirical_data import load_shfe_cu_archives
            from .empirical_research import (
                export_research_evidence,
                load_research_config,
                run_research_suite,
            )

            config = load_research_config(arguments.config)
            paths = sorted(arguments.data.glob("shfe-futures-*.zip"))
            if not paths:
                raise ValueError("no shfe-futures-YYYY.zip archives in --data directory")
            panel = load_shfe_cu_archives(paths)
            output = arguments.output or arguments.data / "runs" / config.name
            run_research_suite(panel, config, output, baseline_run=arguments.baseline_run)
            if arguments.export is not None:
                export_research_evidence(output, arguments.export)
        elif arguments.command == "empirical-shfe":
            from .empirical_data import audit_shfe_daily_samples, load_shfe_cu_archives
            from .empirical_study import (
                export_empirical_evidence,
                load_empirical_config,
                run_empirical_study,
            )

            config = load_empirical_config(arguments.config)
            paths = sorted(arguments.data.glob("shfe-futures-*.zip"))
            if not paths:
                raise ValueError("no shfe-futures-YYYY.zip archives in --data directory")
            panel = load_shfe_cu_archives(paths)
            output = arguments.output or arguments.data / "runs" / config.name
            samples = sorted((arguments.data / "daily_samples").glob("kx*.dat"))
            checks = audit_shfe_daily_samples(panel, samples) if samples else None
            run_empirical_study(panel, config, output, data_validation=checks)
            if arguments.export is not None:
                export_empirical_evidence(output, arguments.export)
        elif arguments.command == "benchmark-statistical-completion":
            from .statistical_completion_benchmark import (
                load_statistical_completion_config,
                run_statistical_completion,
            )

            config = load_statistical_completion_config(arguments.config)
            output = arguments.output or Path("experiments/runs") / config.name
            run_statistical_completion(config, output)
        elif arguments.command == "benchmark-hedging":
            from .hedging_pilot import run_hedging_pilot

            output = run_hedging_pilot(arguments.config, arguments.output)
        elif arguments.command == "benchmark-partial":
            from .partial_benchmark import run_partial_benchmark

            output = run_partial_benchmark(arguments.config, arguments.output)
        elif arguments.command == "benchmark-infinite-transport":
            from .infinite_benchmark import run_infinite_transport_benchmark

            output = run_infinite_transport_benchmark(arguments.config, arguments.output)
        elif arguments.command == "benchmark-rq3":
            from .rq3 import run_rq3_benchmark

            output = run_rq3_benchmark(arguments.config, arguments.output)
        elif arguments.command == "benchmark-rq2":
            from .rq2 import run_rq2_benchmark

            output = run_rq2_benchmark(arguments.config, arguments.output)
        else:
            from .benchmarks import run_benchmark

            output = run_benchmark(arguments.config, arguments.output)
    except (ValueError, FileExistsError, OSError, TypeError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
