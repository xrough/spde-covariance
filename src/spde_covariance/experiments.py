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
        ("benchmark-infinite-transport", "audit a fixed infinite-rank transport model"),
        (
            "benchmark-partial",
            "study partial innovations, covariance selection and fragment completion",
        ),
    ):
        benchmark = commands.add_parser(command, help=help_text)
        benchmark.add_argument("config", type=Path)
        benchmark.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "benchmark-hedging":
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
