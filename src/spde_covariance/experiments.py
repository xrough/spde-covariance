"""Run reproducible transport-innovation covariance benchmarks."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="run the paired synthetic benchmark suite")
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        from .benchmarks import run_benchmark

        output = run_benchmark(arguments.config, arguments.output)
    except (ValueError, FileExistsError, OSError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
