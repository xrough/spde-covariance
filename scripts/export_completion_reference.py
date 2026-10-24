"""Export deterministic CSV fixtures; the separate base-R script checks parity."""

import argparse

from spde_covariance.fragment_reference import export_author_fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="directory for CSV fixtures and provenance")
    args = parser.parse_args()
    rows = export_author_fixtures(args.output)
    print(f"Exported {len(rows)} explicit-rank author-reference fixtures to {args.output}")


if __name__ == "__main__":
    main()
