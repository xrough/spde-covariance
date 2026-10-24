"""Reproduce matched-factor contrasts from saved ablation estimates (no fitting)."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTRASTS = {
    "PSD coherence at rank2": ("coherent_eta0_zero_rank2", "pairwise_rank2"),
    "PSD coherence at full support": ("coherent_eta0_zero_support", "pairwise_support"),
    "entrywise shrinkage at rank2": ("ridge_rank2", "pairwise_rank2"),
    "shrinkage within coherent support": (
        "coherent_eta0.001_zero_support",
        "coherent_eta0_zero_support",
    ),
    "positive floor within coherent support": (
        "coherent_eta0.001_resolution_support",
        "coherent_eta0.001_zero_support",
    ),
    "rank2 versus support after coherent fitting": (
        "coherent_eta0.001_resolution_rank2",
        "coherent_eta0.001_resolution_support",
    ),
    "global PSD projection after raw rank2": ("pairwise_rank2_psd", "pairwise_rank2"),
    "stability guard versus validation alone": ("stability_guarded", "validation_only"),
}
METRICS = ("complete_weighted_error", "missing_weighted_error", "test_observed_score")
FIELDS = ("model", "base_grid", "cover", "N")


def contrasts(rows):
    """Form differences within independent replications before computing MCSE."""
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[f] for f in FIELDS)
        slot = (int(row["replication"]), row["policy"])
        if slot in groups[key]:
            raise ValueError("duplicate policy/replication in an ablation cell")
        groups[key][slot] = row
    records = []
    for key, data in groups.items():
        replications = sorted({rep for rep, _ in data})
        for name, (first, second) in CONTRASTS.items():
            for metric in METRICS:
                values = []
                for rep in replications:
                    a, b = data.get((rep, first)), data.get((rep, second))
                    if a is None or b is None:
                        continue
                    if not all(str(r["available"]).lower() == "true" for r in (a, b)):
                        continue
                    if any(r.get(metric) in (None, "") for r in (a, b)):
                        continue
                    difference = float(a[metric]) - float(b[metric])
                    if not np.isfinite(difference):
                        raise ValueError("nonfinite paired contrast")
                    values.append(difference)
                records.append(
                    {
                        **dict(zip(FIELDS, key, strict=True)),
                        "contrast": name,
                        "first": first,
                        "second": second,
                        "metric": metric,
                        "replications": len(replications),
                        "available_pairs": len(values),
                        "mean": float(np.mean(values)) if values else None,
                        "mcse": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else None,
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be a new directory")
    opener = gzip.open if args.estimates.suffix == ".gz" else open
    with opener(args.estimates, "rt", newline="") as handle:
        records = contrasts(list(csv.DictReader(handle)))
    args.output.mkdir(parents=True)
    (args.output / "paired.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    with (args.output / "paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "input": str(args.estimates),
        "input_sha256": hashlib.sha256(args.estimates.read_bytes()).hexdigest(),
        "script": "scripts/completion_contrasts.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": "first minus second; paired within replication; MCSE across replications",
        "scope": "postprocessing only; no fitting, model or parameter selection",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
