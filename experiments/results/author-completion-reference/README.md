# Pinned author-R explicit-rank verification

The [successful Ubuntu workflow](https://github.com/xrough/spde-covariance/actions/runs/40053664886) compared eight identical matrix fixtures with the pinned author implementation. All passed; maximum absolute discrepancy was **8.33e-13**. This checks explicit ranks, including the raw-indefiniteness control, and makes no claim about the legacy FVE normalization.

[r-parity-results.csv](r-parity-results.csv) records the comparisons. [provenance.json](provenance.json) pins the author revision and source checksum; [verification.json](verification.json) records the clean toolkit source revision and the downloaded workflow artifact hashes. Fixtures are reproducible from that revision via `python scripts/export_completion_reference.py OUTPUT`, then `Rscript scripts/check_author_completion.R OUTPUT`. The [fixture manifest](manifest.csv) declares ranks and join counts.
