"""Parity booleans must not be mistaken for private price observations."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


def test_public_checker_allows_exact_parity_flags_but_rejects_prices(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "check_mechanisms", root / "scripts/check_mechanisms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.load_mechanism_evidence
    real = original(root / "experiments/results/copper-mechanisms")
    # Use the committed aggregate fixture without exchange files. Avoid testing
    # presentation hashes here; its complete manifest has a separate verifier.
    module.load_mechanism_evidence = lambda directory: real
    module.check(tmp_path)
    changed = deepcopy(real)
    parity = changed["manifest"]["importer_audit"]["price_and_raw_count_parity_2022_2025"]
    parity["settlement"] = [100.0, 101.0]
    module.load_mechanism_evidence = lambda directory: changed
    with pytest.raises(AssertionError):
        module.check(tmp_path)
    changed = deepcopy(real)
    changed["manifest"]["importer_audit"]["other"] = {"settlement": [100.0, 101.0]}
    module.load_mechanism_evidence = lambda directory: changed
    with pytest.raises(AssertionError):
        module.check(tmp_path)
