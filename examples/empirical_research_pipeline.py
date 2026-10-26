"""Small synthetic control for all empirical notebook research angles.

No exchange data or Jupyter installation is needed. Numerical algorithms live
in the package, and the same runner produces the historical aggregate bundle.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from empirical_pipeline import synthetic_panel

from spde_covariance.empirical_research import (
    ResearchSuiteConfig,
    export_research_evidence,
    load_research_evidence,
    run_research_suite,
)
from spde_covariance.empirical_study import EmpiricalConfig, run_empirical_study


def main():
    config = EmpiricalConfig(
        training_end="2022-01-31",
        validation_end="2022-02-28",
        test_end="2022-03-31",
        windows=(8, 16),
        min_direct_count=3,
        bootstrap_replications=20,
        sensitivities=False,
    )
    panel = synthetic_panel()
    research = ResearchSuiteConfig(
        empirical=config,
        score_blocks=(2, 4),
        representation_dimensions=(3,),
        representation_smoothing=(1e-6,),
    )
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_empirical_study(panel, config, root / "baseline")
        run_research_suite(panel, research, root / "suite", baseline_run=root / "baseline")
        export_research_evidence(root / "suite", root / "public")
        evidence = load_research_evidence(root / "public")
        assert evidence["data"]["sessions"] == len(panel.session_dates)
        assert evidence["attribution"]["paired_comparisons"]
        assert "joint" in evidence
        print("Synthetic research suite passed: six notebook angles; verified aggregate export")


if __name__ == "__main__":
    main()
