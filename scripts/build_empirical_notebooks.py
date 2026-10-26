"""Build the six executable research notebooks from plain, reviewable source.

Run this after numerical evidence is available, then execute each notebook with
check_notebook.py --write-outputs. This script does not run market experiments.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path


def markdown(source):
    # GitHub's notebook renderer uses Jupyter math, unlike its README renderer.
    source = re.sub(
        r"```math\n(.*?)\n```", lambda m: "$$\n" + m.group(1) + "\n$$", source, flags=re.S
    )
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": (ast.unparse(ast.parse(source.strip())) + "\n").splitlines(True),
    }


SETUP = """
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from spde_covariance.empirical_research import load_research_evidence

evidence = load_research_evidence(Path("../experiments/results/shfe-research"))
print(evidence["protocol"]["status"])
print("Numerical source:", evidence["manifest"]["git_revision"])
print("Original baseline source:", evidence["protocol"]["baseline_source_revision"])
assert evidence["protocol"]["baseline_summary_sha256"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

def table(rows, columns):
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(f"{row.get(c):.6g}" if isinstance(row.get(c), float) else str(row.get(c)) for c in columns))
"""

FOOTER = markdown(r"""
## Reproduction and research boundary

This notebook reads checksummed aggregate evidence, not raw exchange files.
The numerical algorithms are in `spde_covariance`; use the
[protocol](../docs/empirical-research-protocol.md) to rerun local calibration.
Execute this notebook with `uv run python scripts/check_notebook.py notebooks/NAME`.

The original historical benchmark remains fixed. New analyses on the inspected
2024–2025 period are exploratory, even though forecast issuance is past-only.
No final untouched confirmation period, executable trading backtest or physical
forcing identification is claimed. See the [roadmap](../README.md).
""")


def write_notebook(name, cells):
    footer = json.loads(json.dumps(FOOTER))
    footer["source"] = [s.replace("NAME", name) for s in footer["source"]]
    document = {
        "cells": cells + [footer],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = Path(__file__).resolve().parents[1] / "notebooks" / name
    for i, cell in enumerate(document["cells"]):
        cell["id"] = f"research-{i:03}"
    target.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def main():
    write_notebook(
        "01_data_and_risk_targets.ipynb",
        [
            markdown(r"""# 01 · Data, contract identity and economic risk targets

**RQ1:** What is measured, and what historical information is available for it?
The first acceptance gate is a defensible target, not a favorable model score.
We use official SHFE copper settlements, actual contract identifiers and no
rolled continuous-price series. Raw files and individual outcomes remain local.

**Hypothesis:** effective same-contract history can be much shorter than a
nominal trailing window because contracts have finite lives. This matters when
comparing a pooled curve model with a current-contract covariance estimate."""),
            code(SETUP),
            markdown(r"""## Data and quality audit

Rows are one settlement per contract and session. The observed exchange calendar
contains all sessions found in the archives; no missing prices are filled.
Nonpositive volume or open interest disqualifies a quote from origin curve
reconstruction but does not delete its valid future settlement from P&L scoring.
Annual archives are retrospective and do not reconstruct original release vintages."""),
            code("""data = evidence["data"]
print("Sessions / rows / contracts:", data["sessions"], data["rows"], data["distinct_contracts"])
print("Session counts:", data["sessions_by_year"])
print("Quality:", json.dumps(data["quality"], indent=2))
print(data["vintage_scope"])
assert sum(data["sessions_by_year"].values()) == data["sessions"]"""),
            markdown(r"""## Risk target and units

At each origin choose eligible contracts nearest 90, 180 and 270 days, then hold
their **same IDs** to the next observed session. The coordinate is the 15th of
the contract month, not an actual expiry or a delivery-average map.

```math
Y_{i+1}=5\,a^\top(p_{i+1}^{\mathrm{same\ IDs}}-p_i),\qquad
v_i\approx\mathbb E[Y_{i+1}^2\mid\mathcal F_i].
```

The three holdings are outright $(1,0,0)$, spread $(1,-1,0)$ and butterfly
$(1,-2,1)$. Prices are CNY/tonne and P&L is CNY. With zero conditional mean this
second moment is a variance; otherwise a variance interpretation is an
additional assumption. These are next-session risks, not accumulated or annual returns."""),
            code("""table(data["tenor_coverage"], ["lower_days", "upper_days", "rows", "positive_volume_rows", "positive_open_interest_rows"])
fig, ax = plt.subplots(figsize=(7,3))
rows = data["tenor_coverage"]
ax.bar(np.arange(len(rows)), [r["rows"] for r in rows], color="0.35")
ax.set_xticks(np.arange(len(rows)), [f"{r['lower_days']}–{r['upper_days']}" for r in rows], rotation=30)
ax.set(xlabel="Maturity coordinate (days)", ylabel="Contract-session rows", title="Coverage before curve eligibility filtering")
fig.tight_layout(); plt.show()"""),
            markdown(r"""## Effective history for currently held contracts

Count only adjacent sessions where all three selected actual contracts have
settlements at both endpoints. Absent quotes do not create an artificial long
return. Counts below the declared minimum are unavailable, not zero covariance.
Exposure-specific baselines can use fewer legs; this three-leg audit is the
stricter common support used by the joint model and hedge experiment."""),
            code("""table(data["same_contract_history"], ["window", "count", "joint_count_quantiles", "sufficient_count"])
fig, ax = plt.subplots(figsize=(6,3))
for row in data["same_contract_history"]:
    q = row["joint_count_quantiles"]
    if q:
        ax.errorbar(row["window"], q[2], yerr=[[q[2]-q[0]],[q[-1]-q[2]]], fmt="o", color="black")
ax.plot([0,260],[0,260],linestyle=":",color="0.6")
ax.set(xlabel="Nominal trailing sessions", ylabel="Joint history count (median/range)", title="Contract life limits effective samples")
fig.tight_layout(); plt.show()"""),
            markdown("""## Interpretation and next decision

Read the history counts before interpreting differences between nominal windows.
In this sample the nominal 252-session window contains only 49–77 joint changes
(median 62), the same range as the 126-session window.
This audit supports target definitions and reproducibility; it cannot certify
point-in-time data vintages, exchange execution, bid/ask costs or an exact
financial delivery map. Proceed to representation diagnostics before choosing
a more complex covariance model."""),
        ],
    )
    write_notebook(
        "02_representation_and_identification.ipynb",
        [
            markdown(r"""# 02 · Curve representation and identification

**RQ1:** Which risks survive compression into a smooth finite curve?
**Hypothesis:** good price-level reconstruction does not imply accurate spread
or butterfly risk, and increasing dimension may trade approximation error for
conditioning. This is separate from optimizer convergence."""),
            code(SETUP),
            markdown(r"""## Exact transport and the residual identity

For the invariant Laguerre span, $E_{i+1}M_{\Delta_i}=E_i$ at the same contract
maturities. If $a_i$ denotes the fitted state, then

```math
\Delta p_i=E_{i+1}(a_{i+1}-M_{\Delta_i}a_i)+u_i,\qquad
Y_i=S_i+U_i.
```

Therefore raw sample second moments obey

```math
\overline{Y^2}=\overline{S^2}+\overline{U^2}+2\overline{SU}.
```

The signed cross term is retained. Residuals are realized endpoint
reconstruction-error changes, not fresh independent price noise. This is a
post hoc decomposition; future reconstructed states never enter a forecast."""),
            code("""fits = evidence["representation"]["fits"]
table([{"dimension": r["dimension"], "smoothing": r["smoothing"], "median_rmse": r["rmse_quantiles"][2], "median_condition": r["condition_quantiles"][1], "available": r["available_sessions"]} for r in fits], ["dimension","smoothing","median_rmse","median_condition","available"])
for r in fits:
    for target in r["decomposition"]["exposures"]:
        if target["count"]:
            assert target["identity_relative_error"] < 1e-10"""),
            code("""base = next(r for r in fits if r["dimension"] == 3 and r["smoothing"] == 1e-6)
rows = base["decomposition"]["exposures"]
table(rows,["exposure","count","modeled_share","residual_share","cross_share"])
fig, ax = plt.subplots(figsize=(8,3))
x=np.arange(len(rows)); width=.24
for j,(key,label,color) in enumerate([("modeled_share","Fitted curve","0.25"),("residual_share","Residual","0.6"),("cross_share","Signed cross","0.85")]):
    ax.bar(x+(j-1)*width,[r[key] for r in rows],width,label=label,color=color,edgecolor="black",linewidth=.4)
ax.axhline(0,color="black",linewidth=.6); ax.set_xticks(x,[r["exposure"] for r in rows],rotation=12)
ax.set(ylabel="Share of realized second moment",title="Curve fit and financial risk are different targets"); ax.legend()
fig.tight_layout(); plt.show()"""),
            markdown(r"""## A clean identification control

Suppose observations reveal covariance blocks on nodes $(1,2)$ and $(2,3)$,
but never the pair $(1,3)$. All matrices below are PSD and give the same observed
blocks. The unobserved spread risk is $2-2c\in[0,4]$. Regularization or canonical
completion chooses a matrix; the observations do not identify the choice."""),
            code("""for c in (-1.0,0.0,1.0):
    K=np.array([[1.,0.,c],[0.,1.,0.],[c,0.,1.]])
    a=np.array([1.,0.,-1.])
    assert np.linalg.eigvalsh(K)[0] >= -1e-12
    print("Missing covariance",c,"spread risk",float(a@K@a))
print("This is an analytical identification illustration, not a market confidence interval.")"""),
            markdown("""## Interpretation and next decision

The three-mode fitted curve accounts for 99.1% of outright second moment but
only 38.3% of spread and 11.4% of butterfly second moment. Four modes reduce
median level RMSE from 48.58 to 44.22 CNY/tonne while raising median conditioning
from about 70 to 1,611.

Separate missing-pair ambiguity, basis restriction, conditioning, sampling and
numerical error. The market decomposition measures representation losses; it
does not observe the true SPDE covariance. Carry material residual risk forward
into a joint model rather than equating reconstructed coefficients with clean
market truth."""),
        ],
    )
    write_notebook(
        "03_performance_attribution.ipynb",
        [
            markdown("""# 03 · Attribution: transport, window and information

**RQ2:** Which mechanism changes forecast quality? The original practical rules
selected windows independently. This study fixes windows for mechanism comparisons,
keeps dimension and smoothing fixed, and changes transport or retained nodes.
No setting is selected using these exploratory evaluation scores."""),
            code(SETUP),
            markdown(r"""## Protocol and fair comparisons

The adjusted factor model fits $C$ through $V_\Delta(C)$; its A=0 counterpart
uses ordinary coefficient increments. Both use identical reconstructed states,
window and future actual-contract outcomes. Compare raw direct covariance and
10% diagonal shrinkage at the same nominal windows. Actual historical support
can differ because raw current contracts have finite history.

Nested masks retain all nodes, alternating nodes, then four nodes from the
alternating set. They affect factor reconstruction only. Raw direct baselines
retain full contract history, so masked-factor versus direct comparisons are
information stress tests. Within-mask adjusted/A=0 comparisons isolate transport
more closely. A 16-interval history cannot meet the original 20-pair baseline
minimum; do not relax that minimum to manufacture a comparison."""),
            code("""study = evidence["attribution"]
print(json.dumps(study["scope"],indent=2))
rows=[r for r in study["scores"] if r["mask"]=="all" and r["year"]=="all"]
table(rows,["window","exposure","method","count","qlike","risk_ratio"])"""),
            code("""fig, axes=plt.subplots(1,3,figsize=(12,3.2))
for ax,exposure in zip(axes,["outright_3m","spread_3m_6m","butterfly_3m_6m_9m"],strict=True):
    for method,style in [("factor_adjusted","o-"),("factor_unadjusted","x--"),("direct","s-"),("shrinkage","^:")]:
        points=sorted([r for r in rows if r["exposure"]==exposure and r["method"]==method],key=lambda r:r["window"])
        ax.plot([r["window"] for r in points],[r["qlike"] for r in points],style,label=method,linewidth=1)
    ax.set(title=exposure,xlabel="Matched nominal window",ylabel="Mean QLIKE (lower is better)")
axes[0].legend(fontsize=7);fig.tight_layout();plt.show()"""),
            markdown("""## Paired transport effect and masking stress

Differences are calculated on each method pair's identical scored observations.
Negative means favor the adjusted model. Report block uncertainty and availability
alongside the mean. A very small adjusted/A=0 effect is not evidence that transport
is mathematically unnecessary; it limits the measured forecasting claim here."""),
            code("""paired=[r for r in study["paired_comparisons"] if r["reference"]=="factor_unadjusted" and r["score"]=="qlike" and r["year"]=="all"]
table(paired,["mask","window","exposure","count","mean","bootstrap_standard_error","bootstrap_percentile_95"])
table([r for r in study["availability"] if r["window"]==16 and r["mask"]=="all"],["window","exposure","method","candidate_origins","issued_count","scored_count"])"""),
            code("""for row in study["representation"]:
    print(row["mask"],json.dumps(row["reconstruction"],indent=2))
table([r for r in study["available_scores"] if r["window"]==126 and r["year"]=="all" and r["method"]=="factor_adjusted"],["mask","exposure","count","qlike","risk_ratio"])
print("Numerical diagnostics:")
for row in study["numerical_diagnostics"]:
    print(row["mask"],row["window"],row["unavailable_reasons"],row["relative_covariance_error_bound_max"])"""),
            markdown("""## Interpretation and next decision

At 252 intervals with all nodes, adjusted minus A=0 QLIKE is approximately
−0.000142 (outright), +0.00530 (spread) and +0.0323 (butterfly). The last difference
favors A=0; no favorable transport effect is established. Some masked estimates
improve under this fixed misspecified estimator, which does not make less
information intrinsically preferable.

Distinguish practical policy comparison from mechanism attribution. Neither a
masked curve model beating a baseline under one loss nor more coordinates alone
establishes an SPDE advantage. Use the joint-risk study to assess whether retained
residual information improves the economically relevant targets."""),
        ],
    )
    write_notebook(
        "04_forecast_scores_and_model_risk.ipynb",
        [
            markdown(r"""# 04 · Forecast scores, calibration and model risk

**RQ3:** Does the conclusion survive changing the loss and dependence assumptions?
The frozen benchmark has an informative ranking reversal. QLIKE remains primary;
the quadratic score is retained with paired uncertainty, rather than promoted
after seeing favorable outcomes."""),
            code(SETUP),
            markdown(r"""## What the scores measure

For $z=Y^2$ and positive forecast $v$,

```math
\ell_Q(v,z)=\log v+z/v,\qquad \ell_2(v,z)=v^2-2vz.
```

Paired differences are affine in $z$. If $m=\mathbb E[z\mid\mathcal F]$, expected
regrets are $\log(v/m)+m/v-1$ and $(v-m)^2$. Both target the conditional second
moment but weight misspecification differently. QLIKE is sensitive to severe
relative underprediction; quadratic loss weights cash-scale errors. A Gaussian
distribution is not needed. QLIKE uses the original positive floor; quadratic
loss uses the unfloored nonnegative forecast. See
[Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)."""),
            code("""scores=evidence["scores"]
print(json.dumps(scores["protocol"],indent=2))
paired=[r for r in scores["paired_comparisons"] if r["stratum"]=="pooled" and r["method"]=="spde_local"]
table(paired,["exposure","loss","common_count","mean","bootstrap_standard_error","bootstrap_percentile_95"])"""),
            code("""fig,axes=plt.subplots(1,2,figsize=(10,3.2))
for ax,loss,scale in zip(axes,["qlike","quadratic"],[1.,1e12],strict=True):
    rows=[r for r in paired if r["loss"]==loss and r["available"]]
    for i,r in enumerate(rows):
        lo,hi=np.asarray(r["bootstrap_percentile_95"])/scale
        ax.plot([lo,hi],[i,i],color="black");ax.plot(r["mean"]/scale,i,"o",color="black")
    ax.axvline(0,color="0.5",linestyle=":");ax.set_yticks(range(len(rows)),[r["exposure"] for r in rows])
    ax.set(title=loss,xlabel="SPDE − direct"+(" (10¹² CNY⁴)" if scale>1 else ""))
fig.tight_layout();plt.show()"""),
            markdown("""## Dependence, calibration and period sensitivity

Circular moving blocks resample chronologically ordered paired scored sessions.
The number of sessions is not a Monte Carlo replication count. The percentile
ranges are descriptive and do not correct for the many explored comparisons.
Calibration bins are based on forecast risk, and realized-outcome strata are
explicitly ex post. Neither is used to tune a policy on the evaluation outcomes."""),
            code("""sensitivity=[r for r in scores["block_sensitivity"] if r["method"]=="spde_local" and r["exposure"]=="spread_3m_6m"]
table(sensitivity,["loss","requested_block_length","mean","bootstrap_standard_error","bootstrap_percentile_95"])
table([r for r in scores["method_summaries"] if r["method"] in ("spde_local","direct") and r["stratum"]=="year"],["group","exposure","method","count","mean_qlike","risk_ratio"])
table([r for r in scores["method_summaries"] if r["method"] in ("spde_local","direct") and r["stratum"]=="calendar_gap"],["group","exposure","method","count","mean_qlike","risk_ratio"])
table([r for r in scores["realized_outcome_strata"] if r["method"] in ("spde_local","direct")],["realized_squared_pnl_third","exposure","method","count","mean_qlike","risk_ratio"])"""),
            code("""rows=[r for r in scores["calibration_bins"] if r["method"]=="spde_local"]
table(rows,["exposure","forecast_variance_quintile","count","predicted_rms_cny","realized_rms_cny","risk_ratio"])
fig,ax=plt.subplots(figsize=(5,4))
for exposure in sorted({r["exposure"] for r in rows}):
    selected=[r for r in rows if r["exposure"]==exposure and r["available"]]
    ax.plot([r["predicted_rms_cny"] for r in selected],[r["realized_rms_cny"] for r in selected],"o-",label=exposure)
limit=ax.get_xlim()[1];ax.plot([0,limit],[0,limit],":",color="0.5")
ax.set(xlabel="Forecast RMS (CNY)",ylabel="Realized RMS within bin (CNY)",title="Forecast-risk calibration bins");ax.legend(fontsize=7)
fig.tight_layout();plt.show()"""),
            markdown("""## Influential paired differences

The following diagnostic removes the largest 1% or 5% of absolute paired score
differences. It measures influence, not a preferred cleaned evaluation sample.
No dates or individual market outcomes are exported; small cells are suppressed."""),
            code("""table([r for r in scores["score_influence"] if r["method"]=="spde_local"],["exposure","loss","fraction","full_mean_difference","remaining_mean_difference","removed_absolute_difference_share"])
table([r for r in scores["availability"] if r["method"] in ("direct","spde_local")],list(scores["availability"][0].keys()))"""),
            markdown("""## Interpretation and next decision

The spread difference is +0.953 under primary QLIKE (bootstrap range
[0.188, 1.839]) but −0.391×10¹² CNY⁴ under quadratic loss (range
[−0.665, −0.177]×10¹²). Outright and butterfly quadratic ranges include zero.

Read score ranking, calibration and uncertainty together. Favorable quadratic
loss on one exposure does not remove a QLIKE or underprediction failure. The
matched-window experiment addresses attribution; the joint model addresses
missing risk. An uninspected confirmation period is needed for a revised policy."""),
        ],
    )
    write_notebook(
        "05_joint_residual_risk_and_clocks.ipynb",
        [
            markdown("""# 05 · Joint residual risk and diffusion clocks

**RQ2–RQ3:** Can explicit residual risk repair the smooth-factor model? Does
stochastic time scaling explain some calendar-gap behavior? This is a revised
development model, distinct from the frozen source-covariance benchmark."""),
            code(SETUP),
            markdown(r"""## Model and estimand

For the current actual-contract set, collect historical adjacent changes and
their exact transport decomposition. For diffusion duration $h_i$, estimate

```math
z_i=\frac{1}{\sqrt{h_i}}\begin{pmatrix}r_i\\u_i\end{pmatrix},\qquad
\widehat Q=\frac1n\sum_i z_i z_i^\top,
\qquad \widehat K_{n+1}=h_{n+1}[E_{n+1},I]\widehat Q[E_{n+1},I]^\top.
```

This retains factor/residual cross blocks and is PSD by construction. Optional
diagonal shrinkage changes the joint moment explicitly. The locally common
joint moment is an approximation across contract ages, not physical forcing
identification. Compare with factor-only and raw direct second moments on the
**same complete historical intervals**.

Physical transport always uses ACT/365.25 maturity time. Only the diffusion
normalizer changes: $h_i=\Delta_i$ or one session/252. The original source-$C$
pipeline remains a separate calendar-time control."""),
            code("""joint=evidence["joint"]
print(joint["clock_convention"])
print(joint["selection_criterion"])
print("Joint settings:",json.dumps(joint["joint_config"],indent=2))
for method,selection in joint["selection"].items():
    print(method,selection["selected"],"validation origins",selection["common_validation_origins"],"last outcome",selection["selection_last_outcome"])
    assert selection["selection_last_outcome"] <= joint["chronology"]["validation_end"]"""),
            code("""table(joint["risk_summary"],["exposure","method","common_count","qlike","realized_over_predicted_second_moment","predicted_rms_cny","realized_rms_cny"])
table([r for r in joint["paired_scores"] if r["score"]=="qlike"],["exposure","method","reference","count","mean","bootstrap_standard_error","bootstrap_percentile_95"])"""),
            code("""fig,axes=plt.subplots(1,3,figsize=(12,3.2))
for ax,exposure in zip(axes,["outright_3m","spread_3m_6m","butterfly_3m_6m_9m"],strict=True):
    rows=[r for r in joint["risk_summary"] if r["exposure"]==exposure]
    ax.bar(range(len(rows)),[r["realized_over_predicted_second_moment"] for r in rows],color="0.45")
    ax.axhline(1,color="black",linestyle=":");ax.set_xticks(range(len(rows)),[r["method"] for r in rows],rotation=80,fontsize=7)
    ax.set(title=exposure,ylabel="Realized / predicted second moment")
fig.tight_layout();plt.show()"""),
            markdown("""## Cross-covariances and numerical checks

Dropping cross terms would impose independence without evidence. The components
below sum to a joint-model forecast; factor/direct controls can use a different
kernel. Numerical PSD/rank diagnostics establish matrix feasibility, not a
statistical or market-model guarantee. Availability is reported separately."""),
            code("""table(joint["risk_components"],["method","exposure","count","components_cny2","cross_over_total"])
table([r for r in joint["clock_gap_summary"] if r["method"] in ("joint_calendar","joint_session")],["method","gap_group","exposure","count","qlike","realized_over_predicted_second_moment"])
print(json.dumps(joint["numerical_diagnostics"],indent=2))
print(json.dumps(joint["availability"],indent=2))"""),
            markdown("""## Interpretation and next decision

Validation selects 63 intervals, calendar diffusion and zero shrinkage for the
joint model. Joint spread/butterfly QLIKE is 13.710/15.069, versus factor-only
15.172/23.760 and direct 13.637/15.066. Residual risk repairs the weak factor
model but does not establish superiority over direct covariance. Session scaling
improves curvature scores while worsening outright scores in this development
sample; the selected calendar policy remains unchanged.

Compare the selected practical policies and the fixed-clock companions separately.
Residual risk and a different clock can improve a weak factor forecast without
establishing an SPDE advantage over raw direct risk. Use the matched attribution
results and effective sample counts to qualify the conclusion. Test the selected
revised policy on an uninspected period before promoting it beyond development."""),
        ],
    )
    write_notebook(
        "06_hedging_decisions.ipynb",
        [
            markdown("""# 06 · Covariance forecasts as hedging decisions

**RQ3:** Does a covariance estimate support a useful economic decision?
Keep one near contract and use the middle/far contracts as hedges. The objective
is reduced settlement risk, not an expected-return or trading-alpha strategy."""),
            code(SETUP),
            markdown(r"""## Decision problem and chronology

Given the issued PSD raw contract covariance $\widehat K_n$, solve

```math
\min_{h\in[-2,2]^2}(1,h^\top)\widehat K_n(1,h^\top)^\top.
```

The solver checks the unconstrained stationary point, edges and corners, so
singular PSD controls remain well-defined. It receives no future price. Each
rule selects its risk-model settings on 2023 validation QLIKE, then issues
positions using data through the origin. The same actual contracts are held
until next session. These are risk-selected policies, not separately
hedging-loss-tuned policies."""),
            code("""joint=evidence["joint"]
print(joint["hedge_scope"])
print("Bounds:",joint["joint_config"]["hedge_bounds"])
table(joint["hedge_summary"],["method","common_count","realized_rms_cny","reduction_vs_unhedged","predicted_rms_cny","realized_over_predicted_second_moment"])"""),
            code("""rows=joint["hedge_summary"]
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
axes[0].bar(range(len(rows)),[r["realized_rms_cny"] for r in rows],color="0.4")
axes[0].set_xticks(range(len(rows)),[r["method"] for r in rows],rotation=75,fontsize=7)
axes[0].set(ylabel="Realized RMS settlement P&L (CNY)",title="One near contract plus bounded hedges")
hedged=[r for r in rows if r["method"]!="unhedged"]
axes[1].bar(range(len(hedged)),[r["mean_position_change_proxy"] or 0 for r in hedged],color="0.65")
axes[1].set_xticks(range(len(hedged)),[r["method"] for r in hedged],rotation=75,fontsize=7)
axes[1].set(ylabel="Mean absolute hedge-position change",title="Instability proxy; excludes roll/execution costs")
fig.tight_layout();plt.show()"""),
            markdown("""## Paired realized risk and position stability

All displayed rules use the common scored-origin cohort. Negative paired
squared-P&L differences favor the listed rule. Bootstrap blocks resample
paired sessions, not independent strategies. Mean hedge positions and bound
hits help detect unstable optimization even when average risk is small."""),
            code("""table(joint["hedge_paired"],["method","reference","count","mean","bootstrap_standard_error","bootstrap_percentile_95"])
table([r for r in joint["hedge_summary"] if r["method"]!="unhedged"],["method","mean_hedge_positions","bound_hit_fraction","mean_position_change_proxy"])
print(json.dumps(joint["availability"],indent=2))"""),
            markdown("""## Interpretation and next decision

Realized RMS is CNY 3,487 unhedged, 501 with direct covariance, 533 with joint
factor/residual, 624 with the original source model and 685 with factor-only.
Joint minus direct squared P&L averages +32,657 CNY², with descriptive bootstrap
range [17,545, 47,609]; direct has lower realized risk in this sample.

This closes the pipeline from data and curve reconstruction to model risk and
a constrained decision. It does not establish an executable backtest: contract
switching, actual fills, margin, roll mechanics, transaction costs and liquidity
capacity are not modeled. A later execution-aware study needs these inputs and
an uninspected evaluation period. Do not turn lower settlement RMS alone into
a profitability or CV performance claim."""),
        ],
    )


if __name__ == "__main__":
    main()
    paths = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("0*.ipynb"))
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", *map(str, paths)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", *map(str, paths)], check=True)
