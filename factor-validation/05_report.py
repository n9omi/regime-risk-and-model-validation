"""
05_report.py | Step 5: write the factor-validation report (Markdown for GitHub, PDF for reading).

Outputs: outputs/factor_validation_summary.md / .html, reports/factor_validation_summary.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib.report import Report, fmt_num, fmt_p, fmt_pct, verdict  # noqa: E402

TAB, FIG = C.FACT_OUT / "tables", C.FACT_OUT / "figures"
t = lambda n: pd.read_csv(TAB / f"{n}.csv")
log, uni, perf, dec = t("01_data_pull_log"), t("01_universe"), t("02_published_performance"), t("02_decay_windows")
cv, pbo, ds, gridp = t("03_cv_results"), t("04_pbo"), t("04_deflated_sharpe"), t("03_grid_performance")
corr = pd.read_csv(TAB / "02_strategy_correlations.csv", index_col=0)
plc = t("02_placebo_test")
byreg = t("02_performance_by_regime") if (TAB / "02_performance_by_regime.csv").exists() else None
print("Step 5 | writing the factor-validation report")

net = perf[perf.returns == "net"].set_index("strategy")
rep = Report("Factor Validation: Commodity Strategies",
             f"{int(log.commodities[0])} commodities · monthly {log['first'][0]}–{log['last'][0]} · source: {log.source[0]}",
             C.FACT_OUT, "factor_validation_summary")

# ---------------------------------------------------------------- headline
pb = pbo.set_index("variant_set")
dsi = ds.set_index("candidate")
rep.section("Headline numbers")
rep.kpis([
    ("Value (5y) net Sharpe", fmt_num(net.loc["VALUE", "sharpe"], 2), f"t = {fmt_num(net.loc['VALUE','nw_t'],1)} (Newey-West)"),
    ("TSMOM (12-1) net Sharpe", fmt_num(net.loc["TSMOM", "sharpe"], 2), f"t = {fmt_num(net.loc['TSMOM','nw_t'],1)}"),
    ("XS momentum net Sharpe", fmt_num(net.loc["XSMOM", "sharpe"], 2), f"t = {fmt_num(net.loc['XSMOM','nw_t'],1)}"),
    ("PBO, momentum grid", fmt_pct(pb.loc["TSMOM", "pbo"], 0), "chance the TSMOM winner is below median OOS"),
    ("Variants tried", f"{int(pb.loc['All variants','n_variants'])}", "every one is charged in the deflated Sharpe"),
    ("Lag-1 autocorrelation", fmt_num(uni.lag1_autocorr.median(), 2), f"vs {fmt_num(uni.lag2_autocorr.median(),2)} at lag 2: averaging artefact"),
])
dw = dec.pivot(index="strategy", columns="window", values="sharpe")
IN_SAMPLE = "In paper's sample"
xs_pos = int((gridp[gridp.family == "XSMOM"].sharpe_full_sample > 0).sum())
val_ok = bool(dsi.loc["Published VALUE", "dsr"] > 0.95 and net.loc["VALUE", "nw_t"] > 3)
max_short = cv[cv.method == "Purged CPCV"].oos_shortfall.abs().max()
n_turb = int(byreg[byreg.wti_regime_last_month == "Turbulent"].months.max()) if byreg is not None and (byreg.wti_regime_last_month == "Turbulent").any() else 0
obs = [
    ("**Value is the standout and it survives every check.**" if val_ok else "**Value is the strongest strategy.**") +
    f" A 5-year reversal signal earns a net Sharpe of "
    f"{fmt_num(net.loc['VALUE','sharpe'],2)} with a Newey-West t of {fmt_num(net.loc['VALUE','nw_t'],1)} (above the Harvey–Liu–Zhu t > 3 bar), "
    f"a deflated Sharpe of {fmt_num(dsi.loc['Published VALUE','dsr'],3)}, and a post-publication Sharpe of "
    f"{fmt_num(dw.loc['VALUE','After publication'],2)}.",
    f"**Time-series momentum is real but modest.** Net Sharpe {fmt_num(net.loc['TSMOM','sharpe'],2)} "
    f"({fmt_num(dw.loc['TSMOM', IN_SAMPLE],2)} in the paper's window, {fmt_num(dw.loc['TSMOM','After publication'],2)} after publication); "
    f"t = {fmt_num(net.loc['TSMOM','nw_t'],2)} falls short of t > 3, and choosing its best look-back has a "
    f"{fmt_pct(pb.loc['TSMOM','pbo'],0)} probability of being overfit.",
    (f"**Cross-sectional momentum does not replicate on spot prices** (net Sharpe {fmt_num(net.loc['XSMOM','sharpe'],2)}; "
     f"{xs_pos} of its {int(pb.loc['XSMOM','n_variants'])} variants is positive). The paper used futures returns; see the caveat below."
     if net.loc["XSMOM", "sharpe"] <= 0 else
     f"**Cross-sectional momentum replicates** with a net Sharpe of {fmt_num(net.loc['XSMOM','sharpe'],2)}."),
    f"**Value and momentum hedge each other** (correlation {fmt_num(corr.loc['VALUE','XSMOM'],2)} with XS momentum, "
    f"{fmt_num(corr.loc['VALUE','TSMOM'],2)} with TSMOM), as Asness, Moskowitz and Pedersen report across asset classes.",
    f"**Parameter selection is stable.** Out-of-sample Sharpe from purged CPCV is within "
    f"{fmt_num(max_short,2)} of the in-sample best in every family: "
    "rankings barely change between halves of the sample, which is what a low PBO means.",
]
rep.bullets(obs)
rep.callout("**Spot is not futures.** The Pink Sheet reports monthly *average spot* prices. Spot prices mean-revert more "
            "than futures returns do, because futures curves already price expected reversion, so value's Sharpe here is an "
            "upper bound for a tradable version. Monthly averaging also smooths volatility and adds lag-1 autocorrelation, "
            "which is why every signal skips the most recent month.", "warn")

# ---------------------------------------------------------------- data
rep.section("1. Data", new_page=True)
rep.text(f"World Bank *Pink Sheet* monthly prices for {int(log.commodities[0])} commodities with a liquid futures "
         f"analogue, {log['first'][0]} to {log['last'][0]}. Returns are monthly log changes. A month is tradable once at "
         f"least {C.MIN_ASSETS} commodities have prices.")
u = uni.copy()
u = pd.DataFrame({"Commodity": u.commodity, "Sector": u.sector, "Unit": u.unit, "Ann. vol": u.ann_vol_pct.map(lambda v: f"{v:.1f}%"),
                  "Lag-1 autocorr": u.lag1_autocorr.map(lambda v: fmt_num(v, 2)), "Lag-2 autocorr": u.lag2_autocorr.map(lambda v: fmt_num(v, 2))})
rep.table(u, "Universe", small=True)
rep.figure(FIG / "01_autocorrelation.png", "Averaging within the month creates lag-1 autocorrelation (≈0.25 for a random walk, Working 1960); lag 2 is clean.")

# ---------------------------------------------------------------- replication
rep.section("2. Replicating the published strategies", new_page=True)
spec = pd.DataFrame([dict(Strategy=v["name"], Paper=v["paper"], Settings=", ".join(f"{a}={round(b,3) if isinstance(b,float) else b}" for a, b in v["params"].items()),
                          **{"Sample ends": v["sample_end"], "Published": v["published"]}) for v in C.PUBLISHED.values()])
rep.table(spec, "What is being replicated", small=True)
p = perf.copy()
p = pd.DataFrame({"Strategy": p.strategy, "Returns": p.returns, "Months": p.months, "Ann. return": p.ann_return.map(lambda v: fmt_pct(v, 1)),
                  "Ann. vol": p.ann_vol.map(lambda v: fmt_pct(v, 1)), "Sharpe": p.sharpe.map(lambda v: fmt_num(v, 2)),
                  "NW t": p.nw_t.map(lambda v: fmt_num(v, 2)), "Max DD": p.max_drawdown.map(lambda v: fmt_pct(v, 0)),
                  "Turnover/mo": p.avg_turnover.map(lambda v: "–" if pd.isna(v) else fmt_num(v, 2))})
rep.table(p, f"Full-sample performance (costs: {C.COST_BPS} bp per unit of turnover)", small=True)
rep.figure(FIG / "02_cumulative_returns.png", "Growth of $1, net of costs, with each paper's sample end and publication date.")
d = dec.copy()
d = pd.DataFrame({"Strategy": d.strategy, "Window": d.window, "From": d.start, "To": d.end, "Months": d.months,
                  "Sharpe": d.sharpe.map(lambda v: fmt_num(v, 2)), "NW t": d.nw_t.map(lambda v: fmt_num(v, 2))})
rep.table(d, "McLean–Pontiff windows: before and after the paper", small=True)
rep.figure(FIG / "02_decay.png", "Sharpe ratio in each window. Short windows (under 3 years) are noisy.")
pl = pd.DataFrame({"Strategy": plc.strategy, "Real gross Sharpe": plc.real_gross_sharpe.map(lambda v: fmt_num(v, 2)),
                   "Placebo mean": plc.placebo_mean_sharpe.map(lambda v: fmt_num(v, 2)),
                   "Placebo 5–95%": [f"{fmt_num(a,2)} to {fmt_num(b,2)}" for a, b in zip(plc.placebo_p05, plc.placebo_p95)],
                   "p-value": plc.p_value.map(fmt_p)})
pi = plc.set_index("strategy")
rep.text("**Placebo test.** Shuffling each month's positions across commodities keeps the portfolio's shape but destroys "
         "which commodity gets which position. For the market-neutral strategies the placebos centre near zero "
         f"(XS momentum {fmt_num(pi.loc['XSMOM','placebo_mean_sharpe'],2)}, value {fmt_num(pi.loc['VALUE','placebo_mean_sharpe'],2)}), "
         "so the portfolio code carries no structural bias, and value sits far in the placebo tail "
         f"(p = {fmt_p(pi.loc['VALUE','p_value'])}). TSMOM is different: a shuffle keeps each month's net long/short tilt, "
         f"and that alone earns a Sharpe of {fmt_num(pi.loc['TSMOM','placebo_mean_sharpe'],2)} against the real "
         f"{fmt_num(pi.loc['TSMOM','real_gross_sharpe'],2)}. Most of time-series momentum's edge here is timing the "
         "commodity complex as a whole, not picking commodities.")
rep.table(pl, "Permutation test (200 shuffles, gross of costs)", small=True)
if byreg is not None:
    b = byreg.copy()
    b = pd.DataFrame({"Strategy": b.strategy, "WTI regime last month": b.wti_regime_last_month, "Months": b.months,
                      "Ann. return": b.ann_return.map(lambda v: fmt_pct(v, 1)), "Sharpe": b.sharpe.map(lambda v: fmt_num(v, 2))})
    rep.text("**Link to the regime workflow.** Months are labelled by the WTI regime the Markov-switching model "
             "reported the month before (so the label is known in advance). Turbulent months are few, so treat this as descriptive.")
    rep.table(b, "Strategy performance by energy regime (2007 onward)", small=True)

# ---------------------------------------------------------------- selection
rep.section("3. Choosing parameters without fooling yourself", new_page=True)
rep.text(f"The research grid holds **{len(gridp)} variants**: look-backs of 1–12 months, one- or two-month skips, "
         "vol-scaling on or off, terciles or quartiles, and 3–5 year value windows. Picking the best one after seeing "
         "the results is the classic way to overfit. **Combinatorial purged cross-validation** (López de Prado, 2018) "
         f"splits {C.FACTOR_START[:4]}–{log['last'][0][:4]} into {C.CPCV_GROUPS} blocks, holds out {C.CPCV_TEST_GROUPS} at a time "
         f"(45 splits, 9 complete out-of-sample paths), chooses the best variant on the rest, and drops {C.PURGE} month "
         f"before and {C.EMBARGO} months after each held-out block so overlapping look-back windows cannot leak.")
rep.figure(FIG / "03_grid_sharpes.png", "Full-sample net Sharpe of every variant, coloured by family.")
c = cv.copy()
c = pd.DataFrame({"Variant set": c.variant_set, "N": c.n_variants, "Method": c.method,
                  "In-sample best": c.in_sample_best_sharpe.map(lambda v: fmt_num(v, 2)),
                  "Out-of-sample": c.mean_oos_sharpe.map(lambda v: fmt_num(v, 2)),
                  "Shortfall": c.oos_shortfall.map(lambda v: fmt_num(v, 2)), "Most chosen": c.most_chosen})
rep.table(c, "In-sample promise vs out-of-sample delivery (annualised net Sharpe)", small=True)
rep.figure(FIG / "03_cpcv_vs_insample.png", "Each dot is one CPCV path. The diamond is what a backtest of the best variant would report.")
rep.text("Purging changes little here because the strategies are simple rules: the only thing fitted is the choice of "
         "variant, and rankings are stable. Purging matters far more for fitted models (regressions, trees) trained on "
         "overlapping labels; the code handles both.")

# ---------------------------------------------------------------- overfitting
rep.section("4. Overfitting diagnostics", new_page=True)
q = pbo.copy()
q = pd.DataFrame({"Variant set": q.variant_set, "Variants": q.n_variants, "CSCV splits": q.splits.map(lambda v: f"{v:,}"),
                  "PBO": q.pbo.map(lambda v: fmt_pct(v, 1)), "OOS Sharpe < 0": q.share_oos_sharpe_negative.map(lambda v: fmt_pct(v, 0)),
                  "Verdict": [verdict(v < 0.5, yes="Low risk", no="High risk") for v in q.pbo]})
rep.table(q, f"Probability of backtest overfitting (CSCV, {C.PBO_BLOCKS} blocks)")
rep.figure(FIG / "04_pbo_logits.png", "Logit of the in-sample winner's out-of-sample rank. Mass left of zero = overfitting.")
z = ds.copy()
z = pd.DataFrame({"Candidate": z.candidate, "Variant": z.variant, "Trials N": z.n_trials, "Sharpe": z.sharpe_ann.map(lambda v: fmt_num(v, 2)),
                  "Luck bar SR0": z.sr0_ann.map(lambda v: fmt_num(v, 2)), "PSR vs 0": z.psr_vs_0.map(lambda v: fmt_num(v, 3)),
                  "DSR": z.dsr.map(lambda v: fmt_num(v, 3)), "NW t": z.nw_t.map(lambda v: fmt_num(v, 2)),
                  "Verdict": [verdict(bool(a), yes="Survives", no="Not proven") for a in z.passes_dsr_95]})
rep.table(z, "Probabilistic and deflated Sharpe ratios (annualised Sharpe; DSR > 0.95 to pass)", small=True)
rep.figure(FIG / "04_deflated_sharpe.png", "Deflated Sharpe ratios; bars in green pass the 95% bar.")
rep.text("**Reading the luck bar.** SR0 is the Sharpe the best of N variants would show with zero true skill, given how "
         "much the variants' Sharpes differ. Across all 46 variants the bar is high because the families differ in genuine "
         "skill, not only luck; within a family it is the fairer test, which is why each published strategy is deflated by "
         "its own family.")

# ---------------------------------------------------------------- validation
rep.section("5. Validation summary", new_page=True)
val = pd.DataFrame([
    ("Look-ahead", "Signals use prices through t−1; placebo and no-look-ahead tests",
     f"Market-neutral placebos centre on {fmt_num(plc[plc.strategy != 'TSMOM'].placebo_mean_sharpe.abs().max(),2)} or less",
     verdict(bool(plc[plc.strategy != 'TSMOM'].placebo_mean_sharpe.abs().max() < 0.2))),
    ("Data artefact", "Lag-1 autocorrelation from monthly averaging", f"Median {fmt_num(uni.lag1_autocorr.median(),2)}; neutralised by skipping a month", verdict(True)),
    ("Replication", "Published settings, gross and net",
     "Positive net Sharpe: " + (", ".join(k for k in C.PUBLISHED if net.loc[k, "sharpe"] > 0) or "none") +
     "; negative: " + (", ".join(k for k in C.PUBLISHED if net.loc[k, "sharpe"] <= 0) or "none"),
     verdict(None, warn="Partial") if (net.loc[list(C.PUBLISHED), "sharpe"] <= 0).any() else verdict(True)),
    ("Decay", "McLean–Pontiff windows", f"Post-publication Sharpe: value {fmt_num(dw.loc['VALUE','After publication'],2)}, TSMOM {fmt_num(dw.loc['TSMOM','After publication'],2)}", verdict(True)),
    ("Selection bias", "Purged CPCV, walk-forward", f"Out-of-sample within {fmt_num(max_short,2)} of in-sample in every family",
     verdict(bool(max_short < 0.25))),
    ("Overfitting", "PBO (CSCV), deflated Sharpe", f"PBO {fmt_pct(pb.loc['TSMOM','pbo'],0)} for TSMOM; value DSR {fmt_num(dsi.loc['Published VALUE','dsr'],3)}", verdict(None, warn="Family-dependent")),
], columns=["Check", "How", "Result", "Status"])
rep.table(val, "Validation evidence")
rep.sub("Limitations")
rep.bullets([
    "Spot, monthly-average prices: not the futures excess returns the papers use. No roll yield, so no carry strategy.",
    "Costs are a flat 10 bp per unit of turnover; real costs vary by market and by era.",
    "The grid is what one researcher might try. The literature as a whole tried far more, so true N is larger.",
    f"The regime split has only {n_turb} turbulent months: descriptive only.",
])
rep.sub("References")
rep.bullets([
    "Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2).",
    "Miffre, J. & Rallis, G. (2007). Momentum strategies in commodity futures markets. *Journal of Banking & Finance*, 31(6).",
    "Asness, C., Moskowitz, T. & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3).",
    "McLean, R. D. & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1).",
    "Bailey, D. & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2).",
    "Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5).",
    "Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4).",
    "López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 7, 11, 12.",
    "Harvey, C., Liu, Y. & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1).",
    "Working, H. (1960). Note on the correlation of first differences of averages in a random chain. *Econometrica*, 28(4).",
    "World Bank (2026). Commodity Price Data (The Pink Sheet), monthly historical data.",
])
md, html = rep.write()
pdf = rep.pdf(C.REPORTS)
print(f"  wrote {md.name}, {html.name}" + (f", {pdf}" if pdf else ""))
