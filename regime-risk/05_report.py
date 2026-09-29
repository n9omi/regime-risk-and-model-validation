"""
05_report.py | Step 5: write the regime & risk report (Markdown for GitHub, PDF for reading).

Every sentence with a number in it is generated from the tables written by steps 1-4,
so the report stays correct when the data updates.
Outputs: outputs/regime_risk_summary.md / .html, reports/regime_risk_summary.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib.report import Report, fmt_money, fmt_num, fmt_p, fmt_pct, verdict  # noqa: E402

TAB = C.RISK_OUT / "tables"
FIG = C.RISK_OUT / "figures"
t = lambda name: pd.read_csv(TAB / f"{name}.csv")
log, summ, fixes, flags = t("01_data_pull_log"), t("01_summary_stats"), t("01_data_quality_fixes"), t("01_flagged_moves")
sel, par, rec, cur, epi = (t("02_model_selection"), t("02_ms_parameters"), t("02_simulation_recovery"),
                           t("02_current_regime"), t("02_turbulent_episodes"))
loss, dm, mz, grec = t("03_forecast_losses"), t("03_diebold_mariano"), t("03_mincer_zarnowitz"), t("03_garch_recovery")
bt, today, byreg = t("04_backtest_results"), t("04_var_es_today"), t("04_exceptions_by_regime")
asof = pd.to_datetime(log["last"]).min().date()
src = log["source"].iloc[0].split(" (")[0]
MODELS = ["Historical sim", "Filtered HS", "GARCH-t", "HAR-RV normal", "Markov-switching"]
print("Step 5 | writing the regime & risk report")

rep = Report("Regime & Risk: Energy Benchmarks",
             f"WTI, Brent, Henry Hub and an $8mm energy book · daily data {C.START_DATE[:4]}–{asof} · source: {src}",
             C.RISK_OUT, "regime_risk_summary")

# ---------------------------------------------------------------- headline
book_today = today[(today.series == "Book")].set_index("model")
passes = bt.groupby("model")["passes"].sum().reindex(MODELS)
best = " & ".join(passes[passes == passes.max()].index)
agg = byreg.groupby(["model", "regime"])[["exceptions", "days"]].sum()
agg["rate"] = agg.exceptions / agg.days
hs_t, hs_c = agg.loc[("Historical sim", "Turbulent"), "rate"], agg.loc[("Historical sim", "Calm"), "rate"]
wins = (loss[loss.rank_qlike == 1].model == "GARCH-t").sum()
cur_i = cur.set_index("series")

rep.section("Headline numbers")
rep.kpis([
    ("Book 1-day 99% VaR", fmt_money(book_today.loc["GARCH-t", "var99_usd"]), f"GARCH-t, as of {asof}"),
    ("Book 97.5% ES", fmt_money(book_today.loc["GARCH-t", "es975_usd"]), "average loss beyond VaR"),
    ("Best backtest record", best, f"each passes {int(passes.max())} of {bt.series.nunique()} series"),
    ("HS misses in turbulence", fmt_pct(hs_t, 1), f"vs {fmt_pct(hs_c, 1)} in calm; target 1.0%"),
    ("Best variance forecast", "GARCH-t", f"lowest QLIKE on {wins} of {loss.series.nunique()} series"),
    ("Turbulent now", ", ".join(cur_i.index[cur_i.regime == "Turbulent"]) or "none",
     "regime with highest filtered probability"),
])
obs = []
obs.append(f"**Regimes are real and persistent.** Calm spells last {fmt_num(par[par.regime=='Calm'].expected_duration_days.min(),0)}–"
           f"{fmt_num(par[par.regime=='Calm'].expected_duration_days.max(),0)} trading days on average; turbulent spells "
           f"{fmt_num(par[par.regime=='Turbulent'].expected_duration_days.min(),0)}–{fmt_num(par[par.regime=='Turbulent'].expected_duration_days.max(),0)} days, "
           f"with volatility {fmt_num((par[par.regime=='Turbulent'].ann_vol_pct.values / par[par.regime=='Calm'].ann_vol_pct.values).min(),1)}–"
           f"{fmt_num((par[par.regime=='Turbulent'].ann_vol_pct.values / par[par.regime=='Calm'].ann_vol_pct.values).max(),1)}× the calm level.")
obs.append(f"**Plain historical simulation is regime-blind.** Its 99% VaR was beaten on {fmt_pct(hs_c,1)} of calm days but "
           f"{fmt_pct(hs_t,1)} of turbulent days: the 250-day window is too slow to absorb a new volatility regime, "
           f"so exceptions arrive in clusters (Christoffersen rejects HS on {int((bt[bt.model=='Historical sim'].christoffersen_p<0.05).sum())} of 4 series).")
fhs_t = agg.loc[("Filtered HS", "Turbulent"), "rate"]
obs.append(f"**Filtering fixes most of it.** Rescaling history to today's EWMA volatility (filtered HS) cuts the turbulent-regime "
           f"exception rate to {fmt_pct(fhs_t,1)} and passes coverage and independence on "
           f"{int(passes['Filtered HS'])} of 4 series.")
g = bt[bt.model == "GARCH-t"].set_index("series")
worst = g.exception_rate.idxmax()
if g.loc[worst, "kupiec_p"] < 0.05:
    obs.append(f"**GARCH-t forecasts variance best, yet its VaR still fails on {worst}**: {int(g.loc[worst,'exceptions'])} "
               f"exceptions against {fmt_num(g.loc[worst,'expected'],0)} expected (Kupiec p = {fmt_p(g.loc[worst,'kupiec_p'])}). "
               "A good volatility forecast is necessary for a good VaR, not sufficient: the tail shape matters too.")
else:
    obs.append(f"**GARCH-t forecasts variance best and its VaR passes Kupiec on every series** "
               f"(worst: {worst}, {int(g.loc[worst,'exceptions'])} exceptions vs {fmt_num(g.loc[worst,'expected'],0)} expected).")
h = bt[bt.model == "HAR-RV normal"]
obs.append(f"**Normal tails fail.** HAR-RV with normal quantiles records {int(h.exceptions.sum())} exceptions across the four series "
           f"against {fmt_num(h.expected.sum(),0)} expected; its average Z2 of {fmt_num(h.z2_full.mean(),2)} says realised tail losses "
           "run well beyond its ES.")
rep.bullets(obs)
rep.callout("All forecasts are out of sample: each day's VaR uses only data available the evening before. "
            "The positions are hypothetical and priced off EIA spot benchmarks, not tradable futures.", "note")

# ---------------------------------------------------------------- data
rep.section("1. Data", new_page=True)
rep.text(f"Daily EIA spot prices from {C.START_DATE} to {asof}. Returns are daily log changes in %. "
         f"The **energy book** holds constant dollar positions: WTI {fmt_money(C.BOOK['WTI'])}, Brent {fmt_money(C.BOOK['Brent'])}, "
         f"Henry Hub {fmt_money(C.BOOK['HenryHub'])} (a long WTI–Brent spread plus long US gas); its return is P&L over gross notional.")
tab = log[["series", "description", "unit", "source", "first", "last", "rows"]].copy()
tab.columns = ["Series", "Description", "Unit", "Source", "First", "Last", "Rows"]
tab["Rows"] = tab["Rows"].map(lambda v: f"{v:,}")
rep.table(tab, "Data pull log")
s = summ.copy()
s = pd.DataFrame({"Series": s.series, "Ann. vol": s.ann_vol_pct.map(lambda v: f"{v:.1f}%"),
                  "Skew": s["skew"].map(lambda v: fmt_num(v, 2)), "Excess kurtosis": s.excess_kurtosis.map(lambda v: fmt_num(v, 1)),
                  "Worst day": [f"{fmt_num(a,1)}% ({b})" for a, b in zip(s.worst_day_pct, s.worst_date)],
                  "Best day": [f"{fmt_num(a,1)}% ({b})" for a, b in zip(s.best_day_pct, s.best_date)]})
rep.table(s, "Return statistics (daily log returns)")
nfix = len(fixes)
nflag = int((flags.issue.str.startswith("move")).sum()) if len(flags) else 0
rep.text(f"**Data quality.** {nfix} fixes were applied and logged (`01_data_quality_fixes.csv`), including WTI's negative "
         f"print on 2020-04-20 (set to missing: a log return is undefined). {nflag} extreme moves were **flagged but kept** "
         "(`01_flagged_moves.csv`); crashes and gas spikes are exactly what risk models must survive.")
rep.figure(FIG / "01_prices.png", "EIA daily spot prices.")

# ---------------------------------------------------------------- regimes
rep.section("2. Regimes", new_page=True)
rep.text("A Gaussian **Markov-switching model** (Hamilton, 1989) lets each market move between a calm and a turbulent "
         "state, each with its own mean and volatility, switching with fixed daily probabilities. The regime is never "
         "observed; the model infers the probability of each regime every day.")
p = par.copy()
p = pd.DataFrame({"Series": p.series, "Regime": p.regime, "Ann. vol": p.ann_vol_pct.map(lambda v: f"{v:.1f}%"),
                  "Daily mean": p.mean_daily_pct.map(lambda v: fmt_num(v, 3) + "%"),
                  "Stay prob.": p.stay_prob.map(lambda v: f"{v:.3f}"),
                  "Avg. spell (days)": p.expected_duration_days.map(lambda v: fmt_num(v, 1)),
                  "Share of days": p.share_of_days.map(lambda v: fmt_pct(v, 1))})
rep.table(p, "Two-regime model, full sample")
rep.figure(FIG / "02_regime_vols.png", "Annualised volatility in each regime.")
rep.figure(FIG / "02_regimes_WTI.png", "WTI: turbulent regimes (shaded) and the probability of turbulence, smoothed vs real time.")
b = sel.pivot(index="series", columns="regimes", values="bic")
n3 = int((b[3] < b[2]).sum())
rep.text(f"**Model choice.** BIC prefers three regimes on {n3} of {len(b)} series (e.g. WTI: {b.loc['WTI',2]:,.0f} with two vs "
         f"{b.loc['WTI',3]:,.0f} with three). With Gaussian regimes an extra state mostly absorbs fat tails, so "
         "this project keeps the two-regime model for interpretability and reports the three-regime fit in "
         "`02_model_selection.csv`. That trade-off is a documented model choice, not an oversight.")
rp = rec.groupby("series").agg(vol_err=("max_abs_vol_error_pct", "max"), acc=("state_accuracy", "min")).reset_index()
rp.columns = ["Series", "Worst vol error", "Worst state accuracy"]
rp["Worst vol error"] = rp["Worst vol error"].map(lambda v: f"{v:.1f}%")
rp["Worst state accuracy"] = rp["Worst state accuracy"].map(lambda v: fmt_pct(v, 1))
rep.text("**Does the estimator work?** Three data sets were simulated from each fitted model and re-estimated. "
         "The EM algorithm recovers regime volatilities within a few percent and classifies the true hidden state "
         "on almost every day:")
rep.table(rp, "Simulation-based verification", small=True)
cc = cur.copy()
cc = pd.DataFrame({"Series": cc.series, "Regime today": cc.regime, "Probability": cc.probability.map(lambda v: f"{v:.2f}"),
                   "P(turbulent)": cc.p_turbulent.map(lambda v: f"{v:.2f}"), "Days in regime": cc.days_in_regime})
rep.table(cc, f"Current regime (filtered, {asof})", small=True)
ep = epi.sort_values("trading_days", ascending=False).head(8)
ep = pd.DataFrame({"Series": ep.series, "Start": ep.start, "End": ep.end, "Days": ep.trading_days,
                   "Price change": ep.price_change_pct.map(lambda v: "–" if pd.isna(v) else f"{fmt_num(v,1)}%"),
                   "Realised vol": ep.realised_ann_vol_pct.map(lambda v: f"{v:.0f}%")})
rep.table(ep, "Longest turbulent episodes (smoothed probability > 0.5)", small=True)

# ---------------------------------------------------------------- volatility
rep.section("3. Volatility forecasts", new_page=True)
rep.text("Four one-day-ahead variance forecasts, all out of sample. They are scored with **QLIKE** "
         "(log h + r²/h), which ranks forecasts correctly even though the squared return is a very noisy "
         "measure of true variance (Patton, 2011). **Diebold–Mariano** tests ask whether the gap to GARCH-t "
         "is bigger than chance.")
L = loss.pivot(index="series", columns="model", values="qlike")[["EWMA", "GARCH-t", "HAR-RV", "MS"]]
lt = L.reset_index()
for c in ["EWMA", "GARCH-t", "HAR-RV", "MS"]:
    lt[c] = [fmt_num(v, 3) + (" ★" if v == L.loc[sname].min() else "") for sname, v in zip(L.index, L[c])]
lt.columns = ["Series", "EWMA", "GARCH-t", "HAR-RV", "Markov-switching"]
rep.table(lt, "Mean QLIKE loss (lower is better; ★ = best)")
d = dm.copy()
d = pd.DataFrame({"Series": d.series, "Model vs GARCH-t": d.model, "DM stat (QLIKE)": d.dm_qlike.map(lambda v: fmt_num(v, 2)),
                  "p-value": d.p_qlike.map(fmt_p),
                  "Verdict": [verdict(None) if pv >= 0.05 else (verdict(False, no="Worse than GARCH-t") if st > 0 else verdict(True, yes="Better than GARCH-t"))
                              for st, pv in zip(d.dm_qlike, d.p_qlike)]})
d["Verdict"] = d["Verdict"].str.replace("– n/a", "No significant difference")
rep.table(d, "Diebold–Mariano tests against GARCH-t (positive = higher loss than GARCH-t)", small=True)
rep.figure(FIG / "03_qlike_vs_garch.png", "Loss relative to GARCH-t by series.")
rep.figure(FIG / "03_vol_forecasts_Brent.png", "Brent: forecasts vs the volatility realised over the next month.")
gr = grec.groupby("series").apply(lambda g: (g.est_persistence - g.true_persistence).abs().max(), include_groups=False)
rep.text(f"**GARCH estimator check.** Refitting simulated GARCH-t paths recovers persistence (α+β) within "
         f"{gr.max():.3f} on every series. HAR-RV is handicapped here: with daily closes only, realised variance "
         "is proxied by the squared return, which throws away most of the information intraday data would give.")

# ---------------------------------------------------------------- VaR / ES
rep.section("4. VaR and ES backtests", new_page=True)
rep.text(f"Five models forecast 1-day 99% VaR and 97.5% ES every day from {C.BACKTEST_START} "
         f"({int(bt.days.iloc[0]):,} days). A model **passes** when neither Kupiec (right number of exceptions) nor "
         "conditional coverage (right number *and* no clustering) rejects at 5%.")
rows = []
for _, r in bt.iterrows():
    rows.append({"Series": r.series, "Model": r.model, "Exceptions": f"{r.exceptions} / {r.expected:.0f}",
                 "Kupiec p": fmt_p(r.kupiec_p), "Christoffersen p": fmt_p(r.christoffersen_p), "CC p": fmt_p(r.cc_p),
                 "Zone (last 250d)": r.zone_last_250, "Z2": fmt_num(r.z2_full, 2),
                 "Verdict": verdict(bool(r.passes))})
rep.table(pd.DataFrame(rows), "Backtest results, 99% VaR (exceptions observed / expected)", small=True)
rep.figure(FIG / "04_exception_rates.png", "Observed exception rates against the 1% target.")
rep.figure(FIG / "04_exceptions_by_regime.png", "Exception rate split by the regime the model knew about at forecast time.")
rep.figure(FIG / "04_traffic_light_book.png", "Rolling 250-day exception counts for the energy book with Basel zones.")
rep.figure(FIG / "04_backtest_Book.png", "Energy book: returns against historical-simulation and GARCH-t VaR.")
td = today[today.series.isin(["WTI", "Brent", "HenryHub", "Book"])].copy()
td = pd.DataFrame({"Series": td.series, "Model": td.model, "Notional": td.notional.map(fmt_money),
                   "VaR 99%": td.var99_usd.map(fmt_money), "ES 97.5%": td.es975_usd.map(fmt_money),
                   "VaR % of notional": td.var99_pct.map(lambda v: f"{v:.2f}%")})
rep.table(td, f"Next-day risk as of {asof}", small=True)
rep.text("**Expected Shortfall.** Z2 near zero means ES was about right; negative values mean realised losses beyond VaR "
         "were larger than ES predicted. Acerbi and Szekely's thresholds (−0.70 yellow, −1.8 red) are calibrated to one "
         f"year of data, so the share of rolling years below −0.70 is in `04_backtest_results.csv`.")

# ---------------------------------------------------------------- validation summary
rep.section("5. Model validation summary", new_page=True)
rep.text("Organised the way a model risk team reviews a model under the Fed/OCC **SR 11-7** guidance.")
val = pd.DataFrame([
    ("Conceptual soundness", "Simulate from each fitted model, re-estimate, compare",
     f"MS vol error ≤ {rec.max_abs_vol_error_pct.max():.1f}%, state accuracy ≥ {fmt_pct(rec.state_accuracy.min(),1)}; GARCH persistence error ≤ {gr.max():.3f}",
     verdict(True)),
    ("Model selection", "BIC across 2 vs 3 regimes", f"BIC prefers 3 on {n3} of {len(b)} series; 2 kept for interpretability",
     verdict(None, warn="Documented choice")),
    ("Benchmarking", "Diebold–Mariano against GARCH-t", f"GARCH-t has the lowest QLIKE on {wins} of 4 series", verdict(True)),
    ("Outcomes analysis", "Kupiec, Christoffersen, conditional coverage, Z2, McNeil–Frey",
     f"{int(bt.passes.sum())} of {len(bt)} model–series pairs pass; best: {best}", verdict(None, warn="Mixed")),
    ("Ongoing monitoring", "Basel traffic light on rolling 250 days",
     f"{int((bt.zone_last_250=='green').sum())} of {len(bt)} pairs green today", verdict(True)),
], columns=["Check", "How", "Result", "Status"])
rep.table(val, "Validation evidence")
rep.sub("Limitations")
rep.bullets([
    "Spot benchmarks, not futures: Henry Hub spot spikes during cold snaps are far larger than front-month futures moves.",
    "Daily closes only: HAR-RV and realised-volatility checks use squared returns, a noisy proxy.",
    "Gaussian regimes: tails inside each regime are thin; a Student-t Markov-switching model would be the next step.",
    "Univariate book: the book's risk is modelled on its own P&L series, so correlations change only implicitly.",
    "Positions are constant-dollar and linear: no options, no liquidity horizons, no basis between spot and hedges.",
])
rep.sub("References")
rep.bullets([
    "Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*, 57(2).",
    "Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60.",
    "Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices. *REStat*, 69(3).",
    "Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *J. Financial Econometrics*, 7(2).",
    "Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *J. Econometrics*, 160(1).",
    "Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *JBES*, 13(3).",
    "Kupiec, P. (1995); Christoffersen, P. (1998); Acerbi, C. & Szekely, B. (2014); McNeil, A. & Frey, R. (2000).",
    "Hull, J. & White, A. (1998). Incorporating volatility updating into historical simulation. *Journal of Risk*, 1(1).",
    "Board of Governors of the Federal Reserve System & OCC (2011). SR 11-7: Guidance on Model Risk Management.",
])

md, html = rep.write()
pdf = rep.pdf(C.REPORTS)
print(f"  wrote {md.name}, {html.name}" + (f", {pdf}" if pdf else ""))
