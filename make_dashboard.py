"""
make_dashboard.py | Build the static dashboard (docs/index.html) and its README screenshots.

Reads only the CSV/pickle outputs of the two workflows, so run it after `bash run.sh`
(run.sh calls it last). Open docs/index.html in any browser, or publish with GitHub Pages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import config as C  # noqa: E402
from lib.dashboard import Dashboard, style_fig  # noqa: E402
from lib.report import fmt_money, fmt_num, fmt_p, fmt_pct, verdict  # noqa: E402
from lib.style import SERIES, STATUS  # noqa: E402

RT, FT = C.RISK_OUT / "tables", C.FACT_OUT / "tables"
rt = lambda n: pd.read_csv(RT / f"{n}.csv")
ft = lambda n: pd.read_csv(FT / f"{n}.csv")
D = pd.read_pickle(C.DATA_PROC / "daily.pkl")
G = pd.read_pickle(C.DATA_PROC / "regimes.pkl")
V = pd.read_pickle(C.DATA_PROC / "vol_forecasts.pkl")
F = pd.read_pickle(C.DATA_PROC / "var_forecasts.pkl")
PUB = pd.read_pickle(C.DATA_PROC / "published.pkl")
GR = pd.read_pickle(C.DATA_PROC / "grid.pkl")
R, PX = D["returns"], D["prices"]
MODELS = ["Historical sim", "Filtered HS", "GARCH-t", "HAR-RV normal", "Markov-switching"]
MCOL = dict(zip(MODELS, SERIES))
VMOD = ["EWMA", "GARCH-t", "HAR-RV", "MS"]
VCOL = dict(zip(VMOD, SERIES))
asof = R.index[-1].date()
REPO = "https://github.com/n9omi/regime-risk-and-model-validation"
print("Building dashboard")

bt, today, cur, par = rt("04_backtest_results"), rt("04_var_es_today"), rt("02_current_regime"), rt("02_ms_parameters")
loss, dm, byreg = rt("03_forecast_losses"), rt("03_diebold_mariano"), rt("04_exceptions_by_regime")
perf, dec, cv, pbo, ds, gridp = (ft("02_published_performance"), ft("02_decay_windows"), ft("03_cv_results"),
                                 ft("04_pbo"), ft("04_deflated_sharpe"), ft("03_grid_performance"))
net = perf[perf.returns == "net"].set_index("strategy")

d = Dashboard("Regime Risk & Model Validation",
              f"Energy regimes, volatility, VaR/ES backtests and commodity factor validation · daily data to {asof}",
              links=[("Risk report (PDF)", f"{REPO}/blob/main/reports/regime_risk_summary.pdf"),
                     ("Factor report (PDF)", f"{REPO}/blob/main/reports/factor_validation_summary.pdf"),
                     ("GitHub", REPO)],
              footer="Data: EIA daily spot prices (via FRED or its public GitHub mirror) and the World Bank Pink Sheet. "
                     "Hypothetical positions; educational project, not investment advice.")


def dx(idx):
    """Dates as compact YYYY-MM-DD strings (keeps the page small)."""
    idx = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else pd.DatetimeIndex(idx)
    return idx.strftime("%Y-%m-%d").tolist()


def ry(a, d=4):
    """Round values for the page; the CSVs keep full precision."""
    return [None if not np.isfinite(v) else v for v in np.round(np.asarray(a, float), d).tolist()]


# =============================================================== Overview
t = d.tab("overview", "Overview",
          "Two workflows, one question: <b>can we trust the numbers?</b> The first tracks volatility regimes in energy "
          "benchmarks and backtests five VaR/ES models out of sample. The second rebuilds three published commodity factor "
          "strategies and checks them for overfitting.")
agg = byreg.groupby(["model", "regime"])[["exceptions", "days"]].sum()
agg["rate"] = agg.exceptions / agg.days
bk = today[today.series == "Book"].set_index("model")
passes = bt.groupby("model")["passes"].sum().reindex(MODELS)
pbi = pbo.set_index("variant_set")
turb_now = cur[cur.regime == "Turbulent"].series.tolist()
t.kpis([
    ("Book 99% VaR (1-day)", fmt_money(bk.loc["GARCH-t", "var99_usd"]), f"GARCH-t · $8mm gross · {asof}"),
    ("Regime today", "Turbulent" if turb_now else "Calm", ", ".join(turb_now) + " turbulent" if turb_now else "all four series calm",
     "warning" if turb_now else "good"),
    ("Best VaR models", " & ".join(passes[passes == passes.max()].index), f"each passes {int(passes.max())} of 4 series", "good"),
    ("HS misses in turbulence", fmt_pct(agg.loc[("Historical sim", "Turbulent"), "rate"], 1), "target 1.0% · regime-blind", "critical"),
    ("Value net Sharpe", fmt_num(net.loc["VALUE", "sharpe"], 2), f"Newey-West t = {fmt_num(net.loc['VALUE','nw_t'],1)}", "good"),
    ("PBO, TSMOM grid", fmt_pct(pbi.loc["TSMOM", "pbo"], 0), "chance the chosen look-back is overfit", "warning"),
])
c = cur.copy()
t.table(pd.DataFrame({"Series": c.series, "Regime": c.regime, "Probability": c.probability.map(lambda v: f"{v:.2f}"),
                      "P(turbulent)": c.p_turbulent.map(lambda v: f"{v:.2f}"), "Days in regime": c.days_in_regime}),
        "Current regime", f"Filtered probability, {asof}", half=True)
tb = today[today.series == "Book"]
t.table(pd.DataFrame({"Model": tb.model, "VaR 99%": tb.var99_usd.map(fmt_money), "ES 97.5%": tb.es975_usd.map(fmt_money),
                      "% of gross": tb.var99_pct.map(lambda v: f"{v:.2f}%")}),
        "Energy book: next-day risk", "+$4mm WTI, −$2mm Brent, +$2mm Henry Hub", half=True)
fig = go.Figure()
for m in MODELS:
    fig.add_bar(x=["Calm", "Turbulent"], y=[agg.loc[(m, "Calm"), "rate"] * 100, agg.loc[(m, "Turbulent"), "rate"] * 100],
                name=m, marker_color=MCOL[m], hovertemplate="%{x}: %{y:.2f}%<extra>" + m + "</extra>")
fig.add_hline(y=1, line_color="#52514e", line_width=1, annotation_text="target 1%", annotation_position="top left")
style_fig(fig, 320, unified=False).update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, yaxis_title="Exception rate (%)")
t.chart(fig, "VaR exceptions by regime", "Share of days a loss beat 99% VaR, split by the regime known at forecast time", half=True)
fig = go.Figure()
for k, cc in zip(C.PUBLISHED, SERIES):
    r = PUB["returns"][k]["net"]
    fig.add_scatter(x=dx(r.index), y=ry((1 + r).cumprod(), 4), name=C.PUBLISHED[k]["name"], line=dict(color=cc, width=1.6))
style_fig(fig, 320).update_layout(yaxis=dict(type="log", dtick=1, title="Growth of $1 (log)"))
t.chart(fig, "Published factor strategies", "Net of 10bp costs, monthly", half=True)

# =============================================================== Regimes
t = d.tab("regimes", "Regimes", "A two-state Gaussian Markov-switching model (Hamilton, 1989). Shaded bands mark days the smoothed "
          "probability of the turbulent regime exceeds 0.5. The lower panel compares the hindsight (smoothed) probability "
          "with the real-time filtered one that the VaR models actually used.")
figs = {}
for s in R.columns:
    fit = G["fits"][s]
    lvl = PX[s] if s in PX else (1 + R[s] / 100).cumprod() * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    fig.add_scatter(x=dx(lvl.index), y=ry(lvl.values, 3), name="Price" if s in PX else "Book index",
                    line=dict(color=SERIES[0], width=1.2), row=1, col=1)
    rtp = G["realtime"][s]["filt"].iloc[:, -1].dropna()
    fig.add_scatter(x=dx(rtp.index), y=ry(rtp.values, 3), name="Real-time P(turbulent)",
                    line=dict(color=SERIES[2], width=0.7), opacity=0.8, row=2, col=1)
    fig.add_scatter(x=dx(R.index), y=ry(fit.smoothed[:, -1], 3), name="Smoothed P(turbulent)",
                    line=dict(color=SERIES[1], width=1.4), row=2, col=1)
    turb = pd.Series(fit.smoothed[:, -1] > 0.5, index=R.index)
    grp = (turb != turb.shift()).cumsum()
    for _, g in turb[turb].groupby(grp[turb]):
        if len(g) >= 3:
            fig.add_vrect(x0=g.index[0].strftime("%Y-%m-%d"), x1=g.index[-1].strftime("%Y-%m-%d"), fillcolor="#f3d9cf",
                          opacity=0.9, line_width=0, layer="below", row=1, col=1)
    style_fig(fig, 470)
    fig.update_yaxes(title_text=C.DAILY[s]["unit"] if s in C.DAILY else "Index", row=1, col=1)
    fig.update_yaxes(range=[0, 1], title_text="Probability", row=2, col=1)
    figs[s] = fig
t.chart_set("Prices and regimes", figs, "Choose a series")
p = par.copy()
t.table(pd.DataFrame({"Series": p.series, "Regime": p.regime, "Ann. vol": p.ann_vol_pct.map(lambda v: f"{v:.1f}%"),
                      "Stay prob.": p.stay_prob.map(lambda v: f"{v:.3f}"), "Avg. spell (days)": p.expected_duration_days.map(lambda v: fmt_num(v, 1)),
                      "Share of days": p.share_of_days.map(lambda v: fmt_pct(v, 1))}),
        "Regime parameters", "Full-sample fit", half=True)
rec = rt("02_simulation_recovery").groupby("series").agg(e=("max_abs_vol_error_pct", "max"), a=("state_accuracy", "min")).reset_index()
sel = rt("02_model_selection").pivot(index="series", columns="regimes", values="bic").reset_index()
t.table(pd.DataFrame({"Series": rec.series, "Worst vol error": rec.e.map(lambda v: f"{v:.1f}%"),
                      "Worst state accuracy": rec.a.map(lambda v: fmt_pct(v, 1)),
                      "BIC (2 regimes)": sel.set_index("series").loc[rec.series, 2].map(lambda v: f"{v:,.0f}").values,
                      "BIC (3 regimes)": sel.set_index("series").loc[rec.series, 3].map(lambda v: f"{v:,.0f}").values}),
        "Estimator checks", "Simulate from the fit, re-estimate, compare; model choice by BIC", half=True,
        note="BIC prefers three Gaussian regimes (the extra state absorbs fat tails); two are kept for interpretability.")

# =============================================================== Volatility
t = d.tab("volatility", "Volatility", "One-day-ahead variance forecasts, all out of sample, scored with QLIKE against the squared return "
          "(Patton, 2011). Grey is the volatility realised over the following month, for reference.")
figs = {}
for s in R.columns:
    h = V["forecasts"][s]
    real = (R[s] ** 2).rolling(22).mean().shift(-21).loc[h.index].pipe(np.sqrt) * np.sqrt(252)
    fig = go.Figure()
    fig.add_scatter(x=dx(real.index), y=ry(real.values, 2), name="Realised (next 22d)", line=dict(color="#c3c2b7", width=2.2))
    for m in VMOD:
        fig.add_scatter(x=dx(h.index), y=ry(np.sqrt(h[m] * 252), 2), name=m, line=dict(color=VCOL[m], width=1))
    style_fig(fig, 400).update_layout(yaxis=dict(type="log", dtick=1, title="Annualised vol (%), log"))
    figs[s] = fig
t.chart_set("Volatility forecasts", figs)
L = loss.pivot(index="series", columns="model", values="qlike")[VMOD]
tab = L.reset_index()
for m in VMOD:
    tab[m] = [fmt_num(v, 3) + (" ★" if v == L.loc[s].min() else "") for s, v in zip(L.index, L[m])]
t.table(tab.rename(columns={"series": "Series"}), "Mean QLIKE (lower is better)", "★ = best forecaster for that series", half=True)
t.table(pd.DataFrame({"Series": dm.series, "Model": dm.model, "DM stat": dm.dm_qlike.map(lambda v: fmt_num(v, 2)),
                      "p": dm.p_qlike.map(fmt_p),
                      "Result": [("✖ Worse" if st > 0 else "✔ Better") if pv < 0.05 else "– Tie" for st, pv in zip(dm.dm_qlike, dm.p_qlike)]}),
        "Diebold–Mariano vs GARCH-t", "Positive statistic = higher loss than GARCH-t", half=True)

# =============================================================== VaR backtests
t = d.tab("var", "VaR backtests", "Five models forecast 1-day 99% VaR and 97.5% ES every day since 2010 using only past data. "
          "A model passes when neither Kupiec nor conditional coverage rejects at 5%.")
t.kpis([(f"{m}", f"{int(bt[bt.model == m].passes.sum())} / 4", "series passed",
         "good" if bt[bt.model == m].passes.sum() >= 3 else ("warning" if bt[bt.model == m].passes.sum() >= 2 else "critical"))
        for m in MODELS])
figs = {}
for s in R.columns:
    f = F[s]
    x = f[("actual", "x")]
    fig = go.Figure()
    fig.add_scatter(x=dx(x.index), y=ry(x.values, 2), name="Daily return", mode="lines", line=dict(color="#c3c2b7", width=0.8))
    for m in ("Historical sim", "Filtered HS", "GARCH-t"):
        v = -f[(m, "var99")]
        fig.add_scatter(x=dx(v.index), y=ry(v.values, 2), name=f"−VaR {m}", line=dict(color=MCOL[m], width=1.1))
        e = x < v
        fig.add_scatter(x=dx(x.index[e]), y=ry(x[e].values, 2), mode="markers", name=f"Exceptions {m}", showlegend=False,
                        marker=dict(color=MCOL[m], size=6, line=dict(color="#fcfcfb", width=1)))
    lo = np.nanpercentile(x, 0.05)
    style_fig(fig, 420).update_layout(yaxis=dict(range=[max(lo * 1.4, -60), np.nanpercentile(x, 99.95) * 1.2], title="% return"))
    figs[s] = fig
t.chart_set("Returns vs VaR forecasts", figs, "Dots mark exceptions; historical simulation, filtered HS and GARCH-t shown")
rows = pd.DataFrame({"Series": bt.series, "Model": bt.model, "Exceptions": [f"{a} / {b:.0f}" for a, b in zip(bt.exceptions, bt.expected)],
                     "Kupiec p": bt.kupiec_p.map(fmt_p), "Christoffersen p": bt.christoffersen_p.map(fmt_p),
                     "CC p": bt.cc_p.map(fmt_p), "Zone (250d)": bt.zone_last_250, "Z2": bt.z2_full.map(lambda v: fmt_num(v, 2)),
                     "Verdict": [verdict(bool(v)) for v in bt.passes]})
t.table(rows, "Backtest results", "99% VaR; exceptions observed / expected; Z2 ≈ 0 means ES was about right")
fig = go.Figure()
fb = F["Book"]
xb = fb[("actual", "x")].values
ymax = 12
for m in MODELS:
    exc = (xb < -fb[(m, "var99")].values).astype(int)
    cnt = np.convolve(exc, np.ones(250, int), mode="valid")
    ymax = max(ymax, cnt.max() + 1)
    fig.add_scatter(x=dx(fb.index[249:]), y=cnt, name=m, line=dict(color=MCOL[m], width=1.2, shape="hv"))
for y0, y1, col in ((0, 4.5, "#e8f5e8"), (4.5, 9.5, "#fdf3dc"), (9.5, ymax, "#fbe4e4")):
    fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.7, line_width=0, layer="below")
style_fig(fig, 340).update_layout(yaxis=dict(range=[0, ymax], title="Exceptions, last 250 days"))
t.chart(fig, "Basel traffic light: energy book", "Green 0–4, yellow 5–9, red 10+", half=True)
fig = go.Figure()
for m in MODELS:
    sub = bt[bt.model == m]
    fig.add_scatter(x=sub.exception_rate * 100, y=sub.series, mode="markers", name=m,
                    marker=dict(color=MCOL[m], size=11, line=dict(color="#fcfcfb", width=2)),
                    hovertemplate="%{y}: %{x:.2f}%<extra>" + m + "</extra>")
fig.add_vline(x=1, line_color="#52514e", line_width=1, annotation_text="target 1%", annotation_position="bottom right")
style_fig(fig, 340, unified=False).update_layout(xaxis_title="Exception rate (%)")
fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
fig.update_yaxes(showgrid=False)
t.chart(fig, "Exception rates", "Observed share of days beyond 99% VaR", half=True)

# =============================================================== Factors
t = d.tab("factors", "Factor replication", "Three published commodity strategies rebuilt on World Bank Pink Sheet spot prices with "
          "their papers' settings. <b>Caveat:</b> spot averages are not futures returns, so read these as upper bounds for value "
          "and as indicative for momentum.")
t.kpis([(C.PUBLISHED[k]["name"], fmt_num(net.loc[k, "sharpe"], 2), f"net Sharpe · t = {fmt_num(net.loc[k,'nw_t'],1)}",
         "good" if net.loc[k, "nw_t"] > 3 else ("warning" if net.loc[k, "sharpe"] > 0 else "critical")) for k in C.PUBLISHED])
figs = {}
for k, cc in zip(C.PUBLISHED, SERIES):
    r = PUB["returns"][k]
    fig = go.Figure()
    for kind, col in (("gross", "#c3c2b7"), ("net", cc)):
        fig.add_scatter(x=dx(r.index), y=ry((1 + r[kind]).cumprod(), 4), name=kind.title(), line=dict(color=col, width=1.6))
    for dt, lab in ((C.PUBLISHED[k]["sample_end"], "sample ends"), (C.PUBLISHED[k]["published"], "published")):
        fig.add_vline(x=pd.Period(dt, "M").to_timestamp(), line_color="#52514e", line_width=1)
        fig.add_annotation(x=pd.Period(dt, "M").to_timestamp(), y=1.02, yref="paper", text=lab, showarrow=False,
                           xanchor="right" if lab == "sample ends" else "left", font=dict(size=11, color="#52514e"))
    style_fig(fig, 380).update_layout(yaxis=dict(type="log", dtick=1, title="Growth of $1 (log)"))
    figs[C.PUBLISHED[k]["name"]] = fig
t.chart_set("Cumulative returns", figs, "Gross and net of costs, with each paper's sample end and publication date")
wins = ["In paper's sample", "After sample, before publication", "After publication"]
fig = go.Figure()
for w, cc in zip(wins, (SERIES[0], SERIES[2], SERIES[1])):
    sub = dec[dec.window == w].set_index("strategy").reindex(list(C.PUBLISHED))
    fig.add_bar(x=[C.PUBLISHED[k]["name"] for k in sub.index], y=sub.sharpe, name=w, marker_color=cc,
                customdata=sub.months, hovertemplate="%{x}<br>Sharpe %{y:.2f} over %{customdata} months<extra>" + w + "</extra>")
style_fig(fig, 340, unified=False).update_layout(barmode="group", yaxis_title="Net Sharpe")
t.chart(fig, "Does the edge survive publication?", "McLean–Pontiff windows", half=True)
pp = perf.copy()
t.table(pd.DataFrame({"Strategy": pp.strategy, "Returns": pp.returns, "Sharpe": pp.sharpe.map(lambda v: fmt_num(v, 2)),
                      "Ann. return": pp.ann_return.map(lambda v: fmt_pct(v, 1)), "NW t": pp.nw_t.map(lambda v: fmt_num(v, 2)),
                      "Max DD": pp.max_drawdown.map(lambda v: fmt_pct(v, 0))}),
        "Full-sample performance", "Costs: 10bp per unit of turnover", half=True)

# =============================================================== Overfitting
t = d.tab("overfitting", "Overfitting checks", f"{len(gridp)} strategy variants, tested the way a skeptical reviewer would: "
          "purged combinatorial cross-validation, the probability of backtest overfitting (PBO) and the deflated Sharpe ratio.")
t.kpis([(f"PBO · {k}", fmt_pct(pbi.loc[k, "pbo"], 0), f"{int(pbi.loc[k,'n_variants'])} variants",
         "good" if pbi.loc[k, "pbo"] < 0.25 else ("warning" if pbi.loc[k, "pbo"] < 0.5 else "critical")) for k in pbi.index])
paths = pd.read_csv(FT / "03_cpcv_path_sharpes.csv")
fig = go.Figure()
sets = list(pbi.index)
for i, k in enumerate(sets):
    pth = paths[paths.variant_set == k].oos_sharpe
    row = cv[(cv.variant_set == k) & (cv.method == "Purged CPCV")].iloc[0]
    fig.add_scatter(x=pth, y=[k] * len(pth), mode="markers", name="CPCV path (OOS)", showlegend=i == 0,
                    marker=dict(color=SERIES[0], size=9, opacity=0.75, line=dict(color="#fcfcfb", width=1)))
    fig.add_scatter(x=[row.in_sample_best_sharpe], y=[k], mode="markers", name="In-sample best", showlegend=i == 0,
                    marker=dict(color=SERIES[1], size=13, symbol="diamond", line=dict(color="#fcfcfb", width=2)))
style_fig(fig, 280, unified=False).update_layout(xaxis_title="Annualised net Sharpe")
fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
t.chart(fig, "In-sample promise vs out-of-sample delivery", "Purged CPCV: 45 splits, 9 paths", half=True)
t.table(pd.DataFrame({"Variant set": pbo.variant_set, "Variants": pbo.n_variants, "CSCV splits": pbo.splits.map(lambda v: f"{v:,}"),
                      "PBO": pbo.pbo.map(lambda v: fmt_pct(v, 1)), "OOS Sharpe < 0": pbo.share_oos_sharpe_negative.map(lambda v: fmt_pct(v, 0)),
                      "Verdict": [verdict(v < 0.5, yes="Low risk", no="High risk") for v in pbo.pbo]}),
        "Probability of backtest overfitting", "Share of CSCV splits where the in-sample winner ranks below median out of sample", half=True)
fig = go.Figure()
fam_col = {"TSMOM": SERIES[0], "XSMOM": SERIES[1], "VALUE": SERIES[2]}
o = gridp.sort_values("sharpe_full_sample")
for fam, cc in fam_col.items():
    sub = o[o.family == fam]
    fig.add_bar(y=sub.variant, x=sub.sharpe_full_sample, orientation="h", name=fam, marker_color=cc,
                hovertemplate="%{y}: %{x:.2f}<extra></extra>")
style_fig(fig, 560, unified=False).update_layout(xaxis_title="Full-sample net Sharpe", bargap=0.15,
                                                  yaxis=dict(categoryorder="array", categoryarray=o.variant.tolist(), tickfont=dict(size=10)))
t.chart(fig, "Every variant in the grid", "The best one is what a naive backtest reports", half=True)
dd = ds.copy()
t.table(pd.DataFrame({"Candidate": dd.candidate, "Variant": dd.variant, "Trials": dd.n_trials, "Sharpe": dd.sharpe_ann.map(lambda v: fmt_num(v, 2)),
                      "Luck bar SR0": dd.sr0_ann.map(lambda v: fmt_num(v, 2)), "DSR": dd.dsr.map(lambda v: fmt_num(v, 3)),
                      "NW t": dd.nw_t.map(lambda v: fmt_num(v, 2)), "Verdict": [verdict(bool(v), yes="Survives", no="Not proven") for v in dd.passes_dsr_95]}),
        "Deflated Sharpe ratios", "DSR > 0.95 passes; published strategies deflated by their own family", half=True)

out = d.write(C.DOCS)
print(f"  wrote {out}")
if "--no-shots" not in sys.argv:
    shots = d.screenshot(C.DOCS)
    print(f"  {len(shots)} screenshots in docs/screenshots/")
