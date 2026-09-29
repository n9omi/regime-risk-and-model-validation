"""
04_var_es_backtest.py | Step 4: forecast VaR and ES five ways, then test them against history.

For every day since BACKTEST_START, each model forecasts 1-day 99% VaR, 97.5% VaR and
97.5% ES using only data up to the day before. The forecasts are then compared with
what happened:
  Kupiec (right number of exceptions?), Christoffersen (do exceptions cluster?),
  conditional coverage (both), the Basel traffic light, Acerbi-Szekely Z2 and the
  McNeil-Frey residual test (is ES big enough?), and exception rates by regime.
Outputs: tables/04_*.csv, figures/04_*.png, data/processed/var_forecasts.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

import config as C  # noqa: E402
from lib import regimes as ms, style, var_es as ve  # noqa: E402
from lib.volatility import ewma  # noqa: E402

style.apply()
TAB, FIG = C.RISK_OUT / "tables", C.RISK_OUT / "figures"
D = pd.read_pickle(C.DATA_PROC / "daily.pkl")
G = pd.read_pickle(C.DATA_PROC / "regimes.pkl")
V = pd.read_pickle(C.DATA_PROC / "vol_forecasts.pkl")
R, NOTIONAL = D["returns"], D["notional"]
P99, P975 = 1 - C.VAR_CONF, 1 - C.ES_CONF
MODELS = ["Historical sim", "Filtered HS", "GARCH-t", "HAR-RV normal", "Markov-switching"]
print("Step 4 | VaR / ES forecasts and backtests")


def hs_path(r, start, p, scale=None, scale_now=None):
    """Rolling (filtered) historical simulation for every day from `start`."""
    W = C.HS_WINDOW
    win = sliding_window_view(r, W)[start - W:len(r) - W]           # window for day t = r[t-W:t]
    if scale is not None:
        sw = sliding_window_view(scale, W)[start - W:len(r) - W]
        win = win / sw * scale_now[:, None]
    srt = np.sort(win, axis=1)
    k = max(int(np.floor(W * p)), 1)
    return -srt[:, k - 1], -srt[:, :k].mean(axis=1)


forecasts, results, today, regime_rows = {}, [], [], []
for s in R.columns:
    r = R[s].values
    idx = V["forecasts"][s].index
    start = R.index.get_loc(idx[0])
    x = r[start:]
    h = V["forecasts"][s]
    ew_full = ewma(r, C.EWMA_LAMBDA)       # EWMA variance for every day (the FHS look-back needs it)
    out = {}
    # 1. Historical simulation
    v99, _ = hs_path(r, start, P99)
    v975, e975 = hs_path(r, start, P975)
    out["Historical sim"] = (v99, v975, e975, np.full(len(x), np.std(r[:start])))
    # 2. Filtered HS (Hull & White 1998): rescale past returns to today's EWMA volatility
    sd_now = np.sqrt(ew_full[start:])
    v99, _ = hs_path(r, start, P99, scale=np.sqrt(ew_full), scale_now=sd_now)
    v975, e975 = hs_path(r, start, P975, scale=np.sqrt(ew_full), scale_now=sd_now)
    out["Filtered HS"] = (v99, v975, e975, sd_now)
    # 3. GARCH-t
    sd = np.sqrt(h["GARCH-t"].values)
    nu, mu = V["nu"][s].values, V["mu"][s].values
    v99, _ = ve.t_var_es(mu, sd, nu, P99)
    v975, e975 = ve.t_var_es(mu, sd, nu, P975)
    out["GARCH-t"] = (v99, v975, e975, sd)
    # 4. HAR-RV with normal quantiles (thin tails on purpose: shows what fat tails cost)
    sd = np.sqrt(h["HAR-RV"].values)
    v99, _ = ve.normal_var_es(0.0, sd, P99)
    v975, e975 = ve.normal_var_es(0.0, sd, P975)
    out["HAR-RV normal"] = (v99, v975, e975, sd)
    # 5. Markov-switching normal mixture
    rt = G["realtime"][s]
    w, m, sg = (rt[k].values[start:] for k in ("pred", "mu", "sd"))
    mv = np.array([ms.mixture_var_es(w[i], m[i], sg[i], P99)[0] for i in range(len(x))])
    me = np.array([ms.mixture_var_es(w[i], m[i], sg[i], P975) for i in range(len(x))])
    out["Markov-switching"] = (mv, me[:, 0], me[:, 1], np.sqrt(h["MS"].values))
    forecasts[s] = pd.concat({mname: pd.DataFrame(dict(var99=a, var975=b, es975=c, sd=d), index=idx)
                              for mname, (a, b, c, d) in out.items()}, axis=1)
    forecasts[s][("actual", "x")] = x

    # --- tests --------------------------------------------------------------------------
    turb = (rt["filt"].iloc[:, -1].shift(1).loc[idx] > 0.5).values   # regime known at forecast time
    for mname, (v99, v975, e975, sdv) in out.items():
        exc = x < -v99
        lr_pof, p_pof = ve.kupiec_pof(exc, P99)
        lr_ind, p_ind, trans = ve.christoffersen_ind(exc)
        lr_cc, p_cc = ve.conditional_coverage(exc, P99)
        zones, counts = ve.rolling_zones(exc, C.TRAFFIC_WINDOW)
        z2 = ve.acerbi_szekely_z2(x, v975, e975, P975)
        z2_roll = np.array([ve.acerbi_szekely_z2(x[i:i + 250], v975[i:i + 250], e975[i:i + 250], P975)
                            for i in range(0, len(x) - 250 + 1, 21)])
        mf, mf_p, n_ex = ve.mcneil_frey(x, v975, e975, sdv, seed=C.SEED)
        results.append(dict(
            series=s, model=mname, days=len(x), exceptions=int(exc.sum()), expected=len(x) * P99,
            exception_rate=exc.mean(), kupiec_lr=lr_pof, kupiec_p=p_pof, christoffersen_lr=lr_ind,
            christoffersen_p=p_ind, p_exc_after_exc=trans["p11"], cc_lr=lr_cc, cc_p=p_cc,
            zone_last_250=ve.traffic_light(int(exc[-250:].sum())), exceptions_last_250=int(exc[-250:].sum()),
            share_windows_green=zones["green"], share_windows_yellow=zones["yellow"], share_windows_red=zones["red"],
            z2_full=z2, share_years_z2_below_070=np.mean(z2_roll < -0.70), mcneil_frey_resid=mf,
            mcneil_frey_p=mf_p, es_exceedances=n_ex, avg_var99_pct=v99.mean(),
            passes=bool(p_pof > 0.05 and p_cc > 0.05)))
        for lab, mask in (("Calm", ~turb), ("Turbulent", turb)):
            regime_rows.append(dict(series=s, model=mname, regime=lab, days=int(mask.sum()),
                                    exceptions=int(exc[mask].sum()),
                                    exception_rate=exc[mask].mean() if mask.any() else np.nan))
    # --- next-day forecasts in dollars ---------------------------------------------------
    nd = V["next_day"][s]
    rw = r[-C.HS_WINDOW:]
    e_next = nd["EWMA"]
    z = rw / np.sqrt(ew_full[-C.HS_WINDOW:])
    pf = nd["garch_full"]
    wts = G["realtime"][s]
    todays = {
        "Historical sim": (ve.hs(rw, P99)[0], ve.hs(rw, P975)[1]),
        "Filtered HS": (ve.hs(z * np.sqrt(e_next), P99)[0], ve.hs(z * np.sqrt(e_next), P975)[1]),
        "GARCH-t": (ve.t_var_es(pf["mu"], np.sqrt(nd["GARCH-t"]), pf["nu"], P99)[0],
                    ve.t_var_es(pf["mu"], np.sqrt(nd["GARCH-t"]), pf["nu"], P975)[1]),
        "HAR-RV normal": (ve.normal_var_es(0, np.sqrt(nd["HAR-RV"]), P99)[0],
                          ve.normal_var_es(0, np.sqrt(nd["HAR-RV"]), P975)[1]),
        "Markov-switching": (ms.mixture_var_es(wts["next_weights"], wts["next_mu"], wts["next_sd"], P99)[0],
                             ms.mixture_var_es(wts["next_weights"], wts["next_mu"], wts["next_sd"], P975)[1]),
    }
    n = NOTIONAL[s]
    to_usd = (lambda pct: n * pct / 100) if s == "Book" else (lambda pct: n * (1 - np.exp(-pct / 100)))
    for mname, (v, e) in todays.items():
        today.append(dict(series=s, model=mname, as_of=R.index[-1].date(), notional=n, var99_pct=v, es975_pct=e,
                          var99_usd=to_usd(v), es975_usd=to_usd(e)))
    print(f"  {s:9s} exceptions (expected {len(x) * P99:.0f}): " +
          ", ".join(f"{m} {int((x < -out[m][0]).sum())}" for m in MODELS))

res = pd.DataFrame(results)
res.to_csv(TAB / "04_backtest_results.csv", index=False)
pd.DataFrame(today).to_csv(TAB / "04_var_es_today.csv", index=False)
reg = pd.DataFrame(regime_rows)
reg.to_csv(TAB / "04_exceptions_by_regime.csv", index=False)
bk = forecasts["Book"].copy()
bk.columns = [f"{a}|{b}" for a, b in bk.columns]
bk.round(5).to_csv(TAB / "04_book_daily_forecasts.csv")

# --- Figures ---------------------------------------------------------------------------------
COL = dict(zip(MODELS, style.SERIES))
for s, f in forecasts.items():
    fig, ax = plt.subplots(figsize=(9, 3.9))
    x = f[("actual", "x")]
    ax.plot(x.index, x, color=style.INK["axis"], lw=0.5, label="Daily return")
    for mname in ("Historical sim", "GARCH-t"):
        v = f[(mname, "var99")]
        ax.plot(v.index, -v, color=COL[mname], lw=0.9, label=f"−VaR 99%: {mname}")
        exc = x < -v
        ax.scatter(x.index[exc], x[exc], s=10, color=COL[mname], zorder=3)
    lo = np.nanpercentile(x, 0.05)
    ax.set_ylim(max(lo * 1.4, -60), np.nanpercentile(x, 99.95) * 1.2)
    ax.set_ylabel("% return")
    ax.set_title(f"{s}: daily returns vs VaR forecasts (dots = exceptions)")
    ax.legend(ncol=3, loc="lower left")
    style.save(fig, FIG / f"04_backtest_{s}.png")

fig, ax = plt.subplots(figsize=(9, 3.8))
series = list(R.columns)
for j, mname in enumerate(MODELS):
    sub = res[res.model == mname].set_index("series").loc[series]
    ax.scatter(sub.exception_rate * 100, np.arange(len(series)) + (j - 2) * 0.12, s=42, color=COL[mname],
               label=mname, zorder=3)
ax.axvline(P99 * 100, color=style.INK["secondary"], lw=1)
ax.text(P99 * 100, -0.75, " target 1%", color=style.INK["secondary"], fontsize=8.5, va="bottom")
ax.set_ylim(-0.8, len(series) - 0.5)
ax.set_yticks(range(len(series)), series)
ax.set_xlabel("Observed exception rate (%) for 99% VaR")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_title("How often did losses beat VaR?", pad=28)
ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handletextpad=0.2, columnspacing=1.0)
style.save(fig, FIG / "04_exception_rates.png")

fig, ax = plt.subplots(figsize=(9, 3.8))
f = forecasts["Book"]
x = f[("actual", "x")].values
for mname in MODELS:
    exc = x < -f[(mname, "var99")].values
    c = np.convolve(exc.astype(int), np.ones(250, int), mode="valid")
    ax.plot(f.index[249:], c, color=COL[mname], lw=1.0, label=mname)
ax.axhspan(-0.5, 4.5, color="#e8f5e8", lw=0)
ax.axhspan(4.5, 9.5, color="#fdf3dc", lw=0)
ax.axhspan(9.5, max(ax.get_ylim()[1], 12), color="#fbe4e4", lw=0)
ax.set_ylim(0, max(ax.get_ylim()[1], 12))
ax.set_ylabel("Exceptions in last 250 days")
ax.set_title("Energy book: Basel traffic light over time (green 0-4, yellow 5-9, red 10+)")
ax.legend(ncol=3, loc="upper left", fontsize=8)
style.save(fig, FIG / "04_traffic_light_book.png")

fig, ax = plt.subplots(figsize=(9, 3.6))
agg = reg.groupby(["model", "regime"])[["exceptions", "days"]].sum()
agg["rate"] = agg.exceptions / agg.days * 100
xs = np.arange(len(MODELS))
for i, (lab, c) in enumerate((("Calm", style.SERIES[0]), ("Turbulent", style.SERIES[1]))):
    ax.bar(xs + (i - 0.5) * 0.36, [agg.loc[(m, lab), "rate"] for m in MODELS], width=0.34, color=c, label=f"{lab} regime")
ax.axhline(1, color=style.INK["secondary"], lw=1)
ax.set_xticks(xs, MODELS)
ax.set_ylabel("Exception rate (%)")
ax.set_title("Exception rate by real-time regime (all series pooled; target 1%)")
ax.legend(loc="upper right")
style.save(fig, FIG / "04_exceptions_by_regime.png")

pd.to_pickle(forecasts, C.DATA_PROC / "var_forecasts.pkl")
print("  saved var_forecasts.pkl")
