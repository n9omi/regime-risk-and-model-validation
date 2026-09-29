"""
03_volatility.py | Step 3: forecast tomorrow's variance four ways and score the forecasts.

Models (all strictly out of sample: day t's forecast uses returns up to t-1)
  EWMA      RiskMetrics, lambda = 0.94
  GARCH-t   GARCH(1,1) with Student-t errors; parameters re-estimated every REFIT_EVERY days
  HAR-RV    Corsi (2009) on the squared-return proxy; re-estimated every REFIT_EVERY days
  MS        mixture variance implied by the real-time Markov-switching probabilities (step 2)
Scoring: QLIKE and MSE against r_t^2 (Patton, 2011), Diebold-Mariano tests vs GARCH-t,
Mincer-Zarnowitz calibration regressions, plus a GARCH parameter-recovery check.
Outputs: tables/03_*.csv, figures/03_*.png, data/processed/vol_forecasts.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib import regimes as ms, style, volatility as vol  # noqa: E402

style.apply()
TAB, FIG = C.RISK_OUT / "tables", C.RISK_OUT / "figures"
D = pd.read_pickle(C.DATA_PROC / "daily.pkl")
G = pd.read_pickle(C.DATA_PROC / "regimes.pkl")
R = D["returns"]
MODELS = ["EWMA", "GARCH-t", "HAR-RV", "MS"]
print("Step 3 | volatility forecasts")

start = R.index.get_loc(R.index[R.index >= C.BACKTEST_START][0])
fc, garch_params, har_params, nu_path, mu_path, next_day = {}, [], [], {}, {}, {}
for s in R.columns:
    r = R[s].values
    T = len(r)
    h = pd.DataFrame(index=R.index, columns=MODELS, dtype=float)
    # EWMA: one causal pass
    h["EWMA"] = vol.ewma(r, C.EWMA_LAMBDA)
    # GARCH-t and HAR: re-fit monthly on an expanding window, filter daily with fixed params
    g_h, g_nu, g_mu = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    hr = np.full(T, np.nan)
    rv = r ** 2
    Xh = vol.har_features(rv)
    x0 = None
    for t0 in range(start, T, C.REFIT_EVERY):
        t1 = min(t0 + C.REFIT_EVERY, T)
        p = vol.garch_fit(r[:t0], start=x0)
        x0 = p["x"]
        hh, _ = vol.garch_filter(r[:t1], p["mu"], p["omega"], p["alpha"], p["beta"], h0=np.var(r[:t0]))
        g_h[t0:t1], g_nu[t0:t1], g_mu[t0:t1] = hh[t0:t1], p["nu"], p["mu"]
        b = vol.har_fit(rv[:t0])
        # forecast for day t uses features dated t-1
        hr[t0:t1] = Xh[t0 - 1:t1 - 1] @ b
        garch_params.append(dict(series=s, fit_date=R.index[t0 - 1].date(), **{k: p[k] for k in
                                 ("mu", "omega", "alpha", "beta", "nu", "persistence")}))
        har_params.append(dict(series=s, fit_date=R.index[t0 - 1].date(), b0=b[0], b_day=b[1], b_week=b[2], b_month=b[3]))
    floor = 0.05 * np.var(r[:start])
    h["GARCH-t"], h["HAR-RV"] = g_h, np.maximum(hr, floor)
    # Markov-switching mixture variance from the real-time predicted regime probabilities
    rt = G["realtime"][s]
    w, m, sd = rt["pred"].values, rt["mu"].values, rt["sd"].values
    msv = np.full(T, np.nan)
    ok = np.isfinite(w).all(1)
    msv[ok] = [ms.mixture_variance(w[i], m[i], sd[i]) for i in np.where(ok)[0]]
    h["MS"] = msv
    fc[s] = h.iloc[start:]
    nu_path[s] = pd.Series(g_nu, index=R.index).iloc[start:]
    mu_path[s] = pd.Series(g_mu, index=R.index).iloc[start:]
    # next-day (after last observation) forecasts, using all data
    pf = vol.garch_fit(r, start=x0)
    _, g_next = vol.garch_filter(r, pf["mu"], pf["omega"], pf["alpha"], pf["beta"])
    bf = vol.har_fit(rv)
    e_next = C.EWMA_LAMBDA * h["EWMA"].iloc[-1] + (1 - C.EWMA_LAMBDA) * r[-1] ** 2
    next_day[s] = dict(EWMA=e_next, **{"GARCH-t": g_next}, **{"HAR-RV": max(Xh[-1] @ bf, floor)},
                       MS=ms.mixture_variance(rt["next_weights"], rt["next_mu"], rt["next_sd"]),
                       garch_full=pf, har_full=bf)
    print(f"  {s:9s} {len(fc[s])} forecast days; full-sample GARCH-t persistence {pf['persistence']:.3f}, nu {pf['nu']:.1f}")

# --- Scoring ---------------------------------------------------------------------------
loss_rows, dm_rows, mz_rows = [], [], []
for s, h in fc.items():
    proxy = R[s].loc[h.index].values ** 2
    L = {m: vol.qlike(proxy, h[m].values) for m in MODELS}
    M = {m: vol.mse(proxy, h[m].values) for m in MODELS}
    for m in MODELS:
        a, b, r2 = vol.mincer_zarnowitz(proxy, h[m].values)
        loss_rows.append(dict(series=s, model=m, qlike=np.mean(L[m]), mse=np.mean(M[m]),
                              qlike_vs_garch=np.mean(L[m]) - np.mean(L["GARCH-t"])))
        mz_rows.append(dict(series=s, model=m, mz_intercept=a, mz_slope=b, mz_r2=r2))
        if m != "GARCH-t":
            st, p = vol.diebold_mariano(L[m], L["GARCH-t"])
            st2, p2 = vol.diebold_mariano(M[m], M["GARCH-t"])
            dm_rows.append(dict(series=s, model=m, benchmark="GARCH-t", dm_qlike=st, p_qlike=p, dm_mse=st2, p_mse=p2))
losses = pd.DataFrame(loss_rows)
losses["rank_qlike"] = losses.groupby("series")["qlike"].rank().astype(int)
losses.to_csv(TAB / "03_forecast_losses.csv", index=False)
pd.DataFrame(dm_rows).to_csv(TAB / "03_diebold_mariano.csv", index=False)
pd.DataFrame(mz_rows).to_csv(TAB / "03_mincer_zarnowitz.csv", index=False)
pd.DataFrame(garch_params).to_csv(TAB / "03_garch_params_path.csv", index=False)
pd.DataFrame(har_params).to_csv(TAB / "03_har_params_path.csv", index=False)
full = pd.DataFrame([dict(series=s, **{k: v for k, v in nd["garch_full"].items() if k not in ("x",)},
                          har_b0=nd["har_full"][0], har_b_day=nd["har_full"][1],
                          har_b_week=nd["har_full"][2], har_b_month=nd["har_full"][3]) for s, nd in next_day.items()])
full.to_csv(TAB / "03_full_sample_params.csv", index=False)

# --- GARCH parameter recovery (does the estimator find known parameters?) ------------------
rec = []
for s in R.columns:
    p = next_day[s]["garch_full"]
    for rep in range(3):
        sim = vol.garch_simulate(p["mu"], p["omega"], p["alpha"], p["beta"], p["nu"], len(R), seed=200 + rep)
        q = vol.garch_fit(sim)
        rec.append(dict(series=s, replication=rep + 1, true_alpha=p["alpha"], est_alpha=q["alpha"],
                        true_beta=p["beta"], est_beta=q["beta"], true_nu=p["nu"], est_nu=q["nu"],
                        true_persistence=p["persistence"], est_persistence=q["persistence"]))
rec = pd.DataFrame(rec)
rec.to_csv(TAB / "03_garch_recovery.csv", index=False)
print(f"  GARCH recovery: max |persistence error| {abs(rec.est_persistence - rec.true_persistence).max():.4f}")

# --- Figures ----------------------------------------------------------------------------------
for s, h in fc.items():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    realised = (R[s] ** 2).rolling(22).mean().shift(-21).loc[h.index].pipe(np.sqrt) * np.sqrt(252)
    ax.plot(realised.index, realised, color=style.INK["axis"], lw=1.6, label="Realised (next 22 days)")
    for m, c in zip(MODELS, style.SERIES):
        ax.plot(h.index, np.sqrt(h[m] * 252), color=c, lw=0.8, label=m)
    ax.set_yscale("log")
    ax.set_ylabel("Annualised vol (%), log scale")
    ax.set_title(f"{s}: one-day-ahead volatility forecasts")
    ax.legend(ncol=5, loc="upper left")
    style.save(fig, FIG / f"03_vol_forecasts_{s}.png")

fig, ax = plt.subplots(figsize=(9, 3.6))
piv = losses.pivot(index="series", columns="model", values="qlike_vs_garch")[MODELS]
xs = np.arange(len(piv))
wbar = 0.2
for i, (m, c) in enumerate(zip(MODELS, style.SERIES)):
    ax.bar(xs + (i - 1.5) * wbar, piv[m], width=wbar - 0.03, color=c, label=m)
ax.axhline(0, color=style.INK["axis"], lw=0.8)
ax.set_xticks(xs, piv.index)
ax.set_ylabel("Mean QLIKE minus GARCH-t")
ax.set_title("Forecast loss relative to GARCH-t (below zero = better)")
ax.legend(ncol=4, loc="upper left")
style.save(fig, FIG / "03_qlike_vs_garch.png")

pd.to_pickle(dict(forecasts=fc, nu=nu_path, mu=mu_path, next_day=next_day), C.DATA_PROC / "vol_forecasts.pkl")
print("  saved vol_forecasts.pkl")
