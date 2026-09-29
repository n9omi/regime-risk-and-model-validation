"""
02_regimes.py | Step 2: which volatility regime is each market in?

Fits a Gaussian Markov-switching model (Hamilton, 1989) to daily returns:
  * full sample, K = 2 and 3 regimes, compared by BIC
  * simulation check: simulate from the fitted model, re-fit, confirm the estimator recovers it
  * real-time (out-of-sample) regime probabilities: the model is re-estimated every
    MS_REFIT_EVERY days on data up to that day and filtered forward with no look-ahead.
Outputs: tables/02_*.csv, figures/02_*.png, data/processed/regimes.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib import regimes as ms, style  # noqa: E402

style.apply()
TAB, FIG = C.RISK_OUT / "tables", C.RISK_OUT / "figures"
D = pd.read_pickle(C.DATA_PROC / "daily.pkl")
R, PX = D["returns"], D["prices"]
K = C.MS_STATES[0]
NAMES = {0: "Calm", 1: "Turbulent"} if K == 2 else {0: "Calm", 1: "Normal", 2: "Turbulent"}
print(f"Step 2 | Markov-switching regimes on {list(R.columns)}")

# --- 1. Full-sample fits and model selection ------------------------------------------
fits, bic_rows, par_rows = {}, [], []
for s in R.columns:
    y = R[s].values
    res = {k: ms.fit(y, k, seed=C.SEED) for k in C.MS_STATES}
    fits[s] = res[K]
    for k, r in res.items():
        bic_rows.append(dict(series=s, regimes=k, loglik=r.loglik, params=r.n_params, aic=r.aic, bic=r.bic,
                             iterations=r.n_iter, converged=r.converged))
    r = res[K]
    for j in range(K):
        par_rows.append(dict(series=s, regime=NAMES[j], mean_daily_pct=r.mu[j],
                             ann_vol_pct=r.sigma[j] * np.sqrt(252), stay_prob=r.P[j, j],
                             expected_duration_days=r.durations[j], long_run_share=r.stationary[j],
                             share_of_days=(r.smoothed.argmax(1) == j).mean()))
    print(f"  {s:9s} " + "  ".join(f"K={k}: BIC {r_.bic:,.0f}" for k, r_ in res.items()))
bic = pd.DataFrame(bic_rows)
bic["best_by_bic"] = bic.groupby("series")["bic"].transform("min") == bic["bic"]
bic.to_csv(TAB / "02_model_selection.csv", index=False)
params = pd.DataFrame(par_rows)
params.to_csv(TAB / "02_ms_parameters.csv", index=False)

# --- 2. Simulation-based verification (SR 11-7 'conceptual soundness') -----------------
rec = []
for s, r in fits.items():
    for rep in range(3):
        y_sim, st = ms.simulate(r.mu, r.sigma, r.P, len(R), seed=100 + rep)
        f = ms.fit(y_sim, K, seed=rep)
        acc = (f.smoothed.argmax(1) == st).mean()
        rec.append(dict(series=s, replication=rep + 1,
                        true_vol_calm=r.sigma[0] * np.sqrt(252), est_vol_calm=f.sigma[0] * np.sqrt(252),
                        true_vol_turb=r.sigma[-1] * np.sqrt(252), est_vol_turb=f.sigma[-1] * np.sqrt(252),
                        true_stay_turb=r.P[-1, -1], est_stay_turb=f.P[-1, -1], state_accuracy=acc))
rec = pd.DataFrame(rec)
rec["max_abs_vol_error_pct"] = np.maximum(abs(rec.est_vol_calm / rec.true_vol_calm - 1),
                                          abs(rec.est_vol_turb / rec.true_vol_turb - 1)) * 100
rec.to_csv(TAB / "02_simulation_recovery.csv", index=False)
print(f"  recovery: worst vol error {rec.max_abs_vol_error_pct.max():.1f}%, "
      f"worst state accuracy {rec.state_accuracy.min():.1%}")

# --- 3. Real-time regime probabilities (no look-ahead) -----------------------------------
start = R.index.get_loc(R.index[R.index >= C.BACKTEST_START][0])
realtime = {}
for s in R.columns:
    y = R[s].values
    pred = np.full((len(y), K), np.nan)      # Pr(s_t | y_1..t-1): the forecast for day t
    filt = np.full((len(y), K), np.nan)      # Pr(s_t | y_1..t)
    mu = np.full((len(y), K), np.nan)
    sd = np.full((len(y), K), np.nan)
    prev = None
    for t0 in range(start, len(y), C.MS_REFIT_EVERY):
        prev = ms.fit(y[:t0], K, init=prev, seed=C.SEED, max_iter=200)
        t1 = min(t0 + C.MS_REFIT_EVERY, len(y))
        f, p = ms.filter_fixed(y[:t1], prev)             # filter runs on data up to each day only
        pred[t0:t1], filt[t0:t1] = p[t0:t1], f[t0:t1]
        mu[t0:t1], sd[t0:t1] = prev.mu, prev.sigma
    # forecast for the day after the last observation
    last = ms.fit(y, K, init=prev, seed=C.SEED, max_iter=200)
    f_all, _ = ms.filter_fixed(y, last)
    realtime[s] = dict(pred=pd.DataFrame(pred, index=R.index), filt=pd.DataFrame(filt, index=R.index),
                       mu=pd.DataFrame(mu, index=R.index), sd=pd.DataFrame(sd, index=R.index),
                       next_weights=last.P.T @ f_all[-1], next_mu=last.mu, next_sd=last.sigma)
    print(f"  {s:9s} real-time filter done ({(len(y) - start) // C.MS_REFIT_EVERY + 1} re-fits)")

# --- 4. Today's regime and turbulent episodes --------------------------------------------
cur, epi = [], []
for s, r in fits.items():
    p_now = r.filtered[-1]
    state = int(p_now.argmax())
    run = 1
    lab = r.smoothed.argmax(1)
    while run < len(lab) and lab[-run - 1] == lab[-1]:
        run += 1
    cur.append(dict(series=s, date=R.index[-1].date(), regime=NAMES[state], probability=p_now[state],
                    p_turbulent=p_now[-1], days_in_regime=run,
                    regime_ann_vol_pct=r.sigma[state] * np.sqrt(252)))
    turb = pd.Series(r.smoothed[:, -1] > 0.5, index=R.index)
    grp = (turb != turb.shift()).cumsum()
    for _, g in turb[turb].groupby(grp[turb]):
        if len(g) >= 10:
            a, b = g.index[0], g.index[-1]
            col = s if s in PX else None
            chg = (PX.loc[b, col] / PX.loc[a, col] - 1) if col else np.nan
            epi.append(dict(series=s, start=a.date(), end=b.date(), trading_days=len(g),
                            price_change_pct=100 * chg, realised_ann_vol_pct=R.loc[a:b, s].std() * np.sqrt(252)))
pd.DataFrame(cur).to_csv(TAB / "02_current_regime.csv", index=False)
episodes = pd.DataFrame(epi).sort_values(["series", "start"])
episodes.to_csv(TAB / "02_turbulent_episodes.csv", index=False)

# --- 5. Figures -----------------------------------------------------------------------------
for i, s in enumerate(R.columns):
    r = fits[s]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    level = PX[s] if s in PX else (1 + R[s] / 100).cumprod() * 100
    a1.plot(level.index, level.values, color=style.SERIES[0] if s != "Book" else style.SERIES[6], lw=0.9)
    turb = r.smoothed[:, -1] > 0.5
    a1.fill_between(R.index, 0, 1, where=turb, transform=a1.get_xaxis_transform(), color=style.REGIME_SHADE,
                    lw=0, label="Turbulent regime (smoothed P > 0.5)")
    a1.set_title(f"{s}: {'price' if s in PX else 'cumulative book value (index)'} with turbulent regimes shaded")
    a1.legend(loc="upper left")
    a2.plot(R.index, r.smoothed[:, -1], color=style.SERIES[1], lw=0.8, label="Smoothed (hindsight)")
    rt = realtime[s]["filt"].iloc[:, -1]
    a2.plot(rt.index, rt.values, color=style.SERIES[0], lw=0.6, alpha=0.9, label="Real-time filtered")
    a2.set_ylim(-0.02, 1.02)
    a2.set_ylabel("P(turbulent)")
    a2.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    style.save(fig, FIG / f"02_regimes_{s}.png")

fig, ax = plt.subplots(figsize=(9, 3.4))
tp = params[params.regime == NAMES[K - 1]].set_index("series")
cp = params[params.regime == NAMES[0]].set_index("series")
yy = np.arange(len(tp))
ax.hlines(yy, cp.ann_vol_pct, tp.ann_vol_pct, color=style.INK["axis"], lw=2)
ax.scatter(cp.ann_vol_pct, yy, s=60, color=style.SERIES[0], zorder=3, label="Calm regime")
ax.scatter(tp.ann_vol_pct, yy, s=60, color=style.SERIES[1], zorder=3, label="Turbulent regime")
ax.set_yticks(yy, tp.index)
ax.set_xlabel("Annualised volatility (%)")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_title("Volatility in each regime")
ax.legend(loc="lower right")
style.save(fig, FIG / "02_regime_vols.png")

pd.to_pickle(dict(fits=fits, realtime=realtime, K=K, names=NAMES), C.DATA_PROC / "regimes.pkl")
print("  saved regimes.pkl")
