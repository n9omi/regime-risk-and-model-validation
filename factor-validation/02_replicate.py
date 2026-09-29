"""
02_replicate.py | Step 2: rebuild three published commodity factor strategies with their papers' settings.

For each strategy: full-sample performance (gross and net of costs), and the three windows
McLean & Pontiff (2016) use to measure decay: in the paper's sample, after the sample but
before publication, and after publication. Also: how each strategy did in calm vs
turbulent energy regimes (from the regime-risk workflow, when it has been run).
Outputs: tables/02_*.csv, figures/02_*.png, data/processed/published.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib import strategies as S, style  # noqa: E402

style.apply()
TAB, FIG = C.FACT_OUT / "tables", C.FACT_OUT / "figures"
D = pd.read_pickle(C.DATA_PROC / "monthly.pkl")
px, ret = D["prices"], D["returns"]
print("Step 2 | replicating published strategies")

rets, rows, sub = {}, [], []
for key, spec in C.PUBLISHED.items():
    w = S.weights_for(key, spec["params"], ret, px, C)
    pr = S.portfolio_returns(w, ret, C.COST_BPS)
    rets[key] = pr
    for kind in ("gross", "net"):
        rows.append(dict(strategy=key, name=spec["name"], paper=spec["paper"], returns=kind,
                         start=str(pr.index[0]), end=str(pr.index[-1]), avg_turnover=pr.turnover.mean(),
                         **S.perf(pr[kind])))
    for lab, a, b in (("In paper's sample", None, spec["sample_end"]),
                      ("After sample, before publication", spec["sample_end"], spec["published"]),
                      ("After publication", spec["published"], None)):
        r = pr["net"]
        idx = r.index
        m = np.ones(len(r), bool)
        if a:
            m &= idx > pd.Period(a, "M")
        if b:
            m &= idx <= pd.Period(b, "M")
        p = S.perf(r[m])
        if p:
            sub.append(dict(strategy=key, window=lab, start=str(idx[m][0]), end=str(idx[m][-1]), **p))
    print(f"  {key:6s} net Sharpe {S.perf(pr['net'])['sharpe']:.2f} ({pr.index[0]} -> {pr.index[-1]})")

# --- Placebo (permutation) test: shuffle each month's positions across commodities ---------------
# Keeps the same number of longs/shorts and the same turnover profile, destroys the signal.
# If the portfolio code had a structural bias (look-ahead, timing error), placebos would also win.
rng = np.random.default_rng(C.SEED)
placebo = []
simple = np.expm1(ret)
for key, spec in C.PUBLISHED.items():
    w = S.weights_for(key, spec["params"], ret, px, C)
    real = S.perf(rets[key]["gross"])["sharpe"]
    W = w.fillna(0).values
    held = ~w.isna().all(axis=1).values
    srs = []
    for _ in range(200):
        Wp = rng.permuted(W, axis=1)
        wp = pd.DataFrame(np.where(held[:, None], Wp, np.nan), index=w.index, columns=w.columns)
        pr = S.portfolio_returns(wp, ret, 0.0)["gross"]
        srs.append(pr.mean() / pr.std() * np.sqrt(12))
    srs = np.array(srs)
    placebo.append(dict(strategy=key, real_gross_sharpe=real, placebo_mean_sharpe=srs.mean(),
                        placebo_p05=np.percentile(srs, 5), placebo_p95=np.percentile(srs, 95),
                        p_value=(np.sum(srs >= real) + 1) / (len(srs) + 1)))
pd.DataFrame(placebo).to_csv(TAB / "02_placebo_test.csv", index=False)
print("  placebo: " + ", ".join(f"{p['strategy']} real {p['real_gross_sharpe']:.2f} vs placebo {p['placebo_mean_sharpe']:.2f}" for p in placebo))

# 50/50 value + momentum combination (Asness, Moskowitz & Pedersen 2013 highlight the negative correlation)
combo = pd.concat({k: rets[k]["net"] for k in ("XSMOM", "VALUE")}, axis=1).dropna()
rows.append(dict(strategy="COMBO", name="50/50 value + XS momentum", paper="Asness, Moskowitz & Pedersen (2013)",
                 returns="net", start=str(combo.index[0]), end=str(combo.index[-1]), avg_turnover=np.nan,
                 **S.perf(combo.mean(axis=1))))
perf = pd.DataFrame(rows)
perf.to_csv(TAB / "02_published_performance.csv", index=False)
subp = pd.DataFrame(sub)
subp.to_csv(TAB / "02_decay_windows.csv", index=False)
nets = pd.concat({k: v["net"] for k, v in rets.items()}, axis=1)
nets.corr().to_csv(TAB / "02_strategy_correlations.csv")
nets.to_csv(TAB / "02_strategy_monthly_returns.csv")

# --- Regime link: performance by energy regime (from the regime-risk workflow) --------------
regime_rows = []
rg = C.DATA_PROC / "regimes.pkl"
if rg.exists():
    G = pd.read_pickle(rg)
    sm = pd.Series(G["fits"]["WTI"].smoothed[:, -1], index=pd.read_pickle(C.DATA_PROC / "daily.pkl")["returns"].index)
    monthly_turb = sm.groupby(sm.index.to_period("M")).mean()
    lab = pd.Series(np.where(monthly_turb > 0.5, "Turbulent", "Calm"), index=monthly_turb.index)
    # regime known at the start of the month (previous month's reading)
    lab = lab.shift(1).dropna()
    for k in nets.columns:
        r = nets[k].dropna()
        r = r[r.index.isin(lab.index)]
        for g in ("Calm", "Turbulent"):
            rr = r[lab.loc[r.index] == g]
            if len(rr) >= 6:
                regime_rows.append(dict(strategy=k, wti_regime_last_month=g, months=len(rr),
                                        ann_return=rr.mean() * 12, ann_vol=rr.std() * np.sqrt(12),
                                        sharpe=rr.mean() / rr.std() * np.sqrt(12)))
    pd.DataFrame(regime_rows).to_csv(TAB / "02_performance_by_regime.csv", index=False)
    print("  linked to WTI regimes from the regime-risk workflow")

# --- Figures -------------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
for ax, (k, c) in zip(axes, zip(C.PUBLISHED, style.SERIES)):
    r = rets[k]["net"]
    cum = (1 + r).cumprod()
    ax.plot(r.index.to_timestamp(), cum.values, color=c, lw=1.1)
    ax.set_yscale("log")
    for dt, txt, ha in ((C.PUBLISHED[k]["sample_end"], "sample ends ", "right"),
                        (C.PUBLISHED[k]["published"], " published", "left")):
        x = pd.Period(dt, "M").to_timestamp()
        ax.axvline(x, color=style.INK["secondary"], lw=0.8)
        ax.text(x, 0.96, txt, fontsize=7.5, color=style.INK["secondary"], va="top", ha=ha,
                transform=ax.get_xaxis_transform())
    ax.set_title(f"{C.PUBLISHED[k]['name']} — {C.PUBLISHED[k]['paper']}", fontsize=9.5)
    ax.set_ylabel("Growth of $1 (log)")
fig.suptitle("Published strategies, net of costs", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
style.save(fig, FIG / "02_cumulative_returns.png")

fig, ax = plt.subplots(figsize=(9, 3.6))
wins = ["In paper's sample", "After sample, before publication", "After publication"]
piv = subp.pivot(index="strategy", columns="window", values="sharpe").reindex(columns=wins).loc[list(C.PUBLISHED)]
xs = np.arange(len(piv))
for i, (wn, c) in enumerate(zip(wins, (style.SERIES[0], style.SERIES[2], style.SERIES[1]))):
    ax.bar(xs + (i - 1) * 0.27, piv[wn], width=0.25, color=c, label=wn)
ax.axhline(0, color=style.INK["axis"], lw=0.8)
ax.set_xticks(xs, [C.PUBLISHED[k]["name"] for k in piv.index])
ax.set_ylabel("Annualised Sharpe (net)")
ax.set_title("Does the edge survive publication?")
ax.legend(loc="upper left", fontsize=8)
style.save(fig, FIG / "02_decay.png")

pd.to_pickle(dict(returns=rets, nets=nets), C.DATA_PROC / "published.pkl")
print("  saved published.pkl")
