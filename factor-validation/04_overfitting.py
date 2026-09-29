"""
04_overfitting.py | Step 4: how likely is it that the backtest is a fluke?

  * Probability of backtest overfitting (PBO) by combinatorially symmetric CV (CSCV), for the
    whole grid and within each strategy family
  * Probabilistic and deflated Sharpe ratios (PSR, DSR) for the in-sample winners and for
    the published parameter choices, charging each for the number of variants tried
  * Newey-West t-statistics against the Harvey-Liu-Zhu (2016) t > 3 hurdle
Outputs: tables/04_*.csv, figures/04_*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib import strategies as S, style, validation as V  # noqa: E402

style.apply()
TAB, FIG = C.FACT_OUT / "tables", C.FACT_OUT / "figures"
G = pd.read_pickle(C.DATA_PROC / "grid.pkl")
PUB = pd.read_pickle(C.DATA_PROC / "published.pkl")
M, fams = G["M"], np.array(G["families"])
print("Step 4 | overfitting diagnostics")

sets = {"All variants": np.arange(M.shape[1])} | {f: np.where(fams == f)[0] for f in C.GRID}
pbo_rows, pbo_res = [], {}
for k, ix in sets.items():
    r = V.pbo_cscv(M.iloc[:, ix].values, C.PBO_BLOCKS)
    pbo_res[k] = r
    slope = np.polyfit(r["is_sr"], r["oos_sr"], 1)[0]
    pbo_rows.append(dict(variant_set=k, n_variants=len(ix), splits=r["n_splits"], pbo=r["pbo"],
                         median_logit=np.median(r["logits"]), is_to_oos_slope=slope,
                         share_oos_sharpe_negative=np.mean(r["oos_sr"] < 0)))
    print(f"  {k:13s} PBO {r['pbo']:.1%} over {r['n_splits']:,} CSCV splits")
pbo = pd.DataFrame(pbo_rows)
pbo.to_csv(TAB / "04_pbo.csv", index=False)

# --- PSR / DSR ------------------------------------------------------------------------------------------
trial_sr = M.apply(lambda s: V.sharpe(s))            # per-month Sharpe of every variant
rows = []
for k, ix in sets.items():
    sub = M.iloc[:, ix]
    best = trial_sr.iloc[ix].idxmax()
    d, sr0 = V.dsr(sub[best].values, trial_sr.iloc[ix].values)
    rows.append(dict(candidate=f"Best of {k}", variant=best, n_trials=len(ix), months=len(sub),
                     sharpe_ann=V.sharpe(sub[best], 12), sr0_ann=sr0 * np.sqrt(12), psr_vs_0=V.psr(sub[best].values),
                     dsr=d, nw_t=V.newey_west_t(sub[best].values)))
for key, spec in C.PUBLISHED.items():
    # A paper's authors choose among variants of THEIR strategy, so each published strategy is
    # deflated by its own family's trials. (Deflating by the whole grid would mix families with
    # genuinely different Sharpe ratios into the "luck" dispersion and set the bar too high.)
    r = PUB["returns"][key]["net"].dropna().values
    fam_ix = sets[key]
    d, sr0 = V.dsr(r, trial_sr.iloc[fam_ix].values)
    rows.append(dict(candidate=f"Published {key}", variant=S.label(key, spec["params"]), n_trials=len(fam_ix),
                     months=len(r), sharpe_ann=V.sharpe(r, 12), sr0_ann=sr0 * np.sqrt(12), psr_vs_0=V.psr(r),
                     dsr=d, nw_t=V.newey_west_t(r)))
ds = pd.DataFrame(rows)
ds["passes_dsr_95"] = ds.dsr > 0.95
ds["passes_t3"] = ds.nw_t > 3
ds.to_csv(TAB / "04_deflated_sharpe.csv", index=False)
for _, r in ds.iterrows():
    print(f"  {r.candidate:24s} SR {r.sharpe_ann:5.2f}  SR0 {r.sr0_ann:4.2f}  DSR {r.dsr:.3f}  t {r.nw_t:5.2f}")

# --- Figures -----------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(sets), figsize=(9, 2.9), sharey=True)
for ax, (k, r) in zip(axes, pbo_res.items()):
    ax.hist(r["logits"], bins=25, color=style.SERIES[0], edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=style.SERIES[1], lw=1.2)
    ax.set_title(f"{k}\nPBO = {r['pbo']:.0%}", fontsize=9)
    ax.set_xlabel("logit rank OOS")
axes[0].set_ylabel("CSCV splits")
fig.suptitle("Where does the in-sample winner rank out of sample? (left of red = bottom half)", x=0.01, ha="left",
             fontweight="bold", fontsize=10.5)
fig.tight_layout()
style.save(fig, FIG / "04_pbo_logits.png")

fig, ax = plt.subplots(figsize=(9, 3.8))
for (k, r), c in zip(pbo_res.items(), style.SERIES):
    if k == "All variants":
        continue
    ax.scatter(r["is_sr"] * np.sqrt(12), r["oos_sr"] * np.sqrt(12), s=6, alpha=0.25, color=c, label=k)
lim = ax.get_xlim()
ax.plot(lim, lim, color=style.INK["axis"], lw=0.8)
ax.axhline(0, color=style.INK["axis"], lw=0.8)
ax.set_xlabel("In-sample Sharpe of the chosen variant")
ax.set_ylabel("Its out-of-sample Sharpe")
ax.set_title("Performance degradation: in-sample vs out-of-sample Sharpe across CSCV splits")
ax.legend(loc="upper left", markerscale=3)
style.save(fig, FIG / "04_degradation.png")

fig, ax = plt.subplots(figsize=(9, 3.5))
y = np.arange(len(ds))
ax.barh(y, ds.dsr, color=[style.STATUS["good"] if p else style.STATUS["critical"] for p in ds.passes_dsr_95], height=0.6)
for yi, (v, p) in enumerate(zip(ds.dsr, ds.passes_dsr_95)):
    ax.text(min(v, 0.80) + 0.01 if v < 0.8 else 0.01, yi, ("survives" if p else "not proven") + f"  {v:.3f}",
            va="center", fontsize=8, color="white" if v >= 0.8 else style.INK["primary"])
ax.axvline(0.95, color=style.INK["secondary"], lw=1)
ax.text(0.95, -0.9, " 95%", color=style.INK["secondary"], fontsize=8.5)
ax.set_yticks(y, ds.candidate)
ax.invert_yaxis()
ax.set_xlim(0, 1.02)
ax.set_xlabel("Deflated Sharpe ratio (probability the true Sharpe beats the best-of-N luck benchmark)")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_title("Deflated Sharpe ratios")
style.save(fig, FIG / "04_deflated_sharpe.png")
print("  done")
