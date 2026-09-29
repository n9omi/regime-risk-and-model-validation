"""
03_purged_cv.py | Step 3: pick parameters the honest way, with purged cross-validation.

A researcher who tries 46 variants and reports the best one has not found a 46-to-1
winner; they have found the luckiest draw. This step builds every variant in the grid,
then asks what you would have earned OUT OF SAMPLE if you had chosen the variant with
the best training Sharpe:
  * combinatorial purged CV (CPCV): 10 groups, 2 held out per split -> 45 splits, 9 full paths
  * the same without purging/embargo, to show what leakage does
  * anchored walk-forward: re-choose every 12 months using only the past
Outputs: tables/03_*.csv, figures/03_*.png, data/processed/grid.pkl
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
D = pd.read_pickle(C.DATA_PROC / "monthly.pkl")
px, ret = D["prices"], D["returns"]
print("Step 3 | research grid + purged cross-validation")

cols, labels, fams = {}, [], []
for fam, p in S.grid(C.GRID):
    lab = S.label(fam, p)
    cols[lab] = S.portfolio_returns(S.weights_for(fam, p, ret, px, C), ret, C.COST_BPS)["net"]
    labels.append(lab)
    fams.append(fam)
M = pd.DataFrame(cols).dropna()          # common sample where every variant trades
T, N = M.shape
print(f"  {N} variants, {T} common months ({M.index[0]} -> {M.index[-1]})")
full_sr = M.apply(lambda r: V.sharpe(r, 12))
best_is = full_sr.idxmax()
grid_tab = pd.DataFrame(dict(variant=labels, family=fams, sharpe_full_sample=full_sr.values,
                             ann_return=(M.mean() * 12).values, ann_vol=(M.std() * np.sqrt(12)).values))
grid_tab.to_csv(TAB / "03_grid_performance.csv", index=False)

# --- CPCV with and without purging, for the whole grid and for each family -----------------------
res, path_srs = [], {}
fam_arr = np.array(fams)
sets = {"All variants": np.arange(N)} | {f: np.where(fam_arr == f)[0] for f in C.GRID}
for set_name, ix in sets.items():
    Ms = M.iloc[:, ix]
    insample = Ms.apply(lambda r: V.sharpe(r, 12))
    for name, pg, emb in (("Purged CPCV", C.PURGE, C.EMBARGO), ("Naive CV", 0, 0)):
        P, ch, _ = V.cpcv_paths(Ms.values, C.CPCV_GROUPS, C.CPCV_TEST_GROUPS, pg, emb)
        srs = np.array([V.sharpe(p, 12) for p in P])
        res.append(dict(variant_set=set_name, n_variants=len(ix), method=name, mean_oos_sharpe=srs.mean(),
                        min_oos_sharpe=srs.min(), max_oos_sharpe=srs.max(), in_sample_best_sharpe=insample.max(),
                        in_sample_best=insample.idxmax(), most_chosen=Ms.columns[np.bincount(ch).argmax()],
                        distinct_choices=len(set(ch))))
        if name == "Purged CPCV":
            path_srs[set_name] = srs
    wf, picks = V.walk_forward(Ms.values, C.WALK_FORWARD_MIN, 12)
    wf_s = pd.Series(wf, index=M.index).dropna()
    res.append(dict(variant_set=set_name, n_variants=len(ix), method="Walk-forward", mean_oos_sharpe=V.sharpe(wf_s, 12),
                    min_oos_sharpe=np.nan, max_oos_sharpe=np.nan, in_sample_best_sharpe=insample.max(),
                    in_sample_best=insample.idxmax(), most_chosen=Ms.columns[pd.Series([j for _, j in picks]).mode()[0]],
                    distinct_choices=len(set(j for _, j in picks))))
    if set_name == "All variants":
        wf_r, picks_all = wf_s, picks
cv = pd.DataFrame(res)
cv["oos_shortfall"] = cv.in_sample_best_sharpe - cv.mean_oos_sharpe
cv.to_csv(TAB / "03_cv_results.csv", index=False)
pd.DataFrame([dict(variant_set=k, path=i + 1, oos_sharpe=v) for k, a in path_srs.items() for i, v in enumerate(a)]
             ).to_csv(TAB / "03_cpcv_path_sharpes.csv", index=False)
pd.DataFrame([dict(rebalance=str(M.index[t0]), chosen=M.columns[j]) for t0, j in picks_all]).to_csv(TAB / "03_walk_forward_picks.csv", index=False)
purged_srs = path_srs["All variants"]
for _, r in cv[cv.method != "Naive CV"].iterrows():
    print(f"  {r.variant_set:13s} {r.method:12s} in-sample best {r.in_sample_best_sharpe:5.2f} -> out of sample {r.mean_oos_sharpe:5.2f}")

# --- Figures ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
fam_col = {"TSMOM": style.SERIES[0], "XSMOM": style.SERIES[1], "VALUE": style.SERIES[2]}
o = grid_tab.sort_values("sharpe_full_sample")
ax.barh(range(N), o.sharpe_full_sample, color=[fam_col[f] for f in o.family], height=0.72)
ax.axvline(0, color=style.INK["axis"], lw=0.8)
ax.set_yticks([])
ax.set_xlabel("Full-sample net Sharpe")
ax.set_title(f"All {N} variants in the research grid")
for f, c in fam_col.items():
    ax.barh([], [], color=c, label=f)
ax.legend(loc="lower right")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
style.save(fig, FIG / "03_grid_sharpes.png")

fig, ax = plt.subplots(figsize=(9, 3.6))
names = list(sets)
for i, k in enumerate(names):
    row = cv[(cv.variant_set == k) & (cv.method == "Purged CPCV")].iloc[0]
    ax.scatter(path_srs[k], np.full(len(path_srs[k]), i), s=34, color=style.SERIES[0], alpha=0.75, zorder=3,
               label="CPCV path Sharpe (out of sample)" if i == 0 else None)
    ax.scatter([row.in_sample_best_sharpe], [i], s=90, marker="D", color=style.SERIES[1], zorder=4,
               label="Best in-sample Sharpe" if i == 0 else None)
    wfv = cv[(cv.variant_set == k) & (cv.method == "Walk-forward")].mean_oos_sharpe.iloc[0]
    ax.scatter([wfv], [i], s=70, marker="|", color=style.SERIES[2], zorder=4, linewidths=2.5,
               label="Walk-forward Sharpe" if i == 0 else None)
ax.axvline(0, color=style.INK["axis"], lw=0.8)
ax.set_yticks(range(len(names)), [f"{k} ({len(sets[k])})" for k in names])
ax.set_xlabel("Annualised net Sharpe")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_title("Choosing the best variant: in-sample promise vs out-of-sample delivery", pad=26)
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
style.save(fig, FIG / "03_cpcv_vs_insample.png")

pd.to_pickle(dict(M=M, full_sr=full_sr, families=fams, wf=wf_r, path_srs=path_srs), C.DATA_PROC / "grid.pkl")
print("  saved grid.pkl")
