"""
01_data.py | Step 1: download the World Bank Pink Sheet and build the monthly commodity panel.

Inputs : CMO-Historical-Data-Monthly.xlsx (World Bank, or its GitHub mirror)
Outputs: tables/01_*.csv, figures/01_*.png, data/processed/monthly.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib import data, style  # noqa: E402

style.apply()
TAB, FIG = C.FACT_OUT / "tables", C.FACT_OUT / "figures"
for d in (TAB, FIG, C.DATA_PROC):
    d.mkdir(parents=True, exist_ok=True)

print("Step 1 | World Bank Pink Sheet")
allp, units, src = data.pink_sheet(C.PINK_URLS, C.DATA_RAW)
missing = [k for k in C.UNIVERSE if k not in allp.columns]
if missing:
    print(f"  ! not in this Pink Sheet release, skipped: {missing}")
cols = [k for k in C.UNIVERSE if k in allp.columns]
px = allp[cols].rename(columns={k: C.UNIVERSE[k][0] for k in cols})
px = px.loc[C.FACTOR_START:]
px = px.where(px > 0)
sectors = pd.Series({C.UNIVERSE[k][0]: C.UNIVERSE[k][1] for k in cols})
print(f"  {px.shape[1]} commodities, {px.index[0]} -> {px.index[-1]}  [{src}]")

ret = np.log(px).diff()
# Data-quality flags: stale prices (3+ months unchanged) and extreme monthly moves. Flagged, kept.
flags = []
for c in ret.columns:
    z = (ret[c] == 0).astype(int)
    run = z.groupby((z != z.shift()).cumsum()).transform("sum") * z
    if (run >= 3).any():
        flags.append(dict(commodity=c, issue="3+ months unchanged", months=int((run >= 3).sum())))
    big = ret[c].abs() > 0.40
    for d in ret.index[big]:
        flags.append(dict(commodity=c, issue=f"monthly move {ret.at[d, c]:+.0%}", months=str(d)))
pd.DataFrame(flags).to_csv(TAB / "01_data_quality_flags.csv", index=False)

summ = pd.DataFrame({
    "sector": sectors, "unit": [units.get(k, "") for k in cols], "first": px.apply(lambda s: str(s.first_valid_index())),
    "last": px.apply(lambda s: str(s.last_valid_index())), "months": px.notna().sum(),
    "ann_vol_pct": ret.std() * np.sqrt(12) * 100, "ann_mean_pct": ret.mean() * 12 * 100,
    "lag1_autocorr": ret.apply(lambda s: s.autocorr(1)), "lag2_autocorr": ret.apply(lambda s: s.autocorr(2)),
})
summ.index.name = "commodity"
summ.to_csv(TAB / "01_universe.csv")
pd.DataFrame([dict(source=src, file="CMO-Historical-Data-Monthly.xlsx", first=str(px.index[0]),
                   last=str(px.index[-1]), commodities=px.shape[1])]).to_csv(TAB / "01_data_pull_log.csv", index=False)
print(f"  median lag-1 autocorrelation {summ.lag1_autocorr.median():.2f} vs lag-2 {summ.lag2_autocorr.median():.2f}"
      " (monthly averaging inflates lag 1: every signal skips a month)")

fig, ax = plt.subplots(figsize=(9, 3.2))
n = px.notna().sum(axis=1)
ax.step(n.index.to_timestamp(), n.values, where="post", color=style.SERIES[0])
ax.axhline(C.MIN_ASSETS, color=style.INK["secondary"], lw=0.8)
ax.text(n.index[0].to_timestamp(), C.MIN_ASSETS + 0.4, f"minimum to trade ({C.MIN_ASSETS})", fontsize=8.5,
        color=style.INK["secondary"])
ax.set_ylim(0, n.max() + 2)
ax.set_title("Commodities with a price each month")
style.save(fig, FIG / "01_coverage.png")

fig, ax = plt.subplots(figsize=(9, 3.6))
o = summ.sort_values("lag1_autocorr")
ax.scatter(o.lag1_autocorr, range(len(o)), color=style.SERIES[1], s=26, label="Lag 1", zorder=3)
ax.scatter(o.lag2_autocorr, range(len(o)), color=style.SERIES[0], s=26, label="Lag 2", zorder=3)
ax.axvline(0, color=style.INK["axis"], lw=0.8)
ax.set_yticks(range(len(o)), o.index, fontsize=7.5)
ax.set_xlabel("Autocorrelation of monthly returns")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_title("Monthly averaging shows up as lag-1 autocorrelation")
ax.legend(loc="lower right")
fig.set_size_inches(9, 5.2)
style.save(fig, FIG / "01_autocorrelation.png")

pd.to_pickle(dict(prices=px, returns=ret, sectors=sectors, source=src), C.DATA_PROC / "monthly.pkl")
print("  saved monthly.pkl")
