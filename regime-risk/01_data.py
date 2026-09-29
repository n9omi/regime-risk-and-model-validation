"""
01_data.py | Step 1: download the EIA daily benchmarks, clean them, build returns and P&L.

Inputs : FRED (or GitHub mirror) daily spot prices for WTI, Brent, Henry Hub
Outputs: outputs/tables/01_*.csv, outputs/figures/01_*.png, data/processed/daily.pkl
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
TAB, FIG = C.RISK_OUT / "tables", C.RISK_OUT / "figures"
for d in (TAB, FIG, C.DATA_PROC, C.DATA_RAW):
    d.mkdir(parents=True, exist_ok=True)

# --- 1. Download ----------------------------------------------------------------
print("Step 1 | downloading daily prices")
series, log = {}, []
for name, spec in C.DAILY.items():
    s, src = data.daily_series(name, spec, C.DATA_RAW)
    series[name] = s
    s_ok = s.dropna()
    log.append(dict(series=name, description=spec["label"], unit=spec["unit"], source=src,
                    first=s_ok.index.min().date(), last=s_ok.index.max().date(), rows=len(s_ok)))
    print(f"  {name:9s} {len(s_ok):6d} rows  {s_ok.index.min().date()} -> {s_ok.index.max().date()}  [{src}]")
pd.DataFrame(log).to_csv(TAB / "01_data_pull_log.csv", index=False)

# --- 2. Align and clean -----------------------------------------------------------
prices = pd.DataFrame(series).loc[C.START_DATE:]
prices = prices.dropna(how="all")
clean, rets, fixes, flags = data.clean_daily(prices, C.OUTLIER_SD)
fixes.to_csv(TAB / "01_data_quality_fixes.csv", index=False)
flags.to_csv(TAB / "01_flagged_moves.csv", index=False)
print(f"  {len(fixes)} fixes logged, {len(flags)} moves flagged (kept)")

# --- 3. P&L: single-asset desks ($1mm long) and the energy book --------------------
simple = np.expm1(rets / 100)                               # daily simple returns
book_pnl = sum(simple[k] * v for k, v in C.BOOK.items())   # $ P&L of constant-dollar positions
gross = sum(abs(v) for v in C.BOOK.values())
book_ret = 100 * book_pnl / gross                          # % of gross notional
series_ret = rets.copy()
series_ret["Book"] = book_ret                              # every model runs on % returns
notional = {k: C.UNIT_NOTIONAL for k in C.DAILY} | {"Book": gross}

# --- 4. Summary statistics -------------------------------------------------------
def stats(x):
    return dict(obs=len(x), ann_vol_pct=x.std() * np.sqrt(252), mean_daily_pct=x.mean(),
                skew=x.skew(), excess_kurtosis=x.kurt(), worst_day_pct=x.min(), best_day_pct=x.max(),
                worst_date=x.idxmin().date(), best_date=x.idxmax().date())

summ = pd.DataFrame({k: stats(series_ret[k]) for k in series_ret}).T
summ.index.name = "series"
summ.to_csv(TAB / "01_summary_stats.csv")
corr = rets.corr()
corr.to_csv(TAB / "01_correlations.csv")

# --- 5. Figures ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True)
for ax, (k, c) in zip(axes, zip(C.DAILY, style.SERIES)):
    ax.plot(clean.index, clean[k], color=c, lw=1.0)
    ax.set_ylabel(C.DAILY[k]["unit"])
    ax.set_title(C.DAILY[k]["label"], fontsize=10)
fig.suptitle("Daily spot prices (EIA)", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
style.save(fig, FIG / "01_prices.png")

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(series_ret.index, book_pnl.reindex(series_ret.index) / 1e3, color=style.SERIES[0], lw=0.6)
ax.set_title("Energy book: daily hypothetical P&L")
ax.set_ylabel("$ thousands")
style.save(fig, FIG / "01_book_pnl.png")

pd.to_pickle(dict(prices=clean, returns=series_ret, book_pnl=book_pnl, notional=notional,
                  log=pd.DataFrame(log)), C.DATA_PROC / "daily.pkl")
print(f"  saved {len(series_ret)} days of returns for {list(series_ret.columns)}")
