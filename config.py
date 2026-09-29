"""
config.py | Every assumption in one place. Change numbers here, never in the scripts.

Two workflows share this file:
  A. regime-risk/        daily energy benchmarks: regimes, volatility forecasts, VaR/ES backtests
  B. factor-validation/  monthly commodity panel: published factor strategies + overfitting checks
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"            # untouched downloads (git-ignored)
DATA_PROC = ROOT / "data" / "processed"     # hand-off files between scripts (git-ignored)
REPORTS = ROOT / "reports"                  # published PDFs
DOCS = ROOT / "docs"                        # static dashboard (GitHub Pages)
RISK_OUT = ROOT / "regime-risk" / "outputs"
FACT_OUT = ROOT / "factor-validation" / "outputs"
SEED = 7

# =============================================================================
# A. Regime & risk (daily)
# =============================================================================
# EIA spot benchmarks. Primary source: FRED; fallback: the public datahub.io mirror of
# the same EIA series on GitHub. Both are free and need no key.
GH = "https://raw.githubusercontent.com/datasets"
DAILY = {
    "WTI":      dict(fred="DCOILWTICO",   mirror=f"{GH}/oil-prices/main/data/wti-daily.csv",
                     label="WTI crude, Cushing", unit="$/bbl"),
    "Brent":    dict(fred="DCOILBRENTEU", mirror=f"{GH}/oil-prices/main/data/brent-daily.csv",
                     label="Brent crude, Europe", unit="$/bbl"),
    "HenryHub": dict(fred="DHHNGSP",      mirror=f"{GH}/natural-gas/main/data/daily.csv",
                     label="Henry Hub natural gas", unit="$/MMBtu"),
}
START_DATE = "2007-01-02"        # full daily coverage for all three series from here (earlier years are patchy)

# The "energy book": constant dollar positions, rebalanced daily. Long crude at Cushing,
# short Brent (a WTI-Brent spread position) and long US gas.
BOOK = {"WTI": 4_000_000, "Brent": -2_000_000, "HenryHub": 2_000_000}
UNIT_NOTIONAL = 1_000_000        # each single-asset "desk" is $1mm long

VAR_CONF = 0.99                  # 1-day VaR confidence
ES_CONF = 0.975                  # 1-day ES confidence (FRTB level)
HS_WINDOW = 250                  # historical-simulation look-back (trading days)
EWMA_LAMBDA = 0.94               # RiskMetrics decay
MIN_TRAIN = 750                  # observations (~3 years) before the first model fit
REFIT_EVERY = 21                 # re-estimate GARCH / HAR parameters every month (filters update daily)
MS_REFIT_EVERY = 126             # re-estimate the Markov-switching model every six months
BACKTEST_START = "2010-01-04"    # first out-of-sample forecast date
MS_STATES = [2, 3]               # regime counts compared by BIC; the first is used downstream
TRAFFIC_WINDOW = 250             # Basel traffic-light window
OUTLIER_SD = 6                   # flag (never delete) moves larger than this many rolling SDs

# =============================================================================
# B. Factor validation (monthly)
# =============================================================================
# World Bank "Pink Sheet" monthly prices. Official file first, then a public GitHub mirror.
PINK_URLS = [
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx",
    "https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx",
    "https://raw.githubusercontent.com/unbalancedparentheses/forex-centuries/main/data/sources/worldbank_commodities/wb_commodity_prices_monthly.xlsx",
]
# Pink Sheet column -> (short name, sector). Commodities with a liquid futures analogue.
UNIVERSE = {
    "Crude oil, Brent": ("Brent", "Energy"), "Crude oil, WTI": ("WTI", "Energy"),
    "Natural gas, US": ("US gas", "Energy"), "Coal, Australian": ("Coal", "Energy"),
    "Aluminum": ("Aluminum", "Base metals"), "Copper": ("Copper", "Base metals"),
    "Lead": ("Lead", "Base metals"), "Nickel": ("Nickel", "Base metals"),
    "Tin": ("Tin", "Base metals"), "Zinc": ("Zinc", "Base metals"),
    "Gold": ("Gold", "Precious"), "Silver": ("Silver", "Precious"), "Platinum": ("Platinum", "Precious"),
    "Maize": ("Corn", "Grains"), "Wheat, US SRW": ("Wheat SRW", "Grains"), "Wheat, US HRW": ("Wheat HRW", "Grains"),
    "Soybeans": ("Soybeans", "Grains"), "Soybean oil": ("Soy oil", "Grains"), "Soybean meal": ("Soy meal", "Grains"),
    "Rice, Thai 5%": ("Rice", "Grains"), "Palm oil": ("Palm oil", "Grains"),
    "Cocoa": ("Cocoa", "Softs"), "Coffee, Arabica": ("Coffee", "Softs"), "Sugar, world": ("Sugar", "Softs"),
    "Cotton, A Index": ("Cotton", "Softs"), "Rubber, RSS3": ("Rubber", "Softs"),
}
FACTOR_START = "1980-01"         # prices before this are often administered (flat for months)
MIN_ASSETS = 10                  # a month needs at least this many priced assets to trade
COST_BPS = 10                    # trading cost per unit of turnover (one-way), basis points
TSMOM_TARGET_VOL = 0.40          # per-asset annualised vol target (Moskowitz, Ooi & Pedersen)
VOL_LOOKBACK = 36                # months of returns for the ex-ante volatility estimate

# The three published strategies, with the parameters the papers use.
# The Pink Sheet reports monthly AVERAGE prices, which adds spurious lag-1 autocorrelation
# (Working, 1960). Every signal therefore skips the most recent month (skip = 1).
PUBLISHED = {
    "TSMOM": dict(name="Time-series momentum", paper="Moskowitz, Ooi & Pedersen (2012, JFE)",
                  sample_end="2009-12", published="2012-05", params=dict(lookback=12, skip=1, scaled=True)),
    "XSMOM": dict(name="Cross-sectional momentum", paper="Miffre & Rallis (2007, JBF)",
                  sample_end="2004-12", published="2007-06", params=dict(lookback=12, skip=1, q=1 / 3)),
    "VALUE": dict(name="Value (5-year reversal)", paper="Asness, Moskowitz & Pedersen (2013, JF)",
                  sample_end="2011-12", published="2013-06", params=dict(years=5, q=1 / 3)),
}
# The full grid of variants a researcher might have tried. Its size (N) feeds the
# deflated Sharpe ratio: the more you try, the higher the bar.
GRID = {
    "TSMOM": dict(lookback=[1, 3, 6, 9, 12], skip=[1, 2], scaled=[True, False]),
    "XSMOM": dict(lookback=[1, 3, 6, 9, 12], skip=[1, 2], q=[1 / 3, 1 / 4]),
    "VALUE": dict(years=[3, 4, 5], q=[1 / 3, 1 / 4]),
}
CPCV_GROUPS = 10                 # combinatorial purged CV: N groups ...
CPCV_TEST_GROUPS = 2             # ... k of them held out per split -> C(10,2) = 45 splits, 9 paths
PURGE = 1                        # months dropped before each test block (label overlap)
EMBARGO = 12                     # months dropped after each test block (look-back overlap)
PBO_BLOCKS = 16                  # CSCV blocks for the probability of backtest overfitting
WALK_FORWARD_MIN = 120           # months before the first walk-forward selection
