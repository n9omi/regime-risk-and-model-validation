"""
data.py | Download, cache and parse the two public datasets.

Daily:   EIA spot prices (WTI, Brent, Henry Hub) from FRED, falling back to a GitHub mirror.
Monthly: World Bank Pink Sheet commodity prices, falling back to a GitHub mirror.
Set FRED_API_KEY in your environment (or a .env file) to use the official FRED API.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import fetch


def _read_env_key() -> str | None:
    key = os.environ.get("FRED_API_KEY")
    env = Path(__file__).resolve().parents[1] / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("FRED_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    return key or None


def _parse_two_col_csv(path: Path) -> pd.Series:
    """Parse a date,value CSV (FRED CSV or the GitHub mirror). '.' and blanks become NaN."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0], errors="coerce"))
    s = s[s.index.notna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def _parse_fred_json(path: Path) -> pd.Series:
    import json

    obs = json.loads(path.read_text())["observations"]
    return pd.Series(pd.to_numeric([o["value"] for o in obs], errors="coerce"),
                     index=pd.to_datetime([o["date"] for o in obs])).sort_index()


def daily_series(name: str, spec: dict, raw_dir: Path) -> tuple[pd.Series, str]:
    """One EIA daily series. Returns (prices, source description)."""
    key = _read_env_key()
    fid = spec["fred"]
    if key:
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={fid}"
               f"&api_key={key}&file_type=json")
        try:
            p, src = fetch(url, raw_dir / f"{fid}.json")
            return _parse_fred_json(p), f"FRED API ({fid})"
        except Exception as e:
            print(f"  ! FRED API failed for {fid}: {str(e)[:80]}; trying public CSV")
    p, src = fetch([f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}", spec["mirror"]],
                   raw_dir / f"{fid}.csv")
    label = "FRED CSV" if "fred" in src else ("GitHub mirror of EIA" if "github" in src else src)
    return _parse_two_col_csv(p), f"{label} ({fid})"


def pink_sheet(urls, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """World Bank Pink Sheet 'Monthly Prices' sheet -> (prices wide DataFrame, units, source)."""
    p, src = fetch(urls, raw_dir / "CMO-Historical-Data-Monthly.xlsx", max_age_hours=24 * 7, min_bytes=100_000)
    raw = pd.read_excel(p, sheet_name="Monthly Prices", header=None)
    hdr = next(i for i in range(12) if str(raw.iloc[i, 1]).startswith("Crude oil"))
    names = [str(x).strip() for x in raw.iloc[hdr, 1:].tolist()]
    units = pd.Series([str(x) for x in raw.iloc[hdr + 1, 1:].tolist()], index=names)
    body = raw.iloc[hdr + 2:].copy()
    body.columns = ["date"] + names
    body = body[body["date"].astype(str).str.match(r"^\d{4}M\d{2}$")]
    idx = pd.PeriodIndex(body["date"].str.replace("M", "-"), freq="M")
    prices = body.drop(columns="date").apply(pd.to_numeric, errors="coerce")
    prices.index = idx
    updated = str(raw.iloc[3, 0]) if "Updated" in str(raw.iloc[3, 0]) else ""
    where = ("GitHub mirror of the World Bank file" if "githubusercontent" in src
             else ("World Bank" if "worldbank.org" in src else src))
    return prices, units, f"{where}; {updated}".strip("; ")


def clean_daily(prices: pd.DataFrame, outlier_sd: float = 6.0):
    """
    Data-quality rules (every action is logged, nothing is silently changed):
      * non-positive price -> missing (WTI spot printed below zero on 2020-04-20; a log return is undefined)
      * gaps of 1-2 days -> carry the last price forward; longer gaps -> drop the date
      * 3+ days of exactly 0% change -> flagged as stale
      * |return| > outlier_sd x rolling 1-year SD -> flagged, NOT deleted (crashes are what VaR is for)
    Returns (clean prices, log returns in %, fixes log, flags log).
    """
    fixes, flags = [], []
    px = prices.copy()
    for c in px.columns:
        bad = px[c] <= 0
        for d in px.index[bad]:
            fixes.append(dict(date=d.date(), series=c, issue="non-positive price", value=px.at[d, c],
                              action="set to missing"))
        px.loc[bad, c] = np.nan
    before = px.isna()
    px = px.ffill(limit=2)
    for c in px.columns:
        filled = before[c] & px[c].notna()
        if filled.any():
            fixes.append(dict(date=None, series=c, issue="short gap (1-2 days)", value=int(filled.sum()),
                              action="carried last price forward"))
    n0 = len(px)
    px = px.dropna()
    if len(px) < n0:
        fixes.append(dict(date=None, series="all", issue="dates with a missing series after filling",
                          value=n0 - len(px), action="dropped date"))
    per_year = px.groupby(px.index.year).size()
    for y, n in per_year.items():
        if n < 200 and y not in (px.index[0].year, px.index[-1].year):
            flags.append(dict(date=None, series="all", issue=f"only {n} common trading days in {y}",
                              value=n, action="flagged: returns span missing days"))
    ret = 100 * np.log(px).diff().dropna()
    for c in ret.columns:
        r = ret[c]
        sd = r.rolling(250, min_periods=60).std().shift(1)
        big = (r.abs() > outlier_sd * sd)
        for d in r.index[big]:
            flags.append(dict(date=d.date(), series=c, issue=f"move > {outlier_sd:g} rolling SD",
                              value=round(r[d], 2), action="flagged, kept"))
        zero = (r == 0).astype(int)
        runs = zero.groupby((zero != zero.shift()).cumsum()).transform("sum") * zero
        n_stale = int((runs >= 3).sum())
        if n_stale:
            flags.append(dict(date=None, series=c, issue="3+ days of 0% change", value=n_stale,
                              action="flagged, kept"))
    return px, ret, pd.DataFrame(fixes), pd.DataFrame(flags)
