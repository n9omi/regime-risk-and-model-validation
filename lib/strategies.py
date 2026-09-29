"""
strategies.py | Three published commodity factor strategies, as position weights.

Notation (monthly, t = month end at which positions are set; they earn month t+1's return):
    r_{i,t}          log return of commodity i in month t
    MOM_{i,t}(L, s)  sum of r_i over months t-s-L+1 .. t-s      (L-month look-back, skip the last s)
    sigma_{i,t}      annualised volatility of r_i over the last 36 months

    TSMOM   w_{i,t} = (1/N_t) * sign(MOM_{i,t}) * (0.40 / sigma_{i,t})      Moskowitz, Ooi & Pedersen (2012)
    XSMOM   long the top q of assets by MOM, short the bottom q, equal weights in each leg
                                                                          Miffre & Rallis (2007)
    VALUE   v_{i,t} = log( mean price 4.5-5.5 years ago / price at t-s ); long the cheapest q,
            short the dearest q                                           Asness, Moskowitz & Pedersen (2013)

Portfolio return in month t+1:  R_{t+1} = sum_i w_{i,t} * (exp(r_{i,t+1}) - 1)  -  c * sum_i |w_{i,t} - w_{i,t-1}|
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def momentum(ret: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    return ret.rolling(lookback, min_periods=lookback).sum().shift(skip)


def ann_vol(ret: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    return ret.rolling(lookback, min_periods=24).std() * np.sqrt(12)


def _tradable(sig: pd.DataFrame, min_assets: int) -> pd.DataFrame:
    ok = sig.notna().sum(axis=1) >= min_assets
    return sig.where(ok, np.nan, axis=0)


def w_tsmom(ret, lookback=12, skip=1, scaled=True, target=0.40, vol_lb=36, min_assets=10):
    sig = _tradable(np.sign(momentum(ret, lookback, skip)), min_assets)
    raw = sig * (target / ann_vol(ret, vol_lb)) if scaled else sig
    raw = raw.clip(-5, 5)                           # cap leverage on very quiet assets
    n = raw.notna().sum(axis=1).replace(0, np.nan)
    return raw.div(n, axis=0)


def _rank_long_short(score: pd.DataFrame, q: float, min_assets: int) -> pd.DataFrame:
    score = _tradable(score, min_assets)
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for t, row in score.iterrows():
        s = row.dropna()
        if len(s) < min_assets:
            w.loc[t] = np.nan
            continue
        k = max(int(np.floor(len(s) * q)), 1)
        order = s.sort_values()
        w.loc[t, order.index[-k:]] = 1.0 / k
        w.loc[t, order.index[:k]] = -1.0 / k
    return w


def w_xsmom(ret, lookback=12, skip=1, q=1 / 3, min_assets=10):
    return _rank_long_short(momentum(ret, lookback, skip), q, min_assets)


def w_value(prices, years=5, q=1 / 3, skip=1, min_assets=10):
    lp = np.log(prices)
    m = 12 * years
    past = lp.rolling(13, min_periods=10).mean().shift(m - 6)   # average of months t-m-6 .. t-m+6
    score = past - lp.shift(skip)                                # high = cheap relative to history
    return _rank_long_short(score, q, min_assets)


def portfolio_returns(w: pd.DataFrame, ret: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Gross and net monthly returns of weights w (set at t, earn t+1). Returns DataFrame [gross, net, turnover]."""
    simple = np.expm1(ret)
    wl = w.shift(1)                                              # weights held during month t
    gross = (wl * simple).sum(axis=1, min_count=1)
    turnover = (w.fillna(0) - w.fillna(0).shift(1)).abs().sum(axis=1).shift(1)
    net = gross - cost_bps / 1e4 * turnover
    out = pd.DataFrame(dict(gross=gross, net=net, turnover=turnover))
    return out[wl.notna().any(axis=1)]


def weights_for(family: str, params: dict, ret: pd.DataFrame, prices: pd.DataFrame, cfg) -> pd.DataFrame:
    if family == "TSMOM":
        return w_tsmom(ret, params["lookback"], params["skip"], params["scaled"],
                       cfg.TSMOM_TARGET_VOL, cfg.VOL_LOOKBACK, cfg.MIN_ASSETS)
    if family == "XSMOM":
        return w_xsmom(ret, params["lookback"], params["skip"], params["q"], cfg.MIN_ASSETS)
    if family == "VALUE":
        return w_value(prices, params["years"], params["q"], 1, cfg.MIN_ASSETS)
    raise ValueError(family)


def label(family: str, p: dict) -> str:
    if family == "TSMOM":
        return f"TSMOM {p['lookback']}-{p['skip']}{' vol-scaled' if p['scaled'] else ''}"
    if family == "XSMOM":
        return f"XSMOM {p['lookback']}-{p['skip']} q={'1/3' if abs(p['q'] - 1/3) < 1e-9 else '1/4'}"
    return f"VALUE {p['years']}y q={'1/3' if abs(p['q'] - 1/3) < 1e-9 else '1/4'}"


def grid(cfg_grid: dict):
    """Every (family, params) combination in the research grid."""
    for fam, g in cfg_grid.items():
        keys = list(g)
        for vals in itertools.product(*(g[k] for k in keys)):
            yield fam, dict(zip(keys, vals))


# ---------------------------------------------------------------- performance
def perf(r: pd.Series, periods: int = 12) -> dict:
    r = r.dropna()
    if len(r) < 12:
        return {}
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    from .validation import newey_west_t
    return dict(months=len(r), ann_return=r.mean() * periods, ann_vol=r.std() * np.sqrt(periods),
                sharpe=r.mean() / r.std() * np.sqrt(periods), nw_t=newey_west_t(r.values),
                max_drawdown=dd.min(), skew=r.skew(), hit_rate=(r > 0).mean())
