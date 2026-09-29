"""
var_es.py | Value-at-Risk, Expected Shortfall and the tests used to backtest them.

Conventions: X_t is the day's return or P&L (losses negative). VaR and ES are reported
as POSITIVE loss amounts. p is the tail probability (1% for 99% VaR, 2.5% for 97.5% ES).
An "exception" is a day with X_t < -VaR_t.
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, norm, t as student_t


# ---------------------------------------------------------------- estimators
def hs(window, p):
    """Historical simulation: the empirical p-quantile of past returns, and the mean beyond it."""
    x = np.sort(np.asarray(window, float))
    n = len(x)
    k = max(int(np.floor(n * p)), 1)          # number of tail observations
    var = -x[k - 1]
    es = -x[:k].mean()
    return var, es


def normal_var_es(mu, sd, p):
    z = norm.ppf(p)
    return -(mu + sd * z), -mu + sd * norm.pdf(z) / p


def t_var_es(mu, sd, nu, p):
    """Student-t scaled to standard deviation `sd` (unit-variance t times sd)."""
    scale = sd * np.sqrt((nu - 2) / nu)
    q = student_t.ppf(p, nu)
    es_std = student_t.pdf(q, nu) / p * (nu + q ** 2) / (nu - 1)
    return -(mu + scale * q), -mu + scale * es_std


# ---------------------------------------------------------------- VaR backtests
def kupiec_pof(exc, p):
    """Proportion-of-failures LR test (Kupiec, 1995). H0: exception rate = p."""
    exc = np.asarray(exc, bool)
    T, x = len(exc), int(exc.sum())
    pi = x / T
    ll0 = xlogy(T - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(T - x, 1 - pi) + xlogy(x, pi)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, 1))


def christoffersen_ind(exc):
    """Independence LR test (Christoffersen, 1998). H0: an exception today does not change tomorrow's odds."""
    e = np.asarray(exc, int)
    a, b = e[:-1], e[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(1 - chi2.cdf(lr, 1)), dict(p01=p01, p11=p11)


def conditional_coverage(exc, p):
    lr_pof, _ = kupiec_pof(exc, p)
    lr_ind, _, _ = christoffersen_ind(exc)
    lr = lr_pof + lr_ind
    return float(lr), float(1 - chi2.cdf(lr, 2))


def traffic_light(n_exceptions_250: int) -> str:
    """Basel zone for 250 days of 99% VaR: green 0-4, yellow 5-9, red 10+."""
    return "green" if n_exceptions_250 <= 4 else ("yellow" if n_exceptions_250 <= 9 else "red")


def rolling_zones(exc, window=250):
    """Share of all 250-day windows that fall in each Basel zone."""
    c = np.convolve(np.asarray(exc, int), np.ones(window, int), mode="valid")
    return {z: float(np.mean([traffic_light(x) == z for x in c])) for z in ("green", "yellow", "red")}, c


# ---------------------------------------------------------------- ES backtests
def acerbi_szekely_z2(x, var_p, es_p, p):
    """
    Z2 = sum_t X_t 1{X_t < -VaR_t} / (T p ES_t) + 1   (Acerbi & Szekely, 2014)
    E[Z2] = 0 if ES is right; negative = losses beyond VaR are bigger than ES says.
    With T = 250 the authors' thresholds are -0.70 (yellow, 5%) and -1.8 (red, 0.01%).
    """
    x, var_p, es_p = map(lambda a: np.asarray(a, float), (x, var_p, es_p))
    ind = x < -var_p
    return float(np.sum(x * ind / (len(x) * p * es_p)) + 1)


def mcneil_frey(x, var_p, es_p, sd, n_boot=5000, seed=0):
    """
    Exceedance-residual test (McNeil & Frey, 2000). On days with X_t < -VaR_t, the residual
    r_t = (-X_t - ES_t) / sd_t should average zero. One-sided bootstrap p-value for
    H1: mean > 0 (ES too small). Returns (mean residual, p-value, number of exceedances).
    """
    x, var_p, es_p, sd = map(lambda a: np.asarray(a, float), (x, var_p, es_p, sd))
    ind = x < -var_p
    r = (-x[ind] - es_p[ind]) / sd[ind]
    if len(r) < 5:
        return float("nan"), float("nan"), int(len(r))
    rng = np.random.default_rng(seed)
    rc = r - r.mean()
    boots = rng.choice(rc, size=(n_boot, len(rc)), replace=True).mean(axis=1)
    pval = float(np.mean(boots >= r.mean()))
    return float(r.mean()), pval, int(len(r))
