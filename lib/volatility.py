"""
volatility.py | One-day-ahead variance forecasts and how to judge them.

Notation (returns r_t in %, forecasts made with information up to t-1):
    EWMA      h_t = lam * h_{t-1} + (1 - lam) * r_{t-1}^2                   (RiskMetrics, 1996)
    GARCH-t   r_t = mu + e_t,  e_t = sqrt(h_t) z_t,  z_t ~ unit-variance Student-t(nu)
              h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}                 (Bollerslev, 1986/87)
    HAR-RV    RV_t = b0 + bd RV_{t-1} + bw RV^(5)_{t-1} + bm RV^(22)_{t-1}      (Corsi, 2009)
              RV^(n) = average of the last n daily RVs. With daily closes only, RV_t = r_t^2.
Losses (Patton, 2011: both rank forecasts correctly even with a noisy proxy like r^2):
    QLIKE     L = log h + r^2 / h
    MSE       L = (r^2 - h)^2
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln
from scipy.stats import norm


# ---------------------------------------------------------------- EWMA
def ewma(r, lam=0.94, init_n=30):
    """h_t for every t (forecast of day t's variance from data up to t-1)."""
    r = np.asarray(r, float)
    h = np.empty_like(r)
    h[0] = np.var(r[:init_n]) if len(r) >= init_n else np.var(r)
    # h_t = lam h_{t-1} + (1-lam) r_{t-1}^2  -> a first-order linear filter
    u = np.r_[h[0], (1 - lam) * r[:-1] ** 2]
    h[:] = lfilter([1.0], [1.0, -lam], u)
    return h


# ---------------------------------------------------------------- GARCH(1,1)-t
def garch_filter(r, mu, omega, alpha, beta, h0=None):
    """Conditional variances h_t (t = 0..T-1) and the next-day forecast h_T."""
    e = np.asarray(r, float) - mu
    h0 = np.var(e) if h0 is None else h0
    u = np.r_[h0, omega + alpha * e[:-1] ** 2]
    h = lfilter([1.0], [1.0, -beta], u)
    h_next = omega + alpha * e[-1] ** 2 + beta * h[-1]
    return h, h_next


def _t_nll(params, r):
    mu, omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9995 or nu <= 2.05:
        return 1e12
    h, _ = garch_filter(r, mu, omega, alpha, beta)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e12
    e2 = (r - mu) ** 2
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(e2 / ((nu - 2) * h)))
    return -np.sum(ll)


def garch_fit(r, start=None):
    """Maximum likelihood for GARCH(1,1) with Student-t errors. Returns dict of parameters."""
    r = np.asarray(r, float)
    v = np.var(r)
    x0 = start if start is not None else np.array([np.mean(r), 0.05 * v, 0.07, 0.90, 7.0])
    bounds = [(-abs(np.std(r)), abs(np.std(r))), (1e-8, 10 * v), (1e-6, 0.6), (0.2, 0.9995), (2.1, 80)]
    best = None
    for x in (x0, np.array([np.mean(r), 0.02 * v, 0.05, 0.93, 6.0])):
        res = minimize(_t_nll, x, args=(r,), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    mu, omega, alpha, beta, nu = best.x
    return dict(mu=mu, omega=omega, alpha=alpha, beta=beta, nu=nu, nll=best.fun,
                persistence=alpha + beta, uncond_var=omega / max(1 - alpha - beta, 1e-6), x=best.x)


def garch_simulate(mu, omega, alpha, beta, nu, T, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
    r = np.empty(T)
    h = omega / (1 - alpha - beta)
    for t in range(T):
        r[t] = mu + np.sqrt(h) * z[t]
        h = omega + alpha * (r[t] - mu) ** 2 + beta * h
    return r


# ---------------------------------------------------------------- HAR-RV
def har_features(rv):
    rv = np.asarray(rv, float)
    d = rv
    w = np.convolve(rv, np.ones(5) / 5, mode="full")[: len(rv)]
    m = np.convolve(rv, np.ones(22) / 22, mode="full")[: len(rv)]
    w[:4] = np.nan
    m[:21] = np.nan
    return np.column_stack([np.ones(len(rv)), d, w, m])


def har_fit(rv_train):
    """OLS of RV_{t+1} on [1, RV_t, RV^(5)_t, RV^(22)_t]. Returns coefficient vector."""
    X = har_features(rv_train)[:-1]
    y = np.asarray(rv_train, float)[1:]
    ok = np.all(np.isfinite(X), axis=1)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


# ---------------------------------------------------------------- forecast evaluation
def qlike(proxy, h):
    h = np.maximum(h, 1e-8)
    return np.log(h) + proxy / h


def mse(proxy, h):
    return (proxy - h) ** 2


def newey_west_lrv(x, lags=None):
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9))) if lags is None else lags
    lrv = x @ x / T
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        lrv += 2 * w * (x[l:] @ x[:-l]) / T
    return lrv


def diebold_mariano(loss_a, loss_b):
    """DM test of equal predictive accuracy. Negative stat = model A has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    stat = d.mean() / np.sqrt(newey_west_lrv(d) / len(d))
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def mincer_zarnowitz(proxy, h):
    """Regress proxy on forecast: unbiased forecasts have intercept 0 and slope 1."""
    X = np.column_stack([np.ones(len(h)), h])
    b, *_ = np.linalg.lstsq(X, proxy, rcond=None)
    resid = proxy - X @ b
    r2 = 1 - resid.var() / np.var(proxy)
    return float(b[0]), float(b[1]), float(r2)
