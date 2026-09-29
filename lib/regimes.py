"""
regimes.py | Gaussian Markov-switching model (Hamilton, 1989), written from scratch.

Model: the return y_t is drawn from one of K regimes, s_t in {0..K-1}:

    y_t | s_t = k  ~  N(mu_k, sigma_k^2)
    Pr(s_t = j | s_{t-1} = i) = P[i, j]          (a Markov chain)

The regime is never observed. The Hamilton filter gives Pr(s_t = k | y_1..y_t) (what a
trader knows in real time); the Kim smoother gives Pr(s_t = k | y_1..y_T) (the best
after-the-fact reading). Parameters are estimated by EM (Baum-Welch). Regimes are
ordered by volatility, so state 0 is always the calmest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

VAR_FLOOR = 1e-6


@dataclass
class MSResult:
    mu: np.ndarray
    sigma: np.ndarray
    P: np.ndarray
    pi: np.ndarray
    loglik: float
    n_obs: int
    n_iter: int
    converged: bool
    filtered: np.ndarray = field(repr=False, default=None)   # Pr(s_t | y_1..t)
    predicted: np.ndarray = field(repr=False, default=None)  # Pr(s_t | y_1..t-1)
    smoothed: np.ndarray = field(repr=False, default=None)   # Pr(s_t | y_1..T)

    @property
    def k(self) -> int:
        return len(self.mu)

    @property
    def n_params(self) -> int:
        return self.k * (self.k - 1) + 2 * self.k  # transitions + means + variances

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.n_params * np.log(self.n_obs)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def durations(self) -> np.ndarray:
        """Expected length of a stay in each regime, 1 / (1 - p_kk), in periods."""
        return 1.0 / np.maximum(1 - np.diag(self.P), 1e-12)

    @property
    def stationary(self) -> np.ndarray:
        w, v = np.linalg.eig(self.P.T)
        s = np.real(v[:, np.argmin(np.abs(w - 1))])
        return s / s.sum()


def _densities(y, mu, sigma):
    z = (y[:, None] - mu[None, :]) / sigma[None, :]
    return np.exp(-0.5 * z * z) / (sigma[None, :] * np.sqrt(2 * np.pi))


def hamilton_filter(y, mu, sigma, P, pi):
    """Forward pass. Returns (filtered T x K, predicted T x K, log-likelihood)."""
    y = np.asarray(y, float)
    T, K = len(y), len(mu)
    eta = np.maximum(_densities(y, mu, sigma), 1e-300)
    filt = np.empty((T, K))
    pred = np.empty((T, K))
    ll = 0.0
    p = np.asarray(pi, float)
    PT = P.T
    for t in range(T):
        pred[t] = p
        joint = p * eta[t]
        c = joint.sum()
        ll += np.log(c)
        f = joint / c
        filt[t] = f
        p = PT @ f
    return filt, pred, ll


def kim_smoother(filt, pred, P):
    """Backward pass. Returns (smoothed T x K, expected transition counts K x K)."""
    T, K = filt.shape
    smooth = np.empty_like(filt)
    smooth[-1] = filt[-1]
    trans = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        ratio = smooth[t + 1] / np.maximum(pred[t + 1], 1e-300)
        smooth[t] = filt[t] * (P @ ratio)
        trans += filt[t][:, None] * P * ratio[None, :]
    smooth /= smooth.sum(axis=1, keepdims=True)
    return smooth, trans


def _init(y, K, rng):
    """Start from volatility quantiles: calm days -> low sigma, wild days -> high sigma."""
    vol = np.abs(y - y.mean())
    qs = np.quantile(vol, np.linspace(0, 1, K + 1))
    sig = np.array([max(np.std(y[(vol >= qs[k]) & (vol <= qs[k + 1])]) * 1.5, 1e-3) for k in range(K)])
    sig = np.sort(sig) * (1 + 0.05 * rng.standard_normal(K))
    mu = np.full(K, y.mean())
    P = np.full((K, K), 0.02 / max(K - 1, 1))
    np.fill_diagonal(P, 0.98)
    return mu, np.abs(sig), P, np.full(K, 1.0 / K)


def fit(y, K=2, max_iter=500, tol=1e-7, init=None, seed=0) -> MSResult:
    """EM estimation. `init` = a previous MSResult (warm start for rolling re-fits)."""
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    if init is not None and init.k == K:
        mu, sigma, P, pi = init.mu.copy(), init.sigma.copy(), init.P.copy(), init.stationary
    else:
        mu, sigma, P, pi = _init(y, K, rng)
    prev = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        filt, pred, ll = hamilton_filter(y, mu, sigma, P, pi)
        smooth, trans = kim_smoother(filt, pred, P)
        # M-step
        w = smooth.sum(axis=0)
        mu = (smooth * y[:, None]).sum(axis=0) / w
        var = (smooth * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / w
        sigma = np.sqrt(np.maximum(var, VAR_FLOOR))
        P = trans / trans.sum(axis=1, keepdims=True)
        pi = smooth[0]
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            converged = True
            break
        prev = ll
    order = np.argsort(sigma)                     # state 0 = calmest
    mu, sigma, P, pi = mu[order], sigma[order], P[np.ix_(order, order)], pi[order]
    filt, pred, ll = hamilton_filter(y, mu, sigma, P, pi)
    smooth, _ = kim_smoother(filt, pred, P)
    return MSResult(mu, sigma, P, pi, ll, len(y), it, converged, filt, pred, smooth)


def filter_fixed(y, res: MSResult):
    """Run the real-time filter with fixed parameters (no re-estimation). Returns filtered probs."""
    filt, pred, _ = hamilton_filter(np.asarray(y, float), res.mu, res.sigma, res.P, res.stationary)
    return filt, pred


def simulate(mu, sigma, P, T, seed=0):
    """Draw (y, states) from a Markov-switching model: used to verify the estimator."""
    rng = np.random.default_rng(seed)
    K = len(mu)
    s = np.empty(T, int)
    w, v = np.linalg.eig(np.asarray(P).T)
    st = np.real(v[:, np.argmin(np.abs(w - 1))])
    s[0] = rng.choice(K, p=st / st.sum())
    u = rng.random(T)
    cum = np.cumsum(P, axis=1)
    for t in range(1, T):
        s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t])), K - 1)
    y = np.asarray(mu)[s] + np.asarray(sigma)[s] * rng.standard_normal(T)
    return y, s


# ---------------------------------------------------------------- risk from a mixture
def mixture_var_es(w, mu, sigma, alpha):
    """
    VaR and ES (as positive losses) of a normal mixture sum_k w_k N(mu_k, sigma_k^2).
    alpha is the tail probability (0.01 for 99% VaR). ES uses the closed form
    E[X 1{X<q}] = sum_k w_k (mu_k Phi(z_k) - sigma_k phi(z_k)),  z_k = (q - mu_k) / sigma_k.
    """
    w, mu, sigma = map(np.asarray, (w, mu, sigma))
    f = lambda q: np.sum(w * norm.cdf((q - mu) / sigma)) - alpha
    lo, hi = (mu - 12 * sigma).min(), (mu + 12 * sigma).max()
    q = brentq(f, lo, hi)
    z = (q - mu) / sigma
    tail_mean = np.sum(w * (mu * norm.cdf(z) - sigma * norm.pdf(z))) / alpha
    return -q, -tail_mean


def mixture_variance(w, mu, sigma) -> float:
    m = np.sum(w * mu)
    return float(np.sum(w * (sigma ** 2 + mu ** 2)) - m ** 2)
