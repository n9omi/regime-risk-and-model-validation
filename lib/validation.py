"""
validation.py | Is a backtest real, or the best of many lucky tries?

    Sharpe ratio (per period)      SR = mean(R) / sd(R)
    Probabilistic SR (Bailey & Lopez de Prado, 2012)
        PSR(SR*) = Phi( (SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )
        g3 = skewness, g4 = kurtosis (normal = 3). The probability the true SR beats SR*.
    Deflated SR (Bailey & Lopez de Prado, 2014)
        DSR = PSR(SR0),  SR0 = sd(SR_n) * [ (1-gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)) ]
        SR0 is the Sharpe you'd expect from the best of N worthless strategies; gamma = 0.5772.
    Probability of backtest overfitting, PBO (Bailey, Borwein, Lopez de Prado & Zhu, 2017)
        Split the T x N matrix of trial returns into S blocks. For every half/half split, pick
        the in-sample winner and find its rank out of sample. PBO = share of splits where the
        winner lands in the bottom half out of sample.
    Combinatorial purged cross-validation, CPCV (Lopez de Prado, 2018, ch. 7 & 12)
        N groups, k held out per split; drop `purge` observations before and `embargo` after
        each test block so look-back windows cannot leak test information into training.
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649


def sharpe(r, periods=1):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return r.mean() / r.std(ddof=1) * np.sqrt(periods) if len(r) > 2 and r.std() > 0 else np.nan


def newey_west_t(r, lags=None):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    T = len(r)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9))) if lags is None else lags
    x = r - r.mean()
    lrv = x @ x / T
    for l in range(1, lags + 1):
        lrv += 2 * (1 - l / (lags + 1)) * (x[l:] @ x[:-l]) / T
    return r.mean() / np.sqrt(lrv / T)


def psr(r, sr_star=0.0):
    """Probabilistic Sharpe ratio of per-period returns r against a per-period benchmark SR*."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    T, sr = len(r), sharpe(r)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def expected_max_sr(trial_srs):
    """SR0: expected maximum per-period Sharpe among N trials with zero true skill."""
    s = np.asarray(trial_srs, float)
    s = s[np.isfinite(s)]
    N = len(s)
    if N < 2:
        return 0.0
    return float(np.std(s, ddof=1) * ((1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * np.e))))


def dsr(r, trial_srs):
    """Deflated Sharpe ratio: PSR measured against the best-of-N luck benchmark SR0."""
    sr0 = expected_max_sr(trial_srs)
    return psr(r, sr0), sr0


# ---------------------------------------------------------------- CSCV / PBO
def pbo_cscv(M, S=16):
    """
    M: T x N matrix of trial returns (rows = time). Returns dict with PBO, logits,
    in-sample and out-of-sample Sharpe of the selected trial for every split.
    """
    M = np.asarray(M, float)
    T, N = M.shape
    T = (T // S) * S
    M = M[-T:]
    blocks = np.array_split(np.arange(T), S)
    s1 = np.array([M[b].sum(0) for b in blocks])          # S x N sums
    s2 = np.array([(M[b] ** 2).sum(0) for b in blocks])
    n = np.array([len(b) for b in blocks])[:, None]
    logits, is_sr, oos_sr = [], [], []
    for combo in itertools.combinations(range(S), S // 2):
        mask = np.zeros(S, bool)
        mask[list(combo)] = True
        def sr(m):
            cnt = n[m].sum()
            mu = s1[m].sum(0) / cnt
            var = s2[m].sum(0) / cnt - mu ** 2
            return mu / np.sqrt(np.maximum(var, 1e-18))
        is_, oos = sr(mask), sr(~mask)
        best = int(np.argmax(is_))
        rank = (oos < oos[best]).sum() + 1                  # 1 = worst ... N = best
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_sr.append(is_[best])
        oos_sr.append(oos[best])
    logits = np.array(logits)
    return dict(pbo=float(np.mean(logits <= 0)), logits=logits, is_sr=np.array(is_sr), oos_sr=np.array(oos_sr),
                n_splits=len(logits))


# ---------------------------------------------------------------- CPCV
def cpcv_splits(T, n_groups=10, k=2, purge=1, embargo=12):
    groups = np.array_split(np.arange(T), n_groups)
    out = []
    for test_g in itertools.combinations(range(n_groups), k):
        test = np.concatenate([groups[g] for g in test_g])
        drop = set(test.tolist())
        for g in test_g:
            a, b = groups[g][0], groups[g][-1]
            drop.update(range(max(a - purge, 0), a))                 # purge: labels overlapping the test block
            drop.update(range(b + 1, min(b + 1 + embargo, T)))        # embargo: look-backs reaching into it
        train = np.array([i for i in range(T) if i not in drop])
        out.append(dict(train=train, test_groups=test_g))
    return out, groups


def cpcv_paths(M, n_groups=10, k=2, purge=1, embargo=12, select=None):
    """
    For every split, choose the trial with the best training Sharpe and record its test returns.
    Stitch the test pieces into phi = C(n-1, k-1) complete out-of-sample paths.
    Returns (paths: phi x T array, chosen trial index per split, splits).
    """
    M = np.asarray(M, float)
    T, N = M.shape
    splits, groups = cpcv_splits(T, n_groups, k, purge, embargo)
    phi = comb(n_groups - 1, k - 1)
    paths = np.full((phi, T), np.nan)
    used = {g: 0 for g in range(n_groups)}
    chosen = []
    for sp in splits:
        tr = M[sp["train"]]
        srs = np.nanmean(tr, 0) / np.nanstd(tr, 0, ddof=1)
        j = int(np.nanargmax(srs)) if select is None else select(tr)
        chosen.append(j)
        for g in sp["test_groups"]:
            paths[used[g], groups[g]] = M[groups[g], j]
            used[g] += 1
    return paths, np.array(chosen), splits


def walk_forward(M, min_train=120, step=12):
    """Anchored walk-forward: every `step` months pick the best trial so far, hold it for `step` months."""
    M = np.asarray(M, float)
    T, N = M.shape
    out = np.full(T, np.nan)
    picks = []
    for t0 in range(min_train, T, step):
        tr = M[:t0]
        j = int(np.nanargmax(np.nanmean(tr, 0) / np.nanstd(tr, 0, ddof=1)))
        out[t0:t0 + step] = M[t0:t0 + step, j]
        picks.append((t0, j))
    return out, picks
