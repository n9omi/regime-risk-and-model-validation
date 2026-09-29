"""
test_models.py | Does every estimator and test do what it claims? Run:  python -m pytest -q

These are the unit-level checks a model validator would ask for: known answers,
simulation-based parameter recovery, and no-look-ahead guarantees.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import regimes as ms, strategies as S, validation as V, var_es as ve, volatility as vol  # noqa: E402


# ---------------------------------------------------------------- regimes
def test_markov_switching_recovers_known_parameters():
    P = np.array([[0.98, 0.02], [0.05, 0.95]])
    y, s = ms.simulate([0.05, -0.2], [1.0, 3.0], P, 4000, seed=3)
    fit = ms.fit(y, 2, seed=0)
    assert abs(fit.sigma[0] - 1.0) < 0.1 and abs(fit.sigma[1] - 3.0) < 0.3
    assert abs(fit.P[0, 0] - 0.98) < 0.02
    assert (fit.smoothed.argmax(1) == s).mean() > 0.9


def test_filter_probabilities_are_valid():
    y, _ = ms.simulate([0, 0], [1, 2], np.array([[0.97, 0.03], [0.1, 0.9]]), 500, seed=1)
    fit = ms.fit(y, 2)
    for arr in (fit.filtered, fit.predicted, fit.smoothed):
        assert np.allclose(arr.sum(1), 1) and (arr >= 0).all()


def test_mixture_with_one_component_equals_normal():
    v, e = ms.mixture_var_es([1.0], np.array([0.0]), np.array([2.0]), 0.01)
    nv, ne = ve.normal_var_es(0.0, 2.0, 0.01)
    assert v == pytest.approx(nv, rel=1e-6) and e == pytest.approx(ne, rel=1e-6)


# ---------------------------------------------------------------- volatility
def test_garch_recovers_known_parameters():
    r = vol.garch_simulate(0.0, 0.05, 0.08, 0.90, 6.0, 5000, seed=4)
    p = vol.garch_fit(r)
    assert abs(p["persistence"] - 0.98) < 0.02
    assert 4 < p["nu"] < 10


def test_ewma_matches_loop():
    r = np.random.default_rng(0).standard_normal(300)
    h = vol.ewma(r, 0.94)
    ref = [np.var(r[:30])]
    for t in range(1, 300):
        ref.append(0.94 * ref[-1] + 0.06 * r[t - 1] ** 2)
    assert np.allclose(h, ref)


def test_diebold_mariano_detects_a_better_forecast():
    rng = np.random.default_rng(1)
    h = np.exp(rng.standard_normal(3000) * 0.5)
    r2 = h * rng.standard_normal(3000) ** 2
    stat, p = vol.diebold_mariano(vol.qlike(r2, h), vol.qlike(r2, h * 1.8))
    assert stat < 0 and p < 0.01


# ---------------------------------------------------------------- VaR / ES
def test_t_converges_to_normal():
    v, e = ve.t_var_es(0, 1, 500, 0.01)
    assert v == pytest.approx(norm.ppf(0.99), rel=0.01)


def test_kupiec_and_christoffersen():
    exc = np.zeros(1000, bool)
    exc[::100] = True                        # exactly 1%, evenly spaced
    assert ve.kupiec_pof(exc, 0.01)[1] > 0.99
    assert ve.christoffersen_ind(exc)[1] > 0.3
    clustered = np.zeros(1000, bool)
    clustered[500:510] = True                # same count, all in a row
    assert ve.christoffersen_ind(clustered)[1] < 0.001


def test_traffic_light_zones():
    assert [ve.traffic_light(n) for n in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]


def test_z2_is_near_zero_for_a_correct_model():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200_000)
    v, e = ve.normal_var_es(0, 1, 0.025)
    assert abs(ve.acerbi_szekely_z2(x, np.full_like(x, v), np.full_like(x, e), 0.025)) < 0.05


# ---------------------------------------------------------------- validation
def test_cpcv_splits_purge_and_embargo():
    splits, groups = V.cpcv_splits(200, n_groups=10, k=2, purge=1, embargo=12)
    assert len(splits) == 45
    for sp in splits:
        test = np.concatenate([groups[g] for g in sp["test_groups"]])
        assert not set(sp["train"]) & set(test)
        for g in sp["test_groups"]:
            end = groups[g][-1]
            assert not set(range(end + 1, min(end + 13, 200))) & set(sp["train"])


def test_cpcv_builds_nine_complete_paths():
    M = np.random.default_rng(0).standard_normal((300, 5)) * 0.01
    paths, chosen, _ = V.cpcv_paths(M, 10, 2, 0, 0)
    assert paths.shape == (9, 300) and np.isfinite(paths).all()


def test_pbo_near_half_for_pure_noise_and_low_for_real_skill():
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((480, 20)) * 0.02
    # With no skill the in-sample winner is a coin flip or worse out of sample (the halves are
    # complementary, so a lucky first half implies a weaker second half): PBO >= ~0.5.
    assert V.pbo_cscv(noise, 16)["pbo"] > 0.4
    skill = noise.copy()
    skill[:, 0] += 0.02
    assert V.pbo_cscv(skill, 16)["pbo"] < 0.05


def test_deflated_sharpe_is_stricter_than_psr():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(300) * 0.02 + 0.004
    trials = rng.standard_normal(50) * 0.1
    d, sr0 = V.dsr(r, trials)
    assert sr0 > 0 and d < V.psr(r)


# ---------------------------------------------------------------- strategies
def test_no_look_ahead_in_strategy_weights():
    rng = np.random.default_rng(7)
    idx = pd.period_range("1990-01", periods=200, freq="M")
    ret = pd.DataFrame(rng.standard_normal((200, 12)) * 0.05, index=idx, columns=list("ABCDEFGHIJKL"))
    px = np.exp(ret.cumsum())
    t = 150
    bumped = ret.copy()
    bumped.iloc[t:] += 0.5                                   # change only the future
    px_b = np.exp(bumped.cumsum())
    for f in (lambda r, p: S.w_tsmom(r, 12, 1), lambda r, p: S.w_xsmom(r, 12, 1),
              lambda r, p: S.w_value(p, 5, 1 / 3, 1)):
        w0, w1 = f(ret, px), f(bumped, px_b)
        pd.testing.assert_frame_equal(w0.iloc[:t], w1.iloc[:t])
