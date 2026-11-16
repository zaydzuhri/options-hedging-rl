"""Classical strategies: limiting cases of the Whalley-Wilmott band."""

import math

import pytest

from backtest import HedgingEnv, run_episode
from strategies import STRATEGIES
from strategy import Observation


def obs(**kw):
    base = dict(step=0, steps_left=21, tau=21 / 252, spot=100.0, strike=100.0, is_call=True, vol=0.2, rate=0.0, cost_rate=0.001,
                option_price=2.3, delta=0.51, gamma=0.069, position=0.0, cash=2.3, pnl=0.0)
    return Observation(**{**base, **kw})


def test_whalley_wilmott_without_costs_is_the_daily_delta_hedge(episodes):
    env = HedgingEnv("call", 1.0, rate=0.02, cost_bps=0)
    for ep in episodes[:30]:
        ww = run_episode(env, STRATEGIES["ww_band"](risk_aversion=1.0), ep)
        delta = run_episode(env, STRATEGIES["delta_hedge"](), ep)
        assert ww["pnl"] == pytest.approx(delta["pnl"], abs=1e-12)


def test_whalley_wilmott_band_formula_and_monotonicity():
    ww = STRATEGIES["ww_band"]
    half = (1.5 * 0.001 * 100.0 * 0.069**2 / 1.0) ** (1 / 3)
    assert ww(risk_aversion=1.0).act(obs(position=0.0)) == pytest.approx(0.51 - half)   # trades to the lower edge
    assert ww(risk_aversion=1.0).act(obs(position=1.0)) == pytest.approx(0.51 + half)   # ... or the upper edge
    assert ww(risk_aversion=1.0).act(obs(position=0.5)) == 0.5                         # inside: no trade
    lower = lambda a: ww(risk_aversion=a).act(obs(position=0.0))
    assert lower(10.0) > lower(1.0)                                                     # more risk averse: narrower
    assert math.isclose(ww().act(obs(position=0.0, cost_rate=0.0)), 0.51)                # no costs: delta
