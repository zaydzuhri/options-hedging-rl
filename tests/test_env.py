"""HedgingEnv accounting, limits, splits, and a published discrete-hedging result."""

import math

import numpy as np
import pandas as pd
import pytest

from backtest import HedgingEnv, black_scholes, run_episode, split_train_test
from strategies import STRATEGIES
from strategy import Strategy


class RandomHedge(Strategy):
    def __init__(self, seed=0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return float(self.rng.uniform(-0.5, 1.5))


def test_rewards_sum_to_final_pnl(episodes):
    env = HedgingEnv("call", 1.0, rate=0.03, cost_bps=10)
    strategy = RandomHedge()
    for ep in episodes[:20]:
        obs, done, total = env.reset(ep), False, 0.0
        while not done:
            obs, reward, done, _ = env.step(strategy.act(obs))
            total += reward
        assert total == pytest.approx(env.pnl, abs=1e-10)


def test_unhedged_pnl_is_premium_with_interest_minus_payoff(episodes):
    env = HedgingEnv("put", 1.0, rate=0.05, cost_bps=10)
    for ep in episodes[:20]:
        res = run_episode(env, STRATEGIES["no_hedge"](), ep)
        payoff = max(100.0 - ep.spots[-1], 0.0)
        assert res["pnl"] == pytest.approx(res["premium"] * math.exp(0.05 * 21 / 252) - payoff, abs=1e-10)


def test_position_limits_clip_and_count_violations(episodes):
    class TooMuch(Strategy):
        def act(self, obs):
            return 2.0

    env = HedgingEnv("call", 1.0, cost_bps=0, position_limits=(0.0, 1.0))
    res = run_episode(env, TooMuch(), episodes[0])
    assert res["n_violations"] == 21 and res["max_violation"] == pytest.approx(1.0)
    assert all(h["position"] <= 1.0 for h in env.history)
    assert run_episode(env, STRATEGIES["delta_hedge"](), episodes[0])["n_violations"] == 0


def test_discrete_delta_hedging_error_matches_derman_kamal():
    """Reproduced result: with daily rebalancing and no costs, the P&L of a delta-hedged ATM
    option has std ~ sqrt(pi/4) * vega * sigma / sqrt(N) (Kamal & Derman 1999, "When you
    cannot hedge continuously"). Limiting case: the error shrinks like 1/sqrt(N)."""
    from tests.conftest import gbm_episodes

    sigma, n = 0.2, 21
    env = HedgingEnv("call", 1.0, rate=0.0, cost_bps=0)
    pnl = [run_episode(env, STRATEGIES["delta_hedge"](), ep)["pnl"] for ep in gbm_episodes(2000, n, sigma)]
    tau = n / 252
    vega = 100 * math.exp(-0.5 * (0.5 * sigma * math.sqrt(tau)) ** 2) / math.sqrt(2 * math.pi) * math.sqrt(tau)
    predicted = math.sqrt(math.pi / 4) * vega * sigma / math.sqrt(n)
    assert np.std(pnl) == pytest.approx(predicted, rel=0.10)
    assert abs(np.mean(pnl)) < 0.05


def test_split_is_chronological_and_disjoint(dated_frame):
    train, test = split_train_test(dated_frame, {"test_start": "2018-06-01"})
    assert train.index.max() < pd.Timestamp("2018-06-01") <= test.index.min()
    assert len(train) + len(test) == len(dated_frame)
    with pytest.raises(ValueError):
        split_train_test(dated_frame, {"test_start": "2030-01-01"})
