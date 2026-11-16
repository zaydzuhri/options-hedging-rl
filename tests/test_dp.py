"""DP: structure of the solution, limiting cases, convergence, and agreement with simulation."""

import numpy as np
import pytest

from backtest import HedgingEnv, run_episode
from strategies import STRATEGIES
from strategies.dp import h_grid, optimal_risk, solve_gbm
from strategy import black_scholes
from tests.conftest import gbm_episodes

BASE = dict(strike=100.0, is_call=True, N=21, dt=1 / 252, sigma=0.2, mu=0.0, rate=0.0, cost_rate=0.001,
            risk_aversion=1.0, h=h_grid(True, (0, 1), 201), n_x=201)
PREMIUM = float(black_scholes(100.0, 100.0, 21 / 252, 0.2, 0.0)[0])
width = lambda sol: (sol["hi"] - sol["lo"])[0, len(sol["x"]) // 2]       # at the strike, 21 days left


def test_solution_is_a_no_trade_band():
    sol = solve_gbm(**{**BASE, "n_x": 101}, check=True)
    assert sol["mismatch"] <= 1.0                         # "clip to [lo, hi]" is the best action from every position
    assert np.all(sol["lo"] <= sol["hi"])


def test_band_collapses_without_costs_and_widens_with_them():
    assert width(solve_gbm(**{**BASE, "cost_rate": 0.0})) == 0.0
    assert 0.0 < width(solve_gbm(**{**BASE, "cost_rate": 0.0005})) < width(solve_gbm(**{**BASE, "cost_rate": 0.002}))


def test_more_risk_aversion_narrows_the_band():
    assert width(solve_gbm(**{**BASE, "risk_aversion": 10.0})) < width(solve_gbm(**BASE))


def test_grid_refinement_converges():
    risk = lambda nx, nh: optimal_risk(solve_gbm(**{**BASE, "n_x": nx, "h": h_grid(True, (0, 1), nh)}), 100.0, PREMIUM)
    coarse, mid, fine = risk(201, 101), risk(401, 201), risk(801, 401)      # halve both grid steps twice
    assert coarse > mid > fine and abs(fine - mid) < abs(mid - coarse) / 2 and abs(fine - mid) < 0.003


def test_dp_value_matches_simulated_policy_and_beats_delta():
    value = optimal_risk(solve_gbm(**{**BASE, "h": h_grid(True, (0, 1), 401)}), 100.0, PREMIUM)
    env = HedgingEnv("call", 1.0, 0.0, cost_bps=10, position_limits=(0.0, 1.0))
    eps = gbm_episodes(4000, seed=7)
    entropic = lambda pnl: float(np.log(np.mean(np.exp(-np.asarray(pnl)))))
    policy = STRATEGIES["dp"](risk_aversion=1.0, h_range=[0, 1])       # solved once, reused for every episode
    dp = entropic([run_episode(env, policy, ep)["pnl"] for ep in eps])
    delta = entropic([run_episode(env, STRATEGIES["delta_hedge"](), ep)["pnl"] for ep in eps])
    assert dp == pytest.approx(value, abs=0.02)            # Monte Carlo error of 4000 episodes is ~0.007
    assert dp < delta


def test_interest_is_priced_in():
    """With interest, holding the band costs carry: the DP value must still match simulation."""
    sol = solve_gbm(**{**BASE, "rate": 0.05, "mu": 0.05, "h": h_grid(True, (0, 1), 401)})
    premium = float(black_scholes(100.0, 100.0, 21 / 252, 0.2, 0.05)[0])
    env = HedgingEnv("call", 1.0, 0.05, cost_bps=10, position_limits=(0.0, 1.0))
    eps = gbm_episodes(4000, mu=0.05, seed=8)
    policy = STRATEGIES["dp"](risk_aversion=1.0, h_range=[0, 1])
    dp = [run_episode(env, policy, ep)["pnl"] for ep in eps]
    assert float(np.log(np.mean(np.exp(-np.asarray(dp))))) == pytest.approx(optimal_risk(sol, 100.0, premium), abs=0.02)
