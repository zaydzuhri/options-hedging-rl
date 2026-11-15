"""Exact DP: structure of the solution, limiting cases, convergence, and agreement with simulation."""

import numpy as np
import pytest

from backtest import HedgingEnv, run_episode
from dp_optimal import save, solve
from strategies.optimal import DPOptimal
from tests.conftest import gbm_episodes

BASE = dict(kind="call", moneyness=1.0, rate=0.0, cost_rate=0.001, steps_per_year=252, n_steps=21,
            limits=(0.0, 1.0), mu=0.0, sigma=0.2, pricing_vol=0.2, risk_aversion=1.0, n_spot=401, n_pos=201)


def test_solution_is_a_no_trade_band():
    sol = solve(**BASE)
    assert sol["band_mismatch"] == 0                       # "clip to [lower, upper]" from every position
    assert np.all(sol["lower"] <= sol["upper"])


def test_band_collapses_without_costs_and_widens_with_them():
    width = lambda cost: (lambda s: (s["upper"] - s["lower"])[0, len(s["spots"]) // 2])(solve(**{**BASE, "cost_rate": cost}))
    assert width(0.0) == 0.0
    assert 0.0 < width(0.0005) < width(0.002)


def test_more_risk_aversion_narrows_the_band():
    width = lambda a: (lambda s: (s["upper"] - s["lower"])[0, len(s["spots"]) // 2])(solve(**{**BASE, "risk_aversion": a}))
    assert width(10.0) < width(1.0)


def test_grid_refinement_converges():
    coarse = solve(**{**BASE, "n_spot": 201, "n_pos": 101})["risk"]
    mid = solve(**BASE)["risk"]
    fine = solve(**{**BASE, "n_spot": 801, "n_pos": 401})["risk"]
    assert abs(fine - mid) < abs(mid - coarse) / 2 and abs(fine - mid) < 0.003


def test_dp_value_matches_simulated_policy_and_beats_delta(tmp_path):
    sol = solve(**BASE)
    save(sol, tmp_path / "dp.npz")
    env = HedgingEnv("call", 1.0, 0.0, cost_bps=10, position_limits=(0.0, 1.0))
    eps = gbm_episodes(4000, seed=7)
    entropic = lambda pnl: float(np.log(np.mean(np.exp(-np.asarray(pnl)))))
    dp = entropic([run_episode(env, DPOptimal(model=str(tmp_path / "dp.npz")), ep)["pnl"] for ep in eps])
    from strategies import STRATEGIES
    delta = entropic([run_episode(env, STRATEGIES["delta_hedge"](), ep)["pnl"] for ep in eps])
    assert dp == pytest.approx(sol["risk"], abs=0.02)      # Monte Carlo error of 4000 episodes is ~0.007
    assert dp < delta
