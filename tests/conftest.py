"""Shared fixtures: small simulated Black-Scholes markets, so tests need no network."""

import pandas as pd
import pytest

from backtest import make_episodes, simulate_gbm


def gbm_episodes(n_episodes=400, n_steps=21, sigma=0.2, mu=0.0, seed=0, stride=None):
    """Non-overlapping GBM episodes with a constant pricing vol equal to the true vol."""
    df = simulate_gbm(n_episodes * n_steps, sigma=sigma, mu=mu, seed=seed).to_frame("spot")
    df["vol"] = sigma
    return make_episodes(df, n_steps, stride or n_steps)


@pytest.fixture
def episodes():
    return gbm_episodes()


@pytest.fixture
def dated_frame():
    idx = pd.bdate_range("2018-01-01", periods=300)
    return pd.DataFrame({"spot": 100.0, "vol": 0.2}, index=idx)
