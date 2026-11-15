"""Deep hedging: the differentiable P&L must be the backtester's P&L, and its gradient the true one."""

import math

import numpy as np
import pytest
import torch

from backtest import HedgingEnv, run_episode
from deep_hedge import (ARCHS, DEFAULTS, DTYPE, episode_pnl, episode_tensors, load_policy, risk, save_policy)
from strategies.learned import DeepHedge

MARKET = dict(kind="call", moneyness=1.0, rate=0.03, cost_rate=0.001, steps_per_year=252, n_steps=21, limits=(0.0, 1.0))


def random_policy(arch, seed=0):
    torch.manual_seed(seed)
    policy = ARCHS[arch](torch.zeros(4, dtype=DTYPE), torch.ones(4, dtype=DTYPE), MARKET["limits"], 16, 2).to(DTYPE)
    with torch.no_grad():
        for p in policy.parameters():   # move away from the delta-hedge initialization
            p.add_(0.3 * torch.randn_like(p))
    return policy


def pnl(policy, ep):
    m = MARKET
    return episode_pnl(policy, ep, m["cost_rate"], m["rate"], 1.0 / m["steps_per_year"], m["limits"])


@pytest.mark.parametrize("arch", ["band", "mlp"])
def test_differentiable_pnl_matches_backtester(arch, episodes, tmp_path):
    policy = random_policy(arch)
    path = tmp_path / f"{arch}.pt"
    save_policy(policy, {**DEFAULTS, "arch": arch, "hidden": 16, "layers": 2}, path)
    env = HedgingEnv("call", 1.0, MARKET["rate"], MARKET["cost_rate"] * 1e4, 252, MARKET["limits"])
    strategy = DeepHedge(model=str(path))
    eps = episodes[:30]
    env_pnl = np.array([run_episode(env, strategy, ep)["pnl"] for ep in eps])
    ep = episode_tensors(eps, "call", 1.0, MARKET["rate"], 252)
    with torch.no_grad():
        torch_pnl = pnl(load_policy(path), ep).numpy()
    assert np.abs(env_pnl - torch_pnl).max() < 1e-10


def test_pathwise_gradient_matches_finite_differences(episodes):
    policy = random_policy("band")
    ep = episode_tensors(episodes[:200], "call", 1.0, MARKET["rate"], 252)
    objective = {"type": "entropic", "risk_aversion": 1.0}
    loss = risk(pnl(policy, ep), objective)
    loss.backward()
    bias = policy.net[-1].bias
    h = 1e-6
    with torch.no_grad():
        bias[0] += h
        up = risk(pnl(policy, ep), objective).item()
        bias[0] -= 2 * h
        down = risk(pnl(policy, ep), objective).item()
    assert bias.grad[0].item() == pytest.approx((up - down) / (2 * h), rel=1e-4, abs=1e-8)


def test_risk_measures():
    x = torch.arange(100, dtype=DTYPE) - 50.0                        # P&L -50 .. 49
    assert risk(x, {"type": "cvar", "alpha": 0.05}).item() == pytest.approx(48.0)   # mean loss of worst 5: 50..46
    g = torch.as_tensor(np.random.default_rng(0).normal(0.3, 0.5, 200_000), dtype=DTYPE)
    a = 0.4                                                          # normal P&L: -mean + a/2 var exactly
    assert risk(g, {"type": "entropic", "risk_aversion": a}).item() == pytest.approx(-0.3 + a / 2 * 0.25, abs=3e-3)


def test_policies_respect_position_limits(episodes):
    for arch in ["band", "mlp"]:
        policy = random_policy(arch, seed=1)
        ep = episode_tensors(episodes[:50], "call", 1.0, 0.0, 252)
        with torch.no_grad():
            out = policy(ep["features"][:, 0], ep["delta"][:, 0], torch.full((50,), 5.0, dtype=DTYPE))
        assert out.min() >= 0.0 and out.max() <= 1.0
