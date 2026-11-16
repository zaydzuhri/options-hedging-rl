"""Learned hedging policies (deep hedging): small neural networks trained by train.py, played in the
backtester by the `deep_hedge` strategy.

A policy maps (features, delta, current position) -> target position. Two architectures:
  band: learn a no-trade band [delta - lower, delta + upper] around Black-Scholes delta; inside it, don't
        trade; outside it, trade to the nearest edge. "Don't trade" is built into the architecture
        (Imaki et al., 2021): the shape of the optimal policy, with a learned (and possibly asymmetric) width.
  mlp:  target = delta + correction(features, current position), unconstrained.
Both start from the daily delta hedge (or a 0.05 band), and project onto the position limits, so a policy
never asks for an infeasible position. Feature scaling and limits are stored in the checkpoint.
"""

import math
from pathlib import Path

import torch
from torch import nn

from strategy import Observation, Strategy

DTYPE = torch.float64   # same precision as the numpy backtester, so the two agree to ~1e-12
FEATURES = ["log_moneyness", "sqrt_tau", "vol", "delta"]   # what a policy sees, besides its position


def _mlp(n_in, n_out, hidden, layers):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    last = nn.Linear(d, n_out)
    nn.init.zeros_(last.weight)   # start from a constant output, see each policy's docstring
    return nn.Sequential(*mods, last)


class Policy(nn.Module):
    """Base: standardizes features with training-set statistics and projects the output onto the
    position limits (both stored in the checkpoint)."""

    def __init__(self, feat_mean, feat_std, limits):
        super().__init__()
        self.register_buffer("feat_mean", feat_mean)
        self.register_buffer("feat_std", torch.where(feat_std > 1e-8, feat_std, 1.0))   # constant vol -> std 0
        self.register_buffer("limits", torch.as_tensor(limits, dtype=feat_mean.dtype))

    def norm(self, feats):
        return (feats - self.feat_mean) / self.feat_std

    def project(self, target):
        return torch.minimum(torch.maximum(target, self.limits[0]), self.limits[1])


class NoTransactionBand(Policy):
    """No-trade band around delta with a learned width (starts as a 0.05 band)."""

    def __init__(self, feat_mean, feat_std, limits, hidden=32, layers=2, init_width=0.05):
        super().__init__(feat_mean, feat_std, limits)
        self.net = _mlp(len(FEATURES), 2, hidden, layers)
        nn.init.constant_(self.net[-1].bias, math.log(math.expm1(init_width)))   # softplus^-1

    def band(self, feats, delta):
        lower, upper = nn.functional.softplus(self.net(self.norm(feats))).unbind(-1)
        return delta - lower, delta + upper

    def forward(self, feats, delta, position):
        lo, hi = self.band(feats, delta)
        return self.project(torch.minimum(torch.maximum(position, lo), hi))


class MLPHedge(Policy):
    """target = delta + correction(features, current position); the correction starts at zero."""

    def __init__(self, feat_mean, feat_std, limits, hidden=32, layers=2):
        super().__init__(feat_mean, feat_std, limits)
        self.net = _mlp(len(FEATURES) + 1, 1, hidden, layers)

    def forward(self, feats, delta, position):
        x = torch.cat([self.norm(feats), position.unsqueeze(-1)], dim=-1)
        return self.project(delta + self.net(x).squeeze(-1))


ARCHS = {"band": NoTransactionBand, "mlp": MLPHedge}


def save_policy(policy, p, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(arch=p["arch"], hidden=p["hidden"], layers=p["layers"], params=p,
                    state_dict=policy.state_dict()), path)


def load_policy(path):
    ckpt = torch.load(path, weights_only=True)
    state = ckpt["state_dict"]
    policy = ARCHS[ckpt["arch"]](state["feat_mean"], state["feat_std"], state["limits"], ckpt["hidden"],
                                 ckpt["layers"]).to(DTYPE)
    policy.load_state_dict(state)
    return policy.eval()


class DeepHedge(Strategy):
    """A policy trained with pathwise gradients through whole episodes (see train.py).

    params:
      model (str): checkpoint path; `uv run train.py --config <config>` writes it.
      The other params of the config entry (arch, objective, steps, ...) are only read at training time.
    """

    def __init__(self, **params):
        super().__init__(**params)
        if not Path(params["model"]).exists():
            raise FileNotFoundError(f"{params['model']} not found: train it first with "
                                    "`uv run train.py --config <this config>`")
        self.policy = load_policy(params["model"])

    def act(self, obs: Observation) -> float:
        feats = torch.tensor([[math.log(obs.spot / obs.strike), math.sqrt(obs.tau), obs.vol, obs.delta]], dtype=DTYPE)
        with torch.no_grad():
            target = self.policy(feats, torch.tensor([obs.delta], dtype=DTYPE), torch.tensor([obs.position], dtype=DTYPE))
        return target.item()
