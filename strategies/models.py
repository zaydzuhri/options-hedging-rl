"""Price models that model-based strategies (dp, mpc) plan with. Not the market itself:
the backtest feeds strategies real or simulated prices; these are the strategy's BELIEF about them.

A model is chosen in a strategy's params, e.g.
    "model": {"type": "gbm"}                                   (vol = obs.vol, drift = "mu" or obs.rate)
    "model": {"type": "gjr_garch", "mu": ..., "omega": ..., "alpha": ..., "gamma": ..., "beta": ..., "h0": ...}
GJR-GARCH parameters are per step (e.g. monthly). The strategy can't see the true variance, so the
model filters it from the prices observed so far in the episode (no look-ahead).
"""

import math

import numpy as np


class GBM:
    """Geometric Brownian motion with the currently observed vol (what Black-Scholes assumes)."""

    def __init__(self, mu=None, **_):
        self.mu = mu

    def reset(self):
        pass

    def observe(self, obs):
        pass

    def drift(self, obs):
        return obs.rate if self.mu is None else self.mu

    def vol(self, obs):
        """Annualized vol the model believes in right now."""
        return obs.vol

    def simulate(self, obs, H, dt, M, rng):
        """M price paths (antithetic pairs) for H steps from obs.spot -> (S, vols), both (M, H+1)."""
        w = rng.standard_normal((M // 2, H))
        w = np.concatenate([w, -w])
        sig = obs.vol
        logret = (self.drift(obs) - 0.5 * sig**2) * dt + sig * math.sqrt(dt) * w
        S = obs.spot * np.exp(np.hstack([np.zeros((len(w), 1)), np.cumsum(logret, axis=1)]))
        return S, np.full(S.shape, sig)


class GJRGarch:
    """GJR-GARCH(1,1) log-returns (see backtest.simulate_gjr_garch), per-step parameters.
    The conditional variance v is filtered from the episode's observed returns, starting at h0."""

    def __init__(self, mu, omega, alpha, gamma, beta, h0, **_):
        self.mu, self.omega, self.alpha, self.gamma, self.beta, self.h0 = mu, omega, alpha, gamma, beta, h0

    def next_var(self, v, eps):
        return self.omega + (self.alpha + self.gamma * (eps < 0)) * eps**2 + self.beta * v

    def reset(self):
        self.v, self.last_spot = self.h0, None

    def observe(self, obs):
        """Call once per step, before planning: update the variance with the latest return."""
        if self.last_spot is not None:
            self.v = self.next_var(self.v, math.log(obs.spot / self.last_spot) - self.mu)
        self.last_spot = obs.spot

    def drift(self, obs):
        return self.mu  # per step

    def vol(self, obs):
        dt = obs.tau / obs.steps_left
        return math.sqrt(self.v / dt)

    def simulate(self, obs, H, dt, M, rng):
        z = rng.standard_normal((M // 2, H))
        z = np.concatenate([z, -z])
        S = np.empty((M, H + 1))
        v = np.empty((M, H + 1))
        S[:, 0], v[:, 0] = obs.spot, self.v
        for k in range(H):
            eps = np.sqrt(v[:, k]) * z[:, k]
            S[:, k + 1] = S[:, k] * np.exp(self.mu + eps)
            v[:, k + 1] = self.next_var(v[:, k], eps)
        return S, np.sqrt(v / dt)       # second output: annualized vol along each path


def make_model(spec):
    spec = dict(spec or {"type": "gbm"})
    kind = spec.pop("type", "gbm")
    if kind == "gbm":
        return GBM(**spec)
    if kind == "gjr_garch":
        return GJRGarch(**spec)
    raise ValueError(f"unknown model type {kind!r} (use 'gbm' or 'gjr_garch')")
