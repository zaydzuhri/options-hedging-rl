"""Price models: what the model-based strategies (dp, mpc) plan with, and the GJR-GARCH(1,1) recursion
that every simulator in the repo shares (the backtest's gjr_garch data source, MPC's scenarios, and the
fitted simulator train.py can train on). For the strategies these are a BELIEF about prices, not the
market itself: the backtest feeds them real or simulated prices.

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


# ---------------------------------------------------------------- GJR-GARCH(1,1), per-step parameters
#     log-return  r_t = mu + eps_t,   eps_t = sqrt(v_t) z_t
#     variance    v_{t+1} = omega + (alpha + gamma * 1{eps_t < 0}) eps_t^2 + beta v_t
# Unlike GBM, volatility moves: it stays high after big moves (clustering) and rises more after drops
# than after rallies (gamma: the "leverage effect"). Glosten, Jagannathan & Runkle (1993).

def gjr_garch_var(v, eps, omega, alpha, gamma, beta):
    """Next variance after shock eps (the recursion above)."""
    return omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * v


def gjr_garch_shocks(n_steps, omega, alpha, gamma, beta, v0, draw):
    """Shocks eps (n, n_steps) and variances v (n, n_steps+1) of n paths starting at variances v0, where
    draw(k) gives the n standardized shocks z of step k (normal, antithetic, or resampled residuals)."""
    v = np.empty((len(v0), n_steps + 1))
    eps = np.empty((len(v0), n_steps))
    v[:, 0] = v0
    for k in range(n_steps):
        eps[:, k] = np.sqrt(v[:, k]) * draw(k)
        v[:, k + 1] = gjr_garch_var(v[:, k], eps[:, k], omega, alpha, gamma, beta)
    return eps, v


def fit_gjr_garch(spot, pricing_vol, steps_per_year=252, iters=200) -> dict:
    """Fit GJR-GARCH(1,1) to a price series by Gaussian quasi-maximum likelihood (in % returns, for
    conditioning; the result is in decimal units). Also returns the standardized residuals (to resample:
    they keep the data's fat tails and skew) and a least-squares line from the model's annualized vol to
    the pricing vol (e.g. VIX), so simulated options carry the same volatility premium as real ones."""
    import torch   # only for the likelihood's gradient

    r = np.diff(np.log(np.asarray(spot, dtype=float))) * 100.0
    mu = float(r.mean())
    e = torch.as_tensor(r - mu, dtype=torch.float64)
    var0 = float(e.var())
    inv_softplus = lambda y: math.log(math.expm1(y))
    raw = torch.tensor([inv_softplus(0.05 * var0), inv_softplus(0.03), inv_softplus(0.1), math.log(0.85 / 0.15)],
                       dtype=torch.float64, requires_grad=True)
    unpack = lambda raw: (torch.nn.functional.softplus(raw[0]), torch.nn.functional.softplus(raw[1]),
                          torch.nn.functional.softplus(raw[2]), torch.sigmoid(raw[3]))

    def variances(raw):
        omega, alpha, gamma, beta = unpack(raw)
        var, out = torch.tensor(var0, dtype=torch.float64), []
        for t in range(len(e)):
            out.append(var)
            var = omega + (alpha + gamma * (e[t] < 0)) * e[t] ** 2 + beta * var
        return torch.stack(out), var        # var for each return, and the forecast after the last one

    def nll():
        optim.zero_grad()
        var, _ = variances(raw)
        loss = 0.5 * torch.mean(torch.log(var) + e**2 / var)
        loss.backward()
        return loss

    optim = torch.optim.LBFGS([raw], max_iter=iters, line_search_fn="strong_wolfe")
    optim.step(nll)
    with torch.no_grad():
        var, last = variances(raw)
        omega, alpha, gamma, beta = (float(x) for x in unpack(raw))
    var = var.numpy()
    resid = (r - mu) / np.sqrt(var)
    forecast = np.append(var[1:], float(last)) / 1e4             # at each close: variance of the next return
    slope, intercept = np.polyfit(np.sqrt(forecast * steps_per_year), np.asarray(pricing_vol, float)[1:], 1)
    return dict(mu=mu / 100, omega=omega / 1e4, alpha=alpha, gamma=gamma, beta=beta, resid=resid,
                start_var=forecast, vol_slope=float(slope), vol_intercept=float(intercept),
                persistence=alpha + gamma / 2 + beta, steps_per_year=steps_per_year)


def simulate_fitted_gjr_garch(fit, n_paths, n_steps, seed=0):
    """Paths from a fit_gjr_garch fit: start in volatility regimes seen in the data, resample its
    residuals. Returns spots (n_paths, n_steps+1) from 100 and pricing vols of the same shape."""
    rng = np.random.default_rng(seed)
    v0 = rng.choice(fit["start_var"], n_paths)
    eps, v = gjr_garch_shocks(n_steps, fit["omega"], fit["alpha"], fit["gamma"], fit["beta"], v0,
                              lambda k: rng.choice(fit["resid"], n_paths))
    spots = 100.0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(fit["mu"] + eps, axis=1)]))
    vols = np.maximum(fit["vol_intercept"] + fit["vol_slope"] * np.sqrt(v * fit["steps_per_year"]), 0.01)
    return spots, vols


class GJRGarch:
    """GJR-GARCH(1,1) belief for planning (see the recursion above), per-step parameters.
    The conditional variance v is filtered from the episode's observed returns, starting at h0."""

    def __init__(self, mu, omega, alpha, gamma, beta, h0, **_):
        self.mu, self.omega, self.alpha, self.gamma, self.beta, self.h0 = mu, omega, alpha, gamma, beta, h0

    def next_var(self, v, eps):
        return gjr_garch_var(v, eps, self.omega, self.alpha, self.gamma, self.beta)

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
        eps, v = gjr_garch_shocks(H, self.omega, self.alpha, self.gamma, self.beta, np.full(M, self.v), lambda k: z[:, k])
        S = obs.spot * np.cumprod(np.hstack([np.ones((M, 1)), np.exp(self.mu + eps)]), axis=1)
        return S, np.sqrt(v / dt)       # second output: annualized vol along each path


def make_model(spec):
    spec = dict(spec or {"type": "gbm"})
    kind = spec.pop("type", "gbm")
    if kind == "gbm":
        return GBM(**spec)
    if kind == "gjr_garch":
        return GJRGarch(**spec)
    raise ValueError(f"unknown model type {kind!r} (use 'gbm' or 'gjr_garch')")
