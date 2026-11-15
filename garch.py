"""
A market simulator for training on more than one history: GJR-GARCH(1,1) with bootstrapped residuals.

Real training data is ~2800 overlapping episodes that share a handful of crises, so a policy
trained on it mostly memorizes 2008. Instead we fit a volatility model to the training period
and sample as many new paths as we want from it:

    r_t = mu + e_t,   e_t = sigma_t * z_t,   z_t drawn from the fitted standardized residuals
    sigma_{t+1}^2 = omega + (alpha + gamma * 1[e_t < 0]) * e_t^2 + beta * sigma_t^2

GJR's gamma term lets losses raise volatility more than gains (the leverage effect), and
resampling the real residuals keeps their fat tails and skew. The pricing vol (VIX in the
data) is mapped from the model's one-day-ahead vol with a least-squares line fitted on the
same period, so simulated options carry the same volatility premium as real ones.
Validation and test stay on real data.
"""

import math

import numpy as np
import pandas as pd
import torch

from backtest import Episode


def fit(df: pd.DataFrame, iters: int = 200) -> dict:
    """Gaussian quasi-maximum likelihood on daily log returns (in %), then the pricing-vol line."""
    r = np.diff(np.log(df["spot"].to_numpy())) * 100.0
    mu = float(r.mean())
    e = torch.as_tensor(r - mu, dtype=torch.float64)
    var0 = float(e.var())
    inv_softplus = lambda y: math.log(math.expm1(y))
    raw = torch.tensor([inv_softplus(0.05 * var0), inv_softplus(0.03), inv_softplus(0.1), math.log(0.85 / 0.15)],
                       dtype=torch.float64, requires_grad=True)

    def unpack(raw):
        sp = torch.nn.functional.softplus
        return sp(raw[0]), sp(raw[1]), sp(raw[2]), torch.sigmoid(raw[3])

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
    forecast = np.append(var[1:], float(last))                   # at each close: variance of the next return
    model_vol = np.sqrt(forecast * 252) / 100.0                  # annualized, as a fraction
    slope, intercept = np.polyfit(model_vol, df["vol"].to_numpy()[1:], 1)
    return dict(mu=mu, omega=omega, alpha=alpha, gamma=gamma, beta=beta, resid=resid, start_var=forecast,
                vol_slope=float(slope), vol_intercept=float(intercept),
                persistence=alpha + gamma / 2 + beta)


def pricing_vol(model, var):
    return np.maximum(model["vol_intercept"] + model["vol_slope"] * np.sqrt(var * 252) / 100.0, 0.01)


def simulate(model, n_paths: int, n_steps: int, seed: int = 0) -> list[Episode]:
    """Independent paths of n_steps+1 prices starting at 100, from vol regimes seen in the fit period."""
    rng = np.random.default_rng(seed)
    var = rng.choice(model["start_var"], n_paths)
    log_s = np.zeros(n_paths)
    spots, vols = [np.full(n_paths, 100.0)], [pricing_vol(model, var)]
    for _ in range(n_steps):
        e = np.sqrt(var) * rng.choice(model["resid"], n_paths)
        log_s += (model["mu"] + e) / 100.0
        var = model["omega"] + (model["alpha"] + model["gamma"] * (e < 0)) * e**2 + model["beta"] * var
        spots.append(100.0 * np.exp(log_s))
        vols.append(pricing_vol(model, var))
    spots, vols = np.stack(spots, 1), np.stack(vols, 1)
    return [Episode(f"gjr-garch-{i}", spots[i], vols[i]) for i in range(n_paths)]
