"""
Exact dynamic programming for the hedging problem in the Black-Scholes (GBM) world.

    uv run dp_optimal.py --config configs/gbm_costs.json   # solve every "dp_optimal" strategy in the config
    uv run backtest.py   --config configs/gbm_costs.json   # evaluate the optimal policy next to the others

With entropic risk (exponential utility), the objective is E[exp(-a * P&L)] and the P&L is
a sum of per-step terms, so cash factors out of the value function. What is left depends only
on (t, spot S, shares held q), a grid we can sweep backwards. In time-N dollars (G = e^{r dt}):

    J_N(S, q) = exp(a * (λ|q|S + payoff(S)))                                  sell off, pay the option
    J_t(S, q) = min_u exp(a G^{N-t} λ|u - q| S) * E[ exp(-a u G^{N-t-1} (S' - G S)) J_{t+1}(S', u) ]

and the optimal entropic risk is -G^N * premium + (1/a) * log J_0(S_0, 0). The spot follows a
Markov chain on a log-price grid matched to the GBM increments (each cell gets the normal
probability mass of its interval); positions and actions share one grid inside the position
limits. The grids are the only approximation (see the refinement test).

The optimal policy is a no-trade band [lower(t, S), upper(t, S)]: keep q if it is inside,
otherwise trade to the nearest edge. lower is the best action from the smallest position,
upper the best action from the largest; the solver checks that every other position agrees.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from backtest import black_scholes

_cdf = np.vectorize(NormalDist().cdf, otypes=[float])


def solve(kind, moneyness, rate, cost_rate, steps_per_year, n_steps, limits, mu, sigma, pricing_vol,
          risk_aversion=1.0, n_spot=801, n_pos=401, width=6.0) -> dict:
    """Backward induction on the (spot, position) grid. Returns the band, the grids and the optimal risk."""
    dt, lam, a = 1.0 / steps_per_year, cost_rate, risk_aversion
    G = math.exp(rate * dt)
    strike = 100.0 * moneyness

    # log-spot grid around 100 covering +-`width` standard deviations of the whole option life
    half = width * sigma * math.sqrt(n_steps * dt)
    z = np.linspace(-half, half, n_spot | 1)                   # odd size: 100 is a grid point
    spots = 100.0 * np.exp(z)
    cells = np.concatenate([[-np.inf], (z[:-1] + z[1:]) / 2, [np.inf]])
    step_mean, step_std = (mu - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt)
    P = np.diff(_cdf((cells[None, :] - z[:, None] - step_mean) / step_std), axis=1)   # P[i, j] = Pr(i -> j)

    lo, hi = limits
    q = np.union1d(np.linspace(lo, hi, n_pos), [0.0] if lo <= 0.0 <= hi else [])   # q_0 = 0 must be on it
    payoff = np.maximum(spots - strike, 0.0) if kind == "call" else np.maximum(strike - spots, 0.0)
    log_j = a * (lam * np.abs(q)[None, :] * spots[:, None] + payoff[:, None])         # log J_N [spot, q]
    trade_cost = a * lam * np.abs(q[None, :] - q[:, None])                             # [q, u], times G^{N-t} S

    lower, upper = np.empty((n_steps, len(spots))), np.empty((n_steps, len(spots)))
    band_mismatch = 0
    for t in reversed(range(n_steps)):
        c_next, c_now = G ** (n_steps - t - 1), G ** (n_steps - t)
        # log E[exp(-a u c_next (S' - G S)) J_{t+1}(S', u) | S] for every (S, u); centring at 100 keeps exp in range
        m = -a * q[None, :] * c_next * (spots[:, None] - 100.0) + log_j
        shift = m.max(axis=0)
        log_e = (np.log(P @ np.exp(m - shift)) + shift
                 + a * q[None, :] * c_next * (G * spots[:, None] - 100.0))             # [spot, u]
        new_log_j, best_u = np.empty_like(log_j), np.empty(log_j.shape, dtype=np.int64)
        for rows in np.array_split(np.arange(len(spots)), 16):                          # [spot, q, u] in chunks
            total = c_now * trade_cost[None] * spots[rows, None, None] + log_e[rows, None, :]
            best_u[rows] = total.argmin(axis=-1)
            new_log_j[rows] = np.take_along_axis(total, best_u[rows][..., None], axis=-1)[..., 0]
        log_j = new_log_j
        lower[t], upper[t] = q[best_u[:, 0]], q[best_u[:, -1]]
        band = np.clip(np.arange(len(q))[None, :], best_u[:, :1], best_u[:, -1:])
        band_mismatch = max(band_mismatch, int(np.abs(band - best_u).max()))

    premium = black_scholes(100.0, strike, n_steps * dt, pricing_vol, rate, kind)[0]
    risk = -G**n_steps * premium + log_j[len(spots) // 2, np.searchsorted(q, 0.0)] / a
    return dict(spots=spots, positions=q, lower=lower, upper=upper, risk=risk, premium=premium,
                band_mismatch=band_mismatch, risk_aversion=a)


def solve_from_config(cfg, params) -> dict:
    data, vol, opt, mkt = cfg["data"], cfg["vol"], cfg["option"], cfg["market"]
    if data["source"] != "gbm" or vol["type"] != "constant":
        raise SystemExit("dp_optimal needs the Black-Scholes world: data.source 'gbm' and vol.type 'constant'")
    kind = opt.get("type", "call")
    limits = mkt.get("position_limits") or ((0.0, 1.0) if kind == "call" else (-1.0, 0.0))
    return solve(kind, opt.get("moneyness", 1.0), mkt.get("rate", 0.0), mkt.get("cost_bps", 0.0) / 1e4,
                 mkt.get("steps_per_year", 252), opt["maturity_steps"], limits, data.get("mu", 0.0), data["sigma"],
                 vol["value"], **{k: v for k, v in params.items() if k in ("risk_aversion", "n_spot", "n_pos", "width")})


def save(sol, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **sol)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a gbm config with 'dp_optimal' strategies")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)
    jobs = [s for s in cfg["strategies"] if s["name"] == "dp_optimal"]
    if not jobs:
        raise SystemExit("no 'dp_optimal' strategies in this config")
    for s in jobs:
        params = s["params"]
        sol = solve_from_config(cfg, params)
        save(sol, params["model"])
        print(f"[{name}] {s.get('label', s['name'])}: optimal entropic risk {sol['risk']:.4f} "
              f"(risk aversion {sol['risk_aversion']:g}, grid {len(sol['spots'])} spots x {len(sol['positions'])} "
              f"positions, band check: off by at most {sol['band_mismatch']} grid steps) -> {params['model']}")


if __name__ == "__main__":
    main()
