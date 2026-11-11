"""Model Predictive Control (MPC) hedge, solved by single shooting.

Hedging written as a stochastic optimal control problem. At planning step k (k = 0 is "now"):

  State     x_k = (S_k, h_k, cash_k)       spot price, shares held, cash
  Control   u_k                            shares to hold from step k to k+1
  Noise     w_k ~ N(0, 1)                  random price shock
  Dynamics  x_{k+1} = f(x_k, u_k, w_k):
      S_{k+1}    = S_k * exp((mu - vol^2/2) dt + vol sqrt(dt) w_k)              <- price MODEL (GBM)
      h_{k+1}    = u_k
      cash_{k+1} = (cash_k - (u_k - h_k) S_k - c |u_k - h_k| S_k) * exp(r dt)  <- pay for shares + cost
  Terminal  P&L_H = cash_H + h_H S_H - V(S_H)        (- unwind cost c|h_H|S_H if H is the expiry)
            V = option value at the horizon: the payoff if H is the expiry, else its Black-Scholes price
  Objective J(u_0..u_{H-1}) = -E[P&L_H] + (risk_aversion / 2) * Var[P&L_H]

Notes:
  - The price S does not depend on our controls (we are too small to move the market), so the
    controller needs a MODEL for it. We use GBM with the currently observed vol -- the same
    assumption Black-Scholes makes. The backtest then runs it on real data (plant != model).
  - E and Var are approximated with M sampled scenarios, drawn once per solve, so J becomes a
    deterministic function of u that an ordinary optimizer can minimize.
  - Single shooting: the decision variables are only the controls u_0..u_{H-1}; states are
    obtained by rolling f forward from the current state. The gradient dJ/du comes from a
    backward (adjoint / costate) pass through the same rollout.
  - MPC (receding horizon): solve from the newly observed state every step, apply u_0 only.
  - The plan is open-loop (one u_k for all scenarios), so it can't plan to react to future
    moves -- re-solving every step is what makes the overall policy react.
"""

import numpy as np
from scipy.optimize import minimize

from strategy import Observation, Strategy, black_scholes

EPS = 1e-4  # smoothing for |x| ~ sqrt(x^2 + EPS^2), so the objective is differentiable


class MPCHedge(Strategy):
    """params:
    horizon (int, default 5):           planning steps H (capped at the steps left to expiry)
    risk_aversion (float, default 1.0): lambda; higher = care more about variance, less about costs
    n_scenarios (int, default 256):     Monte Carlo scenarios M per solve
    mu (float, default = obs.rate):     drift the price model assumes. Defaulting to the interest rate means
                                        the model expects no free profit from holding shares (no speculating)
    min_trade (float, default 1e-3):    skip trades smaller than this (noise from the smoothing)
    seed (int, default 0)
    """

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.params.get("seed", 0))
        self.plan = None  # last solution, used to warm-start the next solve

    def act(self, obs: Observation) -> float:
        H = min(self.params.get("horizon", 5), obs.steps_left)
        dt = obs.tau / obs.steps_left
        at_expiry = H == obs.steps_left

        S = self.simulate_prices(obs, H, dt)                      # (M, H+1), S[:, 0] = obs.spot
        tau_H = (obs.steps_left - H) * dt                          # time left after the horizon
        V_H = black_scholes(S[:, H], obs.strike, tau_H, obs.vol, obs.rate, obs.kind)[0]

        # warm start: yesterday's plan shifted by one step, or "hold delta" on the first step
        if self.plan is not None and len(self.plan) > 1:
            u_init = np.append(self.plan[1:], self.plan[-1])[:H]
        else:
            u_init = np.full(H, obs.delta)
        res = minimize(self.objective, u_init, args=(S, V_H, obs, dt, at_expiry), jac=True, method="L-BFGS-B")
        self.plan = res.x

        u0 = float(res.x[0])                                       # receding horizon: apply first control only
        return obs.position if abs(u0 - obs.position) < self.params.get("min_trade", 1e-3) else u0

    def simulate_prices(self, obs: Observation, H: int, dt: float) -> np.ndarray:
        """M scenario paths of the price model. Uses antithetic pairs (w, -w) to reduce noise."""
        M = self.params.get("n_scenarios", 256)
        mu = self.params.get("mu", obs.rate)
        w = self.rng.standard_normal((M // 2, H))
        w = np.concatenate([w, -w])
        logret = (mu - 0.5 * obs.vol**2) * dt + obs.vol * np.sqrt(dt) * w
        return obs.spot * np.exp(np.hstack([np.zeros((len(w), 1)), np.cumsum(logret, axis=1)]))

    def objective(self, u, S, V_H, obs: Observation, dt, at_expiry):
        """J(u) and dJ/du. Forward pass = roll the dynamics f; backward pass = adjoint (costates)."""
        lam = self.params.get("risk_aversion", 1.0)
        c, g = obs.cost_rate, np.exp(obs.rate * dt)
        M, H = S.shape[0], len(u)
        absval = lambda x: np.sqrt(x**2 + EPS**2)
        sign = lambda x: x / np.sqrt(x**2 + EPS**2)                 # derivative of absval

        # ---- forward pass: x_{k+1} = f(x_k, u_k), all M scenarios at once ----
        cash, h = np.full(M, obs.cash), obs.position
        trades = np.empty(H)
        for k in range(H):
            trades[k] = u[k] - h
            cash = (cash - trades[k] * S[:, k] - c * absval(trades[k]) * S[:, k]) * g
            h = u[k]
        pnl = cash + h * S[:, H] - V_H
        if at_expiry:
            pnl -= c * absval(h) * S[:, H]
        J = -pnl.mean() + 0.5 * lam * pnl.var()

        # ---- backward pass: costates p_cash = dP&L/dcash_k, p_h = dP&L/dh_k (per scenario) ----
        dJ_dpnl = (-1.0 + lam * (pnl - pnl.mean())) / M
        p_cash = 1.0
        p_h = S[:, H] * (1.0 - (c * sign(h) if at_expiry else 0.0))
        grad = np.empty(H)
        for k in reversed(range(H)):
            dcash_dtrade = -g * S[:, k] * (1.0 + c * sign(trades[k]))   # d cash_{k+1} / d trade_k
            dpnl_du = p_cash * dcash_dtrade + p_h                       # u_k moves trade_k and h_{k+1}
            grad[k] = dJ_dpnl @ dpnl_du
            p_h = -p_cash * dcash_dtrade                                # h_k only enters via trade_k = u_k - h_k
            p_cash = p_cash * g
        return J, grad
