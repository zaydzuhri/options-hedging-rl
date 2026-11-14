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
  - What the plan is (param `feedback`):
      open-loop (default):  u_k = a_k, one number per step shared by all scenarios. The plan can't
                            plan to react to future moves (re-solving every step is what makes the
                            overall policy react), so it over-estimates future risk, more so for long H.
      feedback:             u_k = a_k + b_k (S_k - S_0), a rule that reacts to where the price has gone
                            in each scenario ("affine disturbance feedback"). The optimizer picks the
                            a's and b's. u_0 = a_0 is still one number, since S_0 is already observed.
    Either way only u_0 is applied, then everything is re-solved next step: still MPC.
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
    feedback (bool, default False):     plan with the feedback rule u_k = a_k + b_k (S_k - S_0) instead of fixed u_k
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

        # Decision variables theta = [a_0..a_{H-1}] (open-loop) or [a_0..a_{H-1}, b_0..b_{H-1}] (feedback).
        # Warm start: yesterday's plan shifted by one step; on the first step "hold delta"
        # (and, for feedback, b = gamma: delta's own sensitivity to the price).
        feedback = self.params.get("feedback", False)
        if self.plan is not None and len(self.plan[0]) > 1:
            theta0 = np.concatenate([np.append(x[1:], x[-1])[:H] for x in self.plan])
        else:
            theta0 = np.concatenate([np.full(H, obs.delta)] + ([np.full(H, obs.gamma)] if feedback else []))
        res = minimize(self.objective, theta0, args=(S, V_H, obs, dt, at_expiry, feedback),
                       jac=True, method="L-BFGS-B")
        self.plan = np.split(res.x, 2) if feedback else [res.x]

        u0 = float(res.x[0])                                       # receding horizon: apply first control only (a_0)
        return obs.position if abs(u0 - obs.position) < self.params.get("min_trade", 1e-3) else u0

    def simulate_prices(self, obs: Observation, H: int, dt: float) -> np.ndarray:
        """M scenario paths of the price model. Uses antithetic pairs (w, -w) to reduce noise."""
        M = self.params.get("n_scenarios", 256)
        mu = self.params.get("mu", obs.rate)
        w = self.rng.standard_normal((M // 2, H))
        w = np.concatenate([w, -w])
        logret = (mu - 0.5 * obs.vol**2) * dt + obs.vol * np.sqrt(dt) * w
        return obs.spot * np.exp(np.hstack([np.zeros((len(w), 1)), np.cumsum(logret, axis=1)]))

    def objective(self, theta, S, V_H, obs: Observation, dt, at_expiry, feedback):
        """J(theta) and dJ/dtheta. Forward pass = roll the dynamics f; backward pass = adjoint (costates)."""
        lam = self.params.get("risk_aversion", 1.0)
        c, g = obs.cost_rate, np.exp(obs.rate * dt)
        M, H = S.shape[0], S.shape[1] - 1
        absval = lambda x: np.sqrt(x**2 + EPS**2)
        sign = lambda x: x / np.sqrt(x**2 + EPS**2)                 # derivative of absval

        # controls for every scenario and step: U[m, k] = shares held from step k to k+1 in scenario m
        a = theta[:H]
        X = S[:, :H] - S[:, :1]                                     # price move since now, at each step
        U = a + theta[H:] * X if feedback else np.broadcast_to(a, (M, H))

        # ---- forward pass: x_{k+1} = f(x_k, u_k), all M scenarios at once ----
        cash, h = np.full(M, obs.cash), np.full(M, obs.position)
        trades = np.empty((M, H))
        for k in range(H):
            trades[:, k] = U[:, k] - h
            cash = (cash - trades[:, k] * S[:, k] - c * absval(trades[:, k]) * S[:, k]) * g
            h = U[:, k]
        pnl = cash + h * S[:, H] - V_H
        if at_expiry:
            pnl -= c * absval(h) * S[:, H]
        J = -pnl.mean() + 0.5 * lam * pnl.var()

        # ---- backward pass: costates p_cash = dP&L/dcash_k, p_h = dP&L/dh_k (per scenario) ----
        dJ_dpnl = (-1.0 + lam * (pnl - pnl.mean())) / M
        p_cash = 1.0
        p_h = S[:, H] * (1.0 - (c * sign(h) if at_expiry else 0.0))
        dpnl_dU = np.empty((M, H))
        for k in reversed(range(H)):
            dcash_dtrade = -g * S[:, k] * (1.0 + c * sign(trades[:, k]))   # d cash_{k+1} / d trade_k
            dpnl_dU[:, k] = p_cash * dcash_dtrade + p_h                    # u_k moves trade_k and h_{k+1}
            p_h = -p_cash * dcash_dtrade                                   # h_k only enters via trade_k = u_k - h_k
            p_cash = p_cash * g
        dJ_dU = dJ_dpnl[:, None] * dpnl_dU
        grad = dJ_dU.sum(0)                                          # dJ/da_k  (dU/da = 1)
        if feedback:
            grad = np.concatenate([grad, (dJ_dU * X).sum(0)])        # dJ/db_k  (dU/db = X)
        return J, grad
