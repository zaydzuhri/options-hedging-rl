"""Model Predictive Control (MPC) hedge, solved by single shooting.

Hedging written as a stochastic optimal control problem. At planning step k (k = 0 is "now"):

  State     x_k = (S_k, h_k, cash_k)       spot price, shares held, cash
  Control   u_k                            shares to hold from step k to k+1
  Noise     the price model (GBM or GJR-GARCH, see strategies/models.py)
  Dynamics  x_{k+1} = f(x_k, u_k, noise):
      S_{k+1}    = price model                                                  <- MODEL of the market
      h_{k+1}    = u_k
      cash_{k+1} = (cash_k - (u_k - h_k) S_k - c |u_k - h_k| S_k) * exp(r dt)  <- pay for shares + cost
  Terminal  P&L_H = cash_H + h_H S_H - V(S_H)        (- unwind cost c|h_H|S_H if H is the expiry)
            V = option value at the horizon: the payoff if H is the expiry, else its Black-Scholes price
  Objective (param `objective`), over the M scenarios of P&L_H:
      mean_var: -E[P&L] + (risk_aversion / 2) Var[P&L]
      cara:     log E[exp(-risk_aversion P&L)] / risk_aversion       (= minus the certainty equivalent)
      rsqp:     sqrt(E[max(-P&L, 0)^2])                               (only losses count)
      cvar:     -E[P&L] + risk_aversion * CVaR_q(-P&L)                (CVaR_q = average of the worst
                                                                       (1-q) share of losses, q = cvar_level)

Notes:
  - E and Var are approximated with M sampled scenarios, drawn once per solve, so J becomes a
    deterministic function of the plan that an ordinary optimizer can minimize.
  - Single shooting: the decision variables are only the plan's parameters; states are obtained by
    rolling f forward from the current state. The gradient comes from a backward (adjoint) pass.
  - What the plan is (param `plan`):
      open:    u_k = a_k, one number per step shared by all scenarios. Can't plan to react to future
               moves, so it over-estimates future risk, more so for long horizons.
      affine:  u_k = a_k + b_k (S_k - S_0), a rule that reacts to where the price has gone in each
               scenario ("affine disturbance feedback").
      band:    u_0 = a_0 (today's trade is a free decision), and for k >= 1
               u_k = clip(h_k, D_k + a_k - w_k, D_k + a_k + w_k), where D_k is the Black-Scholes delta
               in that scenario: "only trade when outside a band around (shifted) delta, and then only
               to its edge". The optimizer picks u_0, the shifts a_k and half-widths w_k >= 0. This is the
               shape of the optimal policy under proportional costs (see strategies/dp.py). (A band for
               u_0 too would be degenerate: all scenarios start at the same h, so whenever h is inside
               the band the gradient for today's trade is exactly zero and the optimizer stalls.)
    Either way only u_0 is applied, then everything is re-solved next step (receding horizon).
"""

import numpy as np
from scipy.optimize import minimize

from strategy import Observation, Strategy, black_scholes
from strategies.models import make_model

EPS = 1e-4  # smoothing for |x| ~ sqrt(x^2 + EPS^2), so the objective is differentiable


class MPCHedge(Strategy):
    """params:
    horizon (int, default 5):           planning steps H (capped at the steps left to expiry)
    objective (str, default mean_var):  mean_var | cara | rsqp | cvar (see module docstring)
    cvar_level (float, default 0.95):   q for the cvar objective
    risk_aversion (float, default 1.0): lambda (mean_var) or g (cara); unused by rsqp
    plan (str, default open):           open | affine | band (see module docstring). feedback=True means affine.
    shared_band (bool, default false):  band plan only: one shift a and width w for all future steps
                                        (3 decision variables instead of 2H: less overfitting to the scenarios)
    model (dict, default gbm):          price model for the scenarios (strategies/models.py)
    mu (float, optional):               drift of the default GBM model (default: obs.rate, i.e. no speculating)
    n_scenarios (int, default 256):     Monte Carlo scenarios M per solve
    min_trade (float, default 1e-3):    skip trades smaller than this (noise from the smoothing)
    seed (int, default 0)
    """

    def __init__(self, **params):
        super().__init__(**params)
        spec = params.get("model") or {"type": "gbm", "mu": params.get("mu")}
        self.model = make_model(spec)
        self.plan_type = params.get("plan", "affine" if params.get("feedback") else "open")

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.params.get("seed", 0))
        self.model.reset()
        self.plan = None  # last solution, used to warm-start the next solve

    def act(self, obs: Observation) -> float:
        self.model.observe(obs)
        H = min(self.params.get("horizon", 5), obs.steps_left)
        dt = obs.tau / obs.steps_left
        at_expiry = H == obs.steps_left

        S, vols = self.model.simulate(obs, H, dt, self.params.get("n_scenarios", 256), self.rng)  # (M, H+1)
        tau = (obs.steps_left - np.arange(H + 1)) * dt                                          # time left at each step
        V_H = black_scholes(S[:, H], obs.strike, tau[H], vols[:, H], obs.rate, obs.kind)[0]
        D = None
        if self.plan_type == "band":   # Black-Scholes delta along each scenario: the band's centre
            D = np.column_stack([black_scholes(S[:, k], obs.strike, tau[k], vols[:, k], obs.rate, obs.kind)[1]
                                 for k in range(H)])

        theta0, bounds = self.initial_plan(obs, H)
        if self.plan_type == "band" and self.params.get("shared_band", False) and H > 1:
            # optimize (u_0, a, w) only; expand to the full per-step layout [u_0, a.., 0, w..]
            expand = lambda t: np.concatenate([[t[0]], np.full(H - 1, t[1]), [0.0], np.full(H - 1, t[2])])
            def small(t, *args):
                J, g = self.objective(expand(t), *args)
                return J, np.array([g[0], g[1:H].sum(), g[H + 1:].sum()])
            t0 = np.array([theta0[0], np.mean(theta0[1:H]), max(np.mean(theta0[H + 1:]), 0.01)])
            res = minimize(small, t0, args=(S, V_H, D, obs, dt, at_expiry), jac=True, method="L-BFGS-B",
                           bounds=[(None, None), (None, None), (0.0, None)])
            res.x = expand(res.x)
        else:
            res = minimize(self.objective, theta0, args=(S, V_H, D, obs, dt, at_expiry),
                           jac=True, method="L-BFGS-B", bounds=bounds)
        self.plan = res.x

        u0 = float(self.controls(res.x, S, D, np.full(len(S), obs.position), 0)[0])   # receding horizon: first control only
        return obs.position if abs(u0 - obs.position) < self.params.get("min_trade", 1e-3) else u0

    # ------------------------------------------------------------------ plan
    def initial_plan(self, obs, H):
        """Warm start = the previous plan shifted by one step; on the first step a delta-like plan."""
        n_parts = 1 if self.plan_type == "open" else 2
        if self.plan is not None and len(self.plan) // n_parts > 1:
            parts = np.split(self.plan, n_parts)
            theta0 = np.concatenate([np.append(x[1:], x[-1])[:H] for x in parts])
        elif self.plan_type == "open":
            theta0 = np.full(H, obs.delta)
        elif self.plan_type == "affine":
            theta0 = np.concatenate([np.full(H, obs.delta), np.full(H, obs.gamma)])
        else:
            theta0 = np.concatenate([np.zeros(H), np.full(H, 0.05)])
        if self.plan_type == "band":
            # today's trade: start from what the previous plan's band says (or delta on the first step)
            H_prev = 0 if self.plan is None else len(self.plan) // 2
            a, w = (self.plan[1], self.plan[H_prev + 1]) if H_prev > 1 else (0.0, 0.0)
            theta0[0] = float(np.clip(obs.position, obs.delta + a - w, obs.delta + a + w))
            theta0[H] = 0.0
        bounds = [(None, None)] * H + [(0.0, 0.0)] + [(0.0, None)] * (H - 1) if self.plan_type == "band" else None
        return theta0, bounds

    def controls(self, theta, S, D, h, k):
        """u_k for every scenario, given the holdings h before trading at step k."""
        H = S.shape[1] - 1
        a = theta[k]
        if self.plan_type == "open":
            return np.broadcast_to(a, h.shape)
        if self.plan_type == "affine":
            return a + theta[H + k] * (S[:, k] - S[:, 0])
        if k == 0:                                   # band plan: today's trade is a free decision
            return np.broadcast_to(a, h.shape)
        return np.clip(h, D[:, k] + a - theta[H + k], D[:, k] + a + theta[H + k])

    # ------------------------------------------------------------------ objective
    def objective(self, theta, S, V_H, D, obs: Observation, dt, at_expiry):
        """J(theta) and dJ/dtheta. Forward pass = roll the dynamics f; backward pass = adjoint (costates)."""
        c, g = obs.cost_rate, np.exp(obs.rate * dt)
        M, H = S.shape[0], S.shape[1] - 1
        absval = lambda x: np.sqrt(x**2 + EPS**2)
        sign = lambda x: x / np.sqrt(x**2 + EPS**2)                 # derivative of absval

        # ---- forward pass: x_{k+1} = f(x_k, u_k), all M scenarios at once ----
        cash, h = np.full(M, obs.cash), np.full(M, obs.position)
        trades, lo_hit, hi_hit = np.empty((M, H)), np.zeros((M, H), bool), np.zeros((M, H), bool)
        for k in range(H):
            u = self.controls(theta, S, D, h, k)
            if self.plan_type == "band" and k > 0:
                lo_hit[:, k] = h < D[:, k] + theta[k] - theta[H + k]
                hi_hit[:, k] = h > D[:, k] + theta[k] + theta[H + k]
            trades[:, k] = u - h
            cash = (cash - trades[:, k] * S[:, k] - c * absval(trades[:, k]) * S[:, k]) * g
            h = u
        pnl = cash + h * S[:, H] - V_H
        if at_expiry:
            pnl -= c * absval(h) * S[:, H]
        J, dJ_dpnl = self.risk(pnl)

        # ---- backward pass: costates p_cash = dP&L/dcash_{k+1}, p_h = dP&L/dh_{k+1} (per scenario) ----
        p_cash = 1.0
        p_h = S[:, H] * (1.0 - (c * sign(h) if at_expiry else 0.0))
        dpnl_du = np.empty((M, H))
        for k in reversed(range(H)):
            dcash_dtrade = -g * S[:, k] * (1.0 + c * sign(trades[:, k]))   # d cash_{k+1} / d trade_k
            dpnl_du[:, k] = p_cash * dcash_dtrade + p_h                    # u_k moves trade_k and h_{k+1}
            du_dh = (~(lo_hit[:, k] | hi_hit[:, k])) if self.plan_type == "band" and k > 0 else 0.0
            p_h = -p_cash * dcash_dtrade + dpnl_du[:, k] * du_dh           # h_k enters via trade_k (and the clip)
            p_cash = p_cash * g
        dJ_du = dJ_dpnl[:, None] * dpnl_du
        if self.plan_type == "open":
            return J, dJ_du.sum(0)
        if self.plan_type == "affine":
            return J, np.concatenate([dJ_du.sum(0), (dJ_du * (S[:, :H] - S[:, :1])).sum(0)])
        clipped = (lo_hit | hi_hit).astype(float)                          # u = edge: moves with a, and with w (+/-)
        clipped[:, 0] = 1.0                                                # u_0 = a_0 directly
        return J, np.concatenate([(dJ_du * clipped).sum(0), (dJ_du * (hi_hit.astype(float) - lo_hit)).sum(0)])

    def risk(self, pnl):
        """The objective over scenarios and its derivative w.r.t. each scenario's P&L."""
        kind, lam, M = self.params.get("objective", "mean_var"), self.params.get("risk_aversion", 1.0), len(pnl)
        if kind == "mean_var":
            return -pnl.mean() + 0.5 * lam * pnl.var(), (-1.0 + lam * (pnl - pnl.mean())) / M
        if kind == "cara":
            z = -lam * pnl
            w = np.exp(z - z.max())
            return (z.max() + np.log(w.mean())) / lam, -w / w.sum()
        if kind == "rsqp":
            loss = np.maximum(-pnl, 0.0)
            J = np.sqrt(np.mean(loss**2))
            return J, -loss / (M * max(J, 1e-12))
        if kind == "cvar":
            n_tail = max(1, int(np.ceil((1 - self.params.get("cvar_level", 0.95)) * M)))
            worst = np.argsort(pnl)[:n_tail]                       # lowest P&L = largest losses
            grad = np.full(M, -1.0 / M)
            grad[worst] -= lam / n_tail
            return -pnl.mean() - lam * pnl[worst].mean(), grad
        raise ValueError(f"unknown objective {kind!r} (use mean_var, cara, rsqp or cvar)")
