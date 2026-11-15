"""iLQG hedge: iterative Linear-Quadratic-Gaussian trajectory optimization (Todorov & Li, 2005).

Like iLQR, it optimizes a nominal trajectory by repeatedly (1) rolling it out, (2) taking a quadratic
model of the cost-to-go around it in a backward (Riccati-like) pass, which yields a locally optimal
AFFINE FEEDBACK law, and (3) line-searching the update. The "G" (Gaussian) part: the noise enters the
dynamics as F(x, u) * eps and its effect on the expected cost is kept in the backward pass. That matters
here because the noise is CONTROL-DEPENDENT: the shares you hold multiply the price shock.

Coordinates matter. In raw coordinates (price, shares, wealth) a second-order expansion cannot represent
gamma: "the right hedge changes with the price" is a THIRD derivative of the value function, so iLQG
would think the right hedge is fixed around the nominal path. So we plan in delta-hedged coordinates,
where the option's nonlinearity lives in the change of coordinates and what is left is nearly LQ:

  State     x = (y, z, e):   y = price,
                             z = h - Delta(t, y)        exposure mismatch: shares held minus BS delta
                             e = wealth - V(t, y)       P&L marked to the model (wealth = cash + h*y)
  Control   v = u - Delta(t, y)     how far from delta to hold over the next step (u = shares)
  Dynamics  y' = y (1 + m_k + s_k eps)                  (m_k, s_k from the price model, strategies/models.py)
            z' = u - Delta(t+1, y')                     random-walks: delta moves by ~ Gamma * dy
            e' = e + V(t,y) - c |v - z|_eps y + u (y' - y) - V(t+1, y')
               ~ e + v (y' - y) - c |v - z| y + (gamma/theta terms no hedge can remove)
  Cost      at the horizon: -P + (risk_aversion / 2) (P - P_bar)^2,  P = e - unwind cost at expiry
            i.e. "high mean P&L, small variance around it".
  |.|_eps   = sqrt(x^2 + eps_cost^2), so costs are smooth enough for a quadratic model.
V and Delta are Black-Scholes with the model's vol (smoothed over smooth_tau_steps at expiry). The
dynamics are statistically linearized (mean over the noise, and noise loading E[f * eps], by 5-point
Gauss-Hermite); their derivatives are taken numerically; the second-order term kept is the cost curvature.

Two ways to use it (param `replan`):
  true  (default): receding horizon -- re-solve from the observed state every step, apply u_0.
  false: solve once at the start over the whole option life, then only follow the feedback law
         v = v_bar_k + K_k (x - x_bar_k). The purest form: shows what "locally optimal" costs.
  false + trust_radius (HYBRID): follow the law only while the state stays where its expansion is
         valid: price within trust_radius standard deviations of the nominal price path, and exposure
         mismatch z within z_radius shares of the nominal. Outside that region, a fallback controller
         (the Whalley-Wilmott band) acts for this step -- it brings z back toward delta, the part of
         the state we CAN steer (the price we can't) -- and iLQG is re-solved from the next step on.
Line search: each candidate feedback law is scored on M sampled price scenarios with the TRUE objective
(exact |.| and payoff) and accepted only if it improves.
"""

import numpy as np

from strategy import Observation, Strategy, black_scholes
from strategies.models import make_model


class ILQGHedge(Strategy):
    """params:
    horizon (int, default 5):            planning steps (capped at steps left); ignored when replan=false
    risk_aversion (float, default 1.0):  lambda in -P + (lambda/2)(P - P_bar)^2
    replan (bool, default true):         see module docstring
    model (dict, default gbm):           price model (strategies/models.py)
    eps_cost (float or "auto", default 0.03): smoothing of |trade| (in shares); "auto" = the current
                                         Whalley-Wilmott half-width, so the smoothing matches the no-trade region
    smooth_tau_steps (float, default 1): payoff smoothing at expiry, in steps of time
    max_iter (int, default 30), tol (float, default 1e-7), n_scenarios (int, default 256), seed (int, default 0)
    min_trade (float, default 1e-3):     skip trades smaller than this
    trust_radius (float, optional):      hybrid mode (with replan=false): max |log(y / y_bar)| in units of the
                                         per-step std * sqrt(steps since the solve)
    z_radius (float, default 0.1):       hybrid mode: max |z - z_bar| in shares
    exact_first_step (bool, default true): decide today's trade with the exact |.| cost (a true no-trade band)
                                         instead of the smoothed one; see the end of solve()
    exact_law (bool, default false):     solve-once / hybrid only: the same exact-cost decision on every step that
                                         follows the law (the law's local quadratic model + the exact |.| cost),
                                         instead of the smooth affine law that nudges the position every day
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.model = make_model(params.get("model"))

    def reset(self):
        self.rng = np.random.default_rng(self.params.get("seed", 0))
        self.model.reset()
        self.law = None

    # ------------------------------------------------------------------ acting
    def act(self, obs: Observation) -> float:
        self.model.observe(obs)
        replan = self.params.get("replan", True)
        if replan or self.law is None:
            H = min(self.params.get("horizon", 5), obs.steps_left) if replan else obs.steps_left
            self.law = self.solve(obs, H)
        v_bar, x_bar, K, sig, start, tau0, local = self.law
        j = obs.step - start
        tau = tau0 - j * (obs.tau / obs.steps_left)
        x = self.coords(obs.spot, obs.position, obs.cash + obs.position * obs.spot, tau, sig, obs)
        D = self.bs(obs.spot, tau, sig, obs)[1]
        u = float(D + v_bar[j] + K[j] @ (x - x_bar[j]))
        if j > 0 and self.params.get("exact_law", False) and local[j] is not None and local[j][1] > 0:
            # follow the law's local quadratic model, but with the exact |trade| cost (a no-trade band every step)
            q_u, q_uu, q_ux, kap = local[j]
            v_free = v_bar[j] - (q_u + q_ux @ (x - x_bar[j])) / q_uu
            u = float(D + soft_threshold(v_free, x[1], kap * (x[0] / x_bar[j, 0]) / q_uu))
        r = self.params.get("trust_radius")
        if r is not None and not replan and j > 0:
            step_sd = sig * np.sqrt(obs.tau / obs.steps_left)
            off_price = abs(np.log(x[0] / x_bar[j, 0])) > r * step_sd * np.sqrt(j)
            off_z = abs(x[1] - x_bar[j, 1]) > self.params.get("z_radius", 0.1)
            if off_price or off_z or not np.isfinite(u):
                self.law = None                         # outside the local region: re-solve next step
                self.fallbacks = getattr(self, "fallbacks", 0) + 1
                u = self.fallback(obs)
        return obs.position if abs(u - obs.position) < self.params.get("min_trade", 1e-3) else u

    def fallback(self, obs):
        """Whalley-Wilmott band: trade back to the band's edge if outside it (see baselines.py)."""
        half = (3 * obs.cost_rate * obs.spot * np.exp(-obs.rate * obs.tau) * obs.gamma**2
                / (2 * self.params.get("risk_aversion", 1.0))) ** (1 / 3)
        return float(np.clip(obs.position, obs.delta - half, obs.delta + half))

    def bs(self, y, tau, sig, obs):
        """Black-Scholes (value, delta, gamma); at expiry smoothed over smooth_tau_steps steps."""
        dt = obs.tau / obs.steps_left
        return black_scholes(y, obs.strike, max(tau, self.params.get("smooth_tau_steps", 1.0) * dt), sig, obs.rate, obs.kind)

    def coords(self, y, h, w, tau, sig, obs):
        V, D, _ = self.bs(y, tau, sig, obs)
        return np.array([y, h - D, w - V])

    # ------------------------------------------------------------------ the iLQG solve
    def solve(self, obs, H):
        p = self.params
        lam, c, eps = p.get("risk_aversion", 1.0), obs.cost_rate, p.get("eps_cost", 0.03)
        if eps == "auto":   # smooth over the Whalley-Wilmott no-trade half-width (scales like c^(1/3) Gamma^(2/3))
            eps = max(1e-3, (3 * c * obs.spot * obs.gamma**2 / (2 * lam)) ** (1 / 3))
        dt = obs.tau / obs.steps_left
        m, s, _ = self.model.schedule(obs, H, dt)
        sig = self.model.schedule(obs, 0, dt)[2]           # vol for V and Delta: average over the remaining life
        taus = obs.tau - dt * np.arange(H + 1)
        at_expiry = H == obs.steps_left
        absv = lambda d: np.sqrt(d**2 + eps**2)

        def f(X, v, k, noise):
            """One step of the dynamics in (y, z, e) coordinates for a batch: X (n, 3), v (n,), noise (n,)."""
            y, z, e = X[:, 0], X[:, 1], X[:, 2]
            V0, D0, _ = self.bs(y, taus[k], sig, obs)
            u = D0 + v
            y1 = y * (1 + m[k] + s[k] * noise)
            V1, D1, _ = self.bs(y1, taus[k + 1], sig, obs)
            return np.column_stack([y1, u - D1, e + V0 - c * absv(v - z) * y + u * (y1 - y) - V1])

        # Statistical linearization: the nominal path follows the EXPECTED next state, and the noise
        # loading is the regression slope E[f * eps]. A noise-free path would miss the Ito drift of delta
        # (it moves by Gamma_S * dS^2 / 2 on average), and iLQG would pre-trade toward a delta path that
        # does not happen on average.
        zq, wq = np.polynomial.hermite_e.hermegauss(5)
        wq = wq / wq.sum()

        def moments(X, v, k):
            """E[f] and E[f * eps] for a batch of (x, v), all quadrature nodes in one vectorized call."""
            n = len(X)
            out = f(np.tile(X, (len(zq), 1)), np.tile(v, len(zq)), k, np.repeat(zq, n)).reshape(len(zq), n, 3)
            return np.einsum("q,qnd->nd", wq, out), np.einsum("q,qnd->nd", wq * zq, out)

        def linearize(x, v, k, hu=1e-5):
            """E[f], noise loading F, and their derivatives w.r.t. x and v, by central differences (one batch)."""
            hs = np.array([1e-4 * abs(x[0]), 1e-5, 1e-5])
            X = np.tile(x, (9, 1)); vv = np.full(9, float(v))
            for i in range(3):
                X[1 + 2 * i, i] += hs[i]; X[2 + 2 * i, i] -= hs[i]
            vv[7] += hu; vv[8] -= hu
            fm, fn = moments(X, vv, k)
            dfm = np.column_stack([(fm[1 + 2 * i] - fm[2 + 2 * i]) / (2 * hs[i]) for i in range(3)])
            dfn = np.column_stack([(fn[1 + 2 * i] - fn[2 + 2 * i]) / (2 * hs[i]) for i in range(3)])
            return fm[0], fn[0], dfm, (fm[7] - fm[8]) / (2 * hu), dfn, (fn[7] - fn[8]) / (2 * hu)

        steps = lambda x0: np.array([1e-4 * abs(x0[0]), 1e-5, 1e-5])     # finite-difference steps for (y, z, e)

        def grad_hess(fun, x0):
            """Gradient and Hessian of a scalar fun(x) by central differences."""
            hs = steps(x0)
            g, Hm = np.empty(3), np.empty((3, 3))
            for i in range(3):
                di = np.zeros(3); di[i] = hs[i]
                g[i] = (fun(x0 + di) - fun(x0 - di)) / (2 * hs[i])
                for j in range(3):
                    dj = np.zeros(3); dj[j] = hs[j]
                    Hm[i, j] = (fun(x0 + di + dj) - fun(x0 + di - dj) - fun(x0 - di + dj) + fun(x0 - di - dj)) / (4 * hs[i] * hs[j])
            return g, 0.5 * (Hm + Hm.T)

        def P_of(x):  # terminal P&L in the model (unwind cost at expiry)
            y, z, e = x
            return e - (c * absv(z + self.bs(y, taus[H], sig, obs)[1]) * y if at_expiry else 0.0)

        x0 = self.coords(obs.spot, obs.position, obs.cash + obs.position * obs.spot, taus[0], sig, obs)
        S, _ = self.model.simulate(obs, H, dt, p.get("n_scenarios", 256), self.rng)

        def nominal(v_ff, K, x_ref):
            xs, vs = np.empty((H + 1, 3)), np.empty(H)
            xs[0] = x0
            for k in range(H):
                vs[k] = v_ff[k] + K[k] @ (xs[k] - x_ref[k])
                xs[k + 1] = moments(xs[k][None, :], np.array([vs[k]]), k)[0][0]
            return xs, vs

        def score(v_ff, K, x_ref):
            """True objective of the law on the sampled scenarios (exact |.| and payoff)."""
            M = S.shape[0]
            h, w = np.full(M, obs.position), np.full(M, obs.cash + obs.position * obs.spot)
            for k in range(H):
                V, D, _ = self.bs(S[:, k], taus[k], sig, obs)
                dx = np.column_stack([S[:, k] - x_ref[k, 0], h - D - x_ref[k, 1], w - V - x_ref[k, 2]])
                u = D + v_ff[k] + dx @ K[k]
                w = w - c * np.abs(u - h) * S[:, k] + u * (S[:, k + 1] - S[:, k])
                h = u
            Vend = black_scholes(S[:, H], obs.strike, taus[H], sig, obs.rate, obs.kind)[0]
            P = w - Vend - (c * np.abs(h) * S[:, H] if at_expiry else 0.0)
            return -P.mean() + 0.5 * lam * P.var()

        # Per step k: iLQG's quadratic model of the cost-to-go in v WITHOUT that step's smoothed trading cost
        # (q_u, q_uu, q_ux, kappa), so a decision can add the EXACT cost kappa * |v - z| instead (soft threshold).
        local = [None] * H

        def backward(xs, vs, mu):
            # terminal cost l = -P + lam/2 (P - P_bar)^2 at P = P_bar: l_x = -P_x, l_xx = lam P_x P_x^T - P_xx
            Px, Pxx = grad_hess(P_of, xs[H])
            Vx, Vxx = -Px, lam * np.outer(Px, Px) - Pxx
            kff, KK = np.empty(H), np.empty((H, 3))
            for k in reversed(range(H)):
                x, v = xs[k], vs[k]
                _, F, f_x, f_u, F_x, F_u = linearize(x, v, k)                              # F = noise loading
                d = v - x[1]
                A2 = eps**2 / (d**2 + eps**2) ** 1.5                                       # curvature of |.|_eps
                ce = c * A2 * x[0]                                                         # e' = ... - c|v - z| y
                fe_xx = np.array([[0, 0, 0], [0, -ce, 0], [0, 0, 0]])
                fe_ux, fe_uu = np.array([0, ce, 0]), -ce
                VF = Vxx @ F
                Q_x = f_x.T @ Vx + F_x.T @ VF
                Q_u = f_u @ Vx + F_u @ VF
                Q_xx = f_x.T @ Vxx @ f_x + F_x.T @ Vxx @ F_x + Vx[2] * fe_xx
                Q_uu = f_u @ Vxx @ f_u + F_u @ Vxx @ F_u + Vx[2] * fe_uu + mu
                Q_ux = f_u @ Vxx @ f_x + F_u @ Vxx @ F_x + Vx[2] * fe_ux
                if Q_uu <= 0:
                    return None
                kap = -Vx[2] * c * x[0]                                                    # smoothed cost = kap |d|_eps
                local[k] = (Q_u - kap * d / np.sqrt(d**2 + eps**2), Q_uu - mu - kap * A2,
                            Q_ux + kap * A2 * np.array([0.0, 1.0, 0.0]), kap)
                kff[k], KK[k] = -Q_u / Q_uu, -Q_ux / Q_uu
                Vx = Q_x + KK[k] * Q_uu * kff[k] + KK[k] * Q_u + Q_ux * kff[k]
                Vxx = Q_xx + Q_uu * np.outer(KK[k], KK[k]) + np.outer(KK[k], Q_ux) + np.outer(Q_ux, KK[k])
                Vxx = 0.5 * (Vxx + Vxx.T)
            return kff, KK

        # initial law: warm start from the previous plan (shifted), else "hold delta"; no feedback yet
        v_ff = np.zeros(H)
        if self.law is not None and len(self.law[0]) > 1:
            v_ff = np.append(self.law[0][1:], self.law[0][-1])[:H]
        K = np.zeros((H, 3))
        xs, vs = nominal(v_ff, K, np.zeros((H + 1, 3)))
        J, mu = score(vs, K, xs), 1e-6
        for _ in range(p.get("max_iter", 30)):
            out = backward(xs, vs, mu)
            if out is None:
                mu *= 10
                if mu > 1e6:
                    break
                continue
            kff, K_new = out
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
                cand = vs + alpha * kff
                J_new = score(cand, K_new, xs)
                if J_new < J:
                    break
            else:
                mu *= 10
                if mu > 1e6:
                    break
                continue
            improvement = J - J_new
            xs, vs = nominal(cand, K_new, xs)
            K, J, mu = K_new, J_new, max(mu / 10, 1e-8)
            if improvement < p.get("tol", 1e-7) * max(1.0, abs(J)):
                break
        if p.get("exact_first_step", True) and backward(xs, vs, 0.0) is not None:
            # Today's trade with the EXACT cost c|v - z| y: take iLQG's quadratic model of the cost-to-go in v,
            # remove the smoothed cost, add the exact one, and minimize in closed form (soft thresholding):
            # don't trade if the marginal benefit at the current position is below the marginal cost.
            q_u, q_uu, _, kap = local[0]                                  # model without today's (smoothed) cost
            if q_uu > 0:
                vs = vs.copy()
                vs[0] = soft_threshold(vs[0] - q_u / q_uu, x0[1], kap / q_uu)
        return vs, xs, K, sig, obs.step, obs.tau, local


def soft_threshold(v_free, z_now, thr):
    """argmin_v  q_uu/2 (v - v_free)^2 + kappa |v - z_now|  with thr = kappa / q_uu: stay at z_now (no trade)
    if the unconstrained target is within thr of it, else stop thr short of the target."""
    return z_now if abs(v_free - z_now) <= thr else v_free - np.sign(v_free - z_now) * thr
