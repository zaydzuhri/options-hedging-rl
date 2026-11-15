"""Dynamic programming (DP) hedge: the optimal policy for a price model, computed on a grid.

Objective: exponential utility of the final P&L, i.e. minimize E[exp(-risk_aversion * P&L)].
Its certainty equivalent CE = -log E[exp(-g P&L)] / g is ~ mean - (g/2) * variance for small risk,
so it trades off "high average" against "tight". Unlike variance, it is time-consistent, so DP applies;
and the wealth so far drops out of the problem, so the state is only (time, price [, variance], shares).

  State      t, x = log S (and v = model variance for GJR-GARCH), h = shares held before trading
  Value      psi_t(x, h) = log E[exp(-g * (P&L from t to expiry))]     (lower = better)
  Terminal   psi_N(x, h) = g * (payoff(S) + c |h| S)                    (pay the payoff, sell the shares)
  Bellman    G_t(x, h')  = log E[exp(-g h' (S' - S) + psi_{t+1}(x', h'))]    hold h' over one step
             psi_t(x, h) = min_{h'}  g c S |h' - h|  +  G_t(x, h')            trade from h to h'

G is convex in h', so the minimizer is a NO-TRADE BAND [lo, hi]:
    lo = argmin G(h') + g c S h',  hi = argmin G(h') - g c S h',  optimal action = clip(h, lo, hi)
(buy up to lo if below it, sell down to hi if above it, otherwise don't trade). The tables lo/hi are
solved backward from expiry once, then each step is a lookup. The expectation over the next price
is Gauss-Hermite quadrature, with linear interpolation of psi_{t+1} between grid points.

Not modelled: interest on cash (exact when rate = 0).
"""

import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import logsumexp

from strategy import Observation, Strategy
from strategies.models import GJRGarch, make_model


def _band(G, a, h):
    """For each row r: lo = argmin_h' G[r] + a[r] h', hi = argmin G[r] - a[r] h' (refined between
    grid points with a parabola), and the new value psi(h) = G(clip(h, lo, hi)) + a |clip - h|."""
    dh = h[1] - h[0]

    def argmin(F):
        k = np.clip(np.argmin(F, axis=1), 1, F.shape[1] - 2)
        r = np.arange(F.shape[0])
        f0, f1, f2 = F[r, k - 1], F[r, k], F[r, k + 1]
        den = f0 - 2 * f1 + f2
        off = np.where(den > 1e-14, 0.5 * (f0 - f2) / np.where(den > 1e-14, den, 1.0), 0.0)
        return h[0] + (k + np.clip(off, -0.5, 0.5)) * dh

    lo = argmin(G + a[:, None] * h[None, :])
    hi = argmin(G - a[:, None] * h[None, :])
    hi = np.maximum(hi, lo)
    hc = np.clip(h[None, :], lo[:, None], hi[:, None])
    fi = np.clip((hc - h[0]) / dh, 0, len(h) - 1 - 1e-9)                  # linear interpolation of G at hc
    i0 = fi.astype(int)
    f = fi - i0
    rows = np.arange(G.shape[0])[:, None]
    G_hc = G[rows, i0] * (1 - f) + G[rows, np.minimum(i0 + 1, len(h) - 1)] * f
    return lo, hi, G_hc + a[:, None] * np.abs(hc - h[None, :])


def _interp_rows(table, fi):
    """Linear interpolation of table (n, ...) at fractional row indices fi (any shape)."""
    fi = np.clip(fi, 0, table.shape[0] - 1 - 1e-9)
    i0 = fi.astype(int)
    f = fi - i0
    f = f.reshape(f.shape + (1,) * (table.ndim - 1))
    return table[i0] * (1 - f) + table[i0 + 1] * f


class DPHedge(Strategy):
    """params:
    risk_aversion (float, default 1.0): g, per $ of P&L ($ per option on $100 of stock). Must keep
                                        E[exp(-g P&L)] finite under the model: with GJR-GARCH a big drop
                                        makes the next variance explode, two-step losses act like a product
                                        of normals, and large g makes the expectation infinite (the solution
                                        then degenerates). For the paper's monthly setting use g <~ 0.3.
    model (dict, default gbm):          price model to optimize for (see strategies/models.py)
    vol_grid (list, optional):          GBM only: solve for each of these vols and interpolate by
                                        the current obs.vol (needed when vol changes, e.g. implied vol).
                                        Default: solve for the vol at the start of each episode (cached).
    n_x (201), x_width (6.0):           price grid size and half-width in standard deviations of log S
    n_h (401), h_range:                 share grid (default [-0.1, 1.1] for calls, [-1.1, 0.1] for puts)
    n_quad (32):                        quadrature points for the next-step expectation
    n_v (40), v_range:                  GJR-GARCH only: variance grid (log-spaced)
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.model = make_model(params.get("model"))
        self.tables = {}  # solved once per (setting, vol), reused across episodes

    def reset(self):
        self.model.reset()
        self.episode_vol = None

    # ---------------------------------------------------------------- acting
    def act(self, obs: Observation) -> float:
        self.model.observe(obs)
        dt = obs.tau / obs.steps_left
        N = obs.step + obs.steps_left
        if isinstance(self.model, GJRGarch):
            x, logv, lo_t, hi_t = self._garch_tables(obs, dt, N)
            lo, hi = self._lookup2(x, logv, lo_t[obs.step], hi_t[obs.step], math.log(obs.spot), math.log(self.model.v))
        else:
            lo, hi = self._gbm_band(obs, dt, N)
        return float(np.clip(obs.position, lo, hi))

    def _gbm_band(self, obs, dt, N):
        x_now = math.log(obs.spot)
        grid = self.params.get("vol_grid")
        if not grid:
            if self.episode_vol is None:
                self.episode_vol = obs.vol
            sig = self.episode_vol
            x, lo_t, hi_t = self._gbm_tables(obs, dt, N, sig)
            return np.interp(x_now, x, lo_t[obs.step]), np.interp(x_now, x, hi_t[obs.step])
        grid = np.asarray(grid, dtype=float)
        s = float(np.clip(obs.vol, grid[0], grid[-1]))
        j = min(int(np.searchsorted(grid, s, side="right")) - 1, len(grid) - 2)
        w = (s - grid[j]) / (grid[j + 1] - grid[j])
        out = []
        for sig in (grid[j], grid[j + 1]):
            x, lo_t, hi_t = self._gbm_tables(obs, dt, N, sig)
            out.append((np.interp(x_now, x, lo_t[obs.step]), np.interp(x_now, x, hi_t[obs.step])))
        return (1 - w) * out[0][0] + w * out[1][0], (1 - w) * out[0][1] + w * out[1][1]

    @staticmethod
    def _lookup2(x, logv, lo_t, hi_t, xq, vq):
        fi = np.clip((xq - x[0]) / (x[1] - x[0]), 0, len(x) - 1 - 1e-9)
        fj = np.clip((vq - logv[0]) / (logv[1] - logv[0]), 0, len(logv) - 1 - 1e-9)
        i, j = int(fi), int(fj)
        a, b = fi - i, fj - j
        bil = lambda T: ((1 - a) * (1 - b) * T[i, j] + a * (1 - b) * T[i + 1, j]
                         + (1 - a) * b * T[i, j + 1] + a * b * T[i + 1, j + 1])
        return bil(lo_t), bil(hi_t)

    # ---------------------------------------------------------------- solving
    def _common(self, obs):
        p = self.params
        g = p.get("risk_aversion", 1.0)
        h_lo, h_hi = p.get("h_range", (-0.1, 1.1) if obs.is_call else (-1.1, 0.1))
        h = np.linspace(h_lo, h_hi, p.get("n_h", 401))
        z, wq = hermegauss(p.get("n_quad", 32))
        logw = np.log(wq / math.sqrt(2 * math.pi))
        payoff = (lambda S: np.maximum(S - obs.strike, 0.0)) if obs.is_call else (lambda S: np.maximum(obs.strike - S, 0.0))
        return g, h, z, logw, payoff

    def _gbm_tables(self, obs, dt, N, sig):
        key = ("gbm", N, dt, obs.strike, obs.is_call, obs.cost_rate, round(sig, 10))
        if key in self.tables:
            return self.tables[key]
        p = self.params
        g, h, z, logw, payoff = self._common(obs)
        mu = self.model.drift(obs)
        half = p.get("x_width", 6.0) * max(sig, 0.05) * math.sqrt(N * dt)
        x = np.linspace(math.log(obs.strike) - half, math.log(obs.strike) + half, p.get("n_x", 201))
        S, dx, c = np.exp(x), x[1] - x[0], obs.cost_rate
        step = (mu - 0.5 * sig**2) * dt + sig * math.sqrt(dt) * z            # log-price move at each quadrature node
        dS = S[None, :] * np.expm1(step)[:, None]                             # (q, x) price change
        psi = g * (payoff(S)[:, None] + c * np.abs(h)[None, :] * S[:, None])  # value at expiry
        lo_t, hi_t = np.empty((N, len(x))), np.empty((N, len(x)))
        a = g * c * S
        for t in reversed(range(N)):
            nxt = _interp_rows(psi, (x[None, :] + step[:, None] - x[0]) / dx)        # (q, x, h)
            G = logsumexp(logw[:, None, None] - g * dS[:, :, None] * h[None, None, :] + nxt, axis=0)
            lo_t[t], hi_t[t], psi = _band(G, a, h)
        self.tables[key] = (x, lo_t, hi_t)
        return self.tables[key]

    def _garch_tables(self, obs, dt, N):
        key = ("garch", N, obs.strike, obs.is_call, obs.cost_rate)
        if key in self.tables:
            return self.tables[key]
        p, m = self.params, self.model
        g, h, z, logw, payoff = self._common(obs)
        v_lo, v_hi = p.get("v_range", (0.5 * m.omega / (1 - m.beta), 25 * m.h0))
        logv = np.linspace(math.log(v_lo), math.log(v_hi), p.get("n_v", 40))
        V = np.exp(logv)
        half = p.get("x_width", 6.0) * math.sqrt(2 * N * m.h0)
        x = np.linspace(math.log(obs.strike) - half, math.log(obs.strike) + half, p.get("n_x", 121))
        S, dx, c = np.exp(x), x[1] - x[0], obs.cost_rate
        a = g * c * S
        psi = np.repeat((g * (payoff(S)[:, None] + c * np.abs(h)[None, :] * S[:, None]))[:, None, :], len(V), axis=1)
        lo_t, hi_t = np.empty((N, len(x), len(V))), np.empty((N, len(x), len(V)))
        for t in reversed(range(N)):
            new = np.empty_like(psi)
            for j, v in enumerate(V):
                eps = math.sqrt(v) * z                                            # (q,) return shock
                step = m.mu + eps
                fv = (np.log(m.next_var(v, eps)) - logv[0]) / (logv[1] - logv[0])  # (q,) next-variance index
                fv = np.clip(fv, 0, len(V) - 1 - 1e-9)
                j0 = fv.astype(int)
                b = (fv - j0)[:, None, None]
                fi = np.clip((x[None, :] + step[:, None] - x[0]) / dx, 0, len(x) - 1 - 1e-9)   # (q, x)
                i0 = fi.astype(int)
                a_ = (fi - i0)[:, :, None]
                J0, J1 = j0[:, None], j0[:, None] + 1
                nxt = ((1 - a_) * (1 - b) * psi[i0, J0] + a_ * (1 - b) * psi[i0 + 1, J0]
                       + (1 - a_) * b * psi[i0, J1] + a_ * b * psi[i0 + 1, J1])           # (q, x, h)
                dS = S[None, :] * np.expm1(step)[:, None]
                G = logsumexp(logw[:, None, None] - g * dS[:, :, None] * h[None, None, :] + nxt, axis=0)
                lo_t[t, :, j], hi_t[t, :, j], new[:, j, :] = _band(G, a, h)
            psi = new
        self.tables[key] = (x, logv, lo_t, hi_t)
        return self.tables[key]
