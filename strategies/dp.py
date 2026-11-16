"""Dynamic programming (DP) hedge: the optimal policy for a price model, computed on a grid.

Objective: entropic risk of the final P&L, (1/g) log E[exp(-g * P&L)] = -CE, i.e. exponential utility
with risk aversion g. Its certainty equivalent CE ~ mean - (g/2) * variance for small risk, so it trades
off "high average" against "tight". Unlike variance it is time-consistent, so DP applies; and the cash you
already hold drops out, so the state is only (time, price [, variance], shares held).

Money is counted in time-N dollars (G = e^{r dt}): cash grows by G each step, so a cost paid at step t
weighs G^(N-t), and holding h' shares from t to t+1 earns h' (S' - G S) G^(N-t-1).

  State      t, x = log S (and v = model variance for GJR-GARCH), h = shares held before trading
  Value      psi_t(x, h) = log E[exp(-g * (P&L from t to expiry, without the cash already held))]
  Terminal   psi_N(x, h) = g * (payoff(S) + c |h| S)                              pay the payoff, sell the shares
  Bellman    G_t(x, h')  = log E[exp(-g h' G^(N-t-1) (S' - G S) + psi_{t+1}(x', h'))]      hold h' for a step
             psi_t(x, h) = min_{h'}  g c S G^(N-t) |h' - h|  +  G_t(x, h')                 trade from h to h'
  Risk       (1/g) log E[exp(-g P&L)] = -G^N premium + psi_0(x_0, 0) / g

G is convex in h', so the minimizer is a NO-TRADE BAND [lo, hi]:
    lo = argmin G(h') + a h',  hi = argmin G(h') - a h'  (a = g c S G^(N-t)),  optimal action = clip(h, lo, hi)
(buy up to lo if below it, sell down to hi if above it, otherwise don't trade). `check=True` verifies this
by brute force from every grid position. The tables are solved backward from expiry once, then each step
is a lookup. Under GBM the log-price is a Markov chain on the grid (exact normal cell probabilities); under
GJR-GARCH the expectation is Gauss-Hermite quadrature with linear interpolation between grid points. Band
edges are refined between grid points with a parabola.
"""

import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import logsumexp, ndtr

from strategy import Observation, Strategy
from strategies.models import GJRGarch, make_model


def _band(G, a, h, check=False):
    """For each row r: lo = argmin_h' G[r] + a[r] h', hi = argmin G[r] - a[r] h' (refined between grid
    points with a parabola), and the new value psi(h) = G(clip(h, lo, hi)) + a |clip - h|.
    With check, also the largest gap (in grid steps) between clip(h, lo, hi) and the brute-force argmin."""
    dh, n = h[1] - h[0], len(h)

    def argmin(F):
        k = np.argmin(F, axis=1)
        inner = (k > 0) & (k < n - 1)                       # at a grid end (e.g. a position limit): exact
        kc = np.clip(k, 1, n - 2)
        r = np.arange(F.shape[0])
        f0, f1, f2 = F[r, kc - 1], F[r, kc], F[r, kc + 1]
        den = f0 - 2 * f1 + f2
        off = np.where(den > 1e-14, 0.5 * (f0 - f2) / np.where(den > 1e-14, den, 1.0), 0.0)
        return h[0] + np.where(inner, kc + np.clip(off, -0.5, 0.5), k) * dh

    lo = argmin(G + a[:, None] * h[None, :])
    hi = np.maximum(argmin(G - a[:, None] * h[None, :]), lo)
    hc = np.clip(h[None, :], lo[:, None], hi[:, None])
    fi = np.clip((hc - h[0]) / dh, 0, n - 1 - 1e-9)                   # linear interpolation of G at hc
    i0 = fi.astype(int)
    f = fi - i0
    rows = np.arange(G.shape[0])[:, None]
    G_hc = G[rows, i0] * (1 - f) + G[rows, np.minimum(i0 + 1, n - 1)] * f
    mismatch = 0.0
    if check:   # every (row, h, h') at once would be large: one row block at a time
        for blk in np.array_split(np.arange(G.shape[0]), max(1, G.shape[0] // 16)):
            total = G[blk, None, :] + a[blk, None, None] * np.abs(h[None, None, :] - h[None, :, None])
            best = h[np.argmin(total, axis=-1)]                          # (rows, h): brute-force best h'
            mismatch = max(mismatch, float(np.abs(best - hc[blk]).max() / dh))
    return lo, hi, G_hc + a[:, None] * np.abs(hc - h[None, :]), mismatch


def _interp_rows(table, fi):
    """Linear interpolation of table (n, ...) at fractional row indices fi (any shape)."""
    fi = np.clip(fi, 0, table.shape[0] - 1 - 1e-9)
    i0 = fi.astype(int)
    f = fi - i0
    f = f.reshape(f.shape + (1,) * (table.ndim - 1))
    return table[i0] * (1 - f) + table[i0 + 1] * f


def h_grid(is_call, h_range=None, n_h=401):
    """Share grid: h_range, or [-0.1, 1.1] for a call ([-1.1, 0.1] for a put). Use the position
    limits as h_range when the environment has them (e.g. [0, 1])."""
    lo, hi = h_range if h_range is not None else ((-0.1, 1.1) if is_call else (-1.1, 0.1))
    return np.linspace(lo, hi, n_h)


def solve_gbm(strike, is_call, N, dt, sigma, mu, rate, cost_rate, risk_aversion, h,
              n_x=801, x_width=6.0, check=False) -> dict:
    """Backward induction under GBM (drift mu, vol sigma). The log-price follows a Markov chain on a uniform
    grid matched to the GBM increments: each cell gets the exact normal probability of its interval (the end
    cells take the tails), so expectations are a matrix product and refining the grid converges smoothly.
    Returns the log-price grid x, the band edges lo/hi (N, n_x), psi_0 (n_x, n_h) and, with check, the
    largest band mismatch in grid steps."""
    g, c, Gr = risk_aversion, cost_rate, math.exp(rate * dt)
    half = x_width * max(sigma, 0.05) * math.sqrt(N * dt)
    x = np.linspace(math.log(strike) - half, math.log(strike) + half, n_x | 1)   # odd: the strike is on the grid
    S = np.exp(x)
    edges = np.concatenate([[-np.inf], (x[:-1] + x[1:]) / 2, [np.inf]])
    m, s = (mu - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt)
    P = np.diff(ndtr((edges[None, :] - x[:, None] - m) / s), axis=1)        # P[i, j] = Pr(x_i -> x_j)
    payoff = np.maximum(S - strike, 0.0) if is_call else np.maximum(strike - S, 0.0)
    psi = g * (payoff[:, None] + c * np.abs(h)[None, :] * S[:, None])         # value at expiry
    lo_t, hi_t, mismatch = np.empty((N, len(x))), np.empty((N, len(x))), 0.0
    for t in reversed(range(N)):
        k = g * Gr ** (N - t - 1) * h[None, :]                               # exponent is -k (S' - G S)
        A = -k * (S[:, None] - strike) + psi                                  # centred at the strike: keeps exp in range
        top = A.max(axis=0)
        Gt = np.log(P @ np.exp(A - top)) + top + k * (Gr * S[:, None] - strike)
        lo_t[t], hi_t[t], psi, mm = _band(Gt, g * c * S * Gr ** (N - t), h, check)
        mismatch = max(mismatch, mm)
    return dict(x=x, h=h, lo=lo_t, hi=hi_t, psi0=psi, mismatch=mismatch, risk_aversion=g, growth=Gr ** N)


def optimal_risk(sol, spot, premium, position=0.0) -> float:
    """The optimal entropic risk (1/g) log E[exp(-g P&L)] (= -CE) of selling at `premium` and hedging
    optimally from `position` shares, read off a solve_gbm solution."""
    x, h, psi0 = sol["x"], sol["h"], sol["psi0"]
    row = np.array([np.interp(position, h, psi0[i]) for i in range(len(x))])
    return float(-sol["growth"] * premium + np.interp(math.log(spot), x, row) / sol["risk_aversion"])


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
    h_range, n_h (401):                 share grid; set h_range to the position limits if there are any
                                        (default [-0.1, 1.1] for calls, [-1.1, 0.1] for puts)
    n_x (801; 121 for GJR-GARCH), x_width (6.0): price grid size and half-width in standard deviations of log S
    n_quad (32), n_v (40), v_range:     GJR-GARCH only: quadrature points for the next-step expectation,
                                        variance grid (log-spaced)
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
            sol = self._gbm_tables(obs, dt, N, self.episode_vol)
            return np.interp(x_now, sol["x"], sol["lo"][obs.step]), np.interp(x_now, sol["x"], sol["hi"][obs.step])
        grid = np.asarray(grid, dtype=float)
        s = float(np.clip(obs.vol, grid[0], grid[-1]))
        j = min(int(np.searchsorted(grid, s, side="right")) - 1, len(grid) - 2)
        w = (s - grid[j]) / (grid[j + 1] - grid[j])
        out = []
        for sig in (grid[j], grid[j + 1]):
            sol = self._gbm_tables(obs, dt, N, sig)
            out.append((np.interp(x_now, sol["x"], sol["lo"][obs.step]), np.interp(x_now, sol["x"], sol["hi"][obs.step])))
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
    def _h(self, obs):
        return h_grid(obs.is_call, self.params.get("h_range"), self.params.get("n_h", 401))

    def _gbm_tables(self, obs, dt, N, sig):
        key = ("gbm", N, dt, obs.strike, obs.is_call, obs.cost_rate, obs.rate, round(sig, 10))
        if key not in self.tables:
            p = self.params
            self.tables[key] = solve_gbm(obs.strike, obs.is_call, N, dt, sig, self.model.drift(obs), obs.rate,
                                         obs.cost_rate, p.get("risk_aversion", 1.0), self._h(obs),
                                         p.get("n_x", 801), p.get("x_width", 6.0))
        return self.tables[key]

    def _garch_tables(self, obs, dt, N):
        key = ("garch", N, obs.strike, obs.is_call, obs.cost_rate, obs.rate)
        if key in self.tables:
            return self.tables[key]
        p, m = self.params, self.model
        g, h, c, Gr = p.get("risk_aversion", 1.0), self._h(obs), obs.cost_rate, math.exp(obs.rate * dt)
        z, wq = hermegauss(p.get("n_quad", 32))
        logw = np.log(wq / math.sqrt(2 * math.pi))
        v_lo, v_hi = p.get("v_range", (0.5 * m.omega / (1 - m.beta), 25 * m.h0))
        logv = np.linspace(math.log(v_lo), math.log(v_hi), p.get("n_v", 40))
        V = np.exp(logv)
        half = p.get("x_width", 6.0) * math.sqrt(2 * N * m.h0)
        x = np.linspace(math.log(obs.strike) - half, math.log(obs.strike) + half, p.get("n_x", 121))
        S, dx = np.exp(x), x[1] - x[0]
        payoff = np.maximum(S - obs.strike, 0.0) if obs.is_call else np.maximum(obs.strike - S, 0.0)
        psi = np.repeat((g * (payoff[:, None] + c * np.abs(h)[None, :] * S[:, None]))[:, None, :], len(V), axis=1)
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
                gain = S[None, :] * (np.exp(step)[:, None] - Gr)
                Gt = logsumexp(logw[:, None, None] - g * Gr ** (N - t - 1) * gain[:, :, None] * h[None, None, :] + nxt,
                               axis=0)
                lo_t[t, :, j], hi_t[t, :, j], new[:, j, :], _ = _band(Gt, g * c * S * Gr ** (N - t), h)
            psi = new
        self.tables[key] = (x, logv, lo_t, hi_t)
        return self.tables[key]
