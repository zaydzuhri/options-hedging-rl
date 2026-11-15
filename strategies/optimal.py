"""The exact optimal policy from dp_optimal.py (Black-Scholes world only)."""

import math

import numpy as np

from strategy import Observation, Strategy


class DPOptimal(Strategy):
    """No-trade band [lower(t, S), upper(t, S)] solved by dynamic programming: keep the position
    if it is inside, otherwise trade to the nearest edge. Edges are interpolated in log-spot.

    params:
      model (str): the .npz file `uv run dp_optimal.py --config <config>` writes.
      The other params (risk_aversion, n_spot, n_pos, width) are only read by the solver.
    """

    def __init__(self, **params):
        super().__init__(**params)
        sol = np.load(params["model"])
        self.log_spots, self.lower, self.upper = np.log(sol["spots"]), sol["lower"], sol["upper"]

    def act(self, obs: Observation) -> float:
        x = math.log(obs.spot)
        lo = float(np.interp(x, self.log_spots, self.lower[obs.step]))
        hi = float(np.interp(x, self.log_spots, self.upper[obs.step]))
        return min(max(obs.position, lo), hi)
