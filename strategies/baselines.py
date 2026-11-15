"""Classical, non-learned hedging strategies."""

import math

from strategy import Observation, Strategy


class NoHedge(Strategy):
    """Never trade. Shows how much risk the naked option carries."""

    def act(self, obs: Observation) -> float:
        return 0.0


class DeltaHedge(Strategy):
    """Classic Black-Scholes delta hedge: hold `delta` shares, rebalancing every k steps.

    params:
      rebalance_every (int, default 1): trade only on every k-th step.
    """

    def act(self, obs: Observation) -> float:
        k = self.params.get("rebalance_every", 1)
        return obs.delta if obs.step % k == 0 else obs.position


class DeltaBand(Strategy):
    """Delta hedge with a no-trade band: only trade when the hedge has drifted more
    than `band` shares away from delta. Saves costs by ignoring small mismatches.

    params:
      band (float, default 0.05): allowed |delta - position| before trading.
      to_edge (bool, default False): if True, trade only back to the edge of the band
        (the cheapest fix) instead of all the way to delta.
    """

    def act(self, obs: Observation) -> float:
        band = self.params.get("band", 0.05)
        gap = obs.delta - obs.position
        if abs(gap) <= band:
            return obs.position
        if self.params.get("to_edge", False):
            return obs.delta - band if gap > 0 else obs.delta + band
        return obs.delta


class WhalleyWilmott(Strategy):
    """Asymptotically optimal no-trade band under proportional costs (Whalley & Wilmott, 1997).
    Hold the current position while it is inside

        delta +/- (3/2 * exp(-rate*tau) * cost_rate * spot * gamma^2 / risk_aversion)^(1/3)

    and otherwise trade to the nearest edge of the band. The band is wide where gamma is big
    (delta moves a lot, so chasing it is expensive) and narrows as risk aversion grows.
    With zero costs it collapses to the daily delta hedge.

    params:
      risk_aversion (float, default 1.0): exponential-utility risk aversion, per $ of P&L.
    """

    def act(self, obs: Observation) -> float:
        risk_aversion = self.params.get("risk_aversion", 1.0)
        half = (1.5 * math.exp(-obs.rate * obs.tau) * obs.cost_rate * obs.spot * obs.gamma**2
                / risk_aversion) ** (1 / 3)
        return min(max(obs.position, obs.delta - half), obs.delta + half)
