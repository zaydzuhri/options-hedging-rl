"""
The strategy interface: what a hedging strategy SEES (Observation) and what it DOES (act).
Also the Black-Scholes formula, since both the backtester and some strategies need it.

Setup (same for every strategy):
  - At step 0 we SELL one option on one share and receive its premium in cash.
  - At every step t = 0 .. N-1 the strategy picks how many shares of the underlying
    to hold until the next step (the "hedge position"). Trading costs money.
  - At step N the option expires: we pay its payoff and sell off whatever shares we hold.
  - Final P&L = premium + gains/losses on the shares - option payoff - trading costs.
    A perfect hedge ends at P&L ~ 0 every single time; risk = how spread out P&L is.

To write a new strategy: subclass Strategy, implement act(), register it in
strategies/__init__.py. That's it.
"""

from dataclasses import dataclass, astuple, fields

import numpy as np
from scipy.special import ndtr  # standard normal CDF N(x); works on whole arrays at once


def black_scholes(spot, strike, tau, vol, rate, kind="call"):
    """Price, delta and gamma of a European option. `spot` can be a number or a numpy array
    (e.g. many simulated scenarios at once). tau = years to maturity; at tau == 0 this
    returns the payoff (and a 0/1 delta)."""
    spot = np.asarray(spot, dtype=float)
    if tau <= 0 or vol <= 0:
        if kind == "call":
            return np.maximum(spot - strike, 0.0), 1.0 * (spot > strike), 0.0 * spot
        return np.maximum(strike - spot, 0.0), -1.0 * (spot < strike), 0.0 * spot
    sq = vol * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * tau) / sq
    d2 = d1 - sq
    disc = np.exp(-rate * tau)
    if kind == "call":
        price = spot * ndtr(d1) - strike * disc * ndtr(d2)
        delta = ndtr(d1)
    else:
        price = strike * disc * ndtr(-d2) - spot * ndtr(-d1)
        delta = ndtr(d1) - 1.0
    gamma = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) / (spot * sq)
    return price, delta, gamma


@dataclass(frozen=True)
class Observation:
    """Everything a strategy is allowed to know at the current step (no peeking ahead)."""

    step: int             # current step index, 0 .. n_steps-1
    steps_left: int       # steps until the option expires
    tau: float            # time to maturity in years
    spot: float           # underlying price (each episode is rescaled so it starts at 100)
    strike: float         # option strike price
    is_call: bool         # True = we sold a call, False = we sold a put
    vol: float            # volatility estimate used for pricing (implied or realized), annualized
    rate: float           # risk-free interest rate, annualized
    cost_rate: float      # trading cost per $ traded (cost_bps / 10000)
    option_price: float   # Black-Scholes value of the option right now
    delta: float          # Black-Scholes delta = shares to hold to be locally hedged
    gamma: float          # how fast delta changes when spot moves
    position: float       # shares currently held (before this step's trade)
    cash: float           # cash account (premium received +/- trades, with interest)
    pnl: float            # mark-to-market P&L so far: cash + position*spot - option_price

    @property
    def kind(self) -> str:
        return "call" if self.is_call else "put"

    def to_array(self) -> np.ndarray:
        """Flat float vector, in field order (handy as a neural-net input later)."""
        return np.array(astuple(self), dtype=np.float64)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


class Strategy:
    """Base class. A strategy maps an Observation to a target hedge position."""

    def __init__(self, **params):
        self.params = params

    def reset(self) -> None:
        """Called at the start of every episode. Clear any per-episode memory here."""

    def act(self, obs: Observation) -> float:
        """Return the number of shares to hold from now until the next step.
        Returning obs.position means "don't trade"."""
        raise NotImplementedError
