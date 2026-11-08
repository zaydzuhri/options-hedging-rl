"""
The strategy interface: what a hedging strategy SEES (Observation) and what it DOES (act).

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


@dataclass(frozen=True)
class Observation:
    """Everything a strategy is allowed to know at the current step (no peeking ahead)."""

    step: int             # current step index, 0 .. n_steps-1
    steps_left: int       # steps until the option expires
    tau: float            # time to maturity in years
    spot: float           # underlying price (each episode is rescaled so it starts at 100)
    strike: float         # option strike price
    vol: float            # volatility estimate used for pricing (implied or realized), annualized
    rate: float           # risk-free interest rate, annualized
    option_price: float   # Black-Scholes value of the option right now
    delta: float          # Black-Scholes delta = shares to hold to be locally hedged
    gamma: float          # how fast delta changes when spot moves
    position: float       # shares currently held (before this step's trade)
    cash: float           # cash account (premium received +/- trades, with interest)
    pnl: float            # mark-to-market P&L so far: cash + position*spot - option_price

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
