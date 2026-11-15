"""Black-Scholes: identities every price/delta/gamma must satisfy."""

import math

import pytest

from backtest import black_scholes

CASES = [(100, 100, 0.25, 0.2, 0.03), (90, 100, 0.1, 0.35, 0.0), (120, 100, 1.0, 0.15, 0.05)]


@pytest.mark.parametrize("spot,strike,tau,vol,rate", CASES)
def test_put_call_parity(spot, strike, tau, vol, rate):
    call = black_scholes(spot, strike, tau, vol, rate, "call")[0]
    put = black_scholes(spot, strike, tau, vol, rate, "put")[0]
    assert call - put == pytest.approx(spot - strike * math.exp(-rate * tau), abs=1e-10)


@pytest.mark.parametrize("spot,strike,tau,vol,rate", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_and_gamma_are_derivatives_of_price(spot, strike, tau, vol, rate, kind):
    h = 1e-3
    price = lambda s: black_scholes(s, strike, tau, vol, rate, kind)[0]
    _, delta, gamma = black_scholes(spot, strike, tau, vol, rate, kind)
    assert delta == pytest.approx((price(spot + h) - price(spot - h)) / (2 * h), abs=1e-6)
    assert gamma == pytest.approx((price(spot + h) - 2 * price(spot) + price(spot - h)) / h**2, rel=1e-4)


def test_expiry_returns_payoff():
    assert black_scholes(110, 100, 0.0, 0.2, 0.03, "call") == (10.0, 1.0, 0.0)
    assert black_scholes(110, 100, 0.0, 0.2, 0.03, "put") == (0.0, 0.0, 0.0)
