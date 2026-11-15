"""GJR-GARCH: fitting data simulated from known parameters recovers them."""

import numpy as np
import pandas as pd
import pytest

import garch

TRUE = dict(mu=0.03, omega=0.03, alpha=0.02, gamma=0.15, beta=0.88, vol_intercept=0.05, vol_slope=0.9)


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    model = {**TRUE, "resid": rng.standard_normal(200_000), "start_var": np.array([1.0])}
    ep = garch.simulate(model, n_paths=1, n_steps=6000, seed=1)[0]
    df = pd.DataFrame({"spot": ep.spots, "vol": ep.vols})
    fitted = garch.fit(df)
    assert fitted["beta"] == pytest.approx(TRUE["beta"], abs=0.04)
    assert fitted["gamma"] == pytest.approx(TRUE["gamma"], abs=0.06)
    assert fitted["persistence"] == pytest.approx(TRUE["alpha"] + TRUE["gamma"] / 2 + TRUE["beta"], abs=0.02)
    assert fitted["vol_slope"] == pytest.approx(TRUE["vol_slope"], abs=0.1)
    assert np.std(fitted["resid"]) == pytest.approx(1.0, abs=0.05)


def test_simulated_paths_start_at_100_and_have_positive_vol():
    model = {**TRUE, "resid": np.random.default_rng(0).standard_normal(1000), "start_var": np.array([0.5, 2.0])}
    eps = garch.simulate(model, n_paths=50, n_steps=21, seed=0)
    assert all(e.spots[0] == 100.0 and len(e.spots) == 22 and e.vols.min() > 0 for e in eps)
