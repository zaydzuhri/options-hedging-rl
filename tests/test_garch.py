"""GJR-GARCH: fitting data simulated from known parameters recovers them."""

import numpy as np
import pytest

from strategies.models import fit_gjr_garch, simulate_fitted_gjr_garch

# per-day parameters in decimal units (mu 0.03% a day, ...)
TRUE = dict(mu=3e-4, omega=3e-6, alpha=0.02, gamma=0.15, beta=0.88, vol_intercept=0.05, vol_slope=0.9, steps_per_year=252)


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    model = {**TRUE, "resid": rng.standard_normal(200_000), "start_var": np.array([1e-4])}
    spots, vols = simulate_fitted_gjr_garch(model, n_paths=1, n_steps=6000, seed=1)
    fitted = fit_gjr_garch(spots[0], vols[0])
    assert fitted["beta"] == pytest.approx(TRUE["beta"], abs=0.04)
    assert fitted["gamma"] == pytest.approx(TRUE["gamma"], abs=0.06)
    assert fitted["persistence"] == pytest.approx(TRUE["alpha"] + TRUE["gamma"] / 2 + TRUE["beta"], abs=0.02)
    assert fitted["vol_slope"] == pytest.approx(TRUE["vol_slope"], abs=0.1)
    assert np.std(fitted["resid"]) == pytest.approx(1.0, abs=0.05)


def test_simulated_paths_start_at_100_and_have_positive_vol():
    model = {**TRUE, "resid": np.random.default_rng(0).standard_normal(1000), "start_var": np.array([5e-5, 2e-4])}
    spots, vols = simulate_fitted_gjr_garch(model, n_paths=50, n_steps=21, seed=0)
    assert spots.shape == (50, 22) and np.all(spots[:, 0] == 100.0) and vols.min() > 0
